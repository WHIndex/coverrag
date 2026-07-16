#!/usr/bin/env python3
"""Fill a COVER-RAG pairwise evaluation sheet with AI-assisted labels.

Important: the output is *not* human evaluation. It is useful as a pilot
annotation or LLM-as-judge sanity check, but paper-facing human-evaluation
claims should still use human annotators or explicitly call this LLM judging.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from cover_posthoc_audit import OpenAIChatClient, extract_json_object


SYSTEM = """You are a careful evaluator for citation-grounded QA.
You must judge two blinded answers, A and B, using only the question, reference
answer, the answer text, and its cited evidence snippets.

Return strict JSON only. Do not mention which system is better outside JSON.
Be conservative: if a factual claim is not clearly supported by cited evidence,
count it as unsupported. If the answer includes correct facts not in the cited
evidence, it may be correct but not citation-faithful.
"""

USER_TEMPLATE = """Question:
{question}

Reference answer:
{reference_answer}

Answer A:
{system_a_answer}

Cited evidence for A:
{system_a_cited_evidence}

Answer B:
{system_b_answer}

Cited evidence for B:
{system_b_cited_evidence}

Fill the following fields.

Scoring rules:
- correctness_*_1_to_5: 1 wrong/unhelpful, 3 partially correct, 5 fully correct and complete for the question.
- citation_faithfulness_*_1_to_5: 1 citations support little/none, 3 mixed or partial support, 5 cited evidence supports the factual claims.
- unsupported_claims_*_count: integer count of factual claims in the answer not supported by its cited evidence.
- fluency_*_1_to_5: 1 hard to read, 5 clear and fluent.
- preference_a_b_tie: "A", "B", or "Tie" based on overall correctness plus trustworthiness.
- notes: one concise English sentence explaining the main difference.

Return JSON with exactly these keys:
{{
  "correctness_a_1_to_5": 1,
  "correctness_b_1_to_5": 1,
  "citation_faithfulness_a_1_to_5": 1,
  "citation_faithfulness_b_1_to_5": 1,
  "unsupported_claims_a_count": 0,
  "unsupported_claims_b_count": 0,
  "fluency_a_1_to_5": 1,
  "fluency_b_1_to_5": 1,
  "preference_a_b_tie": "Tie",
  "notes": "..."
}}
"""

LABEL_FIELDS = [
    "correctness_a_1_to_5",
    "correctness_b_1_to_5",
    "citation_faithfulness_a_1_to_5",
    "citation_faithfulness_b_1_to_5",
    "unsupported_claims_a_count",
    "unsupported_claims_b_count",
    "fluency_a_1_to_5",
    "fluency_b_1_to_5",
    "preference_a_b_tie",
    "notes",
]

INT_FIELDS = [
    "correctness_a_1_to_5",
    "correctness_b_1_to_5",
    "citation_faithfulness_a_1_to_5",
    "citation_faithfulness_b_1_to_5",
    "unsupported_claims_a_count",
    "unsupported_claims_b_count",
    "fluency_a_1_to_5",
    "fluency_b_1_to_5",
]


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def prompt_for(row: dict[str, str], max_chars: int) -> str:
    return USER_TEMPLATE.format(
        question=truncate(row.get("question", ""), max_chars),
        reference_answer=truncate(row.get("reference_answer", ""), max_chars),
        system_a_answer=truncate(row.get("system_a_answer", ""), max_chars),
        system_a_cited_evidence=truncate(row.get("system_a_cited_evidence", ""), max_chars),
        system_b_answer=truncate(row.get("system_b_answer", ""), max_chars),
        system_b_cited_evidence=truncate(row.get("system_b_cited_evidence", ""), max_chars),
    )


def clamp_int(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(round(float(value)))
    except Exception:
        parsed = low
    return min(high, max(low, parsed))


def normalize_labels(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in INT_FIELDS:
        high = 99 if field.startswith("unsupported_claims") else 5
        out[field] = clamp_int(raw.get(field), 0 if high == 99 else 1, high)
    preference = str(raw.get("preference_a_b_tie", "Tie") or "Tie").strip().upper()
    if preference not in {"A", "B", "TIE"}:
        preference = "Tie"
    if preference == "TIE":
        preference = "Tie"
    out["preference_a_b_tie"] = preference
    out["notes"] = str(raw.get("notes", "") or "")[:500]
    return out


def row_already_labeled(row: dict[str, str]) -> bool:
    return all(str(row.get(field, "")).strip() for field in LABEL_FIELDS if field != "notes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("paper/human_eval/cover_v13_pairwise/annotation_sheet.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/human_eval/cover_v13_pairwise/annotation_sheet_ai_filled.csv"),
    )
    parser.add_argument("--cache-file", type=Path, default=Path("cache/cover_human_eval_ai_labels.json"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--max-chars-per-field", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = root_dir()
    input_path = resolve(root, args.input)
    output_path = resolve(root, args.output)
    cache_path = resolve(root, args.cache_file)

    fieldnames, rows = read_csv(input_path)
    cache = read_cache(cache_path)
    client = OpenAIChatClient(args.model, args.temperature, args.top_p, args.max_retries)

    processed = 0
    changed = 0
    for row in rows:
        eval_id = row.get("eval_id", "")
        if not eval_id:
            continue
        if not args.overwrite and row_already_labeled(row):
            continue
        if eval_id in cache:
            labels = cache[eval_id]
        else:
            if args.limit and processed >= args.limit:
                break
            raw, _usage = client.chat(SYSTEM, prompt_for(row, args.max_chars_per_field), max_tokens=args.max_tokens)
            labels = normalize_labels(extract_json_object(raw))
            cache[eval_id] = labels
            processed += 1
            changed += 1
            if args.sleep:
                time.sleep(args.sleep)
            if changed % max(1, args.save_every) == 0:
                write_cache(cache_path, cache)
                write_csv(output_path, fieldnames, rows)
                print(f"Saved {changed} new labels...")
        for field in LABEL_FIELDS:
            row[field] = str(labels.get(field, ""))

    write_cache(cache_path, cache)
    write_csv(output_path, fieldnames, rows)
    print(f"Wrote AI-assisted labels to {output_path}")
    print(f"New API-labeled examples this run: {changed}; cached labels: {len(cache)}")
    print("Reminder: do not report this output as human evaluation without human review.")


if __name__ == "__main__":
    main()
