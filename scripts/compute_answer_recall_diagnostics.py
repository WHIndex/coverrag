#!/usr/bin/env python3
"""Compute lightweight answer-recall diagnostics for ALCE-style outputs.

This is a local, no-GPU diagnostic. It complements ALCE's official metrics:

* ASQA: exact short-answer presence over qa_pairs, matching eval.py STR-EM.
* QAMPARI: exact answer-group recall/precision over comma-separated outputs,
  matching eval.py's normalization and list matching.
* ELI5: lexical gold-claim coverage approximation. Official ELI5 correctness
  is AutoAIS claims_nli; this script records existing .score claims_nli when
  available, but avoids loading the NLI model.

It also compares pairs of conditions to quantify kept/lost/gained answer units.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import string
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CONDITIONS = [
    ("asqa", "base_raw", "result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json"),
    (
        "asqa",
        "current_v3",
        "result/cover_v3_main_task_boost_v8_intermediate_full/cover_v3/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json",
    ),
    ("asqa", "oracle_raw", "result/origin/asqa-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json"),
    (
        "asqa",
        "oracle_v3",
        "result/cover_v3_oracle_diagnosis/oracle_run/cover_v3/asqa-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.cover_v3.json",
    ),
    ("eli5", "base_raw", "result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json"),
    (
        "eli5",
        "current_v3",
        "result/cover_v3_main_task_boost_v8_intermediate_full/cover_v3/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.cover_v3.json",
    ),
    ("eli5", "oracle_raw", "result/origin/eli5-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json"),
    (
        "eli5",
        "oracle_v3",
        "result/cover_v3_oracle_diagnosis/oracle_run/cover_v3/eli5-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.cover_v3.json",
    ),
    ("qampari", "base_raw", "result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json"),
    (
        "qampari",
        "current_v3",
        "result/cover_v3_main_task_boost_v8_intermediate_full/cover_v3/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json",
    ),
    ("qampari", "oracle_raw", "result/origin/qampari-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json"),
    (
        "qampari",
        "oracle_v3",
        "result/cover_v3_oracle_diagnosis/oracle_run/cover_v3/qampari-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.cover_v3.json",
    ),
]

DEFAULT_PAIRS = [
    ("asqa", "base_raw", "current_v3"),
    ("asqa", "oracle_raw", "oracle_v3"),
    ("asqa", "base_raw", "oracle_raw"),
    ("asqa", "current_v3", "oracle_v3"),
    ("eli5", "base_raw", "current_v3"),
    ("eli5", "oracle_raw", "oracle_v3"),
    ("eli5", "base_raw", "oracle_raw"),
    ("eli5", "current_v3", "oracle_v3"),
    ("qampari", "base_raw", "current_v3"),
    ("qampari", "oracle_raw", "oracle_v3"),
    ("qampari", "base_raw", "oracle_raw"),
    ("qampari", "current_v3", "oracle_v3"),
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\[\d+\]", " ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def remove_citations(text: Any) -> str:
    return re.sub(r"\[\d+\]", " ", str(text or ""))


def content_tokens(text: Any) -> set[str]:
    return {tok for tok in normalize_answer(text).split() if tok and tok not in STOPWORDS}


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported result JSON: {path}")


def load_score(path: Path) -> dict[str, Any]:
    score_path = Path(str(path) + ".score")
    if not score_path.exists():
        return {}
    try:
        value = json.loads(score_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def score_get(score: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = score.get(key)
        if value not in (None, ""):
            return value
    return ""


def condition_spec(text: str) -> tuple[str, str, Path]:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid condition {text!r}; expected dataset:name:path")
    dataset, name, path = parts
    return dataset.strip().lower(), name.strip(), Path(path)


def pair_spec(text: str) -> tuple[str, str, str]:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid pair {text!r}; expected dataset:before:after")
    dataset, before, after = parts
    return dataset.strip().lower(), before.strip(), after.strip()


def asqa_units(item: dict[str, Any]) -> list[dict[str, Any]]:
    units = []
    for idx, qa_pair in enumerate(item.get("qa_pairs", []) or []):
        answers = [str(x) for x in qa_pair.get("short_answers", []) or [] if str(x).strip()]
        if answers:
            units.append({"unit_id": idx, "target": " | ".join(answers[:4]), "variants": answers})
    return units


def qampari_units(item: dict[str, Any]) -> list[dict[str, Any]]:
    units = []
    for idx, group in enumerate(item.get("answers", []) or []):
        answers = [str(x) for x in group if str(x).strip()]
        if answers:
            units.append({"unit_id": idx, "target": " | ".join(answers[:4]), "variants": answers})
    return units


def eli5_units(item: dict[str, Any]) -> list[dict[str, Any]]:
    units = []
    for idx, claim in enumerate(item.get("claims", []) or []):
        text = str(claim or "").strip()
        if text:
            units.append({"unit_id": idx, "target": text, "variants": [text]})
    return units


def units_for(dataset: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    if dataset == "asqa":
        return asqa_units(item)
    if dataset == "qampari":
        return qampari_units(item)
    if dataset == "eli5":
        return eli5_units(item)
    return []


def qampari_predictions(output: str) -> list[str]:
    preds = [normalize_answer(part.strip()) for part in str(output or "").rstrip().rstrip(".").rstrip(",").split(",")]
    return [pred for pred in preds if pred]


def unit_hit(dataset: str, unit: dict[str, Any], output: str, eli5_claim_overlap: float) -> bool:
    if dataset in {"asqa", "qampari"}:
        if dataset == "qampari":
            preds = set(qampari_predictions(output))
            return any(normalize_answer(variant) in preds for variant in unit["variants"])
        normalized_output = normalize_answer(output)
        return any(normalize_answer(variant) and normalize_answer(variant) in normalized_output for variant in unit["variants"])
    if dataset == "eli5":
        output_tokens = content_tokens(remove_citations(output))
        for variant in unit["variants"]:
            claim_tokens = content_tokens(variant)
            if claim_tokens and len(claim_tokens & output_tokens) / len(claim_tokens) >= eli5_claim_overlap:
                return True
    return False


def condition_hits(dataset: str, items: list[dict[str, Any]], eli5_claim_overlap: float) -> list[dict[str, Any]]:
    rows = []
    for item_idx, item in enumerate(items):
        output = str(item.get("output", "") or "")
        for unit in units_for(dataset, item):
            rows.append(
                {
                    "dataset": dataset,
                    "item_index": item_idx,
                    "item_id": str(item.get("sample_id") or item.get("id") or item_idx),
                    "unit_id": unit["unit_id"],
                    "question": str(item.get("question", "")),
                    "target": unit["target"],
                    "hit": unit_hit(dataset, unit, output, eli5_claim_overlap),
                    "output": output,
                }
            )
    return rows


def summarize_condition(dataset: str, name: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_units = len(rows)
    hit_units = sum(1 for row in rows if row["hit"])
    by_item: dict[int, list[bool]] = {}
    for row in rows:
        by_item.setdefault(int(row["item_index"]), []).append(bool(row["hit"]))
    any_items = sum(1 for hits in by_item.values() if any(hits))
    full_items = sum(1 for hits in by_item.values() if hits and all(hits))
    outputs = {}
    for row in rows:
        outputs.setdefault(row["item_index"], row["output"])
    pred_count = ""
    precision = ""
    f1 = ""
    rec_top5 = ""
    if dataset == "qampari":
        item_precisions = []
        item_recalls = []
        item_recalls_top5 = []
        item_f1s = []
        pred_counts = []
        rows_by_item: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            rows_by_item.setdefault(int(row["item_index"]), []).append(row)
        for item_idx, item_rows in rows_by_item.items():
            preds = qampari_predictions(str(item_rows[0]["output"]))
            pred_counts.append(len(preds))
            flat = {normalize_answer(variant) for row in item_rows for variant in str(row["target"]).split(" | ")}
            correct = sum(pred in flat for pred in preds)
            rec = sum(1 for row in item_rows if row["hit"]) / len(item_rows) if item_rows else 0.0
            prec = correct / len(preds) if preds else 0.0
            rec5 = min(5, sum(1 for row in item_rows if row["hit"])) / min(5, len(item_rows)) if item_rows else 0.0
            item_precisions.append(prec)
            item_recalls.append(rec)
            item_recalls_top5.append(rec5)
            item_f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        pred_count = statistics.mean(pred_counts) if pred_counts else ""
        precision = 100 * statistics.mean(item_precisions) if item_precisions else ""
        f1 = 100 * statistics.mean(item_f1s) if item_f1s else ""
        rec_top5 = 100 * statistics.mean(item_recalls_top5) if item_recalls_top5 else ""
    score = load_score(path)
    return {
        "dataset": dataset,
        "condition": name,
        "path": str(path),
        "num_items": len(by_item),
        "total_units": total_units,
        "hit_units": hit_units,
        "answer_recall_micro": 100 * hit_units / total_units if total_units else "",
        "item_any_rate": 100 * any_items / len(by_item) if by_item else "",
        "item_full_rate": 100 * full_items / len(by_item) if by_item else "",
        "avg_output_words": statistics.mean(len(remove_citations(output).split()) for output in outputs.values()) if outputs else "",
        "qampari_num_preds": pred_count,
        "qampari_precision": precision,
        "qampari_recall_top5": rec_top5,
        "qampari_f1": f1,
        "official_str_em": score_get(score, "str_em"),
        "official_claims_nli": score_get(score, "claims_nli"),
        "official_qampari_rec": score_get(score, "qampari_rec"),
        "official_qampari_f1": score_get(score, "qampari_f1", "QAMPARI-F1"),
        "official_citation_recall": score_get(score, "citation_rec"),
        "official_citation_precision": score_get(score, "citation_prec"),
    }


def compare_pair(
    dataset: str,
    before_name: str,
    after_name: str,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    max_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = {(row["item_index"], row["unit_id"]): row for row in before_rows}
    after = {(row["item_index"], row["unit_id"]): row for row in after_rows}
    keys = sorted(set(before) & set(after))
    counts: Counter[str] = Counter()
    examples = []
    for key in keys:
        b = before[key]
        a = after[key]
        if b["hit"] and a["hit"]:
            bucket = "kept"
        elif b["hit"] and not a["hit"]:
            bucket = "lost"
        elif not b["hit"] and a["hit"]:
            bucket = "gained"
        else:
            bucket = "missed_by_both"
        counts[bucket] += 1
        if bucket in {"lost", "gained"} and len(examples) < max_examples:
            examples.append(
                {
                    "dataset": dataset,
                    "before": before_name,
                    "after": after_name,
                    "bucket": bucket,
                    "item_index": b["item_index"],
                    "item_id": b["item_id"],
                    "unit_id": b["unit_id"],
                    "question": b["question"],
                    "target": b["target"],
                    "before_output": str(b["output"])[:600],
                    "after_output": str(a["output"])[:600],
                }
            )
    total = sum(counts.values())
    before_hits = counts["kept"] + counts["lost"]
    after_hits = counts["kept"] + counts["gained"]
    return (
        {
            "dataset": dataset,
            "before": before_name,
            "after": after_name,
            "total_units": total,
            "before_hit_units": before_hits,
            "after_hit_units": after_hits,
            "before_recall": 100 * before_hits / total if total else "",
            "after_recall": 100 * after_hits / total if total else "",
            "recall_delta": 100 * (after_hits - before_hits) / total if total else "",
            "kept": counts["kept"],
            "lost": counts["lost"],
            "gained": counts["gained"],
            "missed_by_both": counts["missed_by_both"],
            "lost_rate": 100 * counts["lost"] / total if total else "",
            "gain_rate": 100 * counts["gained"] / total if total else "",
        },
        examples,
    )


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def write_md(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join(["---"] * len(fieldnames)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute lightweight answer-recall diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=Path("result/answer_recall_diagnostics"))
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help="dataset:name:path. If omitted, uses the current base/current/oracle defaults.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="dataset:before:after. If omitted, uses default comparisons.",
    )
    parser.add_argument("--eli5-claim-overlap", type=float, default=0.55)
    parser.add_argument("--max-examples", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = root_dir()
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_specs = [condition_spec(value) for value in args.condition] if args.condition else [
        (dataset, name, Path(path)) for dataset, name, path in DEFAULT_CONDITIONS
    ]
    pair_specs = [pair_spec(value) for value in args.pair] if args.pair else DEFAULT_PAIRS

    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for dataset, name, raw_path in condition_specs:
        path = resolve(root, raw_path)
        if not path.exists():
            print(f"WARNING: skip missing condition {dataset}:{name}: {path}")
            continue
        items = load_items(path)
        hit_rows = condition_hits(dataset, items, args.eli5_claim_overlap)
        rows_by_key[(dataset, name)] = hit_rows
        summary_rows.append(summarize_condition(dataset, name, path, hit_rows))

    pair_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for dataset, before, after in pair_specs:
        before_rows = rows_by_key.get((dataset, before))
        after_rows = rows_by_key.get((dataset, after))
        if before_rows is None or after_rows is None:
            continue
        pair_row, examples = compare_pair(dataset, before, after, before_rows, after_rows, args.max_examples)
        pair_rows.append(pair_row)
        example_rows.extend(examples)

    summary_fields = [
        "dataset",
        "condition",
        "num_items",
        "total_units",
        "hit_units",
        "answer_recall_micro",
        "item_any_rate",
        "item_full_rate",
        "avg_output_words",
        "qampari_num_preds",
        "qampari_precision",
        "qampari_recall_top5",
        "qampari_f1",
        "official_str_em",
        "official_claims_nli",
        "official_qampari_rec",
        "official_qampari_f1",
        "official_citation_recall",
        "official_citation_precision",
        "path",
    ]
    pair_fields = [
        "dataset",
        "before",
        "after",
        "total_units",
        "before_hit_units",
        "after_hit_units",
        "before_recall",
        "after_recall",
        "recall_delta",
        "kept",
        "lost",
        "gained",
        "missed_by_both",
        "lost_rate",
        "gain_rate",
    ]
    write_csv(output_dir / "answer_recall_summary.csv", summary_rows, summary_fields)
    write_md(output_dir / "answer_recall_summary.md", summary_rows, summary_fields)
    write_csv(output_dir / "answer_recall_pair_deltas.csv", pair_rows, pair_fields)
    write_md(output_dir / "answer_recall_pair_deltas.md", pair_rows, pair_fields)
    write_csv(output_dir / "answer_recall_unit_changes.csv", example_rows)
    print(f"Wrote {output_dir / 'answer_recall_summary.csv'}")
    print(f"Wrote {output_dir / 'answer_recall_pair_deltas.csv'}")
    print(f"Wrote {output_dir / 'answer_recall_unit_changes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
