#!/usr/bin/env python3
"""Compare a COVER-RAG v3 output against its own saved draft answer.

Unlike diagnose_cover_v3_failures.py, this script does not require the raw
baseline JSON. COVER v3 stores the original ALCE answer in
item["cover_v3"]["original_output"], so we can inspect answer-span losses even
when only the revised result file was copied back from the server.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose COVER v3 changes using cover_v3.original_output.")
    parser.add_argument("--cover", nargs="+", required=True, type=Path, help="One or more *.cover_v3.json files.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--examples-csv", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=80)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected ALCE-style result JSON: {path}")


def infer_dataset(path: Path) -> str:
    name = path.name.lower()
    for dataset in ("asqa", "eli5", "qampari"):
        if name.startswith(dataset + "-") or dataset in name:
            return dataset
    return "unknown"


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_present(answers: Sequence[str], output: str) -> tuple[bool, str]:
    norm_output = normalize_answer(output)
    for answer in answers:
        norm_answer = normalize_answer(answer)
        if norm_answer and norm_answer in norm_output:
            return True, str(answer)
    return False, str(answers[0] if answers else "")


def short_answers(qa_pair: dict[str, Any]) -> list[str]:
    answers = qa_pair.get("short_answers") or qa_pair.get("answer") or []
    if isinstance(answers, str):
        return [answers]
    return [str(answer) for answer in answers if str(answer or "").strip()]


def output_shape(output: str) -> dict[str, Any]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(output or "")) if part.strip()]
    citations = [re.findall(r"\[(\d+)\]", sent) for sent in sentences]
    return {
        "word_len": len(re.sub(r"\[\d+\]", "", str(output or "")).split()),
        "num_units": len(sentences),
        "citation_links": sum(len(ids) for ids in citations),
        "no_citation_units": sum(1 for ids in citations if not ids),
        "multi_citation_units": sum(1 for ids in citations if len(ids) > 1),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def diagnose_cover(path: Path, max_examples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = infer_dataset(path)
    items = load_items(path)
    counts: collections.Counter[str] = collections.Counter()
    examples: list[dict[str, Any]] = []
    base_shapes: list[dict[str, Any]] = []
    cover_shapes: list[dict[str, Any]] = []

    for item_id, item in enumerate(items):
        base_output = str((item.get("cover_v3") or {}).get("original_output") or "")
        cover_output = str(item.get("output") or "")
        if base_output or cover_output:
            base_shapes.append(output_shape(base_output))
            cover_shapes.append(output_shape(cover_output))

        if dataset != "asqa":
            continue
        for qa_id, qa_pair in enumerate(item.get("qa_pairs") or []):
            answers = short_answers(qa_pair)
            base_has, matched = answer_present(answers, base_output)
            cover_has, _ = answer_present(answers, cover_output)
            if base_has and cover_has:
                counts["kept"] += 1
            elif base_has and not cover_has:
                counts["lost"] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "dataset": dataset,
                            "item_id": item_id,
                            "qa_id": qa_id,
                            "bucket": "lost",
                            "question": item.get("question", ""),
                            "target": matched,
                            "original_output": base_output,
                            "cover_output": cover_output,
                        }
                    )
            elif not base_has and cover_has:
                counts["gained"] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "dataset": dataset,
                            "item_id": item_id,
                            "qa_id": qa_id,
                            "bucket": "gained",
                            "question": item.get("question", ""),
                            "target": matched,
                            "original_output": base_output,
                            "cover_output": cover_output,
                        }
                    )
            else:
                counts["missed_by_both"] += 1

    def shape_mean(key: str, rows: list[dict[str, Any]]) -> float:
        return mean([float(row.get(key, 0) or 0) for row in rows])

    total = sum(counts.values())
    summary = {
        "path": str(path),
        "dataset": dataset,
        "num_examples": len(items),
        "answer_span_counts": dict(counts),
        "answer_span_total": total,
        "answer_span_lost_rate": counts["lost"] / total if total else 0.0,
        "original_avg_word_len": shape_mean("word_len", base_shapes),
        "cover_avg_word_len": shape_mean("word_len", cover_shapes),
        "original_avg_units": shape_mean("num_units", base_shapes),
        "cover_avg_units": shape_mean("num_units", cover_shapes),
        "original_avg_citation_links": shape_mean("citation_links", base_shapes),
        "cover_avg_citation_links": shape_mean("citation_links", cover_shapes),
    }
    return summary, examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "item_id",
        "qa_id",
        "bucket",
        "question",
        "target",
        "original_output",
        "cover_output",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for path in args.cover:
        summary, path_examples = diagnose_cover(path, args.max_examples)
        summaries.append(summary)
        examples.extend(path_examples)
    result = {"summaries": summaries, "examples": examples}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None),
            encoding="utf-8",
        )
    if args.examples_csv:
        write_csv(args.examples_csv, examples)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
