#!/usr/bin/env python3
"""Diagnose QAMPARI selection changes against the original ALCE output.

The COVER-QAMPARI result stores the draft answer in
``item["cover_qampari"]["original_output"]`` and the revised answer in
``item["output"]``.  This script evaluates both with ALCE's QAMPARI string-match
logic and counts answer groups that are kept, lost, or gained.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Any, Iterable


def remove_citations(text: str) -> str:
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", str(text))).replace(" |", "").replace("]", "")


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return " ".join(remove_articles(remove_punc(str(text).lower())).split())


def split_preds(output: str) -> list[str]:
    text = remove_citations(output).strip().split("\n")[0]
    return [
        pred
        for pred in (normalize_answer(part.strip()) for part in text.rstrip().rstrip(".").rstrip(",").split(","))
        if pred
    ]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def qampari_metrics(items: list[dict[str, Any]], outputs: list[str]) -> dict[str, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    recalls_top5: list[float] = []
    f1s: list[float] = []
    f1s_top5: list[float] = []
    num_preds: list[int] = []
    for item, output in zip(items, outputs):
        preds = split_preds(output)
        num_preds.append(len(preds))
        answer_groups = [[normalize_answer(x) for x in group] for group in item.get("answers", [])]
        flat_answers = [answer for group in answer_groups for answer in group]
        precision = sum(pred in flat_answers for pred in preds) / len(preds) if preds else 0.0
        recall_hits = sum(any(answer in preds for answer in group) for group in answer_groups)
        recall = recall_hits / len(answer_groups) if answer_groups else 0.0
        recall_top5 = min(5, recall_hits) / min(5, len(answer_groups)) if answer_groups else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_top5 = 0.0 if precision + recall_top5 == 0 else 2 * precision * recall_top5 / (precision + recall_top5)
        precisions.append(precision)
        recalls.append(recall)
        recalls_top5.append(recall_top5)
        f1s.append(f1)
        f1s_top5.append(f1_top5)
    return {
        "num_preds": mean(num_preds),
        "qampari_prec": 100 * mean(precisions),
        "qampari_rec": 100 * mean(recalls),
        "qampari_rec_top5": 100 * mean(recalls_top5),
        "qampari_f1": 100 * mean(f1s),
        "qampari_f1_top5": 100 * mean(f1s_top5),
    }


def get_original_output(item: dict[str, Any]) -> str:
    cover = item.get("cover_qampari")
    if isinstance(cover, dict):
        return str(cover.get("original_output", "") or "")
    return ""


def collect_selection_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    selected_counts: list[int] = []
    empty_outputs = 0
    for item in items:
        cover = item.get("cover_qampari") if isinstance(item.get("cover_qampari"), dict) else {}
        selected = cover.get("selected_candidates", []) if isinstance(cover, dict) else []
        selected_counts.append(len(selected))
        if not str(item.get("output", "") or "").strip():
            empty_outputs += 1
        for candidate in selected:
            source = str(candidate.get("source", "unknown"))
            label = str(candidate.get("support_label", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "avg_selected": mean(selected_counts),
        "empty_outputs": empty_outputs,
        "source_counts": source_counts,
        "label_counts": label_counts,
    }


def compare_answer_groups(items: list[dict[str, Any]], original_outputs: list[str], revised_outputs: list[str], max_examples: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = {"kept": 0, "lost": 0, "gained": 0, "missed_by_both": 0}
    examples: list[dict[str, Any]] = []
    for idx, (item, original_output, revised_output) in enumerate(zip(items, original_outputs, revised_outputs)):
        original_preds = set(split_preds(original_output))
        revised_preds = set(split_preds(revised_output))
        for answer_group in item.get("answers", []):
            normalized_group = [normalize_answer(answer) for answer in answer_group]
            original_hit = any(answer in original_preds for answer in normalized_group)
            revised_hit = any(answer in revised_preds for answer in normalized_group)
            if original_hit and revised_hit:
                status = "kept"
            elif original_hit and not revised_hit:
                status = "lost"
            elif not original_hit and revised_hit:
                status = "gained"
            else:
                status = "missed_by_both"
            counts[status] += 1
            if status in {"lost", "gained"} and len(examples) < max_examples:
                examples.append(
                    {
                        "item_id": idx,
                        "status": status,
                        "question": item.get("question", ""),
                        "answer_group": " | ".join(str(x) for x in answer_group),
                        "original_output": remove_citations(original_output),
                        "revised_output": remove_citations(revised_output),
                    }
                )
    return counts, examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose QAMPARI original vs revised answer-set changes.")
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=80)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    items = payload.get("data", [])
    original_outputs = [get_original_output(item) for item in items]
    revised_outputs = [str(item.get("output", "") or "") for item in items]
    answer_group_counts, examples = compare_answer_groups(items, original_outputs, revised_outputs, args.max_examples)
    summary = {
        "result": str(args.result),
        "num_examples": len(items),
        "original_metrics": qampari_metrics(items, original_outputs),
        "revised_metrics": qampari_metrics(items, revised_outputs),
        "delta_metrics": {},
        "answer_group_counts": answer_group_counts,
        "selection_stats": collect_selection_stats(items),
    }
    for key, value in summary["revised_metrics"].items():
        summary["delta_metrics"][key] = value - summary["original_metrics"].get(key, 0.0)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "status", "question", "answer_group", "original_output", "revised_output"],
            )
            writer.writeheader()
            writer.writerows(examples)
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
