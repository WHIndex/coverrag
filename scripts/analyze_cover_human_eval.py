#!/usr/bin/env python3
"""Analyze completed blinded human evaluation for COVER-RAG."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCORE_METRICS = ("correctness", "citation_faithfulness", "unsupported_claims", "fluency")
SYSTEMS = ("base", "cover")


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stderr(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    m = mean(values)
    if m is None:
        return None
    var = sum((value - m) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def key_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["eval_id"]: row for row in rows}


def side_to_system(key: dict[str, str], side: str) -> str:
    return key[f"system_{side.lower()}"].strip().lower()


def score_column(metric: str, side: str) -> str:
    if metric == "unsupported_claims":
        return f"unsupported_claims_{side.lower()}_count"
    return f"{metric}_{side.lower()}_1_to_5"


def analyze(annotation_rows: list[dict[str, str]], key_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = key_map(key_rows)
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    paired_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    preferences: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for row in annotation_rows:
        eval_id = row.get("eval_id", "")
        key = keys.get(eval_id)
        if key is None:
            continue
        dataset = row.get("dataset", key.get("dataset", "")).strip().lower()
        side_scores: dict[tuple[str, str], float] = {}
        for side in ("a", "b"):
            system = side_to_system(key, side)
            if system not in SYSTEMS:
                continue
            for metric in SCORE_METRICS:
                value = as_float(row.get(score_column(metric, side), ""))
                if value is None:
                    continue
                values[(dataset, system, metric)].append(value)
                values[("macro", system, metric)].append(value)
                side_scores[(system, metric)] = value
        for metric in SCORE_METRICS:
            base_value = side_scores.get(("base", metric))
            cover_value = side_scores.get(("cover", metric))
            if base_value is not None and cover_value is not None:
                paired_deltas[(dataset, metric)].append(cover_value - base_value)
                paired_deltas[("macro", metric)].append(cover_value - base_value)

        preference = (row.get("preference_a_b_tie") or "").strip().lower()
        if preference in {"a", "b"}:
            system = side_to_system(key, preference)
            preferences[(dataset, system)].update(["win"])
            preferences[("macro", system)].update(["win"])
        elif preference in {"tie", "t"}:
            preferences[(dataset, "tie")].update(["win"])
            preferences[("macro", "tie")].update(["win"])

    summary_rows: list[dict[str, Any]] = []
    datasets = sorted({key[0] for key in values if key[0] != "macro"}) + ["macro"]
    for dataset in datasets:
        for system in SYSTEMS:
            row: dict[str, Any] = {"dataset": dataset, "system": system}
            counts = []
            for metric in SCORE_METRICS:
                metric_values = values.get((dataset, system, metric), [])
                row[f"{metric}_mean"] = fmt(mean(metric_values))
                row[f"{metric}_stderr"] = fmt(stderr(metric_values))
                counts.append(len(metric_values))
            row["n"] = max(counts or [0])
            row["preference_wins"] = preferences[(dataset, system)].get("win", 0)
            summary_rows.append(row)
        row = {"dataset": dataset, "system": "tie", "n": "", "preference_wins": preferences[(dataset, "tie")].get("win", 0)}
        for metric in SCORE_METRICS:
            row[f"{metric}_mean"] = ""
            row[f"{metric}_stderr"] = ""
        summary_rows.append(row)

    delta_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        row = {"dataset": dataset, "comparison": "cover_minus_base"}
        for metric in SCORE_METRICS:
            deltas = paired_deltas.get((dataset, metric), [])
            row[f"{metric}_delta_mean"] = fmt(mean(deltas))
            row[f"{metric}_delta_stderr"] = fmt(stderr(deltas))
            row[f"{metric}_n"] = len(deltas)
        delta_rows.append(row)

    return summary_rows, delta_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = root_dir()
    annotation_rows = read_csv(resolve(root, args.annotations))
    key_rows = read_csv(resolve(root, args.key))
    summary_rows, delta_rows = analyze(annotation_rows, key_rows)
    output_dir = resolve(root, args.output_dir)

    summary_fields = ["dataset", "system", "n"]
    for metric in SCORE_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_stderr"])
    summary_fields.append("preference_wins")
    delta_fields = ["dataset", "comparison"]
    for metric in SCORE_METRICS:
        delta_fields.extend([f"{metric}_delta_mean", f"{metric}_delta_stderr", f"{metric}_n"])

    write_csv(output_dir / "human_eval_summary.csv", summary_fields, summary_rows)
    write_markdown(output_dir / "human_eval_summary.md", summary_fields, summary_rows)
    write_csv(output_dir / "human_eval_cover_minus_base.csv", delta_fields, delta_rows)
    write_markdown(output_dir / "human_eval_cover_minus_base.md", delta_fields, delta_rows)
    print(f"Wrote human-eval summaries to {output_dir}")


if __name__ == "__main__":
    main()
