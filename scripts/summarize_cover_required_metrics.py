#!/usr/bin/env python3
"""Write the fixed metric table used for COVER-RAG experiment tracking.

The table keeps the columns the project cares about:

* main-task correctness
* main-task recall diagnostic
* sentence-level citation recall / precision
* claim-level citation recall / precision
* claim sentence recall
* claim support

It can read run_cover_* summary CSVs, optional answer_recall_summary.csv, and
optional individual result JSON paths with sidecar .score / .claim_citation_score
files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ("asqa", "eli5", "qampari")
METRIC_FIELDS = (
    "main_correctness",
    "main_recall",
    "sentence_citation_recall",
    "sentence_citation_precision",
    "claim_citation_recall",
    "claim_citation_precision",
    "claim_sentence_recall",
    "claim_support",
)
TASK_FIELDS = {
    "asqa": ("str_em", "alce_str_em", "task_score"),
    "eli5": ("claims_nli", "alce_claims_nli", "task_score"),
    "qampari": ("qampari_f1", "alce_qampari_f1", "task_score"),
}


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else root / value


def norm_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {norm_key(str(key)): value for key, value in row.items()}


def read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [normalize_row(row) for row in csv.DictReader(f)]


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(norm_key(name))
        if value not in (None, ""):
            return value
    return ""


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def infer_dataset(path: Path) -> str:
    lowered = path.name.lower()
    for dataset in DATASETS:
        if dataset in lowered:
            return dataset
    return ""


def summary_row_to_metrics(row: dict[str, Any], dataset: str, version: str) -> dict[str, Any]:
    task_fields = TASK_FIELDS.get(dataset, ("task_score",))
    sentence_rec = get(row, "citation_recall", "citation_rec", "alce_citation_rec")
    sentence_prec = get(row, "citation_precision", "citation_prec", "alce_citation_prec")
    claim_rec = get(row, "claim_citation_recall", "claim_citation_rec", "alce_claim_citation_rec")
    claim_prec = get(row, "claim_citation_precision", "claim_citation_prec", "alce_claim_citation_prec")
    claim_sentence_rec = get(
        row,
        "claim_citation_sentence_recall",
        "claim_citation_sentence_rec",
        "alce_claim_citation_sentence_rec",
    )
    claim_support = get(
        row,
        "final_claim_support_rate",
        "claim_support_rate",
        "cover_v3_final_audit_summary_atomic_claim_support_rate_micro",
    )
    support_number = as_float(claim_support)
    if support_number is not None and support_number <= 1.0:
        claim_support = 100.0 * support_number
    if claim_support in ("", None):
        claim_support = claim_rec
    return {
        "dataset": dataset,
        "version": version,
        "main_correctness": get(row, *task_fields),
        "main_recall": "",
        "sentence_citation_recall": sentence_rec,
        "sentence_citation_precision": sentence_prec,
        "claim_citation_recall": claim_rec,
        "claim_citation_precision": claim_prec,
        "claim_sentence_recall": claim_sentence_rec,
        "claim_support": claim_support,
        "source": "summary_csv",
    }


def score_json_to_metrics(path: Path, dataset: str, version: str) -> dict[str, Any]:
    score = read_json(Path(str(path) + ".score"))
    cover_score = read_json(Path(str(path) + ".cover_score"))
    claim_score = read_json(Path(str(path) + ".claim_citation_score"))
    result = read_json(path)
    merged = normalize_row({**score, **cover_score})
    final_summary = result.get("cover_v3_final_audit_summary", {}) if isinstance(result, dict) else {}
    task_fields = TASK_FIELDS.get(dataset, ("task_score",))
    claim_support = ""
    if final_summary:
        support = as_float(final_summary.get("atomic_claim_support_rate_micro"))
        if support is not None:
            claim_support = 100.0 * support
    if claim_support == "":
        claim_support = claim_score.get("claim_citation_rec", "")
    return {
        "dataset": dataset,
        "version": version,
        "main_correctness": get(merged, *task_fields),
        "main_recall": "",
        "sentence_citation_recall": get(merged, "citation_rec", "citation_recall", "alce_citation_rec"),
        "sentence_citation_precision": get(merged, "citation_prec", "citation_precision", "alce_citation_prec"),
        "claim_citation_recall": claim_score.get("claim_citation_rec", get(merged, "alce_claim_citation_rec")),
        "claim_citation_precision": claim_score.get("claim_citation_prec", get(merged, "alce_claim_citation_prec")),
        "claim_sentence_recall": claim_score.get(
            "claim_citation_sentence_rec",
            get(merged, "alce_claim_citation_sentence_rec"),
        ),
        "claim_support": claim_support,
        "source": str(path),
    }


def load_recall_map(path: Path | None) -> dict[tuple[str, str], Any]:
    rows = read_csv(path)
    out: dict[tuple[str, str], Any] = {}
    for row in rows:
        dataset = str(get(row, "dataset")).lower()
        condition = str(get(row, "condition"))
        recall = get(row, "answer_recall_micro")
        if dataset and condition and recall not in ("", None):
            out[(dataset, condition)] = recall
    return out


def attach_recall(rows: list[dict[str, Any]], recall_map: dict[tuple[str, str], Any], latest_label: str) -> None:
    aliases = {
        "v0": ("base_raw", "v0"),
        latest_label: (latest_label, "recall_repair", "current_v3", "latest"),
        "latest": ("latest", latest_label, "recall_repair", "current_v3"),
        "current": ("current", latest_label, "recall_repair", "current_v3"),
    }
    for row in rows:
        dataset = row["dataset"]
        version = row["version"]
        candidates = aliases.get(version, (version,))
        for condition in candidates:
            recall = recall_map.get((dataset, condition))
            if recall not in ("", None):
                row["main_recall"] = recall
                break


def collect_rows(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path, version in (
        (resolve(root, args.v0_summary_csv), args.v0_label),
        (resolve(root, args.latest_summary_csv), args.latest_label),
    ):
        for row in read_csv(csv_path):
            dataset = str(get(row, "dataset")).lower()
            if dataset in DATASETS:
                rows.append(summary_row_to_metrics(row, dataset, version))

    for spec in args.condition:
        try:
            dataset, version, raw_path = spec.split(":", 2)
        except ValueError:
            raise SystemExit(f"Bad --condition {spec!r}; expected dataset:version:path")
        path = resolve(root, raw_path)
        if path is None:
            continue
        dataset = dataset.lower() or infer_dataset(path)
        if dataset not in DATASETS:
            continue
        rows.append(score_json_to_metrics(path, dataset, version))

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row["dataset"], row["version"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return sorted(deduped, key=lambda row: (DATASETS.index(row["dataset"]), row["version"]))


def write_outputs(rows: list[dict[str, Any]], output_csv: Path, output_md: Path) -> None:
    fields = [
        "dataset",
        "version",
        "main_correctness",
        "main_recall",
        "sentence_citation_recall",
        "sentence_citation_precision",
        "claim_citation_recall",
        "claim_citation_precision",
        "claim_sentence_recall",
        "claim_support",
        "source",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) if field not in {"dataset", "version", "source"} else row.get(field, "") for field in fields})

    md_lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                str(row.get(field, ""))
                if field in {"dataset", "version", "source"}
                else fmt(row.get(field))
                for field in fields
            )
            + " |"
        )
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def validate_required_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    datasets = parse_csv_list(args.require_datasets)
    versions = parse_csv_list(args.require_versions)
    if not datasets and not versions and not args.require_filled:
        return

    selected = rows
    if datasets:
        selected = [row for row in selected if str(row.get("dataset", "")).lower() in datasets]
    if versions:
        selected = [row for row in selected if str(row.get("version", "")).lower() in versions]

    by_key = {
        (str(row.get("dataset", "")).lower(), str(row.get("version", "")).lower()): row
        for row in selected
    }
    errors: list[str] = []

    if datasets and versions:
        for dataset in datasets:
            for version in versions:
                if (dataset, version) not in by_key:
                    errors.append(f"missing row dataset={dataset} version={version}")

    if args.require_filled:
        for row in selected:
            dataset = str(row.get("dataset", ""))
            version = str(row.get("version", ""))
            for field in METRIC_FIELDS:
                if row.get(field) in ("", None):
                    errors.append(f"empty {field} for dataset={dataset} version={version}")

    if errors:
        raise SystemExit("Required metrics validation failed:\n" + "\n".join(f"- {error}" for error in errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the fixed COVER-RAG metric table.")
    parser.add_argument("--v0-summary-csv", type=Path, default=None)
    parser.add_argument("--latest-summary-csv", type=Path, default=None)
    parser.add_argument("--answer-recall-summary", type=Path, default=None)
    parser.add_argument("--condition", action="append", default=[], help="dataset:version:path_to_result_json")
    parser.add_argument("--v0-label", default="v0")
    parser.add_argument("--latest-label", default="latest")
    parser.add_argument("--output-csv", type=Path, default=Path("result/cover_required_metrics.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("result/cover_required_metrics.md"))
    parser.add_argument(
        "--require-datasets",
        default="",
        help="Comma-separated datasets that must appear, e.g. asqa,eli5,qampari.",
    )
    parser.add_argument(
        "--require-versions",
        default="",
        help="Comma-separated versions that must appear for each required dataset, e.g. v0,v3.",
    )
    parser.add_argument(
        "--require-filled",
        action="store_true",
        help="Fail if any required metric field is empty for selected rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = root_dir()
    rows = collect_rows(args, root)
    attach_recall(rows, load_recall_map(resolve(root, args.answer_recall_summary)), args.latest_label)
    validate_required_rows(args, rows)
    output_csv = resolve(root, args.output_csv)
    output_md = resolve(root, args.output_md)
    assert output_csv is not None and output_md is not None
    write_outputs(rows, output_csv, output_md)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
