#!/usr/bin/env python3
"""Archive vanilla-RAG COVER v13 ablations into paper-facing tables.

The script is intentionally small and dependency-free so it can be run after
each long ablation finishes.  It reads the fixed required-metrics summaries,
copies available per-run tables into the paper directory, and rebuilds compact
macro/by-dataset comparison tables against the frozen full v13 run.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any


DATASETS = ("asqa", "eli5", "qampari")
METRICS = (
    "main_correctness",
    "main_recall",
    "sentence_citation_recall",
    "sentence_citation_precision",
    "claim_citation_recall",
    "claim_citation_precision",
    "claim_sentence_recall",
    "claim_support",
)

DEFAULT_ABLATIONS = (
    (
        "no_recovery",
        "Remove targeted citation/claim recovery",
        "result/cover_v3_ablation_vanilla_no_recovery_full",
    ),
    (
        "no_recall_completion",
        "Remove recall-oriented missing-answer completion",
        "result/cover_v3_ablation_vanilla_no_recall_completion_full",
    ),
    (
        "no_expansion",
        "Remove evidence/claim expansion and recall completion",
        "result/cover_v3_ablation_vanilla_no_expansion_full",
    ),
    (
        "no_final_audit",
        "Remove final audit and audit-based output selection (claim metrics unavailable)",
        "result/cover_v3_ablation_vanilla_no_final_audit_full",
    ),
    (
        "no_candidate_selection",
        "Remove audit-based output candidate selection",
        "result/cover_v3_ablation_vanilla_no_candidate_selection_full",
    ),
    (
        "llm_verifier",
        "Replace NLI claim verifier with LLM verifier",
        "result/cover_v3_ablation_vanilla_llm_verifier_full",
    ),
)

UNAVAILABLE_METRICS = {
    "no_final_audit": {
        "claim_citation_recall",
        "claim_citation_precision",
        "claim_sentence_recall",
        "claim_support",
    }
}


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def latest_rows(summary_path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(summary_path)
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset", "").strip().lower()
        version = row.get("version", "").strip()
        if dataset in DATASETS and version == "recall_repair":
            selected[dataset] = row
    return selected


def metric(row: dict[str, str], name: str) -> float | None:
    return as_float(row.get(name, ""))


def comparison_row(
    dataset: str,
    setting: str,
    label: str,
    row: dict[str, str],
    full_row: dict[str, str],
) -> dict[str, str]:
    out: dict[str, str] = {
        "dataset": dataset,
        "setting": setting,
        "ablated_component": label,
    }
    for name in METRICS:
        if name in UNAVAILABLE_METRICS.get(setting, set()):
            out[name] = ""
            out[f"delta_{name}_vs_full"] = ""
            continue
        value = metric(row, name)
        full_value = metric(full_row, name)
        out[name] = fmt(value)
        delta = "" if value is None or full_value is None else value - full_value
        out[f"delta_{name}_vs_full"] = fmt(delta)
    return out


def macro_row(setting: str, label: str, rows: list[dict[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {
        "setting": setting,
        "ablated_component": label,
    }
    for name in METRICS:
        if name in UNAVAILABLE_METRICS.get(setting, set()):
            out[name] = ""
            continue
        values = [as_float(row.get(name, "")) for row in rows]
        values = [value for value in values if value is not None]
        out[name] = fmt(sum(values) / len(values)) if values else ""
    return out


def add_macro_deltas(macro_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not macro_rows:
        return []
    full = macro_rows[0]
    out: list[dict[str, str]] = []
    for row in macro_rows:
        enriched = dict(row)
        for name in METRICS:
            value = as_float(row.get(name, ""))
            full_value = as_float(full.get(name, ""))
            enriched[f"delta_{name}_vs_full"] = (
                "" if value is None or full_value is None else fmt(value - full_value)
            )
        out.append(enriched)
    return out


def copy_required_tables(name: str, run_dir: Path, tables_dir: Path) -> None:
    csv_path = run_dir / "required_metrics_summary.csv"
    if not csv_path.exists():
        return
    shutil.copy2(csv_path, tables_dir / f"{name}_required_metrics_summary.csv")
    md_path = run_dir / "required_metrics_summary.md"
    if md_path.exists():
        shutil.copy2(md_path, tables_dir / f"{name}_required_metrics_summary.md")
    else:
        rows = read_csv(csv_path)
        if rows:
            write_markdown(
                tables_dir / f"{name}_required_metrics_summary.md",
                list(rows[0].keys()),
                rows,
            )


def remove_required_tables(name: str, tables_dir: Path) -> None:
    for suffix in (".csv", ".md"):
        path = tables_dir / f"{name}_required_metrics_summary{suffix}"
        if path.exists():
            path.unlink()


def complete_required_metrics(name: str, rows_by_dataset: dict[str, dict[str, str]]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    unavailable = UNAVAILABLE_METRICS.get(name, set())
    for dataset in DATASETS:
        row = rows_by_dataset.get(dataset)
        if row is None:
            missing.append(f"{dataset}:row")
            continue
        for metric_name in METRICS:
            if metric_name in unavailable:
                continue
            if row.get(metric_name, "") in ("", None):
                missing.append(f"{dataset}:{metric_name}")
    return not missing, missing


def parse_ablation(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--ablation must use name:label:path, for example "
            "no_final_audit:Remove final audit:result/foo"
        )
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-dir",
        default="paper/ablations/vanilla_rag_cover_v13",
        help="Paper ablation directory, relative to alce/ by default.",
    )
    parser.add_argument(
        "--full-summary",
        default="result/cover_v3_recall_completion_v13_integrated_explanation_full/required_metrics_summary.csv",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        type=parse_ablation,
        help="Optional name:label:path entry. If omitted, known vanilla ablations are used.",
    )
    args = parser.parse_args()

    root = root_dir()
    paper_dir = resolve(root, args.paper_dir)
    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    full_summary = resolve(root, args.full_summary)
    full_by_dataset = latest_rows(full_summary)
    if not full_by_dataset:
        raise SystemExit(f"No recall_repair rows found in full summary: {full_summary}")

    ablations = args.ablation or list(DEFAULT_ABLATIONS)
    by_dataset_rows: list[dict[str, str]] = []

    full_dataset_rows = [full_by_dataset[dataset] for dataset in DATASETS if dataset in full_by_dataset]
    full_macro = macro_row("full_v13", "None", full_dataset_rows)
    macro_rows = [full_macro]

    for dataset, full_row in full_by_dataset.items():
        by_dataset_rows.append(comparison_row(dataset, "full_v13", "None", full_row, full_row))

    for name, label, run_dir_text in ablations:
        run_dir = resolve(root, run_dir_text)
        summary = run_dir / "required_metrics_summary.csv"
        if not summary.exists():
            print(f"SKIP {name}: missing {summary}")
            remove_required_tables(name, tables_dir)
            continue
        rows_by_dataset = latest_rows(summary)
        complete, missing = complete_required_metrics(name, rows_by_dataset)
        if not complete:
            print(f"SKIP {name}: incomplete required metrics ({', '.join(missing[:8])})")
            remove_required_tables(name, tables_dir)
            continue
        copy_required_tables(name, run_dir, tables_dir)
        dataset_rows: list[dict[str, str]] = []
        for dataset in DATASETS:
            row = rows_by_dataset.get(dataset)
            full_row = full_by_dataset.get(dataset)
            if row is None or full_row is None:
                continue
            dataset_rows.append(row)
            by_dataset_rows.append(comparison_row(dataset, name, label, row, full_row))
        if dataset_rows:
            macro_rows.append(macro_row(name, label, dataset_rows))

    macro_rows = add_macro_deltas(macro_rows)

    macro_fields = ["setting", "ablated_component"]
    for name in METRICS:
        macro_fields.extend([name, f"delta_{name}_vs_full"])
    dataset_fields = ["dataset", "setting", "ablated_component"]
    for name in METRICS:
        dataset_fields.extend([name, f"delta_{name}_vs_full"])

    write_csv(tables_dir / "ablation_core_modules_macro.csv", macro_fields, macro_rows)
    write_markdown(tables_dir / "ablation_core_modules_macro.md", macro_fields, macro_rows)
    write_csv(tables_dir / "ablation_core_modules_by_dataset.csv", dataset_fields, by_dataset_rows)
    write_markdown(tables_dir / "ablation_core_modules_by_dataset.md", dataset_fields, by_dataset_rows)

    print(f"Wrote {tables_dir / 'ablation_core_modules_macro.csv'}")
    print(f"Wrote {tables_dir / 'ablation_core_modules_by_dataset.csv'}")


if __name__ == "__main__":
    main()
