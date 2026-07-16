#!/usr/bin/env python3
"""Compare a fixed COVER-RAG v0 baseline with the latest COVER-RAG run.

The intended use is method iteration: keep v0 fixed, rerun only the newest
method, then compare task correctness, ALCE citation metrics when available,
and COVER claim-level metrics in one compact table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare COVER-RAG v0 with the latest version summary.")
    parser.add_argument("--v0", required=True, type=Path, help="cover_audit_comparison.csv")
    parser.add_argument("--latest", required=True, type=Path, help="latest cover comparison CSV, e.g. cover_v3_comparison.csv")
    parser.add_argument("--latest-name", default="v3")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--wide-csv", type=Path, default=None)
    parser.add_argument("--wide-md", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"{label} CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{label} CSV has no data rows: {path}")
    return rows


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("dataset") or ""), str(row.get("tag") or ""), str(row.get("model") or ""))


def get(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    parsed = number(value)
    if parsed is None:
        return "" if value in (None, "") else str(value)
    return f"{parsed:.4f}"


def task_metric_for(dataset: str) -> str:
    if dataset == "asqa":
        return "str_em"
    if dataset == "eli5":
        return "claims_nli"
    if dataset == "qampari":
        return "qampari_f1"
    return "qa_f1"


def task_score(row: dict[str, Any], dataset: str) -> str:
    metric = task_metric_for(dataset)
    if metric == "str_em":
        return get(row, "str_em")
    if metric == "claims_nli":
        return get(row, "claims_nli")
    if metric == "qampari_f1":
        return get(row, "qampari_f1", "QAMPARI-F1")
    return get(row, "qa_f1", "QA-F1")


def row_for(version: str, row: dict[str, Any]) -> dict[str, str]:
    dataset = get(row, "dataset")
    if version == "v0":
        claim_support = get(row, "claim_support_rate")
        claim_unsupported = get(row, "claim_unsupported_rate")
        final_claims = get(row, "claim_count")
        draft_claims = get(row, "claim_count")
        kept = rejected = recovered = expanded = expansion_success = ""
    else:
        claim_support = get(row, "final_claim_support_rate", "claim_support_rate")
        claim_unsupported = get(row, "final_claim_unsupported_rate", "claim_unsupported_rate")
        final_claims = get(row, "final_claims", "claim_count")
        prefix = version
        draft_claims = get(row, f"{prefix}_draft_claims")
        kept = get(row, f"{prefix}_kept_claims")
        rejected = get(row, f"{prefix}_rejected_claims")
        recovered = get(row, f"{prefix}_recovered_claims")
        expanded = get(row, f"{prefix}_expanded_claims")
        expansion_success = get(row, f"{prefix}_expansion_success_rate")

    return {
        "dataset": dataset,
        "tag": get(row, "tag"),
        "model": get(row, "model"),
        "version": version,
        "task_metric": task_metric_for(dataset),
        "task_score": task_score(row, dataset),
        "str_em": get(row, "str_em"),
        "qa_f1": get(row, "qa_f1"),
        "mauve": get(row, "mauve"),
        "citation_recall": get(row, "citation_recall", "citation_rec"),
        "citation_precision": get(row, "citation_precision", "citation_prec"),
        "claim_citation_recall": get(row, "claim_citation_recall"),
        "claim_citation_precision": get(row, "claim_citation_precision"),
        "claim_support_rate": claim_support,
        "claim_unsupported_rate": claim_unsupported,
        "draft_claims": draft_claims,
        "final_claims": final_claims,
        "kept_claims": kept,
        "rejected_claims": rejected,
        "recovered_claims": recovered,
        "expanded_claims": expanded,
        "expansion_success_rate": expansion_success,
        "answer_length": get(row, "answer_length"),
        "avg_revised_length": get(row, f"{version}_avg_revised_length"),
        "avg_doc_pool_size": get(row, f"{version}_avg_doc_pool_size"),
    }


LONG_HEADERS = [
    "dataset",
    "tag",
    "model",
    "version",
    "task_metric",
    "task_score",
    "str_em",
    "qa_f1",
    "mauve",
    "citation_recall",
    "citation_precision",
    "claim_citation_recall",
    "claim_citation_precision",
    "claim_support_rate",
    "claim_unsupported_rate",
    "draft_claims",
    "final_claims",
    "kept_claims",
    "rejected_claims",
    "recovered_claims",
    "expanded_claims",
    "expansion_success_rate",
    "answer_length",
    "avg_revised_length",
    "avg_doc_pool_size",
]


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def write_md(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_wide_rows(long_rows: list[dict[str, str]], latest_name: str) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in long_rows:
        by_key.setdefault((row["dataset"], row["tag"], row["model"]), {})[row["version"]] = row

    metrics = [
        "task_score",
        "str_em",
        "qa_f1",
        "mauve",
        "citation_recall",
        "citation_precision",
        "claim_citation_recall",
        "claim_citation_precision",
        "claim_support_rate",
        "claim_unsupported_rate",
    ]
    rows = []
    for key, versions in sorted(by_key.items()):
        dataset, tag, model = key
        out = {"dataset": dataset, "tag": tag, "model": model, "task_metric": task_metric_for(dataset)}
        for version in ("v0", latest_name):
            row = versions.get(version, {})
            for metric in metrics:
                out[f"{version}_{metric}"] = row.get(metric, "")
        v0_task = number(out.get("v0_task_score"))
        latest_task = number(out.get(f"{latest_name}_task_score"))
        out[f"{latest_name}_minus_v0_task"] = (
            fmt(latest_task - v0_task) if v0_task is not None and latest_task is not None else ""
        )
        out[f"{latest_name}_expanded_claims"] = versions.get(latest_name, {}).get("expanded_claims", "")
        out[f"{latest_name}_expansion_success_rate"] = versions.get(latest_name, {}).get("expansion_success_rate", "")
        out[f"{latest_name}_recovered_claims"] = versions.get(latest_name, {}).get("recovered_claims", "")
        out[f"{latest_name}_avg_doc_pool_size"] = versions.get(latest_name, {}).get("avg_doc_pool_size", "")
        rows.append(out)
    return rows


def main() -> None:
    args = parse_args()
    latest_name = args.latest_name
    source_rows = {
        "v0": {key_for(row): row for row in read_rows(args.v0, "v0")},
        latest_name: {key_for(row): row for row in read_rows(args.latest, latest_name)},
    }
    missing_v0 = sorted(set(source_rows[latest_name]) - set(source_rows["v0"]))
    missing_latest = sorted(set(source_rows["v0"]) - set(source_rows[latest_name]))
    if missing_v0 or missing_latest:
        lines = ["v0/latest summaries do not cover the same dataset/tag/model rows."]
        if missing_v0:
            lines.append("Rows missing from v0:")
            lines.extend(f"  - {key}" for key in missing_v0)
        if missing_latest:
            lines.append(f"Rows missing from {latest_name}:")
            lines.extend(f"  - {key}" for key in missing_latest)
        raise SystemExit("\n".join(lines))
    keys = sorted(set(source_rows["v0"]) | set(source_rows[latest_name]))
    long_rows: list[dict[str, str]] = []
    for key in keys:
        for version in ("v0", latest_name):
            row = source_rows[version].get(key)
            if row is not None:
                long_rows.append(row_for(version, row))

    write_csv(args.output_csv, long_rows, LONG_HEADERS)
    write_md(args.output_md, long_rows, LONG_HEADERS)
    print(f"Wrote long comparison CSV: {args.output_csv}")
    print(f"Wrote long comparison Markdown: {args.output_md}")

    if args.wide_csv or args.wide_md:
        wide_rows = build_wide_rows(long_rows, latest_name)
        wide_headers = list(wide_rows[0].keys()) if wide_rows else ["dataset", "tag", "model"]
        if args.wide_csv:
            write_csv(args.wide_csv, wide_rows, wide_headers)
            print(f"Wrote wide comparison CSV: {args.wide_csv}")
        if args.wide_md:
            write_md(args.wide_md, wide_rows, wide_headers)
            print(f"Wrote wide comparison Markdown: {args.wide_md}")


if __name__ == "__main__":
    main()
