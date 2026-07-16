#!/usr/bin/env python3
"""Build a compact raw/v0/v1 result table for COVER-RAG experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


KEY_COLUMNS = [
    "dataset",
    "tag",
    "correctness_note",
    "raw_asqa_str_em",
    "v0_asqa_str_em",
    "v1_asqa_str_em",
    "raw_eli5_claims_nli",
    "v0_eli5_claims_nli",
    "v1_eli5_claims_nli",
    "raw_qampari_f1",
    "v0_qampari_f1",
    "v1_qampari_f1",
    "raw_mauve",
    "v0_mauve",
    "v1_mauve",
    "raw_citation_rec",
    "v0_citation_rec",
    "v1_citation_rec",
    "raw_citation_prec",
    "v0_citation_prec",
    "v1_citation_prec",
    "v0_claim_count",
    "v1_final_claim_count",
    "v0_claim_support_rate",
    "v0_claim_unsupported_rate",
    "v1_draft_claim_support_rate",
    "v1_final_claim_support_rate",
    "v1_final_claim_unsupported_rate",
    "v1_revised_length",
    "v1_revised_citations",
]


CORRECTNESS_NOTE = (
    "ASQA correctness=str_em; ELI5 correctness=claims_nli; "
    "QAMPARI correctness=qampari_f1. Blank means the metric does not apply or was not retained."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact raw/v0/v1 comparison table.")
    parser.add_argument("--raw", type=Path, default=Path("data/4omini_ndoc5_summary.csv"))
    parser.add_argument("--v0-v1", type=Path, default=Path("data/cover_v1_vs_audit_comparison.csv"))
    parser.add_argument("--v0-detail", type=Path, default=Path("data/cover_audit_comparison.csv"))
    parser.add_argument("--v1-detail", type=Path, default=Path("data/cover_v1_comparison.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/key_raw_v0_v1_comparison.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/key_raw_v0_v1_comparison.md"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset", "")).strip(), str(row.get("tag", "")).strip())


def num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pick(row: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        if name in row:
            value = num(row.get(name))
            if value is not None:
                return value
    return None


def asqa_str_em(row: dict[str, Any]) -> float | None:
    return pick(row, ["str_em", "alce_str_em"])


def eli5_claims_nli(row: dict[str, Any]) -> float | None:
    return pick(row, ["claims_nli", "claim_recall", "alce_claims_nli"])


def qampari_f1(row: dict[str, Any]) -> float | None:
    return pick(row, ["qampari_f1", "QAMPARI-F1", "alce_qampari_f1"])


def delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def fmt(value: Any) -> str:
    parsed = num(value)
    if parsed is None:
        return "" if value is None else str(value)
    return f"{parsed:.4f}"


def build_rows(
    raw_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    v0_detail_rows: list[dict[str, str]],
    v1_detail_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw_by_key = {key(row): row for row in raw_rows}
    v0_by_key = {key(row): row for row in v0_detail_rows}
    v1_by_key = {key(row): row for row in v1_detail_rows}
    output = []
    for row in comparison_rows:
        dataset, tag = key(row)
        raw = raw_by_key.get((dataset, tag), {})
        v0_detail = v0_by_key.get((dataset, tag), {})
        v1_detail = v1_by_key.get((dataset, tag), {})
        raw_asqa = asqa_str_em(raw) if dataset == "asqa" else None
        v0_asqa = asqa_str_em(v0_detail) if dataset == "asqa" else None
        v1_asqa = asqa_str_em(v1_detail) if dataset == "asqa" else None
        if dataset == "asqa" and v0_asqa is None:
            v0_asqa = raw_asqa

        raw_eli5 = eli5_claims_nli(raw) if dataset == "eli5" else None
        v0_eli5 = eli5_claims_nli(v0_detail) if dataset == "eli5" else None
        v1_eli5 = eli5_claims_nli(v1_detail) if dataset == "eli5" else None
        if dataset == "eli5" and v0_eli5 is None:
            v0_eli5 = raw_eli5

        raw_qampari = qampari_f1(raw) if dataset == "qampari" else None
        v0_qampari = qampari_f1(v0_detail) if dataset == "qampari" else None
        v1_qampari = qampari_f1(v1_detail) if dataset == "qampari" else None
        if dataset == "qampari" and v0_qampari is None:
            v0_qampari = raw_qampari

        raw_mauve = pick(raw, ["mauve"])
        v0_mauve = pick(row, ["v0_mauve"])
        v1_mauve = pick(row, ["v1_mauve"])

        output.append(
            {
                "dataset": dataset,
                "tag": tag,
                "correctness_note": CORRECTNESS_NOTE,
                "raw_asqa_str_em": raw_asqa,
                "v0_asqa_str_em": v0_asqa,
                "v1_asqa_str_em": v1_asqa,
                "raw_eli5_claims_nli": raw_eli5,
                "v0_eli5_claims_nli": v0_eli5,
                "v1_eli5_claims_nli": v1_eli5,
                "raw_qampari_f1": raw_qampari,
                "v0_qampari_f1": v0_qampari,
                "v1_qampari_f1": v1_qampari,
                "raw_mauve": raw_mauve,
                "v0_mauve": v0_mauve,
                "v1_mauve": v1_mauve,
                "raw_citation_rec": pick(raw, ["citation_rec", "citation_recall"]),
                "v0_citation_rec": pick(row, ["v0_citation_recall"]),
                "v1_citation_rec": pick(row, ["v1_citation_recall"]),
                "raw_citation_prec": pick(raw, ["citation_prec", "citation_precision"]),
                "v0_citation_prec": pick(row, ["v0_citation_precision"]),
                "v1_citation_prec": pick(row, ["v1_citation_precision"]),
                "v0_claim_count": pick(row, ["v0_claim_count"]),
                "v1_final_claim_count": pick(row, ["final_claims"]),
                "v0_claim_support_rate": pick(row, ["v0_claim_support_rate"]),
                "v0_claim_unsupported_rate": pick(row, ["v0_claim_unsupported_rate"]),
                "v1_draft_claim_support_rate": pick(row, ["v1_draft_claim_support_rate"]),
                "v1_final_claim_support_rate": pick(row, ["final_claim_support_rate"]),
                "v1_final_claim_unsupported_rate": pick(row, ["final_claim_unsupported_rate"]),
                "v1_revised_length": pick(row, ["v1_revised_length"]),
                "v1_revised_citations": pick(row, ["v1_revised_citations"]),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt(row.get(col)) for col in KEY_COLUMNS})


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(KEY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(KEY_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in KEY_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_rows = read_csv(args.raw)
    comparison_rows = read_csv(args.v0_v1)
    v0_detail_rows = read_csv(args.v0_detail) if args.v0_detail.exists() else []
    v1_detail_rows = read_csv(args.v1_detail) if args.v1_detail.exists() else []
    rows = build_rows(raw_rows, comparison_rows, v0_detail_rows, v1_detail_rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
