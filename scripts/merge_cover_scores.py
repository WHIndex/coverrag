#!/usr/bin/env python3
"""Merge ALCE `.score` metrics with COVER-RAG audit/revision metrics.

ALCE's `eval.py` should remain the source of the original metrics such as
QA-F1, citation_rec, and citation_prec. COVER-RAG scripts add extra fields to
the result JSON, for example:

    cover_audit_summary
    cover_v1_summary
    cover_v1_final_audit_summary

This utility reads:

    result/foo.cover_audit.json
    result/foo.cover_audit.json.score

and writes:

    result/foo.cover_audit.json.cover_score

or a custom `--output` path.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ALCE score with COVER audit metrics.")
    parser.add_argument("--result", required=True, type=Path, help="COVER output JSON, e.g. result/foo.cover_audit.json")
    parser.add_argument(
        "--alce-score",
        type=Path,
        default=None,
        help="ALCE .score file. Default: <result>.score",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: <result>.cover_score",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional one-row CSV output path.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten(f"{prefix}_{key}" if prefix else str(key), nested, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value


def pick_result_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    return {
        "dataset": args.get("dataset_name"),
        "tag": args.get("tag"),
        "model": args.get("model"),
        "shot": args.get("shot"),
        "ndoc": args.get("ndoc"),
    }


def main() -> None:
    args = parse_args()
    result_path = args.result
    alce_score_path = args.alce_score or Path(str(result_path) + ".score")
    output_path = args.output or Path(str(result_path) + ".cover_score")

    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    alce_score = {}
    if alce_score_path.exists():
        alce_score = json.loads(alce_score_path.read_text(encoding="utf-8"))
    claim_citation_score_path = Path(str(result_path) + ".claim_citation_score")
    if claim_citation_score_path.exists():
        claim_citation_score = json.loads(claim_citation_score_path.read_text(encoding="utf-8"))
        alce_score.update(claim_citation_score)

    merged: dict[str, Any] = {}
    merged.update(pick_result_metadata(result_payload))
    flatten("alce", alce_score, merged)

    for key in [
        "cover_audit_summary",
        "cover_audit_config",
        "cover_audit_token_usage",
        "cover_v1_summary",
        "cover_v1_final_audit_summary",
        "cover_v1_config",
        "cover_v1_token_usage",
        "cover_v2_summary",
        "cover_v2_final_audit_summary",
        "cover_v2_config",
        "cover_v2_token_usage",
        "cover_v3_summary",
        "cover_v3_final_audit_summary",
        "cover_v3_config",
        "cover_v3_token_usage",
        "cover_qampari_summary",
        "cover_qampari_config",
        "cover_qampari_token_usage",
    ]:
        if key in result_payload:
            flatten(key, result_payload[key], merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    print(f"Wrote merged COVER score to {output_path}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged.keys()))
            writer.writeheader()
            writer.writerow(merged)
        print(f"Wrote one-row CSV to {args.csv}")


if __name__ == "__main__":
    main()
