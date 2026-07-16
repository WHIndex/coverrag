#!/usr/bin/env python3
"""Diagnose why COVER-RAG revisions change ASQA correctness and citations.

The script compares a baseline ALCE result and a COVER result. For ASQA it
tracks short-answer exact-match coverage per qa_pair and explains regressions:

    baseline output contains a gold short answer, but COVER output does not.

When COVER metadata is available, it also reports whether the answer span
appeared in draft claims and whether those claims were supported, recovered, or
rejected. This turns "str_em went down" into inspectable failure buckets.
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
    parser = argparse.ArgumentParser(description="Diagnose ASQA COVER-RAG regressions.")
    parser.add_argument("--baseline", required=True, type=Path, help="Raw/v0 ALCE result JSON.")
    parser.add_argument("--cover", required=True, type=Path, help="COVER v1/v2 result JSON.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected result JSON with data list: {path}")


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_presence(short_answers: Sequence[str], output: str) -> tuple[bool, str]:
    norm_output = normalize_answer(output)
    for answer in short_answers:
        norm_answer = normalize_answer(answer)
        if norm_answer and norm_answer in norm_output:
            return True, answer
    return False, str(short_answers[0] if short_answers else "")


def output_sentence_with_answer(output: str, answer: str) -> str:
    norm_answer = normalize_answer(answer)
    if not norm_answer:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", str(output or ""))
    for sentence in sentences:
        if norm_answer in normalize_answer(sentence):
            return sentence.strip()
    return ""


def cover_record(item: dict[str, Any]) -> dict[str, Any]:
    if isinstance(item.get("cover_v2"), dict):
        return item["cover_v2"]
    if isinstance(item.get("cover_v1"), dict):
        return item["cover_v1"]
    if isinstance(item.get("cover_audit"), dict):
        return {"draft_audit": item["cover_audit"]}
    return {}


def claims_containing_answer(item: dict[str, Any], answer: str) -> list[dict[str, Any]]:
    record = cover_record(item)
    audit = record.get("draft_audit") or record.get("final_audit") or {}
    claims = audit.get("claims", []) or []
    norm_answer = normalize_answer(answer)
    matches = []
    for claim in claims:
        haystack = " ".join(
            str(claim.get(key, "") or "")
            for key in ("claim", "source_sentence", "source_sentence_without_citations")
        )
        if norm_answer and norm_answer in normalize_answer(haystack):
            matches.append(claim)
    return matches


def summarize_claim_matches(claims: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = collections.Counter(str(claim.get("label", "unknown")) for claim in claims)
    statuses = collections.Counter(str(claim.get("v2_status", claim.get("label", "unknown"))) for claim in claims)
    examples = []
    for claim in claims[:3]:
        examples.append(
            {
                "claim": claim.get("claim"),
                "label": claim.get("label"),
                "v2_status": claim.get("v2_status"),
                "importance": claim.get("importance"),
                "source_citation_ids": claim.get("source_citation_ids") or claim.get("citation_ids"),
                "evidence_doc_ids": claim.get("evidence_doc_ids"),
            }
        )
    return {
        "matched_claim_count": len(claims),
        "matched_claim_labels": dict(labels),
        "matched_claim_statuses": dict(statuses),
        "matched_claim_examples": examples,
    }


def count_cover_statuses(item: dict[str, Any]) -> dict[str, int]:
    record = cover_record(item)
    statuses = collections.Counter()
    for bucket in ("verified_claims", "recovered_claims", "rejected_claims"):
        for claim in record.get(bucket, []) or []:
            statuses[f"{bucket}:{claim.get('v2_status', claim.get('label', 'unknown'))}"] += 1
    return dict(statuses)


def main() -> None:
    args = parse_args()
    baseline_items = load_items(args.baseline)
    cover_items = load_items(args.cover)
    n = min(len(baseline_items), len(cover_items))
    if args.max_examples is not None:
        n = min(n, args.max_examples)

    rows = []
    summary = collections.Counter()
    for idx in range(n):
        base = baseline_items[idx]
        cover = cover_items[idx]
        qa_pairs = base.get("qa_pairs") or []
        base_output = str(base.get("output", "") or "")
        cover_output = str(cover.get("output", "") or "")

        for qa_id, qa_pair in enumerate(qa_pairs):
            short_answers = qa_pair.get("short_answers") or []
            base_hit, base_answer = exact_presence(short_answers, base_output)
            cover_hit, cover_answer = exact_presence(short_answers, cover_output)
            if base_hit and not cover_hit:
                direction = "lost"
            elif not base_hit and cover_hit:
                direction = "gained"
            elif base_hit and cover_hit:
                direction = "kept"
            else:
                direction = "missed_by_both"
            summary[direction] += 1
            if direction not in {"lost", "gained"}:
                continue

            answer = base_answer if direction == "lost" else cover_answer
            matched_claims = claims_containing_answer(cover, answer)
            claim_summary = summarize_claim_matches(matched_claims)
            rows.append(
                {
                    "item_id": idx,
                    "sample_id": base.get("sample_id", ""),
                    "qa_id": qa_id,
                    "direction": direction,
                    "question": base.get("question", ""),
                    "qa_question": qa_pair.get("question", ""),
                    "answer_span": answer,
                    "baseline_sentence": output_sentence_with_answer(base_output, answer),
                    "cover_sentence": output_sentence_with_answer(cover_output, answer),
                    "baseline_output_len": len(base_output.split()),
                    "cover_output_len": len(cover_output.split()),
                    "matched_claim_count": claim_summary["matched_claim_count"],
                    "matched_claim_labels": json.dumps(claim_summary["matched_claim_labels"], ensure_ascii=False),
                    "matched_claim_statuses": json.dumps(claim_summary["matched_claim_statuses"], ensure_ascii=False),
                    "matched_claim_examples": json.dumps(claim_summary["matched_claim_examples"], ensure_ascii=False),
                    "cover_status_counts": json.dumps(count_cover_statuses(cover), ensure_ascii=False),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "sample_id",
        "qa_id",
        "direction",
        "question",
        "qa_question",
        "answer_span",
        "baseline_sentence",
        "cover_sentence",
        "baseline_output_len",
        "cover_output_len",
        "matched_claim_count",
        "matched_claim_labels",
        "matched_claim_statuses",
        "matched_claim_examples",
        "cover_status_counts",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "num_examples": n,
        "qa_pair_counts": dict(summary),
        "num_regression_rows": len(rows),
        "baseline": str(args.baseline),
        "cover": str(args.cover),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
