#!/usr/bin/env python3
"""Compute citation metrics aligned with atomic-claim verification.

Unlike ALCE sentence-level citation metrics, this script evaluates whether each
atomic claim in the final answer is supported by its cited evidence. It also
estimates citation precision by checking whether a citation id supports at
least one atomic claim in the same answer unit.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute claim-level citation metrics from COVER final audits.")
    parser.add_argument("--result", required=True, type=Path, help="COVER result JSON with final_audit metadata.")
    parser.add_argument("--output", type=Path, default=None, help="Metric JSON path. Default: <result>.claim_citation_score")
    parser.add_argument("--update-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-file", type=Path, default=None, help="Score file to update. Default: <result>.score")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Expected COVER result JSON with top-level data list.")
    return payload


def final_audit_for(item: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("cover_v3", "cover_v2", "cover_v1"):
        record = item.get(key)
        if isinstance(record, dict) and isinstance(record.get("final_audit"), dict):
            return record["final_audit"]
    if isinstance(item.get("cover_audit"), dict):
        return item["cover_audit"]
    return None


def cover_qampari_candidates_for(item: dict[str, Any]) -> list[dict[str, Any]]:
    record = item.get("cover_qampari")
    if not isinstance(record, dict):
        return []
    candidates = record.get("selected_candidates")
    return candidates if isinstance(candidates, list) else []


def as_ints(values: Any) -> list[int]:
    output = []
    if not isinstance(values, list):
        return output
    for value in values:
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in output:
            output.append(parsed)
    return output


def compute(payload: dict[str, Any]) -> dict[str, Any]:
    total_claims = 0
    supported_claims = 0
    unsupported_claims = 0
    no_citation_claims = 0
    total_sentences = 0
    fully_supported_sentences = 0
    total_citation_links = 0
    useful_citation_links = 0
    label_counts: collections.Counter[str] = collections.Counter()

    for item in payload.get("data", []):
        audit = final_audit_for(item)
        if not audit:
            qampari_candidates = cover_qampari_candidates_for(item)
            if not qampari_candidates:
                continue
            for candidate in qampari_candidates:
                if not isinstance(candidate, dict):
                    continue
                total_claims += 1
                total_sentences += 1
                label = str(candidate.get("support_label", "unknown"))
                label_counts[label] += 1
                citation_ids = as_ints(candidate.get("citation_ids") or [])
                support_doc_ids = as_ints(candidate.get("support_doc_ids") or [])
                total_citation_links += len(citation_ids)
                is_supported = label == "supported" and bool(citation_ids)
                if is_supported:
                    supported_claims += 1
                    fully_supported_sentences += 1
                    useful = set(support_doc_ids or citation_ids)
                    useful_citation_links += sum(1 for citation_id in citation_ids if citation_id in useful)
                else:
                    unsupported_claims += 1
                if not citation_ids:
                    no_citation_claims += 1
            continue
        for sentence in audit.get("sentences", []) or []:
            claims = sentence.get("claims", []) or []
            if not claims:
                continue
            total_sentences += 1
            sentence_all_supported = True
            useful_doc_ids = set()
            sentence_citation_ids = set()
            for claim in claims:
                total_claims += 1
                label = str(claim.get("label", "unknown"))
                label_counts[label] += 1
                citation_ids = as_ints(claim.get("citation_ids") or claim.get("source_citation_ids") or [])
                evidence_doc_ids = as_ints(claim.get("evidence_doc_ids") or [])
                sentence_citation_ids.update(citation_ids)
                if label == "supported":
                    supported_claims += 1
                    useful_doc_ids.update(evidence_doc_ids or citation_ids)
                else:
                    unsupported_claims += 1
                    sentence_all_supported = False
                if not citation_ids:
                    no_citation_claims += 1
            if sentence_all_supported:
                fully_supported_sentences += 1
            for citation_id in sentence_citation_ids:
                total_citation_links += 1
                if citation_id in useful_doc_ids:
                    useful_citation_links += 1

    return {
        "claim_citation_rec": 100.0 * supported_claims / total_claims if total_claims else 0.0,
        "claim_citation_prec": 100.0 * useful_citation_links / total_citation_links if total_citation_links else 0.0,
        "claim_citation_sentence_rec": 100.0 * fully_supported_sentences / total_sentences if total_sentences else 0.0,
        "claim_citation_num_claims": total_claims,
        "claim_citation_supported_claims": supported_claims,
        "claim_citation_unsupported_claims": unsupported_claims,
        "claim_citation_no_citation_claims": no_citation_claims,
        "claim_citation_num_sentences": total_sentences,
        "claim_citation_fully_supported_sentences": fully_supported_sentences,
        "claim_citation_num_links": total_citation_links,
        "claim_citation_useful_links": useful_citation_links,
        "claim_citation_label_counts": dict(label_counts),
    }


def main() -> None:
    args = parse_args()
    payload = load_payload(args.result)
    metrics = compute(payload)
    output = args.output or Path(str(args.result) + ".claim_citation_score")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2 if args.pretty else None), encoding="utf-8")
    print(f"Wrote claim citation metrics to {output}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.update_score:
        score_path = args.score_file or Path(str(args.result) + ".score")
        score = {}
        if score_path.exists():
            score = json.loads(score_path.read_text(encoding="utf-8"))
        score.update(metrics)
        score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2 if args.pretty else None), encoding="utf-8")
        print(f"Updated score file {score_path}")


if __name__ == "__main__":
    main()
