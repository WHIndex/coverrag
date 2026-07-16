#!/usr/bin/env python3
"""Diagnose why COVER-RAG v3 changes ALCE task and citation metrics.

This script is intentionally lightweight: it does not run AutoAIS or any LLM.
Instead it inspects the raw/v0 result, the COVER v3 result, and optional run
logs to produce buckets that explain where regressions are likely coming from:

* lost/gained ASQA short-answer coverage
* lost/gained QAMPARI answer groups
* output formatting problems, especially citation-free answer units
* COVER metadata such as rejected/recovered/expanded claims
* expansion attempt failure reasons
* log-level instability such as content-filter and JSON-parse fallbacks
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASET_ORDER = {"asqa": 0, "eli5": 1, "qampari": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose COVER-RAG v3 regressions across ALCE datasets.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Pairs in the form dataset:baseline_json:cover_v3_json, e.g. asqa:result/origin/a.json:result/cover/a.cover_v3.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log", type=Path, default=None, help="Optional nohup log to summarize API/parse fallbacks.")
    parser.add_argument("--max-samples-per-dataset", type=int, default=200)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload
    if isinstance(payload, list):
        return {"data": payload, "args": {}}
    raise ValueError(f"Expected ALCE-style result JSON with data list: {path}")


def infer_dataset_from_path(path: Path) -> str:
    name = path.name.lower()
    for dataset in DATASET_ORDER:
        if name.startswith(dataset + "-") or dataset in name:
            return dataset
    return ""


def parse_pair(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --pairs entry: {spec!r}. Expected dataset:baseline:cover.")
    dataset, baseline, cover = parts
    dataset = dataset.strip().lower()
    if dataset not in DATASET_ORDER:
        raise ValueError(f"Unknown dataset {dataset!r} in pair {spec!r}.")
    return dataset, Path(baseline), Path(cover)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", "", str(text or ""))


def citation_ids(text: str) -> list[int]:
    ids: list[int] = []
    for raw in re.findall(r"\[(\d+)\]", str(text or "")):
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            ids.append(value)
    return ids


def split_sentences(text: str, qampari: bool = False, question: str = "") -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if qampari:
        # Mirrors ALCE's QAMPARI citation evaluation: comma-separated answer items
        # become individual answer units, each prefixed with the question.
        units = [u.strip() for u in text.rstrip().rstrip(".").rstrip(",").split(",") if u.strip()]
        return [f"{question} {unit}".strip() for unit in units]
    # A simple dependency-free approximation of NLTK sent_tokenize.
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def output_shape(item: dict[str, Any], dataset: str) -> dict[str, Any]:
    output = str(item.get("output", "") or "")
    docs = item.get("docs") or []
    sentences = split_sentences(output, qampari=(dataset == "qampari"), question=str(item.get("question", "") or ""))
    sent_citations = [citation_ids(sentence) for sentence in sentences]
    no_citation = sum(1 for ids in sent_citations if not ids)
    multi_citation = sum(1 for ids in sent_citations if len(ids) > 1)
    out_of_range = sum(1 for ids in sent_citations for doc_id in ids if doc_id > len(docs))
    total_links = sum(len(ids) for ids in sent_citations)
    return {
        "word_len": len(remove_citations(output).split()),
        "char_len": len(output),
        "num_units": len(sentences),
        "citation_links": total_links,
        "avg_citations_per_unit": total_links / len(sentences) if sentences else 0.0,
        "no_citation_units": no_citation,
        "no_citation_unit_rate": no_citation / len(sentences) if sentences else 0.0,
        "multi_citation_units": multi_citation,
        "multi_citation_unit_rate": multi_citation / len(sentences) if sentences else 0.0,
        "out_of_range_citation_links": out_of_range,
    }


def exact_presence(short_answers: Sequence[str], output: str) -> tuple[bool, str]:
    norm_output = normalize_answer(output)
    for answer in short_answers:
        norm_answer = normalize_answer(answer)
        if norm_answer and norm_answer in norm_output:
            return True, str(answer)
    return False, str(short_answers[0] if short_answers else "")


def asqa_coverage_rows(
    dataset: str,
    baseline_items: Sequence[dict[str, Any]],
    cover_items: Sequence[dict[str, Any]],
    max_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: collections.Counter[str] = collections.Counter()
    rows: list[dict[str, Any]] = []
    for item_id, (base, cover) in enumerate(zip(baseline_items, cover_items)):
        base_output = str(base.get("output", "") or "")
        cover_output = str(cover.get("output", "") or "")
        for qa_id, qa_pair in enumerate(base.get("qa_pairs") or []):
            short_answers = qa_pair.get("short_answers") or []
            base_hit, base_answer = exact_presence(short_answers, base_output)
            cover_hit, cover_answer = exact_presence(short_answers, cover_output)
            if base_hit and cover_hit:
                bucket = "kept"
            elif base_hit and not cover_hit:
                bucket = "lost"
            elif not base_hit and cover_hit:
                bucket = "gained"
            else:
                bucket = "missed_by_both"
            counts[bucket] += 1
            if bucket in {"lost", "gained"} and len(rows) < max_rows:
                answer = base_answer if bucket == "lost" else cover_answer
                rows.append(
                    {
                        "dataset": dataset,
                        "item_id": item_id,
                        "unit_id": qa_id,
                        "bucket": bucket,
                        "question": base.get("question", ""),
                        "target": answer,
                        "baseline_output": base_output[:600],
                        "cover_output": cover_output[:600],
                    }
                )
    total = sum(counts.values())
    return {
        "asqa_qa_pairs": total,
        "asqa_kept": counts["kept"],
        "asqa_lost": counts["lost"],
        "asqa_gained": counts["gained"],
        "asqa_missed_by_both": counts["missed_by_both"],
        "asqa_lost_rate": counts["lost"] / total if total else 0.0,
        "asqa_gain_rate": counts["gained"] / total if total else 0.0,
    }, rows


def qampari_predictions(output: str) -> list[str]:
    preds = [normalize_answer(x.strip()) for x in str(output or "").rstrip().rstrip(".").rstrip(",").split(",")]
    return [p for p in preds if p]


def qampari_answer_groups(item: dict[str, Any]) -> list[list[str]]:
    return [[normalize_answer(x) for x in group] for group in item.get("answers", []) or []]


def qampari_coverage_rows(
    dataset: str,
    baseline_items: Sequence[dict[str, Any]],
    cover_items: Sequence[dict[str, Any]],
    max_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: collections.Counter[str] = collections.Counter()
    pred_counts_base: list[int] = []
    pred_counts_cover: list[int] = []
    rows: list[dict[str, Any]] = []
    for item_id, (base, cover) in enumerate(zip(baseline_items, cover_items)):
        base_preds = set(qampari_predictions(str(base.get("output", "") or "")))
        cover_preds = set(qampari_predictions(str(cover.get("output", "") or "")))
        pred_counts_base.append(len(base_preds))
        pred_counts_cover.append(len(cover_preds))
        groups = qampari_answer_groups(base)
        for group_id, group in enumerate(groups):
            base_hit = any(ans in base_preds for ans in group if ans)
            cover_hit = any(ans in cover_preds for ans in group if ans)
            if base_hit and cover_hit:
                bucket = "kept"
            elif base_hit and not cover_hit:
                bucket = "lost"
            elif not base_hit and cover_hit:
                bucket = "gained"
            else:
                bucket = "missed_by_both"
            counts[bucket] += 1
            if bucket in {"lost", "gained"} and len(rows) < max_rows:
                rows.append(
                    {
                        "dataset": dataset,
                        "item_id": item_id,
                        "unit_id": group_id,
                        "bucket": bucket,
                        "question": base.get("question", ""),
                        "target": " | ".join(group),
                        "baseline_output": str(base.get("output", "") or "")[:600],
                        "cover_output": str(cover.get("output", "") or "")[:600],
                    }
                )
    total = sum(counts.values())
    return {
        "qampari_answer_groups": total,
        "qampari_kept": counts["kept"],
        "qampari_lost": counts["lost"],
        "qampari_gained": counts["gained"],
        "qampari_missed_by_both": counts["missed_by_both"],
        "qampari_lost_rate": counts["lost"] / total if total else 0.0,
        "qampari_gain_rate": counts["gained"] / total if total else 0.0,
        "qampari_avg_preds_baseline": statistics.mean(pred_counts_base) if pred_counts_base else 0.0,
        "qampari_avg_preds_cover": statistics.mean(pred_counts_cover) if pred_counts_cover else 0.0,
    }, rows


def cover_record(item: dict[str, Any]) -> dict[str, Any]:
    for key in ("cover_v3", "cover_v2", "cover_v1"):
        record = item.get(key)
        if isinstance(record, dict):
            return record
    return {}


def count_claim_labels(claims: Iterable[dict[str, Any]]) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for claim in claims:
        counter[str(claim.get("label", "unknown"))] += 1
    return counter


def summarize_cover_metadata(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    totals: collections.Counter[str] = collections.Counter()
    expansion_reasons: collections.Counter[str] = collections.Counter()
    expansion_sources: collections.Counter[str] = collections.Counter()
    answer_relevance_reasons: collections.Counter[str] = collections.Counter()
    draft_labels: collections.Counter[str] = collections.Counter()
    final_labels: collections.Counter[str] = collections.Counter()
    for item in items:
        record = cover_record(item)
        if not record:
            continue
        draft_audit = record.get("draft_audit") or {}
        final_audit = record.get("final_audit") or {}
        draft_claims = draft_audit.get("claims", []) or []
        final_claims = final_audit.get("claims", []) or []
        draft_labels.update(count_claim_labels(draft_claims))
        final_labels.update(count_claim_labels(final_claims))
        for key in ("verified_claims", "recovered_claims", "expanded_claims", "rejected_claims", "expansion_overflow"):
            totals[key] += len(record.get(key, []) or [])
        for claim in record.get("rejected_claims", []) or []:
            reason = str(claim.get("answer_relevance_reason", "") or "")
            if reason:
                answer_relevance_reasons[reason] += 1
        for attempt in record.get("expansion_attempts", []) or []:
            if attempt.get("attempted"):
                expansion_reasons["attempted"] += 1
            if attempt.get("recovered"):
                expansion_reasons["recovered"] += 1
            reason = str(attempt.get("reason", "") or "")
            if reason:
                expansion_reasons[reason] += 1
            candidate = attempt.get("candidate") or {}
            source = str(candidate.get("source", "") or "")
            if source:
                expansion_sources[source] += 1
    out = {f"cover_{k}": v for k, v in totals.items()}
    out.update({f"expansion_reason_{k}": v for k, v in expansion_reasons.items()})
    out.update({f"expansion_source_{k}": v for k, v in expansion_sources.items()})
    out.update({f"answer_relevance_reason_{k}": v for k, v in answer_relevance_reasons.items()})
    out["draft_label_counts"] = dict(draft_labels)
    out["final_label_counts"] = dict(final_labels)
    return out


def aggregate_shapes(items: Sequence[dict[str, Any]], dataset: str) -> dict[str, Any]:
    shapes = [output_shape(item, dataset) for item in items]
    if not shapes:
        return {}
    keys = list(shapes[0])
    return {f"avg_{key}": statistics.mean(float(row[key]) for row in shapes) for key in keys}


def summarize_log(log_path: Path | None) -> dict[str, Any]:
    if not log_path:
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "log_content_filter_count": text.count("content_filter"),
        "log_claim_decomposition_api_failed": text.count("Claim decomposition API failed"),
        "log_claim_decomposition_json_failed": text.count("Claim decomposition JSON parse failed"),
        "log_expansion_json_failed": text.count("Expansion JSON parse failed"),
        "log_failed_command_count": text.count("FAILED:"),
        "log_traceback_count": text.count("Traceback"),
        "log_checkpoint_count": text.count("checkpoint after"),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []

    for spec in args.pairs:
        dataset, baseline_path, cover_path = parse_pair(spec)
        baseline_payload = load_payload(baseline_path)
        cover_payload = load_payload(cover_path)
        baseline_items = baseline_payload["data"]
        cover_items = cover_payload["data"]
        n = min(len(baseline_items), len(cover_items))
        baseline_items = baseline_items[:n]
        cover_items = cover_items[:n]

        base_shape = aggregate_shapes(baseline_items, dataset)
        cover_shape = aggregate_shapes(cover_items, dataset)
        shape_delta = {
            f"delta_{key.removeprefix('avg_')}": cover_shape.get(key, 0.0) - base_shape.get(key, 0.0)
            for key in base_shape
        }
        row: dict[str, Any] = {
            "dataset": dataset,
            "num_examples": n,
            "baseline": str(baseline_path),
            "cover": str(cover_path),
        }
        row.update({f"baseline_{k}": v for k, v in base_shape.items()})
        row.update({f"cover_{k}": v for k, v in cover_shape.items()})
        row.update(shape_delta)
        row.update(summarize_cover_metadata(cover_items))

        if dataset == "asqa":
            coverage, rows = asqa_coverage_rows(dataset, baseline_items, cover_items, args.max_samples_per_dataset)
            row.update(coverage)
            regression_rows.extend(rows)
        elif dataset == "qampari":
            coverage, rows = qampari_coverage_rows(dataset, baseline_items, cover_items, args.max_samples_per_dataset)
            row.update(coverage)
            regression_rows.extend(rows)
        else:
            # ELI5 correctness is AutoAIS claims_nli in ALCE; without loading the
            # model we record output/citation/claim-side regressions only.
            row["eli5_gold_claims"] = sum(len(item.get("claims", []) or []) for item in baseline_items)

        summary_rows.append(row)

    summary_payload = {
        "log": summarize_log(args.log),
        "datasets": summary_rows,
    }
    (args.output_dir / "diagnosis_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "diagnosis_summary.csv", summary_rows)
    write_csv(args.output_dir / "answer_regressions.csv", regression_rows)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir / 'diagnosis_summary.csv'}")
    print(f"Wrote {args.output_dir / 'answer_regressions.csv'}")


if __name__ == "__main__":
    main()
