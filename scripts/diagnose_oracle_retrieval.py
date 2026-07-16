#!/usr/bin/env python3
"""Diagnose retrieval upper bounds for the current COVER-RAG experiments.

This script answers two questions:

1. Do the retrieved documents contain the gold answer units?
2. How much do task/citation metrics move when the existing oracle-retrieval
   raw outputs are compared with the current base-retrieval outputs?

It is deliberately separate from cover_v3_iterative.py so the diagnostic does
not change the method being tested.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("asqa", "eli5", "qampari")
DEFAULTS = {
    "asqa": {
        "base_raw": "result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json",
        "oracle_raw": "result/origin/asqa-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json",
        "top100": "data/asqa_eval_gtr_top100.json",
        "oracle_docs": "data/asqa_eval_gtr_top100_reranked_oracle.json",
    },
    "eli5": {
        "base_raw": "result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json",
        "oracle_raw": "result/origin/eli5-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json",
        "top100": "data/eli5_eval_bm25_top100.json",
        "oracle_docs": "data/eli5_eval_bm25_top100_reranked_oracle.json",
    },
    "qampari": {
        "base_raw": "result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json",
        "oracle_raw": "result/origin/qampari-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json",
        "top100": "data/qampari_eval_gtr_top100.json",
        "oracle_docs": "data/qampari_eval_gtr_top100_reranked_oracle.json",
    },
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_payload(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON structure: {path}")


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def normalize(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: Any) -> set[str]:
    return {tok for tok in normalize(text).split() if tok and tok not in STOPWORDS}


def doc_text(doc: dict[str, Any]) -> str:
    fields = [
        doc.get("title", ""),
        doc.get("summary", ""),
        doc.get("extraction", ""),
        doc.get("text", ""),
    ]
    return " ".join(str(value or "") for value in fields)


def docs_text(docs: Iterable[dict[str, Any]]) -> str:
    return " ".join(doc_text(doc) for doc in docs)


def prepare_docs(docs: list[dict[str, Any]]) -> tuple[str, set[str]]:
    norm_text = normalize(docs_text(docs))
    return norm_text, {tok for tok in norm_text.split() if tok and tok not in STOPWORDS}


def unit_groups(dataset: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if dataset == "asqa":
        for index, pair in enumerate(item.get("qa_pairs", []) or []):
            answers = [str(x) for x in pair.get("short_answers", []) or [] if str(x).strip()]
            if not answers:
                continue
            groups.append({"index": index, "label": " | ".join(answers[:4]), "variants": answers})
    elif dataset == "qampari":
        for index, answers in enumerate(item.get("answers", []) or []):
            variants = [str(x) for x in answers if str(x).strip()]
            if variants:
                groups.append({"index": index, "label": " | ".join(variants[:4]), "variants": variants})
    elif dataset == "eli5":
        for index, claim in enumerate(item.get("claims", []) or []):
            text = str(claim or "").strip()
            if text:
                groups.append({"index": index, "label": text, "variants": [text]})
    return groups


def lexical_covered(
    dataset: str,
    group: dict[str, Any],
    norm_text: str,
    doc_tokens: set[str],
    threshold: float,
) -> bool:
    for variant in group["variants"]:
        norm_variant = normalize(variant)
        if not norm_variant:
            continue
        if norm_variant in norm_text:
            return True
        if dataset == "eli5":
            claim_tokens = {tok for tok in norm_variant.split() if tok and tok not in STOPWORDS}
            if claim_tokens:
                if len(claim_tokens & doc_tokens) / len(claim_tokens) >= threshold:
                    return True
    return False


def answers_found_covered(groups: list[dict[str, Any]], docs: list[dict[str, Any]]) -> set[int]:
    covered: set[int] = set()
    for doc in docs:
        found = doc.get("answers_found")
        if not isinstance(found, list):
            continue
        for group in groups:
            index = int(group["index"])
            if index < len(found) and bool(found[index]):
                covered.add(index)
    return covered


def lexical_covered_indices(
    dataset: str,
    groups: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    threshold: float,
) -> set[int]:
    norm_text, doc_tokens = prepare_docs(docs)
    return {
        int(group["index"])
        for group in groups
        if lexical_covered(dataset, group, norm_text, doc_tokens, threshold)
    }


def item_id(item: dict[str, Any], index: int) -> str:
    return str(item.get("sample_id") or item.get("id") or index)


def summarize_tier(
    dataset: str,
    tier: str,
    method: str,
    items: list[dict[str, Any]],
    docs_items: list[dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = min(len(items), len(docs_items))
    total_units = 0
    covered_units = 0
    any_items = 0
    full_items = 0
    doc_counts: list[int] = []
    example_rows: list[dict[str, Any]] = []

    for idx in range(n):
        item = items[idx]
        groups = unit_groups(dataset, item)
        docs = docs_items[idx].get("docs", []) or []
        if not isinstance(docs, list):
            docs = []
        if method == "answers_found":
            covered = answers_found_covered(groups, docs)
        else:
            covered = lexical_covered_indices(dataset, groups, docs, threshold)

        total = len(groups)
        got = len(covered)
        total_units += total
        covered_units += got
        doc_counts.append(len(docs))
        if got > 0:
            any_items += 1
        if total > 0 and got == total:
            full_items += 1

        if total > 0 and got < total and len(example_rows) < 100:
            missing = [group["label"] for group in groups if int(group["index"]) not in covered]
            example_rows.append(
                {
                    "dataset": dataset,
                    "tier": tier,
                    "method": method,
                    "item_index": idx,
                    "item_id": item_id(item, idx),
                    "question": str(item.get("question", "")),
                    "total_units": total,
                    "covered_units": got,
                    "missing_units": " || ".join(missing[:8]),
                    "top_doc_titles": " || ".join(str(doc.get("title", "")) for doc in docs[:5]),
                }
            )

    return (
        {
            "dataset": dataset,
            "tier": tier,
            "method": method,
            "num_items": n,
            "total_units": total_units,
            "covered_units": covered_units,
            "coverage_rate": safe_div(covered_units, total_units),
            "items_with_any_unit": any_items,
            "item_any_rate": safe_div(any_items, n),
            "items_with_all_units": full_items,
            "item_full_rate": safe_div(full_items, n),
            "avg_docs": statistics.mean(doc_counts) if doc_counts else 0.0,
        },
        example_rows,
    )


def compare_item_tiers(
    dataset: str,
    items: list[dict[str, Any]],
    base_items: list[dict[str, Any]],
    top100_items: list[dict[str, Any]],
    oracle_items: list[dict[str, Any]],
    threshold: float,
    max_examples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(len(items), len(base_items), len(top100_items), len(oracle_items))
    for idx in range(n):
        item = items[idx]
        groups = unit_groups(dataset, item)
        if not groups:
            continue
        base = lexical_covered_indices(dataset, groups, base_items[idx].get("docs", []) or [], threshold)
        top100 = lexical_covered_indices(dataset, groups, top100_items[idx].get("docs", []) or [], threshold)
        oracle_lex = lexical_covered_indices(dataset, groups, oracle_items[idx].get("docs", []) or [], threshold)
        oracle_found = answers_found_covered(groups, oracle_items[idx].get("docs", []) or [])
        interesting = len(top100) > len(base) or len(oracle_found) > len(base) or len(top100) < len(groups)
        if not interesting:
            continue
        top100_only = [group["label"] for group in groups if int(group["index"]) in top100 - base]
        oracle_only = [group["label"] for group in groups if int(group["index"]) in oracle_found - base]
        top100_missing = [group["label"] for group in groups if int(group["index"]) not in top100]
        rows.append(
            {
                "dataset": dataset,
                "item_index": idx,
                "item_id": item_id(item, idx),
                "question": str(item.get("question", "")),
                "total_units": len(groups),
                "base_top5_units": len(base),
                "top100_units": len(top100),
                "oracle_top5_lexical_units": len(oracle_lex),
                "oracle_top5_answers_found_units": len(oracle_found),
                "top100_extra_units": " || ".join(top100_only[:8]),
                "oracle_answers_found_extra_units": " || ".join(oracle_only[:8]),
                "top100_missing_units": " || ".join(top100_missing[:8]),
            }
        )
        if len(rows) >= max_examples:
            break
    return rows


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metric_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def task_metric(dataset: str) -> str:
    if dataset == "asqa":
        return "str_em"
    if dataset == "eli5":
        return "claims_nli"
    if dataset == "qampari":
        return "qampari_f1"
    return "qa_f1"


def task_from_score(dataset: str, score: dict[str, Any]) -> Any:
    metric = task_metric(dataset)
    if metric == "str_em":
        return metric_value(score, "str_em", "alce_str_em")
    if metric == "claims_nli":
        return metric_value(score, "claims_nli", "alce_claims_nli")
    if metric == "qampari_f1":
        return metric_value(score, "qampari_f1", "QAMPARI-F1", "alce_qampari_f1")
    return metric_value(score, "qa_f1", "QA-F1", "alce_QA-F1")


def score_row(dataset: str, condition: str, retrieval: str, version: str, score_path: Path) -> dict[str, Any]:
    score = load_json(score_path)
    return {
        "dataset": dataset,
        "condition": condition,
        "retrieval": retrieval,
        "version": version,
        "source": str(score_path),
        "task_metric": task_metric(dataset),
        "task_score": task_from_score(dataset, score),
        "str_em": metric_value(score, "str_em", "alce_str_em"),
        "qa_f1": metric_value(score, "QA-F1", "qa_f1", "alce_QA-F1"),
        "claims_nli": metric_value(score, "claims_nli", "alce_claims_nli"),
        "qampari_f1": metric_value(score, "qampari_f1", "QAMPARI-F1", "alce_qampari_f1"),
        "mauve": metric_value(score, "mauve", "alce_mauve"),
        "citation_recall": metric_value(score, "citation_rec", "citation_recall", "alce_citation_rec"),
        "citation_precision": metric_value(score, "citation_prec", "citation_precision", "alce_citation_prec"),
        "claim_citation_recall": metric_value(score, "claim_citation_recall", "alce_claim_citation_rec"),
        "claim_citation_precision": metric_value(score, "claim_citation_precision", "alce_claim_citation_prec"),
        "claim_support_rate": metric_value(
            score,
            "claim_support_rate",
            "final_claim_support_rate",
            "cover_v3_final_audit_summary_atomic_claim_support_rate_micro",
        ),
        "claim_unsupported_rate": metric_value(
            score,
            "claim_unsupported_rate",
            "final_claim_unsupported_rate",
            "cover_v3_final_audit_summary_unsupported_claim_rate_micro",
        ),
    }


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_from_comparison(path: Path | None, retrieval: str, label_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(path):
        dataset = row.get("dataset", "")
        for version in ("v0", "v3"):
            condition = f"{label_prefix}_{version}"
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "retrieval": retrieval,
                    "version": version,
                    "source": str(path),
                    "task_metric": row.get("task_metric", task_metric(dataset)),
                    "task_score": row.get(f"{version}_task_score", ""),
                    "str_em": row.get(f"{version}_str_em", ""),
                    "qa_f1": row.get(f"{version}_qa_f1", ""),
                    "claims_nli": row.get(f"{version}_claims_nli", ""),
                    "qampari_f1": row.get(f"{version}_qampari_f1", ""),
                    "mauve": row.get(f"{version}_mauve", ""),
                    "citation_recall": row.get(f"{version}_citation_recall", ""),
                    "citation_precision": row.get(f"{version}_citation_precision", ""),
                    "claim_citation_recall": row.get(f"{version}_claim_citation_recall", ""),
                    "claim_citation_precision": row.get(f"{version}_claim_citation_precision", ""),
                    "claim_support_rate": row.get(f"{version}_claim_support_rate", ""),
                    "claim_unsupported_rate": row.get(f"{version}_claim_unsupported_rate", ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: format_value(row.get(header, "")) for header in headers})


def write_md(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_metric_rows(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        base_raw = resolve(root, getattr(args, f"{dataset}_base_raw"))
        oracle_raw = resolve(root, getattr(args, f"{dataset}_oracle_raw"))
        if base_raw:
            rows.append(score_row(dataset, "base_raw_score", "base", "raw", Path(str(base_raw) + ".score")))
        if oracle_raw:
            rows.append(score_row(dataset, "oracle_raw_score", "oracle", "raw", Path(str(oracle_raw) + ".score")))
    rows.extend(rows_from_comparison(resolve(root, args.current_comparison_wide), "base", "current"))
    rows.extend(rows_from_comparison(resolve(root, args.oracle_comparison_wide), "oracle", "oracle_run"))
    return [row for row in rows if any(row.get(key) not in ("", None) for key in ("task_score", "citation_recall"))]


def build_delta_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["condition"]): row for row in metric_rows}
    pairs = [
        ("oracle_raw_minus_base_raw", "oracle_raw_score", "base_raw_score"),
        ("current_v3_minus_current_v0", "current_v3", "current_v0"),
        ("oracle_v0_minus_current_v0", "oracle_run_v0", "current_v0"),
        ("oracle_v3_minus_current_v3", "oracle_run_v3", "current_v3"),
        ("oracle_v3_minus_oracle_v0", "oracle_run_v3", "oracle_run_v0"),
    ]
    metrics = [
        "task_score",
        "citation_recall",
        "citation_precision",
        "claim_citation_recall",
        "claim_citation_precision",
        "claim_support_rate",
    ]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for label, after_name, before_name in pairs:
            after = by_key.get((dataset, after_name), {})
            before = by_key.get((dataset, before_name), {})
            if not after or not before:
                continue
            row: dict[str, Any] = {"dataset": dataset, "comparison": label}
            for metric in metrics:
                a = number(after.get(metric))
                b = number(before.get(metric))
                row[f"{metric}_before"] = "" if b is None else b
                row[f"{metric}_after"] = "" if a is None else a
                row[f"{metric}_delta"] = "" if a is None or b is None else a - b
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose oracle retrieval coverage and metric upper bounds.")
    parser.add_argument("--output-dir", type=Path, default=Path("result/cover_v3_oracle_diagnosis/diagnosis"))
    parser.add_argument("--current-comparison-wide", type=Path, default=Path("result/cover_v3_main_task_boost_v8_intermediate_full/cover_v0_v3_comparison_wide.csv"))
    parser.add_argument("--oracle-comparison-wide", type=Path, default=Path("result/cover_v3_oracle_diagnosis/oracle_run/cover_v0_v3_comparison_wide.csv"))
    parser.add_argument("--eli5-claim-overlap", type=float, default=0.55)
    parser.add_argument("--max-comparison-examples", type=int, default=200)
    for dataset, defaults in DEFAULTS.items():
        for key, value in defaults.items():
            parser.add_argument(f"--{dataset}-{key.replace('_', '-')}", dest=f"{dataset}_{key}", default=value)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = root_dir()
    out_dir = resolve(root, args.output_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    comparison_examples: list[dict[str, Any]] = []

    for dataset in DATASETS:
        base_items = load_payload(resolve(root, getattr(args, f"{dataset}_base_raw")))
        top100_items = load_payload(resolve(root, getattr(args, f"{dataset}_top100")))
        oracle_items = load_payload(resolve(root, getattr(args, f"{dataset}_oracle_docs")))
        if not base_items:
            print(f"WARNING: missing base items for {dataset}")
            continue
        threshold = args.eli5_claim_overlap if dataset == "eli5" else 1.0
        tiers = [
            ("base_top5", "lexical", base_items),
            ("candidate_top100", "lexical", top100_items),
            ("oracle_top5", "lexical", oracle_items),
            ("oracle_top5", "answers_found", oracle_items),
        ]
        for tier, method, docs_items in tiers:
            if not docs_items:
                continue
            summary, examples = summarize_tier(dataset, tier, method, base_items, docs_items, threshold)
            coverage_rows.append(summary)
            missing_rows.extend(examples)
        if top100_items and oracle_items:
            comparison_examples.extend(
                compare_item_tiers(
                    dataset,
                    base_items,
                    base_items,
                    top100_items,
                    oracle_items,
                    threshold,
                    args.max_comparison_examples,
                )
            )

    coverage_headers = [
        "dataset",
        "tier",
        "method",
        "num_items",
        "total_units",
        "covered_units",
        "coverage_rate",
        "items_with_any_unit",
        "item_any_rate",
        "items_with_all_units",
        "item_full_rate",
        "avg_docs",
    ]
    write_csv(out_dir / "oracle_retrieval_coverage.csv", coverage_rows, coverage_headers)
    write_md(out_dir / "oracle_retrieval_coverage.md", coverage_rows, coverage_headers)

    missing_headers = [
        "dataset",
        "tier",
        "method",
        "item_index",
        "item_id",
        "question",
        "total_units",
        "covered_units",
        "missing_units",
        "top_doc_titles",
    ]
    write_csv(out_dir / "oracle_retrieval_missing_examples.csv", missing_rows, missing_headers)

    comparison_headers = [
        "dataset",
        "item_index",
        "item_id",
        "question",
        "total_units",
        "base_top5_units",
        "top100_units",
        "oracle_top5_lexical_units",
        "oracle_top5_answers_found_units",
        "top100_extra_units",
        "oracle_answers_found_extra_units",
        "top100_missing_units",
    ]
    write_csv(out_dir / "oracle_retrieval_item_comparison.csv", comparison_examples, comparison_headers)

    metric_rows = build_metric_rows(args, root)
    metric_headers = [
        "dataset",
        "condition",
        "retrieval",
        "version",
        "task_metric",
        "task_score",
        "str_em",
        "qa_f1",
        "claims_nli",
        "qampari_f1",
        "mauve",
        "citation_recall",
        "citation_precision",
        "claim_citation_recall",
        "claim_citation_precision",
        "claim_support_rate",
        "claim_unsupported_rate",
        "source",
    ]
    write_csv(out_dir / "oracle_metric_summary.csv", metric_rows, metric_headers)
    write_md(out_dir / "oracle_metric_summary.md", metric_rows, metric_headers)

    delta_rows = build_delta_rows(metric_rows)
    write_csv(out_dir / "oracle_metric_deltas.csv", delta_rows)

    print(f"Wrote oracle retrieval coverage: {out_dir / 'oracle_retrieval_coverage.csv'}")
    print(f"Wrote oracle metric summary: {out_dir / 'oracle_metric_summary.csv'}")
    print(f"Wrote oracle metric deltas: {out_dir / 'oracle_metric_deltas.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
