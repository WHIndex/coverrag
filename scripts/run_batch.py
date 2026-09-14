#!/usr/bin/env python3
"""Resume-friendly batch runner for CoverRAG.

This script is resume-friendly:
  - existing CoverRAG JSON files are skipped unless --force-revise is set;
  - existing .score files are skipped unless --force-eval is set;
  - failures are logged and the next input continues unless --stop-on-error is set;
  - if ALCE citation evaluation fails, it can retry without --citations so
    correctness metrics such as ASQA str_em are still available.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


NLI_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DATASET_ORDER = {"asqa": 0, "eli5": 1, "qampari": 2}
TAG_ORDER = {"gtr": 0, "bm25": 1, "bge_rerank_default": 2, "reranked_oracle": 3}


SUMMARY_COLUMNS: list[tuple[str, str, str]] = [
    ("dataset", "dataset", "text"),
    ("tag", "tag", "text"),
    ("model", "model", "text"),
    ("shot", "shot", "int"),
    ("ndoc", "ndoc", "int"),
    ("answer_length", "alce_length", "float"),
    ("str_em", "alce_str_em", "float"),
    ("str_hit", "alce_str_hit", "float"),
    ("qa_em", "alce_QA-EM", "float"),
    ("qa_f1", "alce_QA-F1", "float"),
    ("qa_hit", "alce_QA-Hit", "float"),
    ("rougeLsum", "alce_rougeLsum", "float"),
    ("mauve", "alce_mauve", "float"),
    ("claims_nli", "alce_claims_nli", "float"),
    ("num_preds", "alce_num_preds", "float"),
    ("qampari_prec", "alce_qampari_prec", "float"),
    ("qampari_rec", "alce_qampari_rec", "float"),
    ("qampari_rec_top5", "alce_qampari_rec_top5", "float"),
    ("qampari_f1", "alce_qampari_f1", "float"),
    ("qampari_f1_top5", "alce_qampari_f1_top5", "float"),
    ("citation_recall", "alce_citation_rec", "float"),
    ("citation_precision", "alce_citation_prec", "float"),
    ("claim_citation_recall", "alce_claim_citation_rec", "float"),
    ("claim_citation_precision", "alce_claim_citation_prec", "float"),
    ("claim_citation_sentence_recall", "alce_claim_citation_sentence_rec", "float"),
    ("coverrag_examples", "coverrag_summary_num_examples", "int"),
    ("coverrag_draft_claims", "coverrag_summary_draft_num_claims", "int"),
    ("coverrag_initial_supported", "coverrag_summary_initially_supported_claims", "int"),
    ("coverrag_recovered_claims", "coverrag_summary_recovered_claims", "int"),
    ("coverrag_expanded_claims", "coverrag_summary_expanded_claims", "int"),
    ("coverrag_draft_aligned_kept", "coverrag_summary_draft_aligned_kept_claims", "int"),
    ("coverrag_verified_plus_expanded", "coverrag_summary_verified_plus_expanded_claims", "int"),
    ("coverrag_kept_claims", "coverrag_summary_kept_verified_claims", "int"),
    ("coverrag_rejected_claims", "coverrag_summary_rejected_claims", "int"),
    ("coverrag_recovery_attempted", "coverrag_summary_recovery_attempted_claims", "int"),
    ("coverrag_recovery_successful", "coverrag_summary_recovery_successful_claims", "int"),
    ("coverrag_recovery_success_rate", "coverrag_summary_recovery_success_rate_micro", "rate"),
    ("coverrag_expansion_candidates", "coverrag_summary_expansion_candidate_claims", "int"),
    ("coverrag_expansion_attempted", "coverrag_summary_expansion_attempted_claims", "int"),
    ("coverrag_expansion_successful", "coverrag_summary_expansion_successful_claims", "int"),
    ("coverrag_expansion_success_rate", "coverrag_summary_expansion_success_rate_micro", "rate"),
    ("coverrag_expanded_answer_spans", "coverrag_summary_expanded_answer_spans", "int"),
    ("coverrag_recall_target_mismatch", "coverrag_summary_recall_completion_target_mismatches", "int"),
    ("coverrag_recall_no_targets", "coverrag_summary_recall_completion_no_targets", "int"),
    ("coverrag_expansion_low_answer_unit_score", "coverrag_summary_expansion_low_answer_unit_score", "int"),
    ("coverrag_expansion_low_target_relevance", "coverrag_summary_expansion_low_target_relevance", "int"),
    ("coverrag_expansion_low_explanatory_precision", "coverrag_summary_expansion_low_explanatory_precision", "int"),
    ("coverrag_expansion_surface_filtered", "coverrag_summary_expansion_surface_filtered", "int"),
    ("coverrag_expansion_answer_unit_filtered", "coverrag_summary_expansion_answer_unit_filtered", "int"),
    ("coverrag_iterative_completion_claims", "coverrag_summary_iterative_completion_claims", "int"),
    ("coverrag_iterative_completion_attempts", "coverrag_summary_iterative_completion_attempts", "int"),
    ("coverrag_iterative_completion_successful", "coverrag_summary_iterative_completion_successful_claims", "int"),
    ("coverrag_iterative_completion_success_rate", "coverrag_summary_iterative_completion_success_rate_micro", "rate"),
    ("coverrag_dynamic_retrieval_rounds", "coverrag_summary_dynamic_retrieval_rounds", "int"),
    ("coverrag_dynamic_retrieval_queries", "coverrag_summary_dynamic_retrieval_queries", "int"),
    ("coverrag_dynamic_retrieval_added_docs", "coverrag_summary_dynamic_retrieval_added_docs", "int"),
    ("coverrag_dynamic_retrieval_successful_rounds", "coverrag_summary_dynamic_retrieval_successful_rounds", "int"),
    ("coverrag_dynamic_retrieval_added_docs_per_example", "coverrag_summary_dynamic_retrieval_added_docs_per_example", "float"),
    ("coverrag_evidence_seed_claims", "coverrag_summary_evidence_seed_claims", "int"),
    ("coverrag_blueprint_claims", "coverrag_summary_blueprint_claims", "int"),
    ("coverrag_avg_blueprint_claims", "coverrag_summary_avg_blueprint_claims", "float"),
    ("coverrag_reader_api_calls", "coverrag_summary_reader_api_calls", "int"),
    ("coverrag_reader_calls_per_example", "coverrag_summary_reader_api_calls_per_example", "float"),
    ("coverrag_reader_accepted_units", "coverrag_summary_reader_accepted_units", "int"),
    ("coverrag_reader_fallback_units", "coverrag_summary_reader_fallback_units", "int"),
    ("coverrag_coverage_guided_retrieval_rounds", "coverrag_summary_coverage_guided_retrieval_rounds", "int"),
    ("coverrag_coverage_guided_retrieval_targets", "coverrag_summary_coverage_guided_retrieval_targets", "int"),
    (
        "coverrag_coverage_guided_retrieval_skipped_no_targets",
        "coverrag_summary_coverage_guided_retrieval_skipped_no_targets",
        "int",
    ),
    ("coverrag_explanation_render_examples", "coverrag_summary_explanation_render_examples", "int"),
    ("coverrag_explanation_render_selected", "coverrag_summary_explanation_render_selected_claims", "int"),
    ("coverrag_explanation_render_length_limited", "coverrag_summary_explanation_render_length_limited_claims", "int"),
    ("coverrag_candidate_selection_examples", "coverrag_summary_candidate_selection_examples", "int"),
    ("coverrag_candidate_selection_changed", "coverrag_summary_candidate_selection_changed_outputs", "int"),
    ("coverrag_candidate_selection_candidates", "coverrag_summary_candidate_selection_total_candidates", "int"),
    ("coverrag_candidate_selection_audit_failures", "coverrag_summary_candidate_selection_audit_failures", "int"),
    ("coverrag_candidate_selected_current", "coverrag_summary_candidate_selection_selected_labels_current", "int"),
    ("coverrag_candidate_selected_base", "coverrag_summary_candidate_selection_selected_labels_base_preserving", "int"),
    ("coverrag_candidate_selected_integrated", "coverrag_summary_candidate_selection_selected_labels_integrated", "int"),
    ("coverrag_candidate_selected_recall", "coverrag_summary_candidate_selection_selected_labels_recall_heavy", "int"),
    ("coverrag_keep_rate", "coverrag_summary_claim_keep_rate_micro", "rate"),
    ("coverrag_rejected_rate", "coverrag_summary_rejected_claim_rate_micro", "rate"),
    ("coverrag_avg_revised_length", "coverrag_summary_avg_revised_length", "float"),
    ("coverrag_avg_revised_citations", "coverrag_summary_avg_revised_num_citations", "float"),
    ("coverrag_avg_doc_pool_size", "coverrag_summary_avg_doc_pool_size", "float"),
    ("coverrag_draft_supported", "coverrag_summary_draft_label_counts_supported", "int"),
    ("coverrag_draft_wrong_missing", "coverrag_summary_draft_label_counts_wrong_or_missing_citation", "int"),
    ("coverrag_draft_not_supported", "coverrag_summary_draft_label_counts_not_supported", "int"),
    ("coverrag_draft_no_citation", "coverrag_summary_draft_label_counts_no_citation", "int"),
    ("final_claims", "coverrag_final_audit_summary_num_claims", "int"),
    ("final_claim_support_rate", "coverrag_final_audit_summary_atomic_claim_support_rate_micro", "rate"),
    ("final_claim_unsupported_rate", "coverrag_final_audit_summary_unsupported_claim_rate_micro", "rate"),
    ("final_no_citation_claims", "coverrag_final_audit_summary_no_citation_claims", "int"),
]


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_raw_result(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    bad_markers = [
        ".cover_revision",
        ".coverrag",
        ".cover_audit",
        ".cover_score",
        ".score",
        ".debug",
    ]
    return not any(marker in path.name for marker in bad_markers)


def discover_inputs(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.inputs:
        files = [resolve(root, path) for path in args.inputs]
    else:
        files = sorted(resolve(root, args.result_dir).glob(args.pattern))
    files = [path for path in files if is_raw_result(path)]
    if not files:
        raise SystemExit("No raw result JSON files found. Check --result-dir, --pattern, or --inputs.")
    return files


def run(cmd: list[str | Path], root: Path, dry_run: bool) -> None:
    cmd = [str(part) for part in cmd]
    print("+ " + subprocess.list2cmdline(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=root, check=True)


def score_path(path: Path) -> Path:
    return Path(str(path) + ".score")


def cover_score_path(path: Path) -> Path:
    return Path(str(path) + ".cover_score")


def infer_dataset_from_path(path: Path) -> str:
    name = path.name.lower()
    for dataset in ("asqa", "eli5", "qampari"):
        if name.startswith(dataset + "-"):
            return dataset
    return ""


def build_merge_cmd(args: argparse.Namespace, root: Path, cover: Path) -> list[str | Path]:
    cmd: list[str | Path] = [args.python, root / "scripts" / "merge_scores.py", "--result", cover]
    sp = score_path(cover)
    if sp.exists():
        cmd.extend(["--alce-score", sp])
    cmd.append("--pretty")
    return cmd


def build_claim_citation_cmd(args: argparse.Namespace, root: Path, cover: Path) -> list[str | Path]:
    return [
        args.python,
        root / "scripts" / "compute_claim_citation_metrics.py",
        "--result",
        cover,
        "--pretty",
    ]


def fmt(value: Any, kind: str, md: bool = False) -> str:
    if value is None or value == "":
        return ""
    try:
        if kind == "rate":
            return f"{float(value) * 100:.2f}" if md else f"{float(value) * 100:.4f}"
        if kind == "float":
            return f"{float(value):.2f}" if md else f"{float(value):.4f}"
        if kind == "int":
            return str(int(float(value)))
    except Exception:
        return str(value)
    return str(value)


def sort_key(row: dict[str, Any]) -> tuple[int, str, int, str]:
    dataset = str(row.get("dataset") or "")
    tag = str(row.get("tag") or "")
    return (DATASET_ORDER.get(dataset, 99), dataset, TAG_ORDER.get(tag, 99), tag)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run CoverRAG over attributed RAG outputs.")
    parser.add_argument("--result-dir", type=Path, default=Path("result/origin"))
    parser.add_argument("--pattern", default="*-gpt-4o-mini-*-shot2-ndoc5-42.json")
    parser.add_argument("--inputs", nargs="*", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("result/coverrag"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N examples in each input.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Pass through to coverrag.py.")
    parser.add_argument("--control-mode", choices=["answer_seeded", "evidence_seeded"], default="answer_seeded")
    parser.add_argument("--reader-model", default="gpt-4o-mini")
    parser.add_argument("--reader-max-tokens", type=int, default=1200)
    parser.add_argument("--reader-max-plan-claims", type=int, default=24)
    parser.add_argument("--reader-max-unit-words", type=int, default=80)
    parser.add_argument("--reader-conformance-verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidence-seed-max-claims", type=int, default=12)
    parser.add_argument("--evidence-seed-max-verifications", type=int, default=20)
    parser.add_argument("--evidence-seed-evidence-sentences", type=int, default=30)
    parser.add_argument("--evidence-seed-candidate-spans", type=int, default=80)
    parser.add_argument("--evidence-seed-max-candidates", type=int, default=24)

    parser.add_argument("--candidate-docs-file", type=Path, default=None)
    parser.add_argument("--max-doc-pool", type=int, default=100)
    parser.add_argument("--dynamic-retrieval-mode", choices=["off", "global_candidate_docs"], default="off")
    parser.add_argument("--dynamic-retrieval-top-k", type=int, default=12)
    parser.add_argument("--dynamic-retrieval-per-query-docs", type=int, default=4)
    parser.add_argument("--dynamic-retrieval-max-queries", type=int, default=4)
    parser.add_argument("--dynamic-retrieval-max-rejected-claims", type=int, default=4)
    parser.add_argument("--dynamic-retrieval-max-doc-pool", type=int, default=140)
    parser.add_argument("--dynamic-retrieval-query-max-chars", type=int, default=900)
    parser.add_argument("--dynamic-retrieval-min-query-terms", type=int, default=2)
    parser.add_argument("--dynamic-retrieval-sentence-boost", type=float, default=0.12)
    parser.add_argument(
        "--dynamic-retrieval-query-strategy",
        choices=["missing_facet", "question_facet", "hybrid"],
        default="missing_facet",
    )
    parser.add_argument("--dynamic-retrieval-min-score", type=float, default=0.0)
    parser.add_argument("--dynamic-retrieval-require-missing-signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-retrieval-use-supported-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coverage-guided-retrieval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coverage-guided-targets-per-round", type=int, default=3)
    parser.add_argument("--coverage-guided-min-targets", type=int, default=1)
    parser.add_argument("--coverage-guided-min-question-overlap", type=float, default=0.08)
    parser.add_argument("--coverage-guided-query-supported-claims", type=int, default=6)
    parser.add_argument("--coverage-guided-allow-evidence-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coverage-reranker-model", type=Path, default=None)
    parser.add_argument("--coverage-reranker-device", default=None)
    parser.add_argument("--coverage-reranker-batch-size", type=int, default=16)
    parser.add_argument("--coverage-reranker-max-length", type=int, default=384)
    parser.add_argument("--coverage-reranker-doc-prefilter", type=int, default=64)
    parser.add_argument("--coverage-reranker-doc-weight", type=float, default=0.35)
    parser.add_argument("--coverage-reranker-candidate-weight", type=float, default=0.20)

    parser.add_argument("--decomposer", choices=["llm", "sentence"], default="llm")
    parser.add_argument("--decompose-model", default="gpt-4o-mini")
    parser.add_argument("--verifier", choices=["nli", "llm", "lexical", "hybrid"], default="nli")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini")
    parser.add_argument("--nli-model", default=NLI_MODEL)
    parser.add_argument("--device", default=None, help="Optional NLI device passed to coverrag.py.")
    parser.add_argument("--nli-premise-mode", choices=["doc", "sentence_window", "doc_then_sentence"], default="doc")
    parser.add_argument("--nli-top-sentences", type=int, default=4)
    parser.add_argument("--nli-sentence-window-size", type=int, default=1)
    parser.add_argument("--entail-threshold", type=float, default=0.50)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--answer-format", choices=["auto", "paragraph", "qampari"], default="auto")
    parser.add_argument("--evidence-scope", choices=["cited", "all_docs", "cited_then_all"], default="cited")
    parser.add_argument("--citation-policy", choices=["source", "all", "minimal"], default="source")
    parser.add_argument("--sentence-citation-source", choices=["source", "verified", "hybrid"], default="source")
    parser.add_argument("--max-citations-per-claim", type=int, default=3)
    parser.add_argument("--max-citations-per-sentence", type=int, default=4)
    parser.add_argument("--max-verified-claims", type=int, default=24)
    parser.add_argument("--max-final-claims", type=int, default=0)
    parser.add_argument("--preserve-first-pass-claims", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-mode", choices=["citation_aligned", "span_preserving", "answer_preserving", "atomic"], default="citation_aligned")
    parser.add_argument("--preserve-answer-sentences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-rejected-critical-answer-sentences", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-preserved-source-words", type=int, default=48)
    parser.add_argument("--min-preserved-claim-fraction", type=float, default=0.50)
    parser.add_argument("--min-rejected-preserve-question-overlap", type=float, default=0.25)
    parser.add_argument("--max-qampari-items", type=int, default=80)
    parser.add_argument("--qampari-answer-relevance-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-background-claims", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--recover-labels", default="not_supported,no_citation,wrong_or_missing_citation")
    parser.add_argument("--recover-importance", default="critical,supporting")
    parser.add_argument("--recovery-top-k", type=int, default=6)
    parser.add_argument("--recovery-group-size", type=int, default=3)
    parser.add_argument("--min-recovery-score", type=float, default=0.03)
    parser.add_argument("--max-recovered-citations", type=int, default=3)

    parser.add_argument("--expansion-mode", choices=["llm", "extractive", "off"], default="llm")
    parser.add_argument("--expansion-model", default="gpt-4o-mini")
    parser.add_argument("--expansion-max-tokens", type=int, default=900)
    parser.add_argument("--expansion-output-mode", choices=["claims", "span_ids"], default="claims")
    parser.add_argument("--expansion-evidence-sentences", type=int, default=12)
    parser.add_argument("--expansion-candidate-spans", type=int, default=40)
    parser.add_argument("--max-selected-span-ids", type=int, default=8)
    parser.add_argument("--expansion-include-extractive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expansion-filter-general-answer-spans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expansion-min-question-overlap", type=float, default=0.0)
    parser.add_argument("--verify-expansion-with-sentence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-expansion-candidates", type=int, default=10)
    parser.add_argument("--max-expansion-verifications", type=int, default=8)
    parser.add_argument("--max-expanded-claims", type=int, default=6)
    parser.add_argument("--min-expansion-score", type=float, default=0.08)
    parser.add_argument("--min-answer-unit-score", type=float, default=0.25)
    parser.add_argument("--strict-expansion-answer-unit-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-expansion-min-target-relevance", type=float, default=0.45)
    parser.add_argument("--answer-type-gate-mode", choices=["off", "soft", "strict"], default="soft")
    parser.add_argument("--max-expansion-claim-words", type=int, default=22)
    parser.add_argument("--skip-covered-answer-spans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-expansion-answer-span", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--expansion-query-source",
        choices=["question", "question_and_evidence_core", "question_evidence_and_rejected"],
        default="question_and_evidence_core",
    )
    parser.add_argument("--expansion-use-intermediate-answer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--answer-targeted-expansion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--answer-targeted-extractive-top-k", type=int, default=2)
    parser.add_argument("--answer-targeted-min-sentence-overlap", type=float, default=0.16)
    parser.add_argument("--explanatory-expansion-require-phrase-span", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explanatory-title-span-mode", choices=["off", "on"], default="off")
    parser.add_argument("--max-explanatory-expanded-claims", type=int, default=2)
    parser.add_argument("--max-explanatory-expansion-verifications", type=int, default=4)
    parser.add_argument("--min-explanatory-answer-unit-score", type=float, default=0.35)
    parser.add_argument("--explanatory-precision-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explanatory-precision-min-target-relevance", type=float, default=0.58)
    parser.add_argument("--explanatory-precision-min-question-relevance", type=float, default=0.18)
    parser.add_argument("--explanation-integrated-refinement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-explanation-integrated-claims", type=int, default=1)
    parser.add_argument("--max-explanation-length-growth", type=float, default=0.10)
    parser.add_argument("--candidate-output-selection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-output-recall-claims", type=int, default=2)
    parser.add_argument("--candidate-output-recall-growth", type=float, default=0.18)
    parser.add_argument("--candidate-output-min-support", type=float, default=0.94)
    parser.add_argument("--candidate-output-support-tolerance", type=float, default=0.015)
    parser.add_argument("--candidate-output-max-unsupported-rate", type=float, default=0.06)
    parser.add_argument("--candidate-output-min-answer-unit-gain", type=float, default=0.0)
    parser.add_argument("--candidate-output-min-supported-claim-gain", type=int, default=0)
    parser.add_argument("--recall-oriented-completion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recall-completion-max-targets", type=int, default=10)
    parser.add_argument("--recall-completion-rejected-targets", type=int, default=4)
    parser.add_argument("--recall-completion-evidence-sentences", type=int, default=24)
    parser.add_argument("--recall-completion-supported-claims", type=int, default=18)
    parser.add_argument("--recall-completion-min-sentence-overlap", type=float, default=0.0)
    parser.add_argument("--recall-completion-min-target-overlap", type=float, default=0.10)
    parser.add_argument("--answer-unit-relevance-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--answer-unit-min-directness", type=float, default=0.44)
    parser.add_argument("--explanatory-answer-unit-min-directness", type=float, default=0.58)
    parser.add_argument("--iterative-completion-rounds", type=int, default=0)
    parser.add_argument("--iterative-completion-max-new-claims-per-round", type=int, default=2)
    parser.add_argument("--iterative-completion-max-verifications-per-round", type=int, default=4)
    parser.add_argument("--iterative-completion-evidence-sentences", type=int, default=0)
    parser.add_argument("--iterative-completion-candidate-spans", type=int, default=0)
    parser.add_argument("--iterative-completion-max-candidates", type=int, default=0)

    parser.add_argument("--cache-file", type=Path, default=Path("cache/coverrag_cache.json"))
    parser.add_argument(
        "--cache-save-every",
        type=int,
        default=1,
        help="Pass through to coverrag.py to reduce JSON cache write frequency.",
    )
    parser.add_argument(
        "--revision-workers",
        type=int,
        default=1,
        help="Pass through to coverrag.py --workers for concurrent per-example revision.",
    )
    parser.add_argument("--final-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-revise", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval.py; merge only COVER-side metrics.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--no-citations", action="store_true")
    parser.add_argument("--no-qa", action="store_true")
    parser.add_argument("--no-mauve", action="store_true")
    parser.add_argument("--no-claims-nli", action="store_true")
    parser.add_argument(
        "--eval-fallback-no-citations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If eval.py fails while computing ALCE citation metrics, retry without --citations.",
    )
    parser.add_argument("--claim-citation-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-csv", type=Path, default=Path("result/coverrag/coverrag_comparison.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("result/coverrag/coverrag_comparison.md"))
    return parser.parse_args()


def coverrag_path(raw: Path, output_dir: Path) -> Path:
    return output_dir / f"{raw.stem}.coverrag.json"


def build_revise_cmd(args: argparse.Namespace, root: Path, raw: Path, out: Path) -> list[str | Path]:
    cmd: list[str | Path] = [
        args.python,
        root / "scripts" / "coverrag.py",
        "--input",
        raw,
        "--output",
        out,
        "--openai-api",
        "--control-mode",
        args.control_mode,
        "--reader-model",
        args.reader_model,
        "--reader-max-tokens",
        str(args.reader_max_tokens),
        "--reader-max-plan-claims",
        str(args.reader_max_plan_claims),
        "--reader-max-unit-words",
        str(args.reader_max_unit_words),
        "--reader-conformance-verify" if args.reader_conformance_verify else "--no-reader-conformance-verify",
        "--evidence-seed-max-claims",
        str(args.evidence_seed_max_claims),
        "--evidence-seed-max-verifications",
        str(args.evidence_seed_max_verifications),
        "--evidence-seed-evidence-sentences",
        str(args.evidence_seed_evidence_sentences),
        "--evidence-seed-candidate-spans",
        str(args.evidence_seed_candidate_spans),
        "--evidence-seed-max-candidates",
        str(args.evidence_seed_max_candidates),
        "--decomposer",
        args.decomposer,
        "--decompose-model",
        args.decompose_model,
        "--verifier",
        args.verifier,
        "--nli-premise-mode",
        args.nli_premise_mode,
        "--nli-top-sentences",
        str(args.nli_top_sentences),
        "--nli-sentence-window-size",
        str(args.nli_sentence_window_size),
        "--entail-threshold",
        str(args.entail_threshold),
        "--contradiction-threshold",
        str(args.contradiction_threshold),
        "--ambiguous-margin",
        str(args.ambiguous_margin),
        "--answer-format",
        args.answer_format,
        "--evidence-scope",
        args.evidence_scope,
        "--citation-policy",
        args.citation_policy,
        "--sentence-citation-source",
        args.sentence_citation_source,
        "--max-citations-per-claim",
        str(args.max_citations_per_claim),
        "--max-citations-per-sentence",
        str(args.max_citations_per_sentence),
        "--max-verified-claims",
        str(args.max_verified_claims),
        "--max-final-claims",
        str(args.max_final_claims),
        "--render-mode",
        args.render_mode,
        "--max-preserved-source-words",
        str(args.max_preserved_source_words),
        "--min-preserved-claim-fraction",
        str(args.min_preserved_claim_fraction),
        "--min-rejected-preserve-question-overlap",
        str(args.min_rejected_preserve_question_overlap),
        "--max-qampari-items",
        str(args.max_qampari_items),
        "--recover-labels",
        args.recover_labels,
        "--recover-importance",
        args.recover_importance,
        "--recovery-top-k",
        str(args.recovery_top_k),
        "--recovery-group-size",
        str(args.recovery_group_size),
        "--min-recovery-score",
        str(args.min_recovery_score),
        "--max-recovered-citations",
        str(args.max_recovered_citations),
        "--expansion-mode",
        args.expansion_mode,
        "--expansion-model",
        args.expansion_model,
        "--expansion-max-tokens",
        str(args.expansion_max_tokens),
        "--expansion-output-mode",
        args.expansion_output_mode,
        "--expansion-evidence-sentences",
        str(args.expansion_evidence_sentences),
        "--expansion-candidate-spans",
        str(args.expansion_candidate_spans),
        "--max-selected-span-ids",
        str(args.max_selected_span_ids),
        "--max-expansion-candidates",
        str(args.max_expansion_candidates),
        "--max-expansion-verifications",
        str(args.max_expansion_verifications),
        "--max-expanded-claims",
        str(args.max_expanded_claims),
        "--min-expansion-score",
        str(args.min_expansion_score),
        "--min-answer-unit-score",
        str(args.min_answer_unit_score),
        "--strict-expansion-min-target-relevance",
        str(args.strict_expansion_min_target_relevance),
        "--answer-type-gate-mode",
        args.answer_type_gate_mode,
        "--max-expansion-claim-words",
        str(args.max_expansion_claim_words),
        "--expansion-query-source",
        args.expansion_query_source,
        "--expansion-use-intermediate-answer"
        if args.expansion_use_intermediate_answer
        else "--no-expansion-use-intermediate-answer",
        "--answer-targeted-expansion" if args.answer_targeted_expansion else "--no-answer-targeted-expansion",
        "--answer-targeted-extractive-top-k",
        str(args.answer_targeted_extractive_top_k),
        "--answer-targeted-min-sentence-overlap",
        str(args.answer_targeted_min_sentence_overlap),
        (
            "--explanatory-expansion-require-phrase-span"
            if args.explanatory_expansion_require_phrase_span
            else "--no-explanatory-expansion-require-phrase-span"
        ),
        "--explanatory-title-span-mode",
        args.explanatory_title_span_mode,
        "--max-explanatory-expanded-claims",
        str(args.max_explanatory_expanded_claims),
        "--max-explanatory-expansion-verifications",
        str(args.max_explanatory_expansion_verifications),
        "--min-explanatory-answer-unit-score",
        str(args.min_explanatory_answer_unit_score),
        "--explanatory-precision-gate" if args.explanatory_precision_gate else "--no-explanatory-precision-gate",
        "--explanatory-precision-min-target-relevance",
        str(args.explanatory_precision_min_target_relevance),
        "--explanatory-precision-min-question-relevance",
        str(args.explanatory_precision_min_question_relevance),
        (
            "--explanation-integrated-refinement"
            if args.explanation_integrated_refinement
            else "--no-explanation-integrated-refinement"
        ),
        "--max-explanation-integrated-claims",
        str(args.max_explanation_integrated_claims),
        "--max-explanation-length-growth",
        str(args.max_explanation_length_growth),
        "--candidate-output-selection" if args.candidate_output_selection else "--no-candidate-output-selection",
        "--candidate-output-recall-claims",
        str(args.candidate_output_recall_claims),
        "--candidate-output-recall-growth",
        str(args.candidate_output_recall_growth),
        "--candidate-output-min-support",
        str(args.candidate_output_min_support),
        "--candidate-output-support-tolerance",
        str(args.candidate_output_support_tolerance),
        "--candidate-output-max-unsupported-rate",
        str(args.candidate_output_max_unsupported_rate),
        "--candidate-output-min-answer-unit-gain",
        str(args.candidate_output_min_answer_unit_gain),
        "--candidate-output-min-supported-claim-gain",
        str(args.candidate_output_min_supported_claim_gain),
        "--recall-oriented-completion" if args.recall_oriented_completion else "--no-recall-oriented-completion",
        "--recall-completion-max-targets",
        str(args.recall_completion_max_targets),
        "--recall-completion-rejected-targets",
        str(args.recall_completion_rejected_targets),
        "--recall-completion-evidence-sentences",
        str(args.recall_completion_evidence_sentences),
        "--recall-completion-supported-claims",
        str(args.recall_completion_supported_claims),
        "--recall-completion-min-sentence-overlap",
        str(args.recall_completion_min_sentence_overlap),
        "--recall-completion-min-target-overlap",
        str(args.recall_completion_min_target_overlap),
        "--answer-unit-min-directness",
        str(args.answer_unit_min_directness),
        "--explanatory-answer-unit-min-directness",
        str(args.explanatory_answer_unit_min_directness),
        "--iterative-completion-rounds",
        str(args.iterative_completion_rounds),
        "--iterative-completion-max-new-claims-per-round",
        str(args.iterative_completion_max_new_claims_per_round),
        "--iterative-completion-max-verifications-per-round",
        str(args.iterative_completion_max_verifications_per_round),
        "--iterative-completion-evidence-sentences",
        str(args.iterative_completion_evidence_sentences),
        "--iterative-completion-candidate-spans",
        str(args.iterative_completion_candidate_spans),
        "--iterative-completion-max-candidates",
        str(args.iterative_completion_max_candidates),
        "--max-doc-pool",
        str(args.max_doc_pool),
        "--dynamic-retrieval-mode",
        args.dynamic_retrieval_mode,
        "--dynamic-retrieval-top-k",
        str(args.dynamic_retrieval_top_k),
        "--dynamic-retrieval-per-query-docs",
        str(args.dynamic_retrieval_per_query_docs),
        "--dynamic-retrieval-max-queries",
        str(args.dynamic_retrieval_max_queries),
        "--dynamic-retrieval-max-rejected-claims",
        str(args.dynamic_retrieval_max_rejected_claims),
        "--dynamic-retrieval-max-doc-pool",
        str(args.dynamic_retrieval_max_doc_pool),
        "--dynamic-retrieval-query-max-chars",
        str(args.dynamic_retrieval_query_max_chars),
        "--dynamic-retrieval-min-query-terms",
        str(args.dynamic_retrieval_min_query_terms),
        "--dynamic-retrieval-sentence-boost",
        str(args.dynamic_retrieval_sentence_boost),
        "--dynamic-retrieval-query-strategy",
        args.dynamic_retrieval_query_strategy,
        "--dynamic-retrieval-min-score",
        str(args.dynamic_retrieval_min_score),
        "--coverage-guided-targets-per-round",
        str(args.coverage_guided_targets_per_round),
        "--coverage-guided-min-targets",
        str(args.coverage_guided_min_targets),
        "--coverage-guided-min-question-overlap",
        str(args.coverage_guided_min_question_overlap),
        "--coverage-guided-query-supported-claims",
        str(args.coverage_guided_query_supported_claims),
        "--coverage-reranker-batch-size",
        str(args.coverage_reranker_batch_size),
        "--coverage-reranker-max-length",
        str(args.coverage_reranker_max_length),
        "--coverage-reranker-doc-prefilter",
        str(args.coverage_reranker_doc_prefilter),
        "--coverage-reranker-doc-weight",
        str(args.coverage_reranker_doc_weight),
        "--coverage-reranker-candidate-weight",
        str(args.coverage_reranker_candidate_weight),
        "--cache-file",
        resolve(root, args.cache_file),
        "--cache-save-every",
        str(args.cache_save_every),
        "--workers",
        str(args.revision_workers),
        "--pretty",
    ]
    if args.candidate_docs_file is not None:
        cmd.extend(["--candidate-docs-file", resolve(root, args.candidate_docs_file)])
    if args.coverage_reranker_model is not None:
        cmd.extend(["--coverage-reranker-model", resolve(root, args.coverage_reranker_model)])
    if args.coverage_reranker_device:
        cmd.extend(["--coverage-reranker-device", args.coverage_reranker_device])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.dynamic_retrieval_require_missing_signal:
        cmd.append("--dynamic-retrieval-require-missing-signal")
    else:
        cmd.append("--no-dynamic-retrieval-require-missing-signal")
    if args.dynamic_retrieval_use_supported_answer:
        cmd.append("--dynamic-retrieval-use-supported-answer")
    else:
        cmd.append("--no-dynamic-retrieval-use-supported-answer")
    if args.coverage_guided_retrieval:
        cmd.append("--coverage-guided-retrieval")
    else:
        cmd.append("--no-coverage-guided-retrieval")
    if args.coverage_guided_allow_evidence_targets:
        cmd.append("--coverage-guided-allow-evidence-targets")
    else:
        cmd.append("--no-coverage-guided-allow-evidence-targets")
    if args.expansion_include_extractive:
        cmd.append("--expansion-include-extractive")
    else:
        cmd.append("--no-expansion-include-extractive")
    if args.expansion_filter_general_answer_spans:
        cmd.append("--expansion-filter-general-answer-spans")
    else:
        cmd.append("--no-expansion-filter-general-answer-spans")
    if args.strict_expansion_answer_unit_gate:
        cmd.append("--strict-expansion-answer-unit-gate")
    else:
        cmd.append("--no-strict-expansion-answer-unit-gate")
    if args.answer_unit_relevance_gate:
        cmd.append("--answer-unit-relevance-gate")
    else:
        cmd.append("--no-answer-unit-relevance-gate")
    cmd.extend(["--expansion-min-question-overlap", str(args.expansion_min_question_overlap)])
    if args.verify_expansion_with_sentence:
        cmd.append("--verify-expansion-with-sentence")
    else:
        cmd.append("--no-verify-expansion-with-sentence")
    if args.require_expansion_answer_span:
        cmd.append("--require-expansion-answer-span")
    else:
        cmd.append("--no-require-expansion-answer-span")
    if args.preserve_first_pass_claims:
        cmd.append("--preserve-first-pass-claims")
    else:
        cmd.append("--no-preserve-first-pass-claims")
    if args.preserve_answer_sentences:
        cmd.append("--preserve-answer-sentences")
    else:
        cmd.append("--no-preserve-answer-sentences")
    if args.preserve_rejected_critical_answer_sentences:
        cmd.append("--preserve-rejected-critical-answer-sentences")
    else:
        cmd.append("--no-preserve-rejected-critical-answer-sentences")
    if args.qampari_answer_relevance_filter:
        cmd.append("--qampari-answer-relevance-filter")
    else:
        cmd.append("--no-qampari-answer-relevance-filter")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.checkpoint_every:
        cmd.extend(["--checkpoint-every", str(args.checkpoint_every)])
    if args.include_background_claims:
        cmd.append("--include-background-claims")
    else:
        cmd.append("--no-include-background-claims")
    if args.skip_covered_answer_spans:
        cmd.append("--skip-covered-answer-spans")
    else:
        cmd.append("--no-skip-covered-answer-spans")
    if args.final_audit:
        cmd.append("--final-audit")
    else:
        cmd.append("--no-final-audit")
    if args.verifier in {"nli", "hybrid"}:
        cmd.extend(["--nli-model", args.nli_model])
    if args.verifier in {"llm", "hybrid"}:
        cmd.extend(["--llm-verify-model", args.llm_verify_model])
    return cmd


def build_eval_cmd(args: argparse.Namespace, root: Path, cover: Path, citations: bool = True) -> list[str | Path]:
    cmd: list[str | Path] = [args.python, root / "eval.py", "--f", cover]
    dataset = infer_dataset_from_path(cover)
    use_citations = citations and not args.no_citations
    if dataset == "asqa":
        if use_citations:
            cmd.append("--citations")
        if not args.no_qa:
            cmd.append("--qa")
        if not args.no_mauve:
            cmd.append("--mauve")
    elif dataset == "eli5":
        if use_citations:
            cmd.append("--citations")
        if not args.no_claims_nli:
            cmd.append("--claims_nli")
        if not args.no_mauve:
            cmd.append("--mauve")
    elif dataset == "qampari":
        if use_citations:
            cmd.append("--citations")
    else:
        if use_citations:
            cmd.append("--citations")
        if not args.no_qa:
            cmd.append("--qa")
        if not args.no_mauve:
            cmd.append("--mauve")
    return cmd


def run_eval_with_fallback(args: argparse.Namespace, root: Path, cover: Path) -> None:
    try:
        run(build_eval_cmd(args, root, cover, citations=True), root, args.dry_run)
    except Exception:
        if args.no_citations or not args.eval_fallback_no_citations:
            raise
        print("  eval.py with --citations failed; retrying without --citations for task metrics.", flush=True)
        run(build_eval_cmd(args, root, cover, citations=False), root, args.dry_run)


def write_summary(args: argparse.Namespace, root: Path, cover_scores: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in cover_scores:
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=sort_key)
    if not rows:
        print("No cover_score files available for summary.")
        return []

    csv_path = resolve(root, args.summary_csv)
    md_path = resolve(root, args.summary_md)
    if args.dry_run:
        print(f"Would write {display(root, csv_path)}")
        print(f"Would write {display(root, md_path)}")
        return rows

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [header for header, _, _ in SUMMARY_COLUMNS]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: fmt(row.get(key), kind) for header, key, kind in SUMMARY_COLUMNS})

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key), kind, md=True) for _, key, kind in SUMMARY_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote summary CSV: {display(root, csv_path)}")
    print(f"Wrote summary Markdown: {display(root, md_path)}")
    return rows


def main() -> int:
    args = parse_args()
    root = root_dir()
    output_dir = resolve(root, args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolve(root, args.cache_file).parent.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(args, root)
    print(f"Found {len(inputs)} raw result files:")
    for path in inputs:
        print("  - " + display(root, path))

    failures: list[tuple[Path, str]] = []
    cover_scores: list[Path] = []
    for index, raw in enumerate(inputs, start=1):
        cover = coverrag_path(raw, output_dir)
        cscore = cover_score_path(cover)
        cover_scores.append(cscore)
        print(f"\n[{index}/{len(inputs)}] {display(root, raw)}", flush=True)
        if args.summary_only:
            continue
        try:
            if args.force_revise or not cover.exists():
                run(build_revise_cmd(args, root, raw, cover), root, args.dry_run)
            else:
                print(f"  coverrag exists, skip: {display(root, cover)}")

            sp = score_path(cover)
            if args.skip_eval:
                print("  skip eval.py because --skip-eval is set")
            elif args.force_eval or not sp.exists():
                run_eval_with_fallback(args, root, cover)
            else:
                print(f"  score exists, skip: {display(root, sp)}")

            if args.claim_citation_metrics:
                run(build_claim_citation_cmd(args, root, cover), root, args.dry_run)

            if args.force_merge or not cscore.exists():
                run(build_merge_cmd(args, root, cover), root, args.dry_run)
            else:
                print(f"  cover_score exists, skip: {display(root, cscore)}")
        except Exception as exc:
            failures.append((raw, str(exc)))
            print(f"  FAILED: {exc}", flush=True)
            if args.stop_on_error:
                break

    write_summary(args, root, cover_scores)
    if failures:
        print("\nFailures:")
        for path, message in failures:
            print(f"- {display(root, path)}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
