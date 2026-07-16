#!/usr/bin/env python3
"""Batch runner for COVER-RAG v2.

This mirrors the existing v0/v1 batch scripts:

    raw ALCE result JSON
    -> cover_v2_revise.py
    -> eval.py
    -> merge_cover_scores.py
    -> summary CSV/Markdown
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
    ("qampari_f1", "alce_qampari_f1", "float"),
    ("qampari_f1_top5", "alce_qampari_f1_top5", "float"),
    ("citation_recall", "alce_citation_rec", "float"),
    ("citation_precision", "alce_citation_prec", "float"),
    ("v2_examples", "cover_v2_summary_num_examples", "int"),
    ("v2_draft_claims", "cover_v2_summary_draft_num_claims", "int"),
    ("v2_initial_supported", "cover_v2_summary_initially_supported_claims", "int"),
    ("v2_recovered_claims", "cover_v2_summary_recovered_claims", "int"),
    ("v2_kept_claims", "cover_v2_summary_kept_verified_claims", "int"),
    ("v2_rejected_claims", "cover_v2_summary_rejected_claims", "int"),
    ("v2_recovery_attempted", "cover_v2_summary_recovery_attempted_claims", "int"),
    ("v2_recovery_successful", "cover_v2_summary_recovery_successful_claims", "int"),
    ("v2_recovery_success_rate", "cover_v2_summary_recovery_success_rate_micro", "rate"),
    ("v2_keep_rate", "cover_v2_summary_claim_keep_rate_micro", "rate"),
    ("v2_rejected_rate", "cover_v2_summary_rejected_claim_rate_micro", "rate"),
    ("v2_avg_revised_length", "cover_v2_summary_avg_revised_length", "float"),
    ("v2_avg_revised_citations", "cover_v2_summary_avg_revised_num_citations", "float"),
    ("v2_avg_doc_pool_size", "cover_v2_summary_avg_doc_pool_size", "float"),
    ("v2_draft_supported", "cover_v2_summary_draft_label_counts_supported", "int"),
    ("v2_draft_wrong_missing", "cover_v2_summary_draft_label_counts_wrong_or_missing_citation", "int"),
    ("v2_draft_not_supported", "cover_v2_summary_draft_label_counts_not_supported", "int"),
    ("v2_draft_no_citation", "cover_v2_summary_draft_label_counts_no_citation", "int"),
    ("final_claims", "cover_v2_final_audit_summary_num_claims", "int"),
    ("final_claim_support_rate", "cover_v2_final_audit_summary_atomic_claim_support_rate_micro", "rate"),
    ("final_claim_unsupported_rate", "cover_v2_final_audit_summary_unsupported_claim_rate_micro", "rate"),
    ("final_no_citation_claims", "cover_v2_final_audit_summary_no_citation_claims", "int"),
]

DATASET_ORDER = {"asqa": 0, "eli5": 1, "qampari": 2}
TAG_ORDER = {"gtr": 0, "bm25": 1, "bge_rerank_default": 2, "reranked_oracle": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run COVER-RAG v2.")
    parser.add_argument("--result-dir", type=Path, default=Path("result/origin"))
    parser.add_argument("--pattern", default="*-gpt-4o-mini-*-shot2-ndoc5-42.json")
    parser.add_argument("--inputs", nargs="*", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("result/cover_v2"))
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--candidate-docs-file", type=Path, default=None)
    parser.add_argument("--max-doc-pool", type=int, default=100)

    parser.add_argument("--decomposer", choices=["llm", "sentence", "claimify"], default="llm")
    parser.add_argument("--decompose-model", default="gpt-4o-mini")
    parser.add_argument("--verifier", choices=["nli", "llm", "lexical", "hybrid"], default="nli")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini")
    parser.add_argument("--nli-model", default=NLI_MODEL)
    parser.add_argument("--revision-mode", choices=["llm", "template", "atomic", "qampari", "auto"], default="auto")
    parser.add_argument("--v2-render-mode", choices=["source_aware", "alce_llm", "v1"], default="source_aware")
    parser.add_argument("--alce-llm-max-docs", type=int, default=8)
    parser.add_argument("--alce-llm-max-claims", type=int, default=18)
    parser.add_argument("--alce-llm-max-doc-chars", type=int, default=900)
    parser.add_argument("--revision-model", default="gpt-4o-mini")
    parser.add_argument("--answer-format", choices=["auto", "paragraph", "qampari"], default="auto")
    parser.add_argument("--evidence-scope", choices=["cited", "all_docs", "cited_then_all"], default="cited")
    parser.add_argument("--citation-policy", choices=["source", "all", "minimal"], default="source")
    parser.add_argument("--max-citations-per-claim", type=int, default=3)
    parser.add_argument("--max-verified-claims", type=int, default=18)
    parser.add_argument("--max-qampari-items", type=int, default=80)
    parser.add_argument("--include-background-claims", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--recover-labels", default="not_supported,no_citation,wrong_or_missing_citation")
    parser.add_argument("--recover-importance", default="critical,supporting")
    parser.add_argument("--recovery-top-k", type=int, default=6)
    parser.add_argument("--recovery-group-size", type=int, default=3)
    parser.add_argument("--min-recovery-score", type=float, default=0.03)
    parser.add_argument("--max-recovered-citations", type=int, default=3)

    parser.add_argument("--cache-file", type=Path, default=Path("cache/cover_v2_cache.json"))
    parser.add_argument("--final-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-revise", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--no-citations", action="store_true")
    parser.add_argument("--no-qa", action="store_true")
    parser.add_argument("--no-mauve", action="store_true")
    parser.add_argument("--no-claims-nli", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=Path("result/cover_v2/cover_v2_comparison.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("result/cover_v2/cover_v2_comparison.md"))
    return parser.parse_args()


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
    bad_markers = [".cover_v1", ".cover_v2", ".cover_audit", ".cover_score", ".score", ".debug", ".smoke"]
    return not any(marker in path.name for marker in bad_markers)


def discover_inputs(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.inputs:
        files = [resolve(root, p) for p in args.inputs]
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


def cover_v2_path(raw: Path, output_dir: Path) -> Path:
    return output_dir / f"{raw.stem}.cover_v2.json"


def score_path(path: Path) -> Path:
    return Path(str(path) + ".score")


def cover_score_path(path: Path) -> Path:
    return Path(str(path) + ".cover_score")


def build_revise_cmd(args: argparse.Namespace, root: Path, raw: Path, out: Path) -> list[str | Path]:
    cmd: list[str | Path] = [
        args.python,
        root / "scripts" / "cover_v2_revise.py",
        "--input",
        raw,
        "--output",
        out,
        "--openai-api",
        "--decomposer",
        args.decomposer,
        "--decompose-model",
        args.decompose_model,
        "--verifier",
        args.verifier,
        "--revision-mode",
        args.revision_mode,
        "--v2-render-mode",
        args.v2_render_mode,
        "--answer-format",
        args.answer_format,
        "--alce-llm-max-docs",
        str(args.alce_llm_max_docs),
        "--alce-llm-max-claims",
        str(args.alce_llm_max_claims),
        "--alce-llm-max-doc-chars",
        str(args.alce_llm_max_doc_chars),
        "--evidence-scope",
        args.evidence_scope,
        "--citation-policy",
        args.citation_policy,
        "--max-citations-per-claim",
        str(args.max_citations_per_claim),
        "--max-verified-claims",
        str(args.max_verified_claims),
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
        "--max-doc-pool",
        str(args.max_doc_pool),
        "--cache-file",
        resolve(root, args.cache_file),
        "--pretty",
    ]
    if args.candidate_docs_file is not None:
        cmd.extend(["--candidate-docs-file", resolve(root, args.candidate_docs_file)])
    if args.include_background_claims:
        cmd.append("--include-background-claims")
    else:
        cmd.append("--no-include-background-claims")
    if args.final_audit:
        cmd.append("--final-audit")
    else:
        cmd.append("--no-final-audit")
    if args.verifier in {"nli", "hybrid"}:
        cmd.extend(["--nli-model", args.nli_model])
    if args.verifier in {"llm", "hybrid"}:
        cmd.extend(["--llm-verify-model", args.llm_verify_model])
    if args.revision_mode == "llm":
        cmd.extend(["--revision-model", args.revision_model])
    return cmd


def infer_dataset_from_path(path: Path) -> str:
    name = path.name.lower()
    for dataset in ("asqa", "eli5", "qampari"):
        if name.startswith(dataset + "-"):
            return dataset
    return ""


def build_eval_cmd(args: argparse.Namespace, root: Path, cover: Path) -> list[str | Path]:
    cmd: list[str | Path] = [args.python, root / "eval.py", "--f", cover]
    dataset = infer_dataset_from_path(cover)
    if dataset == "asqa":
        if not args.no_citations:
            cmd.append("--citations")
        if not args.no_qa:
            cmd.append("--qa")
        if not args.no_mauve:
            cmd.append("--mauve")
    elif dataset == "eli5":
        if not args.no_citations:
            cmd.append("--citations")
        if not args.no_claims_nli:
            cmd.append("--claims_nli")
        if not args.no_mauve:
            cmd.append("--mauve")
    elif dataset == "qampari":
        if not args.no_citations:
            cmd.append("--citations")
    else:
        if not args.no_citations:
            cmd.append("--citations")
        if not args.no_qa:
            cmd.append("--qa")
        if not args.no_mauve:
            cmd.append("--mauve")
    return cmd


def build_merge_cmd(args: argparse.Namespace, root: Path, cover: Path) -> list[str | Path]:
    cmd: list[str | Path] = [args.python, root / "scripts" / "merge_cover_scores.py", "--result", cover]
    sp = score_path(cover)
    if sp.exists():
        cmd.extend(["--alce-score", sp])
    cmd.append("--pretty")
    return cmd


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
        cover = cover_v2_path(raw, output_dir)
        cscore = cover_score_path(cover)
        cover_scores.append(cscore)
        print(f"\n[{index}/{len(inputs)}] {display(root, raw)}", flush=True)
        if args.summary_only:
            continue
        try:
            if args.force_revise or not cover.exists():
                run(build_revise_cmd(args, root, raw, cover), root, args.dry_run)
            else:
                print(f"  cover_v2 exists, skip: {display(root, cover)}")

            sp = score_path(cover)
            if args.force_eval or not sp.exists():
                run(build_eval_cmd(args, root, cover), root, args.dry_run)
            else:
                print(f"  score exists, skip: {display(root, sp)}")

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
