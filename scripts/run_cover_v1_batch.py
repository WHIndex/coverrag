#!/usr/bin/env python3
"""
Batch run COVER v1 revise -> ALCE eval -> merge COVER scores -> summary table.

Run from the repository/alce root, for example:

  python scripts/run_cover_v1_batch.py

By default this finds raw result files under result/:

  *-gpt-4o-mini-*-shot2-ndoc5-42.json

and excludes generated files such as:

  *.cover_v1.json
  *.cover_audit.json
  *.score
  *.cover_score
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

# Output table fields. Missing fields stay blank.
SUMMARY_COLUMNS: list[tuple[str, str, str]] = [
    ("dataset", "dataset", "text"),
    ("tag", "tag", "text"),
    ("model", "model", "text"),
    ("shot", "shot", "int"),
    ("ndoc", "ndoc", "int"),

    # ALCE metrics.
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

    # Generic COVER columns. For V1 without --final-audit:
    # - cover_claims = draft claims
    # - cover_support_micro_% = draft supported / draft claims
    # - cover_unsupported_% = rejected claims / draft claims
    ("claim_count", "cover_num_claims", "int"),
    ("claim_support_rate", "cover_support_micro", "rate"),
    ("claim_support_rate_macro", "cover_support_macro", "rate"),
    ("claim_unsupported_rate", "cover_unsupported_micro", "rate"),
    ("avg_claims_per_example", "cover_avg_claims", "float"),
    ("no_citation_claim_count", "cover_no_citation_claims", "int"),
    ("wrong_missing_citation_claim_count", "cover_wrong_or_missing_citation_claims", "int"),

    # COVER v1-specific revision metrics.
    ("v1_examples", "cover_v1_summary_num_examples", "int"),
    ("v1_draft_claims", "cover_v1_summary_draft_num_claims", "int"),
    ("v1_kept_claims", "cover_v1_summary_kept_verified_claims", "int"),
    ("v1_rejected_claims", "cover_v1_summary_rejected_claims", "int"),
    ("v1_keep_rate", "cover_v1_summary_claim_keep_rate_micro", "rate"),
    ("v1_rejected_rate", "cover_v1_summary_rejected_claim_rate_micro", "rate"),
    ("v1_avg_revised_length", "cover_v1_summary_avg_revised_length", "float"),
    ("v1_avg_revised_citations", "cover_v1_summary_avg_revised_num_citations", "float"),
    ("v1_draft_supported", "cover_v1_summary_draft_label_counts_supported", "int"),
    ("v1_draft_supported_rate", "cover_v1_draft_supported_rate", "rate"),
    ("v1_draft_wrong_missing", "cover_v1_summary_draft_label_counts_wrong_or_missing_citation", "int"),
    ("v1_draft_wrong_missing_rate", "cover_v1_draft_wrong_missing_rate", "rate"),
    ("v1_draft_not_supported", "cover_v1_summary_draft_label_counts_not_supported", "int"),
    ("v1_draft_no_citation", "cover_v1_summary_draft_label_counts_no_citation", "int"),

    # Populated only when cover_v1_revise.py was run with --final-audit.
    ("final_claims", "cover_v1_final_audit_summary_num_claims", "int"),
    ("final_claim_support_rate", "cover_v1_final_audit_summary_atomic_claim_support_rate_micro", "rate"),
    ("final_claim_unsupported_rate", "cover_v1_final_audit_summary_unsupported_claim_rate_micro", "rate"),
]


COMBINED_COLUMNS: list[tuple[str, str, str]] = [
    ("dataset", "dataset", "text"),
    ("tag", "tag", "text"),
    ("model", "model", "text"),
    ("task_metric", "task_metric", "text"),
    ("v0_task_score", "v0_task_score", "float"),
    ("v1_task_score", "v1_task_score", "float"),
    ("task_delta", "task_delta", "float"),
    ("v0_mauve", "v0_mauve", "float"),
    ("v1_mauve", "v1_mauve", "float"),
    ("mauve_delta", "mauve_delta", "float"),
    ("v0_citation_recall", "v0_citation_recall", "float"),
    ("v1_citation_recall", "v1_citation_recall", "float"),
    ("citation_recall_delta", "citation_recall_delta", "float"),
    ("v0_citation_precision", "v0_citation_precision", "float"),
    ("v1_citation_precision", "v1_citation_precision", "float"),
    ("citation_precision_delta", "citation_precision_delta", "float"),
    ("v0_claim_count", "v0_claim_count", "int"),
    ("v0_claim_support_rate", "v0_claim_support_rate", "float"),
    ("v0_claim_unsupported_rate", "v0_claim_unsupported_rate", "float"),
    ("v0_no_citation_claim_count", "v0_no_citation_claim_count", "int"),
    ("v0_wrong_missing_citation_claim_count", "v0_wrong_missing_citation_claim_count", "int"),
    ("v1_draft_claims", "v1_draft_claims", "int"),
    ("v1_draft_claim_support_rate", "v1_draft_claim_support_rate", "float"),
    ("v1_draft_wrong_missing_citation_rate", "v1_draft_wrong_missing_citation_rate", "float"),
    ("v1_draft_not_supported", "v1_draft_not_supported", "int"),
    ("v1_draft_no_citation", "v1_draft_no_citation", "int"),
    ("v1_keep_rate", "v1_keep_rate", "float"),
    ("v1_rejected_rate", "v1_rejected_rate", "float"),
    ("v1_revised_length", "v1_revised_length", "float"),
    ("v1_revised_citations", "v1_revised_citations", "float"),
    ("final_claims", "final_claims", "int"),
    ("final_claim_support_rate", "final_claim_support_rate", "float"),
    ("final_claim_unsupported_rate", "final_claim_unsupported_rate", "float"),
]


DATASET_ORDER = {"asqa": 0, "eli5": 1, "qampari": 2}
TAG_ORDER = {
    "gtr": 0,
    "bm25": 1,
    "bge_rerank_default": 2,
    "reranked_oracle": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/cover_v1"),
        help="Directory for COVER v1 JSON/score/summary outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*-gpt-4o-mini-*-shot2-ndoc5-42.json",
        help="Glob pattern for raw result JSON files.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit raw result JSON files. Overrides --pattern.",
    )
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--decomposer", choices=["llm", "claimify"], default="llm")
    parser.add_argument("--decompose-model", default="gpt-4o-mini")
    parser.add_argument("--verifier", choices=["nli", "llm", "hybrid"], default="llm")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini")
    parser.add_argument("--revision-model", default="gpt-4o-mini")
    parser.add_argument(
        "--revision-mode",
        choices=["llm", "template", "atomic", "qampari", "auto"],
        default="auto",
        help="Default `auto` preserves QAMPARI list answers and uses atomic sentences otherwise.",
    )
    parser.add_argument("--answer-format", choices=["auto", "paragraph", "qampari"], default="auto")
    parser.add_argument("--citation-policy", choices=["minimal", "all", "source"], default="source")
    parser.add_argument("--max-citations-per-claim", type=int, default=3)
    parser.add_argument("--max-verified-claims", type=int, default=16)
    parser.add_argument("--max-qampari-items", type=int, default=80)
    parser.add_argument(
        "--allow-citation-repair",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use claims whose original citations failed but another retrieved doc supported them.",
    )
    parser.add_argument(
        "--include-background-claims",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep claims marked as background during v1 revision.",
    )
    parser.add_argument("--nli-model", default=NLI_MODEL)
    parser.add_argument("--cache-file", type=Path, default=Path("cache/cover_v1_cache.json"))
    parser.add_argument(
        "--evidence-scope",
        default="cited_then_all",
        choices=["cited", "all_docs", "cited_then_all"],
    )

    parser.add_argument("--force-revise", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--final-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --final-audit to cover_v1_revise.py. Enabled by default; use --no-final-audit to disable.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="Do not pass --citations to eval.py; useful for a low-memory first pass.",
    )
    parser.add_argument("--no-qa", action="store_true")
    parser.add_argument("--no-mauve", action="store_true")
    parser.add_argument(
        "--no-claims-nli",
        action="store_true",
        help="Do not pass --claims_nli for ELI5 files. Leave enabled for ALCE-paper correctness on ELI5.",
    )

    parser.add_argument("--summary-csv", type=Path, default=Path("result/cover_v1/cover_v1_comparison.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("result/cover_v1/cover_v1_comparison.md"))
    parser.add_argument(
        "--audit-summary-csv",
        type=Path,
        default=None,
        help=(
            "COVER audit summary CSV to align with V1. "
            "Default: result/cover_audit/cover_audit_comparison.csv if present, "
            "else data/cover_audit_comparison.csv, else result/cover_audit_comparison.csv."
        ),
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path("result/cover_v1/cover_v1_vs_audit_comparison.csv"),
        help="Compact V0 audit vs V1 comparison CSV output.",
    )
    parser.add_argument(
        "--combined-md",
        type=Path,
        default=Path("result/cover_v1/cover_v1_vs_audit_comparison.md"),
        help="Compact V0 audit vs V1 comparison Markdown output.",
    )
    parser.add_argument(
        "--no-combined-summary",
        action="store_true",
        help="Only write the V1 detailed summary, not the V0 audit vs V1 comparison.",
    )
    return parser.parse_args()


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def is_raw_result(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    bad_markers = [
        ".cover_v1",
        ".cover_audit",
        ".cover_score",
        ".score",
        ".debug",
        ".smoke",
    ]
    return not any(marker in path.name for marker in bad_markers)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def display(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def discover_inputs(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.inputs:
        files = [resolve(root, p) for p in args.inputs]
    else:
        files = sorted(resolve(root, args.result_dir).glob(args.pattern))

    files = [p for p in files if is_raw_result(p)]
    if not files:
        raise SystemExit("No raw result JSON files found. Check --result-dir or --pattern.")
    return files


def run(cmd: list[str | Path], root: Path, dry_run: bool) -> None:
    cmd = [str(part) for part in cmd]
    print("+ " + subprocess.list2cmdline(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=root, check=True)


def cover_v1_path(raw: Path, output_dir: Path) -> Path:
    return output_dir / f"{raw.stem}.cover_v1.json"


def score_path(result_json: Path) -> Path:
    return Path(str(result_json) + ".score")


def cover_score_path(result_json: Path) -> Path:
    return Path(str(result_json) + ".cover_score")


def build_revise_cmd(args: argparse.Namespace, root: Path, raw: Path, out: Path) -> list[str | Path]:
    cmd: list[str | Path] = [
        args.python,
        root / "scripts" / "cover_v1_revise.py",
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
        "--answer-format",
        args.answer_format,
        "--citation-policy",
        args.citation_policy,
        "--max-citations-per-claim",
        str(args.max_citations_per_claim),
        "--max-verified-claims",
        str(args.max_verified_claims),
        "--max-qampari-items",
        str(args.max_qampari_items),
        "--evidence-scope",
        args.evidence_scope,
        "--cache-file",
        resolve(root, args.cache_file),
        "--pretty",
    ]
    if args.allow_citation_repair:
        cmd.append("--allow-citation-repair")
    else:
        cmd.append("--no-allow-citation-repair")
    if args.include_background_claims:
        cmd.append("--include-background-claims")
    else:
        cmd.append("--no-include-background-claims")
    if args.verifier in {"nli", "hybrid"}:
        cmd.extend(["--nli-model", args.nli_model])
    if args.verifier in {"llm", "hybrid"}:
        cmd.extend(["--llm-verify-model", args.llm_verify_model])
    if args.revision_mode == "llm":
        cmd.extend(["--revision-model", args.revision_model])
    if args.final_audit:
        cmd.append("--final-audit")
    return cmd


def build_eval_cmd(args: argparse.Namespace, root: Path, cover: Path) -> list[str | Path]:
    cmd: list[str | Path] = [
        args.python,
        root / "eval.py",
        "--f",
        cover,
    ]
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
        # Conservative fallback for unknown files: keep the old broad eval.
        if not args.no_citations:
            cmd.append("--citations")
        if not args.no_qa:
            cmd.append("--qa")
        if not args.no_mauve:
            cmd.append("--mauve")
    return cmd


def infer_dataset_from_path(path: Path) -> str:
    text = str(path).lower().replace("/", "\\")
    name = path.name.lower()
    for dataset in ("asqa", "eli5", "qampari"):
        if name.startswith(dataset + "-") or ("\\" + dataset + "-") in text:
            return dataset
    return ""


def build_merge_cmd(args: argparse.Namespace, root: Path, cover: Path) -> list[str | Path]:
    cmd: list[str | Path] = [
        args.python,
        root / "scripts" / "merge_cover_scores.py",
        "--result",
        cover,
    ]
    sp = score_path(cover)
    if sp.exists():
        cmd.extend(["--alce-score", sp])
    cmd.append("--pretty")
    return cmd


def flatten_nested(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten_nested(f"{prefix}_{key}" if prefix else str(key), nested, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value


def normalized_raw_score(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept both merge_cover_scores.py output and nested result JSON snippets."""
    out = dict(raw)
    for key in [
        "cover_audit_summary",
        "cover_v1_summary",
        "cover_v1_final_audit_summary",
        "cover_v1_config",
        "cover_v1_token_usage",
    ]:
        if isinstance(raw.get(key), dict):
            flatten_nested(key, raw[key], out)
    return out


def get_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value is None or value == "":
            continue
        return value
    return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_divide(numerator: Any, denominator: Any) -> float | None:
    n = as_float(numerator)
    d = as_float(denominator)
    if n is None or d in (None, 0.0):
        return None
    return n / d


def flatten_score(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize v0 audit, v1 revision, and optional v1 final-audit fields."""
    raw = normalized_raw_score(raw)
    out: dict[str, Any] = dict(raw)

    draft_claims = get_first(
        raw,
        [
            "cover_v1_summary_draft_num_claims",
            "cover_v1_summary_num_claims",
        ],
    )
    v1_examples = get_first(raw, ["cover_v1_summary_num_examples"])
    v1_supported = get_first(raw, ["cover_v1_summary_draft_label_counts_supported"])
    v1_wrong_missing = get_first(raw, ["cover_v1_summary_draft_label_counts_wrong_or_missing_citation"])
    v1_not_supported = get_first(raw, ["cover_v1_summary_draft_label_counts_not_supported"])
    v1_no_citation = get_first(raw, ["cover_v1_summary_draft_label_counts_no_citation"])

    out["cover_v1_draft_supported_rate"] = safe_divide(v1_supported, draft_claims)
    out["cover_v1_draft_wrong_missing_rate"] = safe_divide(v1_wrong_missing, draft_claims)

    final_claims = get_first(
        raw,
        [
            "cover_v1_final_audit_summary_num_claims",
            "cover_audit_summary_num_claims",
        ],
    )
    final_support_rate = get_first(
        raw,
        [
            "cover_v1_final_audit_summary_atomic_claim_support_rate_micro",
            "cover_audit_summary_atomic_claim_support_rate_micro",
        ],
    )
    final_unsupported_rate = get_first(
        raw,
        [
            "cover_v1_final_audit_summary_unsupported_claim_rate_micro",
            "cover_audit_summary_unsupported_claim_rate_micro",
        ],
    )

    # Generic COVER columns used by old summaries.
    # Prefer final-audit metrics when present. If no final audit was run,
    # use draft/revision metrics so V1 rows are still informative.
    out["cover_num_claims"] = get_first(
        raw,
        [
            "cover_num_claims",
            "cover_claims",
            "cover_v1_final_audit_summary_num_claims",
            "cover_v1_summary_draft_num_claims",
            "cover_audit_summary_num_claims",
            "num_claims",
        ],
    )
    out["cover_support_micro"] = get_first(
        raw,
        [
            "cover_support_micro",
            "cover_atomic_claim_support_rate_micro",
            "cover_v1_final_audit_summary_atomic_claim_support_rate_micro",
            "cover_audit_summary_atomic_claim_support_rate_micro",
            "atomic_claim_support_rate_micro",
        ],
    )
    if out["cover_support_micro"] is None:
        out["cover_support_micro"] = out["cover_v1_draft_supported_rate"]

    out["cover_support_macro"] = get_first(
        raw,
        [
            "cover_support_macro",
            "cover_atomic_claim_support_rate_macro",
            "cover_audit_summary_atomic_claim_support_rate_macro",
            "atomic_claim_support_rate_macro",
        ],
    )
    out["cover_unsupported_micro"] = get_first(
        raw,
        [
            "cover_unsupported_micro",
            "cover_unsupported_claim_rate_micro",
            "cover_v1_final_audit_summary_unsupported_claim_rate_micro",
            "cover_audit_summary_unsupported_claim_rate_micro",
            "cover_v1_summary_rejected_claim_rate_micro",
            "unsupported_claim_rate_micro",
        ],
    )
    out["cover_avg_claims"] = get_first(
        raw,
        [
            "cover_avg_claims",
            "cover_audit_summary_avg_claims_per_example",
            "cover_v1_summary_avg_claims_per_example",
            "avg_claims_per_example",
        ],
    )
    if out["cover_avg_claims"] is None:
        out["cover_avg_claims"] = safe_divide(draft_claims, v1_examples)

    out["cover_no_citation_claims"] = get_first(
        raw,
        [
            "cover_no_citation_claims",
            "cover_v1_summary_draft_label_counts_no_citation",
            "cover_audit_summary_no_citation_claims",
            "no_citation_claims",
        ],
    )
    out["cover_wrong_or_missing_citation_claims"] = get_first(
        raw,
        [
            "cover_wrong_or_missing_citation_claims",
            "cover_v1_summary_draft_label_counts_wrong_or_missing_citation",
            "cover_audit_summary_wrong_or_missing_citation_claims",
            "wrong_or_missing_citation_claims",
        ],
    )

    # If a result JSON snippet had nested final-audit fields but not flattened
    # aliases, ensure direct table keys are still available.
    if final_claims is not None:
        out.setdefault("cover_v1_final_audit_summary_num_claims", final_claims)
    if final_support_rate is not None:
        out.setdefault("cover_v1_final_audit_summary_atomic_claim_support_rate_micro", final_support_rate)
    if final_unsupported_rate is not None:
        out.setdefault("cover_v1_final_audit_summary_unsupported_claim_rate_micro", final_unsupported_rate)

    # Keep local variables referenced so future edits do not accidentally remove
    # these parsed labels from the compatibility surface.
    _ = (v1_not_supported, v1_no_citation)
    return out


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
    return (
        DATASET_ORDER.get(dataset, 99),
        dataset,
        TAG_ORDER.get(tag, 99),
        tag,
    )


def write_summary(args: argparse.Namespace, root: Path, cover_scores: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in cover_scores:
        if not path.exists():
            continue
        try:
            rows.append(flatten_score(json.loads(path.read_text(encoding="utf-8"))))
        except Exception as exc:
            print(f"WARNING: failed to read {display(root, path)}: {exc}")

    rows.sort(key=sort_key)
    if not rows:
        print("No .cover_score files available for summary.")
        return rows

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
            writer.writerow(
                {
                    header: fmt(row.get(key), kind, md=False)
                    for header, key, kind in SUMMARY_COLUMNS
                }
            )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                fmt(row.get(key), kind, md=True)
                for _, key, kind in SUMMARY_COLUMNS
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote summary CSV: {display(root, csv_path)}")
    print(f"Wrote summary Markdown: {display(root, md_path)}")
    return rows


def choose_audit_summary_path(args: argparse.Namespace, root: Path) -> Path | None:
    if args.audit_summary_csv is not None:
        return resolve(root, args.audit_summary_csv)
    candidates = [
        root / "result" / "cover_audit" / "cover_audit_comparison.csv",
        root / "data" / "cover_audit_comparison.csv",
        root / "result" / "cover_audit_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def choose_existing_v1_summary_path(args: argparse.Namespace, root: Path) -> Path | None:
    candidates = [
        resolve(root, args.summary_csv),
        root / "data" / "cover_v1_comparison.csv",
        root / "result" / "cover_v1_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def csv_get(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def percent_from_ratio(value: Any) -> float | None:
    parsed = number(value)
    if parsed is None:
        return None
    return parsed * 100.0


def delta(after: Any, before: Any) -> float | None:
    left = number(after)
    right = number(before)
    if left is None or right is None:
        return None
    return left - right


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("tag") or ""),
        str(row.get("model") or ""),
    )


def task_metric_for(dataset: str) -> str:
    if dataset == "asqa":
        return "str_em"
    if dataset == "eli5":
        return "claims_nli"
    if dataset == "qampari":
        return "qampari_f1"
    return "task_metric"


def task_value(row: dict[str, Any], dataset: str, raw_ratio_row: bool = False) -> float | None:
    metric = task_metric_for(dataset)
    if metric == "str_em":
        return number(csv_get(row, ["str_em", "alce_str_em"]))
    if metric == "claims_nli":
        return number(csv_get(row, ["claims_nli", "alce_claims_nli", "claim_recall"]))
    if metric == "qampari_f1":
        return number(csv_get(row, ["qampari_f1", "QAMPARI-F1", "alce_qampari_f1"]))
    return number(csv_get(row, ["qa_f1", "QA-F1", "alce_QA-F1"]))


def audit_value(row: dict[str, Any], canonical: str) -> float | None:
    aliases = {
        "mauve": ["mauve", "alce_mauve"],
        "citation_recall": ["citation_recall", "citation_rec", "alce_citation_rec"],
        "citation_precision": ["citation_precision", "citation_prec", "alce_citation_prec"],
        "claim_count": ["claim_count", "claims", "cover_audit_summary_num_claims"],
        "claim_support_rate": ["claim_support_rate", "support_micro_%"],
        "claim_unsupported_rate": ["claim_unsupported_rate", "unsupported_%"],
        "no_citation_claim_count": ["no_citation_claim_count", "no_citation"],
        "wrong_missing_citation_claim_count": ["wrong_missing_citation_claim_count", "wrong_missing_citation"],
    }
    return number(csv_get(row, aliases[canonical]))


def v1_value(row: dict[str, Any], canonical: str) -> float | None:
    aliases = {
        "mauve": ["alce_mauve", "mauve"],
        "citation_recall": ["alce_citation_rec", "citation_recall", "citation_rec"],
        "citation_precision": ["alce_citation_prec", "citation_precision", "citation_prec"],
        "draft_claims": ["cover_v1_summary_draft_num_claims", "v1_draft_claims"],
        "draft_not_supported": ["cover_v1_summary_draft_label_counts_not_supported", "v1_draft_not_supported"],
        "draft_no_citation": ["cover_v1_summary_draft_label_counts_no_citation", "v1_draft_no_citation"],
        "keep_rate": ["cover_v1_summary_claim_keep_rate_micro", "v1_keep_rate"],
        "rejected_rate": ["cover_v1_summary_rejected_claim_rate_micro", "v1_rejected_rate"],
        "revised_length": ["cover_v1_summary_avg_revised_length", "v1_avg_revised_length"],
        "revised_citations": ["cover_v1_summary_avg_revised_num_citations", "v1_avg_revised_citations"],
        "final_claims": ["cover_v1_final_audit_summary_num_claims", "final_claims"],
        "final_support": [
            "cover_v1_final_audit_summary_atomic_claim_support_rate_micro",
            "final_claim_support_rate",
        ],
        "final_unsupported": [
            "cover_v1_final_audit_summary_unsupported_claim_rate_micro",
            "final_claim_unsupported_rate",
        ],
    }
    if canonical == "keep_rate":
        raw = csv_get(row, ["cover_v1_summary_claim_keep_rate_micro"])
        return percent_from_ratio(raw) if raw is not None else number(csv_get(row, ["v1_keep_rate", "v1_keep_rate_%"]))
    if canonical == "rejected_rate":
        raw = csv_get(row, ["cover_v1_summary_rejected_claim_rate_micro"])
        return percent_from_ratio(raw) if raw is not None else number(csv_get(row, ["v1_rejected_rate", "v1_rejected_rate_%"]))
    if canonical == "final_support":
        raw = csv_get(row, ["cover_v1_final_audit_summary_atomic_claim_support_rate_micro"])
        return percent_from_ratio(raw) if raw is not None else number(csv_get(row, ["final_claim_support_rate", "final_support_micro_%"]))
    if canonical == "final_unsupported":
        raw = csv_get(row, ["cover_v1_final_audit_summary_unsupported_claim_rate_micro"])
        return percent_from_ratio(raw) if raw is not None else number(csv_get(row, ["final_claim_unsupported_rate", "final_unsupported_%"]))
    return number(csv_get(row, aliases[canonical]))


def v1_draft_supported_percent(row: dict[str, Any]) -> float | None:
    raw = csv_get(row, ["cover_v1_draft_supported_rate"])
    if raw is not None:
        return percent_from_ratio(raw)
    return number(csv_get(row, ["v1_draft_claim_support_rate", "v1_draft_supported_rate", "v1_draft_supported_%"]))


def v1_draft_wrong_missing_percent(row: dict[str, Any]) -> float | None:
    raw = csv_get(row, ["cover_v1_draft_wrong_missing_rate"])
    if raw is not None:
        return percent_from_ratio(raw)
    return number(csv_get(row, ["v1_draft_wrong_missing_citation_rate", "v1_draft_wrong_missing_rate", "v1_draft_wrong_missing_%"]))


def write_combined_summary(
    args: argparse.Namespace,
    root: Path,
    v1_rows: list[dict[str, Any]],
) -> None:
    if not v1_rows:
        existing_v1_summary = choose_existing_v1_summary_path(args, root)
        if existing_v1_summary is not None:
            with existing_v1_summary.open("r", newline="", encoding="utf-8") as f:
                v1_rows = list(csv.DictReader(f))

    audit_path = choose_audit_summary_path(args, root)
    if audit_path is None or not audit_path.exists():
        print("No audit summary CSV found; skip combined V0/V1 summary.")
        return

    with audit_path.open("r", newline="", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    audit_by_key = {key_for(row): row for row in audit_rows}

    combined_rows: list[dict[str, Any]] = []
    for v1_row in sorted(v1_rows, key=sort_key):
        key = key_for(v1_row)
        audit_row = audit_by_key.get(key)
        if audit_row is None:
            print(f"WARNING: no audit row for dataset/tag/model={key}; combined row skipped.")
            continue

        dataset, tag, model = key
        task_metric = task_metric_for(dataset)
        v0_task_score = task_value(audit_row, dataset)
        v1_task_score = task_value(v1_row, dataset)
        v0_mauve = audit_value(audit_row, "mauve")
        v1_mauve = v1_value(v1_row, "mauve")
        v0_citation_recall = audit_value(audit_row, "citation_recall")
        v1_citation_recall = v1_value(v1_row, "citation_recall")
        v0_citation_precision = audit_value(audit_row, "citation_precision")
        v1_citation_precision = v1_value(v1_row, "citation_precision")

        combined_rows.append(
            {
                "dataset": dataset,
                "tag": tag,
                "model": model,
                "task_metric": task_metric,
                "v0_task_score": v0_task_score,
                "v1_task_score": v1_task_score,
                "task_delta": delta(v1_task_score, v0_task_score),
                "v0_mauve": v0_mauve,
                "v1_mauve": v1_mauve,
                "mauve_delta": delta(v1_mauve, v0_mauve),
                "v0_citation_recall": v0_citation_recall,
                "v1_citation_recall": v1_citation_recall,
                "citation_recall_delta": delta(v1_citation_recall, v0_citation_recall),
                "v0_citation_precision": v0_citation_precision,
                "v1_citation_precision": v1_citation_precision,
                "citation_precision_delta": delta(v1_citation_precision, v0_citation_precision),
                "v0_claim_count": audit_value(audit_row, "claim_count"),
                "v0_claim_support_rate": audit_value(audit_row, "claim_support_rate"),
                "v0_claim_unsupported_rate": audit_value(audit_row, "claim_unsupported_rate"),
                "v0_no_citation_claim_count": audit_value(audit_row, "no_citation_claim_count"),
                "v0_wrong_missing_citation_claim_count": audit_value(audit_row, "wrong_missing_citation_claim_count"),
                "v1_draft_claims": v1_value(v1_row, "draft_claims"),
                "v1_draft_claim_support_rate": v1_draft_supported_percent(v1_row),
                "v1_draft_wrong_missing_citation_rate": v1_draft_wrong_missing_percent(v1_row),
                "v1_draft_not_supported": v1_value(v1_row, "draft_not_supported"),
                "v1_draft_no_citation": v1_value(v1_row, "draft_no_citation"),
                "v1_keep_rate": v1_value(v1_row, "keep_rate"),
                "v1_rejected_rate": v1_value(v1_row, "rejected_rate"),
                "v1_revised_length": v1_value(v1_row, "revised_length"),
                "v1_revised_citations": v1_value(v1_row, "revised_citations"),
                "final_claims": v1_value(v1_row, "final_claims"),
                "final_claim_support_rate": v1_value(v1_row, "final_support"),
                "final_claim_unsupported_rate": v1_value(v1_row, "final_unsupported"),
            }
        )

    if not combined_rows:
        print("No matched audit/V1 rows; skip combined V0/V1 summary.")
        return

    csv_path = resolve(root, args.combined_csv)
    md_path = resolve(root, args.combined_md)
    if args.dry_run:
        print(f"Would write {display(root, csv_path)}")
        print(f"Would write {display(root, md_path)}")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [header for header, _, _ in COMBINED_COLUMNS]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(
                {
                    header: fmt(row.get(key), kind, md=False)
                    for header, key, kind in COMBINED_COLUMNS
                }
            )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in combined_rows:
        lines.append(
            "| "
            + " | ".join(
                fmt(row.get(key), kind, md=True)
                for _, key, kind in COMBINED_COLUMNS
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote combined CSV: {display(root, csv_path)}")
    print(f"Wrote combined Markdown: {display(root, md_path)}")


def main() -> int:
    args = parse_args()
    root = root_dir()
    output_dir = resolve(root, args.output_dir)

    if not args.dry_run:
        resolve(root, args.cache_file).parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(args, root)
    print(f"Found {len(inputs)} raw result files:")
    for path in inputs:
        print("  - " + display(root, path))

    failures: list[tuple[Path, str]] = []
    cover_scores: list[Path] = []

    for index, raw in enumerate(inputs, start=1):
        cover = cover_v1_path(raw, output_dir)
        cscore = cover_score_path(cover)
        cover_scores.append(cscore)

        print(f"\n[{index}/{len(inputs)}] {display(root, raw)}", flush=True)
        if args.summary_only:
            continue

        try:
            if args.force_revise or not cover.exists():
                run(build_revise_cmd(args, root, raw, cover), root, args.dry_run)
            else:
                print(f"  cover_v1 exists, skip: {display(root, cover)}")

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

    v1_rows = write_summary(args, root, cover_scores)
    if not args.no_combined_summary:
        write_combined_summary(args, root, v1_rows)

    if failures:
        print("\nFailures:")
        for path, message in failures:
            print(f"- {display(root, path)}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
