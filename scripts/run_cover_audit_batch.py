#!/usr/bin/env python3
"""Run COVER post-hoc audit, ALCE eval, merge, and summary for result files.

Default behavior matches the single-file commands used in the experiment:

  1. scripts/cover_posthoc_audit.py
  2. eval.py
  3. scripts/merge_cover_scores.py

The script processes files serially because the shared cache file and the NLI
model are not friendly to concurrent writes / concurrent GPU memory use.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


NLI_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_PATTERN = "*-shot2-ndoc5-42.json"


SUMMARY_COLUMNS: list[tuple[str, str, str]] = [
    ("dataset", "dataset", "text"),
    ("tag", "tag", "text"),
    ("model", "model", "text"),
    ("shot", "shot", "int"),
    ("ndoc", "ndoc", "int"),
    ("answer_length", "alce_length", "float"),
    ("str_em", "alce_str_em", "float"),
    ("str_hit", "alce_str_hit", "float"),
    ("qa_f1", "alce_QA-F1", "float"),
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
    ("claim_count", "cover_audit_summary_num_claims", "int"),
    ("claim_support_rate", "cover_audit_summary_atomic_claim_support_rate_micro", "rate"),
    ("claim_support_rate_macro", "cover_audit_summary_atomic_claim_support_rate_macro", "rate"),
    ("claim_unsupported_rate", "cover_audit_summary_unsupported_claim_rate_micro", "rate"),
    ("avg_claims_per_example", "cover_audit_summary_avg_claims_per_example", "float"),
    ("no_citation_claim_count", "cover_audit_summary_no_citation_claims", "int"),
    ("wrong_missing_citation_claim_count", "cover_audit_summary_wrong_or_missing_citation_claims", "int"),
]


DATASET_ORDER = {"asqa": 0, "eli5": 1, "qampari": 2}
TAG_ORDER = {
    "gtr": 0,
    "bm25": 1,
    "bge_rerank_default": 2,
    "reranked_oracle": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run COVER audit/eval/merge and write comparison tables."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("result"),
        help="Directory containing ALCE result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/cover_audit"),
        help="Directory for COVER audit JSON/score/summary outputs.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob for raw result JSON files. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit result JSON files. Overrides --result-dir/--pattern.",
    )

    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--eval-script", type=Path, default=Path("eval.py"))
    parser.add_argument(
        "--score-mode",
        choices=["auto", "eval", "source"],
        default="auto",
        help=(
            "auto: run eval.py when available, otherwise reuse the source .json.score; "
            "eval: require eval.py; source: always reuse source .json.score."
        ),
    )
    parser.add_argument("--force-cover", action="store_true", help="Rerun cover audit even if output exists.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun eval.py even if output score exists.")
    parser.add_argument("--force-merge", action="store_true", help="Rerun merge even if cover_score exists.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only collect existing .cover_score files and write comparison tables.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed file instead of continuing with the rest.",
    )

    parser.add_argument("--decomposer", choices=["llm", "sentence", "claimify"], default="llm")
    parser.add_argument("--openai-api", dest="openai_api", action="store_true", default=True)
    parser.add_argument("--no-openai-api", dest="openai_api", action="store_false")
    parser.add_argument("--decompose-model", default="gpt-4o-mini")
    parser.add_argument("--verifier", choices=["nli", "llm", "lexical", "hybrid"], default="nli")
    parser.add_argument("--llm-verify-model", default="gpt-4o-mini")
    parser.add_argument("--nli-model", default=NLI_MODEL)
    parser.add_argument("--evidence-scope", choices=["cited", "all_docs", "cited_then_all"], default="cited_then_all")
    parser.add_argument("--cache-file", type=Path, default=Path("cache/cover_audit_cache.json"))
    parser.add_argument("--limit", type=int, default=None, help="Optional cover audit limit for smoke tests.")
    parser.add_argument("--device", default=None, help="Optional NLI device passed to cover_posthoc_audit.py.")
    parser.add_argument("--nli-batch-size", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true", help="Do not pretty-print JSON outputs.")

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("result/cover_audit/cover_audit_comparison.csv"),
        help="Compact comparison CSV output.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("result/cover_audit/cover_audit_comparison.md"),
        help="Compact comparison Markdown table output.",
    )
    return parser.parse_args()


def is_raw_result(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    excluded_markers = [
        ".cover_audit",
        ".cover_v1",
        ".debug",
        ".smoke",
    ]
    return not any(marker in path.name for marker in excluded_markers)


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def discover_inputs(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.inputs:
        candidates = [resolve_path(root, path) for path in args.inputs]
    else:
        result_dir = resolve_path(root, args.result_dir)
        candidates = sorted(result_dir.glob(args.pattern))
    inputs = [path for path in candidates if is_raw_result(path)]
    if not inputs:
        raise SystemExit("No raw result JSON files found.")
    return inputs


def display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_command(cmd: list[str], root: Path, dry_run: bool) -> None:
    print("+ " + subprocess.list2cmdline([str(part) for part in cmd]), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=root, check=True)


def cover_output_for(raw_result: Path, output_dir: Path) -> Path:
    return output_dir / f"{raw_result.stem}.cover_audit.json"


def build_cover_command(args: argparse.Namespace, root: Path, raw_result: Path, cover_result: Path) -> list[str]:
    cmd = [
        args.python,
        root / "scripts" / "cover_posthoc_audit.py",
        "--input",
        raw_result,
        "--output",
        cover_result,
        "--decomposer",
        args.decomposer,
        "--decompose-model",
        args.decompose_model,
        "--verifier",
        args.verifier,
        "--llm-verify-model",
        args.llm_verify_model,
        "--nli-model",
        args.nli_model,
        "--evidence-scope",
        args.evidence_scope,
        "--cache-file",
        resolve_path(root, args.cache_file),
    ]
    if args.openai_api:
        cmd.append("--openai-api")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.nli_batch_size is not None:
        cmd.extend(["--nli-batch-size", str(args.nli_batch_size)])
    if not args.compact_json:
        cmd.append("--pretty")
    return [str(part) for part in cmd]


def score_path_for(path: Path) -> Path:
    return Path(str(path) + ".score")


def choose_score_path(
    args: argparse.Namespace,
    root: Path,
    raw_result: Path,
    cover_result: Path,
) -> tuple[Path | None, bool]:
    cover_score = score_path_for(cover_result)
    source_score = score_path_for(raw_result)
    eval_script = resolve_path(root, args.eval_script)

    if args.score_mode == "source":
        return source_score, False

    if eval_script.exists():
        should_run_eval = args.force_eval or args.force_cover or not cover_score.exists()
        return cover_score, should_run_eval

    if args.score_mode == "eval":
        raise FileNotFoundError(f"eval script not found: {display_path(root, eval_script)}")

    if cover_score.exists():
        print(f"  eval.py not found; using existing {display_path(root, cover_score)}")
        return cover_score, False
    print(f"  eval.py not found; using source score {display_path(root, source_score)}")
    return source_score, False


def build_eval_command(args: argparse.Namespace, root: Path, cover_result: Path) -> list[str]:
    cmd = [
        args.python,
        resolve_path(root, args.eval_script),
        "--f",
        cover_result,
    ]
    dataset = infer_dataset_from_path(cover_result)
    if dataset == "asqa":
        cmd.extend(["--citations", "--qa", "--mauve"])
    elif dataset == "eli5":
        cmd.extend(["--citations", "--claims_nli", "--mauve"])
    elif dataset == "qampari":
        cmd.append("--citations")
    else:
        # Conservative fallback for unknown files: keep the old broad eval.
        cmd.extend(["--citations", "--qa", "--mauve"])
    return [str(part) for part in cmd]


def build_claim_citation_command(
    args: argparse.Namespace,
    root: Path,
    cover_result: Path,
    score_file: Path,
) -> list[str]:
    cmd = [
        args.python,
        root / "scripts" / "compute_claim_citation_metrics.py",
        "--result",
        cover_result,
        "--score-file",
        score_file,
    ]
    if not args.compact_json:
        cmd.append("--pretty")
    return [str(part) for part in cmd]


def infer_dataset_from_path(path: Path) -> str:
    text = str(path).lower().replace("/", "\\")
    name = path.name.lower()
    for dataset in ("asqa", "eli5", "qampari"):
        if name.startswith(dataset + "-") or ("\\" + dataset + "-") in text:
            return dataset
    return ""


def build_merge_command(
    args: argparse.Namespace,
    root: Path,
    cover_result: Path,
    alce_score: Path | None,
) -> list[str]:
    cmd = [
        args.python,
        root / "scripts" / "merge_cover_scores.py",
        "--result",
        cover_result,
    ]
    if alce_score is not None:
        cmd.extend(["--alce-score", alce_score])
    if not args.compact_json:
        cmd.append("--pretty")
    return [str(part) for part in cmd]


def prepare_claim_augmented_score(
    args: argparse.Namespace,
    root: Path,
    alce_score: Path | None,
    cover_result: Path,
) -> Path | None:
    if alce_score is None:
        return None
    claim_augmented_score = score_path_for(cover_result)
    if alce_score == claim_augmented_score:
        return claim_augmented_score
    if args.force_cover or args.force_merge or not claim_augmented_score.exists():
        print(
            f"  copy score for claim metrics: "
            f"{display_path(root, alce_score)} -> {display_path(root, claim_augmented_score)}"
        )
        if not args.dry_run:
            claim_augmented_score.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(alce_score, claim_augmented_score)
    return claim_augmented_score


def normalize_for_csv(value: Any, kind: str) -> str:
    if value is None:
        return ""
    if kind == "rate":
        return f"{float(value) * 100:.4f}"
    if kind == "float":
        return f"{float(value):.4f}"
    if kind == "int":
        return str(int(value))
    return str(value)


def normalize_for_md(value: Any, kind: str) -> str:
    if value is None:
        return ""
    if kind == "rate":
        return f"{float(value) * 100:.2f}"
    if kind == "float":
        return f"{float(value):.2f}"
    if kind == "int":
        return str(int(value))
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


def load_cover_score(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(root: Path, args: argparse.Namespace, cover_scores: list[Path]) -> None:
    rows = [load_cover_score(path) for path in cover_scores if path.exists()]
    rows.sort(key=sort_key)
    if not rows:
        print("No cover_score files available for summary.", flush=True)
        return

    csv_path = resolve_path(root, args.summary_csv)
    md_path = resolve_path(root, args.summary_md)
    if args.dry_run:
        print(f"Would write {display_path(root, csv_path)}")
        print(f"Would write {display_path(root, md_path)}")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[header for header, _, _ in SUMMARY_COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    header: normalize_for_csv(row.get(key), kind)
                    for header, key, kind in SUMMARY_COLUMNS
                }
            )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [header for header, _, _ in SUMMARY_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                normalize_for_md(row.get(key), kind) for _, key, kind in SUMMARY_COLUMNS
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote summary CSV to {display_path(root, csv_path)}")
    print(f"Wrote summary Markdown to {display_path(root, md_path)}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    inputs = discover_inputs(args, root)
    cache_file = resolve_path(root, args.cache_file)
    output_dir = resolve_path(root, args.output_dir)
    if not args.dry_run:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(inputs)} raw result files.")
    failures: list[tuple[Path, str]] = []
    cover_scores: list[Path] = []

    if args.summary_only:
        cover_scores = [
            Path(str(cover_output_for(raw_result, output_dir)) + ".cover_score")
            for raw_result in inputs
        ]
        write_summary(root, args, cover_scores)
        return 0

    for index, raw_result in enumerate(inputs, start=1):
        cover_result = cover_output_for(raw_result, output_dir)
        cover_score_out = Path(str(cover_result) + ".cover_score")
        cover_scores.append(cover_score_out)

        print(f"\n[{index}/{len(inputs)}] {display_path(root, raw_result)}", flush=True)
        try:
            if args.force_cover or not cover_result.exists():
                run_command(build_cover_command(args, root, raw_result, cover_result), root, args.dry_run)
            else:
                print(f"  cover audit exists; skip {display_path(root, cover_result)}")

            alce_score, should_run_eval = choose_score_path(args, root, raw_result, cover_result)
            if should_run_eval:
                run_command(build_eval_command(args, root, cover_result), root, args.dry_run)
            elif alce_score is not None and not alce_score.exists():
                raise FileNotFoundError(f"score file not found: {display_path(root, alce_score)}")
            else:
                print(f"  score ready: {display_path(root, alce_score)}")

            claim_augmented_score = prepare_claim_augmented_score(args, root, alce_score, cover_result)
            claim_score = Path(str(cover_result) + ".claim_citation_score")
            if claim_augmented_score is not None:
                if args.force_cover or args.force_merge or not claim_score.exists():
                    run_command(
                        build_claim_citation_command(args, root, cover_result, claim_augmented_score),
                        root,
                        args.dry_run,
                    )
                else:
                    print(f"  claim citation score exists; skip {display_path(root, claim_score)}")
            else:
                print("  no ALCE score available; skip claim citation score update")

            if args.force_merge or not cover_score_out.exists():
                run_command(build_merge_command(args, root, cover_result, claim_augmented_score or alce_score), root, args.dry_run)
            else:
                print(f"  merged score exists; skip {display_path(root, cover_score_out)}")

        except Exception as exc:
            message = str(exc)
            failures.append((raw_result, message))
            print(f"  FAILED: {message}", flush=True)
            if args.stop_on_error:
                break

    write_summary(root, args, cover_scores)

    if failures:
        print("\nFailures:")
        for path, message in failures:
            print(f"- {display_path(root, path)}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
