#!/usr/bin/env python3
"""Run/evaluate the 4o-mini ndoc=5 ALCE experiments and summarize scores."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml


BASE_DIR = Path(__file__).resolve().parent

ALL_CONFIGS = (
    "configs/asqa_4omini_shot2_ndoc5_gtr_default.yaml",
    "configs/asqa_4omini_shot2_ndoc5_bge_rerank_default.yaml",
    "configs/asqa_4omini_shot2_ndoc5_reranked_oracle.yaml",
    "configs/eli5_4omini_shot2_ndoc5_bm25_default.yaml",
    "configs/eli5_4omini_shot2_ndoc5_bge_rerank_default.yaml",
    "configs/eli5_4omini_shot2_ndoc5_reranked_oracle.yaml",
    "configs/qampari_4omini_shot2_ndoc5_gtr_default.yaml",
    "configs/qampari_4omini_shot2_ndoc5_bge_rerank_default.yaml",
    "configs/qampari_4omini_shot2_ndoc5_reranked_oracle.yaml",
)

DEFAULT_RUN_CONFIGS = (
    "configs/asqa_4omini_shot2_ndoc5_reranked_oracle.yaml",
    "configs/eli5_4omini_shot2_ndoc5_bge_rerank_default.yaml",
    "configs/eli5_4omini_shot2_ndoc5_reranked_oracle.yaml",
    "configs/qampari_4omini_shot2_ndoc5_bge_rerank_default.yaml",
    "configs/qampari_4omini_shot2_ndoc5_reranked_oracle.yaml",
)

BASE_COLUMNS = ("dataset", "tag", "model", "status", "json", "score")

METRIC_COLUMNS = (
    "length",
    "str_em",
    "str_hit",
    "rougeLsum",
    "QA-EM",
    "QA-F1",
    "QA-Hit",
    "mauve",
    "claims_nli",
    "num_preds",
    "qampari_prec",
    "qampari_rec",
    "qampari_rec_top5",
    "qampari_f1",
    "qampari_f1_top5",
    "citation_rec",
    "citation_prec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the remaining 4o-mini ALCE configs, evaluate dataset-specific "
            "metrics, and write a 9-run comparison table."
        )
    )
    parser.add_argument(
        "--run-configs",
        choices=("missing", "all", "none"),
        default="missing",
        help=(
            "Which configs to run before evaluation. 'missing' is the five configs "
            "listed in the experiment TODO; summaries always cover all 9 configs."
        ),
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Run selected configs even if their result JSON already exists.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Recompute score files even if they already exist.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run selected configs and write a summary of existing score files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing run.py/eval.py.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next run/eval command if one command fails.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal places for the Markdown table.",
    )
    parser.add_argument(
        "--summary-md",
        default="result/4omini_ndoc5_summary.md",
        help="Markdown summary output path, relative to this directory by default.",
    )
    parser.add_argument(
        "--summary-csv",
        default="result/4omini_ndoc5_summary.csv",
        help="CSV summary output path, relative to this directory by default.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def load_config(config_rel: str) -> Dict[str, Any]:
    config_path = resolve(config_rel)
    with config_path.open() as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a mapping: {config_rel}")
    return config


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config and config[key] is not None else default


def result_stem(config: Dict[str, Any]) -> str:
    required = ("dataset_name", "tag", "model", "shot", "ndoc")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Config is missing required keys: {', '.join(missing)}")

    model_name = str(config["model"])
    if "/" in model_name:
        model_name = model_name.rsplit("/", 1)[-1]

    name = (
        f"{config['dataset_name']}-{model_name}-{config['tag']}"
        f"-shot{config['shot']}-ndoc{config['ndoc']}-{config_value(config, 'seed', 42)}"
    )
    if as_bool(config_value(config, "azure", False)):
        name += "-azure"
    if config_value(config, "quick_test") is not None:
        name += f"-quick_test{config['quick_test']}"
    if as_bool(config_value(config, "no_doc_in_demo", False)):
        name += "-no_doc_in_demo"
    if as_bool(config_value(config, "fewer_doc_in_demo", False)):
        name += f"-{config_value(config, 'ndoc_in_demo')}_doc_in_demo"
    if int(config_value(config, "num_samples", 1)) > 1:
        name += f"-sample{config['num_samples']}"
    if as_bool(config_value(config, "force_cite_show", False)):
        name += "-forceciteshow"
    return name


def result_json_path(config: Dict[str, Any]) -> Path:
    return BASE_DIR / "result" / f"{result_stem(config)}.json"


def score_path_for_json(json_path: Path) -> Path:
    return Path(str(json_path) + ".score")


def eval_flags(dataset_name: str) -> List[str]:
    if dataset_name == "asqa":
        return ["--citations", "--qa", "--mauve"]
    if dataset_name == "eli5":
        return ["--citations", "--claims_nli", "--mauve"]
    if dataset_name == "qampari":
        return ["--citations"]
    raise ValueError(f"Unknown dataset_name for eval flags: {dataset_name}")


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(cmd: Sequence[str], dry_run: bool, keep_going: bool) -> bool:
    print(f"+ {shell_join(cmd)}")
    if dry_run:
        return False
    try:
        subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
        return True
    except subprocess.CalledProcessError as exc:
        if keep_going:
            print(f"[warn] command failed with exit code {exc.returncode}; continuing")
            return False
        raise


def selected_run_configs(mode: str) -> Sequence[str]:
    if mode == "missing":
        return DEFAULT_RUN_CONFIGS
    if mode == "all":
        return ALL_CONFIGS
    if mode == "none":
        return ()
    raise ValueError(f"Unexpected run mode: {mode}")


def run_selected_configs(args: argparse.Namespace) -> Dict[Path, bool]:
    ran_by_json: Dict[Path, bool] = {}
    for config_rel in selected_run_configs(args.run_configs):
        config = load_config(config_rel)
        json_path = result_json_path(config)
        if json_path.exists() and not args.force_run:
            print(f"[run] skip existing {rel(json_path)}")
            ran_by_json[json_path] = False
            continue

        cmd = [sys.executable, "run.py", "--config", config_rel]
        ran = run_command(cmd, dry_run=args.dry_run, keep_going=args.keep_going)
        ran_by_json[json_path] = ran
        if ran and not json_path.exists():
            message = f"run.py finished, but expected result is missing: {rel(json_path)}"
            if args.keep_going:
                print(f"[warn] {message}")
            else:
                raise FileNotFoundError(message)
    return ran_by_json


def evaluate_all_configs(args: argparse.Namespace, ran_by_json: Dict[Path, bool]) -> None:
    if args.skip_eval:
        print("[eval] skipped by --skip-eval")
        return

    for config_rel in ALL_CONFIGS:
        config = load_config(config_rel)
        json_path = result_json_path(config)
        score_path = score_path_for_json(json_path)
        if not json_path.exists():
            print(f"[eval] missing json, skip {rel(json_path)}")
            continue

        should_eval = (
            args.force_eval
            or ran_by_json.get(json_path, False)
            or not score_path.exists()
        )
        if not should_eval:
            print(f"[eval] skip existing {rel(score_path)}")
            continue

        cmd = [
            sys.executable,
            "eval.py",
            "--f",
            rel(json_path),
            *eval_flags(str(config["dataset_name"])),
        ]
        run_command(cmd, dry_run=args.dry_run, keep_going=args.keep_going)


def load_score(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        score = json.load(f)
    if not isinstance(score, dict):
        raise ValueError(f"Score file is not a JSON object: {rel(path)}")
    return score


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for config_rel in ALL_CONFIGS:
        config = load_config(config_rel)
        json_path = result_json_path(config)
        score_path = score_path_for_json(json_path)
        row: Dict[str, Any] = {
            "dataset": config["dataset_name"],
            "tag": config["tag"],
            "model": config["model"],
            "status": "ok" if score_path.exists() else "missing_score",
            "json": rel(json_path),
            "score": rel(score_path),
        }
        if not json_path.exists():
            row["status"] = "missing_json"
        if score_path.exists():
            row.update(load_score(score_path))
        rows.append(row)
    return rows


def table_columns(rows: Iterable[Dict[str, Any]]) -> List[str]:
    row_list = list(rows)
    metric_cols = [col for col in METRIC_COLUMNS if any(col in row for row in row_list)]
    known = set(BASE_COLUMNS) | set(METRIC_COLUMNS)
    extras = sorted({key for row in row_list for key in row.keys() if key not in known})
    return [*BASE_COLUMNS, *metric_cols, *extras]


def format_value(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str], precision: int) -> str:
    def cell(value: Any) -> str:
        return format_value(value, precision).replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def write_summary(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    columns = table_columns(rows)
    md_path = resolve(args.summary_md)
    csv_path = resolve(args.summary_csv)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = markdown_table(rows, columns, args.precision)
    md_path.write_text(markdown + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})

    print()
    print(markdown)
    print()
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"[summary] score files: {ok_count}/{len(ALL_CONFIGS)}")
    print(f"[summary] wrote {rel(md_path)}")
    print(f"[summary] wrote {rel(csv_path)}")


def main() -> None:
    args = parse_args()
    ran_by_json = run_selected_configs(args)
    evaluate_all_configs(args, ran_by_json)
    write_summary(collect_rows(), args)


if __name__ == "__main__":
    main()
