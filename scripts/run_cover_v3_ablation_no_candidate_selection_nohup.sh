#!/usr/bin/env bash
# Ablation: keep final claim/citation audit enabled, but disable the
# audit-based final-output candidate selection. This isolates whether candidate
# selection contributes beyond the main claim verification/recovery/rewrite.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FAILED: OPENAI_API_KEY is not set. Export it before launching this ablation."
  exit 1
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-cover_v3_ablation_vanilla_no_candidate_selection}"
export USE_EXTERNAL_CACHE_NAMESPACE=1
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-ablation_vanilla_no_candidate_selection_}"
export PREPARE_V0="${PREPARE_V0:-0}"
export V0_SUMMARY_CSV="${V0_SUMMARY_CSV:-result/cover_v3_main_task_boost_v6_api_full/cover_audit/cover_audit_comparison.csv}"
export FINAL_AUDIT=1
export CANDIDATE_OUTPUT_SELECTION=0
export FORCE_V3="${FORCE_V3:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export FORCE_MERGE="${FORCE_MERGE:-1}"

bash scripts/run_cover_v3_recall_repair_nohup.sh
status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"${PYTHON:-/home/anaconda/envs/alce/bin/python}" scripts/archive_vanilla_ablation.py
