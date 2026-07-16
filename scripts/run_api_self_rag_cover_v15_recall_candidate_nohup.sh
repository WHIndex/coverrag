#!/usr/bin/env bash
# Launch the next Self-RAG plug-in experiment.
#
# This keeps the Self-RAG generations fixed and tests a stronger
# recall-oriented COVER-RAG final pass. It reuses the strict v0 summary from
# the clean v13 run, so the comparison is against the same base method without
# spending another long v0 audit.
#
# Usage:
#   bash scripts/run_api_self_rag_cover_v15_recall_candidate_nohup.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/api_self_rag_cover_v15_recall_candidate_${TS}.log}"

export PYTHON="${PYTHON:-/home/anaconda/envs/alce/bin/python}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-api_self_rag_cover_v15_recall_candidate}"
export BASE_DIR="${BASE_DIR:-result/${EXPERIMENT_NAME}_full}"
export RAW_DIR="${RAW_DIR:-$BASE_DIR/self_rag_origin}"

# Reuse the already generated API Self-RAG outputs. Override these if you want
# to test another Self-RAG run.
export SELF_RAG_ASQA_OUTPUT="${SELF_RAG_ASQA_OUTPUT:-result/api_self_rag_cover_v13_full/api_self_rag_raw/asqa-api-self-rag.json}"
export SELF_RAG_ELI5_OUTPUT="${SELF_RAG_ELI5_OUTPUT:-result/api_self_rag_cover_v13_full/api_self_rag_raw/eli5-api-self-rag.json}"
export SELF_RAG_QAMPARI_OUTPUT="${SELF_RAG_QAMPARI_OUTPUT:-result/api_self_rag_cover_v13_full/api_self_rag_raw/qampari-api-self-rag.json}"

# The v0 baseline was rebuilt and validated in the clean v13 run. Use it as
# the fixed baseline for this parameter-only final-pass ablation.
export SELF_RAG_PREPARE_V0="${SELF_RAG_PREPARE_V0:-0}"
export SELF_RAG_V0_SUMMARY_CSV="${SELF_RAG_V0_SUMMARY_CSV:-result/api_self_rag_cover_v13_clean_full/cover_audit/cover_audit_comparison.csv}"
export V0_SCORE_MODE="${V0_SCORE_MODE:-eval}"

# Force the v3 pass/evaluation because this script changes the final-pass
# search and output-selection policy.
export FORCE_V3="${FORCE_V3:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export FORCE_MERGE="${FORCE_MERGE:-1}"
export ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-self_rag_v15_recall_candidate_}"

# More aggressive recall search, still gated by NLI support and answer-unit
# relevance. This is meant to test whether self-RAG has hidden gold-like answer
# units in top100 that v13 was too conservative to insert.
export EXPANSION_EVIDENCE_SENTENCES="${EXPANSION_EVIDENCE_SENTENCES:-24}"
export EXPANSION_CANDIDATE_SPANS="${EXPANSION_CANDIDATE_SPANS:-72}"
export MAX_SELECTED_SPAN_IDS="${MAX_SELECTED_SPAN_IDS:-10}"
export MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-16}"
export MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-10}"
export MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-5}"
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-42}"

export RECALL_COMPLETION_MAX_TARGETS="${RECALL_COMPLETION_MAX_TARGETS:-16}"
export RECALL_COMPLETION_REJECTED_TARGETS="${RECALL_COMPLETION_REJECTED_TARGETS:-7}"
export RECALL_COMPLETION_EVIDENCE_SENTENCES="${RECALL_COMPLETION_EVIDENCE_SENTENCES:-42}"
export RECALL_COMPLETION_SUPPORTED_CLAIMS="${RECALL_COMPLETION_SUPPORTED_CLAIMS:-24}"
export RECALL_COMPLETION_MIN_TARGET_OVERLAP="${RECALL_COMPLETION_MIN_TARGET_OVERLAP:-0.06}"

# Let the final audit choose between the conservative answer and a recall-heavy
# answer, instead of always keeping the first rendered final answer.
export CANDIDATE_OUTPUT_SELECTION="${CANDIDATE_OUTPUT_SELECTION:-1}"
export CANDIDATE_OUTPUT_RECALL_CLAIMS="${CANDIDATE_OUTPUT_RECALL_CLAIMS:-4}"
export CANDIDATE_OUTPUT_RECALL_GROWTH="${CANDIDATE_OUTPUT_RECALL_GROWTH:-0.28}"
export CANDIDATE_OUTPUT_MIN_SUPPORT="${CANDIDATE_OUTPUT_MIN_SUPPORT:-0.93}"
export CANDIDATE_OUTPUT_SUPPORT_TOLERANCE="${CANDIDATE_OUTPUT_SUPPORT_TOLERANCE:-0.02}"
export CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE="${CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE:-0.07}"

# QAMPARI was recall-limited in v13. Preserve more original items and let the
# selector search a larger candidate set, while keeping type/citation guards.
export QAMPARI_SELECT_JOINT_DOCS="${QAMPARI_SELECT_JOINT_DOCS:-16}"
export QAMPARI_SELECT_EXPANSION_DOCS="${QAMPARI_SELECT_EXPANSION_DOCS:-16}"
export QAMPARI_SELECT_JOINT_CANDIDATE_SPANS="${QAMPARI_SELECT_JOINT_CANDIDATE_SPANS:-180}"
export QAMPARI_SELECT_MAX_EXPANSION_CANDIDATES="${QAMPARI_SELECT_MAX_EXPANSION_CANDIDATES:-18}"
export QAMPARI_SELECT_MAX_OUTPUT_ITEMS="${QAMPARI_SELECT_MAX_OUTPUT_ITEMS:-20}"
export QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K="${QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K:-10}"
export QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL="${QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL:-20}"
export QAMPARI_SELECT_REPAIR_TOP_K="${QAMPARI_SELECT_REPAIR_TOP_K:-8}"

echo "Launching Self-RAG v15 recall-candidate experiment"
echo "Log: $LOG_FILE"
echo "Base dir: $BASE_DIR"

nohup bash scripts/run_cover_v3_self_rag_nohup.sh > "$LOG_FILE" 2>&1 &
echo "PID: $!"
