#!/usr/bin/env bash
# Oracle-retrieval diagnostic for COVER-RAG v3.
#
# This script does two things:
#   1. Writes retrieval coverage diagnostics from data/*_reranked_oracle.json.
#   2. Runs the current COVER-RAG v3 setting on existing oracle-retrieval raw
#      outputs, then compares oracle-v0 vs oracle-v3.
#
# Launch:
#   nohup bash scripts/run_cover_v3_oracle_diagnosis_nohup.sh \
#     > logs/cover_v3_oracle_diagnosis_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export PYTHON="${PYTHON:-/home/anaconda/envs/alce/bin/python}"
export BASE_DIR="${BASE_DIR:-result/cover_v3_oracle_diagnosis/oracle_run}"
export DIAG_DIR="${DIAG_DIR:-result/cover_v3_oracle_diagnosis/diagnosis}"
export CURRENT_COMPARISON_WIDE="${CURRENT_COMPARISON_WIDE:-result/cover_v3_main_task_boost_v8_intermediate_full/cover_v0_v3_comparison_wide.csv}"

# Reuse the existing oracle-retrieval raw outputs from result/origin and the
# oracle doc files from data/. These raw outputs are the v0 generation upper
# bound under oracle docs; this script audits/revises them, but does not
# regenerate raw answers.
export ASQA_RAW="${ASQA_RAW:-result/origin/asqa-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json}"
export ELI5_RAW="${ELI5_RAW:-result/origin/eli5-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json}"
export QAMPARI_RAW="${QAMPARI_RAW:-result/origin/qampari-gpt-4o-mini-reranked_oracle-shot2-ndoc5-42.json}"
export ASQA_DOCS="${ASQA_DOCS:-data/asqa_eval_gtr_top100_reranked_oracle.json}"
export ELI5_DOCS="${ELI5_DOCS:-data/eli5_eval_bm25_top100_reranked_oracle.json}"
export QAMPARI_DOCS="${QAMPARI_DOCS:-data/qampari_eval_gtr_top100_reranked_oracle.json}"
export MAX_DOC_POOL="${MAX_DOC_POOL:-5}"
export V0_CACHE_FILE="${V0_CACHE_FILE:-cache/cover_audit_oracle_cache.json}"
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-oracle_}"

# Use the same API-compatible environment protection as the main experiment.
export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DISABLE_PROXY="${DISABLE_PROXY:-1}"
if [[ "$DISABLE_PROXY" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

# Keep the current v8/main-task settings unless the caller overrides them.
export FINAL_AUDIT="${FINAL_AUDIT:-1}"
export RENDER_MODE="${RENDER_MODE:-answer_preserving}"
export NLI_PREMISE_MODE="${NLI_PREMISE_MODE:-sentence_window}"
export NLI_TOP_SENTENCES="${NLI_TOP_SENTENCES:-4}"
export NLI_SENTENCE_WINDOW_SIZE="${NLI_SENTENCE_WINDOW_SIZE:-1}"
export PRESERVE_ANSWER_SENTENCES="${PRESERVE_ANSWER_SENTENCES:-1}"
export PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES="${PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES:-0}"

export EXPANSION_MODE="${EXPANSION_MODE:-llm}"
export ANSWER_TARGETED_EXPANSION="${ANSWER_TARGETED_EXPANSION:-1}"
export ANSWER_TARGETED_EXTRACTIVE_TOP_K="${ANSWER_TARGETED_EXTRACTIVE_TOP_K:-2}"
export ANSWER_TARGETED_MIN_SENTENCE_OVERLAP="${ANSWER_TARGETED_MIN_SENTENCE_OVERLAP:-0.18}"
export EXPANSION_FILTER_GENERAL_ANSWER_SPANS="${EXPANSION_FILTER_GENERAL_ANSWER_SPANS:-1}"
export EXPANSION_MIN_QUESTION_OVERLAP="${EXPANSION_MIN_QUESTION_OVERLAP:-0.12}"
export EXPANSION_QUERY_SOURCE="${EXPANSION_QUERY_SOURCE:-question_evidence_and_rejected}"
export EXPANSION_USE_INTERMEDIATE_ANSWER="${EXPANSION_USE_INTERMEDIATE_ANSWER:-1}"
export EXPANSION_EVIDENCE_SENTENCES="${EXPANSION_EVIDENCE_SENTENCES:-10}"
export EXPANSION_CANDIDATE_SPANS="${EXPANSION_CANDIDATE_SPANS:-24}"
export MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-10}"
export MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-6}"
export MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-3}"
export MIN_EXPANSION_SCORE="${MIN_EXPANSION_SCORE:-0.10}"

export MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-22}"
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-28}"
export CITATION_POLICY="${CITATION_POLICY:-minimal}"
export SENTENCE_CITATION_SOURCE="${SENTENCE_CITATION_SOURCE:-source}"
export MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-2}"
export MAX_CITATIONS_PER_SENTENCE="${MAX_CITATIONS_PER_SENTENCE:-2}"

export ASQA_MAX_FINAL_CLAIMS="${ASQA_MAX_FINAL_CLAIMS:-30}"
export ELI5_EXPANSION_INCLUDE_EXTRACTIVE="${ELI5_EXPANSION_INCLUDE_EXTRACTIVE:-0}"
export ELI5_EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN="${ELI5_EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN:-1}"
export ELI5_ANSWER_TARGETED_EXTRACTIVE_TOP_K="${ELI5_ANSWER_TARGETED_EXTRACTIVE_TOP_K:-6}"
export ELI5_ANSWER_TARGETED_MIN_SENTENCE_OVERLAP="${ELI5_ANSWER_TARGETED_MIN_SENTENCE_OVERLAP:-0.0}"
export ELI5_EXPANSION_MIN_QUESTION_OVERLAP="${ELI5_EXPANSION_MIN_QUESTION_OVERLAP:-0.08}"
export ELI5_MAX_EXPANSION_CANDIDATES="${ELI5_MAX_EXPANSION_CANDIDATES:-8}"
export ELI5_MAX_EXPANSION_VERIFICATIONS="${ELI5_MAX_EXPANSION_VERIFICATIONS:-6}"
export ELI5_MAX_EXPANDED_CLAIMS="${ELI5_MAX_EXPANDED_CLAIMS:-3}"
export ELI5_MAX_FINAL_CLAIMS="${ELI5_MAX_FINAL_CLAIMS:-32}"

export QAMPARI_USE_SELECT="${QAMPARI_USE_SELECT:-1}"
export QAMPARI_SELECT_JOINT_DOCS="${QAMPARI_SELECT_JOINT_DOCS:-5}"
export QAMPARI_SELECT_EXPANSION_DOCS="${QAMPARI_SELECT_EXPANSION_DOCS:-5}"
export QAMPARI_SELECT_JOINT_CANDIDATE_SPANS="${QAMPARI_SELECT_JOINT_CANDIDATE_SPANS:-80}"
export QAMPARI_SELECT_MAX_OUTPUT_ITEMS="${QAMPARI_SELECT_MAX_OUTPUT_ITEMS:-10}"
export QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL="${QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL:-6}"
export QAMPARI_SELECT_BACKFILL_REQUIRES_TYPE_SAFE="${QAMPARI_SELECT_BACKFILL_REQUIRES_TYPE_SAFE:-1}"
export QAMPARI_SELECT_EMPTY_OUTPUT_POLICY="${QAMPARI_SELECT_EMPTY_OUTPUT_POLICY:-empty}"
export QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL="${QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL:-0}"
export QAMPARI_SELECT_DROP_OBVIOUS_BAD="${QAMPARI_SELECT_DROP_OBVIOUS_BAD:-1}"
export QAMPARI_SELECT_MAX_CITATIONS_PER_ITEM="${QAMPARI_SELECT_MAX_CITATIONS_PER_ITEM:-1}"

export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
export CACHE_SAVE_EVERY="${CACHE_SAVE_EVERY:-50}"
export REVISION_WORKERS="${REVISION_WORKERS:-4}"

mkdir -p "$DIAG_DIR" "$BASE_DIR" logs

run_step() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  echo "+ $*"
  "$@"
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "WARNING: $label failed with exit code $status. Continuing."
  fi
  return 0
}

echo "=== COVER-RAG oracle diagnostic ==="
echo "Root: $ROOT_DIR"
echo "Diagnosis dir: $DIAG_DIR"
echo "Oracle run dir: $BASE_DIR"
echo "Current comparison: $CURRENT_COMPARISON_WIDE"
echo "Oracle raw inputs:"
printf '  - %s\n' "$ASQA_RAW" "$ELI5_RAW" "$QAMPARI_RAW"
echo "Oracle doc files:"
printf '  - %s\n' "$ASQA_DOCS" "$ELI5_DOCS" "$QAMPARI_DOCS"

run_step "Stage 1/4: pre-run oracle retrieval coverage" \
  "$PYTHON" scripts/diagnose_oracle_retrieval.py \
  --output-dir "$DIAG_DIR" \
  --current-comparison-wide "$CURRENT_COMPARISON_WIDE" \
  --oracle-comparison-wide "$BASE_DIR/cover_v0_v3_comparison_wide.csv"

if [ "${RUN_ORACLE_COVER:-1}" = "1" ]; then
  run_step "Stage 2/4: run oracle v0 audit and oracle v3" \
    bash scripts/run_cover_v3_three_datasets_nohup.sh
else
  echo
  echo "RUN_ORACLE_COVER=0, skip oracle v0/v3 run."
fi

run_step "Stage 3/4: post-run oracle metric summary" \
  "$PYTHON" scripts/diagnose_oracle_retrieval.py \
  --output-dir "$DIAG_DIR" \
  --current-comparison-wide "$CURRENT_COMPARISON_WIDE" \
  --oracle-comparison-wide "$BASE_DIR/cover_v0_v3_comparison_wide.csv"

echo
echo "=== Stage 4/4: outputs ==="
echo "Coverage: $DIAG_DIR/oracle_retrieval_coverage.csv"
echo "Item comparison: $DIAG_DIR/oracle_retrieval_item_comparison.csv"
echo "Metric summary: $DIAG_DIR/oracle_metric_summary.csv"
echo "Metric deltas: $DIAG_DIR/oracle_metric_deltas.csv"
echo "Oracle v0/v3 comparison: $BASE_DIR/cover_v0_v3_comparison_wide.csv"
echo "Done."
