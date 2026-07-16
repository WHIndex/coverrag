#!/usr/bin/env bash
# API Self-RAG-style base generation + COVER-RAG v13 plug-in.
#
# This avoids the original Self-RAG vLLM/local-checkpoint dependency by using
# an OpenAI-compatible API model (default: gpt-4o-mini) to do self-reflective
# evidence selection and citation-grounded generation.
#
# Launch:
#   nohup bash scripts/run_api_self_rag_cover_nohup.sh \
#     > logs/api_self_rag_cover_v13_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export PYTHON="${PYTHON:-/home/anaconda/envs/alce/bin/python}"
export API_SELF_RAG_MODEL="${API_SELF_RAG_MODEL:-gpt-4o-mini}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-api_self_rag_cover_v13}"
export BASE_DIR="${BASE_DIR:-result/${EXPERIMENT_NAME}_full}"
API_RAW_DIR="$BASE_DIR/api_self_rag_raw"
CONVERTED_RAW_DIR="$BASE_DIR/converted_self_rag_origin"
mkdir -p "$API_RAW_DIR" "$CONVERTED_RAW_DIR" "$BASE_DIR" logs cache

ASQA_INPUT="${ASQA_INPUT:-result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
ELI5_INPUT="${ELI5_INPUT:-result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json}"
QAMPARI_INPUT="${QAMPARI_INPUT:-result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"

ASQA_DOCS="${ASQA_DOCS:-data/asqa_eval_gtr_top100_bge_reranked.json}"
ELI5_DOCS="${ELI5_DOCS:-data/eli5_eval_bm25_top100_bge_reranked.json}"
QAMPARI_DOCS="${QAMPARI_DOCS:-data/qampari_eval_gtr_top100_bge_reranked.json}"

ASQA_RAW_OUT="$API_RAW_DIR/asqa-api-self-rag.json"
ELI5_RAW_OUT="$API_RAW_DIR/eli5-api-self-rag.json"
QAMPARI_RAW_OUT="$API_RAW_DIR/qampari-api-self-rag.json"

API_SELF_RAG_WORKERS="${API_SELF_RAG_WORKERS:-2}"
API_SELF_RAG_MAX_DOCS="${API_SELF_RAG_MAX_DOCS:-8}"
API_SELF_RAG_MAX_DOC_CHARS="${API_SELF_RAG_MAX_DOC_CHARS:-900}"
API_SELF_RAG_LIMIT="${API_SELF_RAG_LIMIT:-}"
FORCE_API_SELF_RAG="${FORCE_API_SELF_RAG:-0}"

run_step() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  echo "+ $*"
  "$@"
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "FAILED: $label exited with $status"
    exit "$status"
  fi
}

run_api_self_rag_one() {
  local dataset="$1"
  local input="$2"
  local docs="$3"
  local output="$4"
  if [ ! -f "$input" ]; then
    echo "WARNING: skip $dataset because input missing: $input"
    return 0
  fi
  if [ "$FORCE_API_SELF_RAG" != "1" ] && [ -f "$output" ]; then
    echo "API Self-RAG output exists, skip $dataset: $output"
    return 0
  fi
  local cmd=(
    "$PYTHON" scripts/run_api_self_rag.py
    --input "$input"
    --output "$output"
    --dataset "$dataset"
    --model "$API_SELF_RAG_MODEL"
    --candidate-docs-file "$docs"
    --max-docs "$API_SELF_RAG_MAX_DOCS"
    --max-doc-chars "$API_SELF_RAG_MAX_DOC_CHARS"
    --workers "$API_SELF_RAG_WORKERS"
    --cache-file "cache/api_self_rag_${dataset}_${API_SELF_RAG_MODEL}_cache.json"
    --pretty
  )
  if [ -n "$API_SELF_RAG_LIMIT" ]; then
    cmd+=(--limit "$API_SELF_RAG_LIMIT")
  fi
  run_step "Generate API Self-RAG $dataset" "${cmd[@]}"
}

echo "=== API Self-RAG-style + COVER-RAG v13 ==="
echo "Base dir: $BASE_DIR"
echo "API model: $API_SELF_RAG_MODEL"
echo "OpenAI base: $OPENAI_API_BASE"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set."
fi

run_api_self_rag_one "asqa" "$ASQA_INPUT" "$ASQA_DOCS" "$ASQA_RAW_OUT"
run_api_self_rag_one "eli5" "$ELI5_INPUT" "$ELI5_DOCS" "$ELI5_RAW_OUT"
run_api_self_rag_one "qampari" "$QAMPARI_INPUT" "$QAMPARI_DOCS" "$QAMPARI_RAW_OUT"

export SELF_RAG_ASQA_OUTPUT="$ASQA_RAW_OUT"
export SELF_RAG_ELI5_OUTPUT="$ELI5_RAW_OUT"
export SELF_RAG_QAMPARI_OUTPUT="$QAMPARI_RAW_OUT"
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export BASE_DIR="$BASE_DIR"
export RAW_DIR="$CONVERTED_RAW_DIR"
# This is a different base method from vanilla RAG. Always clear any inherited
# vanilla V0 summary unless the caller explicitly opts back in.
export V0_SUMMARY_CSV="${SELF_RAG_V0_SUMMARY_CSV:-}"
export PREPARE_V0="${SELF_RAG_PREPARE_V0:-1}"
export V0_SCORE_MODE="${V0_SCORE_MODE:-eval}"
export FORCE_V0="${FORCE_V0:-0}"
export FORCE_V3="${FORCE_V3:-0}"
export FORCE_EVAL="${FORCE_EVAL:-0}"
export FORCE_MERGE="${FORCE_MERGE:-0}"
# Avoid carrying LIMIT=20 from a smoke test into the full COVER run. Use
# COVER_LIMIT if a deliberate COVER-side limit is desired.
export LIMIT="${COVER_LIMIT:-}"

run_step "Run COVER-RAG v13 plug-in on API Self-RAG outputs" \
  bash scripts/run_cover_v3_self_rag_nohup.sh

echo
echo "=== Outputs ==="
echo "API Self-RAG raw outputs: $API_RAW_DIR"
echo "Converted ALCE inputs: $CONVERTED_RAW_DIR"
echo "COVER-RAG result dir: $BASE_DIR"
echo "Required metrics: $BASE_DIR/required_metrics_summary.csv"
echo "Done."
