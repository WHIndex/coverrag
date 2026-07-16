#!/usr/bin/env bash
# Build and train a coverage-aware evidence reranker.
#
# IMPORTANT:
#   Use official train/dev files here. Do not train on alce/data/*_eval_*.json
#   for paper results unless you are only doing diagnostics with
#   ALLOW_EVAL_INPUT=1.
#
# Example:
#   TRAIN_INPUTS="data_train/asqa_train_top100.json data_train/eli5_train_top100.json data_train/qampari_train_top100.json" \
#   DEV_INPUTS="data_train/asqa_dev_top100.json data_train/eli5_dev_top100.json data_train/qampari_dev_top100.json" \
#   nohup bash scripts/run_coverage_reranker_training_nohup.sh \
#     > logs/coverage_reranker_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export PYTHON="${PYTHON:-/home/anaconda/envs/alce/bin/python}"
export TRAIN_INPUTS="${TRAIN_INPUTS:-}"
export DEV_INPUTS="${DEV_INPUTS:-}"
export RERANKER_WORK_DIR="${RERANKER_WORK_DIR:-result/coverage_reranker}"
export BASE_MODEL="${BASE_MODEL:-BAAI/bge-reranker-base}"
export DEVICE="${DEVICE:-cuda}"
export EPOCHS="${EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LR="${LR:-2e-5}"
export MAX_LENGTH="${MAX_LENGTH:-384}"
export SEED="${SEED:-42}"
export ALLOW_EVAL_INPUT="${ALLOW_EVAL_INPUT:-0}"
export MAX_ITEMS="${MAX_ITEMS:-}"
export MAX_DOCS="${MAX_DOCS:-100}"
export MAX_SENTENCES_PER_DOC="${MAX_SENTENCES_PER_DOC:-12}"
export POSITIVES_PER_UNIT="${POSITIVES_PER_UNIT:-2}"
export NEGATIVES_PER_UNIT="${NEGATIVES_PER_UNIT:-4}"
export MIN_UNIT_OVERLAP="${MIN_UNIT_OVERLAP:-0.60}"

mkdir -p "$RERANKER_WORK_DIR" logs

if [ -z "$TRAIN_INPUTS" ]; then
  echo "FAILED: TRAIN_INPUTS is empty. Provide official train split JSON files."
  exit 1
fi

for path in $TRAIN_INPUTS; do
  if [ ! -s "$path" ]; then
    echo "FAILED: missing train input: $path"
    echo "Provide official train top100/full-corpus candidate files, or set ALLOW_EVAL_INPUT=1 only for diagnostics."
    exit 1
  fi
done
for path in $DEV_INPUTS; do
  if [ ! -s "$path" ]; then
    echo "FAILED: missing dev input: $path"
    echo "Unset DEV_INPUTS or provide real dev files."
    exit 1
  fi
done

BUILD_TRAIN_ARGS=(
  "$PYTHON" scripts/build_coverage_reranker_data.py
  --input $TRAIN_INPUTS
  --output "$RERANKER_WORK_DIR/train.jsonl"
  --seed "$SEED"
  --max-docs "$MAX_DOCS"
  --max-sentences-per-doc "$MAX_SENTENCES_PER_DOC"
  --positives-per-unit "$POSITIVES_PER_UNIT"
  --negatives-per-unit "$NEGATIVES_PER_UNIT"
  --min-unit-overlap "$MIN_UNIT_OVERLAP"
)
if [ -n "$MAX_ITEMS" ]; then
  BUILD_TRAIN_ARGS+=(--max-items "$MAX_ITEMS")
fi
if [ "$ALLOW_EVAL_INPUT" = "1" ]; then
  BUILD_TRAIN_ARGS+=(--allow-eval-input)
fi

echo "=== Build coverage-reranker train data ==="
echo "+ ${BUILD_TRAIN_ARGS[*]}"
"${BUILD_TRAIN_ARGS[@]}"

TRAIN_ARGS=(
  "$PYTHON" scripts/train_coverage_reranker.py
  --train-file "$RERANKER_WORK_DIR/train.jsonl"
  --output-dir "$RERANKER_WORK_DIR/model"
  --base-model "$BASE_MODEL"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-length "$MAX_LENGTH"
  --seed "$SEED"
)

if [ -n "$DEV_INPUTS" ]; then
  BUILD_DEV_ARGS=(
    "$PYTHON" scripts/build_coverage_reranker_data.py
    --input $DEV_INPUTS
    --output "$RERANKER_WORK_DIR/dev.jsonl"
    --seed "$SEED"
    --max-docs "$MAX_DOCS"
    --max-sentences-per-doc "$MAX_SENTENCES_PER_DOC"
    --positives-per-unit "$POSITIVES_PER_UNIT"
    --negatives-per-unit "$NEGATIVES_PER_UNIT"
    --min-unit-overlap "$MIN_UNIT_OVERLAP"
  )
  if [ -n "$MAX_ITEMS" ]; then
    BUILD_DEV_ARGS+=(--max-items "$MAX_ITEMS")
  fi
  if [ "$ALLOW_EVAL_INPUT" = "1" ]; then
    BUILD_DEV_ARGS+=(--allow-eval-input)
  fi
  echo
  echo "=== Build coverage-reranker dev data ==="
  echo "+ ${BUILD_DEV_ARGS[*]}"
  "${BUILD_DEV_ARGS[@]}"
  TRAIN_ARGS+=(--dev-file "$RERANKER_WORK_DIR/dev.jsonl")
fi

echo
echo "=== Train coverage-aware reranker ==="
echo "+ ${TRAIN_ARGS[*]}"
"${TRAIN_ARGS[@]}"

echo
echo "Coverage-aware reranker saved to: $RERANKER_WORK_DIR/model"
echo "Use with COVER-RAG:"
echo "  COVERAGE_RERANKER_MODEL=$RERANKER_WORK_DIR/model bash scripts/run_cover_v3_dynamic_retrieval_nohup.sh"
