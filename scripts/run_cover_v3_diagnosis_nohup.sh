#!/usr/bin/env bash
# Diagnose an existing three-dataset COVER-RAG v3 run.
#
# Usage:
#   BASE_DIR=result/cover_v3_three_datasets_v31_full \
#   LOG_FILE=logs/cover_v3_three_v31_full.log \
#   nohup bash scripts/run_cover_v3_diagnosis_nohup.sh > logs/diagnose_cover_v3.log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON="${PYTHON:-python}"
BASE_DIR="${BASE_DIR:-result/cover_v3_three_datasets_v31_full}"
LOG_FILE="${LOG_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/diagnosis}"

ASQA_BASELINE="${ASQA_BASELINE:-result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
ASQA_COVER="${ASQA_COVER:-$BASE_DIR/cover_v3/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json}"
ELI5_BASELINE="${ELI5_BASELINE:-result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json}"
ELI5_COVER="${ELI5_COVER:-$BASE_DIR/cover_v3/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.cover_v3.json}"
QAMPARI_BASELINE="${QAMPARI_BASELINE:-result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
QAMPARI_COVER="${QAMPARI_COVER:-$BASE_DIR/cover_v3/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json}"

mkdir -p "$OUTPUT_DIR" logs

pairs=()
if [ -f "$ASQA_BASELINE" ] && [ -f "$ASQA_COVER" ]; then
  pairs+=("asqa:$ASQA_BASELINE:$ASQA_COVER")
else
  echo "WARNING: ASQA pair missing; skip."
fi
if [ -f "$ELI5_BASELINE" ] && [ -f "$ELI5_COVER" ]; then
  pairs+=("eli5:$ELI5_BASELINE:$ELI5_COVER")
else
  echo "WARNING: ELI5 pair missing; skip."
fi
if [ -f "$QAMPARI_BASELINE" ] && [ -f "$QAMPARI_COVER" ]; then
  pairs+=("qampari:$QAMPARI_BASELINE:$QAMPARI_COVER")
else
  echo "WARNING: QAMPARI pair missing; skip."
fi

if [ "${#pairs[@]}" -eq 0 ]; then
  echo "No complete baseline/cover pairs found."
  exit 1
fi

cmd=(
  "$PYTHON" scripts/diagnose_cover_v3_failures.py
  --pairs "${pairs[@]}"
  --output-dir "$OUTPUT_DIR"
  --pretty
)

if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
  cmd+=(--log "$LOG_FILE")
else
  echo "No LOG_FILE supplied or log file not found; diagnosis will skip log counters."
fi

echo "=== COVER-RAG v3 diagnosis ==="
echo "Base dir: $BASE_DIR"
echo "Output dir: $OUTPUT_DIR"
printf 'Pair: %s\n' "${pairs[@]}"
echo "+ ${cmd[*]}"
"${cmd[@]}"

echo
echo "Diagnosis outputs:"
echo "  $OUTPUT_DIR/diagnosis_summary.json"
echo "  $OUTPUT_DIR/diagnosis_summary.csv"
echo "  $OUTPUT_DIR/answer_regressions.csv"
