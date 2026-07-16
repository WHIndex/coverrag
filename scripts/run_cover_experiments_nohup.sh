#!/usr/bin/env bash
set -euo pipefail

# Run from the ALCE/COVER-RAG repository root:
#   mkdir -p logs
#   nohup bash scripts/run_cover_experiments_nohup.sh > logs/cover_experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# This script only orchestrates scripts/ code:
#   1. rerun COVER-RAG v0 post-hoc audit for raw ALCE result JSON files
#   2. rerun COVER-RAG v1 answer revision with --final-audit enabled
#   3. write v0, v1, and v0-vs-v1 summary tables
#
# Default input discovery:
#   result/origin/*-gpt-4o-mini-*-shot2-ndoc5-42.json
#
# Useful overrides:
#   RESULT_DIR=result/current bash scripts/run_cover_experiments_nohup.sh
#   PATTERN='asqa-*.json' bash scripts/run_cover_experiments_nohup.sh
#   INPUTS='result/a.json result/b.json' bash scripts/run_cover_experiments_nohup.sh
#   PYTHON_BIN=python3 bash scripts/run_cover_experiments_nohup.sh
#   SKIP_V0=1 bash scripts/run_cover_experiments_nohup.sh
#   FINAL_AUDIT=0 bash scripts/run_cover_experiments_nohup.sh
#
# Optional raw answer generation before v0/v1:
#   RUN_RAW=1 RAW_CONFIGS='scripts/asqa.yaml scripts/eli5.yaml scripts/qampari.yaml' \
#     bash scripts/run_cover_experiments_nohup.sh
#
# RAW_CONFIGS should point to configs for your existing ALCE generation script
# (`cover_run.py` or `run.py`). v0/v1 themselves do not generate raw answers;
# they consume the raw result/*.json files produced by ALCE generation.

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-result/origin}"
PATTERN="${PATTERN:-*-gpt-4o-mini-*-shot2-ndoc5-42.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"
CACHE_DIR="${CACHE_DIR:-cache}"
RUN_RAW="${RUN_RAW:-0}"
RAW_CONFIGS="${RAW_CONFIGS:-}"
GENERATOR="${GENERATOR:-}"
SKIP_V0="${SKIP_V0:-0}"
SKIP_V1="${SKIP_V1:-0}"
FORCE_V0="${FORCE_V0:-1}"
FORCE_V1="${FORCE_V1:-1}"
FINAL_AUDIT="${FINAL_AUDIT:-1}"

AUDIT_DIR="${AUDIT_DIR:-$OUTPUT_ROOT/cover_audit}"
V1_DIR="${V1_DIR:-$OUTPUT_ROOT/cover_v1}"

DECOMPOSER="${DECOMPOSER:-llm}"
DECOMPOSE_MODEL="${DECOMPOSE_MODEL:-gpt-4o-mini}"
VERIFIER="${VERIFIER:-llm}"
LLM_VERIFY_MODEL="${LLM_VERIFY_MODEL:-gpt-4o-mini}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"
EVIDENCE_SCOPE="${EVIDENCE_SCOPE:-cited_then_all}"

REVISION_MODE="${REVISION_MODE:-auto}"
ANSWER_FORMAT="${ANSWER_FORMAT:-auto}"
CITATION_POLICY="${CITATION_POLICY:-source}"
MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-3}"
MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-16}"
MAX_QAMPARI_ITEMS="${MAX_QAMPARI_ITEMS:-80}"
ALLOW_CITATION_REPAIR="${ALLOW_CITATION_REPAIR:-0}"
INCLUDE_BACKGROUND_CLAIMS="${INCLUDE_BACKGROUND_CLAIMS:-0}"

mkdir -p logs "$AUDIT_DIR" "$V1_DIR" "$CACHE_DIR"

INPUT_ARGS=(--result-dir "$RESULT_DIR" --pattern "$PATTERN")
if [[ -n "${INPUTS:-}" ]]; then
  # INPUTS is a whitespace-separated list of JSON files.
  # Example:
  #   INPUTS='result/asqa-xxx.json result/eli5-xxx.json'
  read -r -a INPUT_FILES <<< "$INPUTS"
  INPUT_ARGS=(--inputs "${INPUT_FILES[@]}")
fi

find_raw_inputs() {
  FOUND_INPUTS=()
  if [[ -n "${INPUTS:-}" ]]; then
    local file
    for file in "${INPUT_FILES[@]}"; do
      if [[ -f "$file" ]]; then
        FOUND_INPUTS+=("$file")
      fi
    done
    return
  fi

  shopt -s nullglob
  local candidates=("$RESULT_DIR"/$PATTERN)
  shopt -u nullglob

  local file
  for file in "${candidates[@]}"; do
    case "$file" in
      *.cover_audit.json|*.cover_v1.json|*.score|*.cover_score|*".smoke"*|*".debug"*)
        continue
        ;;
    esac
    if [[ -f "$file" ]]; then
      FOUND_INPUTS+=("$file")
    fi
  done
}

choose_generator() {
  if [[ -n "$GENERATOR" ]]; then
    echo "$GENERATOR"
  elif [[ -f cover_run.py ]]; then
    echo "cover_run.py"
  elif [[ -f run.py ]]; then
    echo "run.py"
  else
    echo ""
  fi
}

echo "=== COVER-RAG experiment batch ==="
echo "Python: $PYTHON_BIN"
echo "Input args: ${INPUT_ARGS[*]}"
echo "V0 output dir: $AUDIT_DIR"
echo "V1 output dir: $V1_DIR"
echo "Decomposer: $DECOMPOSER"
echo "Verifier: $VERIFIER"
echo "Evidence scope: $EVIDENCE_SCOPE"
echo "Skip v0: $SKIP_V0"
echo "Skip v1: $SKIP_V1"
echo "Force v0: $FORCE_V0"
echo "Force v1: $FORCE_V1"
echo "Final audit: $FINAL_AUDIT"

if [[ "$RUN_RAW" == "1" || "$RUN_RAW" == "true" || "$RUN_RAW" == "yes" ]]; then
  echo
  echo "=== Optional Stage 0: generate raw ALCE result JSON files ==="
  if [[ -z "$RAW_CONFIGS" ]]; then
    echo "ERROR: RUN_RAW=1 was set, but RAW_CONFIGS is empty."
    echo "Example:"
    echo "  RUN_RAW=1 RAW_CONFIGS='scripts/asqa.yaml scripts/eli5.yaml scripts/qampari.yaml' bash scripts/run_cover_experiments_nohup.sh"
    exit 2
  fi

  GENERATOR_SCRIPT="$(choose_generator)"
  if [[ -z "$GENERATOR_SCRIPT" || ! -f "$GENERATOR_SCRIPT" ]]; then
    echo "ERROR: cannot find raw generation script."
    echo "Set GENERATOR explicitly, for example:"
    echo "  GENERATOR=cover_run.py RUN_RAW=1 RAW_CONFIGS='scripts/asqa.yaml' bash scripts/run_cover_experiments_nohup.sh"
    exit 2
  fi

  read -r -a RAW_CONFIG_FILES <<< "$RAW_CONFIGS"
  for config in "${RAW_CONFIG_FILES[@]}"; do
    echo "+ $PYTHON_BIN $GENERATOR_SCRIPT --config $config"
    "$PYTHON_BIN" "$GENERATOR_SCRIPT" --config "$config"
  done
fi

find_raw_inputs
if [[ "${#FOUND_INPUTS[@]}" -eq 0 ]]; then
  echo
  echo "ERROR: No raw ALCE result JSON files found."
  echo
  echo "What this means:"
  echo "  v0/v1 are post-processing stages. They need raw answer JSON files that"
  echo "  were already produced by your ALCE generation script, usually under result/origin/."
  echo
  echo "Current search:"
  echo "  RESULT_DIR=$RESULT_DIR"
  echo "  PATTERN=$PATTERN"
  if [[ -n "${INPUTS:-}" ]]; then
    echo "  INPUTS=$INPUTS"
  fi
  echo
  echo "Fix option A: point this script to existing raw JSON files:"
  echo "  INPUTS='result/asqa-xxx.json result/eli5-xxx.json result/qampari-xxx.json' bash scripts/run_cover_experiments_nohup.sh"
  echo
  echo "Fix option B: loosen the pattern if your file names differ:"
  echo "  PATTERN='*.json' bash scripts/run_cover_experiments_nohup.sh"
  echo
  echo "Fix option C: generate raw answers first, then v0/v1:"
  echo "  RUN_RAW=1 RAW_CONFIGS='scripts/asqa.yaml scripts/eli5.yaml scripts/qampari.yaml' bash scripts/run_cover_experiments_nohup.sh"
  echo
  echo "Tip: raw files should look like normal ALCE outputs, not .cover_audit.json,"
  echo "     not .cover_v1.json, and not .score files."
  exit 2
fi

echo
echo "Found ${#FOUND_INPUTS[@]} raw input JSON file(s):"
printf '  - %s\n' "${FOUND_INPUTS[@]}"

echo
echo "=== Stage 1/3: rerun COVER-RAG v0 post-hoc audit ==="
if [[ "$SKIP_V0" == "1" || "$SKIP_V0" == "true" || "$SKIP_V0" == "yes" ]]; then
  echo "Skipping v0. Existing summary should be available at:"
  echo "  $AUDIT_DIR/cover_audit_comparison.csv"
else
  V0_FORCE_ARGS=()
  if [[ "$FORCE_V0" == "1" || "$FORCE_V0" == "true" || "$FORCE_V0" == "yes" ]]; then
    V0_FORCE_ARGS=(--force-cover --force-eval --force-merge)
  fi

  "$PYTHON_BIN" scripts/run_cover_audit_batch.py \
    --python "$PYTHON_BIN" \
    "${INPUT_ARGS[@]}" \
    --output-dir "$AUDIT_DIR" \
    --summary-csv "$AUDIT_DIR/cover_audit_comparison.csv" \
    --summary-md "$AUDIT_DIR/cover_audit_comparison.md" \
    --decomposer "$DECOMPOSER" \
    --decompose-model "$DECOMPOSE_MODEL" \
    --verifier "$VERIFIER" \
    --llm-verify-model "$LLM_VERIFY_MODEL" \
    --nli-model "$NLI_MODEL" \
    --evidence-scope "$EVIDENCE_SCOPE" \
    --cache-file "$CACHE_DIR/cover_audit_cache.json" \
    "${V0_FORCE_ARGS[@]}"
fi

echo
echo "=== Stage 2/3: rerun COVER-RAG v1 revise + final audit ==="
if [[ "$SKIP_V1" == "1" || "$SKIP_V1" == "true" || "$SKIP_V1" == "yes" ]]; then
  echo "Skipping v1. Existing summary should be available at:"
  echo "  $V1_DIR/cover_v1_comparison.csv"
else
  V1_FORCE_ARGS=()
  if [[ "$FORCE_V1" == "1" || "$FORCE_V1" == "true" || "$FORCE_V1" == "yes" ]]; then
    V1_FORCE_ARGS=(--force-revise --force-eval --force-merge)
  fi
  V1_FINAL_AUDIT_ARGS=(--final-audit)
  if [[ "$FINAL_AUDIT" == "0" || "$FINAL_AUDIT" == "false" || "$FINAL_AUDIT" == "no" ]]; then
    V1_FINAL_AUDIT_ARGS=(--no-final-audit)
  fi

  "$PYTHON_BIN" scripts/run_cover_v1_batch.py \
    --python "$PYTHON_BIN" \
    "${INPUT_ARGS[@]}" \
    --output-dir "$V1_DIR" \
    --summary-csv "$V1_DIR/cover_v1_comparison.csv" \
    --summary-md "$V1_DIR/cover_v1_comparison.md" \
    --audit-summary-csv "$AUDIT_DIR/cover_audit_comparison.csv" \
    --combined-csv "$V1_DIR/cover_v1_vs_audit_comparison.csv" \
    --combined-md "$V1_DIR/cover_v1_vs_audit_comparison.md" \
    --decomposer "$DECOMPOSER" \
    --decompose-model "$DECOMPOSE_MODEL" \
    --verifier "$VERIFIER" \
    --llm-verify-model "$LLM_VERIFY_MODEL" \
    --revision-mode "$REVISION_MODE" \
    --answer-format "$ANSWER_FORMAT" \
    --citation-policy "$CITATION_POLICY" \
    --max-citations-per-claim "$MAX_CITATIONS_PER_CLAIM" \
    --max-verified-claims "$MAX_VERIFIED_CLAIMS" \
    --max-qampari-items "$MAX_QAMPARI_ITEMS" \
    --nli-model "$NLI_MODEL" \
    --evidence-scope "$EVIDENCE_SCOPE" \
    --cache-file "$CACHE_DIR/cover_v1_cache.json" \
    "$(if [[ "$ALLOW_CITATION_REPAIR" == "0" || "$ALLOW_CITATION_REPAIR" == "false" || "$ALLOW_CITATION_REPAIR" == "no" ]]; then echo "--no-allow-citation-repair"; else echo "--allow-citation-repair"; fi)" \
    "$(if [[ "$INCLUDE_BACKGROUND_CLAIMS" == "0" || "$INCLUDE_BACKGROUND_CLAIMS" == "false" || "$INCLUDE_BACKGROUND_CLAIMS" == "no" ]]; then echo "--no-include-background-claims"; else echo "--include-background-claims"; fi)" \
    "${V1_FINAL_AUDIT_ARGS[@]}" \
    "${V1_FORCE_ARGS[@]}"
fi

echo
echo "=== Stage 3/3: summaries written ==="
echo "V0 detail:      $AUDIT_DIR/cover_audit_comparison.csv"
echo "V1 detail:      $V1_DIR/cover_v1_comparison.csv"
echo "V0 vs V1 table: $V1_DIR/cover_v1_vs_audit_comparison.csv"
echo "Markdown table: $V1_DIR/cover_v1_vs_audit_comparison.md"

if [[ -n "${RAW_SUMMARY:-}" ]]; then
  echo
  echo "=== Optional: build raw/v0/v1 compact table from RAW_SUMMARY ==="
  "$PYTHON_BIN" scripts/make_key_result_table.py \
    --raw "$RAW_SUMMARY" \
    --v0-v1 "$V1_DIR/cover_v1_vs_audit_comparison.csv" \
    --v0-detail "$AUDIT_DIR/cover_audit_comparison.csv" \
    --v1-detail "$V1_DIR/cover_v1_comparison.csv" \
    --output-csv "$V1_DIR/key_raw_v0_v1_comparison.csv" \
    --output-md "$V1_DIR/key_raw_v0_v1_comparison.md"
  echo "Raw/V0/V1 table: $V1_DIR/key_raw_v0_v1_comparison.csv"
else
  echo
  echo "RAW_SUMMARY is not set, so raw/v0/v1 compact table is skipped."
  echo "This is intentional: v0/v1 rerun and v0-vs-v1 summary do not depend on stale raw summaries."
fi
