#!/usr/bin/env bash
# QAMPARI-only COVER-RAG run aligned with ALCE's QAMPARI metrics.
#
# This uses answer-set selection instead of claim rewriting:
#   original cited answer items -> item-level verification/citation repair
#   -> optional top100 evidence expansion -> max-5 cited answer list.
#
# Recommended launch:
#   nohup bash scripts/run_cover_qampari_select_nohup.sh \
#     > logs/qampari_cover_select_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON="${PYTHON:-python}"
RAW="${RAW:-result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
DOCS="${DOCS:-data/qampari_eval_gtr_top100.json}"
OUT_DIR="${OUT_DIR:-result/qampari_gtr_cover_select}"
OUT_JSON="$OUT_DIR/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_qampari.json"
CACHE_FILE="${CACHE_FILE:-cache/cover_qampari_select_cache.json}"

ITEM_VERIFIER="${ITEM_VERIFIER:-llm}"
ITEM_VERIFY_MODEL="${ITEM_VERIFY_MODEL:-gpt-4o-mini}"
SELECTION_MODE="${SELECTION_MODE:-joint}"       # joint | itemwise
JOINT_MODEL="${JOINT_MODEL:-gpt-4o-mini}"
JOINT_DOCS="${JOINT_DOCS:-12}"
JOINT_CANDIDATE_SPANS="${JOINT_CANDIDATE_SPANS:-80}"
JOINT_REQUIRE_TYPE_MATCH="${JOINT_REQUIRE_TYPE_MATCH:-1}"
STRICT_QUESTION_TYPE_GUARD="${STRICT_QUESTION_TYPE_GUARD:-0}"
EXPAND_FROM_DOCS="${EXPAND_FROM_DOCS:-llm}"   # off | extractive | llm
EXPANSION_MODEL="${EXPANSION_MODEL:-gpt-4o-mini}"

MAX_OUTPUT_ITEMS="${MAX_OUTPUT_ITEMS:-5}"
BACKFILL_ORIGINAL_UNTIL="${BACKFILL_ORIGINAL_UNTIL:-5}"
BACKFILL_REQUIRES_TYPE_SAFE="${BACKFILL_REQUIRES_TYPE_SAFE:-0}"
EMPTY_OUTPUT_POLICY="${EMPTY_OUTPUT_POLICY:-empty}"  # empty | original
MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-8}"
EXPANSION_DOCS="${EXPANSION_DOCS:-8}"
REPAIR_TOP_K="${REPAIR_TOP_K:-5}"

FORCE_SELECT="${FORCE_SELECT:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
FORCE_CLAIM_METRICS="${FORCE_CLAIM_METRICS:-0}"
CLAIM_CITATION_METRICS="${CLAIM_CITATION_METRICS:-1}"
DIAGNOSE_SELECTION="${DIAGNOSE_SELECTION:-1}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUT_DIR" logs cache

echo "=== COVER-QAMPARI answer-set selection ==="
echo "Root: $ROOT_DIR"
echo "Raw: $RAW"
echo "Candidate docs: $DOCS"
echo "Output: $OUT_JSON"
echo "Item verifier: $ITEM_VERIFIER ($ITEM_VERIFY_MODEL)"
echo "Selection mode: $SELECTION_MODE ($JOINT_MODEL)"
echo "Joint docs/spans: $JOINT_DOCS / $JOINT_CANDIDATE_SPANS"
echo "Expansion: $EXPAND_FROM_DOCS ($EXPANSION_MODEL)"
echo "Max output items: $MAX_OUTPUT_ITEMS"
echo "Backfill original until: $BACKFILL_ORIGINAL_UNTIL"
echo "Strict question type guard: $STRICT_QUESTION_TYPE_GUARD"
echo "Backfill requires type-safe: $BACKFILL_REQUIRES_TYPE_SAFE"
echo "Empty output policy: $EMPTY_OUTPUT_POLICY"
echo "Diagnose selection: $DIAGNOSE_SELECTION"

if [ "$FORCE_SELECT" = "1" ] || [ ! -f "$OUT_JSON" ]; then
  cmd=(
    "$PYTHON" scripts/cover_qampari_select.py
    --input "$RAW"
    --output "$OUT_JSON"
    --openai-api
    --selection-mode "$SELECTION_MODE"
    --joint-model "$JOINT_MODEL"
    --joint-docs "$JOINT_DOCS"
    --joint-candidate-spans "$JOINT_CANDIDATE_SPANS"
    --item-verifier "$ITEM_VERIFIER"
    --item-verify-model "$ITEM_VERIFY_MODEL"
    --expand-from-docs "$EXPAND_FROM_DOCS"
    --expansion-model "$EXPANSION_MODEL"
    --candidate-docs-file "$DOCS"
    --max-doc-pool 100
    --repair-top-k "$REPAIR_TOP_K"
    --repair-group-size 3
    --expansion-docs "$EXPANSION_DOCS"
    --max-expansion-candidates "$MAX_EXPANSION_CANDIDATES"
    --max-output-items "$MAX_OUTPUT_ITEMS"
    --backfill-original-until "$BACKFILL_ORIGINAL_UNTIL"
    --empty-output-policy "$EMPTY_OUTPUT_POLICY"
    --keep-unsupported-original
    --drop-obvious-bad
    --max-citations-per-item 1
    --cache-file "$CACHE_FILE"
    --cache-save-every 25
    --pretty
  )
  if [ "$JOINT_REQUIRE_TYPE_MATCH" = "1" ]; then
    cmd+=(--joint-require-type-match)
  else
    cmd+=(--no-joint-require-type-match)
  fi
  if [ "$STRICT_QUESTION_TYPE_GUARD" = "1" ]; then
    cmd+=(--strict-question-type-guard)
  else
    cmd+=(--no-strict-question-type-guard)
  fi
  if [ "$BACKFILL_REQUIRES_TYPE_SAFE" = "1" ]; then
    cmd+=(--backfill-requires-type-safe)
  else
    cmd+=(--no-backfill-requires-type-safe)
  fi
  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "WARNING: COVER-QAMPARI selection failed with exit code $status"
  fi
else
  echo "Selection output exists, skip: $OUT_JSON"
fi

SCORE="$OUT_JSON.score"
if [ -f "$OUT_JSON" ]; then
  NEED_MERGE=0
  if [ "$FORCE_EVAL" = "1" ] || [ ! -f "$SCORE" ]; then
    echo "+ $PYTHON eval.py --f $OUT_JSON --citations"
    "$PYTHON" eval.py --f "$OUT_JSON" --citations
    eval_status=$?
    if [ "$eval_status" -ne 0 ]; then
      echo "WARNING: eval.py --citations failed with exit code $eval_status; retrying task metrics without citations."
      "$PYTHON" eval.py --f "$OUT_JSON"
    fi
    NEED_MERGE=1
  else
    echo "Score exists, skip: $SCORE"
  fi

  CLAIM_SCORE="$OUT_JSON.claim_citation_score"
  if [ "$CLAIM_CITATION_METRICS" = "1" ]; then
    if [ "$FORCE_CLAIM_METRICS" = "1" ] || [ ! -f "$CLAIM_SCORE" ]; then
      echo "+ $PYTHON scripts/compute_claim_citation_metrics.py --result $OUT_JSON --pretty"
      "$PYTHON" scripts/compute_claim_citation_metrics.py \
        --result "$OUT_JSON" \
        --pretty
      NEED_MERGE=1
    else
      echo "Claim citation score exists, skip: $CLAIM_SCORE"
    fi
  fi

  COVER_SCORE="$OUT_JSON.cover_score"
  if [ "$FORCE_MERGE" = "1" ] || [ "$NEED_MERGE" = "1" ] || [ ! -f "$COVER_SCORE" ]; then
    echo "+ $PYTHON scripts/merge_cover_scores.py --result $OUT_JSON --pretty --csv $OUT_DIR/cover_qampari_select_summary.csv"
    "$PYTHON" scripts/merge_cover_scores.py \
      --result "$OUT_JSON" \
      --pretty \
      --csv "$OUT_DIR/cover_qampari_select_summary.csv"
  else
    echo "Merged cover score exists, skip: $COVER_SCORE"
  fi

  if [ "$DIAGNOSE_SELECTION" = "1" ]; then
    echo "+ $PYTHON scripts/diagnose_qampari_selection.py --result $OUT_JSON --output-json $OUT_DIR/qampari_selection_diagnosis.json --output-csv $OUT_DIR/qampari_selection_diagnosis_examples.csv --pretty"
    "$PYTHON" scripts/diagnose_qampari_selection.py \
      --result "$OUT_JSON" \
      --output-json "$OUT_DIR/qampari_selection_diagnosis.json" \
      --output-csv "$OUT_DIR/qampari_selection_diagnosis_examples.csv" \
      --pretty
  fi
fi

echo
echo "Outputs:"
echo "  $OUT_JSON"
echo "  $SCORE"
echo "  $OUT_JSON.cover_score"
echo "  $OUT_DIR/cover_qampari_select_summary.csv"
