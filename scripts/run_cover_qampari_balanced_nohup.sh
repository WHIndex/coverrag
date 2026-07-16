#!/usr/bin/env bash
# QAMPARI balanced configuration for COVER-RAG.
#
# This is the recommended QAMPARI setting after the type-guard ablation:
# it keeps empty-output cleanup and answer-type-aware joint selection, but does
# not apply the stricter deterministic type guard that hurt QAMPARI Rec.-5.
#
# Recommended launch:
#   FORCE_SELECT=1 FORCE_EVAL=1 FORCE_MERGE=1 FORCE_CLAIM_METRICS=1 \
#   nohup bash scripts/run_cover_qampari_balanced_nohup.sh \
#     > logs/qampari_balanced_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

export OUT_DIR="${OUT_DIR:-result/qampari_gtr_cover_select_balanced}"

# Reuse the same joint-selection cache when possible. The joint selector cache
# key does not include the stricter post-filtering flags, so this avoids paying
# for the same 1000 LLM selection calls again.
export CACHE_FILE="${CACHE_FILE:-cache/cover_qampari_type_guard_cache.json}"

export SELECTION_MODE="${SELECTION_MODE:-joint}"
export JOINT_REQUIRE_TYPE_MATCH="${JOINT_REQUIRE_TYPE_MATCH:-1}"
export STRICT_QUESTION_TYPE_GUARD="${STRICT_QUESTION_TYPE_GUARD:-0}"
export JOINT_DOCS="${JOINT_DOCS:-10}"
export JOINT_CANDIDATE_SPANS="${JOINT_CANDIDATE_SPANS:-120}"

export MAX_OUTPUT_ITEMS="${MAX_OUTPUT_ITEMS:-5}"
export BACKFILL_ORIGINAL_UNTIL="${BACKFILL_ORIGINAL_UNTIL:-5}"
export BACKFILL_REQUIRES_TYPE_SAFE="${BACKFILL_REQUIRES_TYPE_SAFE:-0}"
export EMPTY_OUTPUT_POLICY="${EMPTY_OUTPUT_POLICY:-empty}"

export CLAIM_CITATION_METRICS="${CLAIM_CITATION_METRICS:-1}"
export DIAGNOSE_SELECTION="${DIAGNOSE_SELECTION:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_cover_qampari_select_nohup.sh"
