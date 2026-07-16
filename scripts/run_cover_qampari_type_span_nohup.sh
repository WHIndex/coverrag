#!/usr/bin/env bash
# Next QAMPARI experiment: answer-type-aware joint selection with evidence spans.
#
# This is a thin wrapper around run_cover_qampari_select_nohup.sh. It writes to
# a new directory so the previous qampari_gtr_cover_select result is preserved.
#
# Recommended launch:
#   nohup bash scripts/run_cover_qampari_type_span_nohup.sh \
#     > logs/qampari_type_span_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

export OUT_DIR="${OUT_DIR:-result/qampari_gtr_cover_select_type_span}"
export CACHE_FILE="${CACHE_FILE:-cache/cover_qampari_type_span_cache.json}"
export SELECTION_MODE="${SELECTION_MODE:-joint}"
export JOINT_REQUIRE_TYPE_MATCH="${JOINT_REQUIRE_TYPE_MATCH:-1}"
export STRICT_QUESTION_TYPE_GUARD="${STRICT_QUESTION_TYPE_GUARD:-0}"
export JOINT_DOCS="${JOINT_DOCS:-10}"
export JOINT_CANDIDATE_SPANS="${JOINT_CANDIDATE_SPANS:-120}"
export MAX_OUTPUT_ITEMS="${MAX_OUTPUT_ITEMS:-5}"
export BACKFILL_ORIGINAL_UNTIL="${BACKFILL_ORIGINAL_UNTIL:-5}"
export BACKFILL_REQUIRES_TYPE_SAFE="${BACKFILL_REQUIRES_TYPE_SAFE:-0}"

# This reproduces the earlier type-span run that achieved the best QAMPARI
# correctness tradeoff.  The cleaner balanced script uses empty instead.
export EMPTY_OUTPUT_POLICY="${EMPTY_OUTPUT_POLICY:-original}"
export CLAIM_CITATION_METRICS="${CLAIM_CITATION_METRICS:-1}"
export DIAGNOSE_SELECTION="${DIAGNOSE_SELECTION:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_cover_qampari_select_nohup.sh"
