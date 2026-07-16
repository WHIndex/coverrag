#!/usr/bin/env bash
# QAMPARI next run: type-span selection plus deterministic question-type guard.
#
# Recommended smoke test:
#   LIMIT=100 FORCE_SELECT=1 FORCE_EVAL=1 FORCE_MERGE=1 FORCE_CLAIM_METRICS=1 \
#   nohup bash scripts/run_cover_qampari_type_guard_nohup.sh \
#     > logs/qampari_type_guard_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Recommended full run:
#   FORCE_SELECT=1 FORCE_EVAL=1 FORCE_MERGE=1 FORCE_CLAIM_METRICS=1 \
#   nohup bash scripts/run_cover_qampari_type_guard_nohup.sh \
#     > logs/qampari_type_guard_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

export OUT_DIR="${OUT_DIR:-result/qampari_gtr_cover_select_type_guard}"
export CACHE_FILE="${CACHE_FILE:-cache/cover_qampari_type_guard_cache.json}"
export SELECTION_MODE="${SELECTION_MODE:-joint}"
export JOINT_REQUIRE_TYPE_MATCH="${JOINT_REQUIRE_TYPE_MATCH:-1}"
export STRICT_QUESTION_TYPE_GUARD="${STRICT_QUESTION_TYPE_GUARD:-1}"
export JOINT_DOCS="${JOINT_DOCS:-10}"
export JOINT_CANDIDATE_SPANS="${JOINT_CANDIDATE_SPANS:-120}"
export MAX_OUTPUT_ITEMS="${MAX_OUTPUT_ITEMS:-5}"

# Slightly lower than the old value 5. This keeps QAMPARI Rec.-5 from collapsing
# while reducing unverified original items that passed only by backfill.
export BACKFILL_ORIGINAL_UNTIL="${BACKFILL_ORIGINAL_UNTIL:-3}"
export BACKFILL_REQUIRES_TYPE_SAFE="${BACKFILL_REQUIRES_TYPE_SAFE:-1}"

# Empty outputs are cleaner than preserving refusal text such as
# "The provided documents do not contain...".
export EMPTY_OUTPUT_POLICY="${EMPTY_OUTPUT_POLICY:-empty}"

export CLAIM_CITATION_METRICS="${CLAIM_CITATION_METRICS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_cover_qampari_select_nohup.sh"
