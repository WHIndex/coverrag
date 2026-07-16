#!/usr/bin/env bash
# COVER-RAG vNext: iterative recall completion.
#
# This keeps the v13 recall-completion recipe intact, then adds extra rounds:
# supported claims -> missing targets -> dynamic reranking over the doc pool ->
# candidate generation -> NLI verification -> merge.
#
# Launch:
#   nohup bash scripts/run_cover_v3_iterative_completion_nohup.sh \
#     > logs/cover_v3_iterative_completion_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-cover_v3_iterative_completion_v1}"
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-iterative_completion_v1_}"
export USE_EXTERNAL_CACHE_NAMESPACE="${USE_EXTERNAL_CACHE_NAMESPACE:-1}"

# Preserve the stable v13 baseline audit unless explicitly rebuilding v0.
export PREPARE_V0="${PREPARE_V0:-0}"

# Main change: repeat answer-unit completion after the first verified draft.
export ITERATIVE_COMPLETION_ROUNDS="${ITERATIVE_COMPLETION_ROUNDS:-2}"
export ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND="${ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND:-2}"
export ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND="${ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND:-4}"
export ITERATIVE_COMPLETION_EVIDENCE_SENTENCES="${ITERATIVE_COMPLETION_EVIDENCE_SENTENCES:-30}"
export ITERATIVE_COMPLETION_CANDIDATE_SPANS="${ITERATIVE_COMPLETION_CANDIDATE_SPANS:-64}"
export ITERATIVE_COMPLETION_MAX_CANDIDATES="${ITERATIVE_COMPLETION_MAX_CANDIDATES:-12}"

# Give the iterative loop room to add a few claims without forcing deletion of
# already-supported answer content. The verifier/final-audit gates still decide
# what survives.
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-40}"
export MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-28}"

echo "=== COVER-RAG iterative completion launcher ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Rounds: $ITERATIVE_COMPLETION_ROUNDS"
echo "New claims/round: $ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND"
echo "Verifications/round: $ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND"
echo "Evidence sentences/round: $ITERATIVE_COMPLETION_EVIDENCE_SENTENCES"

bash scripts/run_cover_v3_recall_repair_nohup.sh
