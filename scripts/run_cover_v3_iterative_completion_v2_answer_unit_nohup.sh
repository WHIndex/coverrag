#!/usr/bin/env bash
# COVER-RAG vNext v2: iterative completion with answer-unit relevance gating.
#
# Compared with v1, this run keeps iterative recall completion but rejects
# expansion candidates that are merely supported webpage titles/background facts.
# The goal is to improve main-task recall/correctness without sacrificing
# sentence-level and claim-level citation quality.
#
# Launch:
#   nohup bash scripts/run_cover_v3_iterative_completion_v2_answer_unit_nohup.sh \
#     > logs/cover_v3_iterative_completion_v2_answer_unit_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [[ -n "${LIMIT:-}" && -z "${EXPERIMENT_NAME:-}" ]]; then
  export EXPERIMENT_NAME="cover_v3_iterative_completion_v2_answer_unit_smoke_limit${LIMIT}"
else
  export EXPERIMENT_NAME="${EXPERIMENT_NAME:-cover_v3_iterative_completion_v2_answer_unit}"
fi
if [[ -n "${LIMIT:-}" && -z "${CACHE_NAMESPACE:-}" ]]; then
  export CACHE_NAMESPACE="iterative_completion_v2_answer_unit_smoke_limit${LIMIT}_"
else
  export CACHE_NAMESPACE="${CACHE_NAMESPACE:-iterative_completion_v2_answer_unit_}"
fi
export USE_EXTERNAL_CACHE_NAMESPACE="${USE_EXTERNAL_CACHE_NAMESPACE:-1}"
export PREPARE_V0="${PREPARE_V0:-0}"

# Keep v1's iterative search, but make accepted additions answer-unit precise.
export ITERATIVE_COMPLETION_ROUNDS="${ITERATIVE_COMPLETION_ROUNDS:-2}"
export ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND="${ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND:-2}"
export ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND="${ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND:-5}"
export ITERATIVE_COMPLETION_EVIDENCE_SENTENCES="${ITERATIVE_COMPLETION_EVIDENCE_SENTENCES:-30}"
export ITERATIVE_COMPLETION_CANDIDATE_SPANS="${ITERATIVE_COMPLETION_CANDIDATE_SPANS:-64}"
export ITERATIVE_COMPLETION_MAX_CANDIDATES="${ITERATIVE_COMPLETION_MAX_CANDIDATES:-12}"

# New guard: NLI support is necessary but not sufficient; candidates must also
# directly answer the question/missing target.
export ANSWER_UNIT_RELEVANCE_GATE="${ANSWER_UNIT_RELEVANCE_GATE:-1}"
export ANSWER_UNIT_MIN_DIRECTNESS="${ANSWER_UNIT_MIN_DIRECTNESS:-0.46}"
export EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS="${EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS:-0.62}"

# Allow a little answer growth, while final audit/candidate selection keeps
# unsupported or low-utility additions out.
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-38}"
export MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-28}"

echo "=== COVER-RAG iterative completion v2 answer-unit launcher ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Answer-unit gate: $ANSWER_UNIT_RELEVANCE_GATE"
echo "Directness thresholds: non_explanatory=$ANSWER_UNIT_MIN_DIRECTNESS explanatory=$EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS"
echo "Rounds: $ITERATIVE_COMPLETION_ROUNDS"

bash scripts/run_cover_v3_recall_repair_nohup.sh
