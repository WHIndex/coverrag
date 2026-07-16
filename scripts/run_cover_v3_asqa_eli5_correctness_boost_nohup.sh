#!/usr/bin/env bash
# Conservative correctness-boost run for COVER-RAG v3 on ASQA + ELI5.
#
# This run keeps the stable sentence-window verifier and answer-preserving
# renderer, but turns on a small, gated answer-completion expansion. It is meant
# to test whether we can improve task correctness without losing the large
# claim-level support gains from the sentence-window version.
#
# Launch:
#   nohup bash scripts/run_cover_v3_asqa_eli5_correctness_boost_nohup.sh \
#     > logs/cover_v3_asqa_eli5_correctness_boost_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

export BASE_DIR="${BASE_DIR:-result/cover_v3_asqa_eli5_correctness_boost}"

# Keep the version that preserved correctness in the full sentence-window run.
export FINAL_AUDIT="${FINAL_AUDIT:-1}"
export RENDER_MODE="${RENDER_MODE:-answer_preserving}"
export NLI_PREMISE_MODE="${NLI_PREMISE_MODE:-sentence_window}"
export NLI_TOP_SENTENCES="${NLI_TOP_SENTENCES:-4}"
export NLI_SENTENCE_WINDOW_SIZE="${NLI_SENTENCE_WINDOW_SIZE:-1}"
export PRESERVE_ANSWER_SENTENCES="${PRESERVE_ANSWER_SENTENCES:-1}"
export PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES="${PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES:-0}"

# Turn on only a very small amount of expansion. The goal is to add missing
# answer strings, not to make the answer longer or more encyclopedic.
export EXPANSION_MODE="${EXPANSION_MODE:-llm}"
export EXPANSION_FILTER_GENERAL_ANSWER_SPANS="${EXPANSION_FILTER_GENERAL_ANSWER_SPANS:-1}"
export EXPANSION_MIN_QUESTION_OVERLAP="${EXPANSION_MIN_QUESTION_OVERLAP:-0.08}"
export ANSWER_TARGETED_EXPANSION="${ANSWER_TARGETED_EXPANSION:-1}"
export EXPANSION_EVIDENCE_SENTENCES="${EXPANSION_EVIDENCE_SENTENCES:-10}"
export EXPANSION_CANDIDATE_SPANS="${EXPANSION_CANDIDATE_SPANS:-24}"
export MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-10}"
export MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-6}"
export MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-2}"
export MIN_EXPANSION_SCORE="${MIN_EXPANSION_SCORE:-0.10}"

# Leave the original answer mostly intact. Expansion gets only a little extra
# room, so correctness can improve through missing nuggets without flooding the
# answer with background facts.
export MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-22}"
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-30}"
export MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-2}"
export MAX_CITATIONS_PER_SENTENCE="${MAX_CITATIONS_PER_SENTENCE:-3}"

# Resume-friendly defaults.
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
export CACHE_SAVE_EVERY="${CACHE_SAVE_EVERY:-50}"
export REVISION_WORKERS="${REVISION_WORKERS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_cover_v3_asqa_eli5_nohup.sh"
