#!/usr/bin/env bash
# Rerun only ELI5 for the conservative correctness-boost COVER-RAG v3 setting.
#
# This script does not touch ASQA outputs. It writes ELI5-only summaries under
# BASE_DIR so a completed ASQA run in another directory remains intact.
#
# Launch:
#   FORCE_V3=1 FORCE_EVAL=1 FORCE_MERGE=1 \
#   nohup bash scripts/run_cover_v3_eli5_correctness_boost_nohup.sh \
#     > logs/cover_v3_eli5_correctness_boost_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON="${PYTHON:-python}"
BASE_DIR="${BASE_DIR:-result/cover_v3_eli5_correctness_boost_full}"
V3_DIR="${V3_DIR:-$BASE_DIR/cover_v3}"

ELI5_RAW="${ELI5_RAW:-result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json}"
ELI5_DOCS="${ELI5_DOCS:-data/eli5_eval_bm25_top100.json}"

DECOMPOSER="${DECOMPOSER:-llm}"
DECOMPOSE_MODEL="${DECOMPOSE_MODEL:-gpt-4o-mini}"
VERIFIER="${VERIFIER:-nli}"
LLM_VERIFY_MODEL="${LLM_VERIFY_MODEL:-gpt-4o-mini}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"
NLI_PREMISE_MODE="${NLI_PREMISE_MODE:-sentence_window}"
NLI_TOP_SENTENCES="${NLI_TOP_SENTENCES:-4}"
NLI_SENTENCE_WINDOW_SIZE="${NLI_SENTENCE_WINDOW_SIZE:-1}"

EXPANSION_MODEL="${EXPANSION_MODEL:-gpt-4o-mini}"
EXPANSION_EVIDENCE_SENTENCES="${EXPANSION_EVIDENCE_SENTENCES:-10}"
EXPANSION_CANDIDATE_SPANS="${EXPANSION_CANDIDATE_SPANS:-24}"
MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-6}"
MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-4}"
MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-2}"
MIN_EXPANSION_SCORE="${MIN_EXPANSION_SCORE:-0.10}"
EXPANSION_MIN_QUESTION_OVERLAP="${EXPANSION_MIN_QUESTION_OVERLAP:-0.08}"
ANSWER_TARGETED_EXPANSION="${ANSWER_TARGETED_EXPANSION:-1}"

MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-22}"
MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-30}"
MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-2}"
MAX_CITATIONS_PER_SENTENCE="${MAX_CITATIONS_PER_SENTENCE:-3}"
REVISION_WORKERS="${REVISION_WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
CACHE_SAVE_EVERY="${CACHE_SAVE_EVERY:-50}"

FORCE_V3="${FORCE_V3:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
FINAL_AUDIT="${FINAL_AUDIT:-1}"
LIMIT="${LIMIT:-}"

mkdir -p "$BASE_DIR" "$V3_DIR" logs

echo "=== COVER-RAG v3 ELI5 correctness-boost rerun ==="
echo "Root: $ROOT_DIR"
echo "Raw: $ELI5_RAW"
echo "Candidate docs: $ELI5_DOCS"
echo "Output dir: $V3_DIR"
echo "NLI premise: $NLI_PREMISE_MODE, top=$NLI_TOP_SENTENCES, window=$NLI_SENTENCE_WINDOW_SIZE"
echo "Expansion: llm with filtered answer spans, max expanded=$MAX_EXPANDED_CLAIMS"
echo "Limit: ${LIMIT:-<full dataset>}"

cmd=(
  "$PYTHON" scripts/run_cover_v3_batch.py
  --inputs "$ELI5_RAW"
  --output-dir "$V3_DIR"
  --python "$PYTHON"
  --decomposer "$DECOMPOSER"
  --decompose-model "$DECOMPOSE_MODEL"
  --verifier "$VERIFIER"
  --llm-verify-model "$LLM_VERIFY_MODEL"
  --nli-model "$NLI_MODEL"
  --nli-premise-mode "$NLI_PREMISE_MODE"
  --nli-top-sentences "$NLI_TOP_SENTENCES"
  --nli-sentence-window-size "$NLI_SENTENCE_WINDOW_SIZE"
  --evidence-scope cited
  --citation-policy source
  --max-citations-per-claim "$MAX_CITATIONS_PER_CLAIM"
  --max-citations-per-sentence "$MAX_CITATIONS_PER_SENTENCE"
  --max-verified-claims "$MAX_VERIFIED_CLAIMS"
  --max-final-claims "$MAX_FINAL_CLAIMS"
  --render-mode answer_preserving
  --max-doc-pool 100
  --expansion-mode llm
  --expansion-model "$EXPANSION_MODEL"
  --expansion-evidence-sentences "$EXPANSION_EVIDENCE_SENTENCES"
  --expansion-candidate-spans "$EXPANSION_CANDIDATE_SPANS"
  --max-expansion-candidates "$MAX_EXPANSION_CANDIDATES"
  --max-expansion-verifications "$MAX_EXPANSION_VERIFICATIONS"
  --max-expanded-claims "$MAX_EXPANDED_CLAIMS"
  --min-expansion-score "$MIN_EXPANSION_SCORE"
  --expansion-query-source question_and_evidence_core
  --answer-targeted-expansion
  --expansion-include-extractive
  --expansion-filter-general-answer-spans
  --expansion-min-question-overlap "$EXPANSION_MIN_QUESTION_OVERLAP"
  --verify-expansion-with-sentence
  --require-expansion-answer-span
  --preserve-first-pass-claims
  --preserve-answer-sentences
  --no-preserve-rejected-critical-answer-sentences
  --no-include-background-claims
  --skip-covered-answer-spans
  --candidate-docs-file "$ELI5_DOCS"
  --cache-file cache/cover_v3_eli5_bm25_cache.json
  --cache-save-every "$CACHE_SAVE_EVERY"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --revision-workers "$REVISION_WORKERS"
  --summary-csv "$V3_DIR/eli5_bm25_cover_v3_comparison.csv"
  --summary-md "$V3_DIR/eli5_bm25_cover_v3_comparison.md"
)

if [ "$FINAL_AUDIT" = "1" ]; then
  cmd+=(--final-audit)
else
  cmd+=(--no-final-audit)
fi
if [ "$FORCE_V3" = "1" ]; then
  cmd+=(--force-revise)
fi
if [ "$FORCE_EVAL" = "1" ]; then
  cmd+=(--force-eval)
fi
if [ "$FORCE_MERGE" = "1" ]; then
  cmd+=(--force-merge)
fi
if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"
status=$?
if [ "$status" -ne 0 ]; then
  echo "WARNING: ELI5 rerun failed with exit code $status."
fi
exit 0
