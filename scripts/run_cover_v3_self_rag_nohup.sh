#!/usr/bin/env bash
# Run COVER-RAG v3 as a plug-in on Self-RAG outputs.
#
# This script does not train or run the Self-RAG model. It assumes you already
# have Self-RAG generations from methods/self-rag, converts them to ALCE result
# JSON, then runs the same frozen v13 COVER-RAG configuration used for the
# vanilla RAG main experiment.
#
# Required environment variables:
#   SELF_RAG_ASQA_OUTPUT=/path/to/self_rag_asqa.json_or_jsonl
#   SELF_RAG_ELI5_OUTPUT=/path/to/self_rag_eli5.json_or_jsonl
#   SELF_RAG_QAMPARI_OUTPUT=/path/to/self_rag_qampari.json_or_jsonl
#
# You can run any subset of the three. At least one SELF_RAG_*_OUTPUT must
# point to an existing file.
#
# Launch example:
#   nohup bash scripts/run_cover_v3_self_rag_nohup.sh \
#     > logs/cover_v3_self_rag_v13_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export PYTHON="${PYTHON:-/home/anaconda/envs/alce/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cover_v3_self_rag_v13}"
export BASE_DIR="${BASE_DIR:-result/${EXPERIMENT_NAME}_full}"
RAW_DIR="${RAW_DIR:-$BASE_DIR/self_rag_origin}"
mkdir -p "$BASE_DIR" "$RAW_DIR" logs

# Reference files provide ALCE gold/task metadata. The adapter replaces only
# output/docs with Self-RAG generations.
ASQA_REFERENCE="${ASQA_REFERENCE:-result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
ELI5_REFERENCE="${ELI5_REFERENCE:-result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json}"
QAMPARI_REFERENCE="${QAMPARI_REFERENCE:-result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"

ASQA_CONVERTED="$RAW_DIR/asqa-self-rag.json"
ELI5_CONVERTED="$RAW_DIR/eli5-self-rag.json"
QAMPARI_CONVERTED="$RAW_DIR/qampari-self-rag.json"

convert_one() {
  local dataset="$1"
  local source="$2"
  local reference="$3"
  local output="$4"
  if [ -z "$source" ]; then
    return 0
  fi
  if [ ! -f "$source" ]; then
    echo "WARNING: $dataset Self-RAG output not found, skip: $source"
    return 0
  fi
  if [ ! -f "$reference" ]; then
    echo "WARNING: $dataset reference ALCE result not found, skip: $reference"
    return 0
  fi
  echo
  echo "=== Convert Self-RAG $dataset output ==="
  echo "+ $PYTHON scripts/convert_self_rag_to_alce.py --self-rag-output $source --reference $reference --output $output --dataset $dataset --pretty"
  "$PYTHON" scripts/convert_self_rag_to_alce.py \
    --self-rag-output "$source" \
    --reference "$reference" \
    --output "$output" \
    --dataset "$dataset" \
    --pretty
}

convert_one "asqa" "${SELF_RAG_ASQA_OUTPUT:-}" "$ASQA_REFERENCE" "$ASQA_CONVERTED"
convert_one "eli5" "${SELF_RAG_ELI5_OUTPUT:-}" "$ELI5_REFERENCE" "$ELI5_CONVERTED"
convert_one "qampari" "${SELF_RAG_QAMPARI_OUTPUT:-}" "$QAMPARI_REFERENCE" "$QAMPARI_CONVERTED"

present=()
[ -f "$ASQA_CONVERTED" ] && present+=("$ASQA_CONVERTED")
[ -f "$ELI5_CONVERTED" ] && present+=("$ELI5_CONVERTED")
[ -f "$QAMPARI_CONVERTED" ] && present+=("$QAMPARI_CONVERTED")
if [ "${#present[@]}" -eq 0 ]; then
  echo "No converted Self-RAG ALCE files found."
  echo "Set at least one of SELF_RAG_ASQA_OUTPUT, SELF_RAG_ELI5_OUTPUT, SELF_RAG_QAMPARI_OUTPUT."
  exit 1
fi

export ASQA_RAW="$ASQA_CONVERTED"
export ELI5_RAW="$ELI5_CONVERTED"
export QAMPARI_RAW="$QAMPARI_CONVERTED"

# Use the same post-hoc candidate evidence as the frozen vanilla v13 run. The
# converted Self-RAG docs remain in the raw file for v0 citation evaluation.
export ASQA_DOCS="${ASQA_DOCS:-data/asqa_eval_gtr_top100_bge_reranked.json}"
export ELI5_DOCS="${ELI5_DOCS:-data/eli5_eval_bm25_top100_bge_reranked.json}"
export QAMPARI_DOCS="${QAMPARI_DOCS:-data/qampari_eval_gtr_top100_bge_reranked.json}"

# Self-RAG is a different base method, so rebuild v0 audit from the converted
# Self-RAG outputs rather than reusing vanilla RAG v0.
export PREPARE_V0="${SELF_RAG_PREPARE_V0:-1}"
# Self-RAG is a different base method. Do not accidentally inherit a vanilla
# RAG V0 summary from the shell; reuse only when explicitly requested.
export V0_SUMMARY_CSV="${SELF_RAG_V0_SUMMARY_CSV:-}"
export V0_DIR="${V0_DIR:-$BASE_DIR/cover_audit}"
export V0_CACHE_FILE="${V0_CACHE_FILE:-cache/cover_audit_self_rag_v13_cache.json}"
export V0_SCORE_MODE="${V0_SCORE_MODE:-eval}"
export FORCE_V0="${FORCE_V0:-0}"

# Frozen v13 COVER-RAG plug-in settings.
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-self_rag_v13_}"
export FINAL_AUDIT="${FINAL_AUDIT:-1}"
export RENDER_MODE="${RENDER_MODE:-answer_preserving}"
export NLI_PREMISE_MODE="${NLI_PREMISE_MODE:-sentence_window}"
export NLI_TOP_SENTENCES="${NLI_TOP_SENTENCES:-4}"
export NLI_SENTENCE_WINDOW_SIZE="${NLI_SENTENCE_WINDOW_SIZE:-1}"
export PRESERVE_ANSWER_SENTENCES="${PRESERVE_ANSWER_SENTENCES:-1}"
export PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES="${PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES:-0}"
export CITATION_POLICY="${CITATION_POLICY:-minimal}"
export SENTENCE_CITATION_SOURCE="${SENTENCE_CITATION_SOURCE:-hybrid}"
export MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-2}"
export MAX_CITATIONS_PER_SENTENCE="${MAX_CITATIONS_PER_SENTENCE:-4}"
export EXPANSION_MODE="${EXPANSION_MODE:-llm}"
export EXPANSION_MAX_TOKENS="${EXPANSION_MAX_TOKENS:-240}"
export EXPANSION_OUTPUT_MODE="${EXPANSION_OUTPUT_MODE:-span_ids}"
export EXPANSION_EVIDENCE_SENTENCES="${EXPANSION_EVIDENCE_SENTENCES:-18}"
export EXPANSION_CANDIDATE_SPANS="${EXPANSION_CANDIDATE_SPANS:-56}"
export MAX_SELECTED_SPAN_IDS="${MAX_SELECTED_SPAN_IDS:-8}"
export EXPANSION_QUERY_SOURCE="${EXPANSION_QUERY_SOURCE:-question_evidence_and_rejected}"
export EXPANSION_USE_INTERMEDIATE_ANSWER="${EXPANSION_USE_INTERMEDIATE_ANSWER:-1}"
export ANSWER_TARGETED_EXPANSION="${ANSWER_TARGETED_EXPANSION:-1}"
export RECALL_ORIENTED_COMPLETION="${RECALL_ORIENTED_COMPLETION:-1}"
export RECALL_COMPLETION_MAX_TARGETS="${RECALL_COMPLETION_MAX_TARGETS:-12}"
export RECALL_COMPLETION_REJECTED_TARGETS="${RECALL_COMPLETION_REJECTED_TARGETS:-5}"
export RECALL_COMPLETION_EVIDENCE_SENTENCES="${RECALL_COMPLETION_EVIDENCE_SENTENCES:-30}"
export RECALL_COMPLETION_SUPPORTED_CLAIMS="${RECALL_COMPLETION_SUPPORTED_CLAIMS:-20}"
export RECALL_COMPLETION_MIN_SENTENCE_OVERLAP="${RECALL_COMPLETION_MIN_SENTENCE_OVERLAP:-0.0}"
export RECALL_COMPLETION_MIN_TARGET_OVERLAP="${RECALL_COMPLETION_MIN_TARGET_OVERLAP:-0.08}"
export EXPANSION_FILTER_GENERAL_ANSWER_SPANS="${EXPANSION_FILTER_GENERAL_ANSWER_SPANS:-1}"
export EXPANSION_INCLUDE_EXTRACTIVE="${EXPANSION_INCLUDE_EXTRACTIVE:-1}"
export EXPANSION_MIN_QUESTION_OVERLAP="${EXPANSION_MIN_QUESTION_OVERLAP:-0.08}"
export ANSWER_TARGETED_EXTRACTIVE_TOP_K="${ANSWER_TARGETED_EXTRACTIVE_TOP_K:-3}"
export ANSWER_TARGETED_MIN_SENTENCE_OVERLAP="${ANSWER_TARGETED_MIN_SENTENCE_OVERLAP:-0.08}"
export EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN="${EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN:-1}"
export EXPLANATORY_TITLE_SPAN_MODE="${EXPLANATORY_TITLE_SPAN_MODE:-off}"
export MAX_EXPLANATORY_EXPANDED_CLAIMS="${MAX_EXPLANATORY_EXPANDED_CLAIMS:-1}"
export MAX_EXPLANATORY_EXPANSION_VERIFICATIONS="${MAX_EXPLANATORY_EXPANSION_VERIFICATIONS:-3}"
export MIN_EXPLANATORY_ANSWER_UNIT_SCORE="${MIN_EXPLANATORY_ANSWER_UNIT_SCORE:-0.42}"
export EXPLANATORY_PRECISION_GATE="${EXPLANATORY_PRECISION_GATE:-1}"
export EXPLANATORY_PRECISION_MIN_TARGET_RELEVANCE="${EXPLANATORY_PRECISION_MIN_TARGET_RELEVANCE:-0.58}"
export EXPLANATORY_PRECISION_MIN_QUESTION_RELEVANCE="${EXPLANATORY_PRECISION_MIN_QUESTION_RELEVANCE:-0.18}"
export EXPLANATION_INTEGRATED_REFINEMENT="${EXPLANATION_INTEGRATED_REFINEMENT:-1}"
export MAX_EXPLANATION_INTEGRATED_CLAIMS="${MAX_EXPLANATION_INTEGRATED_CLAIMS:-1}"
export MAX_EXPLANATION_LENGTH_GROWTH="${MAX_EXPLANATION_LENGTH_GROWTH:-0.10}"
export CANDIDATE_OUTPUT_SELECTION="${CANDIDATE_OUTPUT_SELECTION:-0}"
export MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-12}"
export MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-8}"
export MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-4}"
export MIN_EXPANSION_SCORE="${MIN_EXPANSION_SCORE:-0.06}"
export STRICT_EXPANSION_ANSWER_UNIT_GATE="${STRICT_EXPANSION_ANSWER_UNIT_GATE:-1}"
export STRICT_EXPANSION_MIN_TARGET_RELEVANCE="${STRICT_EXPANSION_MIN_TARGET_RELEVANCE:-0.45}"
export ANSWER_TYPE_GATE_MODE="${ANSWER_TYPE_GATE_MODE:-soft}"
export MAX_EXPANSION_CLAIM_WORDS="${MAX_EXPANSION_CLAIM_WORDS:-22}"
export MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-26}"
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-36}"

# QAMPARI v13 selector settings.
export QAMPARI_USE_SELECT="${QAMPARI_USE_SELECT:-1}"
export QAMPARI_SELECT_JOINT_DOCS="${QAMPARI_SELECT_JOINT_DOCS:-12}"
export QAMPARI_SELECT_EXPANSION_DOCS="${QAMPARI_SELECT_EXPANSION_DOCS:-12}"
export QAMPARI_SELECT_JOINT_CANDIDATE_SPANS="${QAMPARI_SELECT_JOINT_CANDIDATE_SPANS:-140}"
export QAMPARI_SELECT_MAX_OUTPUT_ITEMS="${QAMPARI_SELECT_MAX_OUTPUT_ITEMS:-15}"
export QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K="${QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K:-8}"
export QAMPARI_SELECT_PRESERVED_ORIGINAL_LABEL="${QAMPARI_SELECT_PRESERVED_ORIGINAL_LABEL:-supported}"
export QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL="${QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL:-15}"
export QAMPARI_SELECT_BACKFILL_REQUIRES_TYPE_SAFE="${QAMPARI_SELECT_BACKFILL_REQUIRES_TYPE_SAFE:-1}"
export QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL="${QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL:-1}"
export QAMPARI_SELECT_RESTORE_ORIGINAL_SUBSPANS="${QAMPARI_SELECT_RESTORE_ORIGINAL_SUBSPANS:-1}"
export QAMPARI_SELECT_DROP_OBVIOUS_BAD="${QAMPARI_SELECT_DROP_OBVIOUS_BAD:-1}"
export QAMPARI_SELECT_EMPTY_OUTPUT_POLICY="${QAMPARI_SELECT_EMPTY_OUTPUT_POLICY:-empty}"
export QAMPARI_SELECT_MAX_EXPANSION_CANDIDATES="${QAMPARI_SELECT_MAX_EXPANSION_CANDIDATES:-12}"
export QAMPARI_SELECT_MAX_CITATIONS_PER_ITEM="${QAMPARI_SELECT_MAX_CITATIONS_PER_ITEM:-1}"

export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
export CACHE_SAVE_EVERY="${CACHE_SAVE_EVERY:-50}"
export REVISION_WORKERS="${REVISION_WORKERS:-2}"

echo
echo "=== COVER-RAG on Self-RAG outputs ==="
echo "Base dir: $BASE_DIR"
echo "Converted inputs:"
printf '  - %s\n' "${present[@]}"
echo "Self-RAG source tree: methods/self-rag"
echo "Candidate docs:"
printf '  - %s\n' "$ASQA_DOCS" "$ELI5_DOCS" "$QAMPARI_DOCS"

bash scripts/run_cover_v3_three_datasets_nohup.sh

echo
echo "=== Self-RAG COVER-RAG outputs ==="
echo "Converted raw inputs: $RAW_DIR"
echo "v0 summary: $BASE_DIR/cover_audit/cover_audit_comparison.csv"
echo "v3 summary: $BASE_DIR/cover_v3/cover_v3_comparison.csv"
echo "required metrics: $BASE_DIR/required_metrics_summary.csv"
echo "Done."
