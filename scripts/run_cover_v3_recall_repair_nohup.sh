#!/usr/bin/env bash
# Recall-repair COVER-RAG v3 experiment.
#
# Motivation:
#   Oracle diagnostics show that main-task gains mostly come from retrieving
#   answer-bearing evidence, while the current v3 mainly improves support for
#   already-written claims. This run tests a plug-in style recall repair:
#   use a stronger candidate-doc order, ask expansion to add more missing answer
#   units, and make QAMPARI less deletion-heavy.
#
# Launch:
#   nohup bash scripts/run_cover_v3_recall_repair_nohup.sh \
#     > logs/cover_v3_recall_repair_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export PYTHON="${PYTHON:-/home/anaconda/envs/alce/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cover_v3_recall_completion_v14_candidate_select}"
DEFAULT_BASE_DIR="result/${EXPERIMENT_NAME}_full"
DEFAULT_CACHE_NAMESPACE="recall_completion_v14_"
if [[ "${USE_EXTERNAL_BASE_DIR:-0}" == "1" && -n "${BASE_DIR:-}" ]]; then
  export BASE_DIR
else
  export BASE_DIR="$DEFAULT_BASE_DIR"
fi
if [[ "${USE_EXTERNAL_CACHE_NAMESPACE:-0}" == "1" && -n "${CACHE_NAMESPACE:-}" ]]; then
  export CACHE_NAMESPACE
else
  export CACHE_NAMESPACE="$DEFAULT_CACHE_NAMESPACE"
fi
export V0_CACHE_FILE="${V0_CACHE_FILE:-cache/cover_audit_recall_repair_cache.json}"
# Recall-repair keeps the raw v0 generations unchanged, so the v0 audit metrics
# should usually be reused from the stable API run instead of rebuilt. Rebuilding
# v0 is slow and can be noisy when the upstream LLM filters/overloads.
export PREPARE_V0="${PREPARE_V0:-0}"
export V0_SUMMARY_CSV="${V0_SUMMARY_CSV:-result/cover_v3_main_task_boost_v6_api_full/cover_audit/cover_audit_comparison.csv}"

# Keep base raw answers fixed; use reranked top100 only as post-hoc candidate
# evidence. This preserves the plug-in setting and avoids oracle leakage.
export ASQA_RAW="${ASQA_RAW:-result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
export ELI5_RAW="${ELI5_RAW:-result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json}"
export QAMPARI_RAW="${QAMPARI_RAW:-result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
export ASQA_DOCS="${ASQA_DOCS:-data/asqa_eval_gtr_top100_bge_reranked.json}"
export ELI5_DOCS="${ELI5_DOCS:-data/eli5_eval_bm25_top100_bge_reranked.json}"
export QAMPARI_DOCS="${QAMPARI_DOCS:-data/qampari_eval_gtr_top100_bge_reranked.json}"
export MAX_DOC_POOL="${MAX_DOC_POOL:-100}"
export DYNAMIC_RETRIEVAL_MODE="${DYNAMIC_RETRIEVAL_MODE:-off}"
export DYNAMIC_RETRIEVAL_TOP_K="${DYNAMIC_RETRIEVAL_TOP_K:-12}"
export DYNAMIC_RETRIEVAL_PER_QUERY_DOCS="${DYNAMIC_RETRIEVAL_PER_QUERY_DOCS:-4}"
export DYNAMIC_RETRIEVAL_MAX_QUERIES="${DYNAMIC_RETRIEVAL_MAX_QUERIES:-4}"
export DYNAMIC_RETRIEVAL_MAX_REJECTED_CLAIMS="${DYNAMIC_RETRIEVAL_MAX_REJECTED_CLAIMS:-4}"
export DYNAMIC_RETRIEVAL_MAX_DOC_POOL="${DYNAMIC_RETRIEVAL_MAX_DOC_POOL:-140}"
export DYNAMIC_RETRIEVAL_QUERY_MAX_CHARS="${DYNAMIC_RETRIEVAL_QUERY_MAX_CHARS:-900}"
export DYNAMIC_RETRIEVAL_MIN_QUERY_TERMS="${DYNAMIC_RETRIEVAL_MIN_QUERY_TERMS:-2}"
export DYNAMIC_RETRIEVAL_SENTENCE_BOOST="${DYNAMIC_RETRIEVAL_SENTENCE_BOOST:-0.12}"
export DYNAMIC_RETRIEVAL_QUERY_STRATEGY="${DYNAMIC_RETRIEVAL_QUERY_STRATEGY:-missing_facet}"
export DYNAMIC_RETRIEVAL_MIN_SCORE="${DYNAMIC_RETRIEVAL_MIN_SCORE:-0.0}"
export DYNAMIC_RETRIEVAL_REQUIRE_MISSING_SIGNAL="${DYNAMIC_RETRIEVAL_REQUIRE_MISSING_SIGNAL:-1}"
export DYNAMIC_RETRIEVAL_USE_SUPPORTED_ANSWER="${DYNAMIC_RETRIEVAL_USE_SUPPORTED_ANSWER:-1}"
export COVERAGE_RERANKER_MODEL="${COVERAGE_RERANKER_MODEL:-}"
export COVERAGE_RERANKER_DEVICE="${COVERAGE_RERANKER_DEVICE:-}"
export COVERAGE_RERANKER_BATCH_SIZE="${COVERAGE_RERANKER_BATCH_SIZE:-16}"
export COVERAGE_RERANKER_MAX_LENGTH="${COVERAGE_RERANKER_MAX_LENGTH:-384}"
export COVERAGE_RERANKER_DOC_PREFILTER="${COVERAGE_RERANKER_DOC_PREFILTER:-64}"
export COVERAGE_RERANKER_DOC_WEIGHT="${COVERAGE_RERANKER_DOC_WEIGHT:-0.35}"
export COVERAGE_RERANKER_CANDIDATE_WEIGHT="${COVERAGE_RERANKER_CANDIDATE_WEIGHT:-0.20}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DISABLE_PROXY="${DISABLE_PROXY:-1}"
if [[ "$DISABLE_PROXY" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

# Shared support-preserving settings from v8.
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

# Recall repair: more answer-bearing evidence and more verification budget.
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
export CANDIDATE_OUTPUT_SELECTION="${CANDIDATE_OUTPUT_SELECTION:-1}"
export CANDIDATE_OUTPUT_RECALL_CLAIMS="${CANDIDATE_OUTPUT_RECALL_CLAIMS:-2}"
export CANDIDATE_OUTPUT_RECALL_GROWTH="${CANDIDATE_OUTPUT_RECALL_GROWTH:-0.18}"
export CANDIDATE_OUTPUT_MIN_SUPPORT="${CANDIDATE_OUTPUT_MIN_SUPPORT:-0.94}"
export CANDIDATE_OUTPUT_SUPPORT_TOLERANCE="${CANDIDATE_OUTPUT_SUPPORT_TOLERANCE:-0.015}"
export CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE="${CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE:-0.06}"
export CANDIDATE_OUTPUT_MIN_ANSWER_UNIT_GAIN="${CANDIDATE_OUTPUT_MIN_ANSWER_UNIT_GAIN:-0.0}"
export CANDIDATE_OUTPUT_MIN_SUPPORTED_CLAIM_GAIN="${CANDIDATE_OUTPUT_MIN_SUPPORTED_CLAIM_GAIN:-0}"
export MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-12}"
export MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-8}"
export MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-4}"
export MIN_EXPANSION_SCORE="${MIN_EXPANSION_SCORE:-0.06}"
export STRICT_EXPANSION_ANSWER_UNIT_GATE="${STRICT_EXPANSION_ANSWER_UNIT_GATE:-1}"
export STRICT_EXPANSION_MIN_TARGET_RELEVANCE="${STRICT_EXPANSION_MIN_TARGET_RELEVANCE:-0.45}"
export ANSWER_TYPE_GATE_MODE="${ANSWER_TYPE_GATE_MODE:-soft}"
export MAX_EXPANSION_CLAIM_WORDS="${MAX_EXPANSION_CLAIM_WORDS:-22}"
export VERIFY_EXPANSION_WITH_SENTENCE="${VERIFY_EXPANSION_WITH_SENTENCE:-1}"
export REQUIRE_EXPANSION_ANSWER_SPAN="${REQUIRE_EXPANSION_ANSWER_SPAN:-1}"
export SKIP_COVERED_ANSWER_SPANS="${SKIP_COVERED_ANSWER_SPANS:-1}"
export MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-26}"
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-36}"

# QAMPARI is recall-sensitive: deleting original items can remove true gold
# answers. Preserve a small type-safe prefix before adding expanded items, then
# backfill remaining original items up to the output budget.
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
export REVISION_WORKERS="${REVISION_WORKERS:-8}"

mkdir -p "$BASE_DIR" logs

echo "=== COVER-RAG v3 recall-repair run ==="
echo "Base dir: $BASE_DIR"
echo "Candidate docs:"
printf '  - %s\n' "$ASQA_DOCS" "$ELI5_DOCS" "$QAMPARI_DOCS"
echo "V0 summary: ${V0_SUMMARY_CSV:-<rebuild>}; prepare_v0=$PREPARE_V0"
echo "Expansion output: mode=$EXPANSION_OUTPUT_MODE candidate_spans=$EXPANSION_CANDIDATE_SPANS span_id_cap=$MAX_SELECTED_SPAN_IDS strict_gate=$STRICT_EXPANSION_ANSWER_UNIT_GATE min_target_rel=$STRICT_EXPANSION_MIN_TARGET_RELEVANCE answer_type_gate=$ANSWER_TYPE_GATE_MODE explanatory_title=$EXPLANATORY_TITLE_SPAN_MODE explanatory_kept=$MAX_EXPLANATORY_EXPANDED_CLAIMS explanatory_verify=$MAX_EXPLANATORY_EXPANSION_VERIFICATIONS explanatory_min_unit=$MIN_EXPLANATORY_ANSWER_UNIT_SCORE explanatory_precision=$EXPLANATORY_PRECISION_GATE explanation_integrated=$EXPLANATION_INTEGRATED_REFINEMENT explanation_integrated_claims=$MAX_EXPLANATION_INTEGRATED_CLAIMS explanation_growth=$MAX_EXPLANATION_LENGTH_GROWTH candidate_selection=$CANDIDATE_OUTPUT_SELECTION candidate_recall_claims=$CANDIDATE_OUTPUT_RECALL_CLAIMS candidate_recall_growth=$CANDIDATE_OUTPUT_RECALL_GROWTH candidate_min_unit_gain=$CANDIDATE_OUTPUT_MIN_ANSWER_UNIT_GAIN candidate_min_supported_gain=$CANDIDATE_OUTPUT_MIN_SUPPORTED_CLAIM_GAIN"
echo "Dynamic retrieval: mode=$DYNAMIC_RETRIEVAL_MODE strategy=$DYNAMIC_RETRIEVAL_QUERY_STRATEGY coverage_guided=${COVERAGE_GUIDED_RETRIEVAL:-0} top_k=$DYNAMIC_RETRIEVAL_TOP_K per_query=$DYNAMIC_RETRIEVAL_PER_QUERY_DOCS queries=$DYNAMIC_RETRIEVAL_MAX_QUERIES min_score=$DYNAMIC_RETRIEVAL_MIN_SCORE max_doc_pool=$DYNAMIC_RETRIEVAL_MAX_DOC_POOL"
echo "Coverage-aware reranker: ${COVERAGE_RERANKER_MODEL:-off} device=${COVERAGE_RERANKER_DEVICE:-auto} batch=$COVERAGE_RERANKER_BATCH_SIZE doc_weight=$COVERAGE_RERANKER_DOC_WEIGHT candidate_weight=$COVERAGE_RERANKER_CANDIDATE_WEIGHT"
echo "Revision workers: $REVISION_WORKERS"
echo "QAMPARI recall mode: keep_unsupported_original=$QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL preserve_top_k=$QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K preserve_label=$QAMPARI_SELECT_PRESERVED_ORIGINAL_LABEL backfill_until=$QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL"

bash scripts/run_cover_v3_three_datasets_nohup.sh

echo
echo "=== Answer recall diagnostics for recall-repair run ==="
"$PYTHON" scripts/compute_answer_recall_diagnostics.py \
  --output-dir "$BASE_DIR/answer_recall" \
  --condition "asqa:base_raw:$ASQA_RAW" \
  --condition "asqa:recall_repair:$BASE_DIR/cover_v3/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json" \
  --condition "eli5:base_raw:$ELI5_RAW" \
  --condition "eli5:recall_repair:$BASE_DIR/cover_v3/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.cover_v3.json" \
  --condition "qampari:base_raw:$QAMPARI_RAW" \
  --condition "qampari:recall_repair:$BASE_DIR/cover_v3/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json" \
  --pair "asqa:base_raw:recall_repair" \
  --pair "eli5:base_raw:recall_repair" \
  --pair "qampari:base_raw:recall_repair"

echo
echo "Recall summary: $BASE_DIR/answer_recall/answer_recall_summary.csv"
echo "Recall deltas: $BASE_DIR/answer_recall/answer_recall_pair_deltas.csv"

echo
echo "=== Fixed required metrics table ==="
"$PYTHON" scripts/summarize_cover_required_metrics.py \
  --v0-summary-csv "$V0_SUMMARY_CSV" \
  --latest-summary-csv "$BASE_DIR/cover_v3/cover_v3_comparison.csv" \
  --latest-label "recall_repair" \
  --answer-recall-summary "$BASE_DIR/answer_recall/answer_recall_summary.csv" \
  --output-csv "$BASE_DIR/required_metrics_summary.csv" \
  --output-md "$BASE_DIR/required_metrics_summary.md" \
  --condition "asqa:base_raw:$ASQA_RAW" \
  --condition "asqa:recall_repair:$BASE_DIR/cover_v3/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json" \
  --condition "eli5:base_raw:$ELI5_RAW" \
  --condition "eli5:recall_repair:$BASE_DIR/cover_v3/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.cover_v3.json" \
  --condition "qampari:base_raw:$QAMPARI_RAW" \
  --condition "qampari:recall_repair:$BASE_DIR/cover_v3/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json"

echo "Required metrics: $BASE_DIR/required_metrics_summary.csv"
echo "Done."
