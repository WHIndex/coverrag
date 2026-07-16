#!/usr/bin/env bash
# Run COVER-RAG v3 on the three ALCE datasets used in the current experiment:
# ASQA + GTR, ELI5 + BM25, and QAMPARI + GTR.
#
# Usage:
#   nohup bash scripts/run_cover_v3_three_datasets_nohup.sh > logs/cover_v3_three.log 2>&1 &
#
# Useful overrides:
#   FORCE_V3=1 FINAL_AUDIT=1 bash scripts/run_cover_v3_three_datasets_nohup.sh
#   PREPARE_V0=0 V0_SUMMARY_CSV=/path/to/cover_audit_comparison.csv bash ...

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON="${PYTHON:-python}"
BASE_DIR="${BASE_DIR:-result/cover_v3_three_datasets}"
V0_DIR="${V0_DIR:-$BASE_DIR/cover_audit}"
V3_DIR="${V3_DIR:-$BASE_DIR/cover_v3}"
LOG_PREFIX="${LOG_PREFIX:-cover_v3_three}"
OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"
export OPENAI_API_BASE OPENAI_BASE_URL

# Prefer local cached models and avoid stale local proxy settings during batch
# runs. Set DISABLE_PROXY=0 if the current network really requires a proxy.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
DISABLE_PROXY="${DISABLE_PROXY:-1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TOKENIZERS_PARALLELISM DISABLE_PROXY
if [[ "$DISABLE_PROXY" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

ASQA_RAW="${ASQA_RAW:-result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
ELI5_RAW="${ELI5_RAW:-result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json}"
QAMPARI_RAW="${QAMPARI_RAW:-result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
ASQA_DOCS="${ASQA_DOCS:-data/asqa_eval_gtr_top100.json}"
ELI5_DOCS="${ELI5_DOCS:-data/eli5_eval_bm25_top100.json}"
QAMPARI_DOCS="${QAMPARI_DOCS:-data/qampari_eval_gtr_top100.json}"
MAX_DOC_POOL="${MAX_DOC_POOL:-100}"
DYNAMIC_RETRIEVAL_MODE="${DYNAMIC_RETRIEVAL_MODE:-off}"
DYNAMIC_RETRIEVAL_TOP_K="${DYNAMIC_RETRIEVAL_TOP_K:-12}"
DYNAMIC_RETRIEVAL_PER_QUERY_DOCS="${DYNAMIC_RETRIEVAL_PER_QUERY_DOCS:-4}"
DYNAMIC_RETRIEVAL_MAX_QUERIES="${DYNAMIC_RETRIEVAL_MAX_QUERIES:-4}"
DYNAMIC_RETRIEVAL_MAX_REJECTED_CLAIMS="${DYNAMIC_RETRIEVAL_MAX_REJECTED_CLAIMS:-4}"
DYNAMIC_RETRIEVAL_MAX_DOC_POOL="${DYNAMIC_RETRIEVAL_MAX_DOC_POOL:-140}"
DYNAMIC_RETRIEVAL_QUERY_MAX_CHARS="${DYNAMIC_RETRIEVAL_QUERY_MAX_CHARS:-900}"
DYNAMIC_RETRIEVAL_MIN_QUERY_TERMS="${DYNAMIC_RETRIEVAL_MIN_QUERY_TERMS:-2}"
DYNAMIC_RETRIEVAL_SENTENCE_BOOST="${DYNAMIC_RETRIEVAL_SENTENCE_BOOST:-0.12}"
DYNAMIC_RETRIEVAL_QUERY_STRATEGY="${DYNAMIC_RETRIEVAL_QUERY_STRATEGY:-missing_facet}"
DYNAMIC_RETRIEVAL_MIN_SCORE="${DYNAMIC_RETRIEVAL_MIN_SCORE:-0.0}"
DYNAMIC_RETRIEVAL_REQUIRE_MISSING_SIGNAL="${DYNAMIC_RETRIEVAL_REQUIRE_MISSING_SIGNAL:-1}"
DYNAMIC_RETRIEVAL_USE_SUPPORTED_ANSWER="${DYNAMIC_RETRIEVAL_USE_SUPPORTED_ANSWER:-1}"
COVERAGE_GUIDED_RETRIEVAL="${COVERAGE_GUIDED_RETRIEVAL:-0}"
COVERAGE_GUIDED_TARGETS_PER_ROUND="${COVERAGE_GUIDED_TARGETS_PER_ROUND:-3}"
COVERAGE_GUIDED_MIN_TARGETS="${COVERAGE_GUIDED_MIN_TARGETS:-1}"
COVERAGE_GUIDED_MIN_QUESTION_OVERLAP="${COVERAGE_GUIDED_MIN_QUESTION_OVERLAP:-0.08}"
COVERAGE_GUIDED_QUERY_SUPPORTED_CLAIMS="${COVERAGE_GUIDED_QUERY_SUPPORTED_CLAIMS:-6}"
COVERAGE_GUIDED_ALLOW_EVIDENCE_TARGETS="${COVERAGE_GUIDED_ALLOW_EVIDENCE_TARGETS:-1}"
COVERAGE_RERANKER_MODEL="${COVERAGE_RERANKER_MODEL:-}"
COVERAGE_RERANKER_DEVICE="${COVERAGE_RERANKER_DEVICE:-}"
COVERAGE_RERANKER_BATCH_SIZE="${COVERAGE_RERANKER_BATCH_SIZE:-16}"
COVERAGE_RERANKER_MAX_LENGTH="${COVERAGE_RERANKER_MAX_LENGTH:-384}"
COVERAGE_RERANKER_DOC_PREFILTER="${COVERAGE_RERANKER_DOC_PREFILTER:-64}"
COVERAGE_RERANKER_DOC_WEIGHT="${COVERAGE_RERANKER_DOC_WEIGHT:-0.35}"
COVERAGE_RERANKER_CANDIDATE_WEIGHT="${COVERAGE_RERANKER_CANDIDATE_WEIGHT:-0.20}"

DECOMPOSER="${DECOMPOSER:-llm}"
DECOMPOSE_MODEL="${DECOMPOSE_MODEL:-gpt-4o-mini}"
VERIFIER="${VERIFIER:-nli}"
LLM_VERIFY_MODEL="${LLM_VERIFY_MODEL:-gpt-4o-mini}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"
NLI_PREMISE_MODE="${NLI_PREMISE_MODE:-sentence_window}"
NLI_TOP_SENTENCES="${NLI_TOP_SENTENCES:-4}"
NLI_SENTENCE_WINDOW_SIZE="${NLI_SENTENCE_WINDOW_SIZE:-1}"
ENTAIL_THRESHOLD="${ENTAIL_THRESHOLD:-0.50}"
CONTRADICTION_THRESHOLD="${CONTRADICTION_THRESHOLD:-0.50}"
AMBIGUOUS_MARGIN="${AMBIGUOUS_MARGIN:-0.10}"

PREPARE_V0="${PREPARE_V0:-1}"
V0_SUMMARY_CSV="${V0_SUMMARY_CSV:-}"
V0_CACHE_FILE="${V0_CACHE_FILE:-cache/cover_audit_cache.json}"
V0_SCORE_MODE="${V0_SCORE_MODE:-eval}"
FORCE_V0="${FORCE_V0:-0}"
FORCE_V3="${FORCE_V3:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

EXPANSION_MODE="${EXPANSION_MODE:-llm}"
EXPANSION_MODEL="${EXPANSION_MODEL:-gpt-4o-mini}"
EXPANSION_MAX_TOKENS="${EXPANSION_MAX_TOKENS:-700}"
EXPANSION_OUTPUT_MODE="${EXPANSION_OUTPUT_MODE:-claims}"
EXPANSION_INCLUDE_EXTRACTIVE="${EXPANSION_INCLUDE_EXTRACTIVE:-1}"
EXPANSION_EVIDENCE_SENTENCES="${EXPANSION_EVIDENCE_SENTENCES:-16}"
EXPANSION_CANDIDATE_SPANS="${EXPANSION_CANDIDATE_SPANS:-40}"
MAX_SELECTED_SPAN_IDS="${MAX_SELECTED_SPAN_IDS:-8}"
EXPANSION_FILTER_GENERAL_ANSWER_SPANS="${EXPANSION_FILTER_GENERAL_ANSWER_SPANS:-0}"
EXPANSION_MIN_QUESTION_OVERLAP="${EXPANSION_MIN_QUESTION_OVERLAP:-0.0}"
EXPANSION_QUERY_SOURCE="${EXPANSION_QUERY_SOURCE:-question_and_evidence_core}"
EXPANSION_USE_INTERMEDIATE_ANSWER="${EXPANSION_USE_INTERMEDIATE_ANSWER:-0}"
ANSWER_TARGETED_EXPANSION="${ANSWER_TARGETED_EXPANSION:-0}"
ANSWER_TARGETED_EXTRACTIVE_TOP_K="${ANSWER_TARGETED_EXTRACTIVE_TOP_K:-2}"
ANSWER_TARGETED_MIN_SENTENCE_OVERLAP="${ANSWER_TARGETED_MIN_SENTENCE_OVERLAP:-0.16}"
EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN="${EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN:-0}"
EXPLANATORY_TITLE_SPAN_MODE="${EXPLANATORY_TITLE_SPAN_MODE:-off}"
MAX_EXPLANATORY_EXPANDED_CLAIMS="${MAX_EXPLANATORY_EXPANDED_CLAIMS:-2}"
MAX_EXPLANATORY_EXPANSION_VERIFICATIONS="${MAX_EXPLANATORY_EXPANSION_VERIFICATIONS:-4}"
MIN_EXPLANATORY_ANSWER_UNIT_SCORE="${MIN_EXPLANATORY_ANSWER_UNIT_SCORE:-0.35}"
EXPLANATORY_PRECISION_GATE="${EXPLANATORY_PRECISION_GATE:-0}"
EXPLANATORY_PRECISION_MIN_TARGET_RELEVANCE="${EXPLANATORY_PRECISION_MIN_TARGET_RELEVANCE:-0.58}"
EXPLANATORY_PRECISION_MIN_QUESTION_RELEVANCE="${EXPLANATORY_PRECISION_MIN_QUESTION_RELEVANCE:-0.18}"
EXPLANATION_INTEGRATED_REFINEMENT="${EXPLANATION_INTEGRATED_REFINEMENT:-0}"
MAX_EXPLANATION_INTEGRATED_CLAIMS="${MAX_EXPLANATION_INTEGRATED_CLAIMS:-1}"
MAX_EXPLANATION_LENGTH_GROWTH="${MAX_EXPLANATION_LENGTH_GROWTH:-0.10}"
CANDIDATE_OUTPUT_SELECTION="${CANDIDATE_OUTPUT_SELECTION:-0}"
CANDIDATE_OUTPUT_RECALL_CLAIMS="${CANDIDATE_OUTPUT_RECALL_CLAIMS:-2}"
CANDIDATE_OUTPUT_RECALL_GROWTH="${CANDIDATE_OUTPUT_RECALL_GROWTH:-0.18}"
CANDIDATE_OUTPUT_MIN_SUPPORT="${CANDIDATE_OUTPUT_MIN_SUPPORT:-0.94}"
CANDIDATE_OUTPUT_SUPPORT_TOLERANCE="${CANDIDATE_OUTPUT_SUPPORT_TOLERANCE:-0.015}"
CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE="${CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE:-0.06}"
CANDIDATE_OUTPUT_MIN_ANSWER_UNIT_GAIN="${CANDIDATE_OUTPUT_MIN_ANSWER_UNIT_GAIN:-0.0}"
CANDIDATE_OUTPUT_MIN_SUPPORTED_CLAIM_GAIN="${CANDIDATE_OUTPUT_MIN_SUPPORTED_CLAIM_GAIN:-0}"
RECALL_ORIENTED_COMPLETION="${RECALL_ORIENTED_COMPLETION:-0}"
RECALL_COMPLETION_MAX_TARGETS="${RECALL_COMPLETION_MAX_TARGETS:-10}"
RECALL_COMPLETION_REJECTED_TARGETS="${RECALL_COMPLETION_REJECTED_TARGETS:-4}"
RECALL_COMPLETION_EVIDENCE_SENTENCES="${RECALL_COMPLETION_EVIDENCE_SENTENCES:-24}"
RECALL_COMPLETION_SUPPORTED_CLAIMS="${RECALL_COMPLETION_SUPPORTED_CLAIMS:-18}"
RECALL_COMPLETION_MIN_SENTENCE_OVERLAP="${RECALL_COMPLETION_MIN_SENTENCE_OVERLAP:-0.0}"
RECALL_COMPLETION_MIN_TARGET_OVERLAP="${RECALL_COMPLETION_MIN_TARGET_OVERLAP:-0.10}"
ANSWER_UNIT_RELEVANCE_GATE="${ANSWER_UNIT_RELEVANCE_GATE:-0}"
ANSWER_UNIT_MIN_DIRECTNESS="${ANSWER_UNIT_MIN_DIRECTNESS:-0.44}"
EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS="${EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS:-0.58}"
ITERATIVE_COMPLETION_ROUNDS="${ITERATIVE_COMPLETION_ROUNDS:-0}"
ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND="${ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND:-2}"
ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND="${ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND:-4}"
ITERATIVE_COMPLETION_EVIDENCE_SENTENCES="${ITERATIVE_COMPLETION_EVIDENCE_SENTENCES:-0}"
ITERATIVE_COMPLETION_CANDIDATE_SPANS="${ITERATIVE_COMPLETION_CANDIDATE_SPANS:-0}"
ITERATIVE_COMPLETION_MAX_CANDIDATES="${ITERATIVE_COMPLETION_MAX_CANDIDATES:-0}"
MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-10}"
MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-5}"
MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-3}"
MIN_EXPANSION_SCORE="${MIN_EXPANSION_SCORE:-0.06}"
MIN_ANSWER_UNIT_SCORE="${MIN_ANSWER_UNIT_SCORE:-0.25}"
STRICT_EXPANSION_ANSWER_UNIT_GATE="${STRICT_EXPANSION_ANSWER_UNIT_GATE:-0}"
STRICT_EXPANSION_MIN_TARGET_RELEVANCE="${STRICT_EXPANSION_MIN_TARGET_RELEVANCE:-0.45}"
ANSWER_TYPE_GATE_MODE="${ANSWER_TYPE_GATE_MODE:-soft}"
MAX_EXPANSION_CLAIM_WORDS="${MAX_EXPANSION_CLAIM_WORDS:-22}"
RECOVER_LABELS="${RECOVER_LABELS:-not_supported,no_citation,wrong_or_missing_citation}"
RECOVER_IMPORTANCE="${RECOVER_IMPORTANCE:-critical,supporting}"
RECOVERY_TOP_K="${RECOVERY_TOP_K:-6}"
RECOVERY_GROUP_SIZE="${RECOVERY_GROUP_SIZE:-3}"
MIN_RECOVERY_SCORE="${MIN_RECOVERY_SCORE:-0.03}"
MAX_RECOVERED_CITATIONS="${MAX_RECOVERED_CITATIONS:-3}"
MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-22}"
MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-26}"
CITATION_POLICY="${CITATION_POLICY:-minimal}"
SENTENCE_CITATION_SOURCE="${SENTENCE_CITATION_SOURCE:-source}"
MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-2}"
MAX_CITATIONS_PER_SENTENCE="${MAX_CITATIONS_PER_SENTENCE:-4}"
RENDER_MODE="${RENDER_MODE:-answer_preserving}"
PRESERVE_ANSWER_SENTENCES="${PRESERVE_ANSWER_SENTENCES:-1}"
PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES="${PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES:-0}"
MAX_PRESERVED_SOURCE_WORDS="${MAX_PRESERVED_SOURCE_WORDS:-48}"
MIN_PRESERVED_CLAIM_FRACTION="${MIN_PRESERVED_CLAIM_FRACTION:-0.50}"
MIN_REJECTED_PRESERVE_QUESTION_OVERLAP="${MIN_REJECTED_PRESERVE_QUESTION_OVERLAP:-0.25}"
QAMPARI_ANSWER_RELEVANCE_FILTER="${QAMPARI_ANSWER_RELEVANCE_FILTER:-1}"
QAMPARI_USE_SELECT="${QAMPARI_USE_SELECT:-0}"
QAMPARI_SELECT_ITEM_VERIFIER="${QAMPARI_SELECT_ITEM_VERIFIER:-llm}"
QAMPARI_SELECT_ITEM_VERIFY_MODEL="${QAMPARI_SELECT_ITEM_VERIFY_MODEL:-gpt-4o-mini}"
QAMPARI_SELECT_SELECTION_MODE="${QAMPARI_SELECT_SELECTION_MODE:-joint}"
QAMPARI_SELECT_JOINT_MODEL="${QAMPARI_SELECT_JOINT_MODEL:-gpt-4o-mini}"
QAMPARI_SELECT_JOINT_DOCS="${QAMPARI_SELECT_JOINT_DOCS:-10}"
QAMPARI_SELECT_JOINT_CANDIDATE_SPANS="${QAMPARI_SELECT_JOINT_CANDIDATE_SPANS:-120}"
QAMPARI_SELECT_JOINT_REQUIRE_TYPE_MATCH="${QAMPARI_SELECT_JOINT_REQUIRE_TYPE_MATCH:-1}"
QAMPARI_SELECT_STRICT_QUESTION_TYPE_GUARD="${QAMPARI_SELECT_STRICT_QUESTION_TYPE_GUARD:-0}"
QAMPARI_SELECT_EXPAND_FROM_DOCS="${QAMPARI_SELECT_EXPAND_FROM_DOCS:-llm}"
QAMPARI_SELECT_EXPANSION_MODEL="${QAMPARI_SELECT_EXPANSION_MODEL:-gpt-4o-mini}"
QAMPARI_SELECT_MAX_OUTPUT_ITEMS="${QAMPARI_SELECT_MAX_OUTPUT_ITEMS:-5}"
QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL="${QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL:-0}"
QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K="${QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K:-0}"
QAMPARI_SELECT_PRESERVED_ORIGINAL_LABEL="${QAMPARI_SELECT_PRESERVED_ORIGINAL_LABEL:-backfilled}"
QAMPARI_SELECT_BACKFILL_REQUIRES_TYPE_SAFE="${QAMPARI_SELECT_BACKFILL_REQUIRES_TYPE_SAFE:-0}"
QAMPARI_SELECT_EMPTY_OUTPUT_POLICY="${QAMPARI_SELECT_EMPTY_OUTPUT_POLICY:-empty}"
QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL="${QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL:-0}"
QAMPARI_SELECT_RESTORE_ORIGINAL_SUBSPANS="${QAMPARI_SELECT_RESTORE_ORIGINAL_SUBSPANS:-1}"
QAMPARI_SELECT_DROP_OBVIOUS_BAD="${QAMPARI_SELECT_DROP_OBVIOUS_BAD:-1}"
QAMPARI_SELECT_MAX_EXPANSION_CANDIDATES="${QAMPARI_SELECT_MAX_EXPANSION_CANDIDATES:-8}"
QAMPARI_SELECT_EXPANSION_DOCS="${QAMPARI_SELECT_EXPANSION_DOCS:-8}"
QAMPARI_SELECT_REPAIR_TOP_K="${QAMPARI_SELECT_REPAIR_TOP_K:-5}"
QAMPARI_SELECT_MAX_CITATIONS_PER_ITEM="${QAMPARI_SELECT_MAX_CITATIONS_PER_ITEM:-1}"
FINAL_AUDIT="${FINAL_AUDIT:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
REVISION_WORKERS="${REVISION_WORKERS:-8}"
CACHE_SAVE_EVERY="${CACHE_SAVE_EVERY:-50}"
LIMIT="${LIMIT:-}"

NO_CITATIONS="${NO_CITATIONS:-0}"
NO_MAUVE="${NO_MAUVE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$BASE_DIR" "$V0_DIR" "$V3_DIR" logs

RAW_INPUTS=("$ASQA_RAW" "$ELI5_RAW" "$QAMPARI_RAW")

run_step() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  echo "+ $*"
  "$@"
  local status=$?
  if [ "$status" -ne 0 ]; then
    if [ "$ALLOW_PARTIAL" = "1" ]; then
      echo "WARNING: $label failed with exit code $status. Continuing because ALLOW_PARTIAL=1."
    else
      echo "FAILED: $label exited with $status"
      exit "$status"
    fi
  fi
  return "$status"
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -s "$path" ]; then
    echo "FAILED: missing or empty $label: $path"
    exit 1
  fi
}

require_csv_min_data_rows() {
  local path="$1"
  local min_rows="$2"
  local label="$3"
  require_file "$path" "$label"
  local rows
  rows=$("$PYTHON" - "$path" <<'PY'
import csv
import sys
path = sys.argv[1]
with open(path, newline='', encoding='utf-8') as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)
  if [ "$rows" -lt "$min_rows" ]; then
    echo "FAILED: $label has $rows data rows, expected at least $min_rows: $path"
    exit 1
  fi
}

dataset_setting() {
  local prefix="$1"
  local key="$2"
  local fallback="$3"
  local var="${prefix}_${key}"
  printf '%s' "${!var:-$fallback}"
}

present_inputs=()
[ -f "$ASQA_RAW" ] && REQUIRED_DATASETS+=("asqa")
[ -f "$ELI5_RAW" ] && REQUIRED_DATASETS+=("eli5")
[ -f "$QAMPARI_RAW" ] && REQUIRED_DATASETS+=("qampari")
for raw in "${RAW_INPUTS[@]}"; do
  if [ -f "$raw" ]; then
    present_inputs+=("$raw")
  else
    echo "WARNING: raw result not found, skip: $raw"
  fi
done
if [ "${#present_inputs[@]}" -eq 0 ]; then
  echo "No input result JSON found. Stop."
  exit 1
fi
REQUIRED_DATASETS_CSV="$(IFS=,; echo "${REQUIRED_DATASETS[*]}")"

echo "=== COVER-RAG v3 three-dataset run ==="
echo "Root: $ROOT_DIR"
echo "Base dir: $BASE_DIR"
echo "OpenAI base URL: ${OPENAI_API_BASE:-<default>}"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set. LLM-based audit/revision steps will fail."
fi
echo "Inputs:"
printf '  - %s\n' "${present_inputs[@]}"
echo "Decomposer: $DECOMPOSER"
echo "Verifier: $VERIFIER"
echo "NLI premise: mode=$NLI_PREMISE_MODE, top_sentences=$NLI_TOP_SENTENCES, window=$NLI_SENTENCE_WINDOW_SIZE"
echo "Verifier thresholds: entail=$ENTAIL_THRESHOLD, contradiction=$CONTRADICTION_THRESHOLD, ambiguous_margin=$AMBIGUOUS_MARGIN"
echo "Render mode: $RENDER_MODE"
echo "Preserve answer sentences: $PRESERVE_ANSWER_SENTENCES, max words=$MAX_PRESERVED_SOURCE_WORDS, min kept fraction=$MIN_PRESERVED_CLAIM_FRACTION"
echo "Preserve rejected critical answer sentences: $PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES, min question overlap=$MIN_REJECTED_PRESERVE_QUESTION_OVERLAP"
echo "Expansion: $EXPANSION_MODE, sentences=$EXPANSION_EVIDENCE_SENTENCES, candidates=$MAX_EXPANSION_CANDIDATES, verifications=$MAX_EXPANSION_VERIFICATIONS, kept=$MAX_EXPANDED_CLAIMS"
echo "Expansion gate: filter_general_spans=$EXPANSION_FILTER_GENERAL_ANSWER_SPANS, min_question_overlap=$EXPANSION_MIN_QUESTION_OVERLAP, max_claim_words=$MAX_EXPANSION_CLAIM_WORDS, answer_targeted=$ANSWER_TARGETED_EXPANSION, extractive=$EXPANSION_INCLUDE_EXTRACTIVE, extractive_top_k=$ANSWER_TARGETED_EXTRACTIVE_TOP_K, extractive_sentence_overlap=$ANSWER_TARGETED_MIN_SENTENCE_OVERLAP, span_id_cap=$MAX_SELECTED_SPAN_IDS, explanatory_phrase_span=$EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN, explanatory_title=$EXPLANATORY_TITLE_SPAN_MODE, explanatory_kept=$MAX_EXPLANATORY_EXPANDED_CLAIMS, explanatory_verify=$MAX_EXPLANATORY_EXPANSION_VERIFICATIONS, explanatory_min_unit=$MIN_EXPLANATORY_ANSWER_UNIT_SCORE, explanatory_precision=$EXPLANATORY_PRECISION_GATE, explanation_integrated=$EXPLANATION_INTEGRATED_REFINEMENT, explanation_integrated_claims=$MAX_EXPLANATION_INTEGRATED_CLAIMS, explanation_growth=$MAX_EXPLANATION_LENGTH_GROWTH, candidate_selection=$CANDIDATE_OUTPUT_SELECTION, candidate_recall_claims=$CANDIDATE_OUTPUT_RECALL_CLAIMS, candidate_recall_growth=$CANDIDATE_OUTPUT_RECALL_GROWTH"
echo "Answer-unit gate: enabled=$ANSWER_UNIT_RELEVANCE_GATE min_directness=$ANSWER_UNIT_MIN_DIRECTNESS explanatory_min_directness=$EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS"
echo "Iterative completion: rounds=$ITERATIVE_COMPLETION_ROUNDS new_claims_per_round=$ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND verifications_per_round=$ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND evidence_sentences=$ITERATIVE_COMPLETION_EVIDENCE_SENTENCES candidate_spans=$ITERATIVE_COMPLETION_CANDIDATE_SPANS candidates=$ITERATIVE_COMPLETION_MAX_CANDIDATES"
echo "Recovery: labels=$RECOVER_LABELS, importance=$RECOVER_IMPORTANCE, top_k=$RECOVERY_TOP_K, group_size=$RECOVERY_GROUP_SIZE, min_score=$MIN_RECOVERY_SCORE, max_citations=$MAX_RECOVERED_CITATIONS"
echo "QAMPARI selector: use=$QAMPARI_USE_SELECT, joint_docs=$QAMPARI_SELECT_JOINT_DOCS, spans=$QAMPARI_SELECT_JOINT_CANDIDATE_SPANS, max_items=$QAMPARI_SELECT_MAX_OUTPUT_ITEMS"
echo "Final audit: $FINAL_AUDIT"
echo "Limit: ${LIMIT:-<full dataset>}"
echo "Revision workers: $REVISION_WORKERS"
echo "Cache namespace: ${CACHE_NAMESPACE:-<default>}"
echo "V0 score mode: $V0_SCORE_MODE"
echo "Allow partial outputs: $ALLOW_PARTIAL"

V0_COMPARE_CSV="$V0_DIR/cover_audit_comparison.csv"
if [ -n "$V0_SUMMARY_CSV" ] && [ -f "$V0_SUMMARY_CSV" ]; then
  V0_COMPARE_CSV="$V0_SUMMARY_CSV"
  echo
  echo "Using supplied v0 summary: $V0_COMPARE_CSV"
elif [ "$PREPARE_V0" = "1" ]; then
  V0_ARGS=(
    "$PYTHON" scripts/run_cover_audit_batch.py
    --inputs "${present_inputs[@]}"
    --output-dir "$V0_DIR"
    --python "$PYTHON"
    --score-mode "$V0_SCORE_MODE"
    --decomposer "$DECOMPOSER"
    --decompose-model "$DECOMPOSE_MODEL"
    --verifier "$VERIFIER"
    --llm-verify-model "$LLM_VERIFY_MODEL"
    --nli-model "$NLI_MODEL"
    --evidence-scope cited_then_all
    --cache-file "$V0_CACHE_FILE"
    --summary-csv "$V0_COMPARE_CSV"
    --summary-md "$V0_DIR/cover_audit_comparison.md"
  )
  if [ "$FORCE_V0" = "1" ]; then
    V0_ARGS+=(--force-cover --force-eval --force-merge)
  fi
  run_step "Stage 1/5: prepare missing v0 baselines" "${V0_ARGS[@]}"
  require_csv_min_data_rows "$V0_COMPARE_CSV" "${#present_inputs[@]}" "v0 summary"
else
  echo
  echo "PREPARE_V0=0. Using existing v0 summary at $V0_COMPARE_CSV."
  require_csv_min_data_rows "$V0_COMPARE_CSV" "${#present_inputs[@]}" "v0 summary"
fi

run_v3_one() {
  local name="$1"
  local raw="$2"
  local docs="$3"
  if [ ! -f "$raw" ]; then
    echo "Skip $name: input not found: $raw"
    return 0
  fi
  if [ "$name" = "qampari_gtr" ] && [ "$QAMPARI_USE_SELECT" = "1" ]; then
    run_qampari_select_one "$name" "$raw" "$docs"
    return 0
  fi
  local prefix=""
  case "$name" in
    asqa_*) prefix="ASQA" ;;
    eli5_*) prefix="ELI5" ;;
    qampari_*) prefix="QAMPARI" ;;
    *) prefix="DATASET" ;;
  esac

  local decomposer decompose_model
  local expansion_mode expansion_model expansion_max_tokens expansion_evidence_sentences expansion_candidate_spans max_selected_span_ids
  local expansion_output_mode strict_expansion_answer_unit_gate strict_expansion_min_target_relevance answer_type_gate_mode
  local answer_unit_relevance_gate answer_unit_min_directness explanatory_answer_unit_min_directness
  local expansion_include_extractive explanatory_expansion_require_phrase_span explanatory_title_span_mode
  local explanatory_precision_gate explanatory_precision_min_target_relevance explanatory_precision_min_question_relevance
  local explanation_integrated_refinement max_explanation_integrated_claims max_explanation_length_growth
  local candidate_output_selection candidate_output_recall_claims candidate_output_recall_growth
  local candidate_output_min_support candidate_output_support_tolerance candidate_output_max_unsupported_rate
  local candidate_output_min_answer_unit_gain candidate_output_min_supported_claim_gain
  local expansion_min_question_overlap expansion_query_source expansion_use_intermediate_answer
  local max_expansion_candidates max_expansion_verifications max_explanatory_expansion_verifications
  local max_expanded_claims min_expansion_score min_answer_unit_score answer_targeted_extractive_top_k
  local max_explanatory_expanded_claims min_explanatory_answer_unit_score
  local answer_targeted_min_sentence_overlap max_verified_claims max_final_claims
  local recover_labels recover_importance recovery_top_k recovery_group_size min_recovery_score max_recovered_citations
  local recall_oriented_completion recall_completion_max_targets recall_completion_rejected_targets
  local recall_completion_evidence_sentences recall_completion_supported_claims
  local recall_completion_min_sentence_overlap recall_completion_min_target_overlap
  local citation_policy sentence_citation_source max_citations_per_claim max_citations_per_sentence render_mode
  local preserve_answer_sentences preserve_rejected_critical_answer_sentences
  local expansion_filter_general_answer_spans answer_targeted_expansion qampari_answer_relevance_filter
  local coverage_guided_retrieval coverage_guided_targets_per_round coverage_guided_min_targets
  local coverage_guided_min_question_overlap coverage_guided_query_supported_claims coverage_guided_allow_evidence_targets

  decomposer="$(dataset_setting "$prefix" DECOMPOSER "$DECOMPOSER")"
  decompose_model="$(dataset_setting "$prefix" DECOMPOSE_MODEL "$DECOMPOSE_MODEL")"
  expansion_mode="$(dataset_setting "$prefix" EXPANSION_MODE "$EXPANSION_MODE")"
  expansion_model="$(dataset_setting "$prefix" EXPANSION_MODEL "$EXPANSION_MODEL")"
  expansion_max_tokens="$(dataset_setting "$prefix" EXPANSION_MAX_TOKENS "$EXPANSION_MAX_TOKENS")"
  expansion_output_mode="$(dataset_setting "$prefix" EXPANSION_OUTPUT_MODE "$EXPANSION_OUTPUT_MODE")"
  expansion_include_extractive="$(dataset_setting "$prefix" EXPANSION_INCLUDE_EXTRACTIVE "$EXPANSION_INCLUDE_EXTRACTIVE")"
  expansion_evidence_sentences="$(dataset_setting "$prefix" EXPANSION_EVIDENCE_SENTENCES "$EXPANSION_EVIDENCE_SENTENCES")"
  expansion_candidate_spans="$(dataset_setting "$prefix" EXPANSION_CANDIDATE_SPANS "$EXPANSION_CANDIDATE_SPANS")"
  max_selected_span_ids="$(dataset_setting "$prefix" MAX_SELECTED_SPAN_IDS "$MAX_SELECTED_SPAN_IDS")"
  expansion_min_question_overlap="$(dataset_setting "$prefix" EXPANSION_MIN_QUESTION_OVERLAP "$EXPANSION_MIN_QUESTION_OVERLAP")"
  expansion_query_source="$(dataset_setting "$prefix" EXPANSION_QUERY_SOURCE "$EXPANSION_QUERY_SOURCE")"
  expansion_use_intermediate_answer="$(dataset_setting "$prefix" EXPANSION_USE_INTERMEDIATE_ANSWER "$EXPANSION_USE_INTERMEDIATE_ANSWER")"
  max_expansion_candidates="$(dataset_setting "$prefix" MAX_EXPANSION_CANDIDATES "$MAX_EXPANSION_CANDIDATES")"
  max_expansion_verifications="$(dataset_setting "$prefix" MAX_EXPANSION_VERIFICATIONS "$MAX_EXPANSION_VERIFICATIONS")"
  max_expanded_claims="$(dataset_setting "$prefix" MAX_EXPANDED_CLAIMS "$MAX_EXPANDED_CLAIMS")"
  min_expansion_score="$(dataset_setting "$prefix" MIN_EXPANSION_SCORE "$MIN_EXPANSION_SCORE")"
  min_answer_unit_score="$(dataset_setting "$prefix" MIN_ANSWER_UNIT_SCORE "$MIN_ANSWER_UNIT_SCORE")"
  strict_expansion_answer_unit_gate="$(dataset_setting "$prefix" STRICT_EXPANSION_ANSWER_UNIT_GATE "$STRICT_EXPANSION_ANSWER_UNIT_GATE")"
  strict_expansion_min_target_relevance="$(dataset_setting "$prefix" STRICT_EXPANSION_MIN_TARGET_RELEVANCE "$STRICT_EXPANSION_MIN_TARGET_RELEVANCE")"
  answer_type_gate_mode="$(dataset_setting "$prefix" ANSWER_TYPE_GATE_MODE "$ANSWER_TYPE_GATE_MODE")"
  max_expansion_claim_words="$(dataset_setting "$prefix" MAX_EXPANSION_CLAIM_WORDS "$MAX_EXPANSION_CLAIM_WORDS")"
  answer_targeted_extractive_top_k="$(dataset_setting "$prefix" ANSWER_TARGETED_EXTRACTIVE_TOP_K "$ANSWER_TARGETED_EXTRACTIVE_TOP_K")"
  answer_targeted_min_sentence_overlap="$(dataset_setting "$prefix" ANSWER_TARGETED_MIN_SENTENCE_OVERLAP "$ANSWER_TARGETED_MIN_SENTENCE_OVERLAP")"
  recover_labels="$(dataset_setting "$prefix" RECOVER_LABELS "$RECOVER_LABELS")"
  recover_importance="$(dataset_setting "$prefix" RECOVER_IMPORTANCE "$RECOVER_IMPORTANCE")"
  recovery_top_k="$(dataset_setting "$prefix" RECOVERY_TOP_K "$RECOVERY_TOP_K")"
  recovery_group_size="$(dataset_setting "$prefix" RECOVERY_GROUP_SIZE "$RECOVERY_GROUP_SIZE")"
  min_recovery_score="$(dataset_setting "$prefix" MIN_RECOVERY_SCORE "$MIN_RECOVERY_SCORE")"
  max_recovered_citations="$(dataset_setting "$prefix" MAX_RECOVERED_CITATIONS "$MAX_RECOVERED_CITATIONS")"
  explanatory_expansion_require_phrase_span="$(dataset_setting "$prefix" EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN "$EXPLANATORY_EXPANSION_REQUIRE_PHRASE_SPAN")"
  explanatory_title_span_mode="$(dataset_setting "$prefix" EXPLANATORY_TITLE_SPAN_MODE "$EXPLANATORY_TITLE_SPAN_MODE")"
  max_explanatory_expanded_claims="$(dataset_setting "$prefix" MAX_EXPLANATORY_EXPANDED_CLAIMS "$MAX_EXPLANATORY_EXPANDED_CLAIMS")"
  max_explanatory_expansion_verifications="$(dataset_setting "$prefix" MAX_EXPLANATORY_EXPANSION_VERIFICATIONS "$MAX_EXPLANATORY_EXPANSION_VERIFICATIONS")"
  min_explanatory_answer_unit_score="$(dataset_setting "$prefix" MIN_EXPLANATORY_ANSWER_UNIT_SCORE "$MIN_EXPLANATORY_ANSWER_UNIT_SCORE")"
  explanatory_precision_gate="$(dataset_setting "$prefix" EXPLANATORY_PRECISION_GATE "$EXPLANATORY_PRECISION_GATE")"
  explanatory_precision_min_target_relevance="$(dataset_setting "$prefix" EXPLANATORY_PRECISION_MIN_TARGET_RELEVANCE "$EXPLANATORY_PRECISION_MIN_TARGET_RELEVANCE")"
  explanatory_precision_min_question_relevance="$(dataset_setting "$prefix" EXPLANATORY_PRECISION_MIN_QUESTION_RELEVANCE "$EXPLANATORY_PRECISION_MIN_QUESTION_RELEVANCE")"
  explanation_integrated_refinement="$(dataset_setting "$prefix" EXPLANATION_INTEGRATED_REFINEMENT "$EXPLANATION_INTEGRATED_REFINEMENT")"
  max_explanation_integrated_claims="$(dataset_setting "$prefix" MAX_EXPLANATION_INTEGRATED_CLAIMS "$MAX_EXPLANATION_INTEGRATED_CLAIMS")"
  max_explanation_length_growth="$(dataset_setting "$prefix" MAX_EXPLANATION_LENGTH_GROWTH "$MAX_EXPLANATION_LENGTH_GROWTH")"
  candidate_output_selection="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_SELECTION "$CANDIDATE_OUTPUT_SELECTION")"
  candidate_output_recall_claims="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_RECALL_CLAIMS "$CANDIDATE_OUTPUT_RECALL_CLAIMS")"
  candidate_output_recall_growth="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_RECALL_GROWTH "$CANDIDATE_OUTPUT_RECALL_GROWTH")"
  candidate_output_min_support="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_MIN_SUPPORT "$CANDIDATE_OUTPUT_MIN_SUPPORT")"
  candidate_output_support_tolerance="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_SUPPORT_TOLERANCE "$CANDIDATE_OUTPUT_SUPPORT_TOLERANCE")"
  candidate_output_max_unsupported_rate="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE "$CANDIDATE_OUTPUT_MAX_UNSUPPORTED_RATE")"
  candidate_output_min_answer_unit_gain="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_MIN_ANSWER_UNIT_GAIN "$CANDIDATE_OUTPUT_MIN_ANSWER_UNIT_GAIN")"
  candidate_output_min_supported_claim_gain="$(dataset_setting "$prefix" CANDIDATE_OUTPUT_MIN_SUPPORTED_CLAIM_GAIN "$CANDIDATE_OUTPUT_MIN_SUPPORTED_CLAIM_GAIN")"
  recall_oriented_completion="$(dataset_setting "$prefix" RECALL_ORIENTED_COMPLETION "$RECALL_ORIENTED_COMPLETION")"
  recall_completion_max_targets="$(dataset_setting "$prefix" RECALL_COMPLETION_MAX_TARGETS "$RECALL_COMPLETION_MAX_TARGETS")"
  recall_completion_rejected_targets="$(dataset_setting "$prefix" RECALL_COMPLETION_REJECTED_TARGETS "$RECALL_COMPLETION_REJECTED_TARGETS")"
  recall_completion_evidence_sentences="$(dataset_setting "$prefix" RECALL_COMPLETION_EVIDENCE_SENTENCES "$RECALL_COMPLETION_EVIDENCE_SENTENCES")"
  recall_completion_supported_claims="$(dataset_setting "$prefix" RECALL_COMPLETION_SUPPORTED_CLAIMS "$RECALL_COMPLETION_SUPPORTED_CLAIMS")"
  recall_completion_min_sentence_overlap="$(dataset_setting "$prefix" RECALL_COMPLETION_MIN_SENTENCE_OVERLAP "$RECALL_COMPLETION_MIN_SENTENCE_OVERLAP")"
  recall_completion_min_target_overlap="$(dataset_setting "$prefix" RECALL_COMPLETION_MIN_TARGET_OVERLAP "$RECALL_COMPLETION_MIN_TARGET_OVERLAP")"
  answer_unit_relevance_gate="$(dataset_setting "$prefix" ANSWER_UNIT_RELEVANCE_GATE "$ANSWER_UNIT_RELEVANCE_GATE")"
  answer_unit_min_directness="$(dataset_setting "$prefix" ANSWER_UNIT_MIN_DIRECTNESS "$ANSWER_UNIT_MIN_DIRECTNESS")"
  explanatory_answer_unit_min_directness="$(dataset_setting "$prefix" EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS "$EXPLANATORY_ANSWER_UNIT_MIN_DIRECTNESS")"
  iterative_completion_rounds="$(dataset_setting "$prefix" ITERATIVE_COMPLETION_ROUNDS "$ITERATIVE_COMPLETION_ROUNDS")"
  iterative_completion_max_new_claims_per_round="$(dataset_setting "$prefix" ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND "$ITERATIVE_COMPLETION_MAX_NEW_CLAIMS_PER_ROUND")"
  iterative_completion_max_verifications_per_round="$(dataset_setting "$prefix" ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND "$ITERATIVE_COMPLETION_MAX_VERIFICATIONS_PER_ROUND")"
  iterative_completion_evidence_sentences="$(dataset_setting "$prefix" ITERATIVE_COMPLETION_EVIDENCE_SENTENCES "$ITERATIVE_COMPLETION_EVIDENCE_SENTENCES")"
  iterative_completion_candidate_spans="$(dataset_setting "$prefix" ITERATIVE_COMPLETION_CANDIDATE_SPANS "$ITERATIVE_COMPLETION_CANDIDATE_SPANS")"
  iterative_completion_max_candidates="$(dataset_setting "$prefix" ITERATIVE_COMPLETION_MAX_CANDIDATES "$ITERATIVE_COMPLETION_MAX_CANDIDATES")"
  dynamic_retrieval_mode="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_MODE "$DYNAMIC_RETRIEVAL_MODE")"
  dynamic_retrieval_top_k="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_TOP_K "$DYNAMIC_RETRIEVAL_TOP_K")"
  dynamic_retrieval_per_query_docs="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_PER_QUERY_DOCS "$DYNAMIC_RETRIEVAL_PER_QUERY_DOCS")"
  dynamic_retrieval_max_queries="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_MAX_QUERIES "$DYNAMIC_RETRIEVAL_MAX_QUERIES")"
  dynamic_retrieval_max_rejected_claims="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_MAX_REJECTED_CLAIMS "$DYNAMIC_RETRIEVAL_MAX_REJECTED_CLAIMS")"
  dynamic_retrieval_max_doc_pool="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_MAX_DOC_POOL "$DYNAMIC_RETRIEVAL_MAX_DOC_POOL")"
  dynamic_retrieval_query_max_chars="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_QUERY_MAX_CHARS "$DYNAMIC_RETRIEVAL_QUERY_MAX_CHARS")"
  dynamic_retrieval_min_query_terms="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_MIN_QUERY_TERMS "$DYNAMIC_RETRIEVAL_MIN_QUERY_TERMS")"
  dynamic_retrieval_sentence_boost="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_SENTENCE_BOOST "$DYNAMIC_RETRIEVAL_SENTENCE_BOOST")"
  dynamic_retrieval_query_strategy="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_QUERY_STRATEGY "$DYNAMIC_RETRIEVAL_QUERY_STRATEGY")"
  dynamic_retrieval_min_score="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_MIN_SCORE "$DYNAMIC_RETRIEVAL_MIN_SCORE")"
  dynamic_retrieval_require_missing_signal="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_REQUIRE_MISSING_SIGNAL "$DYNAMIC_RETRIEVAL_REQUIRE_MISSING_SIGNAL")"
  dynamic_retrieval_use_supported_answer="$(dataset_setting "$prefix" DYNAMIC_RETRIEVAL_USE_SUPPORTED_ANSWER "$DYNAMIC_RETRIEVAL_USE_SUPPORTED_ANSWER")"
  coverage_guided_retrieval="$(dataset_setting "$prefix" COVERAGE_GUIDED_RETRIEVAL "$COVERAGE_GUIDED_RETRIEVAL")"
  coverage_guided_targets_per_round="$(dataset_setting "$prefix" COVERAGE_GUIDED_TARGETS_PER_ROUND "$COVERAGE_GUIDED_TARGETS_PER_ROUND")"
  coverage_guided_min_targets="$(dataset_setting "$prefix" COVERAGE_GUIDED_MIN_TARGETS "$COVERAGE_GUIDED_MIN_TARGETS")"
  coverage_guided_min_question_overlap="$(dataset_setting "$prefix" COVERAGE_GUIDED_MIN_QUESTION_OVERLAP "$COVERAGE_GUIDED_MIN_QUESTION_OVERLAP")"
  coverage_guided_query_supported_claims="$(dataset_setting "$prefix" COVERAGE_GUIDED_QUERY_SUPPORTED_CLAIMS "$COVERAGE_GUIDED_QUERY_SUPPORTED_CLAIMS")"
  coverage_guided_allow_evidence_targets="$(dataset_setting "$prefix" COVERAGE_GUIDED_ALLOW_EVIDENCE_TARGETS "$COVERAGE_GUIDED_ALLOW_EVIDENCE_TARGETS")"
  coverage_reranker_model="$(dataset_setting "$prefix" COVERAGE_RERANKER_MODEL "$COVERAGE_RERANKER_MODEL")"
  coverage_reranker_device="$(dataset_setting "$prefix" COVERAGE_RERANKER_DEVICE "$COVERAGE_RERANKER_DEVICE")"
  coverage_reranker_batch_size="$(dataset_setting "$prefix" COVERAGE_RERANKER_BATCH_SIZE "$COVERAGE_RERANKER_BATCH_SIZE")"
  coverage_reranker_max_length="$(dataset_setting "$prefix" COVERAGE_RERANKER_MAX_LENGTH "$COVERAGE_RERANKER_MAX_LENGTH")"
  coverage_reranker_doc_prefilter="$(dataset_setting "$prefix" COVERAGE_RERANKER_DOC_PREFILTER "$COVERAGE_RERANKER_DOC_PREFILTER")"
  coverage_reranker_doc_weight="$(dataset_setting "$prefix" COVERAGE_RERANKER_DOC_WEIGHT "$COVERAGE_RERANKER_DOC_WEIGHT")"
  coverage_reranker_candidate_weight="$(dataset_setting "$prefix" COVERAGE_RERANKER_CANDIDATE_WEIGHT "$COVERAGE_RERANKER_CANDIDATE_WEIGHT")"
  max_verified_claims="$(dataset_setting "$prefix" MAX_VERIFIED_CLAIMS "$MAX_VERIFIED_CLAIMS")"
  max_final_claims="$(dataset_setting "$prefix" MAX_FINAL_CLAIMS "$MAX_FINAL_CLAIMS")"
  citation_policy="$(dataset_setting "$prefix" CITATION_POLICY "$CITATION_POLICY")"
  sentence_citation_source="$(dataset_setting "$prefix" SENTENCE_CITATION_SOURCE "$SENTENCE_CITATION_SOURCE")"
  max_citations_per_claim="$(dataset_setting "$prefix" MAX_CITATIONS_PER_CLAIM "$MAX_CITATIONS_PER_CLAIM")"
  max_citations_per_sentence="$(dataset_setting "$prefix" MAX_CITATIONS_PER_SENTENCE "$MAX_CITATIONS_PER_SENTENCE")"
  render_mode="$(dataset_setting "$prefix" RENDER_MODE "$RENDER_MODE")"
  preserve_answer_sentences="$(dataset_setting "$prefix" PRESERVE_ANSWER_SENTENCES "$PRESERVE_ANSWER_SENTENCES")"
  preserve_rejected_critical_answer_sentences="$(dataset_setting "$prefix" PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES "$PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES")"
  expansion_filter_general_answer_spans="$(dataset_setting "$prefix" EXPANSION_FILTER_GENERAL_ANSWER_SPANS "$EXPANSION_FILTER_GENERAL_ANSWER_SPANS")"
  answer_targeted_expansion="$(dataset_setting "$prefix" ANSWER_TARGETED_EXPANSION "$ANSWER_TARGETED_EXPANSION")"
  qampari_answer_relevance_filter="$(dataset_setting "$prefix" QAMPARI_ANSWER_RELEVANCE_FILTER "$QAMPARI_ANSWER_RELEVANCE_FILTER")"

  echo "Dataset $name settings: decomposer=$decomposer expansion=$expansion_mode output_mode=$expansion_output_mode max_tokens=$expansion_max_tokens spans=$expansion_candidate_spans selected_span_cap=$max_selected_span_ids candidates=$max_expansion_candidates verifications=$max_expansion_verifications kept=$max_expanded_claims min_q_overlap=$expansion_min_question_overlap min_answer_unit=$min_answer_unit_score strict_gate=$strict_expansion_answer_unit_gate min_target_rel=$strict_expansion_min_target_relevance answer_type_gate=$answer_type_gate_mode answer_unit_gate=$answer_unit_relevance_gate answer_unit_min=$answer_unit_min_directness explanatory_answer_unit_min=$explanatory_answer_unit_min_directness max_claim_words=$max_expansion_claim_words query_source=$expansion_query_source intermediate_answer=$expansion_use_intermediate_answer answer_targeted=$answer_targeted_expansion recall_completion=$recall_oriented_completion recall_targets=$recall_completion_max_targets iterative_rounds=$iterative_completion_rounds iterative_new_per_round=$iterative_completion_max_new_claims_per_round iterative_verify_per_round=$iterative_completion_max_verifications_per_round dynamic_retrieval=$dynamic_retrieval_mode dynamic_strategy=$dynamic_retrieval_query_strategy dynamic_docs=$dynamic_retrieval_top_k dynamic_min_score=$dynamic_retrieval_min_score dynamic_doc_pool=$dynamic_retrieval_max_doc_pool coverage_guided=$coverage_guided_retrieval coverage_targets=$coverage_guided_targets_per_round coverage_min_targets=$coverage_guided_min_targets coverage_min_q_overlap=$coverage_guided_min_question_overlap coverage_reranker=${coverage_reranker_model:-off} rerank_doc_weight=$coverage_reranker_doc_weight rerank_candidate_weight=$coverage_reranker_candidate_weight extractive=$expansion_include_extractive extractive_top_k=$answer_targeted_extractive_top_k extractive_sentence_overlap=$answer_targeted_min_sentence_overlap recovery_labels=$recover_labels recovery_top_k=$recovery_top_k recovery_max_cites=$max_recovered_citations explanatory_phrase_span=$explanatory_expansion_require_phrase_span explanatory_title=$explanatory_title_span_mode explanatory_kept=$max_explanatory_expanded_claims explanatory_verify=$max_explanatory_expansion_verifications explanatory_min_unit=$min_explanatory_answer_unit_score explanatory_precision=$explanatory_precision_gate explanatory_precision_target=$explanatory_precision_min_target_relevance explanatory_precision_question=$explanatory_precision_min_question_relevance explanation_integrated=$explanation_integrated_refinement explanation_integrated_claims=$max_explanation_integrated_claims explanation_growth=$max_explanation_length_growth candidate_selection=$candidate_output_selection candidate_recall_claims=$candidate_output_recall_claims candidate_recall_growth=$candidate_output_recall_growth candidate_min_unit_gain=$candidate_output_min_answer_unit_gain candidate_min_supported_gain=$candidate_output_min_supported_claim_gain max_final=$max_final_claims citation_policy=$citation_policy sentence_citation_source=$sentence_citation_source max_sent_cites=$max_citations_per_sentence"

  local cmd=(
    "$PYTHON" scripts/run_cover_v3_batch.py
    --inputs "$raw"
    --output-dir "$V3_DIR"
    --python "$PYTHON"
    --decomposer "$decomposer"
    --decompose-model "$decompose_model"
    --verifier "$VERIFIER"
    --llm-verify-model "$LLM_VERIFY_MODEL"
    --nli-model "$NLI_MODEL"
    --nli-premise-mode "$NLI_PREMISE_MODE"
    --nli-top-sentences "$NLI_TOP_SENTENCES"
    --nli-sentence-window-size "$NLI_SENTENCE_WINDOW_SIZE"
    --entail-threshold "$ENTAIL_THRESHOLD"
    --contradiction-threshold "$CONTRADICTION_THRESHOLD"
    --ambiguous-margin "$AMBIGUOUS_MARGIN"
    --evidence-scope cited
    --citation-policy "$citation_policy"
    --sentence-citation-source "$sentence_citation_source"
    --max-citations-per-claim "$max_citations_per_claim"
    --max-citations-per-sentence "$max_citations_per_sentence"
    --max-verified-claims "$max_verified_claims"
    --max-final-claims "$max_final_claims"
    --render-mode "$render_mode"
    --max-preserved-source-words "$MAX_PRESERVED_SOURCE_WORDS"
    --min-preserved-claim-fraction "$MIN_PRESERVED_CLAIM_FRACTION"
    --min-rejected-preserve-question-overlap "$MIN_REJECTED_PRESERVE_QUESTION_OVERLAP"
    --max-doc-pool "$MAX_DOC_POOL"
    --dynamic-retrieval-mode "$dynamic_retrieval_mode"
    --dynamic-retrieval-top-k "$dynamic_retrieval_top_k"
    --dynamic-retrieval-per-query-docs "$dynamic_retrieval_per_query_docs"
    --dynamic-retrieval-max-queries "$dynamic_retrieval_max_queries"
    --dynamic-retrieval-max-rejected-claims "$dynamic_retrieval_max_rejected_claims"
    --dynamic-retrieval-max-doc-pool "$dynamic_retrieval_max_doc_pool"
    --dynamic-retrieval-query-max-chars "$dynamic_retrieval_query_max_chars"
    --dynamic-retrieval-min-query-terms "$dynamic_retrieval_min_query_terms"
    --dynamic-retrieval-sentence-boost "$dynamic_retrieval_sentence_boost"
    --dynamic-retrieval-query-strategy "$dynamic_retrieval_query_strategy"
    --dynamic-retrieval-min-score "$dynamic_retrieval_min_score"
    --coverage-guided-targets-per-round "$coverage_guided_targets_per_round"
    --coverage-guided-min-targets "$coverage_guided_min_targets"
    --coverage-guided-min-question-overlap "$coverage_guided_min_question_overlap"
    --coverage-guided-query-supported-claims "$coverage_guided_query_supported_claims"
    --coverage-reranker-batch-size "$coverage_reranker_batch_size"
    --coverage-reranker-max-length "$coverage_reranker_max_length"
    --coverage-reranker-doc-prefilter "$coverage_reranker_doc_prefilter"
    --coverage-reranker-doc-weight "$coverage_reranker_doc_weight"
    --coverage-reranker-candidate-weight "$coverage_reranker_candidate_weight"
    --recover-labels "$recover_labels"
    --recover-importance "$recover_importance"
    --recovery-top-k "$recovery_top_k"
    --recovery-group-size "$recovery_group_size"
    --min-recovery-score "$min_recovery_score"
    --max-recovered-citations "$max_recovered_citations"
    --expansion-mode "$expansion_mode"
    --expansion-model "$expansion_model"
    --expansion-max-tokens "$expansion_max_tokens"
    --expansion-output-mode "$expansion_output_mode"
    --expansion-evidence-sentences "$expansion_evidence_sentences"
    --expansion-candidate-spans "$expansion_candidate_spans"
    --max-selected-span-ids "$max_selected_span_ids"
    --expansion-min-question-overlap "$expansion_min_question_overlap"
    --max-expansion-candidates "$max_expansion_candidates"
    --max-expansion-verifications "$max_expansion_verifications"
    --max-expanded-claims "$max_expanded_claims"
    --min-expansion-score "$min_expansion_score"
    --min-answer-unit-score "$min_answer_unit_score"
    --strict-expansion-min-target-relevance "$strict_expansion_min_target_relevance"
    --answer-type-gate-mode "$answer_type_gate_mode"
    --max-expansion-claim-words "$max_expansion_claim_words"
    --expansion-query-source "$expansion_query_source"
    --answer-targeted-extractive-top-k "$answer_targeted_extractive_top_k"
    --answer-targeted-min-sentence-overlap "$answer_targeted_min_sentence_overlap"
    --explanatory-title-span-mode "$explanatory_title_span_mode"
    --max-explanatory-expanded-claims "$max_explanatory_expanded_claims"
    --max-explanatory-expansion-verifications "$max_explanatory_expansion_verifications"
    --min-explanatory-answer-unit-score "$min_explanatory_answer_unit_score"
    --explanatory-precision-min-target-relevance "$explanatory_precision_min_target_relevance"
    --explanatory-precision-min-question-relevance "$explanatory_precision_min_question_relevance"
    --max-explanation-integrated-claims "$max_explanation_integrated_claims"
    --max-explanation-length-growth "$max_explanation_length_growth"
    --candidate-output-recall-claims "$candidate_output_recall_claims"
    --candidate-output-recall-growth "$candidate_output_recall_growth"
    --candidate-output-min-support "$candidate_output_min_support"
    --candidate-output-support-tolerance "$candidate_output_support_tolerance"
    --candidate-output-max-unsupported-rate "$candidate_output_max_unsupported_rate"
    --candidate-output-min-answer-unit-gain "$candidate_output_min_answer_unit_gain"
    --candidate-output-min-supported-claim-gain "$candidate_output_min_supported_claim_gain"
    --recall-completion-max-targets "$recall_completion_max_targets"
    --recall-completion-rejected-targets "$recall_completion_rejected_targets"
    --recall-completion-evidence-sentences "$recall_completion_evidence_sentences"
    --recall-completion-supported-claims "$recall_completion_supported_claims"
    --recall-completion-min-sentence-overlap "$recall_completion_min_sentence_overlap"
    --recall-completion-min-target-overlap "$recall_completion_min_target_overlap"
    --answer-unit-min-directness "$answer_unit_min_directness"
    --explanatory-answer-unit-min-directness "$explanatory_answer_unit_min_directness"
    --iterative-completion-rounds "$iterative_completion_rounds"
    --iterative-completion-max-new-claims-per-round "$iterative_completion_max_new_claims_per_round"
    --iterative-completion-max-verifications-per-round "$iterative_completion_max_verifications_per_round"
    --iterative-completion-evidence-sentences "$iterative_completion_evidence_sentences"
    --iterative-completion-candidate-spans "$iterative_completion_candidate_spans"
    --iterative-completion-max-candidates "$iterative_completion_max_candidates"
    --verify-expansion-with-sentence
    --require-expansion-answer-span
    --preserve-first-pass-claims
    --checkpoint-every "$CHECKPOINT_EVERY"
    --revision-workers "$REVISION_WORKERS"
    --cache-save-every "$CACHE_SAVE_EVERY"
    --cache-file "cache/cover_v3_${CACHE_NAMESPACE}${name}_cache.json"
    --summary-csv "$V3_DIR/${name}_cover_v3_comparison.csv"
    --summary-md "$V3_DIR/${name}_cover_v3_comparison.md"
  )
  if [ -f "$docs" ]; then
    cmd+=(--candidate-docs-file "$docs")
  else
    echo "WARNING: candidate docs not found for $name, using docs inside raw result: $docs"
  fi
  if [ -n "$coverage_reranker_model" ]; then
    cmd+=(--coverage-reranker-model "$coverage_reranker_model")
  fi
  if [ -n "$coverage_reranker_device" ]; then
    cmd+=(--coverage-reranker-device "$coverage_reranker_device")
  fi
  if [ "$dynamic_retrieval_require_missing_signal" = "1" ]; then
    cmd+=(--dynamic-retrieval-require-missing-signal)
  else
    cmd+=(--no-dynamic-retrieval-require-missing-signal)
  fi
  if [ "$dynamic_retrieval_use_supported_answer" = "1" ]; then
    cmd+=(--dynamic-retrieval-use-supported-answer)
  else
    cmd+=(--no-dynamic-retrieval-use-supported-answer)
  fi
  if [ "$coverage_guided_retrieval" = "1" ]; then
    cmd+=(--coverage-guided-retrieval)
  else
    cmd+=(--no-coverage-guided-retrieval)
  fi
  if [ "$coverage_guided_allow_evidence_targets" = "1" ]; then
    cmd+=(--coverage-guided-allow-evidence-targets)
  else
    cmd+=(--no-coverage-guided-allow-evidence-targets)
  fi
  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [ "$FINAL_AUDIT" = "1" ]; then
    cmd+=(--final-audit)
  else
    cmd+=(--no-final-audit)
  fi
  if [ "$preserve_answer_sentences" = "1" ]; then
    cmd+=(--preserve-answer-sentences)
  else
    cmd+=(--no-preserve-answer-sentences)
  fi
  if [ "$preserve_rejected_critical_answer_sentences" = "1" ]; then
    cmd+=(--preserve-rejected-critical-answer-sentences)
  else
    cmd+=(--no-preserve-rejected-critical-answer-sentences)
  fi
  if [ "$expansion_filter_general_answer_spans" = "1" ]; then
    cmd+=(--expansion-filter-general-answer-spans)
  else
    cmd+=(--no-expansion-filter-general-answer-spans)
  fi
  if [ "$strict_expansion_answer_unit_gate" = "1" ]; then
    cmd+=(--strict-expansion-answer-unit-gate)
  else
    cmd+=(--no-strict-expansion-answer-unit-gate)
  fi
  if [ "$answer_unit_relevance_gate" = "1" ]; then
    cmd+=(--answer-unit-relevance-gate)
  else
    cmd+=(--no-answer-unit-relevance-gate)
  fi
  if [ "$answer_targeted_expansion" = "1" ]; then
    cmd+=(--answer-targeted-expansion)
  else
    cmd+=(--no-answer-targeted-expansion)
  fi
  if [ "$expansion_include_extractive" = "1" ]; then
    cmd+=(--expansion-include-extractive)
  else
    cmd+=(--no-expansion-include-extractive)
  fi
  if [ "$explanatory_expansion_require_phrase_span" = "1" ]; then
    cmd+=(--explanatory-expansion-require-phrase-span)
  else
    cmd+=(--no-explanatory-expansion-require-phrase-span)
  fi
  if [ "$explanatory_precision_gate" = "1" ]; then
    cmd+=(--explanatory-precision-gate)
  else
    cmd+=(--no-explanatory-precision-gate)
  fi
  if [ "$explanation_integrated_refinement" = "1" ]; then
    cmd+=(--explanation-integrated-refinement)
  else
    cmd+=(--no-explanation-integrated-refinement)
  fi
  if [ "$candidate_output_selection" = "1" ]; then
    cmd+=(--candidate-output-selection)
  else
    cmd+=(--no-candidate-output-selection)
  fi
  if [ "$recall_oriented_completion" = "1" ]; then
    cmd+=(--recall-oriented-completion)
  else
    cmd+=(--no-recall-oriented-completion)
  fi
  if [ "$expansion_use_intermediate_answer" = "1" ]; then
    cmd+=(--expansion-use-intermediate-answer)
  else
    cmd+=(--no-expansion-use-intermediate-answer)
  fi
  if [ "$FORCE_V3" = "1" ]; then
    cmd+=(--force-revise)
  fi
  if [ "$qampari_answer_relevance_filter" = "1" ]; then
    cmd+=(--qampari-answer-relevance-filter)
  else
    cmd+=(--no-qampari-answer-relevance-filter)
  fi
  if [ "$FORCE_EVAL" = "1" ]; then
    cmd+=(--force-eval)
  fi
  if [ "$FORCE_MERGE" = "1" ]; then
    cmd+=(--force-merge)
  fi
  if [ "$NO_CITATIONS" = "1" ]; then
    cmd+=(--no-citations)
  fi
  if [ "$NO_MAUVE" = "1" ]; then
    cmd+=(--no-mauve)
  fi
  if [ "$SKIP_EVAL" = "1" ]; then
    cmd+=(--skip-eval)
  fi
  run_step "Stage 2/5: run v3 for $name" "${cmd[@]}"
}

run_qampari_select_one() {
  local name="$1"
  local raw="$2"
  local docs="$3"
  local stem
  stem="$(basename "$raw" .json)"
  local out_json="$V3_DIR/${stem}.cover_v3.json"
  echo "Dataset $name settings: QAMPARI answer-set selector output=$out_json"

  if [ "$FORCE_V3" = "1" ] || [ ! -f "$out_json" ]; then
    local cmd=(
      "$PYTHON" scripts/cover_qampari_select.py
      --input "$raw"
      --output "$out_json"
      --openai-api
      --selection-mode "$QAMPARI_SELECT_SELECTION_MODE"
      --joint-model "$QAMPARI_SELECT_JOINT_MODEL"
      --joint-docs "$QAMPARI_SELECT_JOINT_DOCS"
      --joint-candidate-spans "$QAMPARI_SELECT_JOINT_CANDIDATE_SPANS"
      --item-verifier "$QAMPARI_SELECT_ITEM_VERIFIER"
      --item-verify-model "$QAMPARI_SELECT_ITEM_VERIFY_MODEL"
      --expand-from-docs "$QAMPARI_SELECT_EXPAND_FROM_DOCS"
      --expansion-model "$QAMPARI_SELECT_EXPANSION_MODEL"
      --candidate-docs-file "$docs"
      --max-doc-pool "$MAX_DOC_POOL"
      --repair-top-k "$QAMPARI_SELECT_REPAIR_TOP_K"
      --repair-group-size 3
      --expansion-docs "$QAMPARI_SELECT_EXPANSION_DOCS"
      --max-expansion-candidates "$QAMPARI_SELECT_MAX_EXPANSION_CANDIDATES"
      --max-output-items "$QAMPARI_SELECT_MAX_OUTPUT_ITEMS"
      --backfill-original-until "$QAMPARI_SELECT_BACKFILL_ORIGINAL_UNTIL"
      --preserve-original-top-k "$QAMPARI_SELECT_PRESERVE_ORIGINAL_TOP_K"
      --preserved-original-label "$QAMPARI_SELECT_PRESERVED_ORIGINAL_LABEL"
      --empty-output-policy "$QAMPARI_SELECT_EMPTY_OUTPUT_POLICY"
      --max-citations-per-item "$QAMPARI_SELECT_MAX_CITATIONS_PER_ITEM"
      --cache-file "cache/cover_qampari_select_${CACHE_NAMESPACE}${name}_cache.json"
      --cache-save-every "$CACHE_SAVE_EVERY"
      --pretty
    )
    if [ "$QAMPARI_SELECT_JOINT_REQUIRE_TYPE_MATCH" = "1" ]; then
      cmd+=(--joint-require-type-match)
    else
      cmd+=(--no-joint-require-type-match)
    fi
    if [ "$QAMPARI_SELECT_STRICT_QUESTION_TYPE_GUARD" = "1" ]; then
      cmd+=(--strict-question-type-guard)
    else
      cmd+=(--no-strict-question-type-guard)
    fi
    if [ "$QAMPARI_SELECT_BACKFILL_REQUIRES_TYPE_SAFE" = "1" ]; then
      cmd+=(--backfill-requires-type-safe)
    else
      cmd+=(--no-backfill-requires-type-safe)
    fi
    if [ "$QAMPARI_SELECT_KEEP_UNSUPPORTED_ORIGINAL" = "1" ]; then
      cmd+=(--keep-unsupported-original)
    else
      cmd+=(--no-keep-unsupported-original)
    fi
    if [ "$QAMPARI_SELECT_RESTORE_ORIGINAL_SUBSPANS" = "1" ]; then
      cmd+=(--restore-original-subspans)
    else
      cmd+=(--no-restore-original-subspans)
    fi
    if [ "$QAMPARI_SELECT_DROP_OBVIOUS_BAD" = "1" ]; then
      cmd+=(--drop-obvious-bad)
    else
      cmd+=(--no-drop-obvious-bad)
    fi
    if [ -n "$LIMIT" ]; then
      cmd+=(--limit "$LIMIT")
    fi
    run_step "Stage 2/5: run QAMPARI selector for $name" "${cmd[@]}"
  else
    echo "QAMPARI selector output exists, skip: $out_json"
  fi

  if [ ! -f "$out_json" ]; then
    echo "WARNING: QAMPARI selector output missing after run: $out_json"
    return 0
  fi

  local score="$out_json.score"
  if [ "$SKIP_EVAL" = "1" ]; then
    echo "skip QAMPARI eval.py because SKIP_EVAL=1"
  elif [ "$FORCE_EVAL" = "1" ] || [ ! -f "$score" ]; then
    run_step "Stage 2b/5: evaluate QAMPARI selector for $name" \
      "$PYTHON" eval.py --f "$out_json" --citations
  else
    echo "QAMPARI selector score exists, skip: $score"
  fi

  local claim_score="$out_json.claim_citation_score"
  if [ "$FORCE_MERGE" = "1" ] || [ ! -f "$claim_score" ]; then
    run_step "Stage 2c/5: compute QAMPARI selector claim-citation metrics for $name" \
      "$PYTHON" scripts/compute_claim_citation_metrics.py --result "$out_json" --pretty
  else
    echo "QAMPARI selector claim-citation score exists, skip: $claim_score"
  fi

  local cover_score="$out_json.cover_score"
  if [ "$FORCE_MERGE" = "1" ] || [ ! -f "$cover_score" ]; then
    run_step "Stage 2d/5: merge QAMPARI selector score for $name" \
      "$PYTHON" scripts/merge_cover_scores.py \
      --result "$out_json" \
      --pretty \
      --csv "$V3_DIR/${name}_cover_v3_comparison.csv"
  else
    echo "QAMPARI selector merged score exists, skip: $cover_score"
  fi
}

run_v3_one "asqa_gtr" "$ASQA_RAW" "$ASQA_DOCS"
run_v3_one "eli5_bm25" "$ELI5_RAW" "$ELI5_DOCS"
run_v3_one "qampari_gtr" "$QAMPARI_RAW" "$QAMPARI_DOCS"

run_step "Stage 3/5: collect all v3 summary rows" \
  "$PYTHON" scripts/run_cover_v3_batch.py \
  --inputs "${present_inputs[@]}" \
  --output-dir "$V3_DIR" \
  --python "$PYTHON" \
  --summary-only \
  --summary-csv "$V3_DIR/cover_v3_comparison.csv" \
  --summary-md "$V3_DIR/cover_v3_comparison.md"
require_csv_min_data_rows "$V3_DIR/cover_v3_comparison.csv" "${#present_inputs[@]}" "v3 summary"

run_step "Stage 4/5: compare fixed v0 with v3" \
  "$PYTHON" scripts/compare_cover_latest.py \
  --v0 "$V0_COMPARE_CSV" \
  --latest "$V3_DIR/cover_v3_comparison.csv" \
  --latest-name v3 \
  --output-csv "$BASE_DIR/cover_v0_v3_comparison_long.csv" \
  --output-md "$BASE_DIR/cover_v0_v3_comparison_long.md" \
  --wide-csv "$BASE_DIR/cover_v0_v3_comparison_wide.csv" \
  --wide-md "$BASE_DIR/cover_v0_v3_comparison_wide.md"
require_csv_min_data_rows "$BASE_DIR/cover_v0_v3_comparison_long.csv" "$(( ${#present_inputs[@]} * 2 ))" "v0-v3 long comparison"
require_csv_min_data_rows "$BASE_DIR/cover_v0_v3_comparison_wide.csv" "${#present_inputs[@]}" "v0-v3 wide comparison"

RECALL_ARGS=(
  "$PYTHON" scripts/compute_answer_recall_diagnostics.py
  --output-dir "$BASE_DIR/answer_recall"
)
REQUIRED_METRICS_ARGS=(
  "$PYTHON" scripts/summarize_cover_required_metrics.py
  --v0-summary-csv "$V0_COMPARE_CSV"
  --latest-summary-csv "$V3_DIR/cover_v3_comparison.csv"
  --latest-label "v3"
  --answer-recall-summary "$BASE_DIR/answer_recall/answer_recall_summary.csv"
  --output-csv "$BASE_DIR/required_metrics_summary.csv"
  --output-md "$BASE_DIR/required_metrics_summary.md"
  --require-datasets "$REQUIRED_DATASETS_CSV"
  --require-versions "v0,v3"
  --require-filled
)

add_metric_conditions() {
  local dataset="$1"
  local raw="$2"
  if [ ! -f "$raw" ]; then
    return 0
  fi
  local stem
  stem="$(basename "$raw" .json)"
  local v3_json="$V3_DIR/${stem}.cover_v3.json"
  RECALL_ARGS+=(--condition "$dataset:base_raw:$raw")
  if [ -f "$v3_json" ]; then
    RECALL_ARGS+=(--condition "$dataset:v3:$v3_json" --pair "$dataset:base_raw:v3")
  else
    echo "WARNING: skip $dataset v3 recall condition because output is missing: $v3_json"
  fi
}

add_metric_conditions "asqa" "$ASQA_RAW"
add_metric_conditions "eli5" "$ELI5_RAW"
add_metric_conditions "qampari" "$QAMPARI_RAW"

run_step "Stage 5/6: answer recall diagnostics" "${RECALL_ARGS[@]}"
require_csv_min_data_rows "$BASE_DIR/answer_recall/answer_recall_summary.csv" "$(( ${#present_inputs[@]} * 2 ))" "answer recall summary"

run_step "Stage 6/6: fixed required metrics table" "${REQUIRED_METRICS_ARGS[@]}"
require_csv_min_data_rows "$BASE_DIR/required_metrics_summary.csv" "$(( ${#present_inputs[@]} * 2 ))" "required metrics summary"

echo
echo "=== Outputs ==="
echo "v0 summary: $V0_COMPARE_CSV"
echo "v3 summary: $V3_DIR/cover_v3_comparison.csv"
echo "long comparison: $BASE_DIR/cover_v0_v3_comparison_long.csv"
echo "wide comparison: $BASE_DIR/cover_v0_v3_comparison_wide.csv"
echo "answer recall: $BASE_DIR/answer_recall/answer_recall_summary.csv"
echo "required metrics: $BASE_DIR/required_metrics_summary.csv"
echo "Done."
