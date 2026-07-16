#!/usr/bin/env bash
set -uo pipefail

# Run one focused ASQA-GTR experiment:
#   raw ALCE output -> COVER v0 audit -> COVER v1 -> COVER v2 -> compact table
#
# The script is resume-friendly by default:
#   - FORCE_V0/FORCE_V1/FORCE_V2 default to 0, so existing JSON/score files are reused.
#   - A failed stage is reported but does not stop later stages.
#   - Set FAIL_ON_ERROR=1 if you want the script to return non-zero when any stage fails.
#
# If eval.py fails while loading AutoAIS for citation metrics, you can still
# get QA/Mauve and COVER metrics with:
#   EVAL_CITATIONS=0 bash scripts/run_cover_v0_v1_v2_asqa_gtr_nohup.sh
#
# Recommended command:
#   mkdir -p logs
#   nohup bash scripts/run_cover_v0_v1_v2_asqa_gtr_nohup.sh > logs/asqa_gtr_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_NAME="${RAW_NAME:-asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result/asqa_gtr_v0_v1_v2}"
CACHE_DIR="${CACHE_DIR:-cache/asqa_gtr_v0_v1_v2}"

DECOMPOSER="${DECOMPOSER:-llm}"
DECOMPOSE_MODEL="${DECOMPOSE_MODEL:-gpt-4o-mini}"
VERIFIER="${VERIFIER:-nli}"
LLM_VERIFY_MODEL="${LLM_VERIFY_MODEL:-gpt-4o-mini}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"

V0_EVIDENCE_SCOPE="${V0_EVIDENCE_SCOPE:-cited_then_all}"
V1_EVIDENCE_SCOPE="${V1_EVIDENCE_SCOPE:-cited_then_all}"
V2_EVIDENCE_SCOPE="${V2_EVIDENCE_SCOPE:-cited}"

REVISION_MODE="${REVISION_MODE:-auto}"
REVISION_MODEL="${REVISION_MODEL:-gpt-4o-mini}"
V2_RENDER_MODE="${V2_RENDER_MODE:-alce_llm}"
CITATION_POLICY="${CITATION_POLICY:-source}"
MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-3}"
MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-18}"
ALCE_LLM_MAX_DOCS="${ALCE_LLM_MAX_DOCS:-8}"
ALCE_LLM_MAX_CLAIMS="${ALCE_LLM_MAX_CLAIMS:-18}"
ALCE_LLM_MAX_DOC_CHARS="${ALCE_LLM_MAX_DOC_CHARS:-900}"

RECOVERY_TOP_K="${RECOVERY_TOP_K:-6}"
RECOVERY_GROUP_SIZE="${RECOVERY_GROUP_SIZE:-3}"
MIN_RECOVERY_SCORE="${MIN_RECOVERY_SCORE:-0.03}"
MAX_RECOVERED_CITATIONS="${MAX_RECOVERED_CITATIONS:-3}"

FORCE_V0="${FORCE_V0:-0}"
FORCE_V1="${FORCE_V1:-0}"
FORCE_V2="${FORCE_V2:-0}"
FAIL_ON_ERROR="${FAIL_ON_ERROR:-0}"

# v0 does not change the answer, so reusing the raw .json.score is usually
# enough and avoids reloading ALCE citation models.
V0_SCORE_MODE="${V0_SCORE_MODE:-source}"

EVAL_CITATIONS="${EVAL_CITATIONS:-1}"
EVAL_QA="${EVAL_QA:-1}"
EVAL_MAUVE="${EVAL_MAUVE:-1}"

RAW_PATH="${RAW_PATH:-}"
if [[ -z "$RAW_PATH" ]]; then
  if [[ -f "result/origin/$RAW_NAME" ]]; then
    RAW_PATH="result/origin/$RAW_NAME"
  elif [[ -f "result/$RAW_NAME" ]]; then
    RAW_PATH="result/$RAW_NAME"
  else
    echo "ERROR: Cannot find raw file:"
    echo "  result/origin/$RAW_NAME"
    echo "  result/$RAW_NAME"
    echo "Set RAW_PATH=/path/to/$RAW_NAME and rerun."
    exit 2
  fi
fi

CANDIDATE_DOCS_FILE="${CANDIDATE_DOCS_FILE:-}"
if [[ -z "$CANDIDATE_DOCS_FILE" ]]; then
  for candidate in \
    "data/asqa_eval_gtr_top100.json" \
    "data/asqa_eval_gtr_top100_bge_reranked.json" \
    "result/raw/asqa_eval_gtr_top100.json"
  do
    if [[ -f "$candidate" ]]; then
      CANDIDATE_DOCS_FILE="$candidate"
      break
    fi
  done
fi

AUDIT_DIR="$OUTPUT_ROOT/cover_audit"
V1_DIR="$OUTPUT_ROOT/cover_v1"
V2_DIR="$OUTPUT_ROOT/cover_v2"
mkdir -p logs "$AUDIT_DIR" "$V1_DIR" "$V2_DIR" "$CACHE_DIR"

FAILURES=()

is_truthy() {
  case "${1:-}" in
    1|true|yes|y|TRUE|YES|Y) return 0 ;;
    *) return 1 ;;
  esac
}

record_failure() {
  local message="$1"
  FAILURES+=("$message")
  echo
  echo "WARNING: $message"
  echo "Continuing with the next stage."
}

echo "=== ASQA-GTR COVER-RAG v0/v1/v2 focused run ==="
echo "Python: $PYTHON_BIN"
echo "Raw input: $RAW_PATH"
echo "Output root: $OUTPUT_ROOT"
echo "Verifier: $VERIFIER"
echo "Decomposer: $DECOMPOSER"
echo "Force v0/v1/v2: $FORCE_V0 / $FORCE_V1 / $FORCE_V2"
echo "V0 score mode: $V0_SCORE_MODE"
echo "Eval citations/QA/Mauve: $EVAL_CITATIONS / $EVAL_QA / $EVAL_MAUVE"
echo "V2 render mode: $V2_RENDER_MODE"
if [[ -n "$CANDIDATE_DOCS_FILE" ]]; then
  echo "V2 candidate docs: $CANDIDATE_DOCS_FILE"
else
  echo "V2 candidate docs: not found; v2 will recover only from docs already stored in the raw result JSON."
fi

V0_FORCE_ARGS=()
if is_truthy "$FORCE_V0"; then
  V0_FORCE_ARGS=(--force-cover --force-eval --force-merge)
fi

echo
echo "=== Stage 1/4: v0 post-hoc audit ==="
if ! "$PYTHON_BIN" scripts/run_cover_audit_batch.py \
  --python "$PYTHON_BIN" \
  --inputs "$RAW_PATH" \
  --output-dir "$AUDIT_DIR" \
  --summary-csv "$AUDIT_DIR/cover_audit_comparison.csv" \
  --summary-md "$AUDIT_DIR/cover_audit_comparison.md" \
  --score-mode "$V0_SCORE_MODE" \
  --decomposer "$DECOMPOSER" \
  --decompose-model "$DECOMPOSE_MODEL" \
  --verifier "$VERIFIER" \
  --llm-verify-model "$LLM_VERIFY_MODEL" \
  --nli-model "$NLI_MODEL" \
  --evidence-scope "$V0_EVIDENCE_SCOPE" \
  --cache-file "$CACHE_DIR/cover_audit_cache.json" \
  "${V0_FORCE_ARGS[@]}"
then
  record_failure "Stage 1 v0 audit/eval/merge failed."
fi

V1_FORCE_ARGS=()
if is_truthy "$FORCE_V1"; then
  V1_FORCE_ARGS=(--force-revise --force-eval --force-merge)
fi
V1_EVAL_ARGS=()
if ! is_truthy "$EVAL_CITATIONS"; then
  V1_EVAL_ARGS+=(--no-citations)
fi
if ! is_truthy "$EVAL_QA"; then
  V1_EVAL_ARGS+=(--no-qa)
fi
if ! is_truthy "$EVAL_MAUVE"; then
  V1_EVAL_ARGS+=(--no-mauve)
fi

echo
echo "=== Stage 2/4: v1 conservative revision ==="
if ! "$PYTHON_BIN" scripts/run_cover_v1_batch.py \
  --python "$PYTHON_BIN" \
  --inputs "$RAW_PATH" \
  --output-dir "$V1_DIR" \
  --summary-csv "$V1_DIR/cover_v1_comparison.csv" \
  --summary-md "$V1_DIR/cover_v1_comparison.md" \
  --audit-summary-csv "$AUDIT_DIR/cover_audit_comparison.csv" \
  --combined-csv "$V1_DIR/cover_v1_vs_audit_comparison.csv" \
  --combined-md "$V1_DIR/cover_v1_vs_audit_comparison.md" \
  --decomposer "$DECOMPOSER" \
  --decompose-model "$DECOMPOSE_MODEL" \
  --verifier "$VERIFIER" \
  --llm-verify-model "$LLM_VERIFY_MODEL" \
  --nli-model "$NLI_MODEL" \
  --revision-mode "$REVISION_MODE" \
  --revision-model "$REVISION_MODEL" \
  --answer-format auto \
  --citation-policy "$CITATION_POLICY" \
  --max-citations-per-claim "$MAX_CITATIONS_PER_CLAIM" \
  --max-verified-claims "$MAX_VERIFIED_CLAIMS" \
  --evidence-scope "$V1_EVIDENCE_SCOPE" \
  --cache-file "$CACHE_DIR/cover_v1_cache.json" \
  --no-allow-citation-repair \
  --no-include-background-claims \
  --final-audit \
  "${V1_EVAL_ARGS[@]}" \
  "${V1_FORCE_ARGS[@]}"
then
  record_failure "Stage 2 v1 revise/eval/merge failed."
fi

V2_FORCE_ARGS=()
if is_truthy "$FORCE_V2"; then
  V2_FORCE_ARGS=(--force-revise --force-eval --force-merge)
fi
V2_CANDIDATE_ARGS=()
if [[ -n "$CANDIDATE_DOCS_FILE" ]]; then
  V2_CANDIDATE_ARGS=(--candidate-docs-file "$CANDIDATE_DOCS_FILE")
fi
V2_EVAL_ARGS=()
if ! is_truthy "$EVAL_CITATIONS"; then
  V2_EVAL_ARGS+=(--no-citations)
fi
if ! is_truthy "$EVAL_QA"; then
  V2_EVAL_ARGS+=(--no-qa)
fi
if ! is_truthy "$EVAL_MAUVE"; then
  V2_EVAL_ARGS+=(--no-mauve)
fi

echo
echo "=== Stage 3/4: v2 targeted evidence recovery ==="
if ! "$PYTHON_BIN" scripts/run_cover_v2_batch.py \
  --python "$PYTHON_BIN" \
  --inputs "$RAW_PATH" \
  --output-dir "$V2_DIR" \
  --summary-csv "$V2_DIR/cover_v2_comparison.csv" \
  --summary-md "$V2_DIR/cover_v2_comparison.md" \
  --decomposer "$DECOMPOSER" \
  --decompose-model "$DECOMPOSE_MODEL" \
  --verifier "$VERIFIER" \
  --llm-verify-model "$LLM_VERIFY_MODEL" \
  --nli-model "$NLI_MODEL" \
  --revision-mode "$REVISION_MODE" \
  --v2-render-mode "$V2_RENDER_MODE" \
  --revision-model "$REVISION_MODEL" \
  --alce-llm-max-docs "$ALCE_LLM_MAX_DOCS" \
  --alce-llm-max-claims "$ALCE_LLM_MAX_CLAIMS" \
  --alce-llm-max-doc-chars "$ALCE_LLM_MAX_DOC_CHARS" \
  --answer-format auto \
  --evidence-scope "$V2_EVIDENCE_SCOPE" \
  --citation-policy "$CITATION_POLICY" \
  --max-citations-per-claim "$MAX_CITATIONS_PER_CLAIM" \
  --max-verified-claims "$MAX_VERIFIED_CLAIMS" \
  --recover-labels not_supported,no_citation,wrong_or_missing_citation \
  --recover-importance critical,supporting \
  --recovery-top-k "$RECOVERY_TOP_K" \
  --recovery-group-size "$RECOVERY_GROUP_SIZE" \
  --min-recovery-score "$MIN_RECOVERY_SCORE" \
  --max-recovered-citations "$MAX_RECOVERED_CITATIONS" \
  --cache-file "$CACHE_DIR/cover_v2_cache.json" \
  --final-audit \
  "${V2_EVAL_ARGS[@]}" \
  "${V2_CANDIDATE_ARGS[@]}" \
  "${V2_FORCE_ARGS[@]}"
then
  record_failure "Stage 3 v2 revise/eval/merge failed."
fi

echo
echo "=== Stage 4/4: v0/v1/v2 compact comparison ==="
if ! "$PYTHON_BIN" scripts/compare_cover_versions.py \
  --v0 "$AUDIT_DIR/cover_audit_comparison.csv" \
  --v1 "$V1_DIR/cover_v1_comparison.csv" \
  --v2 "$V2_DIR/cover_v2_comparison.csv" \
  --output-csv "$OUTPUT_ROOT/cover_v0_v1_v2_comparison_long.csv" \
  --output-md "$OUTPUT_ROOT/cover_v0_v1_v2_comparison_long.md" \
  --wide-csv "$OUTPUT_ROOT/cover_v0_v1_v2_comparison_wide.csv" \
  --wide-md "$OUTPUT_ROOT/cover_v0_v1_v2_comparison_wide.md"
then
  record_failure "Stage 4 v0/v1/v2 comparison failed."
fi

echo
echo "Done."
echo "Long table: $OUTPUT_ROOT/cover_v0_v1_v2_comparison_long.csv"
echo "Wide table: $OUTPUT_ROOT/cover_v0_v1_v2_comparison_wide.csv"

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  echo
  echo "Completed with ${#FAILURES[@]} warning(s):"
  for failure in "${FAILURES[@]}"; do
    echo "  - $failure"
  done
  if is_truthy "$FAIL_ON_ERROR"; then
    exit 1
  fi
fi

exit 0
