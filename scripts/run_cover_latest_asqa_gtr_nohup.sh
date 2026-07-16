#!/usr/bin/env bash
# Resume-friendly runner for the current COVER-RAG ASQA-GTR experiment.
#
# It treats v0 as a fixed baseline. By default, v0 is reused when its
# cover_score already exists; only the latest method, v3, needs to be rerun
# while you iterate on the method.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON="${PYTHON:-python}"
RAW_RESULT="${RAW_RESULT:-result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
BASE_DIR="${BASE_DIR:-result/asqa_gtr_cover_latest}"
V0_DIR="${V0_DIR:-$BASE_DIR/cover_audit}"
V3_DIR="${V3_DIR:-$BASE_DIR/cover_v3}"

DECOMPOSER="${DECOMPOSER:-llm}"
DECOMPOSE_MODEL="${DECOMPOSE_MODEL:-gpt-4o-mini}"
VERIFIER="${VERIFIER:-nli}"
LLM_VERIFY_MODEL="${LLM_VERIFY_MODEL:-gpt-4o-mini}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"
EVIDENCE_SCOPE="${EVIDENCE_SCOPE:-cited}"

EXPANSION_MODE="${EXPANSION_MODE:-llm}"
EXPANSION_MODEL="${EXPANSION_MODEL:-gpt-4o-mini}"
EXPANSION_EVIDENCE_SENTENCES="${EXPANSION_EVIDENCE_SENTENCES:-16}"
EXPANSION_CANDIDATE_SPANS="${EXPANSION_CANDIDATE_SPANS:-40}"
EXPANSION_INCLUDE_EXTRACTIVE="${EXPANSION_INCLUDE_EXTRACTIVE:-1}"
VERIFY_EXPANSION_WITH_SENTENCE="${VERIFY_EXPANSION_WITH_SENTENCE:-1}"
MAX_EXPANSION_CANDIDATES="${MAX_EXPANSION_CANDIDATES:-10}"
MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-8}"
MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-6}"
MIN_EXPANSION_SCORE="${MIN_EXPANSION_SCORE:-0.06}"

MAX_DOC_POOL="${MAX_DOC_POOL:-100}"
MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-24}"
MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-30}"
MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-3}"
RENDER_MODE="${RENDER_MODE:-citation_aligned}"
LIMIT="${LIMIT:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
REVISION_WORKERS="${REVISION_WORKERS:-4}"
CACHE_SAVE_EVERY="${CACHE_SAVE_EVERY:-50}"
FINAL_AUDIT="${FINAL_AUDIT:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"
NO_CITATIONS="${NO_CITATIONS:-0}"
NO_MAUVE="${NO_MAUVE:-0}"
NO_QA="${NO_QA:-0}"

FORCE_V0="${FORCE_V0:-0}"
FORCE_V3="${FORCE_V3:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
V0_SCORE_MODE="${V0_SCORE_MODE:-source}"
SKIP_V0="${SKIP_V0:-0}"
V0_SUMMARY_CSV="${V0_SUMMARY_CSV:-}"

CANDIDATE_DOCS_FILE="${CANDIDATE_DOCS_FILE:-data/asqa_eval_gtr_top100.json}"
if [ -n "$CANDIDATE_DOCS_FILE" ] && [ ! -f "$CANDIDATE_DOCS_FILE" ]; then
  echo "Candidate docs file not found: $CANDIDATE_DOCS_FILE"
  echo "Falling back to docs inside the raw result JSON."
  CANDIDATE_DOCS_FILE=""
fi

RAW_STEM="$(basename "$RAW_RESULT" .json)"
V0_JSON="$V0_DIR/${RAW_STEM}.cover_audit.json"
V0_COVER_SCORE="$V0_JSON.cover_score"
V3_JSON="$V3_DIR/${RAW_STEM}.cover_v3.json"
V3_COVER_SCORE="$V3_JSON.cover_score"

echo "=== COVER-RAG latest ASQA-GTR run ==="
echo "Root: $ROOT_DIR"
echo "Python: $PYTHON"
echo "Raw result: $RAW_RESULT"
echo "Base dir: $BASE_DIR"
echo "V0 dir: $V0_DIR"
echo "V3 dir: $V3_DIR"
echo "Skip v0: $SKIP_V0"
echo "External v0 summary: ${V0_SUMMARY_CSV:-<none>}"
echo "Decomposer: $DECOMPOSER"
echo "Verifier: $VERIFIER"
echo "Expansion mode: $EXPANSION_MODE"
echo "Expansion evidence sentences: $EXPANSION_EVIDENCE_SENTENCES"
echo "Max expansion candidates: $MAX_EXPANSION_CANDIDATES"
echo "Max expansion verifications: $MAX_EXPANSION_VERIFICATIONS"
echo "Max expanded claims: $MAX_EXPANDED_CLAIMS"
echo "Verify expansion with sentence: $VERIFY_EXPANSION_WITH_SENTENCE"
echo "Render mode: $RENDER_MODE"
echo "Candidate docs: ${CANDIDATE_DOCS_FILE:-<raw-result-docs-only>}"
echo "Limit: ${LIMIT:-<full>}"
echo "Final audit: $FINAL_AUDIT"
echo "Skip eval: $SKIP_EVAL"
echo "Revision workers: $REVISION_WORKERS"

mkdir -p "$V0_DIR" "$V3_DIR" "$BASE_DIR"
V0_COMPARE_CSV="$V0_DIR/cover_audit_comparison.csv"

run_step() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  echo "+ $*"
  "$@"
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "WARNING: $label failed with exit code $status. Continuing."
  fi
  return 0
}

V0_ARGS=(
  "$PYTHON" scripts/run_cover_audit_batch.py
  --inputs "$RAW_RESULT"
  --output-dir "$V0_DIR"
  --python "$PYTHON"
  --score-mode "$V0_SCORE_MODE"
  --decomposer "$DECOMPOSER"
  --decompose-model "$DECOMPOSE_MODEL"
  --verifier "$VERIFIER"
  --llm-verify-model "$LLM_VERIFY_MODEL"
  --nli-model "$NLI_MODEL"
  --evidence-scope cited_then_all
  --cache-file cache/cover_audit_cache.json
  --summary-csv "$V0_DIR/cover_audit_comparison.csv"
  --summary-md "$V0_DIR/cover_audit_comparison.md"
)
if [ "$FORCE_V0" = "1" ]; then
  V0_ARGS+=(--force-cover)
fi
if [ "$FORCE_EVAL" = "1" ]; then
  V0_ARGS+=(--force-eval)
fi
if [ "$FORCE_MERGE" = "1" ]; then
  V0_ARGS+=(--force-merge)
fi

if [ -n "$V0_SUMMARY_CSV" ] && [ -f "$V0_SUMMARY_CSV" ]; then
  echo
  echo "=== Stage 1/4: fixed v0 baseline ==="
  echo "Using external v0 summary, skip v0 audit: $V0_SUMMARY_CSV"
  V0_COMPARE_CSV="$V0_SUMMARY_CSV"
elif [ "$SKIP_V0" = "1" ]; then
  echo
  echo "=== Stage 1/4: fixed v0 baseline ==="
  echo "SKIP_V0=1, skip v0 audit."
  if [ -f "$V0_COMPARE_CSV" ]; then
    echo "Using existing v0 summary: $V0_COMPARE_CSV"
  else
    echo "WARNING: v0 summary not found: $V0_COMPARE_CSV"
    echo "Comparison will fail unless you pass V0_SUMMARY_CSV=/path/to/cover_audit_comparison.csv."
  fi
elif [ "$FORCE_V0" = "1" ] || [ ! -f "$V0_COVER_SCORE" ]; then
  run_step "Stage 1/4: prepare or reuse fixed v0 baseline" "${V0_ARGS[@]}"
else
  echo
  echo "=== Stage 1/4: fixed v0 baseline ==="
  echo "v0 cover_score exists, skip: $V0_COVER_SCORE"
  if [ ! -f "$V0_DIR/cover_audit_comparison.csv" ]; then
    V0_SUMMARY_ARGS=("${V0_ARGS[@]}" --summary-only)
    run_step "Stage 1b/4: rebuild v0 summary from existing cover_score" "${V0_SUMMARY_ARGS[@]}"
  fi
fi

V3_ARGS=(
  "$PYTHON" scripts/run_cover_v3_batch.py
  --inputs "$RAW_RESULT"
  --output-dir "$V3_DIR"
  --python "$PYTHON"
  --decomposer "$DECOMPOSER"
  --decompose-model "$DECOMPOSE_MODEL"
  --verifier "$VERIFIER"
  --llm-verify-model "$LLM_VERIFY_MODEL"
  --nli-model "$NLI_MODEL"
  --evidence-scope "$EVIDENCE_SCOPE"
  --citation-policy source
  --max-citations-per-claim "$MAX_CITATIONS_PER_CLAIM"
  --max-verified-claims "$MAX_VERIFIED_CLAIMS"
  --max-final-claims "$MAX_FINAL_CLAIMS"
  --render-mode "$RENDER_MODE"
  --max-doc-pool "$MAX_DOC_POOL"
  --expansion-mode "$EXPANSION_MODE"
  --expansion-model "$EXPANSION_MODEL"
  --expansion-evidence-sentences "$EXPANSION_EVIDENCE_SENTENCES"
  --expansion-candidate-spans "$EXPANSION_CANDIDATE_SPANS"
  --max-expansion-candidates "$MAX_EXPANSION_CANDIDATES"
  --max-expansion-verifications "$MAX_EXPANSION_VERIFICATIONS"
  --max-expanded-claims "$MAX_EXPANDED_CLAIMS"
  --min-expansion-score "$MIN_EXPANSION_SCORE"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --revision-workers "$REVISION_WORKERS"
  --cache-save-every "$CACHE_SAVE_EVERY"
  --cache-file cache/cover_v3_cache.json
  --summary-csv "$V3_DIR/cover_v3_comparison.csv"
  --summary-md "$V3_DIR/cover_v3_comparison.md"
)
if [ -n "$CANDIDATE_DOCS_FILE" ]; then
  V3_ARGS+=(--candidate-docs-file "$CANDIDATE_DOCS_FILE")
fi
if [ -n "$LIMIT" ]; then
  V3_ARGS+=(--limit "$LIMIT")
fi
if [ "$FINAL_AUDIT" = "1" ]; then
  V3_ARGS+=(--final-audit)
else
  V3_ARGS+=(--no-final-audit)
fi
if [ "$SKIP_EVAL" = "1" ]; then
  V3_ARGS+=(--skip-eval)
fi
if [ "$EXPANSION_INCLUDE_EXTRACTIVE" = "1" ]; then
  V3_ARGS+=(--expansion-include-extractive)
else
  V3_ARGS+=(--no-expansion-include-extractive)
fi
if [ "$VERIFY_EXPANSION_WITH_SENTENCE" = "1" ]; then
  V3_ARGS+=(--verify-expansion-with-sentence)
else
  V3_ARGS+=(--no-verify-expansion-with-sentence)
fi
if [ "$NO_CITATIONS" = "1" ]; then
  V3_ARGS+=(--no-citations)
fi
if [ "$NO_MAUVE" = "1" ]; then
  V3_ARGS+=(--no-mauve)
fi
if [ "$NO_QA" = "1" ]; then
  V3_ARGS+=(--no-qa)
fi
if [ "$FORCE_V3" = "1" ]; then
  V3_ARGS+=(--force-revise)
fi
if [ "$FORCE_EVAL" = "1" ]; then
  V3_ARGS+=(--force-eval)
fi
if [ "$FORCE_MERGE" = "1" ]; then
  V3_ARGS+=(--force-merge)
fi

if [ "$FORCE_V3" = "1" ] || [ ! -f "$V3_COVER_SCORE" ]; then
  run_step "Stage 2/4: run latest COVER-RAG v3" "${V3_ARGS[@]}"
else
  echo
  echo "=== Stage 2/4: latest COVER-RAG v3 ==="
  echo "v3 cover_score exists, skip: $V3_COVER_SCORE"
  if [ ! -f "$V3_DIR/cover_v3_comparison.csv" ]; then
    V3_SUMMARY_ARGS=("${V3_ARGS[@]}" --summary-only)
    run_step "Stage 2b/4: rebuild v3 summary from existing cover_score" "${V3_SUMMARY_ARGS[@]}"
  fi
fi

run_step "Stage 3/4: build v0 vs v3 comparison" \
  "$PYTHON" scripts/compare_cover_latest.py \
  --v0 "$V0_COMPARE_CSV" \
  --latest "$V3_DIR/cover_v3_comparison.csv" \
  --latest-name v3 \
  --output-csv "$BASE_DIR/cover_v0_v3_comparison_long.csv" \
  --output-md "$BASE_DIR/cover_v0_v3_comparison_long.md" \
  --wide-csv "$BASE_DIR/cover_v0_v3_comparison_wide.csv" \
  --wide-md "$BASE_DIR/cover_v0_v3_comparison_wide.md"

echo
echo "=== Stage 4/4: outputs ==="
echo "v0 summary: $V0_DIR/cover_audit_comparison.csv"
echo "v3 summary: $V3_DIR/cover_v3_comparison.csv"
echo "long comparison: $BASE_DIR/cover_v0_v3_comparison_long.csv"
echo "wide comparison: $BASE_DIR/cover_v0_v3_comparison_wide.csv"
echo "Done."
