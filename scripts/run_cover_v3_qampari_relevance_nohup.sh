#!/usr/bin/env bash
# Run the next QAMPARI-only COVER-RAG v3 experiment with answer-relevance filtering.
#
# This is the recommended next run before rerunning all datasets. It isolates
# whether question-aware entity filtering improves QAMPARI F1 while preserving
# claim-level citation quality.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON="${PYTHON:-python}"
RAW="${RAW:-result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
DOCS="${DOCS:-data/qampari_eval_gtr_top100.json}"
OUT_DIR="${OUT_DIR:-result/qampari_gtr_v3_answer_relevance/cover_v3}"
CACHE_FILE="${CACHE_FILE:-cache/cover_v3_qampari_answer_relevance_cache.json}"

DECOMPOSER="${DECOMPOSER:-llm}"
DECOMPOSE_MODEL="${DECOMPOSE_MODEL:-gpt-4o-mini}"
VERIFIER="${VERIFIER:-nli}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/deberta-v3-large-zeroshot-v2.0}"
EXPANSION_MODEL="${EXPANSION_MODEL:-gpt-4o-mini}"

FORCE_REVISE="${FORCE_REVISE:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
FINAL_AUDIT="${FINAL_AUDIT:-1}"
LIMIT="${LIMIT:-}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$OUT_DIR" logs

cmd=(
  "$PYTHON" scripts/run_cover_v3_batch.py
  --inputs "$RAW"
  --output-dir "$OUT_DIR"
  --python "$PYTHON"
  --decomposer "$DECOMPOSER"
  --decompose-model "$DECOMPOSE_MODEL"
  --verifier "$VERIFIER"
  --nli-model "$NLI_MODEL"
  --answer-format qampari
  --evidence-scope cited
  --citation-policy minimal
  --max-citations-per-claim 1
  --max-verified-claims 24
  --max-qampari-items 80
  --max-final-claims 80
  --max-doc-pool 100
  --expansion-mode llm
  --expansion-model "$EXPANSION_MODEL"
  --expansion-evidence-sentences 16
  --expansion-candidate-spans 40
  --max-expansion-candidates 10
  --max-expansion-verifications 6
  --max-expanded-claims 4
  --min-expansion-score 0.10
  --expansion-query-source question_and_evidence_core
  --expansion-include-extractive
  --verify-expansion-with-sentence
  --require-expansion-answer-span
  --qampari-answer-relevance-filter
  --preserve-first-pass-claims
  --cache-file "$CACHE_FILE"
  --summary-csv "$OUT_DIR/cover_v3_comparison.csv"
  --summary-md "$OUT_DIR/cover_v3_comparison.md"
)

if [ -f "$DOCS" ]; then
  cmd+=(--candidate-docs-file "$DOCS")
else
  echo "WARNING: candidate docs file not found, using docs inside raw result: $DOCS"
fi
if [ "$FORCE_REVISE" = "1" ]; then
  cmd+=(--force-revise)
fi
if [ "$FORCE_EVAL" = "1" ]; then
  cmd+=(--force-eval)
fi
if [ "$FORCE_MERGE" = "1" ]; then
  cmd+=(--force-merge)
fi
if [ "$FINAL_AUDIT" = "1" ]; then
  cmd+=(--final-audit)
else
  cmd+=(--no-final-audit)
fi
if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi
if [ "$SKIP_EVAL" = "1" ]; then
  cmd+=(--skip-eval)
fi

echo "=== COVER-RAG v3 QAMPARI answer-relevance run ==="
echo "Root: $ROOT_DIR"
echo "Raw: $RAW"
echo "Output dir: $OUT_DIR"
echo "+ ${cmd[*]}"
"${cmd[@]}"

echo
echo "Outputs:"
echo "  $OUT_DIR/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json"
echo "  $OUT_DIR/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.cover_v3.json.cover_score"
echo "  $OUT_DIR/cover_v3_comparison.csv"
