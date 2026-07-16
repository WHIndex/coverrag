#!/usr/bin/env bash
# Oracle-evidence diagnostic for three plug-in settings:
#   1) vanilla RAG + COVER-RAG v13
#   2) BGE-rerank RAG + COVER-RAG v13
#   3) API Self-RAG-style + COVER-RAG v13
#
# This is an upper-bound/diagnostic run.  It keeps each method's base output
# fixed, rebuilds v0 audit for that base output, and gives COVER-RAG the
# oracle-reranked candidate evidence files.  The oracle files are not a fair
# deployable retriever; they are used to test whether evidence quality is the
# bottleneck.
#
# Launch:
#   nohup bash scripts/run_oracle_evidence_cover_v13_three_methods_nohup.sh \
#     > logs/oracle_evidence_cover_v13_three_methods_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

export PYTHON="${PYTHON:-/home/anaconda/envs/alce/bin/python}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-result/oracle_evidence_cover_v13_three_methods}"
mkdir -p "$EXPERIMENT_ROOT" logs cache

ORACLE_ASQA_DOCS="${ORACLE_ASQA_DOCS:-data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORACLE_ELI5_DOCS="${ORACLE_ELI5_DOCS:-data/eli5_eval_bm25_top100_reranked_oracle.json}"
ORACLE_QAMPARI_DOCS="${ORACLE_QAMPARI_DOCS:-data/qampari_eval_gtr_top100_reranked_oracle.json}"

run_step() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  echo "+ $*"
  "$@"
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "FAILED: $label exited with $status"
    exit "$status"
  fi
}

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "FAILED: missing or empty file: $path"
    exit 1
  fi
}

require_file "$ORACLE_ASQA_DOCS"
require_file "$ORACLE_ELI5_DOCS"
require_file "$ORACLE_QAMPARI_DOCS"

echo "=== Oracle-evidence COVER-RAG v13 diagnostic ==="
echo "Root: $ROOT_DIR"
echo "Output root: $EXPERIMENT_ROOT"
echo "OpenAI base: $OPENAI_BASE_URL"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set. LLM-based steps will fail."
fi
echo "Oracle candidate docs:"
printf '  - %s\n' "$ORACLE_ASQA_DOCS" "$ORACLE_ELI5_DOCS" "$ORACLE_QAMPARI_DOCS"

run_step "Vanilla RAG v0/v3 with oracle evidence" bash -c "
  set -uo pipefail
  export EXPERIMENT_NAME=oracle_evidence_vanilla_cover_v13
  export USE_EXTERNAL_BASE_DIR=1
  export BASE_DIR='$EXPERIMENT_ROOT/vanilla'
  export USE_EXTERNAL_CACHE_NAMESPACE=1
  export CACHE_NAMESPACE=oracle_evidence_vanilla_v13_
  export ASQA_RAW=result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json
  export ELI5_RAW=result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json
  export QAMPARI_RAW=result/origin/qampari-gpt-4o-mini-gtr-shot2-ndoc5-42.json
  export ASQA_DOCS='$ORACLE_ASQA_DOCS'
  export ELI5_DOCS='$ORACLE_ELI5_DOCS'
  export QAMPARI_DOCS='$ORACLE_QAMPARI_DOCS'
  export PREPARE_V0=1
  export V0_SUMMARY_CSV='$EXPERIMENT_ROOT/vanilla/cover_audit/cover_audit_comparison.csv'
  export V0_DIR='$EXPERIMENT_ROOT/vanilla/cover_audit'
  export V0_CACHE_FILE=cache/cover_audit_oracle_evidence_vanilla_v13_cache.json
  export FORCE_V0='${FORCE_V0:-0}'
  export FORCE_V3='${FORCE_V3:-1}'
  export FORCE_EVAL='${FORCE_EVAL:-1}'
  export FORCE_MERGE='${FORCE_MERGE:-1}'
  bash scripts/run_cover_v3_recall_repair_nohup.sh
"

run_step "BGE-rerank RAG v0/v3 with oracle evidence" bash -c "
  set -uo pipefail
  export EXPERIMENT_NAME=oracle_evidence_bge_rerank_cover_v13
  export BASE_DIR='$EXPERIMENT_ROOT/bge_rerank'
  export CACHE_NAMESPACE=oracle_evidence_bge_rerank_v13_
  export ASQA_RAW=result/origin/asqa-gpt-4o-mini-bge_rerank_default-shot2-ndoc5-42.json
  export ELI5_RAW=result/origin/eli5-gpt-4o-mini-bge_rerank_default-shot2-ndoc5-42.json
  export QAMPARI_RAW=result/origin/qampari-gpt-4o-mini-bge_rerank_default-shot2-ndoc5-42.json
  export ASQA_DOCS='$ORACLE_ASQA_DOCS'
  export ELI5_DOCS='$ORACLE_ELI5_DOCS'
  export QAMPARI_DOCS='$ORACLE_QAMPARI_DOCS'
  export RERANK_RAG_PREPARE_V0=1
  export RERANK_RAG_V0_SUMMARY_CSV='$EXPERIMENT_ROOT/bge_rerank/cover_audit/cover_audit_comparison.csv'
  export V0_DIR='$EXPERIMENT_ROOT/bge_rerank/cover_audit'
  export V0_CACHE_FILE=cache/cover_audit_oracle_evidence_bge_rerank_v13_cache.json
  export FORCE_V0='${FORCE_V0:-0}'
  export FORCE_V3='${FORCE_V3:-1}'
  export FORCE_EVAL='${FORCE_EVAL:-1}'
  export FORCE_MERGE='${FORCE_MERGE:-1}'
  bash scripts/run_bge_rerank_rag_cover_nohup.sh
"

run_step "API Self-RAG v0/v3 with oracle evidence" bash -c "
  set -uo pipefail
  export EXPERIMENT_NAME=oracle_evidence_self_rag_cover_v13
  export BASE_DIR='$EXPERIMENT_ROOT/self_rag'
  export RAW_DIR='$EXPERIMENT_ROOT/self_rag/self_rag_origin'
  export CACHE_NAMESPACE=oracle_evidence_self_rag_v13_
  export SELF_RAG_ASQA_OUTPUT=result/api_self_rag_cover_v13_full/api_self_rag_raw/asqa-api-self-rag.json
  export SELF_RAG_ELI5_OUTPUT=result/api_self_rag_cover_v13_full/api_self_rag_raw/eli5-api-self-rag.json
  export SELF_RAG_QAMPARI_OUTPUT=result/api_self_rag_cover_v13_full/api_self_rag_raw/qampari-api-self-rag.json
  export ASQA_DOCS='$ORACLE_ASQA_DOCS'
  export ELI5_DOCS='$ORACLE_ELI5_DOCS'
  export QAMPARI_DOCS='$ORACLE_QAMPARI_DOCS'
  export SELF_RAG_PREPARE_V0=1
  export SELF_RAG_V0_SUMMARY_CSV='$EXPERIMENT_ROOT/self_rag/cover_audit/cover_audit_comparison.csv'
  export V0_DIR='$EXPERIMENT_ROOT/self_rag/cover_audit'
  export V0_CACHE_FILE=cache/cover_audit_oracle_evidence_self_rag_v13_cache.json
  export FORCE_V0='${FORCE_V0:-0}'
  export FORCE_V3='${FORCE_V3:-1}'
  export FORCE_EVAL='${FORCE_EVAL:-1}'
  export FORCE_MERGE='${FORCE_MERGE:-1}'
  bash scripts/run_cover_v3_self_rag_nohup.sh
"

echo
echo "=== Build all-method oracle summary ==="
"$PYTHON" - "$EXPERIMENT_ROOT" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = [
    ("vanilla", root / "vanilla" / "required_metrics_summary.csv"),
    ("bge_rerank", root / "bge_rerank" / "required_metrics_summary.csv"),
    ("self_rag", root / "self_rag" / "required_metrics_summary.csv"),
]
fields = [
    "method",
    "dataset",
    "version",
    "main_correctness",
    "main_recall",
    "sentence_citation_recall",
    "sentence_citation_precision",
    "claim_citation_recall",
    "claim_citation_precision",
    "claim_sentence_recall",
    "claim_support",
    "source",
]
rows = []
for method, path in methods:
    if not path.exists():
        print(f"WARNING: missing {path}")
        continue
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            version = row.get("version", "")
            if version not in {"v0", "v3", "recall_repair"}:
                continue
            out = {field: row.get(field, "") for field in fields if field != "method"}
            out["method"] = method
            if out["version"] == "recall_repair":
                out["version"] = "v3"
            out["source"] = str(path)
            rows.append(out)

output_csv = root / "oracle_evidence_required_metrics_all_methods.csv"
with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})

output_md = root / "oracle_evidence_required_metrics_all_methods.md"
lines = [
    "| " + " | ".join(fields) + " |",
    "| " + " | ".join(["---"] * len(fields)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {output_csv}")
print(f"Wrote {output_md}")
PY

echo
echo "=== Done ==="
echo "Vanilla required metrics: $EXPERIMENT_ROOT/vanilla/required_metrics_summary.csv"
echo "BGE-rerank required metrics: $EXPERIMENT_ROOT/bge_rerank/required_metrics_summary.csv"
echo "Self-RAG required metrics: $EXPERIMENT_ROOT/self_rag/required_metrics_summary.csv"
echo "All-method summary: $EXPERIMENT_ROOT/oracle_evidence_required_metrics_all_methods.csv"
