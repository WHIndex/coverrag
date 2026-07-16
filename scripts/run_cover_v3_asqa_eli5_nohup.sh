#!/usr/bin/env bash
# Run COVER-RAG v3 on the two datasets that best match claim-level grounding:
# ASQA + GTR and ELI5 + BM25. QAMPARI is intentionally skipped here because its
# main correctness metric is short-answer list recall rather than claim support.
#
# Recommended launch:
#   FORCE_V3=1 FINAL_AUDIT=1 \
#   nohup bash scripts/run_cover_v3_asqa_eli5_nohup.sh \
#     > logs/cover_v3_asqa_eli5_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail

export BASE_DIR="${BASE_DIR:-result/cover_v3_asqa_eli5}"
export ASQA_RAW="${ASQA_RAW:-result/origin/asqa-gpt-4o-mini-gtr-shot2-ndoc5-42.json}"
export ELI5_RAW="${ELI5_RAW:-result/origin/eli5-gpt-4o-mini-bm25-shot2-ndoc5-42.json}"

# Point QAMPARI to a deliberately missing file so the shared runner skips it.
export QAMPARI_RAW="${QAMPARI_RAW:-__skip_qampari_for_asqa_eli5__.json}"

export FINAL_AUDIT="${FINAL_AUDIT:-1}"
export RENDER_MODE="${RENDER_MODE:-answer_preserving}"
export EXPANSION_MODE="${EXPANSION_MODE:-off}"
export MAX_VERIFIED_CLAIMS="${MAX_VERIFIED_CLAIMS:-22}"
export MAX_FINAL_CLAIMS="${MAX_FINAL_CLAIMS:-26}"
export MAX_CITATIONS_PER_CLAIM="${MAX_CITATIONS_PER_CLAIM:-2}"
export MAX_CITATIONS_PER_SENTENCE="${MAX_CITATIONS_PER_SENTENCE:-4}"
export MAX_EXPANSION_VERIFICATIONS="${MAX_EXPANSION_VERIFICATIONS:-5}"
export MAX_EXPANDED_CLAIMS="${MAX_EXPANDED_CLAIMS:-3}"
export PRESERVE_ANSWER_SENTENCES="${PRESERVE_ANSWER_SENTENCES:-1}"
export PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES="${PRESERVE_REJECTED_CRITICAL_ANSWER_SENTENCES:-1}"
export REVISION_WORKERS="${REVISION_WORKERS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_cover_v3_three_datasets_nohup.sh"
