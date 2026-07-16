# BGE-Rerank RAG + COVER-RAG v13 Main Result

This directory freezes the main BGE-rerank-RAG plug-in result for paper writing.

## Experiment

- Base method: BGE-rerank RAG generations from `alce/result/origin`.
- Reranker: `BAAI/bge-reranker-large`, used by `alce/scripts/rerank_alce_docs.py` to cross-encode each `(question, document)` pair and reorder the top100 candidate documents.
- Generator: `gpt-4o-mini`, using the reranked top-5 documents.
- Plug-in: COVER-RAG v13, same frozen post-hoc configuration used for the vanilla-RAG and Self-RAG main results.
- Main result directory: `alce/result/bge_rerank_rag_cover_v13_full`.
- Main run log: `logs/bge_rerank_rag_cover_v13_20260613_095907.log`.

## Files

- `tables/required_metrics_summary.csv`: fixed paper-facing metric table with v0 and COVER-RAG v13 rows.
- `tables/main_table_bge_rerank_rag_v13.csv`: compact table for the main paper result.
- `tables/cover_audit_comparison.csv`: fixed BGE-rerank-RAG v0 audit summary.
- `tables/cover_v3_comparison.csv`: full ALCE and COVER diagnostics for v13.
- `tables/cover_v0_v3_comparison_long.csv`: v0 vs v13 long comparison.
- `tables/cover_v0_v3_comparison_wide.csv`: v0 vs v13 wide comparison.
- `answer_recall/`: answer-unit recall diagnostics.
- `outputs/`: final v13 JSON outputs for ASQA, ELI5, and QAMPARI.
- `logs/`: log needed to trace the full v0/v3 run.

## Main Takeaway

On BGE-rerank RAG, COVER-RAG v13 is a strong third plug-in result. It improves main-task correctness and answer recall on all three ALCE datasets, improves sentence-level citation recall on all three datasets, and substantially improves claim-level citation/support metrics. ELI5 sentence-level citation precision is the main remaining trade-off, but claim-level precision and support both improve strongly.

