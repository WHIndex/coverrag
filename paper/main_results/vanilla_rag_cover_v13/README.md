# Vanilla RAG + COVER-RAG v13 Main Result

This directory freezes the main vanilla-RAG plug-in result for paper writing.

## Experiment

- Base method: vanilla RAG generations from `alce/result/origin`.
- Plug-in: COVER-RAG v13, `cover_v3_recall_completion_v13_integrated_explanation`.
- Main run log: `logs/cover_v3_recall_completion_v13_integrated_explanation_20260609_094814.log`.
- Full source result directory: `alce/result/cover_v3_recall_completion_v13_integrated_explanation_full`.

## Files

- `tables/required_metrics_summary.csv`: fixed paper-facing metric table.
- `tables/main_table_vanilla_rag_v13.csv`: compact table for the main paper result.
- `tables/cover_v3_comparison.csv`: full ALCE and COVER diagnostics for v13.
- `tables/cover_v0_v3_comparison_long.csv`: v0 vs v13 long comparison.
- `tables/cover_v0_v3_comparison_wide.csv`: v0 vs v13 wide comparison.
- `answer_recall/`: answer-unit recall diagnostics.
- `outputs/`: final v13 JSON outputs for ASQA, ELI5, and QAMPARI.

## Main Takeaway

COVER-RAG v13 is best treated as a plug-and-play post-hoc faithfulness and citation repair module. On vanilla RAG it strongly improves claim support and claim-level citation metrics, improves ASQA/QAMPARI task metrics, and keeps ELI5 task correctness near the original vanilla RAG while substantially improving attribution quality.

