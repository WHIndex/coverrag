# Self-RAG + COVER-RAG v13 Main Result

This directory freezes the main Self-RAG plug-in result for paper writing.

## Experiment

- Base method: API Self-RAG-style generations produced from the Self-RAG source tree under `alce/methods/self-rag`, converted into ALCE result JSON.
- Plug-in: COVER-RAG v13, same frozen post-hoc configuration used for the vanilla-RAG main result.
- Main result directory: `alce/result/api_self_rag_cover_v13_clean_full`.
- Fixed v0 baseline: `alce/result/api_self_rag_cover_v13_clean_full/cover_audit/cover_audit_comparison.csv`.
- Main run logs:
  - `logs/api_self_rag_cover_v13_clean_20260610_220317.log`
  - `logs/api_self_rag_cover_v13_clean_v0audit_20260611_085600.log`
  - `logs/api_self_rag_cover_v13_clean_fixed_v0v3_20260611_173702.log`

## Files

- `tables/required_metrics_summary.csv`: fixed paper-facing metric table with v0 and COVER-RAG v13 rows.
- `tables/main_table_self_rag_v13.csv`: compact table for the main paper result.
- `tables/cover_audit_comparison.csv`: fixed Self-RAG v0 audit summary.
- `tables/cover_v3_comparison.csv`: full ALCE and COVER diagnostics for v13.
- `tables/cover_v0_v3_comparison_long.csv`: v0 vs v13 long comparison.
- `tables/cover_v0_v3_comparison_wide.csv`: v0 vs v13 wide comparison.
- `answer_recall/`: answer-unit recall diagnostics.
- `outputs/`: final v13 JSON outputs for ASQA, ELI5, and QAMPARI.
- `logs/`: logs needed to trace the clean run and the strict v0/v3 re-evaluation.

## Main Takeaway

On Self-RAG, COVER-RAG v13 is best treated as a plug-and-play post-hoc faithfulness and citation repair module. It strongly improves claim support and claim-level citation metrics across all three datasets, improves ASQA task accuracy, keeps ELI5 task correctness unchanged while improving recall and attribution, and leaves QAMPARI task F1 roughly unchanged while greatly improving claim-level citation/support.

The later v15 recall-candidate run is not used as the main result: it slightly improves recall but introduces small task/support trade-offs, so it is better reported as an ablation rather than the frozen Self-RAG main table.
