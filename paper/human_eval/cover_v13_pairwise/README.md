# COVER-RAG Human Evaluation Pack

This pack contains 100 blinded pairwise examples comparing a base
RAG answer against the corresponding COVER-RAG answer. Annotators should use
`annotation_sheet.csv`; keep `blind_key.csv` hidden until analysis.

## Scores

- `correctness_*_1_to_5`: 1 = wrong/unhelpful, 3 = partially correct, 5 = fully correct and complete.
- `citation_faithfulness_*_1_to_5`: 1 = citations do not support most claims, 3 = mixed/partial support, 5 = cited evidence supports the factual claims.
- `unsupported_claims_*_count`: approximate number of factual claims not supported by the cited evidence.
- `fluency_*_1_to_5`: 1 = hard to read, 5 = clear and fluent.
- `preference_a_b_tie`: choose `A`, `B`, or `Tie` based on overall usefulness and trustworthiness.

## Recommended Protocol

Use at least two annotators for a subset of examples if possible. Annotators
should not open `blind_key.csv`. After annotation, run:

```bash
python scripts/analyze_cover_human_eval.py \
  --annotations paper/human_eval/cover_v13_pairwise/annotation_sheet.csv \
  --key paper/human_eval/cover_v13_pairwise/blind_key.csv \
  --output-dir paper/human_eval/cover_v13_pairwise/results
```

## Optional AI-Assisted Pilot Labels

For debugging only, you can fill a separate LLM-as-judge pilot sheet:

```bash
python scripts/auto_label_cover_human_eval.py \
  --input paper/human_eval/cover_v13_pairwise/annotation_sheet.csv \
  --output paper/human_eval/cover_v13_pairwise/annotation_sheet_ai_filled.csv \
  --cache-file cache/cover_human_eval_ai_labels.json
```

This is not human evaluation. Use it only as a pilot sanity check or explicitly
report it as LLM-as-judge.
