# Human Evaluation

In this directory, you can find the human evaluation results in two files: `human_eval_utility_completed.json` and `human_eval_citations_completed.json`. 
We evaluated a sample of ASQA and ELI5 results from select models. For more details, please refer to the paper. 

### Utility
In `human_eval_utility_completed.json`, each model linked to a dictionary where the key is the question id and the value is the model output. This is identical to the data you would find after running evaluation with one additional field -- `utility_score` is rated from 1-5.

### Citations
In `human_eval_citations_completed.json`, each model linked to a dictionary where the key is the question id and the value is the model output. This is identical to the data you would find after running evaluation with a few additional fields:

- `citation_precision_score` can be found for every valid citation in every sentence. The instruction given to the annotator is: "Given the sentence and one of its cited documents, please rate if the document fully supports all claims in the sentence (2), the document partially supports the claims in the sentence (1), or if the document does not support any claims made in the sentence (0)."
- `sentence_recall_score` can be found for every sentence. The instruction given to the annotator is: "Given the sentence and its cited documents, please rate if the model response is fully supported by the documents (1) or if the model response is not fully supported by the documents (0). If the response is fully supported, that means all factually claims made by the response are found in and supported by at least one of the documents. Otherwise, the response is not fully supported."

We then calculate the humans `overall_precision_score` and `overall_recall_score`. Furthermore, we included the automatic evaluation results for ease of evaluation.

## COVER-RAG Pairwise Human Evaluation

For the COVER-RAG paper experiments, use the newer blinded pairwise workflow:

```bash
cd /home/wanghui/rag/CoverRAG/alce

python scripts/build_cover_human_eval_pack.py \
  --num-examples 100 \
  --seed 42 \
  --output-dir paper/human_eval/cover_v13_pairwise
```

Annotators should fill `paper/human_eval/cover_v13_pairwise/annotation_sheet.csv`
without opening `blind_key.csv`. After annotation, summarize the results with:

```bash
python scripts/analyze_cover_human_eval.py \
  --annotations paper/human_eval/cover_v13_pairwise/annotation_sheet.csv \
  --key paper/human_eval/cover_v13_pairwise/blind_key.csv \
  --output-dir paper/human_eval/cover_v13_pairwise/results
```

Recommended annotation dimensions:

- answer correctness: 1 to 5
- citation faithfulness: 1 to 5
- unsupported factual claim count
- fluency: 1 to 5
- overall preference: A, B, or Tie

For a quick pilot that should **not** be reported as human evaluation, use:

```bash
python scripts/auto_label_cover_human_eval.py \
  --input paper/human_eval/cover_v13_pairwise/annotation_sheet.csv \
  --output paper/human_eval/cover_v13_pairwise/annotation_sheet_ai_filled.csv \
  --cache-file cache/cover_human_eval_ai_labels.json
```
