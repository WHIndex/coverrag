# Best Current Non-Oracle Result

This directory is a lightweight paper-facing index for the current best complete non-oracle result.

## Selected Result

- Selected method: **BGE-rerank RAG + COVER-RAG v13**
- Full frozen paper directory: `alce/paper/main_results/bge_rerank_rag_cover_v13`
- Raw result directory: `alce/result/bge_rerank_rag_cover_v13_full`
- Selection criterion: highest average main-task correctness among complete three-dataset, non-oracle runs.

The selected result is already fully archived in `alce/paper/main_results/bge_rerank_rag_cover_v13`, including tables, logs, answer-recall diagnostics, and final JSON outputs.

## Key Averages

Compared with its v0 baseline, BGE-rerank RAG + COVER-RAG v13 improves average main correctness, average main recall, sentence-level citation recall, sentence-level citation precision, and claim-level citation/support metrics across ASQA, ELI5, and QAMPARI.

| metric | v0 avg | v3 avg | delta |
|---|---:|---:|---:|
| main correctness | 26.6348 | 27.4188 | +0.7840 |
| main recall | 23.5364 | 24.3339 | +0.7975 |
| sentence citation recall | 50.8106 | 54.6919 | +3.8813 |
| sentence citation precision | 45.4172 | 45.8593 | +0.4421 |
| claim citation recall | 42.8182 | 96.8868 | +54.0686 |
| claim citation precision | 50.8624 | 80.1234 | +29.2610 |
| claim support | 42.8182 | 96.8868 | +54.0686 |

## Coverage-Guided Controller Status

The code now contains a coverage-guided iterative RAG controller path, but the
actual controller is more detailed than a simple "covered vs. missing" split.

### Fine-Grained Controller Pipeline

1. **Draft answer claim audit.**
   The draft answer is decomposed into atomic claims. Each claim is verified
   against its cited or available evidence and assigned a fine-grained status,
   such as `supported`, `not_supported`, `no_citation`, or
   `wrong_or_missing_citation`. The controller also keeps claim importance
   signals, so background claims are not treated as high-priority answer gaps.

2. **Supported answer state construction.**
   Claims verified as `supported` are converted into a compact intermediate
   supported-answer state. This state is not just a rewritten answer; it is a
   coverage record of which answer units already have evidence support.

3. **Missing target generation.**
   Missing targets are generated from two sources:
   - unsupported or citation-broken draft claims, especially non-background
     claims that look like answer-bearing units;
   - high-ranked evidence sentences that contain plausible answer spans not
     already covered by the supported claims.

   A target is dropped if it is already covered by verified claims, duplicated,
   too generic, title/source-like, or irrelevant to the question. This is the
   step that operationalizes "covered facets" and "missing facets"; the system
   does not directly classify the entire question into facets, but builds
   concrete missing answer-unit targets from claim verification and evidence.

4. **Coverage-guided retrieval query construction.**
   Retrieval is triggered only if there is at least one concrete missing target.
   Each query contains the original question, the missing answer unit, optional
   evidence clues, and a brief summary of already supported answer units. This
   prevents the retriever from repeatedly searching for already covered content.

5. **Candidate generation from newly retrieved/reranked evidence.**
   Candidate answer units are generated from retrieved evidence either by
   span-id selection or extractive answer-span extraction. Each candidate keeps
   a claim, answer span, evidence sentence, document id, and target-gap field.

6. **Missing-answer-unit gate.**
   A candidate is not accepted merely because it is supported. It must first
   match a missing target. Matching is checked through normalized answer-span
   keys and lexical overlap with the target. Then the candidate receives an
   answer-unit score:

   `0.35 * target_relevance + 0.25 * question_relevance + 0.25 * answer_span_bonus + 0.15 * rank_score - compactness_penalty`

   The controller rejects candidates with low target relevance, low question
   relevance, weak answer-span evidence, overly long/generic surfaces, title-like
   spans, or wrong answer type. Explanation-style questions use stricter
   directness checks so that isolated entity names are not accepted as useful
   explanatory coverage.

7. **Evidence support verification.**
   Only after passing the missing-answer-unit gate is the candidate claim sent
   to the verifier. The accepted claim must be entailed by its evidence sentence
   or source document. Unsupported candidates are rejected even if they appear
   to fill a missing facet.

8. **Supported answer update and output guard.**
   Accepted supported claims are merged back with the original supported claims.
   The final answer renderer preserves verified content and adds only a small
   number of high-confidence new claims. A candidate-output selection guard
   compares alternatives and rejects outputs that increase recall at the cost of
   too much unsupported content or insufficient answer-unit gain.

In short, the controller is:

`claim audit -> supported-state construction -> missing target generation -> missing-target retrieval -> answer-unit gate -> NLI support verification -> guarded answer update`

The implemented controller is exposed through the dynamic-retrieval arguments in
`alce/scripts/cover_v3_iterative.py`, especially `--coverage-guided-retrieval`
and the `coverage_guided_*` controls.

However, the latest `cover_v3_dynamic_retrieval_v3_coverage_guided_fast_full`
run is not selected as a paper result because only ASQA has complete v3 metrics.
ELI5 and QAMPARI are incomplete due to runtime/API stability issues.

## Files In This Index

- `tables/selected_required_metrics_summary.csv`: copied metric summary for the selected best result.
- `tables/selected_cover_v0_v3_comparison_long.csv`: copied v0-v3 long comparison table for the selected best result.
- `tables/selected_main_table.csv`: compact selected main table.
- `tables/best_result_ranking.csv`: ranking/selection note across current complete non-oracle candidates.
