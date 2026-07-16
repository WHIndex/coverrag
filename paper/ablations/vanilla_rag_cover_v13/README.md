# Vanilla RAG + COVER-RAG v13 Ablations

This directory tracks ablations for the frozen vanilla-RAG COVER-RAG v13 run.

## Included So Far

- `no_recovery`: disables targeted citation/claim recovery by setting `RECOVER_LABELS=__none__`.
- `no_recall_completion`: disables recall-oriented missing-answer completion by setting `RECALL_ORIENTED_COMPLETION=0`.
- `no_expansion`: disables evidence/claim expansion and recall completion by setting `EXPANSION_MODE=off`, `RECALL_ORIENTED_COMPLETION=0`, and related expansion budgets to zero.
- `no_final_audit`: disables the final audit and audit-based output candidate selection.
- `no_candidate_selection`: keeps final audit enabled, but disables audit-based final-output candidate selection.
- `llm_verifier`: replaces the local NLI verifier with an LLM verifier while keeping the rest of the v13 configuration.

## Files

- `tables/ablation_core_modules_macro.csv`: compact macro-average ablation table.
- `tables/ablation_core_modules_by_dataset.csv`: dataset-level ablation table.
- `tables/no_recovery_required_metrics_summary.csv`: fixed metric table for the no-recovery run.
- `tables/no_recall_completion_required_metrics_summary.csv`: fixed metric table for the no-recall-completion run.
- `tables/no_expansion_required_metrics_summary.csv`: fixed metric table for the no-expansion run.
- `tables/no_final_audit_required_metrics_summary.csv`: fixed metric table for the no-final-audit diagnostic run.
- `tables/no_candidate_selection_required_metrics_summary.csv`: fixed metric table for the no-candidate-selection run.
- `tables/llm_verifier_required_metrics_summary.csv`: fixed metric table for the LLM-verifier sensitivity run.
- `tables/ablation_no_recovery_vs_full.csv`: paper-facing comparison against full v13.
- `logs/cover_v3_ablation_vanilla_no_recovery_20260614_093903.log`: full run log.
- `logs/cover_v3_ablation_vanilla_no_recall_completion_20260614_143029.log`: full run log.
- `logs/cover_v3_ablation_vanilla_no_expansion_20260614_200747.log`: full run log.
- `logs/cover_v3_ablation_vanilla_no_final_audit_20260619_204520.log`: full run log.
- `logs/cover_v3_ablation_vanilla_no_candidate_selection_resume_20260620_171055.log`: resumed full run log.
- `logs/cover_v3_ablation_vanilla_llm_verifier_20260620_195650.log`: full run log.

## Takeaway

Removing targeted recovery lowers macro main correctness by 0.4571 and macro main recall by 0.6421 compared with full v13. It also lowers claim citation recall/support by 3.0381 and claim sentence recall by 7.3431. Sentence-level citation precision increases because the ablated system keeps fewer recovered links/claims, so this is not a better overall system: the full model preserves more supported answer content while maintaining stronger claim-level coverage.

Removing recall-oriented completion has a smaller effect: macro main correctness drops by 0.1130 and macro main recall drops by 0.1727, while claim support slightly increases by 0.3096. This suggests recall completion contributes to answer utility, but its current gains are modest and partly trade against stricter claim-level filtering. The next ablation should remove expansion more broadly to separate completion from the evidence/claim expansion stage.

Removing expansion more broadly lowers macro main correctness by 0.4748 and macro main recall by 0.8111, and lowers sentence-level citation recall/precision by 4.0779/5.1446. Claim support increases by 2.1162 because the output becomes more conservative. This supports the interpretation that expansion is the main utility-oriented component, while verification/recovery is the main faithfulness-oriented component.

The `no_final_audit` run is a diagnostic rather than a direct claim-level
metric ablation: ASQA/ELI5 do not have final-audit claim records when the final
audit is disabled, so paper-facing tables mark claim-level metrics as
unavailable for that row. Its main-task and sentence-citation metrics are still
useful for checking whether final audit changes the rendered answer. The next
cleaner ablation is `no_candidate_selection`, which keeps final audit enabled
for fair claim-level measurement while disabling only audit-based output
selection.

Removing audit-based output candidate selection has almost no effect on macro
main correctness (-0.0010) or macro main recall (-0.0232), and only slightly
lowers claim-level support/citation metrics (claim support -0.1016). This
suggests candidate selection is a small stabilizer rather than a main source of
the gains. The core utility and faithfulness improvements instead come from
claim recovery and evidence/claim expansion.

Replacing the NLI verifier with an LLM verifier improves macro main
correctness (+0.3389), macro main recall (+0.1671), and sentence-level citation
metrics (+4.1986 recall, +6.6226 precision), but substantially lowers
claim-level support (-6.0554) and claim-sentence recall (-9.9459). The effect
is also dataset-dependent: ASQA gains, ELI5 loses, and QAMPARI is nearly
unchanged. This should be reported as verifier sensitivity/robustness rather
than a core module ablation. It supports using the NLI verifier as the default
when the paper's priority is conservative claim-level faithfulness.
