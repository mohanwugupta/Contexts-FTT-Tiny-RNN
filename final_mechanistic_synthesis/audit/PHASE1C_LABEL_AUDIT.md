# Phase 1C Label and Selection Audit

The historical labels described distinct objects and are retained only with explicit qualifiers.
The new aggregate and per-seed stable selections use endpoint stability, familiar A/B separation,
familiar loss, and task-path residual only. Mixed outcomes and rescue never enter selection.

| Qualified label | Cell/object | Aggregation | Mixed used to select? | Firewall |
|---|---|---|---:|---:|
| historical_report_global_minimum_stable_seed_cell_descriptive | P1_correct_logit_alpha_0.1_anchor_0.1 | single seed-cell | True | False |
| weak_path_plot_representative | P1_correct_logit_alpha_0.001_anchor_10 | mean across 20 seeds | False | True |
| mid_path_plot_representative | P1_correct_logit_alpha_0.1_anchor_10 | mean across 20 seeds | False | True |
| strong_path_plot_representative | P1_correct_logit_alpha_30_anchor_10 | mean across 20 seeds | False | True |
| historical_best_stable_plot_representative_descriptive | P1_correct_logit_alpha_10_anchor_30 | mean across 20 seeds | True | False |
| best_stable_aggregate_cell | P1_correct_logit_alpha_10_anchor_30 | mean across 20 seeds | False | True |
| best_stable_seed_cells_per_seed | 13 distinct seed-selected cells | one selected cell per seed, then seed mean | False | True |

Historical plot choices that used Mixed loss as a readability tie-break are marked
descriptive and are not used for inferential cell selection.
