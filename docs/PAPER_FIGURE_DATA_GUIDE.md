# Paper figure data guide

This guide identifies the CSVs needed for the paper-facing figures, the unit of
observation in each table, the filters that must be applied, and a defensible R
plotting strategy. The versioned source tables under `outputs/` should be the
source of truth. The files under `outputs/final/tables/` are checksum-traced
copies, but their longer names make interactive R work less convenient.

## Non-negotiable plotting rules

1. Use seeds as the independent unit for SRN uncertainty. If a table contains
   seven token positions per seed, average within `seed` first and calculate
   intervals across the 20 seed means.
2. Use participants as the independent unit for human uncertainty. If a table
   contains repeated sequences or probe blocks, average within
   `participant_uid × transition_position` first and bootstrap or summarize
   across participants.
3. Keep `checkpoint == "human_matched"` explicit for the primary causal and
   human-SRN analyses. The v2.1 splitting gate itself was evaluated at
   `checkpoint == "final"`; do not silently mix those checkpoints in one
   estimate.
4. Do not calculate standard errors from already aggregated one-row summaries
   such as `mixed_model_comparison.csv` or
   `human_srn_subspace_similarity.csv`.
5. Distinguish descriptive plots from inferential tests. For example, the
   plotted participant A-B profile is descriptive; the Context × Position test
   comes from `context_position_model.csv`.
6. For bootstrap intervals, resample whole participants or whole seeds, never
   individual transition rows.

## Recommended main-figure structure

### Figure 1: conceptual design and inferential order

No CSV is needed. The diagram should show:

`A/B blocked learning -> context-specific recurrent states -> execution readout
-> frozen Mixed/Novel probes -> human transition profiles`.

Visually separate the SRN mechanism/existence proof from the human behavioral
test. Add the explicit boundary that the exact SRN and human transition
geometries did not match.

### Figure 2: SRN learning, endpoint reliability, and frozen probes

#### Panel A: A/B hidden-state separation over training

- Source: `outputs/ftt_srn_execution_v2_final_checkpointed/splitting_metrics.csv`
- Unit in source: one `seed × model_condition × checkpoint` row.
- Filters: none for condition; retain the four ordered checkpoints.
- X axis: checkpoint sequence (`20, 40, 80, 170`), preferably labeled with
  both the sequence count and semantic checkpoint name.
- Y axis: `ab_hidden_separation`.
- Display: three condition lines (`High`, `Low`, `High_seq_only`) with the 20
  seed points faintly visible and a seed-level 95% CI around the mean.

This panel is more informative than a single checkpoint bar chart because it
shows when the High/Low contrast appears. At the human-matched checkpoint the
mean separations are 1.153 (High), 0.076 (Low), and 1.167
(`High_seq_only`). The last comparison shows that high initialization scale,
not the execution objective alone, drives the hidden-state split.

#### Panel B: frozen probe position within each network's endpoints

- Source: `outputs/ftt_srn_execution_v2_final_checkpointed/probe_interpolation.csv`
- Unit in source: one `seed × model_condition × checkpoint × probe_type` row.
- Primary filters:
  `model_condition == "High"`, `checkpoint == "human_matched"`, and
  `probe_type %in% c("Mixed_norm", "Novel")`.
- Sensitivity: show `Mixed_additive` separately or in the supplement.
- Display: a raincloud/dot plot of `interpolation_index`, with reference lines
  at 0 (B endpoint), .5 (midpoint), and 1 (A endpoint). Mark unreliable seeds
  (`endpoint_reliable == FALSE`) in gray rather than dropping them invisibly.
  A second annotation should give `n reliable / 20` and the fraction of those
  reliable seeds that are `between_endpoints`.

At the human-matched checkpoint, 18/20 High networks have reliable endpoints;
12/18 reliable seeds place normalized Mixed between endpoints and 11/18 place
Novel between endpoints. At the final checkpoint only 7/20 endpoints pass the
reliability threshold, so a final-checkpoint claim must report that denominator.

Do not plot only the three grand mean RT bars. Endpoint ordering differs across
seeds, and the grand mean probe can fall outside the grand mean endpoints even
when a majority of within-seed probes are between their own endpoints.

#### Panel C: execution grounding

- Source: `outputs/ftt_srn_execution_v2_final_checkpointed/execution_dynamics.csv`
- Unit in source: one `seed × training sequence × transition_position` row.
- Filters: `model_condition == "High"` and `global_sequence > 90` (the frozen
  late eight-block window, sequences 91-170).
- Display: bin `correct_token_confidence` within seed, average
  `execution_speed` within bin and seed, then show the across-seed mean curve
  separately for A and B. An alternative is a distribution of per-seed
  confidence-speed slopes/correlations.
- Gate annotations: obtain the registered cohort-level correlations and
  controlled coefficient from any one High row of
  `splitting_metrics.csv`, because those fields are duplicated audit values.

Do not put the three correlations and the controlled regression coefficient on
an unlabeled common bar scale; they are different statistics.

### Figure 3: causal specificity and its boundary

#### Panel A: matched versus mismatched causal patching

- Source:
  `outputs/ftt_srn_context_sequence_v23_final/token_structured_causal_effects.csv`
- Unit in source: one `seed × checkpoint × probe_context × token_position` row.
- Filter: `checkpoint == "human_matched"`.
- First aggregation: within each `seed × probe_context`, take the mean of
  `full_patch_effect`, `wrong_token_patch_effect`, and
  `random_patch_effect` over the seven token positions.
- Display: paired seed dots/lines or a half-eye plot for Matched, Mismatched,
  and Orthogonal Random, faceted by `Mixed_norm` and `Novel`. Report CIs across
  the 20 seed means.

The seed-level means are approximately .091/.059/.010 for
Matched/Mismatched/Random in Mixed and .095/.061/.009 in Novel.

#### Panel B: registered mechanism criteria

- Source:
  `outputs/ftt_srn_context_sequence_v23_final/interaction_mechanism_summary.csv`
- Unit in source: one `seed × checkpoint × probe_context` row; group-level
  fractions are duplicated on every seed row.
- Filter: `checkpoint == "human_matched"`.
- Display: for M1-M6, show either a 20-cell seed pass grid from the Boolean
  criterion columns or a point with an exact binomial interval for the seed
  pass proportion. Draw the registered .80 line only for criteria that actually
  use the .80 rule. M2-M4 use majority-plus-group-mean rules, not the same .80
  rule; M6 is a cohort model-comparison result duplicated across seed rows.

The result to communicate is partial: M1-M4 and the random-direction
specificity control pass, while M5 and M6 fail.

#### Panel C: strict necessity boundary

- Source:
  `outputs/ftt_srn_causal_v22_final/ablation_seed_summary.csv`
- Unit in source: one `seed × checkpoint` row.
- Filter: `checkpoint == "human_matched"`.
- Display: two paired seed distributions for
  `ablation_execution_collapse` and `ablation_sequence_preservation`, with
  horizontal lines at .50 and .90, respectively. Because these metrics have
  different thresholds, facet them rather than implying a single common
  criterion.

This panel should be labeled as the failed/limited necessity boundary, not as a
positive necessity result.

### Figure 4: human transition structure and held-out prediction

#### Panel A: pure A-B transition profile

- Source: `outputs/human_configured_state_v3/pure_ab_token_profiles.csv`
- Unit in source: one `participant_uid × transition_position` row.
- Y axis: `delta = r_a - r_b` on the log-RT scale.
- Display: participant dots or faint participant lines plus a participant-level
  mean and 95% participant bootstrap CI, separately for `Combined` and `Novel`.
- Inferential annotation source:
  `outputs/human_configured_state_v3/context_position_model.csv`, filtered to
  `row_type == "joint_test"`.

Report both joint tests, not only the minimum p-value: Combined Wald
statistic=60.39, df=6, p=3.74e-11; Novel Wald statistic=49.06, df=6,
p=7.27e-9. The plotted `delta` values are shrunken participant estimates
(shrinkage strength 8), not raw cell means or model-adjusted contrasts.

#### Panel B: Mixed and Novel midpoint deviations

- Sources:
  `outputs/human_configured_state_v3/mixed_midpoint_deviation.csv` and
  `outputs/human_configured_state_v3/novel_midpoint_deviation.csv`.
- Unit in source: one probe sequence observation at one transition position;
  participants contribute 63-189 rows.
- Required first aggregation: mean `midpoint_deviation` within
  `participant_uid × transition_position`. Then summarize or bootstrap across
  participants.
- Display: two transition-position lines with 95% participant bootstrap CIs;
  optionally facet rather than overlay because the datasets have different
  participants and probe designs.

Do not take the SEM over all source rows. That treats repeated sequences as
independent and gives participants with three probe blocks three times the
weight of test-only participants.

#### Panel C: participant-held-out model comparison

Summary-only option:

- `outputs/human_configured_state_v3/mixed_model_comparison.csv`
- `outputs/human_configured_state_v3/novel_model_comparison.csv`

These files contain one row per model and are appropriate for reporting the
registered ranking, but not for making error bars.

Uncertainty-aware plotting option (recommended):

- `outputs/human_configured_state_v3/model_M0_scalar_context.csv`
- `outputs/human_configured_state_v3/model_M1_scalar_mixture.csv`
- `outputs/human_configured_state_v3/model_M2_token_specific.csv`
- `outputs/human_configured_state_v3/model_M3_low_rank.csv`
- `outputs/human_configured_state_v3/model_M4_srn_informed.csv`, filtered to
  `model == "M4_srn_informed"`.

Aggregate `heldout_log_likelihood` within participant and model, form paired
within-participant differences from a declared reference (M1 is the most useful
scalar reference), and display the distribution and a participant bootstrap CI
of those differences. Keep Combined and Novel in separate facets. The registered
ordering is M3 > M1 > M2 > M4 > M0 for Mixed and M3 > M2 > M1 > M4 > M0 for
Novel.

The primary pipeline sums observation-level log likelihood, so participants
with more probe observations receive more weight in the registered ranking.
The participant-equal plot is a transparency/sensitivity display and should be
labeled as such.

### Figure 5: selected human rank, reproducibility, and SRN mismatch

#### Panel A: rank-2 human component loadings

- Source: `outputs/human_low_rank_interpretation_v31/component_loadings.csv`
- Unit: one `component × transition_position` row.
- Display: the two loading curves with the exported participant-bootstrap 95%
  intervals (`bootstrap_ci_lower`, `bootstrap_ci_upper`). Label them with
  `interpretation_label`, but retain neutral component numbers wherever
  `semantic_label_assigned == FALSE`.

#### Panel B: held-out rank selection

- Source: `outputs/human_low_rank_interpretation_v31/dimension_selection.csv`
- Use `row_type == "summary"` and `method == "PCA"` for the mean curves and
  `probe_log_likelihood_se` for fold-level error bars.
- Display: ranks 1, 2, 3, and 7 on a discrete x axis, separate dataset lines,
  and a vertical mark at selected rank 2. State in the caption that selection
  pooled the PCA fold scores across both datasets and used the smallest rank
  within one SE of the best score.
- Supplement: FactorAnalysis and parallel-analysis results from the same CSV.

#### Panel C: independent human cross-dataset replication

- Source:
  `outputs/human_low_rank_interpretation_v31/cross_dataset_subspace_similarity.csv`
- Unit: one summary row.
- Display: a compact metric panel or annotated subspace diagram reporting both
  principal angles (43.70 degrees and 4.60 degrees), mean matched-component
  cosine (.847), and shared-subspace variance fraction (.758). Do not add an
  error bar to the single row.

#### Panel D: frozen human-SRN mismatch

- Sources:
  `outputs/human_low_rank_interpretation_v31/human_srn_subspace_similarity.csv`
  and, optionally,
  `outputs/human_low_rank_interpretation_v31/human_srn_dimensionality_comparison.csv`.
- Display: show the two principal angles (86.17 degrees and 7.10 degrees) or
  the canonical correlations, with the prespecified match thresholds in the
  caption. A small dimensionality inset can show the human and SRN
  participation-ratio dimensions (2.80 and 2.33).

The conclusion is not that the systems differ in dimensionality alone; their
rank/effective dimensionalities are similar, but their selected subspaces do
not meet the frozen geometric match rule.

## High-value supplemental figures and CSVs

1. SRN learning and controls:
   `training_dynamics.csv`, `transition_costs.csv`, `transfer_metrics.csv`,
   `context_weight_metrics.csv`, `low_vs_high_scale.csv`, and
   `sequence_only_vs_execution.csv` under
   `outputs/ftt_srn_execution_v2_final_checkpointed/`.
2. Full causal steering and diagnostic controls:
   `steering_results.csv`, `random_direction_controls.csv`,
   `sequence_direction_controls.csv`, `off_manifold_metrics.csv`, and
   `causal_mechanism_summary.csv` under `outputs/ftt_srn_causal_v22_final/`.
3. Shared/specific causal decomposition:
   `shared_context_axis.csv`, `full_context_patching.csv`,
   `shared_context_patching.csv`, `specific_context_patching.csv`,
   `wrong_token_patching.csv`, `execution_model_comparison.csv`,
   `mixed_generalization.csv`, and `novel_generalization.csv` under
   `outputs/ftt_srn_context_sequence_v23_final/`.
4. Human validation:
   `pure_ab_profile_reliability.csv`, `temporal_probe_profiles.csv`,
   `counterbalance_sensitivity.csv`, `model_recovery_matrix.csv`, and
   `shared_vs_separate_probe_profile.csv` under
   `outputs/human_configured_state_v3/`.
5. Low-rank stability and semantic limits:
   `component_stability.csv`, `component_feature_associations.csv`,
   `difficulty_residualized_results.csv`, `shared_vs_separate_basis.csv`,
   `semantic_recovery_matrix.csv`, and
   `final_dimension_interpretation_summary.csv` under
   `outputs/human_low_rank_interpretation_v31/`.

## Audit of the existing Python paper figures

The current `scripts/make_paper_figures.py` is useful as a provenance and
diagnostic layer, but the following should be corrected in the R paper figures:

- Figure 3A computes SEM across seed-token rows rather than across seed means.
- Figure 4B computes SEM across repeated probe observations rather than across
  participant profiles.
- Figure 2B uses grand mean endpoints, which can obscure within-network endpoint
  ordering and endpoint unreliability.
- Figure 2C puts correlations and a regression coefficient on one bar scale and
  gives no sampling uncertainty.
- Figure 4C uses already aggregated model rows, so it cannot display uncertainty
  or paired participant differences.
- Figure 5A omits the bootstrap intervals already present in the source table;
  Figure 5B omits the exported fold SEs.
- `paper_figure_sources.json` includes several claim-supporting tables that the
  plotting code does not directly read (`execution_geometry.csv`,
  `context_position_model.csv`, `cross_dataset_subspace_similarity.csv`, and
  `final_dimension_interpretation_summary.csv`). Their hashes remain useful,
  but the manifest is broader than a literal list of plotted inputs.

