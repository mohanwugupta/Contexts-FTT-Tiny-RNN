# Dynamical generalization — final report

The four questions concern algebraic output compatibility versus dynamical access (E1), independent contextual coordinates (E2), a randomized separation × closure training intervention (E3), and replication across fixed sequence classes (E4). The original four-experiment paper freeze remains a separate historical result.

| stage | implementation_status | scientific_status | evidence_status |
| --- | --- | --- | --- |
| e1 | validated | completed_mixed | completed |
| e2 | validated | completed_mixed | completed |
| e3 | validated | insufficient_familiar_support | insufficient_familiar_support |
| e4 | validated | inconclusive | blocked_by_e3 |


# E1 — Static compatibility and dynamical accessibility

A linear output head maps the hidden midpoint to the arithmetic midpoint of logits. Because A and B share a correct label and CE is convex in logits, midpoint CE cannot exceed their mean endpoint CE. This algebraic guarantee does not imply natural recurrent access.

All 840 static transition rows passed affine and convexity checks. Maximum affine discrepancy: 1.776e-15; minimum Jensen gap: 8.176e-09. Float32 recorded-head roundoff is exported separately.

The exported identity is e_t = b_t + [f(m_(t−1)+e_(t−1))−f(m_(t−1))]. J_t e_(t−1) is only a first-order approximation; its Taylor remainder and output-relevant norms are separate columns. Jacobian singular values and finite products describe local finite-horizon amplification, not autonomous attractor stability. Single-patch rows explicitly label the exogenous placement and then audit recurrent continuation.

| cohort | condition | probe | raw_ce | closure_task_norm | error_task_norm | taylor_remainder_task_norm |
| --- | --- | --- | --- | --- | --- | --- |
| canonical | High_monotonic | A | 0.91728 | 3.696 | 2.904 | 4.1213 |
| canonical | High_monotonic | B | 1.4558 | 3.8384 | 2.904 | 4.5798 |
| canonical | High_monotonic | Mixed | 2.2483 | 3.1332 | 3.7911 | 4.4229 |
| canonical | High_monotonic | NoContext | 2.94 | 3.9618 | 5.5247 | 6.9855 |
| canonical | High_monotonic | Novel30Mean | 2.477 | 4.3405 | 5.0499 | 6.5605 |
| canonical | High_monotonic | Novel60Mean | 3.4008 | 5.0169 | 6.1628 | 7.7625 |
| canonical | Low_monotonic | A | 0.04844 | 0.02151 | 0.052815 | 0.0022221 |
| canonical | Low_monotonic | B | 0.050439 | 0.021202 | 0.052815 | 0.0021435 |
| canonical | Low_monotonic | Mixed | 0.048892 | 0.0027614 | 0.0040642 | 1.2934e-05 |
| canonical | Low_monotonic | NoContext | 0.062768 | 0.42247 | 0.38901 | 0.03583 |
| canonical | Low_monotonic | Novel30Mean | 0.057123 | 0.26905 | 0.25241 | 0.018117 |
| canonical | Low_monotonic | Novel60Mean | 0.071878 | 0.57765 | 0.53558 | 0.062336 |
| constructive | C0_canonical | A | 0.90348 | 3.649 | 2.9008 | 3.8813 |
| constructive | C0_canonical | B | 1.4924 | 3.7698 | 2.9008 | 4.3241 |
| constructive | C0_canonical | Mixed | 2.2062 | 3.087 | 3.7224 | 4.4333 |
| constructive | C0_canonical | NoContext | 2.9558 | 3.8931 | 5.6051 | 6.9688 |
| constructive | C0_canonical | Novel30Mean | 2.4959 | 4.2759 | 5.0871 | 6.6889 |
| constructive | C0_canonical | Novel60Mean | 3.4383 | 4.9667 | 6.2672 | 8.2307 |
| constructive | C1_contrast_only | A | 1.1687 | 4.6838 | 3.1257 | 6.098 |
| constructive | C1_contrast_only | B | 0.96278 | 4.5479 | 3.1257 | 6.2903 |
| constructive | C1_contrast_only | Mixed | 3.0275 | 3.8075 | 5.2534 | 5.7569 |
| constructive | C1_contrast_only | NoContext | 3.0275 | 3.8075 | 5.2534 | 5.7569 |
| constructive | C1_contrast_only | Novel30Mean | 1.9381 | 5.0447 | 4.5159 | 7.5784 |
| constructive | C1_contrast_only | Novel60Mean | 1.9381 | 5.0447 | 4.5159 | 7.5784 |
| constructive | C2_closure_only | A | 1.1579 | 3.6763 | 2.8809 | 3.9027 |
| constructive | C2_closure_only | B | 1.3682 | 3.4673 | 2.8809 | 4.1343 |
| constructive | C2_closure_only | Mixed | 1.8859 | 2.984 | 3.3506 | 2.9806 |
| constructive | C2_closure_only | NoContext | 2.7376 | 3.712 | 5.4886 | 6.0639 |
| constructive | C2_closure_only | Novel30Mean | 2.3146 | 4.1645 | 4.7792 | 6.6964 |
| constructive | C2_closure_only | Novel60Mean | 3.3935 | 5.0154 | 6.2615 | 7.885 |
| constructive | C3_full | A | 1.2333 | 4.2679 | 3.1169 | 5.4978 |
| constructive | C3_full | B | 0.96823 | 4.5213 | 3.1169 | 6.0292 |
| constructive | C3_full | Mixed | 2.3332 | 3.7364 | 4.0573 | 4.2542 |
| constructive | C3_full | NoContext | 2.3332 | 3.7364 | 4.0573 | 4.2542 |
| constructive | C3_full | Novel30Mean | 2.1361 | 4.8203 | 4.7049 | 7.5695 |
| constructive | C3_full | Novel60Mean | 2.1361 | 4.8203 | 4.7049 | 7.5695 |

Nested held-out-seed prediction MSE (lower is better):

| cohort | dynamics_augmented | static |
| --- | --- | --- |
| canonical | 0.82371 | 0.96617 |
| constructive | 1.5611 | 1.7132 |

Paired held-out-seed and C0–C3 effects:

| sequence_id | cohort | probe | contrast | metric | n_seeds | mean | ci_low | ci_high | resampling_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | canonical | all_registered_raw_cues | dynamics_minus_static | held_out_MSE | 20 | -0.14246 | -0.25335 | -0.035952 | seed_within_sequence |
| original | constructive | A | C2_minus_C0 | closure_task_norm | 20 | 0.027274 | -0.39826 | 0.40564 | seed_within_sequence |
| original | constructive | A | C2_minus_C0 | error_task_norm | 20 | -0.019885 | -0.29485 | 0.24638 | seed_within_sequence |
| original | constructive | A | C2_minus_C0 | raw_ce | 20 | 0.25444 | 0.051084 | 0.47063 | seed_within_sequence |
| original | constructive | A | C3_minus_C1 | closure_task_norm | 20 | -0.41584 | -0.80234 | -0.061139 | seed_within_sequence |
| original | constructive | A | C3_minus_C1 | error_task_norm | 20 | -0.0087392 | -0.34773 | 0.23588 | seed_within_sequence |
| original | constructive | A | C3_minus_C1 | raw_ce | 20 | 0.064575 | -0.21645 | 0.33496 | seed_within_sequence |
| original | constructive | B | C2_minus_C0 | closure_task_norm | 20 | -0.30248 | -0.70822 | 0.074587 | seed_within_sequence |
| original | constructive | B | C2_minus_C0 | error_task_norm | 20 | -0.019885 | -0.29485 | 0.24638 | seed_within_sequence |
| original | constructive | B | C2_minus_C0 | raw_ce | 20 | -0.12416 | -0.65816 | 0.28928 | seed_within_sequence |
| original | constructive | B | C3_minus_C1 | closure_task_norm | 20 | -0.026617 | -0.41738 | 0.32973 | seed_within_sequence |
| original | constructive | B | C3_minus_C1 | error_task_norm | 20 | -0.0087392 | -0.34773 | 0.23588 | seed_within_sequence |
| original | constructive | B | C3_minus_C1 | raw_ce | 20 | 0.0054527 | -0.2677 | 0.24849 | seed_within_sequence |
| original | constructive | Mixed | C2_minus_C0 | closure_task_norm | 20 | -0.10294 | -0.53934 | 0.25054 | seed_within_sequence |
| original | constructive | Mixed | C2_minus_C0 | error_task_norm | 20 | -0.37178 | -0.99364 | 0.24491 | seed_within_sequence |
| original | constructive | Mixed | C2_minus_C0 | raw_ce | 20 | -0.32032 | -0.76945 | 0.1161 | seed_within_sequence |
| original | constructive | Mixed | C3_minus_C1 | closure_task_norm | 20 | -0.07114 | -0.33591 | 0.20644 | seed_within_sequence |
| original | constructive | Mixed | C3_minus_C1 | error_task_norm | 20 | -1.1961 | -2.1099 | -0.36335 | seed_within_sequence |
| original | constructive | Mixed | C3_minus_C1 | raw_ce | 20 | -0.69422 | -1.3815 | -0.059478 | seed_within_sequence |
| original | constructive | NoContext | C2_minus_C0 | closure_task_norm | 20 | -0.18115 | -0.642 | 0.251 | seed_within_sequence |
| original | constructive | NoContext | C2_minus_C0 | error_task_norm | 20 | -0.11658 | -0.63315 | 0.37309 | seed_within_sequence |
| original | constructive | NoContext | C2_minus_C0 | raw_ce | 20 | -0.21818 | -0.59422 | 0.13179 | seed_within_sequence |
| original | constructive | NoContext | C3_minus_C1 | closure_task_norm | 20 | -0.07114 | -0.33591 | 0.20644 | seed_within_sequence |
| original | constructive | NoContext | C3_minus_C1 | error_task_norm | 20 | -1.1961 | -2.1099 | -0.36335 | seed_within_sequence |
| original | constructive | NoContext | C3_minus_C1 | raw_ce | 20 | -0.69422 | -1.3815 | -0.059478 | seed_within_sequence |
| original | constructive | Novel30Mean | C2_minus_C0 | closure_task_norm | 20 | -0.11137 | -0.38929 | 0.14356 | seed_within_sequence |
| original | constructive | Novel30Mean | C2_minus_C0 | error_task_norm | 20 | -0.30787 | -0.8773 | 0.1854 | seed_within_sequence |
| original | constructive | Novel30Mean | C2_minus_C0 | raw_ce | 20 | -0.18127 | -0.62433 | 0.16086 | seed_within_sequence |
| original | constructive | Novel30Mean | C3_minus_C1 | closure_task_norm | 20 | -0.2244 | -0.55488 | 0.074574 | seed_within_sequence |
| original | constructive | Novel30Mean | C3_minus_C1 | error_task_norm | 20 | 0.18902 | -0.13349 | 0.53403 | seed_within_sequence |
| original | constructive | Novel30Mean | C3_minus_C1 | raw_ce | 20 | 0.19797 | -0.0296 | 0.43548 | seed_within_sequence |
| original | constructive | Novel30_A | C2_minus_C0 | closure_task_norm | 20 | -0.015886 | -0.36626 | 0.3073 | seed_within_sequence |
| original | constructive | Novel30_A | C2_minus_C0 | error_task_norm | 20 | -0.033706 | -0.78395 | 0.66482 | seed_within_sequence |
| original | constructive | Novel30_A | C2_minus_C0 | raw_ce | 20 | -0.059534 | -0.44319 | 0.27826 | seed_within_sequence |
| original | constructive | Novel30_A | C3_minus_C1 | closure_task_norm | 20 | -0.38643 | -0.81985 | -0.0080482 | seed_within_sequence |
| original | constructive | Novel30_A | C3_minus_C1 | error_task_norm | 20 | 0.37945 | -0.027114 | 0.83425 | seed_within_sequence |
| original | constructive | Novel30_A | C3_minus_C1 | raw_ce | 20 | 0.19262 | -0.01506 | 0.40433 | seed_within_sequence |
| original | constructive | Novel30_B | C2_minus_C0 | closure_task_norm | 20 | -0.20686 | -0.59224 | 0.12654 | seed_within_sequence |
| original | constructive | Novel30_B | C2_minus_C0 | error_task_norm | 20 | -0.58203 | -1.0708 | -0.15519 | seed_within_sequence |
| original | constructive | Novel30_B | C2_minus_C0 | raw_ce | 20 | -0.303 | -0.93366 | 0.17071 | seed_within_sequence |
| original | constructive | Novel30_B | C3_minus_C1 | closure_task_norm | 20 | -0.062378 | -0.40381 | 0.26568 | seed_within_sequence |
| original | constructive | Novel30_B | C3_minus_C1 | error_task_norm | 20 | -0.0014163 | -0.53295 | 0.54198 | seed_within_sequence |
| original | constructive | Novel30_B | C3_minus_C1 | raw_ce | 20 | 0.20332 | -0.19516 | 0.65451 | seed_within_sequence |
| original | constructive | Novel60Mean | C2_minus_C0 | closure_task_norm | 20 | 0.048741 | -0.17296 | 0.25757 | seed_within_sequence |
| original | constructive | Novel60Mean | C2_minus_C0 | error_task_norm | 20 | -0.0056378 | -0.48738 | 0.46379 | seed_within_sequence |
| original | constructive | Novel60Mean | C2_minus_C0 | raw_ce | 20 | -0.044785 | -0.26666 | 0.16219 | seed_within_sequence |
| original | constructive | Novel60Mean | C3_minus_C1 | closure_task_norm | 20 | -0.2244 | -0.55488 | 0.074574 | seed_within_sequence |
| original | constructive | Novel60Mean | C3_minus_C1 | error_task_norm | 20 | 0.18902 | -0.13349 | 0.53403 | seed_within_sequence |
| original | constructive | Novel60Mean | C3_minus_C1 | raw_ce | 20 | 0.19797 | -0.0296 | 0.43548 | seed_within_sequence |
| original | constructive | Novel60_A | C2_minus_C0 | closure_task_norm | 20 | -0.059985 | -0.447 | 0.28775 | seed_within_sequence |
| original | constructive | Novel60_A | C2_minus_C0 | error_task_norm | 20 | 0.1093 | -0.56721 | 0.7277 | seed_within_sequence |
| original | constructive | Novel60_A | C2_minus_C0 | raw_ce | 20 | 0.047779 | -0.30243 | 0.41012 | seed_within_sequence |
| original | constructive | Novel60_A | C3_minus_C1 | closure_task_norm | 20 | -0.38643 | -0.81985 | -0.0080482 | seed_within_sequence |
| original | constructive | Novel60_A | C3_minus_C1 | error_task_norm | 20 | 0.37945 | -0.027114 | 0.83425 | seed_within_sequence |
| original | constructive | Novel60_A | C3_minus_C1 | raw_ce | 20 | 0.19262 | -0.01506 | 0.40433 | seed_within_sequence |
| original | constructive | Novel60_B | C2_minus_C0 | closure_task_norm | 20 | 0.15747 | -0.18252 | 0.50662 | seed_within_sequence |
| original | constructive | Novel60_B | C2_minus_C0 | error_task_norm | 20 | -0.12057 | -0.75556 | 0.47896 | seed_within_sequence |
| original | constructive | Novel60_B | C2_minus_C0 | raw_ce | 20 | -0.13735 | -0.38666 | 0.082854 | seed_within_sequence |
| original | constructive | Novel60_B | C3_minus_C1 | closure_task_norm | 20 | -0.062378 | -0.40381 | 0.26568 | seed_within_sequence |
| original | constructive | Novel60_B | C3_minus_C1 | error_task_norm | 20 | -0.0014163 | -0.53295 | 0.54198 | seed_within_sequence |
| original | constructive | Novel60_B | C3_minus_C1 | raw_ce | 20 | 0.20332 | -0.19516 | 0.65451 | seed_within_sequence |
| original | constructive | NovelMean | C2_minus_C0 | closure_task_norm | 20 | -0.031317 | -0.26624 | 0.18643 | seed_within_sequence |
| original | constructive | NovelMean | C2_minus_C0 | error_task_norm | 20 | -0.15675 | -0.54959 | 0.24648 | seed_within_sequence |
| original | constructive | NovelMean | C2_minus_C0 | raw_ce | 20 | -0.11303 | -0.40323 | 0.12588 | seed_within_sequence |
| original | constructive | NovelMean | C3_minus_C1 | closure_task_norm | 20 | -0.2244 | -0.55488 | 0.074574 | seed_within_sequence |
| original | constructive | NovelMean | C3_minus_C1 | error_task_norm | 20 | 0.18902 | -0.13349 | 0.53403 | seed_within_sequence |
| original | constructive | NovelMean | C3_minus_C1 | raw_ce | 20 | 0.19797 | -0.0296 | 0.43548 | seed_within_sequence |
| original | constructive | all_registered_raw_cues | dynamics_minus_static | held_out_MSE | 20 | -0.15213 | -0.24813 | -0.061037 | seed_within_sequence |

The dynamics-augmented model adds only prespecified mean closure and finite-horizon amplification features. All conditions and cues of a seed remain held out together. Incremental prediction is not causal mediation. Familiar acquisition source logs remain historical changing-weight observations; these final rollouts are not learning curves.

Evidence: tables/e1_transition_dynamics.csv, e1_seed_summary.csv, e1_static_convexity_audit.csv, e1_single_patch_replay.csv, e1_grouped_cv_predictions.csv, e1_effects_bootstrap.csv; figures/e1_accessibility.{png,pdf,svg}; audit/e1_historical_regression.csv. All inference resamples trained seed clusters within cohort. Teacher forcing fixes seven external tokens.



# E2 — Context information and cue geometry

The full finite registry contains 35 grid locations plus eight exact anchors and six cue-magnitude rays. Alpha=x−y and beta=x+y; all third-channel values are zero. Rounded grid locations and exact radical Novel anchors retain distinct identities. Raw norm, nearest-endpoint angle and Euclidean distance to the familiar segment are exported separately. Zero-cue angle is undefined.

| cohort | condition | comparison_kind | effect |
| --- | --- | --- | --- |
| canonical | High_monotonic | anchor_Mixed_minus_NoContext | -0.69166 |
| canonical | High_monotonic | exact_Novel_A_beta_sign | -1.0232 |
| canonical | High_monotonic | exact_Novel_B_beta_sign | -0.82445 |
| canonical | Low_monotonic | anchor_Mixed_minus_NoContext | -0.013875 |
| canonical | Low_monotonic | exact_Novel_A_beta_sign | -0.014063 |
| canonical | Low_monotonic | exact_Novel_B_beta_sign | -0.015447 |
| constructive | C0_canonical | anchor_Mixed_minus_NoContext | -0.74961 |
| constructive | C0_canonical | exact_Novel_A_beta_sign | -1.0952 |
| constructive | C0_canonical | exact_Novel_B_beta_sign | -0.78958 |
| constructive | C1_contrast_only | anchor_Mixed_minus_NoContext | 0 |
| constructive | C1_contrast_only | exact_Novel_A_beta_sign | 0 |
| constructive | C1_contrast_only | exact_Novel_B_beta_sign | 0 |
| constructive | C2_closure_only | anchor_Mixed_minus_NoContext | -0.85175 |
| constructive | C2_closure_only | exact_Novel_A_beta_sign | -1.2026 |
| constructive | C2_closure_only | exact_Novel_B_beta_sign | -0.95523 |
| constructive | C3_full | anchor_Mixed_minus_NoContext | 0 |
| constructive | C3_full | exact_Novel_A_beta_sign | 0 |
| constructive | C3_full | exact_Novel_B_beta_sign | 0 |

1240 C1/C3 same-alpha comparisons passed the frozen precision bounds; maximum logit difference 0.0003853, maximum CE difference 0.0003377. Unrestricted models are measured by the same comparisons without requiring an identity.

Same-alpha beta-sign comparisons are norm matched; same-norm/different-direction pairs and A/B magnitude rays are separately labeled. The beta finite difference is descriptive and does not establish psychological familiarity. No beta/rho pathway was trained. All grid points are repeated measurements of a seed, not independent networks.

Evidence: tables/e2_grid_transition.csv, e2_grid_seed.csv, e2_matched_geometry_controls.csv, e2_projection_invariants.csv, e2_controls_bootstrap.csv; figures/e2_context_grid.{png,pdf,svg}. Heatmap color scales are condition-specific, and every panel shows across-seed means.



# E3 — Causal separation × closure

Implementation status: validated. Scientific status: insufficient_familiar_support. Evidence status: insufficient_familiar_support.

No frozen candidate satisfied every prospectively declared familiar-only pilot criterion. All 48 attempted settings and their five reserved seeds are retained, including failures. No confirmatory model or protected probe was evaluated for this factorial. This is insufficient familiar support, not evidence against recurrent closure. The registered grid was not expanded.

Evidence: manifests/e3_selected_config_manifest.json, tables/e3_pilot_ab_only.csv, e3_pilot_candidate_audit.csv, figures/e3_pilot_feasibility.*; all available e3 training, feasibility, transition and contrast tables are hash-indexed in manifests/e3.json.



# E4 — Fixed sequence replication

Design-only; evidence_status=blocked_by_e3. No feasible frozen E3 familiar-only configuration; no replacement model or sequence-dependent calibration permitted.

| sequence_id | tokens | valid | replaced_equivalent_candidate |
| --- | --- | --- | --- |
| original | [0, 1, 2, 0, 3, 1, 2, 3] | True | False |
| S1 | [0, 1, 0, 1, 2, 3, 2, 3] | True | False |
| S2 | [0, 1, 0, 2, 1, 3, 2, 3] | True | False |
| S3 | [0, 1, 2, 0, 3, 2, 1, 3] | True | False |

The registered design retains 20 paired seeds for each sequence and all four factorial arms, with no per-sequence coefficient search. No replication model was trained; empty outcome tables contain no imputed measurements.



## Plain-language interpretation

A good familiar midpoint output is mathematically expected from the shared target and linear head. E1 quantifies whether recurrent steps reach and sustain that state, while keeping exact nonlinear displacement distinct from local linearization. E2 measures contextual information that the bottleneck necessarily discards; it does not identify a psychological novelty signal. Strong causal claims about separation require both successful process manipulation and comparable familiar learning in E3. When familiar-only feasibility fails, that causal question and its sequence replication remain unresolved. No architecture, cue axis, coefficient grid or output head was changed to obtain a preferred probe result.

All predictive validation holds out complete seed groups. Bootstrap draws resample seeds within fixed sequence. Replication is reported sequence by sequence; three additional sequences cannot establish architecture-wide universality. Negative or blocked results terminate the registered program.

## Provenance and reproduction

Each stage manifest records environment, source/config hashes, checkpoint identity, output hashes and run registry. See audit/ and manifests/ for exact before/after integrity evidence and tables/claim_source_map.csv for figure/claim sources. Reproduction instructions are in docs/dynamical_generalization.md.

