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

