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

