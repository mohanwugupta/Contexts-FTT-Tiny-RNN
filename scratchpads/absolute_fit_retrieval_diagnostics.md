# Absolute-fit retrieval diagnostics scratchpad

## Locked decisions

- This PRD validates R0--R3 without changing their scientific definitions or refitting the familiar state-space family.
- R0 retains one A/B latent draw for a complete seven-transition profile. Transition-wise likelihood rows are explicitly labeled as a descriptive marginal decomposition and never used to refit q.
- The long-form experiment table combines exact frozen familiar predictions from `human_state_space_latent_states.csv` with fitted predictions for every eligible probe transition.
- Posterior prediction samples empirical participant anchor profiles, then applies the fitted Gaussian residual model. R0 holds endpoint identity constant for all seven transitions; R1 uses its fitted scalar interpolation.
- Posterior adequacy is summarized transparently by whether five observed aggregate statistics fall within model-generated 95% intervals. This is descriptive coverage, not a formal hypothesis test.
- Anchor uncertainty refits both models and repeats matched participant-fold prediction for every anchor draw. The saved delta is predictive LL(R1) minus predictive LL(R0).
- Mandatory figures A--J are written at the analysis-output root. They are inference-gated and therefore render even when `--no-plots` suppresses the older optional figure suite.
- The report cannot be written unless every mandatory CSV and figure exists.

## Progress

- [x] Implement long-form familiar/probe fit table.
- [x] Implement profile-level R0/R1 posterior simulation and geometry.
- [x] Implement participant and transition likelihood decompositions.
- [x] Implement distribution, participant-bootstrap, and anchor-uncertainty outputs.
- [x] Implement mandatory figures A--J and completeness gate.
- [x] Complete real-data smoke validation. The gate passed with 104,202 familiar rows, 33,516 probe rows, 5,000 posterior profiles per model, all mandatory CSVs, and all ten figures. Visual inspection confirmed readable plots. The smoke diagnostics showed R0 3/5 and R1 2/5 aggregate posterior checks in range, so relative R0 superiority is not reported as sufficient mechanistic support.
