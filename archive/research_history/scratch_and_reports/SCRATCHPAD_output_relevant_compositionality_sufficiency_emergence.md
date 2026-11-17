# Scratchpad: output-relevant compositionality sufficiency and emergence

## Frozen decisions

- Phase 1 trains the unchanged 8-unit tanh Elman architecture for exactly 170 familiar A/B
  updates. Synthetic interpolation cues enter only the geometry regularizer, never task CE.
- Predeclare “substantial” as Mixed loss ≤75% and midpoint output/logit curvature ≤50% of
  the newly trained High baseline. Separation must remain ≥90%, familiar loss ≤110%, q within
  .15 of .5, and separation ≥2× the Low baseline. These thresholds are frozen in config.
- Derive the separation floor only from familiar A/B states in the existing 20 High checkpoints.
- Select curvature/separation weights only from familiar loss, familiar separation, and synthetic
  cue curvature. Never place Mixed/Novel loss in the selection table.
- Use the raw second difference in the training penalty, matching the Phase 1 PRD equation;
  report derivative-scaled curvature in evaluation for continuity with the prior diagnostics.
- Phase 2 is a hard conditional branch. If Phase 1 fails any criterion, write explicit blocked
  placeholder CSVs and do not construct, train, or evaluate Phase 2 models.
- Phase 2 uses eight latent sequences over the same four-token vocabulary and six synthetic
  contexts in the first two context dimensions. Additive and rank-2 multiplicative surface
  transforms preserve the 4+3 input and four-way output architecture.
- Structured/shuffled conditions share exact sequence/context/heldout exposures. Shuffling changes
  only the stable context→transform mapping. Pretraining uses no curvature/separation penalties.
- Every human-task transfer receives a reset Adam optimizer and exactly 170 A/B-only updates;
  Mixed and Novel receive zero optimizer steps.

## Progress

- [x] Read PRD and freeze quantitative gate definitions before training.
- [x] Implement Phase 1 differentiable output, whole, and null curvature losses plus separation floor.
- [x] Implement A/B-only hyperparameter-selection firewall, final evaluation, gate, and patching.
- [x] Implement gated Phase 2 structured/shuffled additive and multiplicative ecologies.
- [x] Implement held-out pair validation, formal models, internal gate, and identical human transfer.
- [x] Implement output contracts, reports, and conditional figures.
- [x] Run unit/smoke tests and correct defects (18 focused tests passing).
- [x] Run full Phase 1. The prespecified gate failed, so Phase 2 was correctly not run.

## Final Phase 1 decision

- Selected weights from the three-seed firewall: C1 `(0.0001, 1.0)`, C2
  `(0.0001, 1.0)`, and C3 `(0.0003, 0.1)` for `(curvature, separation)`.
- C1 retained familiar performance and separation, and Mixed q remained near .5.
- C1 did not meet the prespecified sufficiency thresholds: Mixed loss was 112.6% of C0,
  output curvature was 72.6% of C0, and logit curvature was 73.3% of C0.
- `PHASE1_GATE = FAIL`; Phase 2 artifacts contain the explicit
  `not_run_phase1_failed` status. No Phase 2 model was trained and no human data were used.
