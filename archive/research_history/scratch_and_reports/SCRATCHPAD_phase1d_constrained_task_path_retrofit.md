# Scratchpad: Phase 1D constrained task-path retrofit

## Frozen decisions

- Phase 1D is a new package and output tree; Phase 1C artifacts and checkpoints are read-only.
- Recover the sequence, eight non-midpoint training cues, 170-update cadence, wrong-path seed rule,
  and all seven success thresholds from the Phase 1C manifest. Fail on disagreement.
- Train exactly the 136 recurrent parameters exposed in Phase 1C. Keep the 36 readout parameters
  frozen and retain the fixed-zero Novel input column.
- Define the equality constraint using the 56 centered endpoint logits (2 contexts × 7 transitions
  × 4 logits). Redundancy from centering is handled by relative-threshold SVD.
- Define the correct and wrong targets once from the source checkpoint and hash them. Never rebuild
  targets from moving endpoints.
- Select numerical damping and learning rate globally on calibration seeds using only non-midpoint
  path residual, endpoint constraint residual, familiar guardrails, and numerical stability.
- Select checkpoints from the registered Phase 1C cadence using the same firewall. Midpoint values
  are evaluated only after the checkpoint is fixed.
- Compare Phase 1C using its behavior-blind per-seed stable-cell rule; use alpha=10, anchor=30 only
  as the aggregate plot representative.
- Save seed-complete partials so a long 20-seed run resumes without repeating finished seeds.
- For the PRD's descriptive decision tree, define "broad" as at least 50% of seeds and "high local
  reachability" as mean Q_local at least 0.50. These thresholds classify the outcome only; they do
  not tune training or alter the Phase 1C-comparable success gate.

## Progress

- [x] Read PRD and trace Phase 1B/1C target, patch, evaluation, and provenance interfaces.
- [x] Implement exact Jacobians and constrained update invariants.
- [x] Implement calibration, correct/wrong training, checkpoint selection, and resume.
- [x] Implement master/method tables, all 21 required figures, and 15-section report.
- [x] Implement optional from-scratch Stage 2 behind a disabled flag.
- [x] Add RED→GREEN tests and run a complete smoke pipeline.

## Smoke-test findings

- Exact source-seed Jacobian dimensions were 56 × 136 for endpoint constraints and 224 × 136
  for the eight-cue path residual; seed 0 had rank 43 and 93 free local dimensions under the
  registered relative SVD convention.
- The first smoke exposed an over-strict raw constraint acceptance tolerance that rejected every
  update. This was corrected before validation; the final smoke accepted 10/10 updates for both
  correct and wrong targets while endpoint task drift stayed below 0.001 of baseline scale.
- The correct-path smoke reduced normalized non-midpoint training residual from 1.000 to 0.930;
  the wrong-path condition reached 0.958. These are implementation checks, not scientific results.
- All 17 invariant audits passed, all 6 primary, 5 cue-sweep, and 10 sequence/training figures were
  produced, and resume reused the completed seed without retraining.
- Six Phase 1D tests passed (five unit/contract tests and one end-to-end integration test). Eight
  adjacent Phase 1C/final-synthesis regression tests also passed.
- The full 20-seed run was intentionally not launched; use the registered runner command after review.
