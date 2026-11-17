# Scratchpad: Phase 1B direct task-space path sufficiency

## Frozen decisions made before midpoint evaluation

- Start every D1–D5 run from its seed's untouched `High_monotonic` checkpoint; D0 and D6
  are read-only baselines.
- Freeze `sequence_head.weight` and `sequence_head.bias` exactly. Only the existing recurrent
  input/recurrent weights and biases are optimized.
- Use the eight PRD training cues and structurally reject lambda=.50 in the primary path.
- Normalize endpoint hidden/logit RMSE by the seed-local original A/B endpoint RMS separation.
  Predeclare 10% as the maximum clean endpoint drift for each.
- Define a strong selection-lambda path reduction as <=35% of baseline, matching the held-out
  residual gate. Choose the smallest alpha attaining it while passing endpoint criteria.
- Select D1–D3 only from familiar task loss, endpoint metrics, and non-midpoint path error.
  D4 reuses the Phase 1 output-curvature weight (`1e-4`) and D5 inherits D1's anchor weights.
- Evaluate complete beta×gamma slices in ascending alpha order and prune larger alpha values once
  the first eligible strong-reduction slice is found; those values cannot win the smallest-alpha rule.
- Use the unchanged registered 170-sequence A10/B10 schedule and a fresh Adam optimizer.
- The descriptive full experiment replay is the 170 familiar sequences followed by frozen
  Mixed, NoContextProbe, and Novel probes. NoContextProbe and Novel are marked OOD under the
  current encoding and never enter selection.
- Sequence-level rows always aggregate exactly the seven trained next-token transitions.

## Progress

- [x] Read the PRD and freeze unspecified tolerances/selection rules.
- [x] Implement checkpoint caching, frozen-head fine-tuning, direct path objectives, anchors,
  curvature and anchor-only controls.
- [x] Implement held-out midpoint, dense cue-sweep, patching, rescue, and gate calculations.
- [x] Implement complete familiar/full-experiment sequence replays and all required plots.
- [x] Complete RED/GREEN tests and retained smoke run (13 focused/adjacent tests passing).
- [x] Run the expanded adjacent regression audit (23 tests passing total).
- [ ] Run the 20-seed final experiment. It was intentionally left for the user's terminal after
  the implementation and smoke contract passed; the smoke gate is not a scientific result.
