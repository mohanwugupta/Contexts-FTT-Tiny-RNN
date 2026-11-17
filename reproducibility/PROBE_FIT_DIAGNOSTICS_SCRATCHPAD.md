# Probe fit diagnostics scratchpad

## Goal

Create direct observed-versus-predicted checks for the human probe retrieval
models and low-rank linear-algebra analysis, analogous to the state-space
trajectory overlays.

## Decisions

- Treat the v3 retrieval model and v3.1 linear-algebra analysis as related but
  distinct estimators. The registered v3 `M3_low_rank` model used rank 5 in all
  folds under its frozen 90% variance rule. The later v3.1 analysis independently
  selected rank 2 using the smallest-rank-within-one-SE rule.
- Use only frozen v3 held-out predictions for the retrieval figures; do not refit
  or overwrite any registered result.
- The frozen v3.1 row-level file retained unsigned error magnitudes but not signed
  observed/predicted values. Reconstruct signed predictions with the exact frozen
  v3.1 implementation, then require row-wise squared and absolute errors to match
  the frozen file within `1e-10` before plotting.
- Show observation-level and participant-balanced epoch-by-transition fit. The
  latter answers the visual curve-shape question without allowing participants
  with more trials to dominate.
- State-space metrics are displayed only in a separate reference panel. They are
  in-sample fits to aggregate SRN training trajectories, whereas the probe models
  are held out and target human log RT or midpoint deviation. Their R-squared
  values are not a direct equivalence test.
- Preserve the earlier scientific caveat: the corrected SRN failed the execution
  direction gate, so its state-space RT overlay is exploratory.

## Progress

- Mapped frozen v3 and v3.1 prediction contracts.
- Confirmed 8,232 held-out Mixed observations and 3,780 held-out Novel
  observations per primary retrieval model.
- Added a standalone read-only diagnostics package, CLI, tests, figures, metrics,
  hashes, and reconstruction audit without changing any model-fitting code.
- Visual inspection showed that the low-rank models capture the transition shape,
  while epoch-separated curves retain meaningful mean-level offsets. Added an
  explicit epoch-bias table and used a neutral figure title rather than claiming
  a uniformly close fit in advance.
- Final verification: all five diagnostic figures were visually inspected; the
  new test module passed 5/5 tests; the complete repository suite passed 86/86;
  and `verify_final_outputs.py` confirmed all 112 frozen final CSVs remained
  bitwise identical.
