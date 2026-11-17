# Scratchpad: Phase 1C Pareto path sufficiency

## Frozen implementation decisions

- Use the PRD's compute-reduced grid with `beta = gamma` and anchor strengths
  `{0.1, 1, 10, 30}`; retain every preregistered alpha.
- Treat the entire seed × alpha × anchor surface as the result. No midpoint-selected winner,
  early stopping, or hyperparameter optimization is permitted.
- Construct one deterministic centered, norm-matched orthogonal wrong displacement for every
  seed × training cue × sequence position. Save its seed and numerical audits.
- Reuse the exact alpha=0 anchor-only optimization trajectory across P1/P2/P3/P4 labels because
  their objectives and gradients are mathematically identical at alpha zero; record the reuse.
- Normalize High-family path residuals and rescue by the corresponding frozen High seed. Frozen
  Low is self-normalized descriptively and receives no High rescue denominator.
- Define a substantial correct-path region before the run as at least 10% feasible P1 seed-cells,
  with at least 50% of seeds admitting a feasible solution.
- Define wrong-path specificity before the run as mean correct-minus-wrong rescue advantage at
  least 0.20 and positive advantage in at least 60% of exact matched pairs.
- Save sequence-level rows for every cell, while figures use a readable preregistered subset plus
  the descriptively identified best stable frontier cell.

## Progress

- [x] Read PRD and freeze unspecified frontier/specificity definitions.
- [x] Implement deterministic wrong targets and Pareto training.
- [x] Implement metrics, specificity matching, dose-response, and feasibility analysis.
- [x] Implement all required tables, figures, report sections, progress output, and seed resume.
- [x] Complete smoke and adjacent regression audit (27 tests passing).
- [ ] Run the full 20-seed sweep in the user's terminal; smoke outcomes are not scientific results.
