# Phase 1E implementation scratchpad

## Frozen scientific decisions

- Source checkpoints, sequence, 20 seeds, 170 attempted updates, readout, correct target, wrong-target rule,
  nonmidpoint training cues, and Phase 1C/1D complete gate are unchanged.
- Endpoint and path residuals are divided by the source A/B centered-logit RMS scale. This applies the same
  source-relative unit to both task-space quantities without using probe outcomes.
- One deterministic numerical setting is used globally: initial trust radius `1e-3` of source recurrent
  parameter norm, relative constraint damping `1e-6`, and relative tangent regularization `1e-4`.
- Trust ratio rules are fixed before probe evaluation: reject/shrink below `.25`, accept/hold from `.25` to
  `.75`, accept/enlarge at or above `.75`. Shrink is `.5`; enlargement is `1.5`.
- Nonlinear restoration is capped at four iterations. Failure to restore the scaled endpoint norm to `5e-3`
  rejects the proposal.
- Checkpoint selection uses registered path residual, endpoint feasibility, familiar raw loss, A/B separation,
  and finite numerical status only. Probe rows are generated after selection and are marked read-only.

## Mechanical failure classification (first matching rule)

1. Complete gate passes -> `Complete success`.
2. Nonfinite result, or zero accepted updates with numerical exceptions -> `Numerical failure`.
3. Frozen Phase 1D `Q_local < .50` -> `No local access`.
4. Constraint/guardrail rejection fraction `>= .50` -> `Constraint-limited`.
5. Registered path-loss reduction `< .10` -> `Optimization stagnation`.
6. Otherwise -> `Improved but below complete gate`.

The order is fixed here before the 20-seed Phase 1E run.

## Implementation notes

- The completed Phase 1D output is reused read-only for paired metrics, local access, baselines, and raw-loss
  overlays; Phase 1D is not rerun.
- Seed partials and selected Phase 1E checkpoints are resumable and configuration-hash guarded.
- A parameter-version ledger invalidates Jacobians after accepted updates.
- Mandatory time-series panels use raw cross-entropy and never normalize the plotted values.
