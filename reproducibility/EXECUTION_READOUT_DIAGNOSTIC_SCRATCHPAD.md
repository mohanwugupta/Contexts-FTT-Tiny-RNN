# Execution readout diagnostic scratchpad

## Scope

Diagnose whether the objective-search failure is caused by the unrestricted
hidden-state execution readout. The completed `execution_objective_search`
package and all of its outputs are frozen. No human data or 20-seed paper run is
in scope.

## Decisions

- Built a separate additive `execution_readout_diagnostic` package. The frozen
  objective-search gate and analytical-optimum functions are imported without
  modification; their source hashes are recorded before and after every run.
- Fixed one normative mapping for all conditions: quadratic confidence risk with
  `c_time=.5`, `c_risk=1`, `gamma=2`, and `lambda_exec=.2`. This was chosen from
  the nonhuman analytical calibration because it has a broad central dynamic
  range and is the least aggressive tested mapping that previously passed the
  acquisition-direction gate. It was not selected using human RT.
- All three conditions receive identically seeded scale-2 recurrent and sequence
  parameters. The requested unrestricted execution head alone has its eight
  weights and bias reset to exactly zero.
- Deterministic speed is the detached objective optimum and has no trainable
  readout parameters. The monotonic policy has exactly two scalar parameters,
  `a` and `b`, with positive slope enforced by `softplus(a)`. The unrestricted
  condition retains the existing hidden-state linear-plus-sigmoid readout.
- All conditions train for the same 170 A10/B10 sequences over seeds 0--4. The
  A--E gate implementation and thresholds are exactly those frozen during the
  objective search.
- State-space fits are firewalled: only a condition passing all A--E gates can
  reach the canonical `state_space_bridge.state_space` Lump/Cross/Split fitter.
  Fits use the five-seed mean normalized-RT trajectory, matching the average-
  trace convention in `state-space_SRN.Rmd`.
- A strict L-BFGS-B line-search warning is not silently treated as convergence.
  Any warned fit is cross-checked against the repository's independent
  `srn_validation.state_space` port and is marked verified only when that port
  converges and its BIC and predictions agree within `1e-6`.

## Progress

- Audited the frozen objective-search trainer/gates and both validated Python
  ports of the R no-fatigue state-space models.
- Implemented the three policies, matched initialization audit, training loop,
  unchanged gates, conditional state-space fitting, figures, report, and CLI.
- Completed the full five-seed screen. All three readouts passed every A--E
  gate. Mean late-minus-early normalized RT changes were `-.294`
  (deterministic), `-.383` (monotonic), and `-.163` (zero-initialized
  unrestricted); the unrestricted direction held in exactly 4/5 seeds.
- Because all conditions passed, fitted Lump/Cross/Split to each mean RT trace.
  Cross won for all three readouts. Cross beat Lump by Delta-BIC `102.44`,
  `77.68`, and `72.40`, and beat Split by `15.90`, `13.16`, and `52.04`,
  respectively. Estimated `A_cross` values were `.0158`, `.0130`, and `.0199`.
- The unrestricted Cross fit triggered the canonical port's strict line-search
  warning at an `A_B=.999` boundary. The independent established port converged
  to the same fit (absolute BIC difference `<4e-9`, maximum prediction difference
  `<3.1e-7`), so it is explicitly marked `convergence_verified=True` while the
  original `converged=False` flag remains visible.
- Focused readout plus frozen-objective tests passed `17/17`; the complete
  repository suite passed `103/103`. Compilation and the final deliverable audit
  passed for 2,550 sequence rows, 17,850 token rows, 15 matched initializations,
  and five PNG/PDF figure pairs.
- `reproducibility/verify_final_outputs.py` confirmed all 112 frozen final CSVs
  remained bitwise identical. Objective-search hashes matched before/after,
  human/Mixed/Novel row counts were zero, and the 20-seed rerun flag remained
  false.
