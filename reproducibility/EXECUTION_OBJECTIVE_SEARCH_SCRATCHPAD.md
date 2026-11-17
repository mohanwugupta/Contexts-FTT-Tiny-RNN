# Execution objective search scratchpad

## Scope

Discover a minimal uncertainty-grounded execution loss before any expensive
paper rerun. Existing v2.1, corrected scale-2, human, retrieval, low-rank, and
causal outputs remain frozen.

## Implementation decisions

- Built a separate `execution_objective_search` package rather than editing the
  registered v2.1 trainer or speed-accuracy objective.
- Reused the exact 8-unit `DualHeadElmanSRN`, four-token input, three-context
  basis, learning rate `.005`, and A10/B10 schedule. Scientific screening fixes
  High initialization scale to `2.0`, five seeds, and all 170 sequences.
- All candidates use `c_time*(1-v) + c_risk*uncertainty*v^gamma`. Candidate A is
  quadratic correct-confidence risk; B uses steeper exponents 3 and 4; C uses
  normalized full-distribution entropy; D uses one minus the top-two probability
  margin.
- The uncertainty tensor is detached inside the execution loss. Cross-entropy
  continues to train sequence logits normally, but the execution term cannot
  lower itself by directly manipulating its certainty statistic.
- Candidate B does not duplicate gamma 2 because that is exactly Candidate A.
- The calibration grid varies `c_time` in `{.05, .10}` and `c_risk` in
  `{.50, 1.0}` with `lambda_exec=.20`, yielding 20 interpretable candidates.
- Calibration gates are fixed before neural screening: positive central slope,
  at least `.20` speed range from certainty `.30` to `.90`, a genuine
  intermediate optimum, and no more than 50% central saturation.
- For the entropy family, monotonicity is evaluated from four-choice chance
  confidence (`.25`) upward. Below-chance correct-token probability is not a
  useful certainty range because entropy necessarily increases on the way to a
  uniform distribution.
- Neural gates use no human outcomes. Gate E accepts either switch-specific RT
  jumps exceeding matched within-context changes or a context coefficient after
  controlling progress and within-block position.
- Selection prioritizes the simplest passing family (A before B before C before
  D), then exponent, then a nonhuman quality score based on objective tracking,
  trained speed range, saturation, and seed consistency of RT direction.
- The search pipeline has no code path that starts the 20-seed confirmatory run.
- The initial 20-candidate screen retained 17 analytically and screened all 17.
  Every candidate failed the neural gates: sequence learning was strong, but only
  20–60% of seeds showed RT improvement and objective tracking was inconsistent.
- Because all original families failed, added a targeted second-stage Candidate-A
  scale check. Multiplying both costs by the same factor preserves `v*(p)` while
  strengthening execution gradients against saturation induced by scale-2
  initialization. Ratios `.05`, `.10`, and `.20` are crossed with two stronger
  absolute scales; no human outcome informs this follow-up.
- The stronger Candidate-A follow-up also failed: only 20--60% of seeds showed
  RT improvement and only 40% showed positive policy-to-optimum tracking. This
  rules out a merely underscaled direct economic cost.
- The PRD permits expansion only after all A--D families fail. Added one minimal
  Candidate E at that point: compute Candidate A's analytical risk optimum from
  the network's detached confidence and minimize Bernoulli cross-entropy from
  actual speed to that soft optimum. The speed-logit gradient is `v-v*`, avoiding
  the extra sigmoid-saturation factor. This remains fully nonhuman: `v*` is
  objective-implied, not an observed RT or external competence state.
- Candidate E was calibrated over `c_time/c_risk=.05--.75`; `.75` is the largest
  tested ratio retaining an intermediate optimum and staying under the fixed
  central-saturation threshold. Ratios `.50` and `.60` passed acquisition Gate
  A. Ratios `.70` and `.75` also passed objective-tracking Gate C. None passed
  grounding Gate B: only 3/5 seeds had the correct confidence--RT direction,
  with arbitrary context offsets persisting in two scale-2 initializations.
- Diagnostic execution weights `.50` and `1.0` did not improve that 3/5 seed
  grounding consistency and were therefore not added to the final fixed-weight
  Candidate-E sweep.
- No candidate passed all gates. The acceptance criteria were not relaxed, no
  objective was selected, and no 20-seed confirmatory run was started. Under
  the frozen scale-2 initialization and 170-update exposure, changing the loss
  alone did not reliably overcome random execution-head/context offsets.

## Progress

- Read the full pasted PRD and audited the existing execution model, trainer,
  calibration, grounding gate, transition-cost code, and corrected-scale wrapper.
- Added analytical calibration, isolated training, five gates, seven figures,
  required CSVs, configuration audit, report, CLI, and tests.
- Completed the prespecified A--D screen, stronger-cost diagnostic, and the
  permitted post-failure policy-objective screen. Final status is a documented
  null search rather than an unstable selected objective.
- Focused objective-search tests passed `10/10`; the complete repository suite
  passed `96/96`. Bytecode compilation, smoke CLI, post-failure CLI, required
  deliverable assertions, figure counts, and no-confirmatory-run assertions all
  passed.
- `reproducibility/verify_final_outputs.py` confirmed all 112 frozen final CSVs
  remained bitwise identical. The existing `ftt_srn_execution_v2` package was
  imported but not edited; all new work is additive.
