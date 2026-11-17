# Structural Prior Curriculum Scratchpad

Living implementation record for `PRD_structural_prior_curriculum.md`. This
contains frozen engineering decisions and validation notes, not empirical
claims.

## Frozen decisions

- E0--E3 are independent matched experiments from the same seed-specific
  initialization. The sequential curriculum is an abstraction/gating
  curriculum; weights do not silently carry between ecology levels. Pilot and
  confirmatory runs likewise start independently at their registered doses.
- Each world trains only `[0,0]`, `[1,0]`, and `[0,1]`; `[1,1]` has exactly zero
  optimizer updates and is scored once from the post-world frozen snapshot.
- Transition Pk denotes the kth next-token prediction. Factor 1 therefore
  changes response-token indices 2/3 (P2/P3 targets), and factor 2 changes
  indices 5/6 (P5/P6 targets). P1/P4/P7 remain shared; the initial item is not a
  scored transition.
- E0 uses the two ordinary cue slots directly. E1/E2 use six numeric surface
  dimensions spanning cue and layout slots; E1 retains factor-specific
  three-dimensional subspaces, while E2 permutes/sign-flips those subspaces.
  E3 uses dense world-specific 34-D factor realizations and may swap the two
  factor-to-task transition roles.
- The P1-style shuffled control at every stage is created from the exact
  structured world by a fixed three-cycle over the task assignments. Visible
  contexts, task marginals, difficulty, schedules, and optimizer exposure are
  byte-identical.
- Within-world training uses a deterministic random permutation of a balanced
  multiset containing ten copies of each observed combination (or the smoke
  dose). Hidden state resets per eight-token sequence; weights and Adam moments
  persist across worlds.
- A stage passes only after both pilot and independently initialized
  confirmatory gates pass. The most abstract confirmatory-passing stage is
  selected without human data.
- Human outputs are not created at all unless at least one ecology completes
  the confirmatory gate. First human probes occur after the first complete
  A10/B10 cycle at checkpoint 20.
- Human A1--A4 ablations run only if H2 meets the registered human-transfer
  success criterion.

## Validation log

- 2026-09-09: AST-parsed the eight new package modules and CLI runner.
- Generator checks reconstructed the factor-local task rule, verified P2/P3
  versus P5/P6 support, confirmed simultaneous `[1,1]` composition, and measured
  mapping correlation near 1.0 structured versus -0.5 shuffled while preserving
  exact evidence/task marginals.
- Frozen internal probes retained identical parameter and optimizer hashes and
  reported zero held-out optimizer updates.
- The final two-seed/three-world smoke pipeline attempted only E0, correctly
  failed its internal pilot gate, stopped before E1, and created no
  `curriculum_human_*` files or human figures. This is an engineering result,
  not evidence about the registered 100-world pilot.
- Direct downstream harness exercised H0/H1/H2 checkpoint transfer, fresh-Adam
  reset, 20-sequence A/B acquisition, first/late immutable probes, canonical
  state-space fits, interpolation/corridor tables, and Figures 6--10.
- Constructed and validated all eight success-contingent branches: A1 input
  reset, A2 recurrent reset, A3 output reset, and A4 alpha 0/.25/.5/.75/1.
- Visually inspected internal generalization, corrected last-reached gate panel,
  and first-Mixed human profile figures.
- New PRD tests: **9/9 passed**. Repository-wide regression: **246 tests passed
  in 59.09 seconds**, with bytecode and pytest caches disabled.
- The registered 100-world pilots and 1,000-world confirmatory stages were not
  launched during implementation validation.
