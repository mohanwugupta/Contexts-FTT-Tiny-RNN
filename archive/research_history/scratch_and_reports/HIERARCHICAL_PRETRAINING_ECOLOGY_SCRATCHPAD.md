# Hierarchical Compositional Pretraining Ecology Scratchpad

Living implementation record for
`PRD_hierarchical_compositional_pretraining_ecology.md`. This records frozen
engineering decisions and validation, not scientific conclusions.

## Frozen decisions

- P0/P1/P2 all use the existing `SRNEngine`: 42 inputs, eight tanh Elman hidden
  units, next-token head, and two-parameter monotonic execution readout. The
  experiment changes weight history only.
- Synthetic visible evidence has 34 values (2 cue-like and 32 layout-like), and
  is reordered behind the existing eight ordinal channels to produce the exact
  7x42 input contract. No world identifier or latent factor is supplied.
- World types cycle deterministically through shared-structure,
  context-specific, and compositional families so every finite pilot contains
  all three ecology components.
- In J=2 worlds, four observed context instances may reuse a factor combination
  with independent visible noise. This permits four matched training contexts
  while reserving two of the four unique combinations as genuinely held out.
- P1 is produced from the already-generated P2 world by a fixed non-identity
  cyclic permutation of visible-context assignments per world x seed. It thus
  preserves exact visible and task marginals, sequences, and update counts.
- The two frozen internal probes are an unseen factor combination and an unseen
  new family member. Both are scored against the structured generating rule;
  neither updates parameters.
- The primary first human Mixed probe is checkpoint 20, immediately after the
  first complete A10/B10 cycle. The late probe is checkpoint 170. Smoke mode
  uses checkpoint 20 for both epoch labels solely to test the machinery.
- Pretrained weights and readout carry into the human phase, transient hidden
  state resets naturally at sequence boundaries, and Adam is reconstructed with
  empty moments before the first human A/B update.
- Pilot admission uses internal held-out P2-versus-P1 performance only. Human
  Mixed/Novel probes never control hyperparameters, checkpoints, or execution.
- M1--M4 are deliberately outside this initial implementation because the PRD
  permits them only after a confirmatory P2 success.

## Validation log

- 2026-09-09: AST-parsed all eight new package/runner files successfully.
- 2026-09-09: generator validation passed for all smoke seed/world pairs: no
  exact human cues, layouts, or motor sequence; held-out combinations were
  disjoint; structured mappings reconstructed; P1 mappings were derangements.
- 2026-09-09: P1/P2 smoke training used exactly 24 optimizer steps each for the
  three-world pilot and 32 each for the four-world confirmatory check.
- 2026-09-09: the final smoke pilot passed its internal-only admission rule
  (P2 NLL 5.9781 vs P1 6.0124; accuracy tied at 0.2381), allowing the downstream
  engineering paths to be exercised. These values are not scientific results.
- 2026-09-09: staged smoke completed all P0/P1/P2 human A/B branches, first and
  late frozen probes, canonical state-space fits, bootstrap contrasts, saved
  checkpoints, all required CSVs, report, manifest, and nine figures.
- P0 at checkpoint 20 was independently compared with the existing
  `train_geometry_model("G2_shared_x", ...)` implementation: maximum absolute
  error was exactly zero for transition NLL, confidence, and normalized RT.
- Frozen probe/hash assertions and fresh-Adam assertions passed. Mixed and
  Novel human training counts remained zero for every run.
- New PRD tests: **9/9 passed**. Repository-wide regression: **237 tests passed
  in 56.17 seconds** with bytecode and pytest caches disabled.
- Visually inspected the ecology schematic, first-Mixed profile panel, and
  compact effect summary. Figure 9 uses green as within-column favorability,
  reversing the color direction for Novel NLL deficit while retaining raw-value
  annotations.
- The full 100-world pilot and 1,000-world/20-seed confirmatory experiment were
  intentionally not launched during implementation validation.
- Full runs emit flushed progress at 20 evenly spaced world checkpoints within
  every condition/seed, in addition to stage and seed completion messages.
