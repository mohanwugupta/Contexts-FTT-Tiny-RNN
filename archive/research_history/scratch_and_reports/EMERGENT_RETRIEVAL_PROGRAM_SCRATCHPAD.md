# Emergent Retrieval and Context-Inference Program Scratchpad

Living record for `PRD_emergent_retrieval_context_inference_program.md`.
It documents frozen choices and engineering validation, not empirical claims.

## Frozen implementation choices

- All variants use coordinate arrays by file order only. Registry names/glyphs
  are audit metadata, never model inputs.
- The common evidence vector is 34-D: the validated 2-D cue followed by the two
  16-D G2 shared-x layout banks. E1--E4 receive only this numeric vector; context
  names are output metadata.
- The neutral test evidence has zero cue, pooled human x, zero y, and duplicates
  that 16-D shared layout across the existing familiar banks. It is never trained.
- E2 uses sequence-level truncated recurrence: its two-dimensional state carries
  across acquisition sequences, while gradients are truncated at each optimizer
  step. Both reset and carry probe policies are frozen diagnostics.
- E3 initializes from current evidence and updates its two-dimensional state
  after each transition with differentiable correct-target NLL. Parameters remain
  frozen during probes; latent-state updates are inference, not learning.
- C1 learns one positive RBF bandwidth from repeated A/B evidence only. C2 adds
  the maximum-likelihood A/B transition matrix from the fixed curriculum. Both
  are labeled `averaging_imposed=True` in every output.
- The primary comparison probe policy is reset. Ten-A, ten-B, and reset histories
  are separate diagnostics and never determine model selection.
- Confirmatory extension depends only on familiar A/B gates. Probe results never
  select hyperparameters, architectures, temperature, latent size, or bandwidth.

## Validation log

- Implemented the isolated `emergent_retrieval_program/` package, two frozen
  JSON contracts, CLI runner, nine figures, report generator, and T1--T12 tests.
- 2026-09-08: AST-parsed all 11 new Python files without writing bytecode.
- 2026-09-08: independently trained E1--E4 for one 20-sequence smoke seed;
  every variant emitted 20 acquisition rows, 35 frozen probe transitions, and
  unchanged post-probe parameter/optimizer hashes.
- 2026-09-08: exercised C1/C2 construction and verified each emits 35 frozen
  probe transitions with `averaging_imposed=True` and unchanged V0 weights.
- 2026-09-08: exercised reset/10-A/10-B history branches for C2/E2/E3; all 126
  transition-level diagnostic rows were produced without weight mutation.
- 2026-09-08: exercised all E3 ablations, including the separately retrained
  A/B-only random-frozen context module branch. A1--A5 all emitted both
  transition- and seed-level results.
- 2026-09-08: `python -m pytest -q -p no:cacheprovider
  tests/test_emergent_retrieval_program.py` passed 7/7 tests.
- 2026-09-09: repository-wide regression suite passed: **228 tests** in
  74.83 seconds with the pytest cache disabled (the workspace is read-only to
  bytecode/cache artifacts in this session).
- 2026-09-08: staged CLI smoke run completed through C2 and generated every
  required CSV, report, manifest, and all nine PNG figures. Visual inspection of
  the gate dashboard and 2-D latent-context panels found labels, legends, and
  scales readable.
- Full 170-sequence/20-seed confirmatory fitting was intentionally not launched
  during implementation validation; the CLI is ready for that run.
