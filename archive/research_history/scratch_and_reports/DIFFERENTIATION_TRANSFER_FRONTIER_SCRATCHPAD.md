# Differentiation--Transfer Frontier Scratchpad

Living implementation record for `PRD_differentiation_transfer_frontier.md`.
This records frozen choices and engineering results, not scientific conclusions.

## Frozen implementation decisions

- The human coordinate JSON is coordinate-only. Its four arrays retain the
  previously authorized file-order assignment; registry names and glyphs do not
  enter the model.
- G2 shared-x coordinates, two 16-value familiar layout banks, eight ordinal
  channels, motor sequence `S-A-F-S-D-A-F-D`, A10/B10 curriculum, 170-sequence
  dose, Adam `.005`, execution objective, monotonic readout, and `RT=1-speed`
  remain frozen.
- Phase-A S0--S4 uses 50 features: 8 ordinal + 16 A-bank + 16 B-bank + 10 cue.
  A/B cues always have exactly five active values and unit L2 norm. Index sets
  are fixed in config and shared across seeds.
- Mixed uses the unit-normalized sum of the corresponding A/B cue vectors;
  Novel uses an all-zero cue. This preserves the existing mixture versus
  unknown-context semantics and was fixed before observing frontier outcomes.
- The repository initialization is not fan-in normalized: every weight matrix
  is sampled from `Normal(0, scale^2)`. Phase B reproduces this rule at H=8/16/32
  and logs fan-in, realized standard deviations, Frobenius norms, recurrent
  spectral radius, and frozen hidden activation norm.
- Phase-A selection is fully mechanical: S0 baseline; lowest five-seed High
  Mixed NLL deficit among all gate-passers, Novel as tie-breaker; S4 only if it
  passes. Phase-B cells advance from five to twenty seeds only after passing the
  same A/B gates.
- Mixed/Novel never update parameters. Deficits are computed within seed and
  transition relative to the worse A/B endpoint. Fast/Slow canonicalization is
  whole-profile, A/B-only, and diagnostic.
- The 2-D original G2 model is rerun through the untouched prior engine for the
  equivalence control. Full runs also compare those probe values against saved
  validated G2 outputs at tolerance `5e-7`.
- The PRD defines S0 equivalence qualitatively, not as exact NLL equality. A
  first full-dose control-only engineering check preserved every A/B gate and
  hidden-distance scale but exposed that a provisional ±1-NLL rule was not
  scale-appropriate. Before examining any S1--S4 selection outcome, transfer
  comparability was therefore frozen as same sign plus a 0.25--4 magnitude
  ratio; when both deficits are near zero, a 0.05 absolute tolerance is used.

## Validation log

- T1--T12 and downstream diagnostics: 8 targeted tests passed.
- Full repository regression suite: 221 tests passed.
- Compilation passed with bytecode directed outside the repository.
- The ordinary two-seed/20-sequence smoke run completed and correctly honored
  Stop 1/A1 at the deliberately abbreviated training dose while still writing
  every required table, report, and PNG/PDF figure.
- A separate in-memory engineering-only gate bypass exercised the otherwise
  unreachable downstream smoke branch. It completed S0/S4 at H=8/16/32 and
  produced nonempty Phase-B training, corridor, factorization, bootstrap, model,
  and phase-diagram artifacts. This bypass is not exposed by the CLI and its
  `/tmp` results are not scientific evidence.
- A full-dose control-only check (five paired seeds, 170 sequences) passed the
  finalized qualitative S0 equivalence rule. The fresh original G2 probe replay
  matched saved values to `1.78e-15`, all A/B gates passed, High/Low hidden
  separation ratios were 0.927/0.547, and the High Mixed/Novel deficit-magnitude
  ratios were 2.526/3.536.
- All seven figures were rendered; both populated and gate-stopped variants were
  visually inspected. The full 20-seed frontier itself was not run during
  implementation.
