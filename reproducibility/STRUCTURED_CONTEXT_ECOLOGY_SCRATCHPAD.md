# Structured context ecology scratchpad

## Frozen implementation decisions

1. This analysis is isolated in `structured_context_ecology`; the completed
   `confirmatory_srn` pipeline and all registered v2/v3 code remain read-only.
2. Stage 0 is loaded from `outputs/confirmatory_srn_final` and is never retrained
   or modified by this runner.
3. Stage 1 uses exactly three literal item channels: `G`, `A_specific`, and
   `B_specific`, with unit-norm A/B encodings.
4. Stage 2 has exactly seven trained contextual items: `S`, `A1--A3`, and
   `B1--B3`, using the frozen ordinal grouping `1-2 / 3-4 / 5-7`.
5. Three additional input columns are reserved for `N1--N3` so Stage 5 can be
   evaluated without changing model shape after training. These columns are
   zero initialized, have zero gradients, and are never active during training.
   The trained Stage 2 context bank therefore remains exactly seven-dimensional.
6. Stage 3 Mixed activates `S + Ak + Bk` and is normalized to unit norm. It is
   evaluated without optimizer steps.
7. Stage 4 uses mean-one lognormal gain with sigma in `{0,.05,.15,.30}` on
   active context features only. It does not renormalize after the draw, alter
   sequence tokens, activate a wrong identity, or introduce label flips.
8. A nonzero sigma is confirmatory only if its screen passes all learning/
   splitting/state-space/RT gates and improves both High-scale Mixed between-
   endpoint fraction and hidden segment distance relative to sigma zero. The
   smallest eligible sigma is selected; no human data are loaded.
9. New stages retain the registered sequence tuple, canonical seeds 0--4 for
   screens and 0--19 for confirmation, and the Stage 0 High-minus-Low separation
   difference as the fixed reference. Nontrivial splitting requires at least
   50% retention of that paired Stage 0 difference.
10. The stage firewall writes required empty downstream tables if a prior gate
    fails; it never runs a later stage to rescue an earlier failure.

## Progress

- [x] Parsed the PRD and froze encodings, gates, and selection rule.
- [x] Implemented the model, training, probes, analyses, stage firewall, report,
  figure code, CLI, and T1--T10 tests.
- [x] T1--T10 and smoke integration pass (11 focused tests).
- [x] Full stepwise ecology executed according to the gates (5-seed screens,
  then 20-seed confirmations only after passing screens).
- [x] Required outputs and six PNG/PDF figure pairs validate.
- [x] Repository regressions pass (136 tests) and 112 frozen CSVs remain
  bitwise identical.

## Full-run decisions and results

- Stage 1 and Stage 2 passed all five gates at screening and confirmation.
- All four Stage 4 screens passed the basic gates. The smallest nonzero level
  improving both prespecified Mixed outcomes was sigma = .30; that improvement
  replicated in the 20-seed confirmation.
- On usable High-scale Mixed transitions, between-endpoint fraction changed
  from .418605 (categorical) to .432432 (structured), while mean distance to
  the A/B segment changed from 1.144354 to 1.093355. At confirmed sigma = .30,
  the corresponding values were .501429 and .689654.
- Stage 5 was probe-only. Relative to the unused-axis categorical control, the
  structured High-scale Novel probe reduced CE from 2.939958 to 1.774593 and
  nearest-trained-manifold distance from 1.677407 to 1.113090; confirmed
  salience reduced the manifold distance further to .999587.
- The fixed interpretation rule therefore returns **A: shared structured
  context is sufficient**. No human observations were loaded or fitted.
- A clean independent rerun produced 19/19 byte-identical common scientific
  CSVs and 255/255 byte-identical checkpoints relative to the preceding run.
- The verified artifacts were installed at
  `outputs/structured_context_ecology_final`: 21 CSVs, 255 checkpoints, and
  six figures in both PNG and PDF. Probe audits record zero optimizer steps,
  zero parameter updates, unchanged parameters, and zero human rows.
