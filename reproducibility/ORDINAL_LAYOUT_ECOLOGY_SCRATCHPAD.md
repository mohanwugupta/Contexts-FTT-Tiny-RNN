# Ordinal sequence + spatial layout ecology scratchpad

## Frozen implementation decisions

1. This analysis lives in `ordinal_layout_ecology`; all confirmatory and
   structured-context predecessors are treated as read-only references.
2. Stage 0 loads `outputs/confirmatory_srn_final` without retraining.
3. Stage 1 supplies token identity separately from eight shared ordinal one-hot
   channels and the stable global cues A=`[1,0]`, B=`[0,1]`.
4. Stage 2 uses two 16-value layout banks from the start: A training activates
   `[Layout_A, 0]`, B activates `[0, Layout_B]`, and primary Mixed activates
   `[Layout_A, Layout_B]`. No post-hoc input expansion or adapter is permitted.
5. Layout coordinates are fixed geometric constructions, centered separately,
   and scaled to unit full-layout L2 norm. Consequently both layouts have the
   same coordinate RMS and neither is selected using model outcomes.
6. Primary Mixed uses global cue `[1,1]`. The registered control uses cue

   `[.5,.5]` and scales both layout banks so its complete input norm exactly
   matches a pure A/B input.
7. Novel uses a third fixed, normalized layout in both already-trained layout
   banks with global cue `[0,0]`. This preserves the input norm and avoids any
   arbitrary untrained input dimension.
8. Stage gates implement the PRD literally: sequence learning, paired High >
   Low separation, High Cross/Split, Low Lump, decreasing RT, and negative
   confidence--RT slope must each reach the 80% seed threshold.
9. Mixed is allowed only after the Stage 2 screen and 20-seed confirmation both
   pass. Improvement means a strictly higher usable between-endpoint fraction
   and a strictly lower hidden A/B-segment distance than categorical Stage 0.
10. If that primary Mixed criterion fails, the only follow-up is a fixed
    5-seed overlap sweep with 0, 1, 2, or 4 static shared-object channels. The
    expanded architecture is present from the beginning of every sweep fit.
11. No salience noise, global-cue noise, dropout, human data, or probe training
    is used anywhere in the primary pipeline.

## Progress

- [x] Parsed the PRD and froze encodings, gates, Mixed control, and follow-up.
- [x] T1--T10 are green (10/10 focused structural tests).
- [x] Smoke integration and stage firewall validate (11/11 including contract).
- [x] Full 5/20-seed ecology and six PNG/PDF figure pairs validate.
- [x] Repository regressions pass (149 tests) and 112 frozen final CSVs remain
  bitwise identical.

## Full-run results

- Stage 1 passed every gate: High was Cross/Split in 5/5 seeds and Low was
  Lump in 5/5.
- Stage 2 passed screening and confirmation. In the 20-seed confirmation,
  High was Cross in 13 and Split in 7; Low was Lump in 20/20. High-minus-Low
  hidden separation was 2.842603, close to the frozen categorical difference
  of 2.863402.
- Primary full-layout Mixed did not improve recombination. On usable High
  transitions, between-endpoint fraction changed from .418605 (categorical)
  to .208333, and mean A/B-segment distance changed from 1.144354 to 1.741794.
  The input-norm-matched half-cue control also had .208333 between endpoints
  and segment distance 1.599017.
- The negative primary result triggered the only registered follow-up. The
  two-shared-object condition passed the 5-seed gates (High Cross/Split 5/5,
  Low Lump 5/5) and provisionally improved both metrics to .438095 and
  1.019017. Counts 1 and 4 failed the High Cross/Split gate; count 0 passed but
  did not improve both metrics. Count 2 is recorded as a screen, not a 20-seed
  confirmation.
- Novel remained difficult for High-scale models (mean CE 3.095138, accuracy
  .35) while Low-scale models generalized the sequence (CE .039463, accuracy
  1.00).
- The fixed final interpretation is **B: the ordinal scaffold plus spatial
  realization preserves splitting but not recombination**. The sparse-overlap
  result is explicitly reported as a provisional follow-up.
- An independent full rerun reproduced 150/150 checkpoint files byte-for-byte;
  all unaffected scientific tables were also byte-identical.
