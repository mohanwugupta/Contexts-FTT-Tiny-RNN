# Posthoc transition x time-since-switch scratchpad

## Frozen decisions

1. This is a read-only analysis of the 20-seed `stage2_layout` confirmatory
   High- and Low-monotonic histories. It never constructs an optimizer or
   updates a checkpoint.
2. The exact historical transition stream must contain 47,600 rows:
   2 conditions x 20 seeds x 17 blocks x 10 sequences x 7 transitions.
3. Block 1 is excluded from switch/early/settled summaries because it has no
   preceding A/B context. Blocks 2--17 provide eight actual A switches and
   eight actual B switches for every seed.
4. Switch is within-block sequence 1, early is 1--2, and settled is 8--10.
   Recovery trajectories retain all positions 1--10 from actual switch blocks.
5. Confidence intervals are 1.96 x the SEM across seed-level summaries.
   Inferential summaries use paired seed differences and two-sided t tests;
   exact estimates and confidence intervals remain primary because this is
   explicitly posthoc.
6. Frozen A/B endpoints are read from the already saved Stage 3 probe table.
   Probes are not rerun.
7. The original final checkpoints contained only final parameters, not historic
   transition outputs. Zero-step checkpoint evaluation could not reconstruct
   past within-block trajectories, so the analyzer was implemented to fail
   explicitly rather than impute or silently retrain.
8. The ordinal-layout pipeline will persist the already-computed transition
   stream on future runs. This changes logging only, not model computation.
9. On 2026-09-05 the user explicitly authorized rerunning the frozen pipeline.
   It was rerun into `outputs/ordinal_layout_ecology_posthoc_source`; all 21
   shared CSVs were byte-identical to the original run, and the 40 relevant
   Stage 2 High/Low checkpoint tensor states were identical. The new transition
   table was then added to the canonical final directory without replacing any
   pre-existing file.

## Progress

- [x] Audited the frozen source and identified the missing historical stream.
- [x] Exact source validation and summaries implemented.
- [x] Figures/report/CLI implemented.
- [x] Synthetic integration and safe-failure tests pass (7/7).
- [x] Repository regressions pass (156 tests) and all 112 registered final CSVs
  remain bitwise identical.

## Implementation notes

- `posthoc_transition_switch/data.py` checks the exact row, seed, schedule,
  transition-position, uniqueness, switch-flag, and finite-RT contract before
  any summary is computed.
- All primary summaries are seed-level. Context is retained in transition
  profiles; Figure 2 and inferential switch costs average A/B within seed.
- The report calls the switch-dynamics explanation reliable only when the High
  `D_switch - D_settled` estimate and its 95% CI are both above zero.
- The output audit hashes all available sources before and after analysis and
  explicitly records no human data, checkpoint loading, optimizer, probe rerun,
  or model selection.
- The plotting integration was exercised on a complete synthetic transition
  history and produced all three required PNG/PDF pairs plus optional Figure 4.
- The canonical real-data transition source is now present. It has 47,600
  validated confirmatory High/Low rows: 20 seeds x 2 conditions x 17 blocks x
  10 sequences x 7 transitions. The analysis completed from the default paths.

## Real-data result

- High switch cost: 0.046318 (95% CI 0.040017--0.052620).
- Low switch cost: 0.033180 (95% CI 0.033054--0.033306).
- Paired High-minus-Low switch cost: 0.013138, t(19)=4.057,
  two-sided p=0.000672.
- High `D_switch - D_settled`: -0.012463 (95% CI
  -0.044451--0.019525), t(19)=-0.764, p=0.454.
- Therefore High has the larger switch cost, but its A/B transition profile is
  not more separated at switch than when settled. Time-since-switch dynamics do
  not explain the training-versus-frozen discrepancy under the registered test.

## Verification record

- Targeted posthoc + ordinal tests: 20 passed.
- Complete repository suite: 156 passed in 34.95 seconds.
- Frozen-output verifier: 112/112 final CSVs bitwise identical.
- Missing-history behavior remains tested: it exits with status 2 and writes
  only the explanatory report and a source audit recording zero updates.
- Completed default analysis audit: source hashes unchanged, zero checkpoint
  loads, zero optimizer steps, zero parameter updates, and no probe rerun.

## Frozen High A/B seed-profile follow-up (2026-09-05)

- Added `scripts/run_frozen_high_AB_seed_profiles.py`, a frozen-table-only
  follow-up using the 280 saved Stage 2 confirmatory High A/B probe rows.
- Computed `delta[s,k] = RT(B,s,k) - RT(A,s,k)`, seed magnitudes, the
  across-seed mean profile, and all 190 unique pairwise seed correlations.
- Mean/median individual magnitude: 0.878825 / 0.865562. The magnitude of the
  grand-mean profile was 0.224682, only 25.6% of the mean individual magnitude.
- Mean/median pairwise profile correlation: -0.039 / -0.059; 56.3% of pairs
  were negative. Majority sign agreement by position ranged from 50% to 70%.
- Decision: strong A/B profiles are seed-idiosyncratic and substantially cancel
  in the grand mean; neither their sign nor their seven-position pattern is
  consistent across seeds.
- Final repository verification after this follow-up: 158 tests passed and all
  112 registered final CSVs remained bitwise identical.

## Whole-profile Fast/Slow canonicalization follow-up (2026-09-05)

- Assigned one Fast/Slow A/B swap per seed using only mean frozen A/B RT across
  all seven transitions. Fast=A for 14 seeds and Fast=B for 6 seeds.
- Mixed was excluded from label assignment and read only afterward for the
  requested descriptive profile. Novel and human data were unused.
- Canonicalization increased the grand-mean/mean-individual magnitude ratio
  from 0.256 to 0.447 and mean/median pairwise correlation from
  -0.039/-0.059 to 0.064/0.005.
- Nevertheless, 50% of pairwise correlations remained negative and per-position
  majority-sign agreement was only 50%--70%. The global A/B swap explains some
  cancellation, but substantial transition-pattern idiosyncrasy remains.
- Audit confirmed the source hash was unchanged, with zero checkpoint loads,
  retraining, probe reruns, optimizer steps, parameter updates, Mixed labeling
  rows, Novel rows, or human rows.
- Final verification: 160 tests passed; 112/112 registered final CSVs remained
  bitwise identical.
