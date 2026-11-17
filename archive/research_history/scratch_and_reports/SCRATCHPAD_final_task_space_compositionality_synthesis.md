# Scratchpad: final task-space compositionality synthesis

## Frozen decisions

- This package imports no optimizer or training entry point and performs zero parameter updates.
- Treat the current report's `alpha=.1, anchor=.1` object as a global minimum stable **seed-cell**
  (seed 15), not an aggregate cell.
- Treat the Phase 1C plot-role `alpha=10, anchor=30` as a historical descriptive plot
  representative because the old plotting code accessed Mixed loss as a secondary sort key.
- Define a new `best_stable_aggregate_cell` from P1 aggregate endpoint/familiar stability and
  minimum normalized logit-path residual, with alpha ascending, anchor descending, and cell ID
  as deterministic tie-breakers. Mixed behavior is absent from the selection frame.
- Define one `best_stable_seed_cell` per seed with the same firewall. Do not relax eligibility.
- Formal task error is the saved centered-logit residual divided by seven transitions; normalized
  error is unchanged by this scaling.
- Derive normalized null-path residual from each seed's own frozen High baseline.
- Classify heterogeneity descriptively as continuous unless the largest adjacent gap in ordered
  selected residuals exceeds 35% of their full range with at least three seeds on both sides.
- Bootstrap by resampling seeds, preserving all within-seed cells for specificity summaries.

## Progress

- [x] Read PRD and verify all complete 20-seed Phase 1C artifacts exist.
- [x] Implement label audit and selection firewall.
- [x] Implement seed/formal/statistical synthesis tables.
- [x] Implement four final figures and nine sequence-level figures.
- [x] Generate report, paper-ready text, plain-language explanation, and captions.
- [x] Add formal, firewall, projector, missing-artifact, read-only, matching, and full-pipeline tests.
- [x] Run the final 2,000-bootstrap synthesis into `final_mechanistic_synthesis/`.

## Smoke-test findings

- The strict aggregate rule independently selected `P1_correct_logit_alpha_10_anchor_30`.
- All 20 seeds had at least one endpoint/familiar-stable P1 cell; four of 20 passed the full
  endpoint + path + rescue + remaining-gap gate.
- Exact wrong-path integrity passed for all 880 seed × alpha × anchor pairs, including source
  checkpoint hash, 170-update duration, and final-checkpoint rule.
- Heterogeneity was classified as a continuous frontier (largest adjacent gap = 19.5% of range),
  not a clear multi-seed subgroup.
- The 100-bootstrap end-to-end smoke and five dedicated tests passed; no source hash changed.
- The production synthesis completed with 2,000 seed-bootstrap draws. All invariant audits passed,
  all four primary and nine mandatory time-series figures were visually inspected, and the final
  narrative sections met their requested approximate word-count ranges.
