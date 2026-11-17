# Context failure and selective repair — append-only scratchpad

Authoritative protocol: PRD_SRN_CONTEXT_FAILURE_SELECTIVE_REPAIR.md.
Expected companion: CONTEXT_FAILURE_SELECTIVE_REPAIR_SCRATCHPAD_TEMPLATE.md.

## Initialization and missing-template record

The named template was not present in the repository, Downloads or Codex attachments. The PRD was read in full. A clarification was requested; meanwhile this scratchpad is initialized from the PRD's required sections. It is not represented as a copy of an unavailable file. Any later supplied template will be incorporated through appended sections, preserving history.

## Scientific thread and scope

Identify a selective natural disruption of recurrent prediction under physical Mixed. Capacity, a full-state rescue, or a helpful oracle is insufficient. No model training, optimizer updates, hidden/cue optimization or changes to predecessor evidence. R4 is design only.

## Source/reuse audit (R0)

Verify actual checkpoint/source/cue identities, indexing, pulse semantics, complete state/logit/probability fidelity and unchanged imported future_summary criterion. Locate P4/Phase1K/P2/D1/S3 sources. Prefer exact P4 records for all360 cells; only source-replay fidelity checks allowed before R1. No duplicated matched branches are newly evaluated for scientific inference.

## Fixed protocol decisions, before new outcomes

- Exactly20 original High_monotonic canonical checkpoints at original170; k3 primary, k5 secondary. No optional Low or secondary novel families in v1, avoiding expansion beyond the required primary program.
- R1 allnine inherited A/B/Mixed × current A/B/Mixed cells; every future cue Mixed. Never use persistent rows or select best branch. Current CE separate from subsequent4/2 predictions.
- References: both A-history/Mixed-update and B-history/Mixed-update, no performance choice. Natural full-A/B references are separate diagnostics only.
- Output geometry: existing full-right SVD and P2 invariance/float32 arithmetic audit; declared1e-10 double tolerances, interior margin1e-6, box tolerance1e-10. Source float32 outcomes; no hidden clipping.
- Future geometry: fixed two-dimensional U within output kernel, complement W; familiar-only future centered-logit Jacobians, no target labels or Mixed CE. Absolute rank1e-8, relative1e-4*sigma1. Top-two orientation identifiable only if sigma2>max(1e-8,1e-4*sigma1) and sigma2−sigma3 exceeds that same bound. Export raw spectrum. No rank changes following outcomes.
- Derivatives: double frozen-weight autodiff plus analytic chain check; actual float32 source FD scales1e-3 and1e-4, one-sided within closed tanh bounds at boundaries. Numerical discrepancies exported, not used to choose directions.
- Random/fallback vectors: SHA256-based RNG identity [base seed20261113, seed,k,reference,control kind], frozen before intervention outcomes. Numerical zero norm threshold1e-12.
- Matched U/W/random controls: common amplitude .95 times the largest jointly feasible scale<=1 in the fixed interior box. If the safety factor falls below a required lower feasible bound, mark infeasible. No rescaling, clipping, sign choice or replacement.
- Visible/full-silent arms use that same common scale when individually feasible. Natural/reference observed source states can lie on float32 tanh boundary; retain their exact observed R1 outcomes and interior status, but never clip or pretend a boundary reference is a feasible strict-interior patch.
- Reverse: same fixed forward amplitude and negative U/W/random directions from each originally viable reference; require matched feasibility there, no extra amplitude search. Familiar supporting controls use both signs from each natural A/B poststate, with own familiar future cue.
- All per-reference rows retained. Pair contrasts within reference then average estimable A/B references within seed; record0/1/2 valid-reference counts. Conditional estimable-seed intervals may be shown, but full-cohort selective claim requires20-seed coverage, both favorable specificity intervals and zero softmax violations. Missing/infeasible effects never become zero.
- Bootstrap10000 whole-seed draws, seed20260925. R3 primary U−W and U−random at Mixed/k3. Reverse support is a separate gate and cannot be inferred from rescue alone.

## Stage gates

R0 source/reuse → R1 exact factorial → R2 label-free geometry → R3 selective/reverse → R4 design only → R5 report/seal. Every scientific stage requires RED, GREEN, refactor regression and immutable manifest.

## RED / GREEN / REFACTOR evidence (append)

## Decisions and deviations (append)

## Run ledger and hashes (append)

## Findings, negative cases and coverage (append)

## Claim ledger and researcher teach-back (append)

## Completion and verification (append)

## R0 RED and source-derived fixture corrections
Nine behavioral failures captured in audit/r0_red.log (permissive stub accepted wrong schedules, cues, identities and probabilities). Actual cue audit established physical Mixed=(.5,.5,0); initial synthetic fixture incorrectly assumed (1,1,0), corrected before implementation execution. One negative target fixture accidentally used the true final target3; corrected to0, preserving the first GREEN failure log. No protocol or scientific thresholds changed.

## r0 completed
{"checkpoints": 20, "reused_cells": 360, "reused_rows": 1440, "fidelity_only_rollouts": 76, "scientific_factorial_rollouts": 0, "model_updates": 0, "passed": true}
Manifest verifies original source/checkpoint bytes; no training or optimization.

## r1 completed
{"reused_cells": 360, "new_scientific_rollouts": 0, "primary_viable_reference_seeds": 5, "all20_present": true, "model_updates": 0}
Manifest verifies original source/checkpoint bytes; no training or optimization.

## r2 completed
{"output_rank": 3, "nullity": 5, "identifiable_geometries": 40, "geometries": 40, "gaps": 160, "model_updates": 0, "labels_used_for_geometry": false}
Manifest verifies original source/checkpoint bytes; no training or optimization.

## Explicit version correction: v1_1 supersedes R2 of v1
The R2 implementation passed its runtime algebra/chain/source guards, but its synthetic kernel test averaged float32 weights before asserting a float64 identity. Observed8.69e-9 was arithmetic in the test, not kernel leakage. The driver shell lacked failure short-circuiting and ran R2 despite that failing test. Preserve v1 unchanged including failure logs and original test snapshot; do not cite its R2 as TDD-complete. Correct ONLY the fixture to average float64 weights, retain1e-12 tolerance and unchanged scientific algorithms/thresholds. R0/R1 immutable artifacts and manifests copied byte-for-byte; no matched factorial reruns. Repeat affected R2 algebra in new directory v1_1 after GREEN, then continue. All future stage commands must short-circuit on failures.
