# Paper freeze implementation log

2026-09-25 — Decisions recorded before new Novel behavioral evaluation.

- Follow final_implement_PRD.md; no new training, optimizer steps, cue tuning, or Phase 1M.
- Preserve every historical file by SHA256 snapshot before restructuring. Historical packages remain compatible; obsolete documentation/runners will be archived with wrappers where needed.
- Freeze analytical double-precision cue vectors from the specified radicals; model cue inputs are float32.
- Common evaluator: explicit float32 NumPy recurrence for canonical and contrast models, float64 stable log-sum-exp CE and geometry. Controls retain float64 displacement until next recurrent update. Regression tolerance fixed prospectively: state 2e-4, logits 2e-3, CE 2e-3 against historical Torch; NumPy natural path 1e-10 CE. Report observed discrepancies.
- A2 primary rule: for each seed, match High update 170 to Low registered checkpoints 0,10,...,170; same incoming context (A at 170), each familiar A/B CE within 0.05 nat; choose minimum Euclidean A/B CE discrepancy then earliest checkpoint. No accuracy constraint because common saved checkpoint accuracy is unavailable. Report all candidate/support counts. No threshold relaxation or training reconstruction. If a selected intermediate parameter state is unavailable, only that comparison is blocked. Also export descriptive all-anchor support for registered High checkpoints without probe inspection.
- Fixed cohorts: canonical High/Low monotonic 0–19; constructive C0–C3 sequence-only 0–19. Missing artifacts block only dependent analyses; strict audit exits nonzero on missing inputs.
- Composite interventions: repeated correct/wrong/orthogonal/A/B endpoints plus all seven acute positions. Maintenance: all seven positions, D0/D1/D2. Generation: native/A/B/Mixed/zero identical-prestate branches, all seven pulse/persistent Mixed/A/B takeovers.
- Aggregate positions equally within seed; lag-specific summaries average available positions within seed. Remaining rescue includes the intervention transition; post-injection excludes lag zero. Export both separately.
- Use one fixed 10000×20 bootstrap matrix, seed 20260925, for all complete-cohort contrasts. Missing pairs fail rather than silently dropping seeds.
- All four main figures and predetermined supplements are generated regardless of effects. A7 primary absolute distance; normalized sensitivity separately.
- Human results are indexed as historical context; no human refitting or new collection is included.

## TODO

- [x] Preserve full historical snapshot
- [x] RED novelty / evaluator / intervention / statistics / firewall tests
- [x] GREEN package and geometrically freeze Novel registry
- [x] Historical regression and immutable cohort audit
- [x] Freeze paper contract and A2 matches before Novel evaluation
- [x] Evaluation-only A1–A8, with A2 reported as insufficient common support
- [x] Figures, captions, report, manifests, visual QA
- [x] Shared package extraction, archive crosswalk, README
- [x] Isolated editable installation and reproduction verification

## Pre-evaluation implementation record

- Full historical snapshot completed: 11,159 non-cache files, 12,769,065,510 bytes; no historical evidence moved or changed.
- RED: novelty/core suite failed import before implementation. GREEN: six combined geometry/intervention/statistics tests passed.
- Novel registry frozen and committed as bb6a63e before new-cue model evaluation. Registry digest: 2b28095562ea69a33d96c521817dcd0743f3d66658385ac425212f6951e00727.
- RED: evaluator and integration suites failed missing-module imports. GREEN: 17 focused tests now pass, including all historical-model regressions, C0 identities, frozen arrays, optimizer trap, branch prestates, cue continuation, nested averaging and clean imports.
- Familiar-only matching (no new Novel evaluation): 1/20 primary seed pairs supported, seed 9 High170→Low110. Low110 parameter checkpoint is unavailable. Preserve this as insufficient_common_support and explicitly unavailable matched-state behavior; no optimizer reconstruction, caliper relaxation, or training extension.
- Shared low-level evaluator implemented in the new package without moving any legacy packages. This prevents eager legacy imports; actual historical archival follows verified outputs.

## Completion and correction record

- Initial full evaluation-only freeze `v2_20260925_novelty30_60` completed for 40 canonical and 80 constructive frozen checkpoints. All four main figures, supplement, captions, source data, report, and manifest were generated. A2 was reported as insufficient common support without training reconstruction.
- Visual QA found only overlapping schematic labels in Figures 2 and 4. Their text placement was corrected; no calculations, cue vectors, scientific contrasts, or checkpoint selections changed. A fresh immutable `v2_20260925_novelty30_60_r2` freeze was generated with renewed source and contract hashes, leaving the first output intact.
- All revised figures were visually reviewed. `archive/artifact_index.json` records the moved 69 historical root documents, original README copy, and four copied legacy runners. Original legacy runner paths remain compatibility entry points.
- The paper-facing README and methods/contract/experiment/provenance docs now describe the four experiments and the A2 limitation. `scripts/verify_paper_freeze.py` verifies input and output hashes, 120 checkpoints, panel files, statuses, and figure bundle.
- An isolated virtual environment with system-provided pinned dependencies and locally installed wheel accepted an editable package installation. The focused paper and historical compatibility suite passed 24 tests. Normal dependency download could not be tested because this sandbox has no package-index access.
