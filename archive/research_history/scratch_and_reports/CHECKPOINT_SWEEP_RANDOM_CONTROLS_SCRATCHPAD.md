# Checkpoint Sweep + Random Controls Scratchpad

## Locked implementation decisions

- This is a standalone extension. It imports the already validated architecture, target-training, probe, H8, geometry, and state-space implementations without modifying their frozen files.
- Primary checkpoints are exactly 0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, and 120 familiar updates.
- Random controls use `build_model(architecture, seed)` from the frozen model source and load no synthetic state or optimizer. Reconstructed hashes must equal `initial_state_sha256` in both matched structured and shuffled confirmatory checkpoints.
- The paired sign-flip null uses the seven token-matched A-minus-B representation difference vectors. The observed statistic is the norm of their mean; the exact 2^7 sign-flip distribution is the null. This tests coherent context separation while preserving transition-specific vector magnitudes.
- Familiar selection is a distinct software phase. It has no Mixed/Novel inputs or imports. A SHA-256 sidecar locks `selected_familiar_checkpoints.json` before the probe phase can begin.
- Group checkpoint selection uses the mean across 20 seeds; per-seed optima are exploratory only.
- H8 is a historical familiar anchor, not a matched synthetic-pretraining condition.
- Mixed at all checkpoints is run only after primary selection/probing is locked and is labeled exploratory in every row.

## Progress

- [x] Locked config and group/checkpoint registry
- [x] Initialization reconstruction audit: all 40 architecture × seed initial states match both stored structured/shuffled hashes
- [x] Familiar sweep and transfer indices, with no future-settling leakage
- [x] State-space checkpoint fits, explicit Cross/Split-vs-Lump evidence, H8 and human familiar references
- [x] Familiar-only group-level selection and SHA-256 lock
- [x] Primary Mixed/Novel and post-freeze exploratory all-checkpoint probes
- [x] Paired contrasts, bootstrap intervals, mixed-effects models, 12 figures, and report
- [x] Focused suite: 10/10; combined new/reused regressions: 34/34
- [x] Reduced seven-group full integration: 12 figures, report, primary freeze, 1,764 exploratory raw rows

## Verification notes

- A reduced one-seed integration initially exposed an empty architecture-panel plotting error; fixed before the production command was handed off.
- Checkpoint-15 transfer originally had access to a future B-settling denominator. This was caught during audit: the numerator is now retained at checkpoint 15, while the denominator/index remains missing until checkpoint 20.
- Human preprocessing is cached within the probe process so primary and exploratory stages do not reread both raw datasets.
