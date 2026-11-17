# Human Context-Control Transfer Scratchpad

## Locked decisions

- Implemented as a standalone package so no file in the synthetic confirmatory freeze manifest is modified.
- Consumes only hash-verified, completed A/B structured/shuffled checkpoints for independent seeds 20–39.
- A-S/A-Sh and B-S/B-Sh are the two primary matched contrasts. B-S-RP and the validated H8 bank are prespecified secondary controls. Random controls are excluded because they were not frozen before human-result inspection.
- The old generic 20/50/90/170 transfer loop is not used for the primary analysis. The actual model schedule has 40 familiar A/B updates before Mixed1/Mixed2 and 120 before final tests.
- Human blocks 5 and 6 are frozen counterfactual probes. They make zero optimizer updates; therefore Mixed1 and Mixed2 share checkpoint 40 but retain separate exposure labels. Block 17 uses checkpoint 120.
- All nonfamiliar model probing is hard-gated by familiar acquisition, hidden/context differentiation, RT differentiation, state-space phenotype, and the confidence–RT relationship.
- A/B canonicalization is one whole-profile swap per seed/snapshot/probe, never a transition-wise relabeling.
- Human outcomes are read only after synthetic verification and familiar gating. They are used only for unfitted descriptive profile correspondence.
- Model B restricted plasticity freezes the shared Elman core while leaving the controller, context projection, prediction head, and execution policy trainable.
- Prior retention uses disposable model copies and the frozen synthetic validation generator.

## Timeline derivation

- Human familiar acquisition: blocks 1–4 = A10/B10/A10/B10 (40 updates).
- Human early probes: blocks 5–6 = Mixed1/Mixed2, with parallel Novel and NoContext cohorts (checkpoint 40, no updates).
- Return to familiar learning: blocks 7–14 = four further A10/B10 cycles (80 updates).
- Familiar tests: blocks 15–16 (checkpoint 120, no updates).
- Final MixedTest/NovelTest/NoContextTest: block 17 (checkpoint 120, no updates).

## Safety/audit design

- Preflight defaults to a closed human firewall and verifies upstream freeze hashes, confirmatory gates, architecture selection, checkpoint completeness, and all 80 checkpoint hashes.
- The execution head is newly initialized for every target-task run; synthetic checkpoints contain no RT head.
- Targets are passed only to loss/statistics code and never to either architecture's `forward` method.
- Every probe/intervention restores a disposable copy and checks the parent state hash.
- Full runs require `--open-human-firewall`; `--skip-retention` exists only for diagnostic iteration and produces an explicitly incomplete retention artifact.

## Progress

- [x] Locked configuration and group registry
- [x] Exact timeline mapper
- [x] Checkpoint/source/config freeze checks
- [x] Familiar-only transfer engine and H8 baseline
- [x] Familiar F1–F6 gates and Lump/Cross/Split fits
- [x] Frozen Mixed/Novel/NoContext probes and whole-profile canonicalization
- [x] Behavioral, hidden, controller, human-correspondence metrics
- [x] Model A/B causal interventions
- [x] Synthetic prior-retention battery
- [x] Paired contrasts/bootstrap and 12 required figures
- [x] Focused test suite: 11/11 passing
- [x] Read-only integration preflight: all 80 confirmatory checkpoint contracts and SHA-256 hashes verified
- [x] Three-model integration smoke: GRU, restricted Model B, and H8; gates/state-space, first Mixed, geometry, interventions, and all 12 figures
- [x] Disposable prior-retention smoke at checkpoints 0/20/40/120; parameter hashes unchanged

## Finalization edge case

- The completed full run produced no gate-passing primary group, so all nonfamiliar metric/intervention tables were correctly empty.
- The first reporting attempt nevertheless called `pd.concat([])` for the intervention figure and failed after retention had finished.
- The plotting guard now treats this as a valid empty result and writes an explicit blank panel.
- `--finalize-only` reconstructs the figures and Markdown report from saved artifacts and does not repeat training, probes, or retention.
