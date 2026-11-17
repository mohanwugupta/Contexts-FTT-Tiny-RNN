# Human Meta-Prior Transfer Scratchpad

## Locked scientific decisions

- The upstream evidence was read from `outputs/srn_meta_learning_calibration` and is immutable: H128, I3 orthogonal initialization, inner α=.01, 10 steps, meta-batch 16. The H128 confirmatory and revised structural gates must pass before this runner starts.
- The older `episodic_meta_learning_capacity_v2` H128 checkpoints are scientifically incompatible because they used the earlier high-scale/three-step setup and failed the structural gate. They are rejected.
- The calibration run did not persist revised model objects. This pipeline deterministically materializes checkpoint files from the locked setup and records file and model-state SHA-256 hashes. Seeds 0–4 reproduce validated jobs; seeds 5–19 extend the frozen setup to 20 genuinely distinct paired source models. Five checkpoints are not recycled as 20 pseudoreplicates.
- G1–G4 receive identical G2-shared-x A10/B10 training, sequence, optimizer, learning rate, clipping, and execution objective. G4 uses the registered H128 I3 random initialization; G5 preserves H8 scale-2 initialization.
- Every run gets a new empty Adam optimizer, zero-initialized monotonic execution readout, and zero recurrent state at each sequence.
- A/B are the only training contexts. A/B/Mixed/Novel/Neutral probes are immutable disposable evaluations, and every primary row is tagged `probe_exposure_index=1`.
- Canonicalization assigns one whole seven-transition profile to Fast and the other to Slow. Transitions are never individually relabeled.
- Final A/B hidden separation is compared with a same-width, same-seed random-initialization baseline. Mixed/Novel primary inference requires both run- and group-level familiar gates.
- Empirical Mixed data are loaded only after target training, probe generation, and familiar gating, solely for unfitted z-profile correlation/RMSE/R².
- H256/H512 remain optional later robustness tests and cannot replace H128.

## Implementation map

- `configs/human_meta_transfer_config.json`: locked upstream, target, gate, seed, bootstrap, and smoke contracts.
- `human_meta_prior_transfer/checkpoints.py`: upstream verification, deterministic checkpoint materialization, and identity validation.
- `human_meta_prior_transfer/training.py`: exact checkpoint loading, fresh optimizer/readout, A/B training, frozen probes, and subsystem resets.
- `human_meta_prior_transfer/analysis.py`: familiar gates, canonicalization, Mixed/Novel measures, hidden geometry, empirical correspondence, mixed model, and bootstrap contrasts.
- `human_meta_prior_transfer/retention.py`: disposable synthetic-prior probes at target checkpoints.
- `human_meta_prior_transfer/plotting.py`: Figures 1–10.
- `human_meta_prior_transfer/pipeline.py`: orchestration, firewalls, stopping rules, conditional ablations, outputs, and report.
- `scripts/run_human_meta_prior_transfer.py`: CLI runner.
- `tests/test_human_meta_prior_transfer.py`: T1–T10 and output integration.

## Verification log

- 2026-09-09: Three-seed smoke completed all groups, immutable probes, state-space fits, gated metrics, empirical-profile comparison, and prior retention.
- 2026-09-09: Smoke generated all 16 required outputs, supplementary audits/tables, and Figures 1–10.
- 2026-09-09: Dedicated T1–T10 plus conditional-ablation suite passed: `11 passed in 32.74s`.
- 2026-09-09: Final clean three-seed smoke completed successfully in `/tmp/human-meta-prior-transfer-smoke-final`; all required CSVs were readable, probability vectors summed to one within numerical tolerance, and all checkpoint/initial-state hashes were recorded.
- 2026-09-09: Full repository regression passed after final edits: `282 passed in 93.85s`.

## Interpretation warning

Smoke checkpoints use two synthetic families and 20 target sequences. Their gates and effects are engineering diagnostics only, not scientific results.
