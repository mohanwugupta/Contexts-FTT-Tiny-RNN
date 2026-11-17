# PRD v3: human configured-state behavioral test

This package tests whether Mixed and third-context behavior is better described
by one global A/B responsibility weight or by a sequence-state-specific
configuration. It reuses the validated retrieval preprocessing contract and
loads the already-trained v2.1 High-Scale SRNs without changing their weights.

## Run it

From the repository root, run the fast software/data smoke first:

```bash
python scripts/run_human_configured_state_v3.py \
  --combined data/mixedBreak_clean.csv \
  --novel data/threeContextsBreak_clean.csv \
  --srn-source outputs/ftt_srn_execution_v2_final_checkpointed \
  --output outputs/human_configured_state_v3_smoke \
  --profile smoke
```

Then run the full registered analysis:

```bash
python scripts/run_human_configured_state_v3.py \
  --combined data/mixedBreak_clean.csv \
  --novel data/threeContextsBreak_clean.csv \
  --srn-source outputs/ftt_srn_execution_v2_final_checkpointed \
  --output outputs/human_configured_state_v3 \
  --profile final
```

Progress is printed at data loading, pure-profile estimation, frozen-SRN
extraction, model fitting, recovery, and figure rendering. Add `--no-plots` for
a table-only run or `--quiet` to suppress progress.

## Data and indexing contract

The Combined file uses A, B, and Mixed/Both blocks. The third-context export is
recognized from its actual labels, including `Space Training` and
`test spaceA`, and is reported as Novel. Mixed and Novel are modeled separately
except in the explicitly labeled shared-profile comparison.

The existing retrieval exclusions are preserved: incomplete participants and
blocks, invalid sequences, the first sequence in every block, and incorrect
attempts are excluded. The human token analysis uses seven SRN-aligned
transitions, `1→2` through `7→8`. Maze keypress indices `0,2,...,14` and the
ordinal Melody/Mixed/Novel indices `0,1,...,7` are normalized to that common
coordinate without losing their source labels or counterbalance metadata.

Odd analyzed sequence repetitions calibrate participant-level parameters;
even repetitions are held out. Group token structure is learned only from
training participants in participant-level cross-validation.

## Model ladder and safeguards

The primary held-out ladder contains M0 scalar context, M1 scalar mixture, M2
shrunken token-specific mixture, M3 training-only low-rank configuration, and
M4 frozen-SRN configuration. M4 also exports its zero-free-token strict form
and full/shared/specific representational variants.

Pure A/B profiles are estimated with partial pooling before probe fitting.
Midpoint deviation is the primary stable outcome. Unclipped responsibility
ratios are secondary and are flagged rather than interpreted when A/B endpoint
separation is below the preregistered threshold.

Every registered `human_matched` SRN seed is used. Checkpoint hashes are tested
before and after extraction; optimizers, backward passes, human-driven seed
selection, and human-driven token-profile fitting are all forbidden and
audited. Recovery uses the observed participant count, endpoint scale, noise,
and participant-by-position observation pattern.

## Outputs

The output directory contains the 22 required PRD tables,
`human_run_config.json`, `human_behavioral_report.md`, two additional
preprocessing/frozen-weight audit tables, and 11 figures in
`figures_human_configured_v3/`. `--profile final` with no participant limit is
marked `final_inference_ready`; a smoke run is always labeled preliminary.

