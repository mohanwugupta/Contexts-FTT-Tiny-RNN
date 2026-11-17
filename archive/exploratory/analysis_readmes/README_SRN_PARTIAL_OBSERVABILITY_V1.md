# SRN v1 partial-observability analysis

This is the isolated implementation of `PRD_v1_SRN_partial_observability.md`.
The primary model is a four-unit tanh Elman SRN trained only to minimize
deterministic next-outcome MSE. It does not read human data, the latent context,
latent state values, the exact posterior, or the current target.

Existing sequence-SRN, PRD2, PRD3, and GRU analysis code is unchanged.

## Commands

Fast software smoke test:

```bash
python scripts/run_srn_partial_observability_v1.py \
  --profile smoke \
  --output outputs/srn_partial_observability_v1_smoke
```

Five-seed MVP:

```bash
python scripts/run_srn_partial_observability_v1.py \
  --profile mvp \
  --output outputs/srn_partial_observability_v1_mvp
```

The MVP automatically runs hidden probes and the dimensionality ladder only if
the prespecified behavioral gate passes. The optional GRU control can be added
with `--run-gru-control`. Skipped stages still produce their required CSV and
figure with `status=not_run`.

After the MVP succeeds, the prespecified 20-seed run is:

```bash
python scripts/run_srn_partial_observability_v1.py \
  --profile final \
  --mvp-run-config outputs/srn_partial_observability_v1_mvp/run_config_v1.json \
  --output outputs/srn_partial_observability_v1_final
```

The final profile refuses to start unless that MVP record says the behavioral
stage gate passed. This prevents an accidental 20-seed run after a failed MVP.

All commands print timestamped network/epoch/stage progress. Use `--quiet` only
when terminal progress is not wanted.

## Interpretation

`proof_of_concept_report.md` explicitly assigns PRD Result A–E. Only Result A
supports moving to v2. `run_config_v1.json` records the full configuration,
scientific gate, staged status, forbidden inputs, and output row counts.
