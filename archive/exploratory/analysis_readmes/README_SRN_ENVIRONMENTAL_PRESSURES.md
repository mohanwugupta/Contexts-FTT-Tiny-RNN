# SRN environmental-pressure analysis (PRD v1.2)

This package asks which learning environments make posterior averaging useful
to a fixed eight-unit Elman SRN. It changes context validity, graded cue
accuracy, and A/B state separation while keeping the architecture, optimizer,
learning rate, exact 10A/10B schedule, training exposure, and update cadence
fixed within each scientific run.

The implementation is isolated in `srn_environmental_pressures/`. It neither
imports human Mixed/Novel outcomes nor modifies any existing SRN, PRD2, PRD3,
or GRU model-fitting code.

## Software smoke test

```bash
python scripts/run_srn_environmental_pressures.py \
  --profile smoke \
  --output outputs/srn_environmental_pressures_smoke
```

This checks three representative environments with one seed and writes all
required artifacts. Gated confirmation, ablation, and representation files are
still written with `status=not_run`.

To exercise every staged software path with deliberately non-inferential tiny
settings:

```bash
python scripts/run_srn_environmental_pressures.py \
  --profile smoke \
  --force-staged-analyses \
  --output outputs/srn_environmental_pressures_forced_smoke
```

## Full discovery and confirmation

```bash
python scripts/run_srn_environmental_pressures.py \
  --profile discovery \
  --output outputs/srn_environmental_pressures_discovery
```

The discovery profile runs the prespecified 6×5×4 environmental grid with five
seeds per condition: 600 primary SRNs. Conditions passing in at least four of
five seeds are frozen and evaluated on 20 fresh seeds. Only confirmed regimes
advance to S1/S2/S3 follow-ups, outcome-noise checks, pressure ablations,
hidden probes, and causal interventions.

The command prints timestamps for every network and completed A/B pair. It can
therefore be left running without appearing idle. The full run is intentionally
not launched by the smoke test.

## Outputs and interpretation

The pipeline writes the 19 required CSVs, `run_config.json`, nine figures, and
`environmental_pressures_report.md`. The report assigns the PRD result pattern,
lists confirmed regimes, and evaluates ablation-based operational necessity.
Smoke results are explicitly labeled non-inferential.

