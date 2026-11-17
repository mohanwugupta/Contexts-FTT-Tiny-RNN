# Causal A/B representation analysis (PRD v2.2)

This isolated pipeline loads trained High-Scale v2.1 checkpoints and performs frozen hidden-state
interventions. It contains no optimizer, never calls backward, and verifies model hashes before and
after every seed/checkpoint analysis.

The human-matched checkpoint is primary. Early, several-switch, and final checkpoints are secondary.
Directions are token matched and estimated in a forward pass separate from intervention evaluation.
The smallest SVD subspace explaining at least 90% of A/B difference energy is used; PC1 is never
forced when more dimensions are required.

## One-time v2.1 checkpoint preparation

The original v2.1 output predates checkpoint serialization. Its CSVs cannot reconstruct neural
weights. Run the unchanged registered v2.1 configuration once into a checkpoint-bearing source:

```bash
python scripts/run_ftt_srn_execution_v2.py \
  --profile final \
  --output outputs/ftt_srn_execution_v2_final_checkpointed
```

The updated v2.1 pipeline writes `model_checkpoint_manifest.csv` and checksum-protected `.pt` files
in addition to its original artifact contract. Training, objectives, seeds, and analyses are
unchanged.

## Causal runs

One-seed software smoke test using all four checkpoints:

```bash
python scripts/run_ftt_srn_causal_v22.py \
  --source outputs/ftt_srn_execution_v2_final_checkpointed \
  --profile smoke \
  --output outputs/ftt_srn_causal_v22_smoke
```

Registered 20-seed causal analysis:

```bash
python scripts/run_ftt_srn_causal_v22.py \
  --source outputs/ftt_srn_execution_v2_final_checkpointed \
  --profile final \
  --output outputs/ftt_srn_causal_v22_final
```

Both commands print timestamped model/checkpoint, export, and plotting progress. Result files are
written directly to the requested directory; only Matplotlib's cache uses temporary storage.

## Artifact and decision contract

The run writes 15 required CSVs, `causal_run_config.json`, `causal_report.md`, a frozen-weight audit,
and ten figures. The report separates:

- sufficiency from directional steering;
- partial necessity from A/B-component ablation;
- the Mixed representational pathway from matched-state component patching;
- descriptive geometry from causal evidence;
- Mixed from the separately evaluated Novel mechanism.

Random orthogonal and sequence-position directions are mandatory negative controls. Every nonzero
intervention carries Mahalanobis, nearest-natural-state, and hidden-norm diagnostics.
The full ±2 steering sweep is exported; inferential steering summaries use the near-manifold
`|alpha|≤1` range, leaving larger perturbations as sensitivity analyses.
