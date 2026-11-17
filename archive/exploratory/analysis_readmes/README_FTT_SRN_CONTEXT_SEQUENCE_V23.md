# PRD v2.3: context × sequence interaction

This package analyzes the already-trained v2.1 High-Scale SRNs. It does not
train a network, create an optimizer, call `backward()`, or change a checkpoint.
It requires a v2.1 output directory containing `model_checkpoint_manifest.csv`.

## Run it

From the repository root, first run the two-seed software smoke:

```bash
python scripts/run_ftt_srn_context_sequence_v23.py \
  --source outputs/ftt_srn_execution_v2_final_checkpointed \
  --output outputs/ftt_srn_context_sequence_v23_smoke \
  --profile smoke
```

Then run the registered 20-seed analysis:

```bash
python scripts/run_ftt_srn_context_sequence_v23.py \
  --source outputs/ftt_srn_execution_v2_final_checkpointed \
  --output outputs/ftt_srn_context_sequence_v23_final \
  --profile final
```

Progress is printed once per seed/checkpoint and again before figure rendering.
Use `--no-plots` for a table-only run or `--quiet` to suppress progress.

## What is tested

For each seed, checkpoint, and token position, the analysis constructs

```text
mu_k = (h_A(k) + h_B(k)) / 2
d_k  = (h_A(k) - h_B(k)) / 2
```

and verifies exact reconstruction of both natural endpoints. Uncentered SVD of
the token-specific `d_k` vectors defines a shared PC1 component; the orthogonal
remainder is the token-specific component. Mixed and Novel states are patched
with the full, shared-only, or specific-only natural component. Every other
token is used in the exhaustive wrong-token control after norm matching.

Full/shared/specific ablations, a norm-matched random orthogonal control, and a
norm-matched sequence-position control report execution speed, normalized RT,
next-token cross-entropy, correct-token probability, and token identity. Every
synthetic state also receives nearest-natural-state, Mahalanobis, and hidden-norm
diagnostics.

The sequence-only, context-only, additive, and interaction regressions are
fit only to natural A/B outcomes. Mixed and Novel are never fitting targets.
Evaluation holds out a complete seed. Since independently initialized networks
do not share hidden-unit coordinates, `f_seq(mu_k)` is represented by aligned
token-position indicators rather than invalidly pooling raw hidden-unit axes
across networks. The context coordinate is the projection onto that network's
shared A/B PC1.

## Outputs and inference

The output includes all 23 PRD tables, a frozen-checkpoint audit, transparent
model coefficients, ten registered figures, `interaction_run_config.json`, and
`interaction_report.md`. `seed_summary.csv` retains token-level effects;
`interaction_mechanism_summary.csv` aggregates M1–M6 within network before
group decisions. Mixed and Novel always remain separate.

Only `--profile final` with exactly 20 registered seeds is final-inference
ready. The report requires the registered 80% threshold where applicable and
also reports the required A/B-over-random specificity control.

