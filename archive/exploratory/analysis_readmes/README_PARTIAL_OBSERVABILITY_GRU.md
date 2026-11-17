# Partial-observability GRU analysis

This standalone package implements `PRD_partial_observability_GRU.md` without
changing the original SRN, PRD2, or PRD3 pipelines. It asks whether a generic
GRU spontaneously learns posterior-weighted expression when it must predict
one continuous outcome while the relevant context-specific skill is latent.

## Scientific contract

- Each episode has independently evolving A and B skills and a newly sampled
  context-prototype pair with a random center, direction, and fixed minimum
  separation.
- The network sees the current noisy context observation, the previous
  practiced action, previous observed performance, feedback availability, and
  episode boundary. It never sees current context identity, current A/B skill
  states, the exact posterior, or current performance before predicting.
- The previous action is the PRD-permitted lagged practice indicator. It
  anchors otherwise exchangeable episode-specific A/B labels but reveals
  nothing about a possible context switch at the current timestep.
- The exact forward Bayesian posterior uses the known generative process and
  previous feedback only. It is an evaluation benchmark, never a target for
  the performance-prediction GRUs.
- Exact Midpoint, Both, Novel, Default, and human schedules are absent from
  training and are evaluated zero-shot.
- Human RTs are optional external validation only and never enter training.

## Implemented factorial axes

```text
uncertainty: perfect, low, moderate, high
skill separation: small, medium, large
objective: deterministic, Gaussian, classifier, stochastic, distributional
hidden dimension: 1, 2, 3, 4, 8, 16
```

The primary deterministic and Gaussian models have ordinary GRUs plus direct
readouts; no experts, mixture weights, posterior variables, or hard-coded
averaging are present. The stochastic and distributional two-component heads
are objective controls required by the PRD.

## Running

Start with the reduced end-to-end smoke profile:

```bash
cd "/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN"

MPLCONFIGDIR=/private/tmp/matplotlib-cache \
XDG_CACHE_HOME=/private/tmp/xdg-cache \
python -u scripts/run_partial_observability_gru.py \
  --profile smoke \
  --output outputs/partial_observability_gru_smoke
```

The pilot contains the complete 4×3×5×6 design with five seeds:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
XDG_CACHE_HOME=/private/tmp/xdg-cache \
caffeinate -i python -u scripts/run_partial_observability_gru.py \
  --profile pilot \
  --output outputs/partial_observability_gru_pilot
```

The final profile uses all cells and 20 seeds. It is intentionally explicit
because it is a large computational experiment:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
XDG_CACHE_HOME=/private/tmp/xdg-cache \
caffeinate -i python -u scripts/run_partial_observability_gru.py \
  --profile full \
  --output outputs/partial_observability_gru_full
```

Every factorial axis and training-size setting can be narrowed from the CLI.
For example, `--objectives deterministic classifier --hidden-dims 3 8
--seed-count 2 --epochs 20` runs a targeted diagnostic.

To overlay human means after the separate retrieval analysis exists, add:

```text
--human-reference outputs/retrieval_analysis_full/retrieval_clean_data.csv
```

## Outputs

All thirteen required CSVs and `run_config.json` are written, plus a training
history audit and eight figures. `run_config.json` records the leakage,
zero-shot, architecture, and human-data safeguards used by the run.
