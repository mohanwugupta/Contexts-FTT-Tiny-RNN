# PRD v3.1: low-rank human-dimension interpretation

This package interprets the low-rank human configuration space found by the
frozen v3 behavioral analysis. It estimates every basis from pure A/B profiles,
then projects Mixed, Novel, and temporal probe profiles without allowing those
outcomes to rotate or refit the basis.

## Run it

From the repository root, run a quick data/software check with:

```bash
python scripts/run_human_low_rank_interpretation_v31.py \
  --v3-output outputs/human_configured_state_v3 \
  --combined data/mixedBreak_clean.csv \
  --novel data/threeContextsBreak_clean.csv \
  --output outputs/human_low_rank_interpretation_v31_smoke \
  --profile smoke
```

Run the full registered analysis with:

```bash
python scripts/run_human_low_rank_interpretation_v31.py \
  --v3-output outputs/human_configured_state_v3 \
  --combined data/mixedBreak_clean.csv \
  --novel data/threeContextsBreak_clean.csv \
  --output outputs/human_low_rank_interpretation_v31 \
  --profile final
```

Progress is printed for loading, dimensionality selection, bootstrap alignment,
feature freezing, probe projection, replication/SRN comparison, recovery, and
figure rendering. Use `--no-plots` for a table-only run or `--quiet` to suppress
these updates.

## Analysis contract

- The seven response transitions are the same `1→2` through `7→8` units used
  by v3 and the frozen SRN comparison.
- Candidate PCA ranks are 1, 2, 3, and the full rank 7 ceiling. Rank is selected
  using participant-held-out reconstruction and probe likelihood with the
  one-standard-error rule. Factor analysis and parallel analysis are reported
  as robustness checks.
- Components are aligned across folds and participant bootstraps with Hungarian
  matching and sign correction. Stability requires median cosine at least `.70`.
- The formal seven-row transition-feature table is frozen and SHA-256 hashed
  before semantic tests. Inference enumerates every estimable transition-label
  permutation, up to all `7! = 5,040` arrangements.
- Component naming requires stability, permutation support, adequate semantic
  recovery, and independent cross-dataset replication. Components that miss a
  gate retain neutral names.
- Combined and Novel bases are fit independently before cross-dataset alignment.
  Shared, separate, and scalar probe accounts are compared with participant-held-
  out likelihood. Raw participant identities are held out across both datasets.
- Human/SRN comparisons use frozen v3.0 SRN profiles and abstraction-level
  dimensionality, variance partitioning, principal angles, and canonical
  correlations. Human outcomes never select seeds or modify SRN weights.

## Outputs and current registered result

The final directory contains all 23 required CSV tables, the frozen run config,
the concise interpretation report, additional prediction/audit tables, and 11
figures in `figures_human_low_rank_v31/`.

The current full run selected rank 2. Both components passed the registered
stability threshold. Component 2 passed the semantic gates for late-learning
dynamics; Component 1 remains neutrally named. The pure-A/B subspace replicated
across Combined and Novel, but the generic-difficulty residual test and the
human/SRN abstraction-match gate did not pass. Novel behavior received a
reliable held-out likelihood improvement from its separate pure-A/B basis, so
the registered probe comparison is Result F. See
`outputs/human_low_rank_interpretation_v31/low_rank_interpretation_report.md`
for the compact result statement.

