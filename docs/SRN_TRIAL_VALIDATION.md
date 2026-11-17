# SRN trial-wise validation diagnostics

## Why this layer exists

`analysis_SRN.Rmd` visualizes legacy CogSci SRN loss over cumulative sequences
and, in some figures, normalizes it alongside human RT. `state-space_SRN.Rmd`
fits no-fatigue Lump, Cross, and Split dynamical models to an averaged SRN loss
trace, but its primary evidence tables emphasize RMSE/BIC. The diagnostic layer
implemented here exposes the corresponding current-model trajectories and puts
every state-space prediction directly over the observed trial means.

The translation is informed by the frozen plans and audit documents:

- `archive/exploratory/prds/PRD_v2_1_FTT_high_scale_SRN_execution_control.md`
  defines the current 8-unit SRN, A10/B10 curriculum, execution head, and
  `normalized_rt = 1 - execution_speed`.
- `docs/METHODS_DRAFT.md` defines the frozen objective and warns that model RT is
  unitless rather than milliseconds.
- `docs/PAPER_FIGURE_DATA_GUIDE.md` identifies `training_dynamics.csv` and
  `execution_dynamics.csv` as the trial-level sources for learning and execution
  checks.
- `docs/SRN_RIGOR_GAP_AND_DECISIONS.md` explains why the older CogSci workflow is
  a robustness/diagnostic source rather than a replacement model specification.

Accordingly, this code reads frozen outputs and never changes training,
checkpoints, model selection, or paper claims.

## Command

```bash
python scripts/run_srn_validation_diagnostics.py
```

Optional paths:

```bash
python scripts/run_srn_validation_diagnostics.py \
  --training outputs/ftt_srn_execution_v2_final_checkpointed/training_dynamics.csv \
  --output outputs/srn_trial_validation
```

## Direct trajectory figures

- `figure_01_sequence_ce_over_trials.png` shows next-token cross-entropy for all
  three frozen conditions.
- `figure_02_total_loss_over_trials.png` shows the full optimized objective. In
  the sequence-only condition this equals sequence CE.
- `figure_03_normalized_rt_over_trials.png` shows the execution-trained High and
  Low conditions. Sequence-only is omitted because its speed head was never
  optimized.

Each panel displays all 20 seed traces faintly, cohort means as colored A/B
block segments, and 95% normal-approximation confidence intervals across seeds.
The script validates all 10,200 rows, the fixed 170-sequence A10/B10 schedule,
seed/trial uniqueness, missingness, and the RT identity before plotting.

## State-space translation

The Python implementation mirrors the no-fatigue definitions in
`state-space_SRN.Rmd`:

- Lump: one decaying state, `x[t+1] = A*x[t]`;
- Cross: separately gated A/B states with estimated cross-talk;
- Split: separately gated A/B states with cross-talk fixed at zero;
- all models: OLS intercept/state decoder nested inside bounded L-BFGS-B
  optimization;
- BIC parameter counts: 3, 5, and 4 respectively.

The fit unit is one cohort-average observation per training sequence, matching
the R notebook's averaged-trace analysis. The current experiment has one fixed
A-first schedule, so there is no counterbalance split to reproduce.

- `figure_04_state_space_loss_overlay.png` overlays all three models on observed
  sequence CE.
- `figure_05_state_space_rt_overlay.png` repeats the diagnostic for model RT.
- `figure_06_observed_vs_predicted.png` shows identity-line checks for every
  trial mean.
- `state_space_predictions.csv` and `state_space_fit_metrics.csv` contain all
  plotted values and fitted parameters for later R use.

## What the check shows

All three conditions learn the sequence. Mean sequence CE falls from 1.599 to
0.062 for High, 1.386 to 0.046 for Low, and 1.599 to 0.063 for High sequence-only.
The individual seed traces show that these cohort curves are not produced by a
small number of runs.

Model normalized RT is not a rescaled copy of loss. High mean RT rises from
0.458, peaks at 0.672 near sequence 87, and ends at 0.618. Low mean RT rises from
0.500, peaks at 0.636 near sequence 89, and ends at 0.588. This non-monotonicity
is visible at the seed level and should be kept explicit when translating the
diagnostic to R.

The simple Lump model has the lowest descriptive BIC in every fit. It explains
99.7% of mean loss variance for both High conditions and 96.5% for Low. For
normalized RT it explains 89.3% for High and 81.6% for Low. The overlays explain
why: Cross/Split impose reinstatement jumps when a held context state becomes
active again, whereas the observed current-model means are dominated by a
smooth global trajectory. This result is a diagnostic of the old state-space
model family on the new SRN—not evidence against the separately established
hidden-state split.

All BICs are in-sample and operate on cohort means. They are useful for checking
whether the fitted curves actually resemble the data, but they do not replace
seed-level uncertainty, held-out evaluation, or the frozen causal analyses.

