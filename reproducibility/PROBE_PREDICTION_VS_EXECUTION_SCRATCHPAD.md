# Probe prediction versus execution scratchpad

## Scope and frozen choices

- Implement `MINI_PRD_probe_prediction_vs_execution.md` as a read-only extension
  of the completed time-matched diagnostic.
- Preserve High/Low monotonic models, seeds 0–19, checkpoints 20–170 by tens,
  zero-state A/B/Mixed/Novel branches, seven transitions, one encounter, no
  settling, and no evaluation updates.
- Report checkpoints 20, 100, and 170. Freeze each seed's counterfactual readout
  at checkpoint 20 and use checkpoint 170 as the sensitivity reference.
- Resample whole seeds with all checkpoints for 2,000 draws using seed 54032.

## Data recovery decision

- `same_snapshot_probe_source.csv` retained sequence RT but did not retain
  transition NLL or readout slope/intercept. Therefore the verified deterministic
  replay is reused; no new acquisition or probe protocol is introduced.
- `time_matched_probe_overlay.replay.evaluate_snapshot` now exports correct-token
  probability, logit-derived target NLL, raw slope, positive softplus slope, and
  intercept. This only adds logging to frozen diagnostic branches.

## Computation decisions

- Prediction loss is `mean(correct_token_nll)` across positions 1–7, not
  `-log(mean_probability)`.
- RT reconstruction uses the repository's float32 Torch operations exactly:
  probability clamp to `[1e-6, 1-1e-6]`, logit, positive softplus-transformed
  slope, intercept, sigmoid, then `1-speed`.
- Checkpoint-20 readout parameters are applied transition by transition to each
  later checkpoint's unchanged probabilities. Transformed RT is averaged only
  afterward.
- Prediction and RT gaps are paired within seed/checkpoint and use the worse
  (larger loss or larger RT) same-snapshot A/B endpoint.
- Bootstrap point estimates are the observed paired sample means; seeded
  whole-seed resampling supplies confidence limits. Raw bootstrap draws are not
  saved because the registered seed/algorithm plus gap source fully reproduces
  them and the draw table is approximately 145 MB. Bootstrap summaries and both
  figure-source tables are saved.

## Verification log

- Targeted unit tests cover the softplus/clamp identity, transition-before-
  sequence aggregation, NLL aggregation, worse-endpoint gaps, paired changes,
  whole-seed resampling, replay fields, and isolation.
- Full temporary integration replay matched all prior final state hashes.
- Maximum RT reconstruction error: 1.19e-7 (tolerance 5e-7).
- Maximum NLL identity error: 9.84e-7 (tolerance 5e-6).
- Maximum previous-sequence-RT disagreement: 1.11e-16 (tolerance 5e-7).
- Visual inspection completed for the six-panel primary figure and four-panel
  fixed-readout figure.

## Result recorded after the frozen analysis

- High-scale Mixed prediction-loss excess changes from -0.267 at checkpoint 20
  to 1.438 at 170; RT excess changes from -0.046 to 0.223.
- High-scale Novel prediction-loss excess changes from 0.285 to 1.931; RT excess
  changes from 0.004 to 0.242.
- At checkpoint 170, using checkpoint-20 readouts reduces mean High-scale RT gap
  from 0.223 to 0.198 for Mixed and from 0.242 to 0.204 for Novel.
- Whole-seed intervals support both a widening upstream prediction limitation
  and smaller additional amplification by changing readout parameters.
