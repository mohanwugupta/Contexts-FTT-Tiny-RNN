# Mini PRD: Locate the Probe Deficit in Prediction or Execution

## 1. Goal

Determine whether the growing High-scale Mixed/Novel RT deficit is already present in next-token prediction, is amplified by the learned execution readout, or both.

The time-matched overlay established that the deficit survives sequence averaging and contemporaneous A/B comparisons. Extend that pipeline with prediction metrics. No new architecture, model fitting, human-data analysis, or probe training is required.

## 2. Reuse the verified evaluation

Start from:

```text
scripts/run_time_matched_sequence_probe_overlay.py
time_matched_probe_config.json
outputs/ordinal_layout_ecology_final
ordinal_layout_ecology/probes.py::evaluate_frozen_layouts
```

Resolve the previous diagnostic output directory from its config. Preserve:

- The same 20 seeds and High/Low monotonic models.
- Post-update checkpoints after sequences 20, 30, …, 170.
- A, B, Mixed, and Novel evaluated from identical model/readout snapshots.
- Zero recurrent state at branch entry, one sequence, seven scored transitions, no settling sequence, and no evaluation updates.

Reuse saved probabilities, logits, or checkpoints first. If missing, extend the existing verified replay to export them. Use its current reproducibility and state-preservation checks. Do not substitute online pre-update A/B loss for frozen post-update A/B loss.

These are independent diagnostic encounters at each checkpoint, not repeated practice with the probe contexts.

## 3. Record and aggregate

For every seed × checkpoint × context × transition, save:

```text
correct-token probability p
correct-token negative log likelihood
normalized RT
readout slope a and intercept b
```

Use the repository's actual positive slope, including its softplus transformation, and its existing numerical-stability conventions. Compute loss from logits when available.

For each sequence, calculate:

\[
L_c(t)=-\frac{1}{7}\sum_k\log p_{c,k}(t),
\qquad
P_c(t)=\frac{1}{7}\sum_k p_{c,k}(t),
\qquad
R_c(t)=\frac{1}{7}\sum_k RT_{c,k}(t).
\]

Loss is the mean of negative log probabilities, not the negative log of mean probability. Aggregate transitions within a sequence before summarizing seeds.

For Mixed and Novel separately, retain paired excess over the worse same-checkpoint endpoint:

\[
G_L(t)=L_{probe}(t)-\max[L_A(t),L_B(t)],
\]

\[
G_R(t)=R_{probe}(t)-\max[R_A(t),R_B(t)].
\]

Positive values indicate worse performance than both endpoints. Preserve per-seed values; summarize uncertainty by resampling whole seeds with their checkpoint trajectories. Do not compare the numerical sizes of loss and RT gaps directly.

## 4. Primary figure

Create one six-panel figure:

```text
Rows:    High scale | Low scale
Columns: prediction loss | correct-token confidence | normalized RT
```

Plot frozen A/B/Mixed/Novel means across the same checkpoints, with consistent colors and across-seed uncertainty. Retain common y-axis limits across High/Low within each metric where practical. Use raw units without condition centering.

Report paired loss and RT gaps at checkpoints 20, 100, and 170 and their change from 20 to 170. Do not infer a paired difference solely from overlapping or nonoverlapping marginal error bars.

## 5. Small readout check

Verify that the saved RT is reconstructed transition by transition by:

\[
RT=1-\sigma[a(t)\operatorname{logit}(p)+b(t)].
\]

To assess the contribution of changing readout parameters, also transform every checkpoint's probabilities using that seed's fixed checkpoint-20 readout. Use checkpoint 170 as a prespecified sensitivity reference. Average transformed transition values afterward; never transform sequence-mean confidence.

Plot the resulting probe-versus-A/B RT gaps against the gaps under the actual contemporaneous readout. Label these curves counterfactual readout diagnostics. They change neither training nor predictions and are not replacement model fits.

This comparison isolates the effect of substituting readout parameters on the observed probabilities. It does not establish what training with a fixed readout would have produced.

## 6. Interpretation

| Pattern | Conclusion |
| --- | --- |
| Probe prediction loss remains above A/B, with a widening gap | A transfer limitation is already present upstream of execution. |
| Loss gap is small or stable, but RT gap grows and is reduced under fixed readouts | Changing execution parameters contribute to the growing RT disparity. |
| Prediction deficit persists and readout substitution also changes its RT expression | Both prediction and execution contribute. |
| Mean loss and RT appear inconsistent | Inspect saved transition probabilities and readout reconstruction; nonlinear averaging can make sequence means diverge. |

Prediction failure alone does not identify a hidden-state mechanism. Summarize evidence without automatically recommending gating or architecture changes.

## 7. Outputs and acceptance

```text
probe_prediction_execution_config.json
probe_prediction_execution_transitions.csv
probe_prediction_execution_sequences.csv
probe_prediction_execution_gaps.csv
probe_prediction_vs_execution.png / .pdf
probe_fixed_readout_comparison.png / .pdf
PROBE_PREDICTION_VS_EXECUTION_REPORT.md
```

Reuse existing checks for evaluation isolation and replay fidelity. Verify loss aggregation, RT reconstruction, and agreement with the previous sequence-level RT output within documented numerical tolerance. Save figure source data and one reproducible command.

The report should be brief: show the primary figure, state whether the deficit appears in prediction, explain any readout contribution, and identify remaining uncertainty.
