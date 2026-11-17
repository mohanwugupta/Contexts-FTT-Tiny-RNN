# PRD: Phase 1B — Direct Task-Space Path Sufficiency
## Preserve the Existing A/B Split While Fixing Only the Path Between the Endpoints

## 0. Rationale

Phase 1 reduced output curvature without improving Mixed performance. This showed that:

```text
low curvature != correct interpolation path
```

The earlier causal patching result was more specific:

```text
Replace only the output-relevant component of the Mixed hidden state
with the linear A/B midpoint component
→ large rescue.

Replace only the output-null component
→ approximately zero rescue.
```

Therefore this experiment directly trains the task-relevant path between the already-learned A/B endpoints.

Primary question:

> If the familiar A/B solutions are held essentially fixed, does forcing the task-relevant intermediate representation onto the endpoint-linear path rescue held-out Mixed performance?

This is a positive-control / sufficiency experiment.

Do not proceed to the structured-experience ecology until this test passes.

---

## 1. Start from existing High-Scale checkpoints

For every existing High-Scale + High-Context seed:

1. Load the trained checkpoint.
2. Cache baseline familiar hidden states:
   \[
   h^0_{k,A},\; h^0_{k,B}
   \]
3. Cache baseline familiar logits:
   \[
   \ell^0_{k,A},\; \ell^0_{k,B}
   \]
4. Preserve the original checkpoint unchanged.

Primary experiment fine-tunes from these trained endpoints rather than relearning A/B from scratch.

This isolates:

```text
endpoint geometry
vs.
between-endpoint geometry
```

---

## 2. Core notation

Let:

```text
k = sequence position
lambda = cue interpolation coordinate
h_k(lambda) = hidden state under interpolated context cue
ell_k(lambda) = output logits
W_out^0 = baseline frozen output matrix
P_out^0 = projector onto row space(W_out^0)
P_null^0 = I - P_out^0
```

Cue interpolation:

\[
c(\lambda)=(1-\lambda)c_A+\lambda c_B.
\]

Baseline endpoint-linear hidden target:

\[
h_k^*(\lambda)
=
(1-\lambda)h^0_{k,A}
+
\lambda h^0_{k,B}.
\]

Baseline endpoint-linear logit target:

\[
\ell_k^*(\lambda)
=
(1-\lambda)\ell^0_{k,A}
+
\lambda\ell^0_{k,B}.
\]

---

## 3. Centered logits

Because adding a common scalar to all logits does not affect softmax, use centered logits:

\[
\tilde\ell
=
\ell-\operatorname{mean}_{classes}(\ell).
\]

Target:

\[
\tilde\ell_k^*(\lambda)
=
(1-\lambda)\tilde\ell^0_{k,A}
+
\lambda\tilde\ell^0_{k,B}.
\]

The primary path loss uses centered logits.

---

## 4. Freeze policy

Primary experiment:

```text
Freeze W_out and output bias.
```

This keeps the readout and output-relevant subspace fixed.

Train only existing recurrent/input parameters:

```text
input-to-hidden weights
recurrent weights
hidden bias
context-input weights if represented separately
```

Do not change architecture or hidden size.

A later sensitivity analysis may unfreeze the output head, but it must not replace the frozen-head primary test.

---

## 5. Endpoint anchoring

The model must not improve Mixed by moving the familiar endpoints.

### Hidden endpoint anchor

\[
\mathcal L_{hidden-anchor}
=
\sum_k
\left[
\|h_{k,A}-h^0_{k,A}\|^2
+
\|h_{k,B}-h^0_{k,B}\|^2
\right].
\]

### Logit endpoint anchor

\[
\mathcal L_{logit-anchor}
=
\sum_k
\left[
\|\tilde\ell_{k,A}-\tilde\ell^0_{k,A}\|^2
+
\|\tilde\ell_{k,B}-\tilde\ell^0_{k,B}\|^2
\right].
\]

Also retain familiar A/B task loss:

\[
\mathcal L_{task}=CE(A)+CE(B).
\]

---

## 6. Experimental conditions

### D0 — Frozen High baseline

No fine-tuning.

### D1 — Direct centered-logit path

Primary positive-control condition:

\[
\mathcal L_{path-logit}
=
\sum_{k,\lambda\in\Lambda_{train}}
\|
\tilde\ell_k(\lambda)
-
\tilde\ell_k^*(\lambda)
\|^2.
\]

### D2 — Direct output-subspace hidden path

\[
\mathcal L_{path-out}
=
\sum_{k,\lambda\in\Lambda_{train}}
\|
P_{out}^0
[
h_k(\lambda)-h_k^*(\lambda)
]
\|^2.
\]

### D3 — Direct whole-hidden path

Strong control:

\[
\mathcal L_{path-whole}
=
\sum_{k,\lambda\in\Lambda_{train}}
\|
h_k(\lambda)-h_k^*(\lambda)
\|^2.
\]

### D4 — Curvature-only proxy control

Replicate the best prior curvature regularizer under the same endpoint-anchored fine-tuning setup.

### D5 — Endpoint-anchor-only control

Fine-tune with familiar task + endpoint anchors but no intermediate path objective.

### D6 — Frozen Low-Scale baseline

Known transferable-but-unsplit comparison.

---

## 7. Hold Mixed out of training

Primary training cue set:

```text
Lambda_train =
{0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90}
```

Do not train on:

```text
lambda = 0.50
```

because current Mixed corresponds to the midpoint.

Primary evaluation:

```text
lambda = 0.50
```

After training, also evaluate the dense sweep:

```text
0.00, 0.05, ..., 1.00
```

Optional upper-bound experiment only if D1/D2 fail:

```text
include lambda=.50 during training
```

Label this:

```text
MIDPOINT-SUPERVISED UPPER BOUND
```

Do not treat it as the primary result.

---

## 8. Total losses

### D1

\[
\mathcal L
=
\mathcal L_{task}
+
\alpha\mathcal L_{path-logit}
+
\beta\mathcal L_{hidden-anchor}
+
\gamma\mathcal L_{logit-anchor}.
\]

### D2

\[
\mathcal L
=
\mathcal L_{task}
+
\alpha\mathcal L_{path-out}
+
\beta\mathcal L_{hidden-anchor}
+
\gamma\mathcal L_{logit-anchor}.
\]

### D3

Replace the path term with \(\mathcal L_{path-whole}\).

### D4

Replace the path term with the previous curvature penalty.

### D5

Use only:

\[
\mathcal L_{task}
+
\beta\mathcal L_{hidden-anchor}
+
\gamma\mathcal L_{logit-anchor}.
\]

---

## 9. Hyperparameter firewall

Hyperparameters may be selected using only:

```text
familiar A/B task loss
endpoint anchor error
path loss on Lambda_train
```

Forbidden for selection:

```text
lambda=.50 / Mixed loss
human Mixed behavior
human Novel behavior
probe-to-human similarity
```

Recommended initial grid:

```text
alpha ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1}
beta  ∈ {0.1, 1, 10}
gamma ∈ {0.1, 1, 10}
```

Choose the smallest `alpha` that strongly lowers training-lambda path error while preserving the endpoints.

---

## 10. Endpoint-preservation metrics

Report:

### Familiar loss ratio

\[
R_{familiar}
=
L_{familiar}^{fine}/L_{familiar}^{base}.
\]

### A/B separation ratio

\[
R_{sep}
=
d_{AB}^{fine}/d_{AB}^{base}.
\]

### Endpoint hidden drift

\[
D_h
=
RMSE(
[h_A,h_B],
[h_A^0,h_B^0]
).
\]

### Endpoint centered-logit drift

\[
D_\ell
=
RMSE(
[\tilde\ell_A,\tilde\ell_B],
[\tilde\ell_A^0,\tilde\ell_B^0]
).
\]

Do not interpret the result as clean sufficiency if endpoint drift is substantial.

---

## 11. Direct path metrics

For every lambda compute:

### Whole-state residual

\[
E_{whole}(\lambda)
=
\sum_k
\|
h_k(\lambda)-h_k^*(\lambda)
\|^2.
\]

### Output-subspace residual

\[
E_{out}(\lambda)
=
\sum_k
\|
P_{out}^0
[
h_k(\lambda)-h_k^*(\lambda)
]
\|^2.
\]

### Null-space residual

\[
E_{null}(\lambda)
=
\sum_k
\|
P_{null}^0
[
h_k(\lambda)-h_k^*(\lambda)
]
\|^2.
\]

### Centered-logit residual

\[
E_{logit}(\lambda)
=
\sum_k
\|
\tilde\ell_k(\lambda)-\tilde\ell_k^*(\lambda)
\|^2.
\]

These are now primary metrics.

Curvature remains secondary.

---

## 12. Rescue fraction

Let:

```text
L0 = baseline High Mixed loss
Lpatch0 = baseline output-only linear-path patched Mixed loss
LD = fine-tuned condition Mixed loss
```

Available rescue:

\[
\Delta_{available}=L_0-L_{patch0}.
\]

Achieved rescue:

\[
\Delta_{achieved}=L_0-L_D.
\]

Define:

\[
\boxed{
R_{rescue}
=
\frac{L_0-L_D}
{L_0-L_{patch0}}
}
\]

Interpretation:

```text
0 = no rescue
1 = reaches the prior patching ceiling
>1 = exceeds the patching counterfactual
```

Report continuously.

---

## 13. Remaining patch gap

After fine-tuning, repeat the output-only linear-path patch.

Define:

\[
G_{remaining}
=
L_D-L_{D,patch}.
\]

Successful learning should make this gap small.

---

## 14. Phase 1B gate

D1 or D2 passes if all are true:

```text
A. familiar loss <= 110% of baseline
B. A/B separation remains within [90%, 110%] of baseline
C. endpoint hidden/logit drift stays below prespecified tolerance
D. held-out midpoint output/logit path residual <= 35% of baseline
E. achieved rescue fraction >= 0.50
F. remaining output-only patch gap <= 50% of baseline patch gap
G. Mixed q remains within 0.15 of 0.5
```

Show all continuous metrics regardless of pass/fail.

Do not change thresholds after viewing Mixed.

---

# 15. Mandatory sequence-level time-series standard

**Every run and every report must include sequence-level time-series plots.**

Aggregate bars/scatters alone are insufficient.

The plotting unit must be one complete sequence, with transition values aggregated across the seven trained next-token transitions.

At minimum compute per sequence:

```text
mean cross entropy
sequence accuracy
mean correct-token probability
mean execution / RT proxy if available
```

Never substitute transition/item-level plots for these required sequence-level plots.

---

## 16. TS1 — Fine-tuning trajectory

Plot over actual fine-tuning sequence/update order:

```text
x = training sequence/update index
y = sequence-level familiar loss
```

Show:

```text
D1
D2
D3
D4
D5
```

with across-seed summary and uncertainty band.

Also plot over fine-tuning:

```text
A/B hidden separation
endpoint hidden-anchor error
endpoint logit-anchor error
training-lambda path residual
```

This should show what training actually changes over time.

---

## 17. TS2 — Full familiar A/B sequence trajectory

After fine-tuning, replay the **entire familiar A/B experimental schedule in original order**.

For every complete sequence plot:

```text
sequence-level loss
sequence accuracy
correct-token probability
execution / RT proxy if available
```

Use:

```text
x = sequence number in experiment
```

Mark:

```text
A blocks
B blocks
block boundaries
```

This is mandatory even if familiar means look normal.

---

## 18. TS3 — Full frozen experiment trajectory

Replay the entire registered experiment schedule sequence-by-sequence, including:

```text
familiar A
familiar B
Mixed probes
NoContextProbe / Novel only if explicitly flagged as OOD under current encoding
```

Plot:

```text
sequence-level loss
correct-token probability
execution / RT proxy if available
```

Mark every probe sequence.

This is descriptive only and cannot drive hyperparameter selection.

---

## 19. TS4 — Held-out midpoint during fine-tuning

At fixed preregistered checkpoints:

```text
0%, 10%, 20%, ..., 100%
```

evaluate `lambda=.50`.

These evaluations must not affect:

```text
early stopping
hyperparameters
checkpoint selection
```

Plot together over fine-tuning:

```text
held-out Mixed sequence loss
output/logit path residual
remaining patch gap
```

This asks whether path alignment and behavioral rescue emerge together.

---

## 20. TS5 — Dense cue sweep at sequence level

For:

```text
lambda = 0.00, 0.05, ..., 1.00
```

plot ordered sequence-level lines for:

```text
sequence loss
correct-token probability
output path residual
logit path residual
```

Do not replace these with midpoint-only bars.

---

## 21. Time-series plotting conventions

For every time series:

```text
primary line = across-seed mean or median
band = bootstrap 95% CI or seed quantiles
individual seed traces optional
```

Do not smooth away actual sequence observations.

Always label:

```text
sequence number/time
context
probe locations
condition
```

Do not place incompatible metrics on the same y-axis.

---

## 22. Required time-series figures

Generate:

```text
P1B_TS01_finetuning_sequence_loss.png
P1B_TS02_finetuning_path_and_anchor_metrics.png

P1B_TS03_full_familiar_sequence_loss.png
P1B_TS04_full_familiar_sequence_accuracy.png
P1B_TS05_full_familiar_execution_proxy.png

P1B_TS06_full_experiment_sequence_loss.png
P1B_TS07_full_experiment_correct_token_probability.png
P1B_TS08_full_experiment_execution_proxy.png

P1B_TS09_midpoint_over_finetuning.png

P1B_TS10_cue_sweep_sequence_loss.png
P1B_TS11_cue_sweep_correct_token_probability.png
P1B_TS12_cue_sweep_path_residuals.png
```

If the execution/RT proxy is unavailable, create a short text artifact explaining why instead of silently omitting the corresponding figure.

---

## 23. Required summary figures

Also generate:

```text
P1B_01_endpoint_preservation.png
P1B_02_Mixed_loss_by_condition.png
P1B_03_output_path_residual_by_condition.png
P1B_04_logit_path_residual_by_condition.png
P1B_05_rescue_fraction_by_condition.png
P1B_06_remaining_patch_gap.png
P1B_07_AB_separation_vs_Mixed_loss.png
P1B_08_path_residual_vs_Mixed_loss.png
P1B_09_direct_path_vs_curvature_control.png
```

---

## 24. Required CSVs

```text
phase1b_training_summary.csv
phase1b_seed_metrics.csv
phase1b_endpoint_preservation.csv
phase1b_path_metrics.csv
phase1b_cue_sweep.csv
phase1b_patching.csv
phase1b_rescue_fraction.csv

phase1b_timeseries_finetuning.csv
phase1b_timeseries_familiar_full_schedule.csv
phase1b_timeseries_full_experiment.csv
phase1b_timeseries_midpoint_checkpoints.csv
```

Time-series CSVs must contain one row per:

```text
seed × condition × sequence/timepoint
```

before aggregation.

---

## 25. TDD / RED → GREEN

### T1 — checkpoint provenance
Never overwrite original High checkpoints; verify hashes.

### T2 — frozen output head
For primary D1–D5:

```text
W_out_final == W_out_baseline
b_out_final == b_out_baseline
```

### T3 — endpoint cache reproduction
Cached baseline hidden states/logits reproduce the original forward pass.

### T4 — centered-logit invariance
Adding a common scalar to all logits does not change `L_path-logit`.

### T5 — projector correctness

```text
P_out^2 = P_out
P_null^2 = P_null
P_out @ P_null = 0
P_out + P_null = I
```

### T6 — held-out midpoint firewall

```text
lambda=.50 not in Lambda_train
```

### T7 — exact endpoint targets

```text
lambda=0 -> cached A
lambda=1 -> cached B
```

### T8 — baseline anchor loss
Unmodified baseline gives zero hidden/logit anchor loss within tolerance.

### T9 — output-patch equivalence
With frozen output head, output-only linear-path patch reproduces full-patch logits.

### T10 — null-patch invariance
Null-only patch leaves logits unchanged within tolerance.

### T11 — no Mixed selection
Midpoint evaluation cannot affect training decisions.

### T12 — sequence-level aggregation
Every time-series row equals aggregation over exactly the seven trained next-token transitions of one complete sequence.

### T13 — sequence-time-series enforcement
Fail report generation if any mandatory sequence-level time-series plot is missing.

### T14 — registered schedule integrity
Full familiar/full experiment replay preserves exact registered sequence/context order.

---

## 26. Required report

Generate:

```text
PHASE1B_DIRECT_TASK_SPACE_PATH_SUFFICIENCY.md
```

Required sections:

```text
1. Executive result
2. Baseline reproduction
3. Endpoint preservation
4. Direct centered-logit path
5. Direct output-subspace path
6. Whole-hidden path control
7. Curvature-only control
8. Held-out Mixed result
9. Rescue fraction
10. Remaining patch gap
11. Full familiar sequence-level time series
12. Full experiment sequence-level time series
13. Fine-tuning sequence-level time series
14. Cue-sweep sequence-level time series
15. Mechanistic comparison
16. Phase 1B gate
17. Implication for the emergence/ecology experiment
```

Sections 11–14 are mandatory.

---

## 27. Explicit questions the report must answer

1. Can the task-space path change while familiar A/B endpoints remain effectively fixed?
2. Does direct centered-logit path training rescue held-out Mixed?
3. Does output-subspace hidden-path training rescue held-out Mixed?
4. Does whole-hidden path training add anything beyond output-only alignment?
5. Does curvature-only training remain inferior to direct path training?
6. What fraction of the prior patching rescue is learned naturally?
7. How much patching rescue remains after fine-tuning?
8. What do the **full sequence-level time series** reveal that the condition means hide?

---

## 28. Decision logic

### PASS

If D1 and/or D2:

```text
preserve endpoints
strongly lower held-out midpoint path residual
substantially rescue Mixed
leave little remaining patch gap
preserve familiar sequence-level trajectory
```

then conclude:

> Endpoint-linear task-space alignment is sufficient.

Only then proceed to the structured-experience/emergence experiment.

### FAIL 1

Path residual becomes small but Mixed does not improve.

Conclusion:

```text
patching interpretation is incomplete
```

Stop before ecology.

### FAIL 2

Mixed improves only because familiar endpoints drift or A/B splitting weakens.

Conclusion:

```text
not a clean sufficiency result
```

Stop before ecology.

### FAIL 3

Whole-hidden path succeeds but output-only path fails.

Conclusion:

```text
full representational geometry contributes beyond readout-relevant alignment
```

Revise the emergence target.

---

## 29. Permanent project standard

For all future modeling in this project:

> **Every model report must include full sequence-level time-series plots for familiar training/evaluation and for the complete probe schedule whenever those schedules exist.**

Do not rely on condition means, bars, midpoint summaries, or transition-level plots alone.

The model must be visually inspectable sequence-by-sequence over time.

---

## 30. Scientific success criterion

The experiment succeeds if it cleanly distinguishes:

```text
a smoother path
```

from:

```text
the correct task-space path
```

The strongest result would be:

\[
\boxed{
\text{same A/B endpoints}
+
\text{directly aligned output-relevant interpolation}
+
\text{held-out Mixed rescue}
}
\]

If that occurs, the next question is:

> What prior experience makes a vanilla SRN learn this path without explicit supervision?
