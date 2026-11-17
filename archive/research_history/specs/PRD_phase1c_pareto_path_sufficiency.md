# PRD: Phase 1C — Pareto Sweep and Specificity Test for Task-Space Path Sufficiency
## Does Getting Closer to the Formal Path Cause Better Held-Out Mixed Transfer While Familiar A/B Remain Fixed?

## 0. Scientific objective

Phase 1B established a useful partial result:

```text
Direct task-space path training improved held-out Mixed performance
while familiar A/B separation and endpoint behavior remained almost unchanged.
```

However, the trained networks only moved partway toward the desired path.

This experiment asks the decisive sufficiency question:

> If we progressively force the task-relevant trajectory toward the formally specified A↔B path, does held-out Mixed performance improve monotonically while familiar A/B endpoints remain stable?

A second specificity question is equally important:

> Does performance improve because the network moves toward the predicted task-space path specifically, or would any equally large representational change help?

This PRD therefore implements:

1. a **Pareto / dose-response sweep** over path-forcing strength and endpoint-anchor strength;
2. a **matched wrong-path control**;
3. mandatory **sequence-level time-series plots** for familiar behavior, probe behavior, and training dynamics.

If this succeeds, the ecology/pretraining experiment can be omitted. The paper can stop at formalization + diagnosis + causal sufficiency.

---

# 1. Core formalization

Let:

```text
k = sequence position
lambda = cue interpolation coefficient
h_k(lambda) = hidden state
ell_k(lambda) = output logits
W_out^0 = frozen baseline output matrix
P_out^0 = projector onto row space(W_out^0)
P_null^0 = I - P_out^0
```

Context interpolation:

\[
c(\lambda)
=
(1-\lambda)c_A+\lambda c_B.
\]

Baseline endpoint hidden states:

\[
h^0_{k,A}, \qquad h^0_{k,B}.
\]

Baseline endpoint logits:

\[
\ell^0_{k,A}, \qquad \ell^0_{k,B}.
\]

Use centered logits:

\[
\tilde\ell
=
\ell-\operatorname{mean}_{classes}(\ell).
\]

Desired endpoint-linear logit path:

\[
\boxed{
\tilde\ell_k^*(\lambda)
=
(1-\lambda)\tilde\ell^0_{k,A}
+
\lambda\tilde\ell^0_{k,B}
}
\]

Desired output-relevant hidden path:

\[
\boxed{
P_{out}^0 h_k^*(\lambda)
=
(1-\lambda)P_{out}^0 h^0_{k,A}
+
\lambda P_{out}^0 h^0_{k,B}
}
\]

No requirement is imposed on:

\[
P_{null}^0 h_k(\lambda).
\]

The target hypothesis is therefore:

> task-relevant interpolation should be compositionally well behaved even if the full hidden representation is not.

---

# 2. Starting checkpoints

Use the same trained High-Scale + High-Context baseline checkpoints from Phase 1B.

For every seed:

1. load the original checkpoint;
2. cache:
   - `h_A^0`
   - `h_B^0`
   - `ell_A^0`
   - `ell_B^0`
   - `W_out^0`
   - output bias
3. preserve original checkpoint hashes;
4. never overwrite source checkpoints.

Primary experiment fine-tunes these existing models.

Also retain frozen Low-Scale + Low-Context checkpoints as a descriptive reference condition only.

---

# 3. Output head policy

Primary experiment:

```text
freeze W_out and output bias
```

Reason:

```text
the task-space target should remain fixed during fine-tuning
```

Trainable parameters:

```text
input-to-hidden weights
recurrent weights
hidden bias
context-input parameters
```

if present in the implementation.

Do not add architecture components.

---

# 4. Endpoint preservation losses

## Hidden anchor

\[
\mathcal L_{h-anchor}
=
\sum_k
\left[
\|h_{k,A}-h^0_{k,A}\|^2
+
\|h_{k,B}-h^0_{k,B}\|^2
\right].
\]

## Centered-logit anchor

\[
\mathcal L_{\ell-anchor}
=
\sum_k
\left[
\|\tilde\ell_{k,A}-\tilde\ell^0_{k,A}\|^2
+
\|\tilde\ell_{k,B}-\tilde\ell^0_{k,B}\|^2
\right].
\]

## Familiar task loss

\[
\mathcal L_{task}
=
CE(A)+CE(B).
\]

These anchors exist only to preserve the already-learned familiar solution.

---

# 5. Correct-path objective

Primary correct-path loss:

\[
\boxed{
\mathcal L_{correct}
=
\sum_{\lambda\in\Lambda_{train}}
\sum_k
\|
\tilde\ell_k(\lambda)
-
\tilde\ell_k^*(\lambda)
\|^2
}
\]

Secondary equivalent hidden-space path loss:

\[
\mathcal L_{correct-hidden}
=
\sum_{\lambda,k}
\|
P_{out}^0
[
h_k(\lambda)-h_k^*(\lambda)
]
\|^2.
\]

Primary Pareto sweep should use the centered-logit path objective because Phase 1B showed the strongest rescue there.

Run the output-hidden version as a preregistered secondary replication.

---

# 6. Held-out midpoint firewall

Use:

```text
Lambda_train =
{0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90}
```

Exclude:

```text
lambda = 0.50
```

from every training loss and hyperparameter-selection metric.

The midpoint remains the primary held-out Mixed test.

Dense post-training evaluation:

```text
lambda = 0.00, 0.05, ..., 1.00
```

---

# 7. Pareto sweep

Instead of selecting one regularizer strength, map the full tradeoff surface.

## Path-strength sweep

Use:

```text
alpha ∈ {
0,
1e-3,
3e-3,
1e-2,
3e-2,
1e-1,
3e-1,
1,
3,
10,
30
}
```

## Anchor-strength sweep

Use at least:

```text
beta ∈ {0.1, 1, 10, 30}
gamma ∈ {0.1, 1, 10, 30}
```

If computational cost is too high:

```text
use beta = gamma
```

and sweep:

```text
anchor_strength ∈ {0.1, 1, 10, 30}
```

Primary total loss:

\[
\mathcal L
=
\mathcal L_{task}
+
\alpha\mathcal L_{correct}
+
\beta\mathcal L_{h-anchor}
+
\gamma\mathcal L_{\ell-anchor}.
\]

Do not choose a single “best” condition before plotting the tradeoff surface.

The entire frontier is the scientific object.

---

# 8. Matched wrong-path control

The correct-path result is not sufficient by itself.

We need to test whether moving the task representation in the **predicted direction** matters.

Construct a matched incorrect target.

Primary wrong-path construction:

1. Compute correct target displacement:
   \[
   d_k(\lambda)
   =
   \tilde\ell_k^*(\lambda)
   -
   \tilde\ell_k^0(\lambda)
   \]
   where \(\tilde\ell_k^0(\lambda)\) is the baseline High network's actual centered logit at that lambda.

2. Construct an orthogonal displacement:
   \[
   d_k^\perp(\lambda)
   \]
   such that:
   \[
   d_k^\perp \perp d_k
   \]
   and:
   \[
   \|d_k^\perp\|=\|d_k\|.
   \]

3. Define wrong target:
   \[
   \tilde\ell_k^{wrong}(\lambda)
   =
   \tilde\ell_k^0(\lambda)
   +
   d_k^\perp(\lambda).
   \]

Use deterministic seeded QR / Gram-Schmidt to construct the orthogonal direction.

The wrong target must preserve the same displacement norm as the correct target.

Wrong-path loss:

\[
\mathcal L_{wrong}
=
\sum_{\lambda,k}
\|
\tilde\ell_k(\lambda)
-
\tilde\ell_k^{wrong}(\lambda)
\|^2.
\]

---

# 9. Secondary wrong-path controls

If feasible, add:

## Permuted-sequence-position target

Permute target sequence-position mappings while preserving target magnitude distribution.

Example:

```text
position 1 target <- position 4
position 2 target <- position 6
...
```

Keep the permutation fixed across seeds.

## Reversed interpolation target

\[
\tilde\ell^{rev}_k(\lambda)
=
(1-\lambda)\tilde\ell^0_{k,B}
+
\lambda\tilde\ell^0_{k,A}.
\]

This is a stronger intentionally incorrect control.

Do not make these mandatory if compute is limited.

Primary wrong-path control remains norm-matched orthogonal displacement.

---

# 10. Experimental families

Run:

```text
P0 = frozen High baseline
P1 = correct centered-logit path sweep
P2 = correct output-hidden path sweep
P3 = matched wrong-path sweep
P4 = endpoint-anchor-only sweep
P5 = frozen Low baseline
```

Optional:

```text
P6 = curvature-proxy sweep
```

for continuity with Phase 1B.

Every P1/P2/P3 model starts from the exact same corresponding seed-level High checkpoint.

---

# 11. Primary metrics

For every seed × condition × alpha × anchor strength:

## Familiar preservation

```text
familiar_loss_ratio
AB_separation_ratio
endpoint_hidden_drift
endpoint_logit_drift
```

## Held-out Mixed

```text
Mixed_loss
Mixed_correct_token_probability
Mixed_sequence_accuracy
Mixed_execution_proxy
Mixed_q
```

## Formal path metrics

```text
heldout_logit_path_residual
heldout_output_path_residual
heldout_whole_path_residual
heldout_null_path_residual
```

## Patching

```text
posttraining_output_patch_loss
remaining_patch_gap
rescue_fraction
```

## Training displacement

```text
actual_logit_change_norm
actual_output_hidden_change_norm
```

This is needed to verify correct and wrong controls make comparable-sized changes.

---

# 12. Normalized path residual

For each seed define:

\[
R_{path}
=
\frac{
E_{logit}^{fine}(0.5)
}{
E_{logit}^{baseline}(0.5)
}.
\]

Interpretation:

```text
1.0 = no movement toward target
0.5 = half baseline path error
0.0 = reaches target
```

Also report analogous:

```text
R_output_path
R_whole_path
```

---

# 13. Endpoint-drift metric

Define normalized endpoint drift:

\[
D_{endpoint}
=
\frac{
RMSE(
[h_A,h_B],
[h_A^0,h_B^0]
)
}{
d_{AB}^0+\epsilon
}.
\]

Also report centered-logit endpoint drift.

Primary “stable endpoint” criterion:

```text
D_endpoint <= 0.10
```

and:

```text
AB separation ratio ∈ [0.90, 1.10]
```

Do not hide continuous values behind the threshold.

---

# 14. Rescue fraction

Using the original baseline patch ceiling:

\[
R_{rescue}
=
\frac{
L_0-L_{fine}
}{
L_0-L_{patch,0}
}.
\]

Compute this **per seed first**, then aggregate.

Also separately report the ratio computed from aggregate means.

The report must state explicitly which version is being shown.

Do not apply this High-baseline rescue denominator to Low-scale controls.

---

# 15. Primary Pareto frontier

The core figure is:

```text
x = normalized held-out path residual R_path
y = held-out Mixed loss
point color = endpoint drift
point size = alpha
facet = correct-path vs wrong-path
```

Also create:

```text
x = endpoint drift
y = held-out path residual
color = Mixed loss
```

The target region is:

```text
low endpoint drift
low path residual
low Mixed loss
```

---

# 16. Dose-response analysis

Within the correct-path family test:

\[
R_{path}\downarrow
\Rightarrow
L_{Mixed}\downarrow
\]

while conditioning descriptively on endpoint drift.

Report:

```text
Pearson r
Spearman rho
seed-clustered regression
bootstrap CI
```

Primary regression:

\[
L_{Mixed}
=
\beta_0
+
\beta_1 R_{path}
+
\beta_2 D_{endpoint}
+
u_{seed}.
\]

Use a mixed model or cluster-robust SE if convenient.

Do not overclaim causality from this regression alone.

The experimental manipulation provides the causal leverage.

---

# 17. Correct-path vs wrong-path specificity analysis

For matched path-strength / anchor-strength cells compare:

```text
correct target
vs
wrong target
```

Match on:

```text
actual task-space displacement magnitude
endpoint drift
familiar loss change
```

Primary specificity question:

> For the same amount of representational movement, does movement toward the formal path improve Mixed more than movement in a matched wrong direction?

Report:

```text
delta Mixed loss
delta correct-token probability
delta rescue fraction
```

per seed.

Paired seed-level comparison preferred.

---

# 18. Strong sufficiency criterion

The formalization is considered strongly supported if there exists a substantial region of the correct-path Pareto frontier satisfying:

```text
endpoint drift <= 0.10
AB separation ratio ∈ [0.90, 1.10]
familiar loss ratio <= 1.10
R_path <= 0.35
Mixed rescue fraction >= 0.75
remaining patch gap <= 0.35 × baseline patch gap
```

AND:

```text
matched wrong-path conditions do not produce comparable Mixed rescue
at comparable displacement magnitude and endpoint drift.
```

Do not require every seed to pass.

Report:

```text
fraction of seeds with a feasible solution
```

for each condition.

---

# 19. If target path cannot be reached

A scientifically important alternative result is:

```text
R_path cannot be pushed below ~0.6–0.7
without endpoint drift >0.10
or familiar degradation.
```

Interpretation:

> The split representation and smooth task-space interpolation may be structurally coupled after learning; the representation cannot easily be retrofitted.

If this occurs, stop.

Do not automatically return to the ecology experiment unless specifically desired later.

This is already a substantive mechanistic result.

---

# 20. If target path is reached but Mixed does not improve

If models achieve:

```text
R_path <= 0.35
```

with stable endpoints, but Mixed remains poor:

```text
reject direct path sufficiency
```

This would invalidate the current formal hypothesis.

Do not continue tuning around it.

---

# 21. Mandatory sequence-level time-series standard

**Every run, summary, and final report must include full sequence-level time-series plots.**

This is a permanent requirement.

The plotting unit must be:

```text
one complete sequence
```

not individual transitions.

Per sequence compute:

```text
mean sequence cross entropy
sequence accuracy
mean correct-token probability
mean execution / RT proxy if available
```

Aggregate over the seven trained transitions within the sequence.

---

# 22. Time series TS1 — fine-tuning trajectory

Plot sequence/update order during fine-tuning for:

```text
familiar sequence loss
A/B separation
endpoint drift
training-lambda path residual
```

Show representative anchor/path strengths and the Pareto-frontier conditions.

Do not plot hundreds of sweep lines in one panel.

Instead include:

```text
baseline
weak path forcing
mid path forcing
strong path forcing
best stable-frontier solution
matched wrong-path solution
anchor-only
```

---

# 23. Time series TS2 — held-out Mixed during fine-tuning

At fixed checkpoints:

```text
0%, 10%, 20%, ..., 100%
```

evaluate midpoint without using it for optimization.

Plot:

```text
held-out Mixed sequence loss
held-out path residual
endpoint drift
remaining patch gap
```

This is mandatory.

---

# 24. Time series TS3 — full familiar experimental schedule

Replay the entire familiar A/B schedule in original order.

For every sequence show:

```text
sequence loss
sequence accuracy
correct-token probability
execution proxy
```

Mark:

```text
A blocks
B blocks
block boundaries
```

At minimum include:

```text
frozen High baseline
best stable correct-path model
matched wrong-path model
anchor-only model
frozen Low baseline
```

---

# 25. Time series TS4 — full experiment / probe schedule

Replay the complete frozen schedule sequence-by-sequence.

Include:

```text
A
B
Mixed
NoContextProbe / Novel only with OOD flag under current encoding
```

Plot:

```text
sequence loss
correct-token probability
execution proxy
```

Mark probe positions explicitly.

This is descriptive only.

---

# 26. Time series TS5 — dense cue sweep

For:

```text
lambda = 0.00, 0.05, ..., 1.00
```

plot ordered line series for:

```text
sequence loss
correct-token probability
logit path residual
output-hidden path residual
```

Compare:

```text
baseline High
best stable correct-path model
matched wrong-path model
anchor-only
Low baseline
```

---

# 27. Mandatory sequence-level tables

Save one row per:

```text
seed × condition × hyperparameter cell × sequence/timepoint
```

for all time-series outputs.

Required:

```text
phase1c_timeseries_finetuning.csv
phase1c_timeseries_midpoint_checkpoints.csv
phase1c_timeseries_familiar_schedule.csv
phase1c_timeseries_full_probe_schedule.csv
phase1c_timeseries_cue_sweep.csv
```

---

# 28. Required summary CSVs

```text
phase1c_pareto_grid.csv
phase1c_seed_metrics.csv
phase1c_endpoint_metrics.csv
phase1c_path_metrics.csv
phase1c_specificity_controls.csv
phase1c_rescue_metrics.csv
phase1c_feasible_frontier.csv
```

---

# 29. Required figures

## Pareto / formalization figures

```text
P1C_01_path_residual_vs_Mixed_loss.png
P1C_02_endpoint_drift_vs_path_residual.png
P1C_03_endpoint_drift_vs_Mixed_loss.png
P1C_04_correct_vs_wrong_path.png
P1C_05_displacement_matched_specificity.png
P1C_06_rescue_fraction_frontier.png
P1C_07_remaining_patch_gap_frontier.png
P1C_08_feasible_region_by_seed.png
```

## Mandatory time-series figures

```text
P1C_TS01_finetuning_sequence_loss.png
P1C_TS02_finetuning_path_and_endpoint_metrics.png
P1C_TS03_midpoint_over_finetuning.png

P1C_TS04_full_familiar_sequence_loss.png
P1C_TS05_full_familiar_sequence_accuracy.png
P1C_TS06_full_familiar_correct_token_probability.png
P1C_TS07_full_familiar_execution_proxy.png

P1C_TS08_full_probe_sequence_loss.png
P1C_TS09_full_probe_correct_token_probability.png
P1C_TS10_full_probe_execution_proxy.png

P1C_TS11_cue_sweep_sequence_loss.png
P1C_TS12_cue_sweep_correct_token_probability.png
P1C_TS13_cue_sweep_logit_path_residual.png
P1C_TS14_cue_sweep_output_path_residual.png
```

If execution proxy is unavailable, write an explicit text artifact documenting why.

---

# 30. TDD / RED → GREEN

## T1 — midpoint firewall

Assert:

```text
0.50 not in Lambda_train
```

for every run.

## T2 — source checkpoint integrity

Verify original checkpoint hashes are unchanged.

## T3 — output-head freeze

Primary sweep must satisfy:

```text
W_out_final == W_out^0
b_out_final == b_out^0
```

## T4 — target endpoint exactness

At:

```text
lambda=0 -> target equals cached A
lambda=1 -> target equals cached B
```

## T5 — centered-logit invariance

Common additive logit shifts must not alter correct-path loss.

## T6 — wrong-path orthogonality

For every target vector:

```text
dot(d_correct, d_wrong) ≈ 0
```

## T7 — wrong-path norm matching

Verify:

```text
||d_wrong|| = ||d_correct||
```

within numerical tolerance.

## T8 — wrong-path determinism

Wrong-path construction must be reproducible from saved seed.

## T9 — endpoint anchor zero test

Unmodified baseline checkpoint gives zero anchor loss within tolerance.

## T10 — no midpoint selection

Held-out midpoint results cannot affect:

```text
learning rate
alpha
anchor strength
checkpoint selection
early stopping
```

## T11 — sequence aggregation

Every time-series point must aggregate exactly one full sequence.

## T12 — time-series completeness

Fail report generation if any mandatory sequence-level time-series figure is missing.

## T13 — schedule integrity

Full familiar and probe replays must preserve registered sequence/context order.

## T14 — rescue-fraction transparency

Save both:

```text
per-seed rescue fraction
ratio-of-aggregate-means rescue fraction
```

and label them separately.

## T15 — Low control denominator

Do not compute High-baseline rescue fraction for frozen Low control.

---

# 31. Report

Generate:

```text
PHASE1C_PARETO_PATH_SUFFICIENCY_REPORT.md
```

Required sections:

```text
1. Executive result
2. Formal hypothesis
3. Sweep design
4. Endpoint-preservation frontier
5. Path-residual frontier
6. Mixed dose-response
7. Correct-path vs wrong-path specificity
8. Rescue fraction
9. Remaining patch gap
10. Full sequence-level fine-tuning time series
11. Full sequence-level familiar time series
12. Full sequence-level probe time series
13. Dense cue-sweep time series
14. Seed-level heterogeneity
15. Feasible-region analysis
16. Sufficiency conclusion
17. Paper-level interpretation
```

Sections 10–13 are mandatory.

---

# 32. Required explicit questions

The report must answer:

### Q1
Can the task-space path be pushed substantially closer to the formal target while familiar A/B endpoints remain stable?

### Q2
Is there a dose-response relation:

\[
R_{path}\downarrow \Rightarrow L_{Mixed}\downarrow?
\]

### Q3
Does the best stable frontier approach the original causal-patching ceiling?

### Q4
Does the remaining patch gap shrink as path residual shrinks?

### Q5
Does a matched wrong-path intervention produce the same benefit?

### Q6
For the same amount of representational movement, is the correct-path intervention more effective than the wrong-path control?

### Q7
What fraction of seeds admit a low-drift / low-path-error / low-Mixed-loss solution?

### Q8
Do the full sequence-level time-series plots show any familiar-performance degradation hidden by aggregate metrics?

---

# 33. Final interpretation logic

## Strong support for sufficiency

If:

```text
correct-path conditions show a graded path-error → Mixed improvement relation,
stable endpoints,
approach the patching ceiling,
and outperform displacement-matched wrong-path controls
```

conclude:

> The formally specified task-space interpolation geometry is sufficient for zero-shot Mixed transfer while preserving familiar context splitting.

At this point the ecology experiment is optional and not required for the current paper.

---

## Partial support

If:

```text
correct-path conditions improve Mixed,
but the target cannot be reached without moderate endpoint drift
```

conclude:

> The path is behaviorally useful, but the learned High-Scale representation resists being retrofitted into the desired geometry.

This is still a substantive result.

---

## Reject sufficiency

If:

```text
path residual reaches the target region,
endpoints remain stable,
but Mixed does not improve,
or matched wrong-path moves help equally
```

reject the formal path-sufficiency hypothesis.

Stop iterating on this mechanism.

---

# 34. Permanent project standard

For all future SRN analyses in this project:

> **Always include full sequence-level time-series plots for the familiar schedule, the complete probe schedule, and any training/fine-tuning trajectory.**

Condition means are supplementary.

The primary results must remain visually inspectable sequence-by-sequence over time.

---

# 35. Scientific success criterion

The strongest result would be:

\[
\boxed{
\text{same familiar A/B endpoints}
+
\text{graded movement toward formal task-space path}
+
\text{graded held-out Mixed rescue}
+
\text{no comparable rescue from matched wrong-path movement}
}
\]

That is sufficient to support a mechanistic claim without requiring an additional pretraining-ecology experiment.
