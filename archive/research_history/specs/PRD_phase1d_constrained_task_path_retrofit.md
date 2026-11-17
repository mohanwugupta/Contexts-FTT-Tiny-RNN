# PRD: Phase 1D — Constrained Task-Path Retrofit Across Seeds

## 0. Scientific goal

The current Phase 1C result shows that a High-scale split SRN **can** be moved toward the formal compositional task-space path while preserving familiar A/B behavior, and that doing so improves held-out Mixed performance. However, only a minority of seeds passed the full sufficiency gate.

This phase asks:

> Is the low seed-level success rate a limitation of the proposed representational solution, or a limitation of the soft-penalty optimization procedure used to retrofit that solution?

Replace the previous soft tradeoff

\[
\mathcal L
=
\alpha \mathcal L_{\text{path}}
+
\beta \mathcal L_{\text{anchor}}
\]

with constrained optimization:

\[
\min_\theta \mathcal L_{\text{path}}(\theta)
\]

subject to the familiar endpoint computation remaining fixed.

The primary implementation should use **projected / equality-constrained gradient updates** so the optimizer searches only parameter directions that improve interpolation geometry without changing the familiar task-space endpoints to first order.

This phase also performs a per-seed **acute reachability audit** and a local Jacobian analysis so we can distinguish:

1. the compositional state works behaviorally;
2. the network parameterization can locally realize it without moving endpoints;
3. the optimizer can actually reach it.

Do **not** introduce a new architecture or ecology in the primary experiment.

---

# 1. Core hypotheses

## H1 — Acute representational sufficiency is broad

For most High-scale seeds, directly replacing the output-relevant Mixed state with the formal compositional state should improve Mixed sequence performance.

If this is true but learned retrofit succeeds in only a few seeds, then the weak success rate is primarily a **learnability / optimization problem**, not evidence against the task-space mechanism.

## H2 — Feasible parameter directions exist

For many seeds, there should be directions in parameter space that:

1. preserve the familiar endpoint task states;
2. reduce the non-midpoint task-path error.

## H3 — Constrained optimization improves seed-level retrofit success

Compared with the previous soft-penalty Phase 1C method, constrained updates should increase the fraction of High-scale seeds that:

- preserve familiar A/B performance;
- preserve A/B separation;
- preserve familiar endpoint task states;
- approach the compositional task path;
- improve held-out Mixed performance;
- leave little additional benefit for acute patching.

## H4 — Direction matters

A matched constrained intervention toward a wrong/orthogonal path should not yield the same Mixed improvement.

---

# 2. Non-goals

Do **not**:

```text
change the SRN architecture
change hidden size
add an adapter
add a new decoder
train on Mixed
train on lambda = 0.5
use held-out Mixed performance to choose hyperparameters
change the Phase 1C success thresholds after seeing results
add a new ecology or pretraining curriculum
move to item-level behavioral analysis
```

The primary scientific comparison is:

```text
same High-scale seeds
same formal path target
same familiar endpoints
old soft-penalty retrofit
vs
new constrained retrofit
```

---

# 3. Required source artifacts

Load the exact High-scale checkpoints and Phase 1C artifacts already used previously.

Expected inputs include the equivalents of:

```text
High-scale baseline checkpoints for all seeds
Low-scale baseline checkpoints
Phase 1C correct-path targets
Phase 1C matched wrong-path targets
Phase 1C configuration / manifest
Phase 1C success-gate thresholds
Phase 1C acute patch implementation
Phase 1C sequence-level evaluation code
```

If a formal target, seed checkpoint, selection rule, or success threshold cannot be recovered exactly from provenance, **fail loudly**.

Do not silently reconstruct or change it.

---

# 4. Freeze the held-out test

The held-out ambiguous context remains:

\[
\lambda_{\text{test}}=0.5.
\]

No optimization step, early-stopping decision, learning-rate selection, checkpoint selection, or algorithm selection may use:

```text
Mixed loss
Mixed correct-token probability
Mixed execution proxy
lambda = 0.5 path error
```

Use the **same non-midpoint training cue values used in Phase 1C**.

If the original training cue set cannot be recovered, stop rather than inventing a new one.

The endpoints:

\[
\lambda=0,\;1
\]

are used only as familiar constraints / anchors.

---

# 5. Formal task-space object

For sequence position \(k\), hidden state:

\[
h_k(c;\theta)\in\mathbb R^d.
\]

Frozen output logits:

\[
\ell_k(c;\theta)=W_{out}h_k(c;\theta)+b.
\]

Define:

\[
C=I-\frac1m\mathbf 1\mathbf 1^\top.
\]

Task-space state:

\[
\boxed{
z_k(c;\theta)
=
C\ell_k(c;\theta)
}
\]

The output map remains frozen in all retrofit experiments.

---

# 6. Formal endpoint-linear target

For:

\[
c_\lambda=(1-\lambda)c_A+\lambda c_B,
\]

define source-checkpoint familiar endpoints:

\[
z^0_{k,A}=z_k(c_A;\theta_0),
\qquad
z^0_{k,B}=z_k(c_B;\theta_0).
\]

The formal target is:

\[
\boxed{
z^*_{k}(\lambda)
=
(1-\lambda)z^0_{k,A}
+
\lambda z^0_{k,B}
}
\]

Critically, the target is defined from the **original source checkpoint** and remains fixed throughout retrofit training.

Do not redefine the target using drifting current endpoints.

---

# 7. Phase A — Acute per-seed reachability audit

Before new training, test whether the representational solution itself works seed-by-seed.

For every High-scale seed:

1. evaluate baseline Mixed;
2. apply the existing acute output-relevant compositional patch at \(\lambda=.5\);
3. evaluate the same Mixed sequence.

Save:

```text
seed
baseline_mixed_loss
patched_mixed_loss
delta_mixed_loss = baseline - patched
baseline_correct_prob
patched_correct_prob
delta_correct_prob
baseline_execution
patched_execution
delta_execution
AB_separation
baseline_task_path_residual
```

Primary count:

```text
N seeds with patched_mixed_loss < baseline_mixed_loss
```

Also report:

```text
mean delta loss
median delta loss
seed-bootstrap 95% CI
fraction improved
```

Generate:

```text
PHASE1D_FIG01_acute_patch_by_seed.png
```

with one paired point/line per seed.

Do not use this result to tune the constrained optimizer.

---

# 8. Phase B — Local parameter-space reachability analysis

We want to know whether, around each trained High checkpoint, there are parameter directions that:

1. preserve familiar endpoint task states;
2. reduce the non-midpoint task-path error.

This analysis uses **no Mixed data**.

---

# 9. Define endpoint constraint vector

Let the trainable parameter vector be:

\[
\theta\in\mathbb R^P.
\]

Define the familiar endpoint constraint vector:

\[
F(\theta)
=
\operatorname{vec}
\left(
\left\{
z_k(A;\theta),
z_k(B;\theta)
\right\}_{k=1}^{K}
\right).
\]

Reference:

\[
F_0=F(\theta_0).
\]

Constraint:

\[
\boxed{
c(\theta)=F(\theta)-F_0=0
}
\]

Primary hard constraint is in **task space**, not full hidden space.

This is deliberate: the theory says the full hidden representation may remain context-specific and may move in output-null directions.

Continue to **measure** hidden endpoint drift and A/B separation as guardrails.

---

# 10. Constraint Jacobian

For each seed compute:

\[
J_F
=
\frac{\partial F}{\partial\theta}.
\]

Use autograd Jacobian computation.

Because the network is small, exact Jacobians are preferred.

Use numerically stable SVD / pseudoinverse.

Save:

```text
constraint_rank
parameter_count
free_local_dimension = parameter_count - constraint_rank
constraint_singular_values
constraint_condition_number
```

Use a relative SVD threshold unless the codebase already has a fixed numerical convention.

---

# 11. Define path residual vector

Across the exact non-midpoint training cue values and all 7 sequence transitions, define:

\[
r(\theta)
=
\operatorname{vec}
\left(
z_k(c_\lambda;\theta)-z_k^*(\lambda)
\right).
\]

Path loss:

\[
\mathcal L_{\text{path}}
=
\frac12\|r(\theta)\|_2^2.
\]

Compute:

\[
J_r
=
\frac{\partial r}{\partial\theta}.
\]

---

# 12. First-order local feasibility diagnostic

Let \(N\) be an orthonormal basis for the null space of the endpoint Jacobian:

\[
J_F N = 0.
\]

Any first-order endpoint-preserving parameter update can be written:

\[
\delta\theta=N a.
\]

Solve:

\[
a^*
=
\arg\min_a
\|
r + J_rNa
\|_2^2.
\]

Then:

\[
r_{\text{best,local}}
=
r+J_rNa^*.
\]

Define:

\[
\boxed{
Q_{\text{local}}
=
1-
\frac{
\|r_{\text{best,local}}\|_2^2
}{
\|r\|_2^2+\epsilon
}
}
\]

Interpretation:

```text
Q_local ≈ 1:
large first-order path correction is available while preserving endpoints.

Q_local ≈ 0:
the local parameterization offers almost no endpoint-preserving direction
that fixes the task path.
```

Save per seed.

---

# 13. Projected-gradient availability

Let:

\[
g=\nabla_\theta\mathcal L_{\text{path}}.
\]

Define the tangent-space projector:

\[
P_T
=
I
-
J_F^\top
(J_FJ_F^\top+\epsilon I)^{+}
J_F.
\]

Compute:

\[
g_T=P_Tg.
\]

Save:

\[
\boxed{
R_g
=
\frac{\|g_T\|_2}{\|g\|_2+\epsilon}
}
\]

and:

```text
projected_gradient_norm
raw_gradient_norm
projected_gradient_fraction
predicted_first_order_path_decrease
```

---

# 14. Phase B diagnostic figure

Generate:

```text
PHASE1D_FIG02_local_reachability.png
```

Panels:

```text
A. Q_local by seed
B. projected-gradient fraction by seed
C. free local dimension by seed
D. Q_local vs previous Phase 1C retrofit success
```

No Mixed behavior is needed for Panels A–C.

---

# 15. Phase C — Primary constrained retrofit

Start from the exact original High checkpoint \(\theta_0\) for every seed.

Do not start from Phase 1C fine-tuned weights.

Optimize only:

\[
\mathcal L_{\text{path}}.
\]

The familiar endpoint computation is enforced by the constrained update.

---

# 16. Equality-constrained update

At current \(\theta\), compute:

\[
c=F(\theta)-F_0,
\]

endpoint Jacobian:

\[
J=J_F(\theta),
\]

and unconstrained path step:

\[
d_0=-\eta\nabla_\theta\mathcal L_{\text{path}}.
\]

Find the closest step to \(d_0\) satisfying the linearized equality correction:

\[
Jd=-c.
\]

Use:

\[
\boxed{
d
=
d_0
-
J^\top
(JJ^\top+\rho I)^{+}
(Jd_0+c)
}
\]

Then:

\[
\theta\leftarrow\theta+d.
\]

Use an SVD/pseudoinverse implementation.

This is preferred over a pure tangent projection because it also corrects accumulated endpoint constraint drift.

---

# 17. Numerical damping and step-size selection

\(\rho\) is numerical damping, not a model parameter.

Use a very small fixed candidate set, e.g.:

```text
rho_relative ∈ {1e-8, 1e-6, 1e-4}
```

Select **globally** using only:

```text
endpoint constraint residual
training-lambda task-path residual
numerical stability
```

For learning rate, use a small fixed candidate set, e.g.:

```text
learning_rate ∈ {1e-4, 3e-4, 1e-3}
```

but reuse Phase 1C scales if already established.

Select globally using only training objective and constraints.

Never use Mixed.

---

# 18. Trust-region safeguard

Because the equality constraint is linearized locally, cap step norm:

\[
\|d\|_2 \le \tau.
\]

Define \(\tau\) relative to \(\|\theta_0\|\) or the established Phase 1C update scale.

If a proposed step violates the cap:

```text
rescale d
recompute endpoint residual after step
```

If endpoint residual increases beyond tolerance:

```text
reject step
halve step size
retry
```

No Mixed metric may participate in line search.

---

# 19. Trainable parameters

Primary condition:

```text
all recurrent/core SRN parameters that were trainable in Phase 1C
```

Keep:

```text
output/readout weights frozen
output bias frozen
```

If Phase 1C froze additional parameters, preserve the same freeze mask.

Do not increase trainable parameter count relative to the previous retrofit without explicitly reporting it.

---

# 20. Checkpoint selection

Save checkpoints at the same cadence as Phase 1C.

Select checkpoint using:

1. endpoint constraint residual below tolerance;
2. familiar A/B guardrails satisfied;
3. minimum **training-lambda** task-path residual.

No held-out midpoint metric may affect checkpoint selection.

---

# 21. Familiar guardrails

Evaluate:

```text
full familiar sequence loss
full familiar correct-token probability
full familiar execution proxy
A/B hidden separation ratio
endpoint hidden drift
endpoint task-space drift
```

Use the exact Phase 1C thresholds from provenance.

Do **not** change them.

Report two labels:

### `phase1c_comparable_success`

Uses the exact original Phase 1C full gate.

### `task_space_mechanistic_success`

Uses:

```text
task endpoint preservation
familiar performance preservation
A/B separation preservation
task-path target
held-out behavioral rescue
remaining patch-gap target
```

but does not require tiny full-hidden endpoint drift.

This secondary label is explanatory only.

Do not replace the Phase 1C-comparable success rate.

---

# 22. Phase D — Constrained matched wrong-path control

Run the exact same constrained optimization for a matched wrong target.

Reuse the Phase 1C wrong-path target construction if available.

For each seed and training cue:

1. wrong target must be orthogonal to the correct task-space correction;
2. target displacement norm must match the correct-path displacement;
3. familiar endpoint constraints must be identical;
4. trainable parameters must be identical;
5. optimizer must be identical;
6. number of updates must be identical;
7. checkpoint rule must be identical.

If desired correct displacement is:

\[
\Delta z_{\text{correct}},
\]

construct:

\[
\Delta z_{\text{wrong}}
\]

such that:

\[
\Delta z_{\text{wrong}}
\perp
\Delta z_{\text{correct}}
\]

and:

\[
\|\Delta z_{\text{wrong}}\|
=
\|\Delta z_{\text{correct}}\|.
\]

---

# 23. Phase E — Held-out evaluation

Only after optimization and checkpoint selection are frozen, evaluate:

```text
lambda = 0.5 Mixed
```

for:

```text
Baseline High
Acute correct-path patch
Previous Phase 1C soft-penalty retrofit
New constrained correct-path retrofit
New constrained wrong-path retrofit
Baseline Low
```

Primary metric:

```text
sequence-level Mixed loss
```

Secondary:

```text
correct-token probability
sequence accuracy
execution proxy
```

---

# 24. Remaining acute patch test

For each constrained correct-path checkpoint, apply the acute output-path patch at Mixed again.

Define:

\[
L_0 = \text{baseline High Mixed loss}
\]

\[
L_{\text{patch0}} = \text{baseline acute-patched Mixed loss}
\]

\[
L_{\text{constrained}} = \text{constrained retrofit Mixed loss}
\]

\[
L_{\text{constrained+patch}}
=
\text{constrained retrofit + acute patch Mixed loss}
\]

Then:

\[
F_{\text{rescue}}
=
\frac{
L_0-L_{\text{constrained}}
}{
L_0-L_{\text{patch0}}
}
\]

and:

\[
G_{\text{remain}}
=
L_{\text{constrained}}
-
L_{\text{constrained+patch}}.
\]

Also report:

\[
D_{\text{available}}
=
L_0-L_{\text{patch0}}.
\]

This denominator must be shown per seed.

Flag near-zero denominators and emphasize absolute rescue / remaining patch gap for those seeds.

---

# 25. Primary success comparison

Generate:

```text
PHASE1D_FIG03_success_rate_comparison.png
```

Show:

```text
Phase 1C soft-penalty: N/20
Phase 1D constrained correct path: N/20
Phase 1D constrained wrong path: N/20
```

using the **exact Phase 1C-comparable gate**.

Also report Wilson 95% intervals.

---

# 26. Per-seed outcome matrix

Generate:

```text
PHASE1D_FIG04_seed_outcome_matrix.png
```

Rows:

```text
seed
```

Columns:

```text
acute patch improves Mixed
high Q_local
soft-penalty success
constrained correct-path success
constrained wrong-path success
```

Sort by \(Q_{local}\) or constrained training-path residual.

---

# 27. Mechanistic diagnostic plot

Generate:

```text
PHASE1D_FIG05_reachability_predicts_learning.png
```

Panels:

```text
A. Q_local vs final constrained training-path residual
B. projected-gradient fraction vs final constrained training-path residual
C. Q_local vs held-out Mixed improvement
D. available acute rescue vs constrained Mixed improvement
```

Interpretation:

```text
High acute rescue + high Q_local + failed optimization
    -> optimizer problem.

High acute rescue + low Q_local
    -> state-level solution exists but is locally hard to realize in parameters.

Low acute rescue
    -> task-space formalization itself may not solve that seed.
```

---

# 28. Correct-vs-wrong specificity

Generate:

```text
PHASE1D_FIG06_correct_vs_wrong_constrained.png
```

Panel A:

```text
x = constrained correct-path Mixed loss
y = constrained wrong-path Mixed loss
```

Identity line.

Panel B:

```text
wrong - correct Mixed loss
```

paired distribution.

Report:

```text
mean paired difference
median paired difference
seed-bootstrap 95% CI
fraction correct < wrong
```

Verify achieved correct/wrong displacement norms are matched.

---

# 29. Mandatory cue-sweep plots

Generate sequence-level:

```text
PHASE1D_CUE01_task_path_residual.png
PHASE1D_CUE02_sequence_loss.png
PHASE1D_CUE03_correct_token_probability.png
PHASE1D_CUE04_execution_proxy.png
PHASE1D_CUE05_AB_axis_and_off_axis_geometry.png
```

Conditions:

```text
Baseline High
Baseline Low
Constrained correct path
Constrained wrong path
Previous best Phase 1C soft-penalty condition
```

Mark \(\lambda=.5\) as held out.

---

# 30. Mandatory sequence-level familiar time series

Every point must summarize one complete sequence over seven transitions.

Generate:

```text
PHASE1D_TS01_familiar_sequence_loss.png
PHASE1D_TS02_familiar_correct_token_probability.png
PHASE1D_TS03_familiar_execution_proxy.png
```

Across the complete familiar A/B schedule.

Conditions:

```text
Baseline High
Constrained correct path
Constrained wrong path
Previous Phase 1C correct-path retrofit
Baseline Low
```

The plots must make it visually obvious whether the intervention preserved the original alternating A/B behavior.

---

# 31. Mandatory full probe sequence-level time series

Generate:

```text
PHASE1D_TS04_full_probe_sequence_loss.png
PHASE1D_TS05_full_probe_correct_token_probability.png
PHASE1D_TS06_full_probe_execution_proxy.png
```

Include all available probe contexts:

```text
Mixed
Novel
NoContextProbe
```

Probes remain read-only evaluation.

---

# 32. Mandatory training trajectories

Generate:

```text
PHASE1D_TS07_training_path_residual.png
PHASE1D_TS08_endpoint_constraint_violation.png
PHASE1D_TS09_AB_separation_ratio.png
PHASE1D_TS10_familiar_loss_ratio.png
```

Compare:

```text
soft-penalty Phase 1C
constrained correct path
constrained wrong path
```

where possible.

---

# 33. Quantitative comparison table

Create:

```text
PHASE1D_TABLE01_method_comparison.csv
```

Rows:

```text
Baseline High
Acute patch
Phase 1C soft-penalty correct path
Phase 1D constrained correct path
Phase 1D constrained wrong path
Baseline Low
```

Columns:

```text
n_seeds
phase1c_comparable_success_n
phase1c_comparable_success_fraction
mixed_loss_mean
mixed_loss_median
mixed_loss_bootstrap_low
mixed_loss_bootstrap_high
mixed_correct_prob_mean
execution_proxy_mean
task_path_residual_mean
AB_separation_ratio_mean
endpoint_hidden_drift_mean
endpoint_task_drift_mean
absolute_rescue_mean
per_seed_rescue_fraction_mean
remaining_patch_gap_mean
remaining_patch_gap_ratio_mean
```

---

# 34. Per-seed master table

Create:

```text
PHASE1D_seed_master.csv
```

Required columns:

```text
seed

baseline_high_mixed_loss
acute_patch_mixed_loss
acute_patch_delta
available_rescue

constraint_rank
parameter_count
free_local_dimension
Q_local
raw_gradient_norm
projected_gradient_norm
projected_gradient_fraction
predicted_first_order_path_decrease

phase1c_best_path_residual
phase1c_mixed_loss
phase1c_success

constrained_correct_training_path_residual
constrained_correct_midpoint_path_residual
constrained_correct_mixed_loss
constrained_correct_correct_prob
constrained_correct_execution
constrained_correct_AB_separation_ratio
constrained_correct_endpoint_hidden_drift
constrained_correct_endpoint_task_drift
constrained_correct_familiar_loss_ratio
constrained_correct_rescue_fraction
constrained_correct_remaining_patch_gap
constrained_correct_phase1c_comparable_success
constrained_correct_task_space_mechanistic_success

constrained_wrong_training_path_residual
constrained_wrong_midpoint_path_residual
constrained_wrong_mixed_loss
constrained_wrong_correct_prob
constrained_wrong_execution
constrained_wrong_AB_separation_ratio
constrained_wrong_endpoint_hidden_drift
constrained_wrong_endpoint_task_drift
constrained_wrong_familiar_loss_ratio

wrong_minus_correct_mixed_loss
wrong_minus_correct_correct_prob
correct_displacement
wrong_displacement
displacement_ratio

rescue_denominator_small
```

---

# 35. Decision logic

The report must explicitly distinguish:

## Outcome 1 — Broad acute rescue + broad constrained retrofit

> The task-space compositional solution is broadly sufficient and the previous 4/20 result was largely an optimization artifact.

## Outcome 2 — Broad acute rescue + high local reachability + poor constrained retrofit

> The solution is available and locally parameter-accessible, but the constrained optimizer is not solving the numerical problem well.

## Outcome 3 — Broad acute rescue + low local reachability in failed seeds

> The compositional state would solve behavior, but many trained High solutions lie in parameter basins where that geometry cannot be reached locally while familiar endpoints remain fixed.

This motivates a future from-scratch inductive-bias experiment.

## Outcome 4 — Acute patch rescue itself is rare

> The formal task-space path is not a broadly sufficient state-level solution across High seeds.

Do not keep optimizing merely to increase the success rate.

---

# 36. Optional Stage 2 — From-scratch compositional inductive bias

**Implement behind a disabled configuration flag. Do not run by default.**

Purpose:

> Test whether the desired geometry is easier to learn during acquisition than retrofit after a strongly split solution has formed.

Training objective:

\[
\mathcal L
=
\mathcal L_{\text{sequence}}
+
\alpha\mathcal L_{\text{local-composition}}.
\]

Never train on:

\[
\lambda=.5.
\]

A clean future design would regularize only local cue neighborhoods near familiar endpoints and test the unseen center.

---

# 37. TDD / RED → GREEN tests

## T1 — Held-out midpoint firewall

Fail if any training/selection function accesses \(\lambda=.5\) outcomes.

## T2 — Frozen formal target

Fail if \(z^*(\lambda)\) changes during fine-tuning.

## T3 — Output weights frozen

Hash before/after. Fail on change.

## T4 — Constraint identity at initialization

```text
||F(theta0) - F0|| < tolerance
```

## T5 — Projector correctness

Verify:

\[
J_FP_T \approx 0.
\]

## T6 — Linearized constrained update

Verify:

\[
\|Jd+c\|
\]

is below numerical tolerance before applying the step.

## T7 — Null-basis correctness

Verify:

\[
J_FN\approx0
\]

and:

\[
N^\top N\approx I.
\]

## T8 — Local feasibility bounded

Verify:

```text
0 <= Q_local <= 1 + numerical_tolerance
```

## T9 — Wrong-path orthogonality

Verify:

\[
\Delta z_{\text{wrong}}^\top
\Delta z_{\text{correct}}
\approx0.
\]

## T10 — Wrong-path norm match

Verify:

\[
\frac{
\|\Delta z_{\text{wrong}}\|
}{
\|\Delta z_{\text{correct}}\|
}
\approx1.
\]

## T11 — Same optimizer exposure

Correct and wrong conditions must have identical:

```text
number of updates
learning-rate rule
projection rule
trainable parameter mask
checkpoint cadence
```

## T12 — No per-seed Mixed selection

Fail if Mixed appears in checkpoint ranking.

## T13 — Phase 1C gate reuse

Fail if Phase 1D redefines the Phase 1C-comparable thresholds.

## T14 — Sequence-level reporting

Each mandatory time-series observation must aggregate exactly seven transitions.

## T15 — No item-level replacement

Fail the final report build if a mandatory sequence-level figure is absent.

## T16 — Rescue denominator exposed

Fail if rescue fraction is reported without available rescue per seed.

## T17 — Provenance

All headline numbers must be reproducible from saved CSVs.

---

# 38. Output directory

Create:

```text
phase1d_constrained_retrofit/
```

with:

```text
audit/
jacobians/
checkpoints/
tables/
figures/
timeseries/
reports/
```

Required outputs:

```text
tables/PHASE1D_seed_master.csv
tables/PHASE1D_TABLE01_method_comparison.csv
tables/PHASE1D_acute_patch_audit.csv
tables/PHASE1D_local_reachability.csv
tables/PHASE1D_constrained_correct.csv
tables/PHASE1D_constrained_wrong.csv

figures/PHASE1D_FIG01_acute_patch_by_seed.png
figures/PHASE1D_FIG02_local_reachability.png
figures/PHASE1D_FIG03_success_rate_comparison.png
figures/PHASE1D_FIG04_seed_outcome_matrix.png
figures/PHASE1D_FIG05_reachability_predicts_learning.png
figures/PHASE1D_FIG06_correct_vs_wrong_constrained.png

figures/PHASE1D_CUE01_task_path_residual.png
figures/PHASE1D_CUE02_sequence_loss.png
figures/PHASE1D_CUE03_correct_token_probability.png
figures/PHASE1D_CUE04_execution_proxy.png
figures/PHASE1D_CUE05_AB_axis_and_off_axis_geometry.png

timeseries/PHASE1D_TS01_familiar_sequence_loss.png
timeseries/PHASE1D_TS02_familiar_correct_token_probability.png
timeseries/PHASE1D_TS03_familiar_execution_proxy.png
timeseries/PHASE1D_TS04_full_probe_sequence_loss.png
timeseries/PHASE1D_TS05_full_probe_correct_token_probability.png
timeseries/PHASE1D_TS06_full_probe_execution_proxy.png
timeseries/PHASE1D_TS07_training_path_residual.png
timeseries/PHASE1D_TS08_endpoint_constraint_violation.png
timeseries/PHASE1D_TS09_AB_separation_ratio.png
timeseries/PHASE1D_TS10_familiar_loss_ratio.png
```

---

# 39. Final report

Generate:

```text
reports/PHASE1D_CONSTRAINED_RETROFIT_REPORT.md
```

Required structure:

```text
1. Executive conclusion
2. Why Phase 1D was run
3. Acute patch success across seeds
4. Local endpoint-preserving reachability
5. Constrained optimizer implementation
6. Familiar endpoint preservation
7. Held-out Mixed results
8. Comparison with Phase 1C soft penalties
9. Correct-vs-wrong specificity
10. Seed-level success rate
11. Sequence-level familiar trajectories
12. Sequence-level probe trajectories
13. Failure-mode decomposition
14. Whether Stage 2 is warranted
15. Final scientific interpretation
```

---

# 40. Key scientific question

Do **not** summarize Phase 1D merely as:

> constrained optimization improved performance.

The report must determine which is true:

### Representational limitation
The compositional state itself does not rescue most seeds.

### Parameter-access limitation
The state would rescue them, but the trained recurrent solution has few endpoint-preserving parameter directions capable of producing it.

### Optimization limitation
The needed directions exist, and hard-constrained optimization reaches them more reliably than the old soft-penalty method.

---

# 41. Target paper-level interpretation if successful

If acute patching works broadly and constrained retrofit substantially increases seed-level success while wrong-path controls do not, the paper can support:

> Strong context-specific representations do not inherently prevent generalization. The critical constraint is the geometry of the task-relevant mapping between familiar solutions. When optimization is restricted to parameter changes that preserve familiar endpoint computations, split SRNs can acquire a more compositional interpolation geometry and recover ambiguous-context transfer without collapsing the familiar representations.

Do not claim necessity or universality unless the results actually establish them.
