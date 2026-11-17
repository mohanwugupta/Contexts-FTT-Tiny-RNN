# PRD — Phase 1L: Constructive Recovery of Shared Transfer
## Familiar-Contrast Bottleneck + Neutral-Trajectory Closure

## Status

**Proposed final constructive SRN experiment.**

This phase is not another diagnostic search. It is a prospective test of whether the representational understanding from Phases 1J–1K is sufficient to *induce* the missing generalization phenotype.

The core prediction is:

> If familiar contexts are represented as opposite modulations of a shared sequence computation, and the neutral/shared sequence trajectory is made dynamically executable, then Mixed, NoContext, and Novel cues should all access the shared route without ever being trained as behavioral targets.

The experiment must preserve the main positive property of the canonical High-scale SRN:

- strong A/B contextual separation,

while recovering the missing property:

- good transfer/generalization under Mixed, NoContext, and Novel cues.

This is intended to be the final modeling experiment. If the full intervention succeeds or cleanly fails, stop and write.

---

# 1. Mechanistic motivation

The previous phases established two separable limitations.

### Limitation 1 — The useful shared/composite state is not naturally generated

Directly placing the High-scale SRN onto the familiar A/B-derived composite trajectory can strongly improve behavior, but Phase 1K showed that the physical Mixed cue does not reliably generate that state from ordinary A- or B-specific prestates.

This motivates changing the **context-routing geometry**.

### Limitation 2 — The useful shared state is not automatically self-maintaining

A single shared-state injection can help immediately, but the state can drift away over subsequent recurrent updates. Repeated placement or appropriate continued contextual drive provides much larger sequence-wide rescue.

This motivates making the **neutral/shared trajectory dynamically closed under the recurrent update**.

The constructive hypothesis therefore has two components:

1. **Familiar-contrast bottleneck**  
   Only evidence relevant to the learned A-vs-B distinction should modulate the sequence state.

2. **Neutral-trajectory closure**  
   The context-neutral/shared sequence trajectory should itself be executable by the recurrent dynamics.

Neither component may use Mixed, NoContext, or Novel behavioral outcomes during training, tuning, seed selection, or model selection.

---

# 2. Scientific question

> Are the two diagnosed representational failures sufficient to explain why the canonical High-scale SRN fails to generalize to Mixed, NoContext, and Novel contexts?

More specifically:

> If we restrict context to a familiar A/B contrast and train the resulting neutral state to form a valid recurrent sequence trajectory, can the network retain strong A/B separation while spontaneously generalizing to all three probe families?

---

# 3. Core representational hypothesis

Let the original raw context cue be \(c\).

From the two familiar training cues only, define:

\[
\bar c = \frac{c_A+c_B}{2}
\]

and

\[
d_c = \frac{c_A-c_B}{2}.
\]

Define the scalar familiar-context coordinate:

\[
\alpha(c)
=
\frac{(c-\bar c)^\top d_c}
{\|d_c\|^2+\epsilon}.
\]

By construction:

\[
\alpha(c_A)=+1
\]

and

\[
\alpha(c_B)=-1.
\]

If Mixed is the physical midpoint of A and B:

\[
c_M=\frac{c_A+c_B}{2},
\]

then:

\[
\alpha(c_M)=0.
\]

NoContext and Novel are **not manually assigned to zero**. Their \(\alpha\) values must be obtained by applying the same frozen A/B-derived projection to their original cue vectors.

For the current cue geometry, verify rather than assume whether:

\[
\alpha(c_{\rm NoContext})\approx0
\]

and

\[
\alpha(c_{\rm Novel_j})\approx0.
\]

The implementation must fail loudly if these values are not what the cue geometry implies.

---

# 4. Architectural change

## 4.1 Canonical SRN

The original model is conceptually:

\[
h_t=
\tanh(
W_{\rm seq}x_t
+
W_{\rm ctx}c_t
+
W_hh_{t-1}
+
b_h
).
\]

The full Phase 1L model replaces the unrestricted context pathway \(W_{\rm ctx}c_t\) with a rank-1 familiar-contrast pathway:

\[
h_t=
\tanh(
W_{\rm seq}x_t
+
u\,\alpha(c_t)
+
W_hh_{t-1}
+
b_h
),
\]

where:

- \(u\in\mathbb{R}^{8}\) is learned;
- \(\alpha(c_t)\) is fixed by the raw cue and the A/B-derived projection;
- no other raw-context direction directly enters the hidden state.

Equivalent implementation:

\[
W_{\rm ctx}^{\rm eff}
=
u
\frac{d_c^\top}{\|d_c\|^2+\epsilon},
\]

with cue centering by \(\bar c\) before multiplication.

Do not special-case cue labels.

---

## 4.2 What remains unchanged

Keep identical to the canonical experiment unless required by the contrast pathway:

- hidden size = 8;
- tanh recurrence;
- sequence token encoding;
- output layer;
- next-token objective;
- A/B sequence content;
- A/B training order;
- number of external training sequences = 170;
- optimizer;
- learning rate;
- batch/update convention;
- teacher forcing;
- High-scale initialization family;
- Low-scale comparison family if run;
- random seeds where possible;
- evaluation sequence;
- raw CE definition.

The purpose is to alter only the diagnosed representational organization.

---

# 5. High-scale initialization

The full model should remain a genuine High-scale model.

Initialize:

- recurrent weights using the canonical High-scale scheme;
- sequence-input weights using the canonical High-scale scheme;
- output weights using the canonical High-scale scheme;
- learned context vector \(u\) from the same High-scale distribution used for an ordinary context-input column.

Do not shrink \(u\) merely to make probes easier.

The contrast bottleneck is the inductive bias; low-magnitude context is not.

---

# 6. Neutral/shared trajectory

For an A/B training sequence, compute the natural familiar trajectories:

\[
h_t^A
\]

and

\[
h_t^B.
\]

Define the A/B-derived neutral target:

\[
s_t
=
\frac{h_t^A+h_t^B}{2}.
\]

Use:

\[
s_t^{*}
=
\operatorname{stopgrad}(s_t)
\]

as the closure target.

The stop-gradient is required so the model cannot reduce closure loss simply by moving the target trajectories together.

The closure target must be derived only from A and B.

---

# 7. Neutral-trajectory closure objective

The key constructive constraint is:

> Starting from the ordinary initial state and receiving neutral familiar-context evidence \(\alpha=0\), the recurrent model should be able to execute the A/B-derived shared sequence trajectory.

For the same sequence tokens, define a neutral rollout:

\[
q_0=h_0
\]

and recursively:

\[
q_t
=
f(q_{t-1},x_t,\alpha=0).
\]

Define:

\[
L_{\rm closure}
=
\frac{1}{T}
\sum_{t=1}^{T}
\frac{
\|q_t-s_t^{*}\|_2^2
}{
\|h_t^A-h_t^B\|_2^2+\epsilon
}.
\]

The A/B separation in the denominator makes the loss approximately scale-normalized.

Use all available hidden positions associated with the seven predicted transitions.

If implementation indexing includes an initial input state before the first scored transition, document the exact mapping and keep it consistent across all conditions.

---

# 8. Primary training objective

For the full model:

\[
L
=
L_{\rm CE}^{A/B}
+
\lambda_{\rm closure}L_{\rm closure}.
\]

Primary preregistered value:

\[
\lambda_{\rm closure}=1.
\]

Because \(L_{\rm closure}\) is normalized by familiar A/B separation, do not tune \(\lambda\) on probe performance.

Optional preregistered sensitivity values:

\[
\lambda\in\{0.3,3.0\}.
\]

Sensitivity runs may be inspected only after the primary \(\lambda=1\) results are frozen.

No value may be selected because it performs best on Mixed, NoContext, or Novel.

---

# 9. Four required model conditions

Use the same seeds and A/B training stream for every condition.

## C0 — Canonical High baseline

Original unconstrained context pathway.

Training:

\[
L=L_{\rm CE}^{A/B}.
\]

Purpose:

- reproduce the established phenotype.

---

## C1 — Familiar-contrast bottleneck only

Use:

\[
u\,\alpha(c)
\]

instead of unrestricted \(W_{\rm ctx}c\).

Training:

\[
L=L_{\rm CE}^{A/B}.
\]

Purpose:

- tests whether fixing cue routing alone is sufficient.

Prediction:

- probe states may begin closer to the shared route;
- maintenance may remain poor.

---

## C2 — Trajectory closure only

Keep the original unrestricted context pathway.

Add the A/B-derived neutral-trajectory closure objective.

For the neutral rollout used in \(L_{\rm closure}\), use the **true physical midpoint cue**:

\[
c=\bar c.
\]

Training:

\[
L=
L_{\rm CE}^{A/B}
+
L_{\rm closure}.
\]

Purpose:

- tests whether making the shared route dynamically viable is sufficient without changing the cue representation.

Prediction:

- a shared trajectory may become executable;
- Novel/NoContext may still fail to access it because their raw cue directions remain uncontrolled.

---

## C3 — Full constructive model

Use both:

- familiar-contrast bottleneck;
- neutral-trajectory closure.

Training:

\[
L=
L_{\rm CE}^{A/B}
+
L_{\rm closure}.
\]

Purpose:

- tests whether the two diagnosed failures jointly explain the missing phenotype.

This is the primary experimental condition.

---

# 10. Optional matched structural control

If cheap to run, include a rank-matched context-bottleneck control.

Replace the A/B contrast direction \(d_c\) with a fixed random unit direction \(r\) of the same dimensionality and use the same single learned vector \(u\).

The random direction must be fixed before training and must not use any probe behavior.

Purpose:

- distinguishes benefit of lower-rank context processing from benefit of the **specific familiar A/B contrast**.

This control is secondary and must not delay the primary experiment.

---

# 11. Probe firewall

The following may **never** influence:

- gradients;
- hyperparameters;
- early stopping;
- seed selection;
- model selection;
- learning rate;
- \(\lambda_{\rm closure}\);
- architecture choice after results are viewed.

Protected probes:

- Mixed;
- NoContext;
- Novel1;
- Novel2;
- Novel3;
- Novel4.

Only A/B data may be used during training and training-time selection.

---

# 12. Evaluation conditions

After training is complete and checkpoints are frozen, evaluate:

- A;
- B;
- Mixed;
- NoContext;
- Novel1;
- Novel2;
- Novel3;
- Novel4;
- NovelMean within seed.

Use the original raw cue vectors as inputs.

For bottleneck models, raw cues are passed through the frozen A/B-derived \(\alpha(c)\) transform.

Do not manually replace a probe cue with zero at evaluation time.

---

# 13. Mandatory cue-geometry audit

Before any behavioral evaluation, export:

\[
\alpha(A),\alpha(B),\alpha(Mixed),\alpha(NoContext),
\alpha(Novel1),\ldots,\alpha(Novel4).
\]

Required checks:

\[
\alpha(A)=+1
\]

\[
\alpha(B)=-1
\]

to numerical tolerance.

For all other cues, report the actual values.

If Mixed, NoContext, or Novel project near zero, that must arise from their original cue geometry and the A/B-derived projection—not from label-specific code.

Create:

`phase1l_cue_projection_audit.csv`

with:

```text
cue_name
raw_cue_hash
alpha
distance_from_cbar
projection_on_dc
orthogonal_norm
```

---

# 14. Primary phenotype

The desired phenotype is:

\[
\boxed{
\text{strong A/B separation}
+
\text{good A/B performance}
+
\text{generalization to Mixed, NoContext, and Novel}
}
\]

The full model is not successful merely because probe loss decreases.

It must satisfy all three components.

---

# 15. Primary behavioral readout

For every final frozen checkpoint retain all seven raw transition CEs:

\[
CE_1,\ldots,CE_7.
\]

Sequence loss:

\[
L_{\rm seq}
=
\frac{1}{7}
\sum_{k=1}^{7}CE_k.
\]

Primary comparisons are between C3 and C0 for:

- Mixed;
- NoContext;
- NovelMean.

Also report Novel1–4 separately.

Do not standardize before computing these effects.

---

# 16. Familiar-performance preservation

For each seed calculate:

\[
\Delta L_A
=
L_A^{C3}-L_A^{C0}
\]

\[
\Delta L_B
=
L_B^{C3}-L_B^{C0}.
\]

Report raw differences and relative changes.

The full model should not obtain probe generalization by simply degrading A/B until everything becomes equally poor.

---

# 17. A/B representational separation

Retain both hidden and task-space separation.

For transition \(t\):

\[
D^h_{AB,t}
=
\|h^A_t-h^B_t\|
\]

and:

\[
D^z_{AB,t}
=
\|z^A_t-z^B_t\|.
\]

Report:

- transition-level values;
- sequence mean;
- per-seed values.

Compare C3 against C0.

Do not define success as exact equality to the baseline. The question is whether **substantial context-specific separation remains**.

---

# 18. Shared-trajectory geometry

For every model and seed define:

\[
m^h_t
=
\frac{h^A_t+h^B_t}{2}
\]

and centered-logit midpoint:

\[
m^z_t
=
\frac{z^A_t+z^B_t}{2}.
\]

For every probe \(p\):

\[
R^h_{p,t}
=
\frac{
\|h^p_t-m^h_t\|
}{
\|h^A_t-h^B_t\|+\epsilon
}
\]

and:

\[
R^z_{p,t}
=
\frac{
\|z^p_t-m^z_t\|
}{
\|z^A_t-z^B_t\|+\epsilon
}.
\]

These are primary mechanistic readouts.

The intended result is that C3 moves:

- Mixed;
- NoContext;
- Novel

toward the A/B-derived neutral trajectory while preserving A/B separation.

---

# 19. Endpoint coordinate and off-axis residual

In task space:

\[
\alpha^{task}_{p,t}
=
\frac{
(z^p_t-z^A_t)^\top(z^B_t-z^A_t)
}{
\|z^B_t-z^A_t\|^2+\epsilon
}.
\]

Off-axis residual:

\[
r^\perp_{p,t}
=
\frac{
\left\|
z^p_t-
[z^A_t+\alpha^{task}_{p,t}(z^B_t-z^A_t)]
\right\|
}{
\|z^A_t-z^B_t\|+\epsilon
}.
\]

This prevents a false success in which probe loss improves via a completely different task-space state.

---

# 20. Neutral-rollout closure diagnostic

After training, independently recompute the neutral rollout:

\[
q_t=f(q_{t-1},x_t,\alpha=0)
\]

and compare it with:

\[
m_t^h=\frac{h_t^A+h_t^B}{2}.
\]

Export:

- transition-level neutral closure residual;
- cumulative drift;
- raw neutral rollout CE;
- centered-logit residual;
- endpoint coordinate.

This evaluation must be performed from scratch rather than reusing training loss buffers.

---

# 21. No hidden-state interventions in the final phenotype test

Phase 1L evaluation must be completely natural.

Do not:

- patch hidden states;
- patch logits;
- clamp output states;
- substitute target trajectories;
- use learned adapters after training.

The point is to determine whether the altered learning architecture makes the desired behavior emerge autonomously.

Previous patching results are references only.

---

# 22. Predictions for the ablations

Freeze these qualitative predictions before running.

| Condition | Routing | Shared trajectory | Expected phenotype |
|---|---|---|---|
| C0 Baseline | unconstrained | unstable | strong A/B split; poor probe transfer |
| C1 Contrast only | improved | not explicitly stabilized | better initial probe geometry; incomplete sustained transfer |
| C2 Closure only | unconstrained | stabilized | neutral route exists; probe access remains inconsistent |
| C3 Full | improved | stabilized | strong A/B split + transfer across Mixed/NoContext/Novel |

The exact numerical effect sizes are not preregistered.

---

# 23. Primary confirmatory comparisons

Use paired seeds wherever initialization correspondence makes this possible.

## P1 — Full model improves all three probe families

C3 vs C0 raw sequence CE for:

- Mixed;
- NoContext;
- NovelMean.

Success requires the direction of improvement for **all three**, not just Mixed.

---

## P2 — Full model preserves familiar execution

C3 must retain strong A/B performance.

Report:

- A raw CE;
- B raw CE;
- pooled familiar raw CE.

---

## P3 — Full model preserves contextual separation

C3 must retain meaningful A/B hidden and task separation.

---

## P4 — Full model improves shared-state geometry

Relative to C0, C3 should reduce:

- Mixed midpoint residual;
- NoContext midpoint residual;
- NovelMean midpoint residual;

especially in centered-logit/task space.

---

## P5 — Joint intervention beats the component ablations

Compare C3 against C1 and C2.

The strongest support for the proposed mechanism is:

\[
C3 > C1
\]

and

\[
C3 > C2
\]

on the combined three-probe phenotype while preserving A/B separation.

Do not require every individual transition to show this ordering.

---

# 24. Composite success index

Do **not** use a hidden weighted score as the main result.

For compact descriptive reporting, define a transparent three-probe generalization vector:

\[
G=
(
\Delta L_{\rm Mixed},
\Delta L_{\rm NoContext},
\Delta L_{\rm NovelMean}
).
\]

All three components should be reported separately.

If a scalar is useful for figures, use the unweighted mean improvement only as secondary:

\[
\bar G
=
\frac{
G_{\rm Mixed}+G_{\rm NoContext}+G_{\rm NovelMean}
}{3}.
\]

Never let this scalar hide a failure on one probe family.

---

# 25. Statistics

Seed is the unit of inference.

Use the canonical 20-seed set per family/condition if computationally feasible.

For Novel:

1. evaluate Novel1–4 separately;
2. average within seed;
3. create NovelMean;
4. only then bootstrap seeds.

Use paired seed bootstrap C3–C0, C3–C1, and C3–C2.

Report:

- mean paired difference;
- median paired difference;
- 95% bootstrap CI;
- seed-level scatter.

Do not bootstrap transitions.

---

# 26. Mandatory raw-loss figures

## FIG01 — Canonical acquisition audit

For C0–C3:

- x = actual external sequence index 1–170;
- y = raw pre-update sequence CE;
- separate A/B;
- actual changing parameter hashes retained.

This ensures the intervention did not silently alter the training protocol.

---

## FIG02 — Final familiar A/B transition profiles

For C0–C3:

- raw CE for transitions 1–7;
- separate A and B;
- High-scale primary.

Purpose:

- show familiar execution is preserved.

---

## FIG03 — All three probe families: raw transition CE

For C0–C3, plot full seven-transition profiles for:

- Mixed;
- NoContext;
- NovelMean.

Novel1–4 should be available in supplement.

This is a primary figure.

---

## FIG04 — Probe sequence-level loss by seed

For every seed and condition:

- Mixed;
- NoContext;
- NovelMean.

Show paired seed points/lines.

No bar-only visualization.

---

## FIG05 — A/B representational separation

For C0–C3:

- hidden A/B separation by transition;
- task-space A/B separation by transition.

---

## FIG06 — Shared-trajectory residual

For Mixed, NoContext, NovelMean:

- hidden midpoint residual;
- task midpoint residual;
- by transition;
- C0–C3.

This should directly test the representational prediction.

---

## FIG07 — Endpoint coordinate / off-axis geometry

For Mixed, NoContext, NovelMean under C0–C3:

- \(\alpha^{task}\);
- off-axis residual;
- transition-level profiles.

---

## FIG08 — Neutral trajectory closure

For C0–C3:

- neutral rollout hidden residual from A/B midpoint;
- neutral rollout task residual;
- neutral rollout raw CE.

This is the mechanistic manipulation check.

---

## FIG09 — Cue projection audit

Plot or tabulate:

- A;
- B;
- Mixed;
- NoContext;
- Novel1–4

against their frozen \(\alpha(c)\) coordinate.

The figure must make clear that probe mappings arise from the A/B-derived projection rather than label-specific assignment.

---

## FIG10 — Ablation summary

A matrix with rows = seeds and columns including:

- familiar CE preserved;
- A/B hidden separation;
- A/B task separation;
- Mixed improvement;
- NoContext improvement;
- NovelMean improvement;
- Mixed task-midpoint residual;
- NoContext task-midpoint residual;
- NovelMean task-midpoint residual;
- neutral closure residual.

Use continuous values or signed differences. Avoid post-hoc red/green thresholds.

---

# 27. Required output files

```text
phase1l/
├── phase1l_provenance.csv
├── phase1l_cue_projection_audit.csv
├── phase1l_training_sequence_metrics.csv
├── phase1l_final_transition_metrics.csv
├── phase1l_neutral_rollout_metrics.csv
├── phase1l_geometry_metrics.csv
├── phase1l_seed_summary.csv
├── phase1l_novel_within_seed.csv
├── phase1l_bootstrap_summary.csv
├── PHASE1L_CONSTRUCTIVE_RECOVERY_REPORT.md
└── figures/
```

---

# 28. Minimum provenance columns

`phase1l_provenance.csv`

```text
condition
family
seed
checkpoint_path
checkpoint_hash
architecture_hash
training_data_hash
sequence_hash
cue_hash
context_projection_hash
lambda_closure
optimizer
learning_rate
hidden_dim
init_scale
final_parameter_hash
```

---

# 29. Minimum transition-metric columns

`phase1l_final_transition_metrics.csv`

```text
condition
family
seed
probe
novel_id
transition
raw_ce
correct_token_probability
hidden_midpoint_residual
task_midpoint_residual
alpha_task
off_axis_task
AB_hidden_separation
AB_task_separation
cue_alpha
parameter_hash
```

---

# 30. Minimum seed-summary columns

`phase1l_seed_summary.csv`

```text
condition
family
seed
A_seq_ce
B_seq_ce
familiar_mean_ce
AB_hidden_separation
AB_task_separation
Mixed_seq_ce
NoContext_seq_ce
NovelMean_seq_ce
Mixed_task_midpoint_residual
NoContext_task_midpoint_residual
NovelMean_task_midpoint_residual
neutral_hidden_closure_residual
neutral_task_closure_residual
neutral_seq_ce
```

---

# 31. TDD / RED → GREEN requirements

Implement tests before scientific runs.

## Test 1 — Canonical training stream identity

Assert C0–C3 receive the exact same:

- A/B sequence order;
- tokens;
- training schedule;
- number of external sequences.

---

## Test 2 — A/B projection identity

Numerically assert:

\[
\alpha(A)=+1
\]

and

\[
\alpha(B)=-1.
\]

---

## Test 3 — No probe special-casing

Search implementation for conditional branches based on:

- `"Mixed"`
- `"NoContext"`
- `"Novel"`

inside the context-projection function.

Fail if any probe label changes its \(\alpha\) value.

The projection function must depend only on cue vectors plus frozen A/B references.

---

## Test 4 — Mixed geometry

If the raw Mixed cue is exactly:

\[
(c_A+c_B)/2,
\]

assert:

\[
\alpha(Mixed)=0
\]

to numerical tolerance.

---

## Test 5 — NoContext/Novel audit

Do not assert zero in advance.

Instead:

- compute values from raw cue vectors;
- save them;
- assert evaluation uses those computed values unchanged.

---

## Test 6 — Rank-1 context pathway

For C1 and C3, verify every context-induced hidden preactivation lies in the one-dimensional span of \(u\).

---

## Test 7 — High-scale initialization

Verify \(u\) uses the canonical High-scale initialization distribution and has not been zero-initialized unless running a specifically named control.

---

## Test 8 — Stop-gradient targets

Assert no gradient flows from \(L_{\rm closure}\) through the stored midpoint targets \(s_t^*\).

---

## Test 9 — Probe firewall

Training dataloader and loss must never contain:

- Mixed;
- NoContext;
- Novel1–4.

---

## Test 10 — Hyperparameter firewall

Primary \(\lambda=1\) must be fixed before probe evaluation.

No code path may choose \(\lambda\) using probe loss.

---

## Test 11 — Neutral rollout recursion

Verify:

\[
q_t=f(q_{t-1},x_t,\alpha=0)
\]

uses the previous predicted neutral state, not the target \(s_{t-1}\), after initialization.

This ensures the loss trains autonomous recurrent closure rather than only one-step teacher-forced matching.

---

## Test 12 — Closure target is A/B-only

Assert midpoint targets depend only on paired A/B trajectories.

---

## Test 13 — A/B separation denominator

Verify normalized closure residual uses detached familiar A/B separation plus \(\epsilon\).

---

## Test 14 — No hidden patch during evaluation

Fail if evaluation writes or replaces hidden states after recurrence begins.

---

## Test 15 — Raw CE preservation

Every final condition × seed × probe must have exactly seven raw transition CE values.

---

## Test 16 — Novel nesting

Novel1–4 must be averaged within seed before family inference.

---

## Test 17 — No probability averaging

Fail if any target or shared representation is built by averaging softmax probabilities.

---

## Test 18 — Centered-logit implementation

Assert centered logits have zero class mean to tolerance.

---

## Test 19 — Acquisition is pre-update

Training time-series CE must be logged before the corresponding optimizer update.

---

## Test 20 — Parameter hashes change during training

The acquisition trace cannot be a frozen final-checkpoint replay.

---

## Test 21 — Final checkpoint identity

Each exported evaluation row must include the same final parameter hash for its seed/condition.

---

## Test 22 — Figure completeness

Fail report generation if missing:

- raw acquisition;
- familiar raw transition profiles;
- all-three-probe raw transition profiles;
- seed-level probe loss;
- A/B separation;
- midpoint residuals;
- endpoint/off-axis geometry;
- neutral closure;
- cue projection audit;
- ablation summary.

---

# 32. Implementation order

## Step 1 — Lock existing canonical configuration

Create immutable references to:

- architecture;
- optimizer;
- training schedule;
- cue vectors;
- sequence;
- seeds.

---

## Step 2 — Implement cue projection

Build:

```python
fit_familiar_contrast(c_A, c_B)
project_context(c)
```

The fitting function accepts **only** A and B cue vectors.

---

## Step 3 — Unit-test projection before model training

Verify all cue coordinates and save the audit table.

Do not modify them after seeing behavior.

---

## Step 4 — Implement contrast-bottleneck SRN

Replace unrestricted context multiplication with:

```text
u * alpha(c)
```

for C1/C3.

---

## Step 5 — Implement paired A/B trajectory extraction

For each A/B training item, produce aligned hidden trajectories.

---

## Step 6 — Implement detached midpoint targets

Compute:

```text
s_t = stopgrad((h_A_t + h_B_t) / 2)
```

---

## Step 7 — Implement recursive neutral rollout

Start from the ordinary initial state.

Run the sequence under neutral familiar evidence.

---

## Step 8 — Implement normalized closure loss

Add only for C2/C3.

---

## Step 9 — Train C0–C3 on A/B only

Use matched seeds.

Do not inspect probes during training.

---

## Step 10 — Freeze checkpoints

Hash everything.

---

## Step 11 — Evaluate A/B first

Confirm training did not catastrophically fail.

This may be inspected before probes because A/B are training/familiar conditions.

---

## Step 12 — Evaluate all protected probes once

Run:

- Mixed;
- NoContext;
- Novel1–4.

Treat this as the confirmatory reveal.

---

## Step 13 — Compute seed bootstrap and figures

No model changes after Step 12.

---

# 33. Decision table

Freeze before probe evaluation.

## Outcome A — Full constructive recovery

C3:

- preserves good A/B performance;
- preserves substantial A/B separation;
- improves Mixed;
- improves NoContext;
- improves NovelMean;
- moves all three toward the shared trajectory;
- outperforms C1/C2 in the predicted joint fashion.

Interpretation:

> The prior mechanistic diagnosis was constructively sufficient: familiar context should act as modulation around a shared sequence backbone, and that shared backbone must form an autonomous recurrent trajectory.

This is the strongest stopping outcome.

---

## Outcome B — Behavioral recovery with correct geometry but reduced A/B separation

All probes improve and approach the shared trajectory, but A/B separation collapses substantially.

Interpretation:

> The intervention recovers generalization by reverting toward a lumped solution rather than solving the separation/generalization tradeoff.

Do not count as full success.

---

## Outcome C — Contrast bottleneck sufficient

C1 already recovers most of C3 behavior.

Interpretation:

> The dominant failure was context routing; explicit recurrent closure adds little.

This is still a successful mechanistic refinement.

---

## Outcome D — Closure sufficient

C2 already recovers most of C3 behavior.

Interpretation:

> The dominant failure was recurrent viability of the shared trajectory; cue geometry was less limiting than expected.

Still informative.

---

## Outcome E — Mixed/NoContext recover, Novel remains poor

C3 improves Mixed and NoContext but not Novel.

Interpretation:

> Familiar-contrast neutrality is sufficient for ambiguity and absence, but unfamiliar context produces additional dynamics not captured by the A/B contrast bottleneck.

Do not invent a Novel-specific fix in this project.

Stop and report the boundary.

---

## Outcome F — Shared geometry improves without behavioral recovery

Probe states move toward the midpoint/shared trajectory but raw CE remains poor.

Interpretation:

> The identified state geometry is not sufficient when induced through learning, despite acute patch sufficiency.

Audit implementation first; if correct, report the mismatch.

---

## Outcome G — No constructive recovery

C3 does not improve the three-probe phenotype.

Interpretation:

> The causal patch results identify a sufficient internal state, but the proposed representation-learning principles are not sufficient to make the network autonomously realize it.

Stop and report the boundary.

---

# 34. Stop rule

After the confirmatory C0–C3 experiment:

- no new architecture search;
- no new context encodings;
- no Novel-specific adapter;
- no probe-supervised fine-tuning;
- no lambda search based on probes;
- no hidden-state repair;
- no larger SRN;
- no additional model families unless needed for a paper-requested robustness check.

If Outcome A–G is obtained with a clean provenance audit, the computational program is complete.

---

# 35. Manuscript interpretation if successful

A successful full intervention would support:

> Context-specific representations need not require separate sequence memories. The network can preserve a shared autonomous sequence computation while familiar contexts modulate that computation in opposite directions. When contextual evidence is ambiguous, absent, or orthogonal to the learned familiar distinction, the system falls back onto the shared trajectory. This organization preserves contextual separation while enabling generalization.

Do **not** claim:

- that humans literally encode context as a one-dimensional signed axis;
- that the intervention proves the biological mechanism;
- that all unfamiliar contexts should be ignored.

The experiment establishes **constructive sufficiency** of the representational principle in the SRN.

---

# 36. Manuscript interpretation if unsuccessful

A clean failure is also theoretically useful:

> Direct causal intervention identified a shared state capable of supporting generalization, but imposing a familiar-context bottleneck and recurrent closure was not sufficient to make that state emerge naturally through learning. Thus state-level sufficiency does not imply learnability under this representational constraint.

That is a legitimate final boundary and should end the modeling program.

---

# 37. Definition of done

Phase 1L is complete when:

1. the A/B-derived cue projection is frozen before probe evaluation;
2. C0–C3 are trained on the same A/B experience;
3. no probe condition enters training or tuning;
4. all 20 matched High seeds are completed if computationally feasible;
5. raw A/B and probe transition CE are retained;
6. A/B representational separation is quantified;
7. neutral trajectory closure is independently audited;
8. Mixed, NoContext, and NovelMean are evaluated together;
9. C1/C2 ablations are compared with C3;
10. the frozen decision-table outcome is assigned;
11. no further model changes are made after confirmatory probe evaluation.

At that point, write the paper.
