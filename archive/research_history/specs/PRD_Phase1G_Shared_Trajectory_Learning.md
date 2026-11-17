# PRD — Phase 1G: Learn a Stable Shared Sequence Trajectory Without Collapsing A/B Context Separation

## 1. Purpose

Phase 1F showed that the High-scale SRNs contain a behaviorally useful shared A/B sequence state, but that state is not naturally sustained by the recurrent dynamics.

The key empirical pattern motivating this phase is:

- High-scale networks have strong A/B contextual separation but poor transfer.
- Replacing a probe state with the correct A/B shared state substantially rescues raw sequence loss.
- Correct sequence-position shared-state patches outperform wrong-position patches.
- Norm-matched orthogonal movements do not reproduce the rescue.
- A one-step / transition-wise shared-state patch helps much more than a single-entry patch followed by free recurrent rollout.
- Mixed is already near the A/B midpoint along the familiar contextual axis but still has off-axis distortion.
- Novel and NoContext are much farther from the familiar shared trajectory.
- In the current canonical implementation, Novel and NoContext are not independent inputs: the Novel input column is effectively zero, making the two probes identical. That must be fixed before any new Novel-transfer claim is made.

The scientific question is now:

> Can an SRN preserve strong A/B context-specific representations while also learning a stable, reusable sequence trajectory that remains available when context is mixed, absent, or genuinely novel?

This phase should test a small set of targeted training interventions rather than a broad hyperparameter search.

---

# 2. Core representational hypothesis

The desired organization is:

\[
h_{A,k} = s_k + d_k
\]

\[
h_{B,k} = s_k - d_k
\]

where:

- \(s_k\) is a sequence-position-specific shared backbone,
- \(d_k\) is context-specific modulation.

The critical requirement is not just that \(s_k\) exists geometrically, but that it is itself predictive and dynamically stable.

The desired shared trajectory is:

\[
s_1 \rightarrow s_2 \rightarrow \dots \rightarrow s_7.
\]

Human-like transfer is expected when:

1. A and B remain distinct,
2. \(s_k\) remains predictive of the sequence,
3. the recurrent system can move from \(s_k\) to \(s_{k+1}\),
4. mixed, absent, and novel context do not destroy access to that shared sequence trajectory.

---

# 3. Canonical model

Use the audited canonical SRN only.

## High condition

```text
architecture: tanh Elman SRN
hidden_size: 8
initialization_scale: 2.0
```

## Low reference

```text
architecture: tanh Elman SRN
hidden_size: 8
initialization_scale: 0.02
```

Training schedule remains the canonical 170-sequence A/B schedule.

Do not change:

- hidden size,
- recurrence type,
- activation,
- sequence,
- optimizer family,
- A/B training schedule,
- output head,
- task definition.

The goal is to alter only the **training pressure on the shared sequence component**.

---

# 4. Conditions

Train the same number of seeds per condition.

Recommended:

```text
20 seeds per condition
```

Use identical seed lists when possible.

## G0 — Baseline High

Canonical High-scale SRN.

No additional intervention.

Purpose:

- reference for contextual splitting,
- reference for poor transfer.

---

## G1 — Context dropout only

On a small proportion of ordinary A/B training sequences, remove the context cue while keeping the sequence input unchanged.

For context vector \(c_t\):

\[
c'_t =
\begin{cases}
c_t & \text{with probability } 1-p \\
0 & \text{with probability } p.
\end{cases}
\]

Important:

- the target sequence remains unchanged,
- A/B labels are not changed,
- no Mixed, Novel, or NoContext probe is used for training,
- context dropout is an auxiliary pressure to keep sequence information usable independently of context.

Use a very small fixed set of dropout probabilities:

```text
p ∈ {0.025, 0.05, 0.10}
```

Select one globally using only A/B training diagnostics and the shared-trajectory objective.

Do not select \(p\) using probe performance.

---

## G2 — Shared-state prediction only

At each sequence position \(k\), compute the context-average hidden state:

\[
s_k^h = \frac{h_{A,k}+h_{B,k}}{2}.
\]

Feed \(s_k^h\) through the existing frozen/shared output head and require it to predict the correct next item.

Auxiliary loss:

\[
L_{\text{shared-pred}}
=
\frac{1}{K}
\sum_k
\operatorname{CE}
\left(
W_{out}s_k^h+b_{out},
y_{k+1}
\right).
\]

Total loss:

\[
L
=
L_{\text{sequence}}
+
\lambda_{\text{pred}}L_{\text{shared-pred}}.
\]

Important:

- do **not** penalize \(h_A-h_B\),
- do **not** force A and B to become similar,
- do **not** directly train on Mixed / Novel / NoContext,
- do **not** regularize toward the Low network.

This intervention asks only:

> Does the A/B-shared component itself retain enough sequence information to predict correctly?

---

## G3 — Shared-trajectory stability only

Define the current shared state:

\[
s_k^h
=
\frac{h_{A,k}+h_{B,k}}{2}.
\]

Starting from \(s_k^h\), apply the canonical recurrent transition under the shared sequence input but **without a context-specific push**.

Let:

\[
\hat{s}_{k+1}
=
f_{\theta}(x_{k+1}, c_{\mathrm{neutral}}, s_k^h).
\]

Primary neutral-context choice:

\[
c_{\mathrm{neutral}} = 0.
\]

Target:

\[
s_{k+1}^h
=
\frac{h_{A,k+1}+h_{B,k+1}}{2}.
\]

Auxiliary trajectory loss:

\[
L_{\text{traj}}
=
\frac{1}{K-1}
\sum_k
\left\|
\hat{s}_{k+1}
-
\operatorname{stopgrad}(s_{k+1}^h)
\right\|^2.
\]

Use `stopgrad` on the target shared state by default so the auxiliary objective trains the transition toward the currently learned shared trajectory instead of trivially moving both sides.

Total loss:

\[
L
=
L_{\text{sequence}}
+
\lambda_{\text{traj}}L_{\text{traj}}.
\]

This asks:

> If the network is placed on the shared trajectory, will its own recurrent dynamics carry it to the next shared sequence state?

---

## G4 — Shared prediction + shared trajectory

Combine G2 and G3:

\[
L
=
L_{\text{sequence}}
+
\lambda_{\text{pred}}L_{\text{shared-pred}}
+
\lambda_{\text{traj}}L_{\text{traj}}.
\]

This is the primary mechanistic intervention.

Expected interpretation if successful:

> The shared sequence component is not only informative at each instant but forms a self-consistent recurrent trajectory.

---

## G5 — Context dropout + shared trajectory

Optional but recommended if computationally affordable.

\[
L
=
L_{\text{sequence}}
+
\lambda_{\text{traj}}L_{\text{traj}}
\]

plus context dropout with selected \(p\).

Purpose:

- test whether a simple weak pressure to use sequence information complements trajectory stabilization,
- determine whether explicit shared-state prediction is necessary.

Do not add more conditions unless one of these fails for an implementation reason.

---

# 5. Hyperparameter policy

Keep the search small and globally fixed.

Suggested values:

```text
lambda_pred ∈ {0.01, 0.05, 0.10, 0.25}
lambda_traj ∈ {0.01, 0.05, 0.10, 0.25}
p_dropout ∈ {0.025, 0.05, 0.10}
```

Use a small pilot subset of seeds if needed.

Hyperparameters may be selected using only:

- A/B raw sequence loss,
- A/B separation,
- shared-state prediction loss,
- shared-trajectory loss,
- training stability.

Do **not** use:

- Mixed loss,
- Novel loss,
- NoContext loss,
- human behavior,
- probe success rate.

Those remain held out.

---

# 6. A/B separation guardrail

The intervention is only interesting if A and B remain meaningfully split.

Track:

\[
D_{AB}
=
\frac{1}{K}
\sum_k
\|h_{A,k}-h_{B,k}\|.
\]

Also track task-space A/B separation:

\[
D^{task}_{AB}
=
\frac{1}{K}
\sum_k
\|z_{A,k}-z_{B,k}\|.
\]

Primary guardrail:

```text
A/B separation ratio relative to Baseline High >= 0.75
```

Use the exact prior project threshold if one is already frozen and comparable.

Also require familiar A/B loss to remain within the existing Phase 1C/1D familiar-loss guardrail.

Do not declare success if transfer improves only because A/B collapse into the Low-network solution.

---

# 7. Shared-backbone metrics

For every trained model compute:

\[
s_k = \frac{z_{A,k}+z_{B,k}}{2}
\]

\[
d_k = \frac{z_{A,k}-z_{B,k}}{2}.
\]

For every probe:

\[
z_{P,k}
=
s_k
+
\alpha_{P,k}d_k
+
r^\perp_{P,k}.
\]

Track:

```text
alpha
absolute off-axis norm
normalized off-axis norm
distance to shared state
sequence-specificity margin
```

Do this in:

- centered task/logit space,
- hidden-state space.

Task/logit space remains primary.

---

# 8. Shared-trajectory diagnostics

The main new metric is whether the shared state is dynamically self-consistent.

For each \(k\):

\[
\hat{s}_{k+1}
=
f_{\theta}(x_{k+1},0,s_k).
\]

Compute:

\[
E_{\text{traj},k}
=
\|\hat{s}_{k+1}-s_{k+1}\|.
\]

Aggregate:

\[
E_{\text{traj}}
=
\frac1{K-1}\sum_kE_{\text{traj},k}.
\]

Also compute cosine similarity and task-space deviation after the rollout step.

---

# 9. Free-rollout shared-trajectory test

This is mandatory.

Start at the shared state for the first analyzed sequence position:

\[
h_1=s_1.
\]

Then allow the network to run recurrently using the shared sequence inputs with no familiar A/B context contribution.

Record:

\[
\hat{h}_2,\hat{h}_3,\dots,\hat{h}_7.
\]

Compare each state to:

\[
s_2,s_3,\dots,s_7.
\]

Metrics:

```text
shared_rollout_state_error
shared_rollout_task_error
shared_rollout_raw_loss
shared_rollout_correct_token_probability
```

This directly tests the failure identified in Phase 1F: local shared-state sufficiency without sustained shared dynamics.

---

# 10. Novel probe must become genuinely novel

The current canonical Novel and NoContext probes are effectively identical.

That must not continue in Phase 1G.

## 10.1 Preserve NoContext

NoContext remains:

```text
context = [0, 0, 0]
```

or the exact canonical zero-context representation.

---

## 10.2 Define genuine Novel cues

Novel must be:

- unseen during training,
- nonzero,
- distinct from A,
- distinct from B,
- distinct from zero.

Do **not** use a single arbitrary vector.

Create a small preregistered set of norm-matched novel cues.

Preferred construction:

1. take the trained A/B context vector norms,
2. generate deterministic seed-independent novel vectors with matched norm,
3. ensure they are not collinear with A or B,
4. use the same novel cue set across all model conditions.

Suggested:

```text
4 novel cue vectors
```

If the context input is only 2D, use angularly distinct directions not equal to A or B.

If the context input is 3D or higher, use deterministic orthogonal / off-axis vectors.

Save the exact vectors in the manifest.

Report both:

```text
mean Novel performance across novel cue vectors
per-novel-cue performance
```

Do not tune the vectors using model performance.

---

# 11. Held-out probes

After training and checkpoint selection, freeze the network.

Evaluate:

```text
A
B
Mixed
NoContext
Novel cue 1
Novel cue 2
Novel cue 3
Novel cue 4
```

No probe enters training or checkpoint selection.

---

# 12. Human-like transfer criterion

Do not use only “loss improved.”

The desired qualitative pattern is:

1. A/B remain separated,
2. familiar A/B raw loss remains good,
3. Mixed raw loss moves toward the A/B familiar range,
4. NoContext raw loss moves toward the A/B familiar range,
5. mean Novel raw loss moves toward the A/B familiar range.

Define:

\[
L_{\min AB}
=
\min(L_A,L_B)
\]

\[
L_{\max AB}
=
\max(L_A,L_B).
\]

For each probe classify:

```text
below A/B range
within A/B range
above A/B range
```

This is descriptive.

Primary continuous metric:

\[
\Delta L_P
=
L_P^{BaselineHigh}
-
L_P^{Condition}.
\]

---

# 13. Required raw-loss sequence plots

These are mandatory.

## TS1 — True A/B acquisition

For every training condition:

x-axis:

```text
actual training sequence index
```

y-axis:

```text
raw sequence loss
```

Show A and B across all 170 training sequences.

No frozen-checkpoint artifact.

---

## TS2 — A/B + held-out probes at final checkpoint

For each condition, show:

```text
A
B
Mixed
NoContext
Novel mean
```

on the same raw-loss axis.

Also show per-novel-cue points.

---

## TS3 — Shared free-rollout raw loss

Start on the shared state and roll out without A/B context.

Plot raw sequence loss across the seven transitions for:

```text
Baseline High
Context dropout
Shared prediction
Shared trajectory
Shared prediction + shared trajectory
```

This is a central mechanistic figure.

---

## TS4 — Probe loss by transition

For each probe:

```text
Mixed
NoContext
Novel
```

plot raw transition-level cross entropy across transitions 1–7.

Overlay:

```text
A reference
B reference
```

---

## TS5 — A/B separation over training

Show hidden and task-space separation across the training schedule.

This catches solutions that improve transfer only by collapsing contexts.

---

# 14. Required causal post-training checks

For the best globally selected intervention, repeat the Phase 1F causal tests.

Specifically:

- shared-state patch,
- A/B-axis projection,
- correct vs wrong sequence-position patch,
- norm-matched orthogonal control,
- transition-wise hidden patch,
- single-entry free rollout.

Reason:

> If training succeeds, the amount of rescue required from the external shared-state patch should shrink because the network should already be expressing the shared trajectory more naturally.

Track:

\[
\text{residual rescue need}
=
L_{\text{baseline probe}}
-
L_{\text{shared-patched probe}}.
\]

A successful learned solution should improve the unpatched probe and reduce the additional benefit of patching.

---

# 15. Primary comparisons

The most important comparison is:

```text
G0 Baseline High
G1 Context dropout
G2 Shared prediction
G3 Shared trajectory
G4 Shared prediction + shared trajectory
```

Optional:

```text
G5 Context dropout + shared trajectory
```

Report for each:

```text
A raw loss
B raw loss
A/B separation
Mixed raw loss
NoContext raw loss
Novel mean raw loss
Novel cue dispersion
shared trajectory error
shared free-rollout loss
```

---

# 16. Decision logic

## Outcome A — G4 succeeds while preserving A/B split

Interpretation:

> Human-like transfer can emerge when the network preserves a predictive shared sequence backbone and recurrent dynamics that sustain it, while context continues to modulate familiar A/B states.

This is the strongest support for the proposed mechanism.

---

## Outcome B — G3 succeeds without G2

Interpretation:

> The critical missing ingredient is dynamical stability of the shared trajectory, not whether the midpoint is directly predictive under the output head.

---

## Outcome C — G2 succeeds without G3

Interpretation:

> Shared sequence information matters, but recurrent stability is not the main transfer bottleneck.

---

## Outcome D — Context dropout succeeds as well as explicit shared losses

Interpretation:

> A weak pressure to maintain context-independent sequence information is sufficient; explicit factorization is unnecessary.

This would favor the simpler mechanism.

---

## Outcome E — Transfer improves only when A/B split collapses

Interpretation:

> The current architecture still exhibits a tradeoff between contextual differentiation and shared transfer.

Do not call this human-like success.

---

## Outcome F — Mixed improves but Novel/NoContext remain poor

Interpretation:

> Preserving a shared trajectory alone is insufficient for unfamiliar/absent-context routing.

Then the next target should be cue-to-shared-trajectory gating.

---

# 17. TDD / RED → GREEN

Implement tests before full runs.

## RED 1 — Canonical architecture

Attempt:

```text
hidden_size = 6
```

Expected:

```text
FAIL: NONCANONICAL_SRN
```

---

## RED 2 — Probe leakage

Attempt to include Mixed, Novel, or NoContext in training tensors.

Expected:

```text
FAIL: PROBE_USED_FOR_TRAINING
```

---

## RED 3 — Shared prediction identity

Verify that:

```python
shared = 0.5 * (h_A + h_B)
```

is exactly what enters the shared prediction auxiliary loss.

---

## RED 4 — No A/B collapse penalty

Assert there is no direct loss term of the form:

```python
||h_A - h_B||
```

or equivalent similarity penalty.

---

## RED 5 — Shared trajectory target detachment

Verify:

```python
target_shared_next.requires_grad == False
```

for the primary G3/G4 implementation.

---

## RED 6 — Genuine Novel cue

Assert for every novel vector:

```python
novel != A
novel != B
novel != 0
```

and that its norm satisfies the preregistered norm-matching rule.

---

## RED 7 — Novel/NoContext independence

Run both probes through the frozen network.

Assert their context tensors differ.

The analysis should fail if they are identical.

---

## RED 8 — Probe firewall

Frozen probe evaluation must not alter:

```text
parameters
optimizer state
RNG state
persistent hidden state
```

---

## RED 9 — Raw-loss identity

Plotting/exported raw sequence loss must equal the canonical training/evaluation loss within tolerance.

---

## RED 10 — True acquisition

Assert parameter hashes or parameter norms change across actual training updates.

Do not permit a frozen final checkpoint to masquerade as an acquisition trajectory.

---

# 18. Required outputs

```text
PHASE1G_SHARED_TRAJECTORY_REPORT.md
PHASE1G_manifest.json
PHASE1G_training_metrics.csv
PHASE1G_probe_metrics.csv
PHASE1G_shared_geometry.csv
PHASE1G_shared_rollout.csv
PHASE1G_novel_cue_vectors.csv
PHASE1G_posttraining_causal_checks.csv
```

Figures:

```text
PHASE1G_FIG01_transfer_by_condition.png
PHASE1G_FIG02_AB_separation_by_condition.png
PHASE1G_FIG03_shared_trajectory_error.png
PHASE1G_FIG04_shared_rollout_loss.png
PHASE1G_FIG05_mixed_nocontext_novel_raw_loss.png
PHASE1G_FIG06_novel_cue_generalization.png
PHASE1G_FIG07_probe_geometry.png
PHASE1G_FIG08_residual_patch_need.png

PHASE1G_TS01_true_AB_acquisition_raw_loss.png
PHASE1G_TS02_final_AB_probe_raw_loss.png
PHASE1G_TS03_shared_free_rollout_raw_loss.png
PHASE1G_TS04_probe_transition_raw_loss.png
PHASE1G_TS05_AB_separation_over_training.png
```

---

# 19. Report structure

## 1. Executive result

State which of G0–G5 best satisfies the frozen criteria.

## 2. Model/provenance audit

Confirm 8-unit High/Low canonical architecture.

## 3. Training interventions

Document exact auxiliary losses and selected weights.

## 4. Familiar acquisition

Show raw A/B loss across all training sequences.

## 5. A/B separation

Verify contextual splitting survives.

## 6. Shared-state predictiveness

Report shared prediction loss.

## 7. Shared-trajectory stability

Report one-step and free-rollout metrics.

## 8. Mixed transfer

Raw loss and geometry.

## 9. NoContext transfer

Raw loss and geometry.

## 10. Genuine Novel transfer

Mean and per-vector raw loss.

## 11. High-vs-Low comparison

Do not conflate Low success with the desired split+shared solution.

## 12. Post-training causal checks

Report whether residual shared-state patch rescue is reduced.

## 13. Decision-tree classification

Use only the frozen outcome definitions.

## 14. Paper-level implication

Keep descriptive, causal, and learnability claims separate.

---

# 20. Paper-level target interpretation if successful

If G4 or a simpler intervention produces:

- preserved A/B splitting,
- good familiar A/B performance,
- lower Mixed loss,
- lower NoContext loss,
- lower genuinely Novel loss,
- improved shared free rollout,
- reduced residual need for shared-state patching,

then the paper-level interpretation can be:

> Context-specific representations can be learned as modulations around a shared sequence backbone. Human-like transfer emerges when that shared sequence representation remains both predictive and dynamically stable, allowing the learner to reuse the same sequence computation when context is mixed, absent, or unfamiliar.

Do not use this conclusion unless the full pattern is observed.

---

# 21. Scope discipline

Do not turn Phase 1G into a large architecture search.

The intended scientific comparison is:

```text
baseline High
context dropout
shared prediction
shared trajectory
shared prediction + shared trajectory
```

Optional:

```text
context dropout + shared trajectory
```

Keep everything else fixed.

The goal is to determine **which minimal pressure preserves a reusable shared sequence trajectory without eliminating context-specific splitting**.
