# PRD — Phase 1J: Single-Injection Persistence and Contextual Stabilization

## 0. Purpose

Phase 1I established two important facts in the canonical 8-unit SRN:

1. Familiar A and B can remain strongly separated in representation and in backward credit assignment.
2. A familiar A/B composite hidden state is sufficient to strongly rescue Mixed, Novel, and NoContext when that state is repeatedly imposed.

The remaining mechanistic question is narrower:

> **Why can Mixed naturally approach/use the familiar A/B composite, while Novel and NoContext do not, even though all three can perform well when forcibly placed on the composite trajectory?**

Phase 1J should determine whether the difference is caused by **stability of the composite state under the ongoing context signal and recurrent dynamics**.

The key test is to inject the familiar composite state **once**, release the network, and measure how long the benefit persists.

This is an analysis/intervention phase only. **Do not retrain the SRN.**

---

# 1. Current evidence to preserve

Use the exact Phase 1I canonical setup and checkpoint provenance.

Phase 1I established:

- canonical 8-unit tanh SRNs;
- High initialization scale = 2.0;
- Low initialization scale = 0.02;
- 20 seeds per family;
- exact canonical replay;
- true raw CE acquisition trajectories;
- fixed familiar A/B reference trajectories;
- Mixed cue encoding satisfies:

\[
c_M = \tfrac12 c_A + \tfrac12 c_B
\]

with zero cue residual;

- therefore the familiar composite hidden target used in Phase 1I is exactly

\[
m_k = \tfrac12\left(h^A_k+h^B_k\right)
\]

at sequence transition \(k\);

- repeated composite-state intervention strongly rescues High-network Mixed, Novel, and NoContext;
- NovelMean must always average all four preregistered Novel cues.

Do not modify these definitions.

---

# 2. Scientific question

We want to distinguish four possibilities.

### H1 — Mixed naturally stabilizes the familiar composite

The Mixed cue is itself the A/B cue midpoint. After a one-time injection onto the familiar composite trajectory, the ongoing Mixed cue and recurrent dynamics keep the network relatively close to that trajectory.

Prediction:

\[
D^{Mixed}_{j} < D^{Novel}_{j},D^{NoContext}_{j}
\]

for lags after the injection, and Mixed retains a larger fraction of the immediate behavioral rescue.

---

### H2 — Novel cues actively push the system away

Novel cues may drive the hidden state away from the familiar A/B composite even after the correct sequence state is inserted.

Prediction:

- Novel under its actual cue rapidly loses rescue;
- Novel improves substantially when its cue is removed after the injection;
- Novel improves even more when its cue is replaced by the familiar composite cue.

---

### H3 — NoContext lacks a stabilizing contextual drive

NoContext has no unfamiliar cue pushing it away, but also lacks the Mixed cue that could keep the network near the familiar composite.

Prediction:

- NoContext remains unstable under \(c=0\);
- replacing \(c=0\) with \(c_M\) after the injection increases persistence.

---

### H4 — The composite is only locally useful, not dynamically stable

A single injection may help immediately but all probes rapidly leave the composite trajectory.

If so, the repeated-patch rescue from Phase 1I establishes **state sufficiency**, but not an attractor-like or self-maintaining composite mechanism.

This outcome is scientifically acceptable and should not be hidden.

---

# 3. Scope

## 3.1 Primary models

Run all analyses on:

- `High_monotonic`, all 20 canonical seeds;
- `Low_monotonic`, all 20 canonical seeds as a control.

High is the primary scientific target.

Low is a control for a network family with near-shared A/B solutions and near-ceiling familiar/probe performance.

---

## 3.2 Probe contexts

Evaluate:

- Mixed
- NoContext
- Novel1
- Novel2
- Novel3
- Novel4
- NovelMean = arithmetic mean over Novel1–Novel4

Never select the best Novel cue.

---

## 3.3 No training

Forbidden:

- optimizer steps;
- parameter updates;
- new losses used to fit the model;
- choosing intervention strengths from probe performance;
- refitting any composite coefficients to Mixed/Novel/NoContext behavior.

Every intervention strength is fixed at 1.0 unless explicitly stated otherwise below.

---

# 4. Canonical state definitions

For every seed and each predicted transition \(k\in\{1,\dots,7\}\), replay the final canonical model from the same zero initial state under familiar A and B.

Save:

\[
h^A_k,\qquad h^B_k
\]

and the familiar midpoint/composite:

\[
m_k = \frac{h^A_k+h^B_k}{2}.
\]

Let the output logits be

\[
\ell(h)=W_{out}h+b_{out}.
\]

Use centered task-space logits:

\[
z(h)=C\ell(h),
\qquad
C=I-\frac{1}{V}\mathbf 1\mathbf 1^\top.
\]

Define the corresponding midpoint task target:

\[
z^m_k=z(m_k).
\]

Important: do **not** average output probabilities. The target is defined in hidden state and evaluated through the actual output layer.

---

# 5. Intervention semantics

The sequence is evaluated with the same teacher-forced external sequence tokens as Phase 1I.

“Free rollout” in this PRD means:

> no additional hidden-state intervention after the initial patch.

It does **not** mean sampling the model’s predicted token and feeding it back.

---

# 6. Experiment A — Natural trajectory baseline

For every seed and probe cue, run the untouched final-checkpoint sequence.

At each transition \(k\), record:

- raw CE;
- hidden state \(h^{probe}_k\);
- centered logits \(z^{probe}_k\);
- hidden distance to midpoint target;
- task-space distance to midpoint target;
- distance to A;
- distance to B.

Use normalized residuals:

\[
D^h_k =
\frac{\|h^{probe}_k-m_k\|}
{\|h^A_k-h^B_k\|+\epsilon}
\]

and

\[
D^z_k =
\frac{\|z^{probe}_k-z^m_k\|}
{\|z^A_k-z^B_k\|+\epsilon}.
\]

This provides the no-intervention trajectory to which all patch conditions are compared.

---

# 7. Experiment B — Single-position composite injection

This is the primary experiment.

For each probe and each injection position

\[
k\in\{1,\dots,7\},
\]

perform exactly one hidden-state replacement:

\[
h_k \leftarrow m_k.
\]

Then:

- compute the output for transition \(k\);
- use the patched hidden state as the recurrent state entering transition \(k+1\);
- apply **no further hidden-state patches**;
- keep the probe’s natural context cue active unless a clamp condition below specifies otherwise;
- continue with the canonical teacher-forced sequence tokens.

Export all seven injection-position variants separately.

---

## 7.1 Behavioral quantities

For each injection position \(k\), compute:

### Immediate rescue

\[
R^{imm}_k
=
CE^{base}_k-CE^{patch}_k.
\]

### Remaining-sequence rescue

\[
R^{remain}_k
=
\sum_{j=k}^{7}
\left(
CE^{base}_j-CE^{patch}_j
\right).
\]

Also export the mean remaining CE difference so results are not dependent on number of transitions remaining.

### Full-sequence rescue

\[
R^{full}_k
=
\sum_{j=1}^{7}
\left(
CE^{base}_j-CE^{patch}_j
\right).
\]

The preferred quantity for mechanism is `remaining_sequence_rescue`, because transitions before the patch cannot be affected.

---

# 8. Experiment C — Post-injection drift / persistence

After each one-time patch at transition \(k\), measure where the system goes.

For lag

\[
j=0,\dots,7-k,
\]

define:

\[
D^h_{k,j}
=
\frac{
\|h^{patch}_{k+j}-m_{k+j}\|
}{
\|h^A_{k+j}-h^B_{k+j}\|+\epsilon
}
\]

and

\[
D^z_{k,j}
=
\frac{
\|z^{patch}_{k+j}-z^m_{k+j}\|
}{
\|z^A_{k+j}-z^B_{k+j}\|+\epsilon
}.
\]

By construction:

\[
D^h_{k,0}=0.
\]

Task-space distance at lag zero may be tiny numerical error but should also be approximately zero.

---

## 8.1 Persistence of behavioral rescue

For each lag, define:

\[
R_{k,j}
=
CE^{base}_{k+j}
-
CE^{patch}_{k+j}.
\]

Normalize by the immediate rescue only when the denominator is stable:

\[
P_{k,j}
=
\frac{R_{k,j}}
{R_{k,0}}.
\]

If

\[
|R_{k,0}|<\epsilon_R,
\]

flag the normalized persistence as undefined and retain the raw rescue.

Do not silently divide by small denominators.

---

## 8.2 Summary persistence score

For each probe and seed, calculate:

\[
S^{beh}
=
\operatorname{mean}_{k,j>0}
R_{k,j}
\]

and

\[
S^{state}
=
-\operatorname{mean}_{k,j>0}
D^h_{k,j}.
\]

Larger values indicate greater post-injection persistence.

Also report task-space persistence using \(D^z\).

Do not collapse to these summaries without retaining every transition-level value.

---

# 9. Experiment D — Context-clamp decomposition

Repeat Experiment B under three post-injection context conditions.

The injection itself is always the same familiar midpoint state \(m_k\).

Only the context input used **after** the patch differs.

## D0 — Natural cue continuation

Continue using the actual probe cue:

\[
c_t=c_{probe}.
\]

This is Experiment B.

---

## D1 — Cue removed after patch

For transitions after the injection:

\[
c_t=0.
\]

Purpose:

> Does the probe cue actively push the state away from the familiar composite?

Predicted diagnostic:

- if Novel persists better under D1 than D0, the Novel cue is actively destabilizing;
- Mixed should not improve under D1 if its own cue is stabilizing;
- NoContext D0 and D1 should be exactly identical because its natural cue is already zero.

That final equality is a required unit test.

---

## D2 — Familiar composite cue after patch

For transitions after the injection:

\[
c_t=c_M
=
\tfrac12(c_A+c_B).
\]

Because Phase 1I found exact cue-level composition, this is also the actual Mixed cue.

Purpose:

> Is the familiar composite cue sufficient to stabilize the familiar composite sequence state?

Predictions:

- Mixed D0 and D2 must be numerically identical;
- Novel should show greater persistence under D2 than D0 if its own cue is the problem;
- NoContext should show greater persistence under D2 than D0 if the absent context fails to stabilize the shared solution.

---

# 10. Experiment E — Directional controls

The primary result must not be explainable by merely perturbing the hidden state.

Repeat the single-injection experiment with the existing Phase 1I controls.

## E1 — Wrong-position midpoint

Use the same fixed derangement as Phase 1I:

\[
h_k \leftarrow m_{\pi(k)}
\]

where \(\pi(k)\neq k\).

The derangement must be preregistered/fixed globally.

---

## E2 — Norm-matched orthogonal movement

Construct a deterministic hidden displacement orthogonal to

\[
m_k-h^{probe}_k
\]

with equal norm.

Verify:

\[
\Delta h_\perp^\top
(m_k-h^{probe}_k)
\approx 0
\]

and

\[
\|\Delta h_\perp\|
=
\|m_k-h^{probe}_k\|.
\]

---

## E3 — A endpoint

\[
h_k\leftarrow h^A_k.
\]

---

## E4 — B endpoint

\[
h_k\leftarrow h^B_k.
\]

---

## E5 — Repeated midpoint patch upper bound

Reuse the Phase 1I repeated composite intervention as an **upper-bound reference**, not as the primary Phase 1J condition.

This makes it possible to distinguish:

- immediate state sufficiency;
- natural persistence;
- maximum rescue under forced trajectory maintenance.

---

# 11. Main predictions

The strongest version of the mechanism predicts the following pattern in High networks.

## Mixed

After one midpoint injection:

- immediate loss improves;
- hidden/task residual grows slowly;
- remaining-sequence rescue stays positive;
- D0 ≈ D2;
- D1 is worse than D0/D2.

Interpretation:

> the physical Mixed cue stabilizes the familiar A/B composite.

---

## Novel

After one midpoint injection:

- immediate loss improves;
- the state rapidly drifts away under its actual Novel cue;
- D1 improves persistence relative to D0 if Novel cue is actively destabilizing;
- D2 improves persistence even more if the familiar composite cue can stabilize the shared solution.

Interpretation:

> Novel has access to the shared sequence computation but its unfamiliar context transformation sends recurrent state away from it.

---

## NoContext

After one midpoint injection:

- immediate loss improves;
- persistence under D0 is limited;
- D0 = D1 exactly;
- D2 improves persistence.

Interpretation:

> absence of context does not actively introduce a competing transformation, but it fails to provide the stabilizing drive supplied by the Mixed cue.

---

# 12. Primary comparisons

Do not define success from a single arbitrary threshold.

Report continuous effects first.

Primary paired seed-level comparisons in High:

1. Mixed vs NovelMean post-injection hidden drift.
2. Mixed vs NoContext post-injection hidden drift.
3. Mixed vs NovelMean remaining-sequence rescue.
4. Mixed vs NoContext remaining-sequence rescue.
5. Novel D1 − D0 persistence improvement.
6. Novel D2 − D0 persistence improvement.
7. NoContext D2 − D0 persistence improvement.
8. Mixed D2 − D0 numerical equality.
9. NoContext D1 − D0 numerical equality.

Use seed bootstrap intervals.

The resampling unit is **seed**, never transitions.

---

# 13. Secondary decomposition: recurrence versus cue drive

For every probe calculate:

\[
\Delta_{\text{remove}}
=
S(D1)-S(D0)
\]

and

\[
\Delta_{\text{composite}}
=
S(D2)-S(D0).
\]

Interpretation:

### Novel

- \(\Delta_{\text{remove}}>0\): Novel cue actively destabilizes.
- \(\Delta_{\text{composite}}>0\): composite familiar cue stabilizes.

### NoContext

Because D0 = D1:

- \(\Delta_{\text{remove}}=0\) by construction;
- \(\Delta_{\text{composite}}>0\) means the missing factor is stabilizing context input.

### Mixed

Because D0 = D2:

- \(\Delta_{\text{composite}}=0\) by construction;
- D1 worsening indicates the Mixed cue contributes to persistence beyond the recurrent state alone.

---

# 14. Decision table

## Outcome A — Full contextual stabilization account

Required qualitative pattern:

- single midpoint injection produces immediate rescue;
- Mixed rescue persists better than Novel/NoContext;
- Mixed D0 = D2;
- Novel persistence improves when its cue is removed and/or replaced by \(c_M\);
- NoContext D0 = D1 but improves under D2;
- directional controls do not reproduce the effect.

Interpretation:

> Shared sequence structure exists, but context controls whether the composite state is dynamically maintained.

This is the strongest current paper mechanism.

---

## Outcome B — Mixed-specific persistence without clean cue decomposition

- Mixed persists substantially better;
- Novel/NoContext do not;
- D1/D2 manipulations do not cleanly localize why.

Interpretation:

> Mixed naturally supports the composite trajectory, but the precise stabilizing mechanism remains unresolved.

Still useful.

---

## Outcome C — All probes retain the injection similarly

Interpretation:

> Natural generation, not post-generation stability, is the main failure.

Then the next experiment should target **how the initial probe state is constructed from the cue**, not recurrent maintenance.

---

## Outcome D — No probe retains the injection

Interpretation:

> The familiar midpoint is behaviorally useful when forced but is not a self-maintaining trajectory.

The averaging account would then require repeated online contextual recomputation rather than stabilization of a single shared state.

---

## Outcome E — Controls rescue as well as the midpoint

Interpretation:

> The effect is nonspecific and the current midpoint/composite mechanism is not supported.

Do not proceed to stronger claims.

---

# 15. Mandatory raw-loss figures

All behavioral figures must use **raw cross-entropy**, never normalized loss.

## FIG01 — Canonical acquisition audit

Reproduce the true canonical High and Low A/B acquisition trajectories across all 170 external training sequences.

Requirements:

- raw mean CE;
- actual chronological training sequence index;
- A and B shown separately;
- changing checkpoint/parameter hashes verified;
- no frozen-checkpoint schedule plot.

This figure is an audit/control and should not change relative to Phase 1I.

---

## FIG02 — Final natural probe transition profiles

For High and Low separately, show raw CE across the seven predicted transitions for:

- A
- B
- Mixed
- NoContext
- Novel1
- Novel2
- Novel3
- Novel4
- NovelMean

No intervention.

---

## FIG03 — Single-injection raw-loss trajectories

For each probe family and injection position, show raw CE from transition 1–7 for:

- no patch;
- one-time midpoint patch;
- repeated midpoint patch upper bound;
- wrong-position patch;
- orthogonal patch;
- A endpoint;
- B endpoint.

Because this can become visually dense, make:

- a primary panel using injection positions 1, 3, 5;
- a supplementary full 7-position grid.

Do not omit the complete version.

---

## FIG04 — Post-injection hidden drift

Lag from patch on x-axis.

Plot normalized hidden midpoint residual for:

- Mixed
- NoContext
- NovelMean

under:

- D0 natural cue
- D1 zero cue
- D2 composite cue

High primary, Low control.

---

## FIG05 — Post-injection task-space drift

Same as FIG04 using centered-logit residual.

---

## FIG06 — Behavioral rescue persistence

Lag from injection on x-axis.

Plot:

\[
CE^{baseline}-CE^{single\ patch}
\]

for Mixed, NoContext, NovelMean.

Show raw rescue first.

Normalized rescue may be supplemental.

---

## FIG07 — Context-clamp causal decomposition

Seed-level paired summary of:

- D1 − D0
- D2 − D0

for Mixed, NoContext, NovelMean.

Use connected seed points, not bars alone.

---

## FIG08 — Immediate versus remaining rescue

Per seed:

x-axis = immediate rescue

y-axis = remaining-sequence rescue after the immediate transition.

Color by probe.

This separates a purely local patch effect from a persistent one.

---

## FIG09 — Repeated-patch upper bound versus single-patch persistence

For each probe:

- baseline CE;
- single-patch remaining CE;
- repeated-patch CE.

This directly quantifies the gap between:

> state sufficiency

and

> natural recurrent maintenance.

---

# 16. Mandatory full-sequence raw-loss reference plot

Create one final paper-facing figure that puts familiar and probe performance on the same raw-loss scale.

For High:

- A raw CE by transition;
- B raw CE by transition;
- Mixed natural;
- Mixed single-patch;
- NovelMean natural;
- NovelMean single-patch;
- NoContext natural;
- NoContext single-patch.

This must make it visually obvious whether the one-shot intervention moves each probe toward familiar A/B performance and whether that improvement persists.

Also export the same figure for Low as a control.

---

# 17. Output tables

## `phase1j_transition_metrics.csv`

One row per:

- family
- seed
- probe
- injection_position
- transition
- intervention
- context_clamp

Columns must include at minimum:

- `raw_ce`
- `baseline_raw_ce`
- `raw_rescue`
- `hidden_midpoint_residual`
- `task_midpoint_residual`
- `distance_to_A_hidden`
- `distance_to_B_hidden`
- `distance_to_A_task`
- `distance_to_B_task`
- `lag_from_patch`
- `parameter_hash`
- `checkpoint_hash`

---

## `phase1j_seed_summary.csv`

One row per:

- family
- seed
- probe
- intervention
- context_clamp

Include:

- immediate rescue;
- remaining-sequence rescue;
- mean post-patch hidden residual;
- mean post-patch task residual;
- behavioral persistence score;
- hidden persistence score;
- task persistence score;
- repeated-patch upper-bound rescue;
- fraction of upper-bound rescue retained after a single patch.

For the last quantity:

\[
F_{\text{retained}}
=
\frac{R_{\text{single}}}
{R_{\text{repeated}}+\epsilon}
\]

with denominator instability explicitly flagged.

---

## `phase1j_statistical_summary.csv`

Seed-bootstrap means/medians and 95% CIs for every primary comparison.

---

## `phase1j_provenance.csv`

For every seed:

- source checkpoint path;
- source checkpoint hash;
- architecture;
- hidden size;
- activation;
- initialization scale;
- cue vectors/hash;
- sequence/hash;
- output-layer hash.

---

# 18. Statistical analysis

Use seeds as the independent sampling unit.

Primary summaries:

- mean;
- median;
- paired seed difference;
- bootstrap 95% CI;
- fraction of seeds with effect in predicted direction.

Do not treat transitions, injection positions, or Novel variants as independent observations.

Novel1–Novel4 are repeated conditions nested within seed.

NovelMean must be computed within seed before seed-level inference.

---

# 19. TDD / RED → GREEN requirements

Implement tests before analysis code.

## Test 1 — canonical architecture

For every checkpoint assert:

- hidden size = 8;
- tanh recurrence;
- High scale = 2.0;
- Low scale = 0.02.

---

## Test 2 — no parameter mutation

Hash all model parameters before and after every probe/intervention evaluation.

Assert exact equality.

---

## Test 3 — exact midpoint target

For every seed and transition:

\[
m_k=\tfrac12(h^A_k+h^B_k)
\]

to numerical tolerance.

---

## Test 4 — exact Mixed cue composition

Assert:

\[
c_M=\tfrac12(c_A+c_B)
\]

to Phase 1I tolerance.

Fail loudly if not true.

---

## Test 5 — Mixed D0 equals D2

Because D2 uses \(c_M\), natural Mixed continuation and composite-cue continuation must be identical.

Assert transition-level equality.

---

## Test 6 — NoContext D0 equals D1

Because natural NoContext already uses \(c=0\), D0 and cue-removal D1 must be identical.

Assert transition-level equality.

---

## Test 7 — exactly one hidden patch

For the single-injection condition, instrument the forward pass.

Assert:

- hidden replacement occurs exactly once;
- at the registered transition;
- no later hidden state is overwritten.

---

## Test 8 — recurrent inheritance

Verify that the patched hidden state is the state passed into the next recurrent transition.

This is essential.

---

## Test 9 — teacher-forced token identity

Assert that all conditions receive the exact same external sequence tokens.

Only hidden intervention/context clamp may differ.

---

## Test 10 — seven predicted transitions

Every sequence-level raw CE must average exactly seven next-token CE terms.

No first-token or padding leakage.

---

## Test 11 — wrong-position derangement

Assert:

\[
\pi(k)\neq k
\]

for every position.

The derangement must be fixed before evaluation.

---

## Test 12 — orthogonal control

Assert:

\[
|\Delta h_\perp^\top \Delta h_{comp}|<\epsilon
\]

and norm matching within tolerance.

---

## Test 13 — NovelMean

Assert NovelMean contains Novel1–Novel4 exactly once each.

---

## Test 14 — no behavioral fitting

Composite coefficients and targets must be created by functions that receive only:

- familiar cue encodings;
- familiar A/B hidden states;
- registered deterministic control seed where needed.

They must not receive probe losses.

---

## Test 15 — bootstrap unit

Assert bootstrap resampling indexes seeds only.

---

## Test 16 — acquisition figure provenance

The canonical acquisition plot must use actual pre-update replay states with changing parameter hashes.

Reject any plot generated from a single frozen final checkpoint.

---

# 20. Implementation order

Use RED → GREEN.

### Phase A — provenance + target reconstruction

1. Load canonical checkpoints.
2. Verify architecture/hashes.
3. Reconstruct A/B trajectories.
4. Reconstruct midpoint targets.
5. Verify exact Mixed cue composition.

No probe evaluation until all tests pass.

---

### Phase B — natural probe baselines

Run untouched Mixed, NoContext, Novel1–4.

Export raw CE and state/task residuals.

---

### Phase C — single-injection experiment

Run all seven injection positions under natural cue continuation.

Export full transition-level results.

---

### Phase D — context clamp

Repeat single-injection experiment under D1 and D2.

Run equality tests for Mixed and NoContext.

---

### Phase E — directional controls

Run wrong-position, orthogonal, A endpoint, B endpoint, and repeated midpoint upper bound.

---

### Phase F — aggregation/statistics

Aggregate only after raw transition CSVs are frozen.

Bootstrap seeds.

---

### Phase G — figures/report

Generate every mandatory figure directly from frozen CSVs.

---

# 21. Report structure

Create:

`PHASE1J_SINGLE_INJECTION_CONTEXT_STABILITY_REPORT.md`

Sections:

1. Executive summary
2. Provenance audit
3. Exact midpoint/composite verification
4. Natural probe trajectories
5. Immediate single-injection rescue
6. Post-injection hidden-state drift
7. Post-injection task-space drift
8. Behavioral rescue persistence
9. Context-clamp decomposition
10. Directional controls
11. Repeated-patch upper bound versus single-patch persistence
12. High-versus-Low comparison
13. Per-seed heterogeneity
14. Decision-table outcome
15. Implications for the human averaging account
16. Limitations

---

# 22. Interpretation constraints

Do not call the midpoint:

- an attractor;
- a stable manifold;
- a naturally generated shared representation;

unless the corresponding persistence evidence supports that claim.

If a one-time midpoint patch helps only at the patched transition, say:

> the state is locally sufficient but not dynamically maintained.

If Mixed retains the state substantially better than Novel/NoContext, say:

> the Mixed context supports persistence of a useful familiar composite state.

If Novel improves when its cue is removed, say:

> the Novel context actively displaces the familiar composite.

If NoContext improves only when \(c_M\) is supplied, say:

> absent context fails to provide the stabilizing drive supplied by the familiar composite cue.

---

# 23. Paper-level mechanism if Outcome A holds

The intended mechanistic statement is:

> The network can separate familiar contexts strongly enough to insulate learning while retaining a common sequence solution that can be expressed through different context-conditioned states. Because the Mixed cue is itself a composition of the familiar cue codes, it can stabilize an intermediate familiar sequence state. Novel and absent contexts do not naturally maintain that state, even though forcing the state demonstrates that the shared sequence computation remains usable.

The corresponding conceptual decomposition is:

\[
\boxed{
\text{shared sequence computation}
+
\text{context-specific expression}
+
\text{context-specific credit assignment}
}
\]

This explains how strong A/B separation can coexist with an averaging-like Mixed probe.

---

# 24. Stop condition

Do not add new architectures, new training objectives, or new human-model fitting in Phase 1J.

First answer this question cleanly:

> **After placing the network once onto the familiar A/B composite sequence state, which contexts keep it there, which contexts push it away, and which contexts simply fail to stabilize it?**

Only after that result is known should the next phase be designed.
