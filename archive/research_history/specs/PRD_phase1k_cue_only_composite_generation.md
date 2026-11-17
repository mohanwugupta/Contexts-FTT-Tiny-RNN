# PRD — Phase 1K: Cue-Only Generation of the Composite Sequence State

## Status

**Proposed final SRN mechanism test.**

This experiment should be implemented as a **frozen-checkpoint causal diagnostic**. It must not train, fine-tune, select, or otherwise modify model parameters.

The purpose is to close the remaining mechanistic gap left by Phase 1I and Phase 1J:

- Phase 1I showed that a familiar A+B-derived composite state can causally rescue Mixed, Novel, and NoContext.
- Phase 1J showed that a single composite-state injection gives a large immediate rescue, but that the rescue persists best when the physical Mixed cue continues to be present.
- What remains unresolved is whether the **Mixed cue itself can naturally generate the useful composite trajectory**, rather than merely stabilize it after we manually place the network there.

If this test is clean, stop the SRN analysis program and write the paper.

---

# 1. Scientific question

The core question is:

> **Can the overlapping A+B cue itself route an already context-specific sequence state toward the shared/composite trajectory that produces intermediate behavior?**

This is the missing link between two facts:

1. High-scale networks strongly separate A and B.
2. A+B overlap nevertheless produces an intermediate/composite solution.

The proposed mechanism is:

\[
\text{shared sequence structure}
\;\xrightarrow{\text{context transformation}}\;
\text{A-specific or B-specific trajectory}
\]

while the overlapping cue applies an intermediate transformation:

\[
c_M = \frac{1}{2}(c_A+c_B)
\]

so that a state currently evolving under A, B, Novel, or NoContext can be redirected toward the familiar composite trajectory.

The key distinction is between:

- **state sufficiency**: manually putting the network on the composite trajectory improves behavior;
- **cue generation**: the physical Mixed cue itself moves the network toward that trajectory;
- **cue maintenance**: continued Mixed input keeps the network near that trajectory once it has moved there.

Phase 1J already addressed state sufficiency and part of maintenance. Phase 1K should test generation directly.

---

# 2. Main hypotheses

## H1 — Mixed cue generates movement toward the familiar midpoint/composite state

Starting from a state produced by A, B, NoContext, or Novel, replacing only the context input with the physical Mixed cue should move the next hidden state and task state **toward the A/B midpoint**, without directly patching the hidden state.

For transition \(k\):

\[
m^h_k = \frac{h^A_k+h^B_k}{2}
\]

and in centered-logit/task space:

\[
m^z_k = \frac{z^A_k+z^B_k}{2}.
\]

The strongest version predicts that, from the same prestate,

\[
d(h^{\text{Mixed cue}}_k,m^h_k)
<
d(h^{\text{native cue}}_k,m^h_k)
\]

and similarly in task space.

---

## H2 — Mixed cue can redirect already separated A and B trajectories toward the same composite trajectory

This is the critical bridge to the human interpretation.

Run the network naturally under A or B for the beginning of the sequence. At a selected transition, **switch only the cue to Mixed** and leave the hidden state untouched.

If the proposed mechanism is correct, both:

\[
A \rightarrow Mixed
\]

and

\[
B \rightarrow Mixed
\]

should converge toward the same intermediate trajectory.

This would show directly that strong A/B separation does not prevent later averaging because the overlapping context can transform either context-specific state toward a common composite route.

---

## H3 — Mixed cue can rescue Novel and NoContext without state injection

Run Novel or NoContext naturally for the first part of the sequence, then replace the cue with Mixed.

If their poor natural transfer reflects failure to generate the composite state, Mixed takeover should reduce their raw CE and their distance from the familiar midpoint.

This is a cue-only rescue:

- no hidden patch;
- no output patch;
- no parameter update;
- no learned adapter.

---

## H4 — Continuous Mixed input is more effective than a one-step Mixed pulse

Compare:

1. **single-pulse Mixed cue**: use Mixed for exactly one transition, then return to the source cue;
2. **persistent Mixed takeover**: use Mixed from the intervention point through the rest of the sequence.

If context continually shapes the sequence trajectory, persistent takeover should produce more durable convergence and behavioral rescue than a one-step pulse.

This tests **generation + stabilization** separately from a single perturbation.

---

## H5 — Cue direction matters

From the same pre-intervention hidden state, compare cue branches using:

- source/native cue;
- physical Mixed cue;
- A cue;
- B cue;
- zero cue where meaningful.

The Mixed cue should preferentially move the state toward the familiar midpoint, while A and B cues should preferentially move it toward their respective familiar endpoint trajectories.

Behavior alone is not sufficient for this control because A or B may also lower loss. Geometry must show that different cues route toward different targets.

---

# 3. Scope and hard constraints

## 3.1 Model families

Use the already registered canonical models:

- `High_monotonic`: 20 seeds, 8-unit tanh SRN, initialization scale 2.0.
- `Low_monotonic`: 20 seeds, 8-unit tanh SRN, initialization scale 0.02.

The High family is the primary mechanistic target.

The Low family is a comparison/control family and must not be treated as a second discovery set.

---

## 3.2 Frozen models only

For every intervention:

- model parameters frozen;
- output layer frozen;
- hidden state changed only by the normal recurrent update;
- no optimizer instantiated;
- no backward pass required;
- no checkpoint selection;
- no intervention strength fitting;
- no fitting to Mixed, NoContext, or Novel behavior.

A hash of every parameter tensor before and after the full analysis must be identical.

---

## 3.3 No direct hidden-state patching in the primary experiment

Phase 1K is specifically a **cue-only generation test**.

Do **not** insert:

\[
(h_A+h_B)/2
\]

or any other hidden target during the primary cue-transplant conditions.

Previous midpoint/composite patch results may be loaded as references, but must not be rerun as the primary manipulation.

---

## 3.4 No probability averaging

Never average output probabilities.

All reference composition is defined in hidden state and centered-logit/task space.

---

## 3.5 Analyze all Novel cues

Use:

- Novel1
- Novel2
- Novel3
- Novel4

and compute `NovelMean` **within seed first**.

Do not select the easiest Novel cue.

---

# 4. Exact SRN indexing

Avoid any ambiguity about when the cue acts.

For transition \(k\in\{1,\dots,7\}\):

\[
h_k =
\tanh(
W_x[x_k;c_k] + W_h h_{k-1}+b_h
)
\]

\[
\ell_k = W_o h_k+b_o
\]

\[
p_k = \operatorname{softmax}(\ell_k)
\]

\[
CE_k = -\log p_k(x_{k+1}).
\]

A cue intervention at transition \(k\) means:

- history through \(h_{k-1}\) is completely natural;
- the altered cue is first used in the update that produces \(h_k\);
- the resulting \(h_k\) generates the prediction for transition \(k\);
- that new \(h_k\) is inherited by all later recurrent updates.

Write a unit test specifically for this off-by-one convention.

---

# 5. Frozen familiar references

For each checkpoint and seed, first compute natural zero-initialized rollouts under:

- A cue;
- B cue;
- Mixed cue;
- NoContext/zero cue;
- Novel1–4.

Store every transition-level:

- hidden state;
- logits;
- centered logits;
- correct-token probability;
- raw CE.

Define centered logits:

\[
C = I-\frac{1}{m}\mathbf{1}\mathbf{1}^{\top}
\]

\[
z_k = C\ell_k.
\]

Define the familiar midpoint references:

\[
m^h_k = \frac{h^A_k+h^B_k}{2}
\]

\[
m^z_k = \frac{z^A_k+z^B_k}{2}.
\]

Also retain endpoint separation denominators:

\[
D^h_{AB,k}=\|h^A_k-h^B_k\|
\]

\[
D^z_{AB,k}=\|z^A_k-z^B_k\|.
\]

These references must be computed **once from the untouched checkpoint** and then frozen.

---

# 6. Phase A — Local cue-generation branch test

This is the cleanest direct test of whether the physical cue points the dynamics toward the desired state.

For each:

- family;
- seed;
- source condition \(s\);
- transition \(k\);

take the **same exact natural source prestate**:

\[
h^{s}_{k-1}.
\]

Source conditions:

- A
- B
- NoContext
- Novel1
- Novel2
- Novel3
- Novel4

From that identical prestate and identical sequence token \(x_k\), branch the recurrent update under:

1. `native`
2. `Mixed`
3. `A`
4. `B`
5. `zero`

where duplicate branches may collapse by definition, e.g. NoContext native = zero.

No branch may inherit another branch's post-update state in Phase A. This is strictly a one-step matched-prestate comparison.

---

## 6.1 Primary local generation metric

For each branch:

\[
R^h_{mid}(s,k,c)
=
\frac{\|h_k(s,c)-m^h_k\|}
{D^h_{AB,k}+\epsilon}
\]

and:

\[
R^z_{mid}(s,k,c)
=
\frac{\|z_k(s,c)-m^z_k\|}
{D^z_{AB,k}+\epsilon}.
\]

Define Mixed-cue midpoint improvement:

\[
G^h_{mid}
=
R^h_{mid}(s,k,\text{native})
-
R^h_{mid}(s,k,\text{Mixed})
\]

and:

\[
G^z_{mid}
=
R^z_{mid}(s,k,\text{native})
-
R^z_{mid}(s,k,\text{Mixed}).
\]

Positive values mean the physical Mixed cue moved the next state closer to the familiar midpoint.

---

## 6.2 Vector-direction alignment

From the common native prestate, define the cue-induced hidden displacement:

\[
\Delta h^{M}_k
=
h_k(s,\text{Mixed})
-
h_k(s,\text{native})
\]

and target direction:

\[
v^h_k
=
m^h_k-h_k(s,\text{native}).
\]

Calculate:

\[
A^h_{dir}
=
\cos(\Delta h^{M}_k,v^h_k).
\]

Repeat in centered-logit space:

\[
A^z_{dir}
=
\cos(\Delta z^{M}_k,v^z_k).
\]

Interpretation:

- positive and large: Mixed cue locally pushes toward midpoint;
- near zero: cue has little directional relation;
- negative: cue pushes away.

Do not hide cases where the target direction has negligible norm. Flag denominator instability explicitly.

---

## 6.3 Endpoint specificity

Using the same prestate:

\[
R^h_A(c)=
\frac{\|h_k(s,c)-h^A_k\|}
{D^h_{AB,k}+\epsilon}
\]

\[
R^h_B(c)=
\frac{\|h_k(s,c)-h^B_k\|}
{D^h_{AB,k}+\epsilon}.
\]

Test whether:

- A cue is closest to A reference;
- B cue is closest to B reference;
- Mixed cue is closest to the midpoint.

Repeat in task space.

This is the main directional specificity control.

---

## 6.4 Local behavioral effect

Compute the raw CE from each one-step branch.

Define:

\[
\Delta CE^{local}_{Mixed}
=
CE_{\text{native}}
-
CE_{\text{Mixed cue}}.
\]

Behavior is secondary to geometry in Phase A.

A cue can move toward the midpoint even if one transition does not immediately lower CE.

---

# 7. Phase B — Persistent cue takeover

This is the main natural-generation intervention.

For each source condition:

- A
- B
- NoContext
- Novel1–4

and each takeover transition:

\[
k\in\{1,\dots,7\},
\]

run the source cue naturally through transition \(k-1\).

At transition \(k\):

- do **not** alter the hidden state;
- replace the cue with the physical Mixed cue;
- keep Mixed active for every remaining transition.

Formally:

\[
c_t =
\begin{cases}
c_s, & t < k\\
c_M, & t\ge k.
\end{cases}
\]

Call this condition:

`persistent_mixed_takeover@k`

The source hidden state at \(k-1\) must be bit-identical to the source natural rollout.

---

# 8. Phase C — One-step cue pulse

Repeat the exact same setup, but Mixed is applied for only one transition:

\[
c_t =
\begin{cases}
c_s, & t<k\\
c_M, & t=k\\
c_s, & t>k.
\end{cases}
\]

Call this:

`single_mixed_pulse@k`.

This distinguishes:

- **generation from one cue update**
from
- **continued contextual stabilization**.

Do not change any other aspect of the rollout.

---

# 9. Phase D — Endpoint takeover controls

For a compact directional control, repeat persistent takeover using:

- A cue;
- B cue.

Do this at minimum for source conditions:

- A
- B
- NoContext
- NovelMean analysis derived from Novel1–4.

Conditions:

`persistent_A_takeover@k`

`persistent_B_takeover@k`

Primary purpose:

- A cue should drive toward A endpoint;
- B cue should drive toward B endpoint;
- Mixed cue should drive toward midpoint.

Do not treat endpoint behavioral improvement as evidence against the compositional account. The geometry is the discriminating readout.

---

# 10. Primary readouts

All readouts must be exported at:

- seed level;
- source condition level;
- transition-of-intervention level;
- lag level;
- intervention level.

---

## 10.1 Raw CE trajectory

Retain:

\[
CE_1,\dots,CE_7
\]

for every rollout.

Never replace the raw transition profile with only a mean.

---

## 10.2 Immediate rescue

At the intervention transition:

\[
Rescue^{imm}_{k}
=
CE^{native}_{k}
-
CE^{intervention}_{k}.
\]

---

## 10.3 Remaining-sequence rescue

For the available post-intervention transitions:

\[
Rescue^{remain}_{k}
=
\sum_{t=k}^{7}
\left(
CE^{native}_t-CE^{intervention}_t
\right).
\]

Also calculate the mean rescue:

\[
\overline{Rescue}^{remain}_{k}
=
\frac{1}{8-k}
Rescue^{remain}_{k}.
\]

Keep both because the number of remaining transitions varies with \(k\).

---

## 10.4 Hidden midpoint residual over lag

For each lag \(j\ge0\):

\[
R^h_{mid}(k+j)
=
\frac{
\|h^{intervention}_{k+j}-m^h_{k+j}\|
}{
D^h_{AB,k+j}+\epsilon
}.
\]

Plot the full lag trajectory.

---

## 10.5 Task-space midpoint residual over lag

\[
R^z_{mid}(k+j)
=
\frac{
\|z^{intervention}_{k+j}-m^z_{k+j}\|
}{
D^z_{AB,k+j}+\epsilon
}.
\]

This is a primary metric because behavior is generated through the output-relevant task state.

---

## 10.6 Endpoint coordinate

For each transition, project onto the familiar A→B axis:

\[
\alpha_k
=
\frac{
(z_k-z^A_k)^\top(z^B_k-z^A_k)
}{
\|z^B_k-z^A_k\|^2+\epsilon
}.
\]

Interpretation:

- \(\alpha=0\): A endpoint;
- \(\alpha=1\): B endpoint;
- \(\alpha=.5\): familiar midpoint.

Also compute normalized off-axis residual:

\[
r^\perp_k
=
\frac{
\|z_k-[z^A_k+\alpha_k(z^B_k-z^A_k)]\|
}{
\|z^B_k-z^A_k\|+\epsilon
}.
\]

A successful Mixed takeover should move states toward \(\alpha\approx .5\) **without large off-axis excursion**.

This prevents a misleading conclusion in which the intervention merely finds a different low-loss state.

---

## 10.7 Pulse-versus-takeover persistence

For every source and injection position compare:

\[
CE^{pulse}_t
\quad\text{vs}\quad
CE^{takeover}_t
\]

and:

\[
R^{z,pulse}_{mid,t}
\quad\text{vs}\quad
R^{z,takeover}_{mid,t}.
\]

Define:

\[
\Delta Persist_k
=
\overline{Rescue}^{remain}_{takeover,k}
-
\overline{Rescue}^{remain}_{pulse,k}.
\]

Positive values support continuous contextual stabilization.

---

# 11. The critical A→Mixed / B→Mixed test

This should receive its own primary figure and its own report section.

For every seed and intervention position:

1. Roll out A naturally until \(k-1\).
2. Continue under Mixed.
3. Roll out B naturally until \(k-1\).
4. Continue under Mixed.
5. Compare both resulting trajectories with:
   - natural A;
   - natural B;
   - natural Mixed;
   - familiar midpoint target.

The mechanistic prediction is not merely lower CE.

It is:

\[
A\rightarrow M
\quad\text{and}\quad
B\rightarrow M
\]

should **converge toward each other** and toward the same intermediate task trajectory.

Define convergence:

\[
D^{AB\rightarrow M}_{k+j}
=
\frac{
\|z^{A\rightarrow M}_{k+j}
-
z^{B\rightarrow M}_{k+j}\|
}{
\|z^A_{k+j}-z^B_{k+j}\|+\epsilon
}.
\]

A value below 1 means the same Mixed cue has reduced the original A/B separation.

Also calculate midpoint error for each branch separately.

This is the most direct test of the proposed explanation for averaging after context-specific learning.

---

# 12. Novel and NoContext interpretation

Do not require Novel and NoContext to behave identically.

The final experiment should distinguish at least three possible failures.

## Failure type 1 — Generation failure

Mixed cue does not move Novel/NoContext toward the midpoint even locally.

Then their state is positioned such that the familiar composite cue cannot construct the target from that prestate.

## Failure type 2 — Generation succeeds, stabilization fails

One-step geometry moves correctly, but continued rollout drifts away despite persistent Mixed cue.

Then recurrence/dynamics prevent maintaining the composite trajectory.

## Failure type 3 — Geometry succeeds and behavior remains poor

Hidden/task state moves toward the midpoint but raw CE remains high.

Then the midpoint itself is not behaviorally sufficient from that history, contrary to the earlier patch result or because the cue-only path reaches a different state neighborhood.

This should trigger an audit before any new mechanism is proposed.

---

# 13. High-versus-Low comparison

High is the primary condition.

For Low:

- run the same code path;
- report the same metrics;
- expect small behavioral rescue because natural performance is already near floor;
- do not use raw CE improvement alone to compare mechanism strength.

The more meaningful Low comparison is geometric:

- how much Mixed cue changes \(\alpha\);
- midpoint residual;
- off-axis residual;
- convergence of A→Mixed and B→Mixed.

---

# 14. Required figures

All figures must use **raw CE/loss** when showing model performance.

Do not substitute execution proxy, normalized performance, or correct-token probability for the raw-loss plots.

---

## FIG01 — Canonical provenance: true raw A/B acquisition

Reproduce the genuine canonical pre-update acquisition trajectory for High and Low:

- x = actual external training sequence index;
- y = raw sequence CE, mean of exactly seven transitions;
- separate A and B;
- changing parameter hash across training;
- final replay must reproduce registered checkpoints.

This is only a provenance panel, not a new result.

---

## FIG02 — Natural final-checkpoint transition profiles

For High and Low separately:

- A;
- B;
- Mixed;
- NoContext;
- Novel1–4;
- NovelMean.

x = transition 1–7.

y = raw CE.

No intervention.

---

## FIG03 — Local cue-generation geometry

For each source family:

- native cue;
- Mixed cue;
- A cue;
- B cue;
- zero cue.

Show:

1. hidden midpoint residual;
2. task midpoint residual;
3. cue-to-target directional cosine;
4. raw CE.

Primary sources to emphasize:

- A;
- B;
- NoContext;
- NovelMean.

---

## FIG04 — A→Mixed and B→Mixed convergence

For each intervention position, show lag-wise:

- normalized A/B branch distance;
- each branch's task midpoint residual;
- \(\alpha\) coordinate;
- off-axis residual.

Include natural A, natural B, and natural Mixed references.

This is a primary paper-candidate figure.

---

## FIG05 — Raw transition CE after A→Mixed and B→Mixed takeover

Show full 7-transition raw CE profiles for representative intervention points:

- early, e.g. \(k=1\);
- middle, e.g. \(k=3\) or 4;
- late, e.g. \(k=5\).

Do not show only averaged rescue.

---

## FIG06 — NoContext cue takeover

For every takeover position:

- natural NoContext;
- single Mixed pulse;
- persistent Mixed takeover;
- persistent A takeover;
- persistent B takeover;
- natural Mixed reference.

Show full 7-transition raw CE.

---

## FIG07 — Novel cue takeover

Same structure as FIG06, with Novel1–4 retained and NovelMean shown only after within-seed averaging.

---

## FIG08 — Pulse versus persistent takeover

Lag-wise comparison for High:

- Mixed;
- NoContext;
- NovelMean;
- A source;
- B source.

Show:

- raw CE rescue;
- task midpoint residual;
- hidden midpoint residual.

---

## FIG09 — Geometry specificity

At matched prestates show the destination of:

- A cue;
- B cue;
- Mixed cue.

Plot endpoint coordinate \(\alpha\) and off-axis residual.

The expected ordering is:

- A cue → \(\alpha\) near 0;
- Mixed cue → \(\alpha\) near .5;
- B cue → \(\alpha\) near 1.

Do not hard-code this as a pass criterion if natural endpoint references are imperfect; report actual distances as well.

---

## FIG10 — Per-seed outcome matrix

Rows = High seeds.

Columns:

- A→Mixed converges;
- B→Mixed converges;
- NoContext local generation;
- NoContext persistent rescue;
- Novel local generation;
- Novel persistent rescue;
- Mixed pulse < takeover;
- direction specificity.

This is descriptive. Do not create arbitrary green/red thresholds after seeing the data.

If thresholds are needed, define them before evaluation from the metrics below.

---

# 15. Required raw sequence-level visualizations

Because previous frozen-checkpoint plots were misleading, raw-loss provenance is mandatory.

## 15.1 Training sequence time series

For canonical High and Low:

- actual external sequence index;
- raw mean CE over seven transitions;
- A/B identity;
- values taken pre-update;
- model hash must change during training;
- final hash must equal registered checkpoint.

## 15.2 Probe sequence profiles

Every probe/cue intervention must retain the **seven raw transition losses**.

Do not only save:

- full-sequence mean;
- rescue score;
- normalized residual.

The report must include raw transition-level profiles for:

- A;
- B;
- Mixed;
- NoContext;
- each Novel;
- cue-pulse conditions;
- persistent cue-takeover conditions;
- A/B endpoint controls.

---

# 16. Statistics

Seed is the unit of inference.

## 16.1 Bootstrap

Use paired bootstrap resampling over seeds.

For Novel:

1. calculate Novel1–4 metrics within seed;
2. average to NovelMean;
3. bootstrap the seed-level NovelMean.

Never treat Novel variants or transitions as independent samples.

---

## 16.2 Primary comparisons

Pre-specify these:

### P1
High, A/B source states:

\[
R^z_{mid}(\text{native})
-
R^z_{mid}(\text{Mixed cue})
\]

local matched-prestate branch.

### P2
High, A→Mixed/B→Mixed:

normalized branch convergence after takeover.

### P3
High, NoContext:

persistent Mixed takeover remaining-sequence raw CE rescue.

### P4
High, NovelMean:

persistent Mixed takeover remaining-sequence raw CE rescue.

### P5
High:

persistent takeover minus single pulse remaining-sequence rescue.

### P6
High:

Mixed-cue midpoint specificity versus A/B endpoint cues in task space.

Report paired seed bootstrap mean/median and 95% CI.

Do not reduce the result to significance tests alone.

---

# 17. Decision table

Freeze this qualitative table before looking at results.

## Outcome A — Full cue-generated compositional routing

Evidence:

- Mixed cue locally moves A and B states toward midpoint;
- A→Mixed and B→Mixed converge toward the same intermediate trajectory;
- persistent Mixed takeover rescues NoContext and Novel at least directionally and geometrically;
- persistent takeover outlasts a one-step pulse;
- A and B cue controls route toward their respective endpoints.

Interpretation:

> Context supplies a transformation of shared sequence structure. Familiar A/B cues generate separated trajectories, while their overlap can redirect those trajectories toward a shared intermediate state.

This is the desired stopping outcome.

---

## Outcome B — Mixed explains averaging, Novel/NoContext remain generation failures

Evidence:

- A→Mixed and B→Mixed cleanly converge;
- Mixed natural trajectory is recovered;
- NoContext and/or Novel do not move sufficiently toward the composite under cue takeover.

Interpretation:

> The mechanism explains the core A/B-overlap averaging phenomenon. Novel and context removal fail because their starting states are not naturally routed into the familiar composite trajectory.

This is still sufficient to stop the main SRN program.

Novel/NoContext become a clearly localized generalization boundary, not a reason to reopen the core mechanism.

---

## Outcome C — Cue points correctly but recurrence blocks convergence

Evidence:

- one-step Mixed branch points toward midpoint;
- persistent takeover fails to converge or rescue.

Interpretation:

> The correct contextual transformation exists locally, but recurrent dynamics prevent the state from reaching or maintaining the composite trajectory from those histories.

This localizes the remaining failure to dynamics.

---

## Outcome D — State patch works but cue cannot generate it

Evidence:

- previous direct midpoint/composite patch rescues;
- Mixed cue does not move source states toward midpoint.

Interpretation:

> The composite state is sufficient but is not naturally generated by the physical overlapping cue from the tested states.

Do not claim natural contextual composition.

---

## Outcome E — Nonspecific cue effects

Evidence:

- A/B/Mixed/zero cues produce similar midpoint movement;
- or behavioral rescue occurs without geometric specificity.

Interpretation:

> Cue takeover acts as a generic perturbation rather than a compositional transformation.

This would weaken the proposed mechanism.

---

# 18. Stop rule

This experiment is intended to end the SRN exploration.

After Phase 1K:

- do not add new architectures;
- do not add larger models;
- do not add ecology/pretraining;
- do not add more cue families;
- do not optimize probe performance;
- do not train on Mixed/Novel/NoContext;
- do not add adapters.

If Outcome A or B occurs, move to manuscript writing.

If Outcome C or D occurs, report the mechanistic boundary and move to writing unless there is evidence of a code/provenance error.

Only reopen modeling if the result exposes a demonstrable implementation error.

---

# 19. Data products

Create:

```text
phase1k/
├── phase1k_provenance.csv
├── phase1k_natural_rollouts.csv
├── phase1k_local_cue_branches.csv
├── phase1k_takeover_transition_metrics.csv
├── phase1k_takeover_summary.csv
├── phase1k_pulse_transition_metrics.csv
├── phase1k_endpoint_control_metrics.csv
├── phase1k_ab_to_mixed_convergence.csv
├── phase1k_novel_within_seed.csv
├── phase1k_seed_summary.csv
├── phase1k_bootstrap_summary.csv
├── PHASE1K_CUE_GENERATION_REPORT.md
└── figures/
```

---

# 20. Minimum columns

## `phase1k_local_cue_branches.csv`

```text
family
seed
source_probe
novel_id
transition
branch_cue
prestate_hash
hidden_state_hash
raw_ce
hidden_midpoint_residual
task_midpoint_residual
hidden_direction_cosine
task_direction_cosine
distance_to_A_hidden
distance_to_B_hidden
distance_to_A_task
distance_to_B_task
alpha_task
off_axis_task
```

---

## `phase1k_takeover_transition_metrics.csv`

```text
family
seed
source_probe
novel_id
takeover_transition
intervention
transition
lag
raw_ce
native_raw_ce
raw_ce_rescue
hidden_midpoint_residual
task_midpoint_residual
alpha_task
off_axis_task
hidden_state_hash
parameter_hash
```

---

## `phase1k_seed_summary.csv`

At minimum:

```text
family
seed
A_to_M_local_task_generation
B_to_M_local_task_generation
A_to_M_final_convergence
B_to_M_final_convergence
AB_to_M_mean_convergence
NoContext_local_task_generation
NoContext_takeover_remaining_rescue
NovelMean_local_task_generation
NovelMean_takeover_remaining_rescue
pulse_vs_takeover_difference
mixed_midpoint_specificity
A_endpoint_specificity
B_endpoint_specificity
```

---

# 21. TDD / RED → GREEN requirements

Write tests before implementing the scientific runner.

## Test 1 — Checkpoint identity

RED:
analysis loads a checkpoint not matching the registered canonical source.

GREEN:
all 20 High and 20 Low checkpoint hashes match the existing registry.

---

## Test 2 — Architecture identity

Assert:

- hidden size = 8;
- tanh recurrence;
- High init scale metadata = 2.0;
- Low init scale metadata = 0.02.

---

## Test 3 — Frozen parameters

Hash parameters before and after every intervention batch.

Assert exact equality.

---

## Test 4 — No optimizer/backward path

Scientific runner must not instantiate an optimizer or call backward.

A static test may inspect the runner module if useful.

---

## Test 5 — Exact physical Mixed cue

Assert:

\[
c_M = (c_A+c_B)/2
\]

to numerical tolerance already used in Phase 1J.

---

## Test 6 — Natural-history identity

For takeover at \(k\), assert that all hidden states through \(k-1\) are bit-identical to the source natural rollout.

---

## Test 7 — Correct intervention index

For takeover at \(k\):

- cue at \(k-1\) is native;
- cue at \(k\) is Mixed;
- output at transition \(k\) reflects the first altered hidden state.

---

## Test 8 — Pulse duration exactly one

For `single_mixed_pulse@k`, assert:

- Mixed only at \(k\);
- native cue restored at \(k+1\).

---

## Test 9 — Persistent takeover

For `persistent_mixed_takeover@k`, assert Mixed is used for all:

\[
t \ge k.
\]

---

## Test 10 — Matched-prestate branch identity

In Phase A, every cue branch at a given source/seed/transition must begin from an identical prestate hash.

---

## Test 11 — Familiar references frozen

A/B midpoint references must be computed before any intervention evaluation and must never depend on source probe behavior.

---

## Test 12 — No probability averaging

Search the scientific runner and target builder for any averaging of softmax probabilities.

Fail if found.

---

## Test 13 — Centered-logit implementation

Numerically confirm centered logits have zero mean across output classes.

---

## Test 14 — Endpoint coordinate sanity

Synthetic unit test:

- \(z=z_A \Rightarrow \alpha=0\);
- \(z=z_B \Rightarrow \alpha=1\);
- \(z=(z_A+z_B)/2 \Rightarrow \alpha=.5\);
- off-axis residual = 0 in all three cases.

---

## Test 15 — Novel nesting

Assert Novel1–4 are averaged within seed before any bootstrap or family statistic.

---

## Test 16 — Raw CE preservation

For every intervention rollout assert exactly seven raw transition CE values exist.

---

## Test 17 — Sequence loss definition

If a sequence-level scalar is calculated, assert:

\[
L_{seq}
=
\frac{1}{7}
\sum_{k=1}^{7} CE_k.
\]

---

## Test 18 — Acquisition provenance

Replay canonical training independently and assert the final parameter hash matches the registered final checkpoint for every seed.

---

## Test 19 — Pre-update acquisition loss

The raw acquisition time series must be logged before each optimizer update, not from a final checkpoint replay.

---

## Test 20 — Figure completeness

Automated test must fail if the report is missing:

- canonical raw acquisition plot;
- natural probe raw transition profiles;
- A→Mixed/B→Mixed convergence plot;
- A→Mixed/B→Mixed raw CE profiles;
- NoContext takeover raw CE plot;
- Novel takeover raw CE plot;
- pulse-versus-takeover plot;
- endpoint-specificity plot;
- per-seed summary.

---

# 22. Implementation order

## Step 1
Lock provenance and checkpoint registry.

## Step 2
Implement natural rollout cache and frozen familiar references.

## Step 3
Implement Phase A one-step matched-prestate cue branching.

## Step 4
Implement persistent Mixed takeover.

## Step 5
Implement single Mixed pulse.

## Step 6
Implement A/B endpoint takeover controls.

## Step 7
Implement transition-level geometry metrics.

## Step 8
Implement A→Mixed/B→Mixed convergence metrics.

## Step 9
Export raw transition-level data.

## Step 10
Run all 20 High and 20 Low seeds.

## Step 11
Bootstrap seeds.

## Step 12
Generate mandatory figures and report.

Do not look at aggregate results to tune intervention definitions between Steps 3 and 10.

---

# 23. Report structure

`PHASE1K_CUE_GENERATION_REPORT.md`

Use:

```text
1. Executive summary
2. Provenance audit
3. Natural familiar and probe behavior
4. Local cue-generation test
5. A→Mixed and B→Mixed convergence
6. NoContext cue takeover
7. Novel cue takeover
8. One-step pulse versus persistent takeover
9. Endpoint-direction controls
10. High-versus-Low comparison
11. Per-seed heterogeneity
12. Frozen decision-table outcome
13. Mechanistic interpretation
14. Limitations
15. Final stopping decision
```

---

# 24. Interpretation language if the result succeeds

If Outcome A:

> The network does not need to erase its context-specific A and B representations to produce intermediate behavior. Instead, the same sequence structure can occupy strongly separated context-dependent trajectories, while the overlapping A+B cue causally redirects either trajectory toward an intermediate task state. Thus context-dependent separation and compositional generalization coexist because context controls the transformation applied to shared sequence structure.

If Outcome B:

> The overlapping A+B cue is sufficient to redirect familiar A and B trajectories toward a common intermediate solution, explaining the core averaging phenomenon despite strong context-specific separation. Novel and context-absent probes fail at the generation stage: their natural states are not routed into this familiar composite trajectory.

Do not claim that humans literally implement the SRN mechanism.

The human result establishes the behavioral computational constraint; the SRN establishes a concrete sufficient mechanism capable of satisfying that constraint.

---

# 25. What would count as “done”

Phase 1K is complete when:

1. all provenance/TDD tests pass;
2. all 20 High and 20 Low seeds run with frozen parameters;
3. A→Mixed and B→Mixed cue-only interventions are evaluated;
4. Novel1–4 and NoContext cue-only generation are evaluated;
5. pulse and persistent takeover are separated;
6. raw seven-transition CE profiles are retained and plotted;
7. hidden and task-space geometry are both reported;
8. endpoint-direction controls are complete;
9. the frozen outcome table is assigned;
10. the report explicitly says whether the core A/B averaging mechanism is explained.

At that point, begin manuscript writing rather than adding another modeling phase.
