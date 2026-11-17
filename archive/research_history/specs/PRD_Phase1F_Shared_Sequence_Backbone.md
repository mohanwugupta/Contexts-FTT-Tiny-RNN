# PRD — Phase 1F: Shared Sequence Backbone and Context-Modulation Decomposition

## 1. Goal

Test the hypothesis that human-like transfer arises because the learner preserves a **shared sequence representation** and represents familiar contexts A and B as context-specific modulations around that shared sequence backbone.

The proposed representational form is:

\[
z_{k,A} = s_k + d_k,
\]

\[
z_{k,B} = s_k - d_k,
\]

where:

- \(k\) indexes sequence position / transition,
- \(s_k\) is the shared sequence representation,
- \(d_k\) is the context-specific modulation.

This implies:

\[
s_k = \frac{z_{k,A}+z_{k,B}}{2},
\]

\[
d_k = \frac{z_{k,A}-z_{k,B}}{2}.
\]

For a probe \(P\in\{\text{Mixed, Novel, NoContext}\}\), decompose:

\[
z_{k,P}
=
s_k
+
\alpha_{k,P}d_k
+
r^\perp_{k,P}.
\]

Here:

- \(\alpha=-1\) corresponds to B,
- \(\alpha=+1\) corresponds to A,
- \(\alpha=0\) corresponds to the shared sequence state,
- \(r^\perp\) is the component not explained by the familiar A/B sequence axis.

The central hypothesis is:

> Human-like transfer occurs when probe states remain close to the familiar shared sequence manifold, with context acting primarily as a modulation around that shared backbone.

This Phase 1F should determine whether the High-scale SRNs fail Novel and NoContext because they do not preserve or express this shared sequence backbone strongly enough.

---

# 2. Canonical model and provenance

Use the audited canonical SRNs only.

## High condition

```text
architecture: tanh Elman SRN
hidden_size: 8
initialization_scale: 2.0
```

## Low condition

```text
architecture: tanh Elman SRN
hidden_size: 8
initialization_scale: 0.02
```

Use the exact 20 canonical High and 20 canonical Low seeds verified by the training-timeline audit.

Before analysis, hard assert:

```python
assert hidden_size == 8
assert high_init_scale == 2.0
assert low_init_scale == 0.02
```

Record:

```text
seed
checkpoint_path
checkpoint_hash
hidden_size
init_scale
training_schedule_hash
git_commit
```

Do not use any older 6-unit or scale-0.5 checkpoints.

---

# 3. Scientific questions

## Q1. Does a shared A/B sequence backbone exist?

For each transition \(k\), compute the A/B midpoint \(s_k\) and A/B modulation vector \(d_k\).

Ask whether the familiar sequence states can be described well as:

\[
s_k \pm d_k.
\]

This is partly definitional for the two endpoints, but the important question is whether:

- the same \(s_k\) is stable and sequence-specific,
- probe states align with that shared structure,
- and the structure is behaviorally useful.

---

## Q2. Where do Mixed, Novel, and NoContext fall relative to that backbone?

For each probe state, estimate:

\[
\alpha_{k,P}
=
\frac{
(z_{k,P}-s_k)^\top d_k
}{
d_k^\top d_k+\epsilon
}
\]

and:

\[
r^\perp_{k,P}
=
z_{k,P}
-
s_k
-
\alpha_{k,P}d_k.
\]

Report:

\[
\|r^\perp_{k,P}\|
\]

and normalized off-axis residual:

\[
R^\perp_{k,P}
=
\frac{
\|r^\perp_{k,P}\|
}{
\|d_k\|+\epsilon
}.
\]

The key descriptive question is whether Mixed is already relatively aligned with the familiar A/B sequence axis while Novel and NoContext lie farther off it.

---

## Q3. Is the shared sequence component causally sufficient?

Patch Novel and NoContext toward the shared sequence representation while leaving the rest of the network unchanged.

If this improves raw sequence loss, then the shared sequence backbone is behaviorally useful rather than merely geometric.

---

## Q4. Is the effect sequence-specific?

A midpoint between A and B is only theoretically meaningful if it corresponds to the **correct sequence position**.

Therefore, patches using \(s_k\) must outperform matched patches using the wrong sequence position \(s_j,\;j\neq k\).

This is mandatory.

---

## Q5. Do High and Low networks differ in how they preserve the shared sequence backbone?

The expected qualitative contrast is:

- Low: strong shared sequence representation, weak A/B contextual separation.
- High: strong A/B contextual separation, weaker or less accessible shared sequence representation.

Do not assume this result. Test it directly.

---

# 4. Representational spaces

Run the decomposition in two spaces.

## 4.1 Primary: centered task/logit space

Let:

\[
\ell_{k,c}
\]

be output logits.

Center them:

\[
C = I-\frac{1}{m}\mathbf 1\mathbf 1^\top,
\]

\[
z_{k,c}=C\ell_{k,c}.
\]

Use centered task/logit space as the **primary analysis** because it is directly tied to model behavior and avoids softmax-gauge ambiguity.

All primary \(\alpha\), off-axis residual, and causal task-space results should use \(z\).

---

## 4.2 Secondary: hidden-state space

Repeat the same decomposition on:

\[
h_{k,c}.
\]

Define:

\[
s^h_k = \frac{h_{k,A}+h_{k,B}}{2},
\]

\[
d^h_k = \frac{h_{k,A}-h_{k,B}}{2}.
\]

Then:

\[
h_{k,P}
=
s^h_k
+
\alpha^h_{k,P}d^h_k
+
r^{h,\perp}_{k,P}.
\]

Hidden-space analyses are secondary because output-null dimensions can move without affecting behavior.

---

# 5. Data extraction

For each seed and each of the seven analyzed transitions:

Evaluate frozen states for:

```text
A
B
Mixed
Novel
NoContextProbe
```

Use the canonical final trained checkpoint.

Save:

```text
seed
condition
transition
probe_type
hidden_state[8]
raw_logits
centered_logits
correct_token
raw_sequence_loss
correct_token_probability
execution_proxy
```

No parameters may update during this phase.

---

# 6. Phase A — Geometry of the shared sequence backbone

For every seed and transition \(k\):

\[
s_k=\frac{z_{k,A}+z_{k,B}}{2}
\]

\[
d_k=\frac{z_{k,A}-z_{k,B}}{2}.
\]

For each probe \(P\):

\[
\alpha_{k,P}
=
\frac{
(z_{k,P}-s_k)^\top d_k
}{
d_k^\top d_k+\epsilon
}
\]

\[
r^\perp_{k,P}
=
z_{k,P}-s_k-\alpha_{k,P}d_k.
\]

Save the following metrics:

```text
alpha
off_axis_norm
normalized_off_axis_norm
distance_to_shared
distance_to_A
distance_to_B
```

Also compute the fraction of probe displacement explained by the familiar A/B axis:

\[
E_{k,P}
=
1-
\frac{
\|r^\perp_{k,P}\|^2
}{
\|z_{k,P}-s_k\|^2+\epsilon
}.
\]

Interpret this only as a geometric descriptive statistic.

---

# 7. Sequence-specificity analysis

The shared representation should be tied to the correct sequence position.

For each true transition \(k\), compare:

\[
\|z_{k,P}-s_k\|
\]

against all wrong-position shared states:

\[
\|z_{k,P}-s_j\|,\quad j\neq k.
\]

Compute:

```text
correct_position_distance
mean_wrong_position_distance
nearest_wrong_position_distance
sequence_specificity_margin
```

where:

\[
M_{k,P}
=
\operatorname{mean}_{j\neq k}\|z_{k,P}-s_j\|
-
\|z_{k,P}-s_k\|.
\]

Positive \(M\) means the probe is closer to the shared state for the correct sequence position.

This is important because the theory is about **compositionality of sequence structure**, not merely geometric interpolation between two context endpoints.

---

# 8. Phase B — Causal task-space interventions

Run the following frozen interventions independently for Mixed, Novel, and NoContextProbe.

Do not train.

## B0 — Baseline

No intervention:

\[
z'_{k,P}=z_{k,P}.
\]

---

## B1 — Shared-midpoint patch

Replace the probe task state with:

\[
z'_{k,P}=s_k.
\]

This asks:

> If the probe expressed the shared sequence representation directly, would the learned readout produce good behavior?

This is a strong state-level sufficiency test.

---

## B2 — A/B-axis projection

Preserve the probe's own inferred contextual coordinate but remove off-axis structure:

\[
z'_{k,P}
=
s_k+\alpha_{k,P}d_k.
\]

This asks:

> Is the main problem that the probe leaves the familiar sequence/context manifold?

---

## B3 — Clamped-axis projection

Clamp:

\[
\tilde{\alpha}_{k,P}
=
\operatorname{clip}(\alpha_{k,P},-1,1),
\]

then:

\[
z'_{k,P}
=
s_k+\tilde{\alpha}_{k,P}d_k.
\]

This tests whether extreme extrapolation along the A/B axis contributes to poor transfer.

B2 is primary; B3 is diagnostic.

---

# 9. Phase C — Sequence-specific causal controls

For each of B1 and B2, run matched wrong-sequence-position controls.

## C1 — Wrong shared-state patch

For transition \(k\), patch using:

\[
s_j,\quad j\neq k.
\]

Use either:

- all six wrong positions and average, or
- a deterministic seed-local permutation fixed before evaluation.

Prefer all wrong positions if computationally cheap.

The correct \(s_k\) must outperform wrong \(s_j\) if the rescue is truly sequence-specific.

---

## C2 — Wrong A/B-axis projection

Use the wrong transition's basis:

\[
s_j,\ d_j.
\]

Project:

\[
z'_{k,P}
=
s_j+\alpha^{(j)}_{k,P}d_j.
\]

This is the stronger control.

---

## C3 — Displacement-matched orthogonal control

Construct a perturbation:

\[
\Delta z_{\text{wrong}}
\perp
\Delta z_{\text{correct}}
\]

with:

\[
\|\Delta z_{\text{wrong}}\|
=
\|\Delta z_{\text{correct}}\|.
\]

Apply to the same probe state.

This tests whether improvement is specific to the shared-sequence direction rather than generic movement in task space.

---

# 10. Phase D — Hidden-state causal interventions

Task-space patches establish representational sufficiency but bypass the recurrent computation.

Therefore repeat the critical interventions in hidden space.

For each transition:

\[
s^h_k
=
\frac{h_{k,A}+h_{k,B}}{2}
\]

\[
d^h_k
=
\frac{h_{k,A}-h_{k,B}}{2}.
\]

Run:

```text
D0 baseline
D1 shared hidden-state patch
D2 hidden A/B-axis projection
D3 wrong-sequence shared-state control
D4 displacement-matched orthogonal control
```

After patching hidden state, allow the canonical recurrent/output computation to continue normally.

Do not patch logits afterward.

---

# 11. Two hidden-patch modes

Use both.

## 11.1 Transition-wise patch

At each analyzed transition \(k\), replace the current hidden state and evaluate the immediate next-token prediction.

This is the clean local causal test.

---

## 11.2 Single-entry patch with free rollout

At the first informative transition, patch:

\[
h\rightarrow s^h_k
\]

or the axis-projected equivalent, then allow the recurrent network to evolve without further intervention.

This asks whether entering the shared sequence state is sufficient for the network to sustain appropriate sequence execution.

Keep these two modes separate in all plots and tables.

---

# 12. Raw sequence loss is the primary behavioral metric

For all baseline and causal conditions, compute the actual model sequence loss.

Do not normalize it in the primary figures.

For sequence \(s\):

\[
L_s
=
-\sum_t \log p_\theta(y_{s,t}\mid y_{s,<t},c_s)
\]

or use the repository's canonical raw sequence-loss definition if it averages rather than sums.

The plotting function must call the same canonical loss implementation used by training/evaluation.

Also save:

```text
mean_token_loss
correct_token_probability
sequence_accuracy
execution_proxy
```

but raw loss is primary.

---

# 13. Required probe-vs-A/B raw-loss figures

For every intervention condition, show:

```text
Context A
Context B
Mixed
Novel
NoContextProbe
```

on the same raw-loss scale.

The figure must make it obvious whether a probe lies:

```text
below A/B
between A and B
above both
```

Do not plot only normalized ratios.

---

# 14. Required sequence-level time-series figures

These are mandatory.

## TS1 — True canonical acquisition

Use the audited replay trajectory.

x-axis:

```text
actual training sequence index
```

y-axis:

```text
raw model sequence loss
```

Show A/B learning over all 170 training sequences.

This is context, not part of the causal patch.

---

## TS2 — Frozen final-checkpoint baseline probe comparison

At the final checkpoint, show raw losses for:

```text
A
B
Mixed
Novel
NoContextProbe
```

across all seven sequence transitions / full sequence.

---

## TS3 — Shared-midpoint patch

Show full raw sequence loss after B1/D1 for:

```text
Mixed
Novel
NoContextProbe
```

with A and B overlaid as reference.

---

## TS4 — A/B-axis projection

Same format for B2/D2.

---

## TS5 — Correct vs wrong sequence patch

Plot full sequence-level raw loss for:

```text
baseline
correct-sequence shared patch
wrong-sequence shared patch
```

separately for Mixed, Novel, and NoContextProbe.

---

# 15. Primary outcome variables

For each seed and probe, calculate:

## Raw rescue

\[
\Delta L_P
=
L_P^{baseline}
-
L_P^{patch}.
\]

Positive means improvement.

---

## Familiar-reference placement

Let:

\[
L_{\min AB}
=
\min(L_A,L_B)
\]

and:

\[
L_{\max AB}
=
\max(L_A,L_B).
\]

Classify probe loss as:

```text
below familiar range
inside familiar A/B range
above familiar range
```

This is descriptive only.

Do not redefine success based on this classification after seeing results.

---

## Sequence-specific rescue

\[
S_P
=
\Delta L_P^{correct\ sequence}
-
\Delta L_P^{wrong\ sequence}.
\]

Positive values support the shared-sequence interpretation.

---

# 16. High-vs-Low comparison

For each metric compare High and Low:

```text
A/B separation
distance to shared backbone
alpha
off-axis residual
sequence-specificity margin
shared-patch rescue
axis-projection rescue
```

The important possible pattern is:

```text
Low:
    lower A/B separation
    lower probe off-axis residual
    better baseline transfer

High:
    higher A/B separation
    higher probe off-axis residual
    worse baseline Novel/NoContext transfer
```

Do not assume this pattern.

Report what the data show.

---

# 17. Critical predictions

## Prediction 1 — Mixed

Mixed should already align relatively well with the A/B familiar task axis.

Expected:

\[
R^\perp_{Mixed}
<
R^\perp_{Novel}
\]

and/or

\[
R^\perp_{Mixed}
<
R^\perp_{NoContext}.
\]

This would help explain why Phase 1D/1E improved Mixed preferentially.

---

## Prediction 2 — Novel and NoContext

Novel and NoContext should either:

1. contain larger off-axis components, or
2. fail to express the correct sequence-specific \(s_k\), or
3. both.

---

## Prediction 3 — Shared-state rescue

If the shared sequence backbone is sufficient:

\[
L_{Novel}^{shared\ patch}
<
L_{Novel}^{baseline}
\]

and:

\[
L_{NoContext}^{shared\ patch}
<
L_{NoContext}^{baseline}.
\]

---

## Prediction 4 — Correct sequence position matters

The correct shared state \(s_k\) should rescue more than \(s_j\), \(j\neq k\).

This is essential.

Without sequence specificity, a midpoint rescue would not establish a sequence-compositional mechanism.

---

# 18. Decision tree

## Outcome A — Strong support

Findings:

- probes are organized relative to \(s_k,d_k\),
- correct shared-state / axis patches improve Novel and NoContext,
- correct sequence-position patches beat wrong-position controls,
- matched orthogonal perturbations do not reproduce rescue.

Interpretation:

> The SRN contains a behaviorally meaningful shared sequence representation around which familiar contexts are differentiated. Novel and absent-context failures arise because those probes fail to access that shared sequence backbone appropriately.

This directly motivates a later training intervention that preserves shared sequence structure during contextual splitting.

---

## Outcome B — Task-space rescue but hidden-space rescue fails

Interpretation:

> A suitable shared task-space solution exists, but the recurrent network does not contain an accessible hidden-state realization of it.

Then the next problem is representational implementation.

---

## Outcome C — Hidden shared-state rescue works only transition-wise

Interpretation:

> The shared state is locally sufficient, but recurrent dynamics do not preserve it over sequence execution.

Then the next experiment should target recurrent dynamics, not cue encoding.

---

## Outcome D — Shared-state patch does not rescue Novel/NoContext

Interpretation:

> The A/B midpoint is not the missing computational ingredient.

Do not proceed to shared-backbone training regularization.

---

# 19. Optional Stage 2 — disabled by default

Do **not** automatically train new models.

Only enable Stage 2 if Phase 1F provides causal evidence that the shared sequence backbone rescues Novel and/or NoContext.

The later training question would be:

> Can the network learn strong A/B contextual differentiation while retaining an independently usable shared sequence backbone?

Possible minimal interventions could include:

- context dropout,
- shared-sequence auxiliary consistency,
- factorized context modulation,
- regularization preserving sequence-position geometry across contexts.

But none should be run in Phase 1F.

Phase 1F is first a causal/mechanistic localization experiment.

---

# 20. Mandatory figures

Produce at least:

```text
PHASE1F_FIG01_probe_alpha_by_condition.png
PHASE1F_FIG02_off_axis_residual_by_probe.png
PHASE1F_FIG03_sequence_specificity_margin.png
PHASE1F_FIG04_shared_patch_rescue.png
PHASE1F_FIG05_axis_projection_rescue.png
PHASE1F_FIG06_correct_vs_wrong_sequence_patch.png
PHASE1F_FIG07_correct_vs_orthogonal_control.png
PHASE1F_FIG08_high_vs_low_geometry.png
PHASE1F_FIG09_hidden_vs_task_space_rescue.png
```

And sequence/raw-loss figures:

```text
PHASE1F_TS01_true_A_B_acquisition_raw_loss.png
PHASE1F_TS02_baseline_A_B_Mixed_Novel_NoContext_raw_loss.png
PHASE1F_TS03_shared_patch_raw_loss.png
PHASE1F_TS04_axis_projection_raw_loss.png
PHASE1F_TS05_correct_vs_wrong_sequence_raw_loss.png
```

No normalized-loss plot may replace these raw-loss figures.

---

# 21. Required CSVs

```text
PHASE1F_task_geometry.csv
PHASE1F_hidden_geometry.csv
PHASE1F_task_patch_results.csv
PHASE1F_hidden_patch_results.csv
PHASE1F_sequence_specificity.csv
PHASE1F_raw_sequence_losses.csv
PHASE1F_seed_summary.csv
```

Each row must include:

```text
seed
scale_condition
probe
transition
intervention
raw_loss
alpha
off_axis_norm
sequence_position_source
checkpoint_hash
```

where applicable.

---

# 22. Required report

Create:

```text
PHASE1F_SHARED_SEQUENCE_BACKBONE_REPORT.md
```

Sections:

1. Executive result
2. Canonical checkpoint audit
3. Formal shared-backbone hypothesis
4. Task-space geometry
5. Hidden-state geometry
6. Mixed geometry
7. Novel geometry
8. NoContext geometry
9. Sequence-specificity analysis
10. Task-space causal patches
11. Hidden-state causal patches
12. Correct-sequence controls
13. Orthogonal controls
14. High-vs-Low comparison
15. Full raw-loss time series
16. Decision-tree classification
17. Implications for later training interventions

The report must distinguish:

```text
descriptive geometry
state-level sufficiency
hidden-state sufficiency
sequence specificity
learnability
```

Do not conflate them.

---

# 23. RED → GREEN tests

Write tests before the full run.

## RED 1 — Canonical model guard

Attempt to load:

```text
hidden_size = 6
```

Expected:

```text
FAIL: NONCANONICAL_SRN
```

---

## RED 2 — Centered-logit identity

Verify:

\[
\mathbf 1^\top z = 0
\]

for every centered logit vector.

---

## RED 3 — A/B decomposition reconstructs endpoints

Assert:

\[
s_k+d_k=z_{k,A}
\]

and:

\[
s_k-d_k=z_{k,B}
\]

within numerical tolerance.

---

## RED 4 — Projection residual orthogonality

For every probe:

\[
d_k^\top r^\perp_{k,P}\approx0.
\]

---

## RED 5 — Correct-sequence control integrity

Wrong-position patch must never use \(j=k\).

Fail if it does.

---

## RED 6 — Displacement matching

For the orthogonal control:

\[
|\|\Delta z_{wrong}\|-\|\Delta z_{correct}\||<\epsilon.
\]

and:

\[
|\Delta z_{wrong}^\top\Delta z_{correct}|<\epsilon.
\]

---

## RED 7 — Frozen causal evaluation

All patches must leave model parameters unchanged.

Hash before and after every intervention batch.

---

## RED 8 — Raw-loss identity

The value exported to plotting must equal the canonical model sequence-loss function exactly within numerical tolerance.

---

## RED 9 — Seven-transition completeness

Every seed × probe × intervention condition must contain exactly the seven registered transitions.

---

## RED 10 — Probe labels

Mixed, Novel, and NoContextProbe must remain distinct throughout all tables and plots.

No pooling across probe types.

---

# 24. Important guardrails

Do not force the data to support a midpoint story.

In particular:

- do not set Novel or NoContext to the midpoint during descriptive analysis;
- first measure where they actually are;
- midpoint/shared-state patching is a causal counterfactual test;
- report if the patch fails.

Do not interpret \(\alpha\approx0\) alone as evidence of successful transfer.

A probe may have:

\[
\alpha\approx0
\]

while also having a large:

\[
r^\perp,
\]

and therefore poor behavior.

The full decomposition matters.

---

# 25. Paper-level target claim if successful

If the geometry, causal rescue, and sequence-specific controls all support the hypothesis, the paper-level claim can be:

> Familiar contexts split a common sequence representation into context-specific variants rather than replacing the shared sequence structure. Ambiguous, novel, and absent-context probes can generalize when they access this shared sequence backbone. Human-like intermediate transfer therefore reflects compositional reuse of sequence structure together with graded context modulation.

Do not use this wording unless Phase 1F supports it.

---

# 26. Implementation order

Implement in this order:

1. provenance + canonical checkpoint assertions;
2. task-space A/B decomposition;
3. hidden-state A/B decomposition;
4. probe \(\alpha\) and off-axis metrics;
5. sequence-specificity diagnostics;
6. task-space shared-state and axis-projection patches;
7. wrong-sequence controls;
8. displacement-matched orthogonal controls;
9. hidden-state patches;
10. raw-loss sequence figures;
11. High-vs-Low comparison;
12. report generation.

Do not run optional Stage 2 training automatically.
