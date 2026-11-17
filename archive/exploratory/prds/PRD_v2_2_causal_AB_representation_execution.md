# PRD v2.2: Causal Test of the A/B Representational Subspace in FTT Execution Control

## 1. Purpose

The current High-Scale SRN result establishes three important prerequisites:

1. **A/B representations split.**
2. **The execution head is grounded in learned sequence confidence.**
3. **Mixed and Novel probes can produce intermediate execution behavior.**

The next goal is to determine whether these facts are causally connected.

The core mechanistic question is:

> **Does the context-specific A/B component of the SRN hidden state causally control execution speed, and does Mixed behavior arise because its hidden state occupies an intermediate position in that causally relevant representational subspace?**

This PRD adds **no new training objective**.

It operates on already-trained High-Scale SRNs and uses frozen-network interventions.

The desired causal chain is:

\[
\boxed{
\text{Context}
\rightarrow
\text{A/B-specific hidden representation}
\rightarrow
\text{execution readout}
\rightarrow
\text{speed}
}
\]

and, for Mixed:

\[
\boxed{
\text{Mixed cue}
\rightarrow
\text{intermediate A/B representation}
\rightarrow
\text{intermediate execution}
}
\]

---

# 2. Existing result to preserve

Use the trained v2.1 model unchanged:

```text
8-unit tanh Elman SRN
High-Scale initialization = 0.5
Low-Scale control = 0.02
4D sequence one-hot
3D context basis
A = [1,0,0]
B = [0,1,0]
Novel = [0,0,1]
Mixed_norm = [1/sqrt(2), 1/sqrt(2), 0]
10-sequence alternating A/B blocks
next-token head
execution-speed head
```

Do not retrain the model to make the intervention results cleaner.

Primary mechanistic analyses should use the registered High-Scale final-profile seeds that passed:

```text
representational splitting gate
execution grounding gate
```

The prior final run used 20 seeds and passed both Stage 1 and Stage 2 gates.

---

# 3. Primary checkpoint

The **human-matched checkpoint** is the primary checkpoint.

Reason:

- A/B hidden-state splitting is already established;
- A/B execution endpoints are still more often meaningfully separated;
- later in training, A/B execution speed can converge even while hidden representations remain split.

Secondary checkpoints:

```text
early_after_initial_ab
several_switches
final
```

The final checkpoint is mechanistically informative but should not be the primary behavioral interpolation test when A/B execution endpoints are too close.

---

# 4. Core representational object

For each trained network and each sequence position \(k\), collect pure-context hidden states:

\[
h_A^{(k)},\qquad h_B^{(k)}.
\]

Define the token-matched A/B difference vector:

\[
d_{AB}^{(k)}
=
h_A^{(k)}
-
h_B^{(k)}.
\]

Normalize:

\[
\hat d_{AB}^{(k)}
=
\frac{d_{AB}^{(k)}}{\|d_{AB}^{(k)}\|}.
\]

This is the simplest candidate **context-specific skill direction** for sequence position \(k\).

Do not begin by assuming that one global A/B direction applies to every sequence position.

---

# 5. Independent estimation and evaluation split

Directions used for intervention must be estimated independently from the trajectories on which their causal effects are evaluated.

For each seed:

## Estimation set

Use frozen pure A/B counterfactual runs with one deterministic hidden-state initialization / sequence pass.

Estimate:

```text
h_A(k)
h_B(k)
d_AB(k)
```

## Evaluation set

Run new frozen counterfactual passes using the same trained weights but separate forward passes.

Apply interventions only in the evaluation set.

This prevents intervention evaluation from reusing the exact hidden-state samples that defined the direction.

No optimizer steps are permitted.

---

# 6. Step 1: determine whether the A/B difference is one-dimensional or subspace-like

Before causal intervention, characterize the geometry of:

\[
\{d_{AB}^{(k)}\}_{k=1}^{K}.
\]

Stack token-position-specific A/B differences into matrix:

\[
D=
\begin{bmatrix}
d_{AB}^{(1)}\\
\vdots\\
d_{AB}^{(K)}
\end{bmatrix}.
\]

Run SVD / PCA on \(D\).

Report:

```text
variance explained PC1
variance explained PC1–2
variance explained PC1–3
pairwise cosine similarity across token-specific d_AB(k)
effective dimensionality
```

Decision:

### Approximately 1D

If one component explains most A/B difference structure, define a shared direction:

\[
d_{AB}^{shared}.
\]

### Multi-dimensional

If not, define the smallest A/B subspace \(U_{AB}\) explaining a prespecified proportion of variance.

Primary threshold:

```text
>= 90% variance explained
```

Do not force a 1D interpretation when the representation is genuinely multi-dimensional.

---

# 7. Primary causal test A: directional steering

Take an evaluation hidden state \(h_t\) at a fixed sequence position.

Apply:

\[
h_t'
=
h_t+\alpha \hat d_{AB}^{(k)}.
\]

Sweep:

```text
alpha = -2.0
        -1.5
        -1.0
        -0.5
         0
        +0.5
        +1.0
        +1.5
        +2.0
```

Use a scale normalized to the observed A/B separation at that sequence position.

A convenient implementation is:

\[
h_t'
=
h_t+
\alpha
\frac{\|h_A^{(k)}-h_B^{(k)}\|}{2}
\hat d_{AB}^{(k)}.
\]

Thus:

```text
alpha > 0 → more A-like
alpha < 0 → more B-like
```

---

# 8. Directional-steering outcomes

For every intervention record:

```text
execution speed v
normalized RT = 1-v
next-token logits
correct-token probability
next-token prediction
hidden-state norm
distance from natural hidden-state manifold
```

Primary prediction:

\[
\alpha
\rightarrow
v
\]

should be monotonic in the same direction as the natural A/B execution difference.

If:

\[
v_A>v_B,
\]

then:

\[
\alpha\uparrow
\Rightarrow
v\uparrow.
\]

If:

\[
v_A<v_B,
\]

the sign reverses.

Use a sign-normalized effect:

\[
v_t^{signed}
=
v_t
\cdot
sign(v_A-v_B)
\]

or equivalently orient \(d_{AB}\) so positive always points toward the faster pure-context state.

---

# 9. Directional-steering success criterion

For each seed:

1. intervention effect on execution speed is monotonic;
2. slope has the natural A→B sign;
3. moving toward A shifts behavior toward \(v_A\);
4. moving toward B shifts behavior toward \(v_B\);
5. next-token identity remains largely intact.

Primary group metric:

```text
fraction of seeds with correctly signed steering slope
mean standardized steering slope
R²(alpha, execution speed)
```

Suggested strong criterion:

```text
>= 16 / 20 seeds with correctly signed effect
```

Do not require exact endpoint recovery.

---

# 10. Primary causal test B: remove the A/B component

For a natural hidden state \(h_t\), remove its component in the A/B direction/subspace.

For 1D:

\[
h_t^{ablated}
=
h_t
-
\operatorname{proj}_{d_{AB}}(h_t-\mu),
\]

where:

\[
\mu
=
\frac{h_A^{(k)}+h_B^{(k)}}{2}.
\]

This removes only the signed A/B displacement from the local midpoint.

For a multi-dimensional A/B subspace \(U_{AB}\):

\[
h_t^{ablated}
=
h_t
-
U_{AB}U_{AB}^{T}(h_t-\mu).
\]

---

# 11. Ablation predictions

If the A/B representational component drives execution differences:

### Pure A/B

Removing that component should reduce:

\[
|v_A-v_B|.
\]

### Mixed

If Mixed's execution depends on its location in the A/B subspace, removing the A/B component should move it toward a common/default execution value.

### Sequence prediction

Because A and B contain the same sequence:

\[
\text{next-token accuracy}
\]

should be relatively preserved.

This is a particularly important dissociation.

The desired pattern is:

\[
\boxed{
\text{A/B execution difference decreases}
}
\]

while:

\[
\boxed{
\text{sequence prediction remains mostly intact}.
}
\]

---

# 12. Ablation success metrics

Compute before versus after intervention:

```text
A/B execution-speed separation
A/B normalized-RT separation
next-token accuracy
next-token CE
correct-token probability
```

Primary causal metric:

\[
C_{\text{exec}}
=
1-
\frac{
|v_A^{ablated}-v_B^{ablated}|
}{
|v_A-v_B|
}.
\]

Interpretation:

```text
0 = no causal effect
1 = complete elimination of A/B execution difference
```

Also compute sequence-preservation metric:

\[
C_{\text{seq}}
=
\frac{
Accuracy_{ablated}
}{
Accuracy_{natural}
}.
\]

The strongest result is:

```text
large execution collapse
minimal sequence-performance collapse
```

---

# 13. Primary causal test C: matched-state A/B component patching

This is the cleanest causal test.

For the same sequence position \(k\), decompose hidden state into:

\[
h
=
h_{\parallel}
+
h_{\perp},
\]

where \(h_{\parallel}\) is the A/B subspace component.

For a Mixed hidden state:

\[
h_M.
\]

Construct:

\[
h_{M\leftarrow A}
=
h_{M,\perp}
+
h_{A,\parallel}
\]

and:

\[
h_{M\leftarrow B}
=
h_{M,\perp}
+
h_{B,\parallel}.
\]

Everything outside the A/B representational component remains exactly the original Mixed state.

Only the A/B component is patched.

---

# 14. Patching predictions

The causal representational account predicts:

\[
v(h_{M\leftarrow A})
\rightarrow
v_A
\]

and:

\[
v(h_{M\leftarrow B})
\rightarrow
v_B.
\]

The same test can be run for Novel:

\[
h_{N\leftarrow A},
\qquad
h_{N\leftarrow B}.
\]

This asks whether Novel's execution behavior is sensitive to the same learned A/B representational component.

---

# 15. Why patching is especially clean in this task

A and B contain the same underlying motor sequence.

Therefore at matched token position:

```text
sequence identity is fixed
target next token is fixed
architecture is fixed
network weights are fixed
```

The intervention changes only the context-specific hidden-state component.

This substantially reduces the risk that an execution effect is caused by a change in sequence identity or motor target.

---

# 16. Readout-only versus recurrent-rollout interventions

Run two levels of intervention.

## Level 1 — Instantaneous readout intervention

Intervene on \(h_t\) immediately before the two output heads.

Measure:

```text
execution speed
next-token logits
```

Do not feed the modified hidden state back into recurrence.

Question:

> Is the A/B representational component directly sufficient to alter the current execution readout?

## Level 2 — Recurrent intervention

Intervene on \(h_t\), then use the modified hidden state as the recurrent state for the next sequence step.

Measure effects over:

```text
t
t+1
t+2
...
```

Question:

> Does manipulating the A/B representational state causally alter the subsequent trajectory and execution policy?

The instantaneous test is primary.

The recurrent test is stronger but more likely to move hidden states off-manifold.

---

# 17. Off-manifold control

Large hidden-state perturbations may generate uninterpretable states.

For each intervention, compute distance to the natural pure/probe hidden-state distribution.

Use at least:

```text
Mahalanobis distance
nearest-natural-state Euclidean distance
hidden-state norm
```

Primary causal conclusions should emphasize intervention magnitudes that remain within or near the natural hidden-state range.

Large-\(\alpha\) extrapolations are sensitivity analyses only.

---

# 18. Random-direction negative control

For every causal intervention, generate norm-matched random control directions.

Random directions must:

```text
match ||d_AB||
be orthogonal to d_AB when possible
use identical alpha sweep
```

Prediction:

\[
Effect(d_{AB})
>
Effect(d_{random}).
\]

This is required.

Without it, a speed change could simply reflect generic sensitivity of the execution head to hidden-state perturbation.

---

# 19. Sequence-direction negative control

Estimate a direction that distinguishes two sequence positions while averaging across context.

For example:

\[
d_{seq}^{(k,j)}
=
\bar h^{(k)}
-
\bar h^{(j)}.
\]

Use norm-matched interventions.

Prediction:

- sequence-direction perturbation may affect next-token prediction;
- A/B direction should disproportionately affect execution/context expression.

This provides a useful functional dissociation.

---

# 20. Execution-head geometry

The execution head is:

\[
v_t
=
\sigma(w_v^\top h_t+b_v).
\]

Compute:

\[
cos(w_v,d_{AB}^{(k)})
\]

and, for a subspace:

\[
\|U_{AB}^{T}w_v\|.
\]

This is descriptive, not by itself causal.

Use it to ask:

> How much of the execution readout lies inside the A/B skill subspace?

Report this alongside intervention effects.

Do not require large cosine alignment for the causal test to pass, because nonlinear recurrent geometry can make simple global alignment misleading.

---

# 21. Mixed representational mediation test

For natural A, B, and Mixed states at matched sequence positions:

1. compute projection onto A/B subspace;
2. compute execution speed;
3. test whether Mixed's position in the subspace predicts its execution speed.

Then use the intervention effects to establish causality.

The desired sequence is:

\[
\text{Mixed cue}
\rightarrow
\text{A/B subspace position}
\rightarrow
\text{execution speed}.
\]

Do not call this statistical mediation unless a formal mediation model is actually fit.

Use “representational pathway” as the default language.

---

# 22. Novel analysis

Novel should be treated separately from Mixed.

Novel has:

```text
[0,0,1]
```

with a neutral frozen novel input column.

Its natural hidden state may not lie on the A/B interpolation path.

Questions:

1. Does Novel naturally occupy the A/B subspace?
2. Is its execution speed predicted by A/B subspace position?
3. Does A/B component patching causally move Novel execution toward A/B endpoints?

Possible outcomes:

### Novel uses same A/B pathway

A/B patching strongly controls Novel execution.

### Novel uses a different default pathway

Novel's natural execution is intermediate, but A/B subspace intervention has weak causal influence.

Both are scientifically informative.

Do not force Mixed and Novel into one mechanism.

---

# 23. Checkpoint dependence

Run the complete causal suite at:

```text
human_matched       PRIMARY
early_after_initial_ab
several_switches
final
```

Primary interpretation should emphasize checkpoints where:

```text
splitting gate passed
execution grounding gate passed
A/B execution endpoints are reliable
```

If:

\[
|RT_A-RT_B|
\]

is below the existing reliability threshold, do not interpret interpolation-index effects at that checkpoint.

Representational interventions may still be reported descriptively.

---

# 24. Seed-level analysis

All directions/subspaces must be estimated **within each network**.

Do not average hidden states across separately initialized networks before defining causal directions.

For each seed export:

```text
seed
checkpoint
token_position
AB_distance
AB_subspace_dim
steering_slope
steering_R2
random_control_slope
ablation_execution_collapse
ablation_sequence_preservation
Mixed_patch_A_effect
Mixed_patch_B_effect
Novel_patch_A_effect
Novel_patch_B_effect
off_manifold_metrics
```

Group-level summaries are computed only after within-network causal effects are defined.

---

# 25. Main mechanistic success criteria

The strongest causal account requires all four:

## M1 — Representation

High-Scale networks contain a reliable A/B context-specific direction/subspace.

## M2 — Sufficiency

Moving hidden state along the A/B subspace systematically changes execution toward the corresponding pure-context policy.

## M3 — Necessity

Removing the A/B component reduces the A/B execution difference while relatively preserving sequence prediction.

## M4 — Probe mechanism

Replacing the A/B component of Mixed with A versus B components shifts Mixed execution toward the corresponding endpoint.

If M1–M4 pass:

\[
\boxed{
\text{the split A/B representation is causally involved in execution control}
}
\]

and:

\[
\boxed{
\text{Mixed interpolation is generated through that representation}.
}
\]

---

# 26. Strong versus weak conclusions

## Strong conclusion

If M1–M4 pass:

> Context-specific hidden-state geometry is not merely correlated with execution behavior. The A/B representational component is both sufficient and partly necessary for the context-dependent execution policy, and Mixed behavior is causally controlled by its position in that representational subspace.

## Intermediate conclusion

If steering and patching work but ablation does not:

> The A/B subspace is sufficient to control execution but may be redundant with other hidden-state features.

## Weak conclusion

If natural Mixed states are intermediate but interventions fail:

> Mixed geometry and execution covary, but there is no evidence that the measured A/B subspace causally generates execution behavior.

## Negative conclusion

If random directions work as well as A/B directions:

> Execution is generically sensitive to hidden-state perturbation; the A/B representational interpretation is not supported.

---

# 27. TDD requirement

All implementation follows:

\[
\boxed{
RED\rightarrow GREEN\rightarrow REFACTOR
}
\]

---

# 28. RED → GREEN tests

## T1 — Frozen weights

Intervention pipeline must assert:

```text
no optimizer
no backward
no parameter mutation
```

during all causal analyses.

## T2 — Token matching

A/B directions must be computed only from matched sequence positions.

## T3 — Independent direction estimation

Evaluation trajectories must not be the exact forward-pass samples used to estimate intervention directions.

## T4 — Direction orientation

Positive A/B direction must consistently point toward A.

A separate sign-normalized execution orientation may be created for group analysis.

## T5 — Projection math

For a synthetic vector:

\[
h=\mu+a d+b r
\]

with \(r\perp d\), A/B ablation must remove only the \(ad\) component.

## T6 — Patching math

Mixed→A patch must replace only the A/B parallel component.

Orthogonal residual must remain numerically unchanged.

## T7 — Random control matching

Random-control direction norm must equal A/B-direction norm within tolerance.

## T8 — Random direction orthogonality

When hidden dimension permits, random control must be orthogonal to the A/B direction/subspace.

## T9 — No sequence-target change

A/B/Mixed/Novel matched-state interventions must retain the same next-token target.

## T10 — Instantaneous intervention isolation

Readout-only interventions must not alter the recurrent state stored for the next step.

## T11 — Recurrent intervention propagation

Recurrent-intervention mode must explicitly feed the modified state into the next recurrence.

## T12 — Off-manifold metrics

Every nonzero intervention must export off-manifold diagnostics.

## T13 — Zero intervention identity

At:

```text
alpha = 0
```

intervened outputs must equal natural outputs within numerical tolerance.

## T14 — Subspace dimensionality

If the 90%-variance threshold requires >1 dimension, pipeline must use the multi-dimensional subspace rather than silently forcing PC1.

## T15 — Probe gate preservation

Mixed/Novel causal interpretation must retain the existing splitting and execution-grounding gate status.

---

# 29. Proposed code structure

```text
ftt_srn_causal_representation/
    config.py

    load_models.py
    checkpoints.py

    representation/
        matched_states.py
        ab_directions.py
        ab_subspace.py
        dimensionality.py

    interventions/
        steering.py
        ablation.py
        patching.py
        recurrent_rollout.py

    controls/
        random_directions.py
        sequence_directions.py
        off_manifold.py

    analysis/
        execution_effects.py
        sequence_preservation.py
        probe_pathway.py
        execution_head_geometry.py
        seed_summary.py

    plotting/
        steering_curves.py
        ablation_figures.py
        patching_figures.py
        subspace_figures.py
        control_figures.py

    pipeline.py

tests/
    test_frozen_weights.py
    test_token_matching.py
    test_independent_estimation.py
    test_direction_orientation.py
    test_projection_math.py
    test_patching_math.py
    test_random_control.py
    test_target_identity.py
    test_readout_intervention.py
    test_recurrent_intervention.py
    test_off_manifold.py
    test_zero_intervention.py
    test_subspace_dimensionality.py
    test_gate_preservation.py
```

---

# 30. Required outputs

```text
causal_run_config.json

ab_directions.csv
ab_subspace_summary.csv
token_direction_similarity.csv

steering_results.csv
steering_seed_summary.csv

ablation_results.csv
ablation_seed_summary.csv

mixed_patching_results.csv
novel_patching_results.csv

random_direction_controls.csv
sequence_direction_controls.csv

off_manifold_metrics.csv

execution_head_alignment.csv
probe_representational_pathway.csv

causal_mechanism_summary.csv
```

---

# 31. Required figures

## Figure 1 — Causal hypothesis

```text
Context
  ↓
A/B hidden subspace
  ↓
execution head
  ↓
speed
```

with Mixed shown at an intermediate hidden-state location.

## Figure 2 — Token-specific A/B directions

Show A/B difference vectors across sequence positions and their pairwise cosine similarity.

## Figure 3 — A/B subspace dimensionality

Scree plot of variance explained by components of \(D\).

## Figure 4 — Steering curve

X:

```text
alpha along B→A direction
```

Y:

```text
execution speed
```

Overlay natural A/B endpoints.

## Figure 5 — Random-direction control

Compare execution effect from:

```text
A/B direction
random orthogonal direction
```

## Figure 6 — A/B component ablation

Before versus after:

```text
A/B execution separation
next-token accuracy
```

## Figure 7 — Mixed patching

Show:

```text
natural B
Mixed←B
natural Mixed
Mixed←A
natural A
```

for execution speed.

## Figure 8 — Novel patching

Same structure for Novel.

## Figure 9 — Sequence preservation

Plot execution change against next-token CE/accuracy change for interventions.

## Figure 10 — Checkpoint dependence

Show causal effect sizes at:

```text
early
human-matched
several switches
final
```

---

# 32. Primary decision logic

## Result A — Steering, ablation, and Mixed patching all succeed

Strong causal representational support.

Proceed to formalize the mechanism as:

\[
\boxed{
\text{split representation}
\rightarrow
\text{graded execution readout}
}
\]

## Result B — Steering and patching succeed, ablation fails

A/B representation is causally sufficient but not uniquely necessary.

Interpret as redundant coding.

## Result C — Natural Mixed interpolation exists but interventions fail

The observed A/B axis/subspace is descriptive, not causal.

Search for a different execution-relevant subspace.

## Result D — Random controls produce equivalent effects

Reject specificity of the A/B representational mechanism.

## Result E — Mixed uses A/B pathway, Novel does not

Mixed and Novel reach intermediate behavior through different mechanisms.

This is a scientifically acceptable and potentially important outcome.

## Result F — Both Mixed and Novel use the A/B pathway

Strongest general retrieval result.

---

# 33. Definition of done

- [ ] Existing trained v2.1 networks are loaded without retraining.
- [ ] Human-matched checkpoint is primary.
- [ ] Token-matched A/B directions are estimated within seed.
- [ ] Shared 1D versus multidimensional A/B structure is tested.
- [ ] Direction estimation and evaluation are independent.
- [ ] Directional steering is implemented.
- [ ] A/B-component ablation is implemented.
- [ ] Mixed A/B-component patching is implemented.
- [ ] Novel A/B-component patching is implemented.
- [ ] Readout-only interventions are primary.
- [ ] Recurrent-rollout interventions are secondary.
- [ ] Random orthogonal controls are implemented.
- [ ] Sequence-direction controls are implemented.
- [ ] Off-manifold diagnostics are exported.
- [ ] Sequence prediction is monitored during every intervention.
- [ ] All causal effects are computed within network before aggregation.
- [ ] Existing splitting and execution gates remain enforced.
- [ ] Final report distinguishes sufficiency, necessity, and correlation.
- [ ] Mixed and Novel are allowed to have different mechanisms.

---

# 34. Target mechanistic claim

The strongest supported claim would be:

> **High-Scale context learning creates a context-specific hidden-state subspace for the same motor sequence. The execution head functionally reads this subspace: shifting the hidden state toward the A or B representation causally shifts execution toward the corresponding context-specific policy, while removing that component selectively reduces the execution difference. Mixed-context behavior arises because its hidden state occupies an intermediate position in this causally relevant representational space.**

Compactly:

\[
\boxed{
\text{Context splits the hidden representation; position within that split space controls execution.}
}
