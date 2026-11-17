# PRD v2.3: Natural-State Causal Decomposition of Context × Sequence Execution Control

## 1. Purpose

The v2.2 causal analysis produced a specific pattern:

- A/B context-specific hidden structure was present in essentially all High-Scale networks.
- Removing the A/B component strongly reduced context-specific execution differences while largely preserving same-sequence prediction.
- Mixed and Novel A/B-component patching often shifted execution in the predicted direction.
- Simple continuous steering along a single A/B direction was much less reliable.

This suggests that the A/B representation is **causally important**, but that it is probably not a single scalar “speed axis.”

The refined hypothesis is:

> **Context configures the motor state, and execution speed is computed from the interaction between that context-configured state and the currently prepared sequence state.**

This PRD tests that hypothesis using **natural-state-constrained interventions** on the already-trained v2.1 High-Scale SRNs.

No new training objective is introduced.

No model weights are changed.

The main scientific question is:

\[
oxed{
	ext{Does execution depend on a context × sequence interaction rather than on context or sequence state alone?}
\]

---

## 2. Target representational story

The working psychological/computational account is:

\[
	ext{sequence position}
ightarrow
h_{	ext{sequence},k}
\]

\[
	ext{context}
ightarrow
h_{	ext{context},k}
\]

and:

\[
oxed{
[h_{	ext{sequence},k},
h_{	ext{context},k}]
ightarrow
	ext{execution policy}.
}
\]

The context-specific state is therefore not assumed to encode a literal RT or vigor value.

Instead, it configures the currently active motor representation.

The execution head reads the resulting configured state.

For Mixed:

\[
	ext{ambiguous context}
ightarrow
	ext{intermediate / recombined context component}
\]

while the underlying sequence state remains intact.

That interaction can yield intermediate execution.

---

## 3. Existing models and checkpoints

Load the registered v2.1 trained networks.

Primary model:

```text
8-unit tanh Elman SRN
High-Scale initialization = 0.5
4D sequence one-hot
3D context basis
A = [1,0,0]
B = [0,1,0]
Novel = [0,0,1]
Mixed_norm = [1/sqrt(2), 1/sqrt(2), 0]
next-token head
execution-speed head
```

Do not retrain.

Do not modify weights.

Primary checkpoint:

```text
human_matched
```

Secondary checkpoints:

```text
early_after_initial_ab
several_switches
final
```

The human-matched checkpoint remains primary because A/B splitting is established, execution grounding is established, and A/B execution endpoints remain more behaviorally informative than at asymptote.

---

## 4. Gate requirements

Only interpret causal results for seeds satisfying the existing v2.1 prerequisites:

```text
splitting_gate_passed = True
execution_grounding_gate_passed = True
```

This PRD does **not** re-open whether High Scale causes splitting. It asks what computation is performed **after splitting exists**.

---

## 5. Core decomposition

For each seed and token position \(k\), obtain frozen natural hidden states:

\[
h_A^{(k)},\qquad h_B^{(k)}.
\]

Define the local midpoint:

\[
\mu_k
=
rac{
h_A^{(k)}+h_B^{(k)}
}{2}.
\]

Define the local context-difference vector:

\[
d_k
=
rac{
h_A^{(k)}-h_B^{(k)}
}{2}.
\]

Thus:

\[
h_A^{(k)}
=
\mu_k+d_k
\]

and:

\[
h_B^{(k)}
=
\mu_k-d_k.
\]

Interpretation:

- \(\mu_k\): context-averaged sequence state at token \(k\);
- \(d_k\): context-specific configuration of that token state.

This is the primary local decomposition.

---

## 6. Shared versus token-specific context structure

The prior causal analysis indicated that A/B differences are low-dimensional but not perfectly one-dimensional.

Stack:

\[
d_1,\dots,d_K
\]

into matrix:

\[
D.
\]

Run SVD/PCA.

Define a **shared context component** using PC1:

\[
d_k^{shared}
=
\operatorname{proj}_{u_{	ext{shared}}}(d_k).
\]

Define the **token-specific context residual**:

\[
d_k^{specific}
=
d_k-d_k^{shared}.
\]

Therefore:

\[
d_k
=
d_k^{shared}
+
d_k^{specific}.
\]

If execution depends only on a generic context axis, the shared component should be sufficient.

If context configures the current sequence state in a token-specific way, the full token-matched component should outperform the shared component.

---

## 7. Nested mechanistic hypotheses

### H1 — Sequence-only account

\[
v=f(\mu_k).
\]

Context-specific differences are epiphenomenal.

### H2 — Shared-context-axis account

\[
v=f(\mu_k,d_k^{shared}).
\]

The same context axis is used across sequence positions.

### H3 — Configured-state interaction account

\[
v=f(\mu_k,d_k^{shared},d_k^{specific}).
\]

Equivalent interpretation:

> the effect of context depends on the currently active sequence state.

Predictions:

1. full token-matched patching outperforms shared-axis-only patching;
2. correct-token context components outperform wrong-token context components;
3. patch-effect magnitude varies across token positions in the same direction as natural token-specific A/B execution differences;
4. models containing context × sequence interaction terms outperform additive models.

---

## 8. Primary intervention A: natural-state token-matched patching

For each token \(k\), obtain natural A, B, Mixed_norm, and Novel hidden states.

For Mixed:

\[
h_M^{(k)}.
\]

Decompose relative to \(\mu_k\):

\[
h_M^{(k)}
=
\mu_k+c_M^{(k)}+r_M^{(k)},
\]

where:

\[
c_M^{(k)}
=
\operatorname{proj}_{d_k}
(h_M^{(k)}-\mu_k)
\]

and \(r_M^{(k)}\) is the residual orthogonal to \(d_k\).

Construct:

\[
h_{M\leftarrow A}^{(k)}
=
\mu_k+d_k+r_M^{(k)}
\]

and:

\[
h_{M\leftarrow B}^{(k)}
=
\mu_k-d_k+r_M^{(k)}.
\]

This keeps the entire Mixed residual fixed. Only the **naturally observed token-matched A/B context component** is replaced.

Repeat for Novel.

No arbitrary large-\(lpha\) extrapolation is used in the primary analysis.

---

## 9. Primary intervention B: shared-context-only patching

Construct the same patch using only:

\[
d_k^{shared}.
\]

Compare the recovered A/B execution contrast with full token-matched patching.

Define:

\[
R_{	ext{shared}}
=
rac{
|\Delta v_{	ext{shared patch}}|
}{
|\Delta v_{	ext{full patch}}|
}.
\]

If \(R_{	ext{shared}}pprox1\), a generic context axis is sufficient.

If substantially below 1, token-specific structure matters.

---

## 10. Primary intervention C: token-specific residual patching

Patch only:

\[
d_k^{specific}.
\]

Compare:

```text
shared only
specific only
full token-matched
```

Primary expected ordering under H3:

\[
Effect_{	ext{full}}
>
Effect_{	ext{shared}}
\]

with:

\[
Effect_{	ext{specific}}>0.
\]

Do not require the effects to add linearly.

---

## 11. Critical control: wrong-token context patching

At token \(k\), instead of using \(d_k\), patch in the context component from another token:

\[
d_j,\qquad j
eq k.
\]

Construct:

\[
h_{M\leftarrow A,j}^{(k)}
=
\mu_k+d_j+r_M^{(k)}
\]

after norm-matching \(d_j\) to \(d_k\).

Prediction:

\[
oxed{
Effect_{	ext{correct token}}
>
Effect_{	ext{wrong token}}.
}
\]

This is the strongest direct test that context does not merely provide a generic A/B label.

---

## 12. Wrong-token control construction

For each target token \(k\):

1. use every \(j
eq k\);
2. norm-match:
   \[
   	ilde d_j
   =
   d_j
   rac{\|d_k\|}{\|d_j\|};
   \]
3. patch into the same target hidden state;
4. average effect over all wrong-token \(j\).

Do not select a convenient wrong token.

---

## 13. Token-specific causal-effect prediction

For each token \(k\), calculate the natural pure-context execution difference:

\[
\Delta v_k^{natural}
=
v_A^{(k)}
-
v_B^{(k)}.
\]

Calculate the causal patch effect:

\[
\Delta v_k^{patch}
=
v_{M\leftarrow A}^{(k)}
-
v_{M\leftarrow B}^{(k)}.
\]

Primary structural prediction:

\[
oxed{
\Delta v_k^{natural}
\propto
\Delta v_k^{patch}.
}
\]

Report within-seed and group-level:

```text
correlation
slope
R²
```

---

## 14. Context × sequence statistical model comparison

Fit transparent predictive models of natural execution speed.

Use natural A/B states only for fitting.

Do not train on Mixed or Novel outcomes.

### Model S — sequence only

Predict execution from context-averaged sequence state:

\[
v=f_{	ext{seq}}(\mu_k).
\]

### Model C — context only

Predict execution from context coordinate:

\[
z_{ctx}
=
\langle h-\mu_k,u_{	ext{shared}}angle.
\]

### Model A — additive sequence + context

\[
v=f_{	ext{seq}}(\mu_k)+eta_{ctx}z_{ctx}.
\]

### Model I — sequence × context interaction

\[
v
=
f_{	ext{seq}}(\mu_k)
+
eta_{ctx}z_{ctx}
+
\sum_k
\gamma_k I(token=k)z_{ctx}.
\]

Equivalent transparent, capacity-controlled implementations are acceptable.

---

## 15. Model comparison

Use leave-one-seed-out evaluation where practical.

Report:

```text
held-out MSE
cross-validated R²
held-out log likelihood if probabilistic
AIC/BIC for descriptive within-dataset fits
```

Primary comparison:

\[
Model\ I
\quad vs\quad
Model\ A.
\]

Prediction:

\[
oxed{
Interaction model > additive model.
}
\]

Do not use an unconstrained neural network as the headline model comparison.

---

## 16. Mixed generalization test

After fitting on natural A/B execution only, predict Mixed execution **without fitting to Mixed**.

Compare:

```text
sequence-only
context-only
additive
interaction
```

The interaction model should best predict Mixed if the configured-state account is correct.

---

## 17. Novel generalization test

Repeat for Novel.

Do not assume the same mechanism as Mixed.

Novel may use:

- the A/B context subspace;
- a default state;
- a separate extrapolative mechanism.

Report Novel separately.

---

## 18. Necessity re-analysis with natural-state decomposition

Repeat A/B component removal, but separately remove:

```text
shared context component
token-specific context component
full token-matched context component
```

For each condition report:

\[
	ext{A/B execution collapse}
\]

and:

\[
	ext{same-sequence prediction preservation}.
\]

Prediction under H3:

\[
Collapse_{	ext{full}}
>
Collapse_{	ext{shared}}
\]

and:

\[
Collapse_{	ext{specific}}>0.
\]

Sequence performance should remain relatively intact.

---

## 19. Functional dissociation

For every intervention measure:

```text
execution-speed change
normalized RT change
next-token CE change
correct-token probability change
next-token identity change
```

The target pattern is:

\[
oxed{
	ext{large execution change}
+
	ext{small same-sequence prediction change}.
}
\]

---

## 20. Natural-state constraint

Primary analyses must use hidden-state components derived from actual frozen A/B states.

Do not use:

- arbitrary random scaling;
- large extrapolative steering;
- synthetic directions not observed in natural A/B differences.

The only permitted synthetic combination is recombination of natural components under explicitly defined controls.

---

## 21. Off-manifold checks

For every patched state export:

```text
distance to nearest natural A/B/Mixed/Novel state
Mahalanobis distance
hidden-state norm
```

Compare correct-token, wrong-token, shared-only, and full patches.

Main causal interpretation should not rely on extreme off-manifold states.

---

## 22. Random-direction control

Retain the v2.2 random orthogonal control.

Prediction:

\[
Effect_{AB}
>
Effect_{random}.
\]

This remains required for specificity.

---

## 23. Sequence-direction control

Retain a token-position direction control.

Prediction:

- sequence-direction intervention should affect next-token prediction more strongly;
- context-component intervention should disproportionately affect execution policy.

This provides the desired functional dissociation.

---

## 24. Seed-level analysis

All decompositions are estimated within network.

Never average hidden states across seeds before defining directions.

For every seed export:

```text
checkpoint
token_position
natural_A_speed
natural_B_speed

shared_context_norm
specific_context_norm

full_patch_effect
shared_patch_effect
specific_patch_effect
wrong_token_patch_effect
random_patch_effect

natural_vs_patch_token_correlation

full_ablation_collapse
shared_ablation_collapse
specific_ablation_collapse

sequence_CE_change
off_manifold_distance
```

---

## 25. Primary mechanistic success criteria

### M1 — Full natural patching works

Token-matched A/B patching reliably shifts Mixed execution toward the corresponding pure-context policy.

Target:

```text
>= 16 / 20 seeds correctly signed
```

at the human-matched checkpoint.

### M2 — Correct-token specificity

\[
Effect_{	ext{correct token}}
>
Effect_{	ext{wrong token}}
\]

at the group level and in a clear majority of seeds.

### M3 — Token-structured causal correspondence

Natural token-wise A/B execution differences correlate positively with token-wise patch effects.

### M4 — Full component beats shared-only

Full token-matched context patching explains more execution difference than the generic shared context axis alone.

### M5 — Necessity with sequence preservation

Removing the full context component collapses A/B execution differences while largely preserving same-sequence prediction.

### M6 — Interaction model advantage

Cross-validated sequence × context model predicts natural and Mixed execution better than the additive model.

---

## 26. Interpretation matrix

### Result A — M1–M6 pass

Strong support:

> **Context configures the currently prepared sequence state in a token-specific manner, and execution is computed from that configured motor state. Mixed behavior emerges by recombining context-specific components of the same sequence representation.**

### Result B — Full patching works, but shared-only is equally good

A generic context axis is sufficient.

### Result C — Full patching > shared, but wrong-token patching works equally well

Token-specific residual structure matters, but not in a sequence-position-specific way.

### Result D — Correct-token patching outperforms wrong-token patching

Key interaction result:

> The same nominal context has different causal representational effects depending on which motor-sequence state is active.

### Result E — Mixed passes; Novel fails

Mixed uses the learned A/B configured-state pathway; Novel uses a different mechanism.

### Result F — Neither natural-state patching nor interaction model improves over additive account

Retain the weaker necessity result from v2.2.

---

## 27. TDD requirement

All implementation follows:

\[
oxed{
REDightarrow GREENightarrow REFACTOR
}
\]

---

## 28. RED → GREEN tests

### T1 — Frozen-network contract

Assert no optimizer, backward pass, or parameter mutation.

### T2 — Exact token matching

Natural A/B components must be computed from the same token position.

### T3 — Local decomposition reconstruction

\[
\mu_k+d_k=h_A^{(k)}
\]

and:

\[
\mu_k-d_k=h_B^{(k)}
\]

within tolerance.

### T4 — Shared + specific reconstruction

\[
d_k^{shared}+d_k^{specific}=d_k.
\]

### T5 — Residual orthogonality

Token-specific residual is orthogonal to shared PC1 within tolerance.

### T6 — Patch preservation

Full patching preserves the target state's residual orthogonal to the replaced component.

### T7 — Wrong-token exclusion

Wrong-token controls never use \(j=k\).

### T8 — Wrong-token norm matching

Rescaled wrong-token context vector norm matches target-token vector norm.

### T9 — Identity patch

Replacing natural A's A component with itself reproduces the original state/output; same for B.

### T10 — No Mixed/Novel fitting

Predictive models may not use Mixed or Novel outcomes during fitting.

### T11 — Leave-one-seed-out split

No held-out seed rows appear in training.

### T12 — Sequence target invariant

Matched-state interventions retain the same next-token target.

### T13 — Random control matching

Random control norm matches the causal context component.

### T14 — Off-manifold export

Every synthetic state gets off-manifold diagnostics.

### T15 — Gate preservation

Every mechanistic row includes splitting/execution gate status.

---

## 29. Proposed code structure

```text
ftt_srn_context_sequence_interaction/
    config.py
    load_models.py

    representation/
        natural_states.py
        local_decomposition.py
        shared_context_axis.py
        token_specific_residuals.py

    interventions/
        full_context_patch.py
        shared_context_patch.py
        specific_context_patch.py
        wrong_token_patch.py
        context_ablation.py

    models/
        sequence_only.py
        context_only.py
        additive.py
        interaction.py
        cross_validation.py

    controls/
        random_direction.py
        sequence_direction.py
        off_manifold.py

    analysis/
        patch_effects.py
        token_structure.py
        necessity.py
        mixed_generalization.py
        novel_generalization.py
        model_comparison.py
        seed_summary.py

    plotting/
        decomposition_figures.py
        patching_figures.py
        wrong_token_figures.py
        token_effect_figures.py
        model_comparison_figures.py
        ablation_figures.py

    pipeline.py
```

---

## 30. Required outputs

```text
interaction_run_config.json

natural_state_decomposition.csv
shared_context_axis.csv
token_specific_context_components.csv

full_context_patching.csv
shared_context_patching.csv
specific_context_patching.csv
wrong_token_patching.csv

token_structured_causal_effects.csv

full_context_ablation.csv
shared_context_ablation.csv
specific_context_ablation.csv

random_direction_controls.csv
sequence_direction_controls.csv
off_manifold_metrics.csv

execution_model_sequence_only.csv
execution_model_context_only.csv
execution_model_additive.csv
execution_model_interaction.csv
execution_model_comparison.csv

mixed_generalization.csv
novel_generalization.csv

interaction_mechanism_summary.csv
seed_summary.csv
```

---

## 31. Required figures

### Figure 1 — Refined representational hypothesis

```text
Sequence position ───────┐
                         ├→ configured motor state → execution head → speed
Context ─────────────────┘
```

### Figure 2 — Shared versus token-specific A/B structure

### Figure 3 — Natural token-matched Mixed patching

```text
B
Mixed←B
Mixed
Mixed←A
A
```

### Figure 4 — Correct-token versus wrong-token patch

Primary interaction figure.

### Figure 5 — Natural versus causal token-specific execution difference

### Figure 6 — Shared-only versus full patch effect

### Figure 7 — Component necessity with sequence preservation

### Figure 8 — Cross-validated model comparison

```text
sequence only
context only
additive
interaction
```

### Figure 9 — Mixed out-of-sample prediction

### Figure 10 — Novel out-of-sample prediction

---

## 32. Definition of done

- [ ] Existing v2.1 networks are loaded without retraining.
- [ ] Human-matched checkpoint is primary.
- [ ] Local token-matched A/B decompositions are constructed.
- [ ] Shared and token-specific context components are separated.
- [ ] Full natural-state patching is implemented.
- [ ] Shared-only patching is implemented.
- [ ] Specific-only patching is implemented.
- [ ] Wrong-token patching control is implemented.
- [ ] Token-specific causal effects are compared with natural A/B effects.
- [ ] Full/shared/specific ablations are implemented.
- [ ] Random orthogonal control remains implemented.
- [ ] Sequence-direction functional control remains implemented.
- [ ] Sequence-only, context-only, additive, and interaction execution models are fit.
- [ ] Model comparison uses held-out seeds.
- [ ] Mixed is never used for fitting.
- [ ] Novel is never used for fitting.
- [ ] Mixed and Novel are reported separately.
- [ ] Off-manifold diagnostics are included for every synthetic state.
- [ ] Same-sequence prediction is monitored for every intervention.
- [ ] Final report distinguishes generic context-axis control from token-specific configured-state interaction.

---

## 33. Target mechanistic claim

The strongest supported claim would be:

> **Context does not simply encode a scalar execution speed. Instead, it configures the hidden representation of the currently prepared sequence state. The causal effect of context therefore depends on which point in the sequence is active. Replacing the token-matched context component shifts execution toward the corresponding learned context policy, whereas generic or mismatched context components are less effective. Mixed-context execution arises from recombination of these learned context-specific configurations of the same motor sequence.**

Compactly:

\[
oxed{
	ext{Context configures the motor representation; execution reads the configured state.}
\]
