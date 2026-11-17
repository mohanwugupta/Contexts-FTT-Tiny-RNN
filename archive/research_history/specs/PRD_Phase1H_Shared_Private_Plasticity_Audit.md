# PRD — Phase 1H: Representational Plasticity Audit of Shared vs Context-Specific Learning

## 1. Purpose

Test the hypothesis that A and B can share substantial sequence structure while showing little cross-context transfer because **learning-related plasticity is concentrated in context-specific representational components rather than in the shared sequence component**.

The candidate factorization is:

\[
h_{c,k}=U_S s_k + U_c p_{c,k} + \epsilon_{c,k},
\]

where:

- \(k\) indexes sequence position / transition,
- \(c\in\{A,B\}\),
- \(U_Ss_k\) is the shared sequence component,
- \(U_cp_{c,k}\) is the context-specific/private component.

The critical learning hypothesis is:

\[
\boxed{\|\Delta h^{private}\| \gg \|\Delta h^{shared}\|}
\]

during familiar A/B acquisition.

If true, this would explain how the system can simultaneously exhibit:

- strong A/B representational separation,
- little behavioral A↔B transfer,
- a causally useful shared sequence backbone,
- and intermediate probe behavior when familiar contextual routing weakens.

This phase is an **analysis/audit**, not another training intervention.

Do not modify the model architecture or retrain models except where deterministic replay is required to recover time-resolved states.

---

## 2. Canonical data and model

Use only the audited canonical SRNs:

```text
architecture: tanh Elman SRN
hidden_size: 8
High initialization scale: 2.0
Low initialization scale: 0.02
training schedule: canonical 170-sequence A/B schedule
```

Use the exact deterministic training replay already validated in the timeline audit.

High-scale networks are primary.

Low-scale networks are a reference/control condition.

Hard assertions:

```python
assert hidden_size == 8
assert high_init_scale == 2.0
assert low_init_scale == 0.02
assert replay_matches_final_checkpoint
```

Record:

```text
seed
condition
checkpoint_hash
training_schedule_hash
git_commit
model_config_hash
```

---

## 3. Core hypotheses

### H1 — Shared sequence structure exists throughout training

For each transition \(k\) and training time \(t\):

\[
s_k(t)=\frac{h_{A,k}(t)+h_{B,k}(t)}{2}.
\]

The shared state should remain sequence-specific.

### H2 — Context-specific separation grows over training

Define:

\[
d_k(t)=\frac{h_{A,k}(t)-h_{B,k}(t)}{2}.
\]

Then \(\|d_k(t)\|\) should grow substantially in High networks as contextual splitting develops.

### H3 — Learning-related representational change is mostly private

\[
\Delta s_k(t)=s_k(t+\Delta t)-s_k(t),
\]

\[
\Delta d_k(t)=d_k(t+\Delta t)-d_k(t).
\]

Prediction:

\[
\|\Delta d_k(t)\| > \|\Delta s_k(t)\|
\]

for much of High-scale learning.

### H4 — Cross-context transfer is predicted by gradient/update alignment, not raw representational overlap

\[
g_A(t)=\nabla_\theta L_A(\theta_t),\qquad
g_B(t)=\nabla_\theta L_B(\theta_t).
\]

Compute:

\[
\cos(g_A,g_B)=
\frac{g_A^\top g_B}{\|g_A\|\|g_B\|}.
\]

Low transfer is expected when A/B learning gradients are weakly aligned even if the hidden states retain a shared component.

### H5 — Shared and private components make distinct behavioral contributions

The shared component should preserve sequence-position information and remain causally useful, while private components should carry contextual differentiation and account for much of learning-related change.

---

## 4. Important conceptual distinction

Do not equate:

```text
representational overlap
```

with:

```text
learning transfer
```

The same network can have:

- strong shared sequence structure,
- strong A/B context separation,
- and low A→B learning transfer,

if the parameter updates that improve A are concentrated in directions that do not improve B.

The analysis must therefore include both:

1. **state-space decomposition**, and
2. **parameter/update/gradient analysis**.

---

## 5. Replay resolution

Use the true deterministic replay.

Preferred resolution:

```text
every external sequence update, 1...170
```

At minimum save states after every external training sequence.

If computational cost is high, compute expensive Jacobian/gradient diagnostics at a preregistered subset of checkpoints, e.g.:

```text
0, 10, 20, 30, ..., 170
```

but keep raw loss and hidden-state replay at all 170 sequences.

Do not infer intermediate states from final checkpoints.

---

## 6. Frozen evaluation at each training state

At each replay state \(\theta_t\), freeze parameters and evaluate the canonical sequence under:

```text
Context A
Context B
```

for all seven analyzed transitions.

Save:

```text
seed
training_sequence_index
transition
context
hidden_state[8]
raw_logits
centered_logits
raw_sequence_loss
correct_token_probability
execution_proxy
```

Probe evaluation must not alter parameters, optimizer state, recurrent persistent state, or RNG state.

---

## 7. Midpoint/shared-difference decomposition over training

For each seed, training time \(t\), and transition \(k\):

\[
s_k(t)=\frac{h_{A,k}(t)+h_{B,k}(t)}{2},
\]

\[
d_k(t)=\frac{h_{A,k}(t)-h_{B,k}(t)}{2}.
\]

Save:

```text
shared_state_norm
private_difference_norm
AB_distance
shared_state_change
private_state_change
```

with:

\[
\Delta s_k(t)=s_k(t+1)-s_k(t),
\]

\[
\Delta d_k(t)=d_k(t+1)-d_k(t).
\]

Primary plasticity ratio:

\[
R_{private/shared}(t,k)
=
\frac{\|\Delta d_k(t)\|^2}{\|\Delta s_k(t)\|^2+\epsilon}.
\]

Also report the unnormalized numerator and denominator.

Do not rely only on the ratio.

---

## 8. Task-space version

Repeat the same analysis in centered logit/task space.

For \(z_{A,k}(t),z_{B,k}(t)\), define:

\[
s^z_k(t)=\frac{z_{A,k}(t)+z_{B,k}(t)}{2},
\]

\[
d^z_k(t)=\frac{z_{A,k}(t)-z_{B,k}(t)}{2}.
\]

Compute:

```text
task_shared_change
task_private_change
task_private/shared_ratio
task_AB_separation
```

Task space is important because hidden-state changes may include output-null movement.

---

## 9. Sequence-information decoding in shared vs private components

At selected training checkpoints, train lightweight linear decoders using frozen states.

### Shared decoder

Input: \(s_k(t)\)

Target:

```text
sequence position / transition k
```

Report:

```text
sequence_position_accuracy_shared
```

### Private decoder

Input: \(d_k(t)\)

Targets:

1. sequence position,
2. context identity.

Report:

```text
sequence_position_accuracy_private
context_accuracy_private
```

Use proper cross-validation / leave-seed-out evaluation.

Do not treat decoder accuracy as causal evidence.

---

## 10. Context-invariant subspace analysis

The arithmetic midpoint is an operationalization, not the whole theory.

### 10.1 Shared subspace \(P_S\)

Using A and B states at the same sequence positions:

1. average paired A/B states,
2. run PCA/SVD on the averaged states,
3. retain a fixed preregistered rank or report ranks 1–7 as sensitivity analysis.

Define projector:

\[
P_S.
\]

### 10.2 Context-discriminative/private subspace \(P_C\)

Use A-B difference vectors:

\[
h_{A,k}-h_{B,k}.
\]

Estimate their span via SVD/PCA.

Define:

\[
P_C.
\]

Do not assume \(P_S\perp P_C\).

Quantify principal angles between them.

If needed for energy decomposition, construct an orthogonalized private projector:

\[
P_{C\perp S}.
\]

---

## 11. Project actual representational updates

For every actual training update \(t\to t+1\), measure how the frozen A and B representations change:

\[
\Delta h_{c,k}(t)=h_{c,k}(t+1)-h_{c,k}(t).
\]

Project:

\[
\Delta h^S_{c,k}=P_S\Delta h_{c,k},
\]

\[
\Delta h^C_{c,k}=P_{C\perp S}\Delta h_{c,k}.
\]

Compute energy fractions:

\[
F_S=
\frac{\|\Delta h^S\|^2}{\|\Delta h\|^2+\epsilon},
\]

\[
F_C=
\frac{\|\Delta h^C\|^2}{\|\Delta h\|^2+\epsilon}.
\]

Report residual energy separately if the decomposition is not exhaustive.

---

## 12. Update specificity by training context

For each actual A training update, evaluate its effect on both A and B frozen representations and losses.

For an A update:

\[
T_{A\rightarrow B}
=
L_B(\theta_t)-L_B(\theta_{t+1}),
\]

\[
T_{A\rightarrow A}
=
L_A(\theta_t)-L_A(\theta_{t+1}).
\]

Define:

\[
R_{transfer,A}
=
\frac{T_{A\rightarrow B}}{T_{A\rightarrow A}+\epsilon}.
\]

Do the analogous calculation for B→A.

Always report the raw numerator and denominator as well.

Save:

```text
within_context_state_change
cross_context_state_change
within_context_loss_change
cross_context_loss_change
```

---

## 13. Parameter-gradient alignment

At registered replay checkpoints, compute frozen gradients:

\[
g_A(t)=\nabla_\theta L_A,
\]

\[
g_B(t)=\nabla_\theta L_B.
\]

Compute:

```text
cosine_gradient_alignment
dot_product_gradient_alignment
gradient_norm_A
gradient_norm_B
```

Prediction:

High networks should show weaker A/B gradient alignment than Low networks.

---

## 14. Per-parameter-block gradient alignment

Decompose gradients by model block:

```text
input weights
recurrent weights
hidden bias
output weights
output bias
```

For each block \(b\):

\[
\cos(g_A^{(b)},g_B^{(b)}).
\]

This can reveal where context-specific learning is implemented.

Do not assume a particular block will dominate.

---

## 15. Functional transfer via one-step counterfactual updates

At checkpoint \(t\):

1. clone \(\theta_t\),
2. take exactly one canonical optimizer-equivalent A update,
3. evaluate A and B raw losses,
4. restore,
5. take one B update,
6. evaluate A and B raw losses.

Save:

```text
delta_A_after_A_update
delta_B_after_A_update
delta_A_after_B_update
delta_B_after_B_update
```

This is a direct local functional transfer test.

No probes enter this analysis.

---

## 16. Connect transfer to shared/private plasticity

Across seeds and training checkpoints, test whether cross-context transfer is associated with:

```text
shared update energy
private update energy
gradient alignment
A/B separation
```

Primary descriptive relationships:

\[
T_{A\rightarrow B}\sim F_S
\]

and

\[
T_{A\rightarrow B}\sim\cos(g_A,g_B).
\]

Do not treat correlation as causation.

---

## 17. Optional causal projection control

If technically clean, take an actual A update and compare:

```text
full update
shared-dominant component
private-dominant component
norm-matched orthogonal control
```

Only implement if parameter-to-representation projection can be defined without a fragile or arbitrary mapping.

This is optional and must not block Phase 1H.

---

## 18. High-vs-Low comparison

Repeat the main analyses for Low networks.

### Expected Low pattern if the theory is right

- small A/B private difference,
- stronger A/B gradient alignment,
- more shared functional learning,
- higher cross-context transfer,
- less private-dominant plasticity.

### Expected High pattern

- larger A/B private difference,
- weaker gradient alignment,
- lower cross-context transfer,
- more update energy in context-discriminative directions.

Do not force this pattern.

---

## 19. Required raw-loss time series

### TS1 — True A/B acquisition

x-axis:

```text
actual external training sequence index
```

y-axis:

```text
raw sequence loss
```

Show A and B across all 170 sequences.

### TS2 — Within- vs cross-context one-step transfer

Across actual training updates, plot:

```text
within-context raw loss improvement
cross-context raw loss improvement
```

### TS3 — Shared vs private representational change

Plot:

```text
||Δs||
||Δd||
```

across training.

### TS4 — Shared/private update energy

Plot:

```text
shared subspace energy
private/context-discriminative subspace energy
```

across training.

### TS5 — Gradient alignment

Plot:

\[
\cos(g_A,g_B)
\]

at registered checkpoints.

Raw model loss must remain the primary behavioral quantity. Do not replace it with normalized loss ratios.

---

## 20. Required figures

```text
PHASE1H_FIG01_shared_vs_private_change.png
PHASE1H_FIG02_private_shared_ratio_by_seed.png
PHASE1H_FIG03_sequence_decoding_shared_private.png
PHASE1H_FIG04_AB_gradient_alignment.png
PHASE1H_FIG05_parameter_block_gradient_alignment.png
PHASE1H_FIG06_within_vs_cross_context_transfer.png
PHASE1H_FIG07_transfer_vs_shared_update_energy.png
PHASE1H_FIG08_transfer_vs_gradient_alignment.png
PHASE1H_FIG09_high_vs_low_plasticity.png
PHASE1H_FIG10_shared_private_subspace_angles.png

PHASE1H_TS01_true_AB_acquisition_raw_loss.png
PHASE1H_TS02_within_cross_transfer_raw_loss.png
PHASE1H_TS03_shared_private_change_over_training.png
PHASE1H_TS04_update_energy_over_training.png
PHASE1H_TS05_gradient_alignment_over_training.png
```

---

## 21. Required CSVs

```text
PHASE1H_replay_states.csv
PHASE1H_shared_private_states.csv
PHASE1H_representational_updates.csv
PHASE1H_subspace_metrics.csv
PHASE1H_gradient_alignment.csv
PHASE1H_block_gradient_alignment.csv
PHASE1H_functional_transfer.csv
PHASE1H_decoder_results.csv
PHASE1H_seed_summary.csv
```

Every row must retain provenance:

```text
seed
scale_condition
training_sequence_index
training_context
transition
checkpoint_hash
```

where applicable.

---

## 22. Primary summary metrics

For each seed report:

```text
mean_shared_change
mean_private_change
median_private/shared_ratio
shared_update_energy_fraction
private_update_energy_fraction
mean_gradient_alignment
mean_A_to_B_transfer
mean_B_to_A_transfer
AB_hidden_separation
AB_task_separation
shared_sequence_decoder_accuracy
private_context_decoder_accuracy
```

---

## 23. Statistical summaries

Use seed as the independent inferential unit.

Report:

- mean,
- median,
- bootstrap 95% CI,
- paired High-vs-Low differences where appropriate.

Do not treat 170 sequence updates as independent observations for inferential statistics.

Time-resolved plots may show all updates.

---

## 24. Decision logic

### Outcome A — Strong support

Require convergent evidence:

1. shared component carries reliable sequence-position information,
2. A/B context-specific structure grows strongly in High networks,
3. actual learning-related changes are larger in private than shared components,
4. A/B gradient alignment is weak in High networks,
5. cross-context functional transfer is small,
6. Low networks show a more shared/aligned learning regime.

Interpretation:

> A and B share sequence structure, but familiar-context learning is implemented predominantly through context-specific representational and parameter changes. This explains how shared sequence content can coexist with little A↔B transfer.

### Outcome B — Shared structure exists but plasticity is not private-dominant

If the shared component is sequence-informative but shared and private changes are similar, do not use the private-plasticity explanation.

### Outcome C — Gradient alignment predicts transfer but state decomposition does not

Interpretation:

> The transfer limitation is more naturally parameter-functional than geometrically factorized.

### Outcome D — High and Low show similar plasticity structure

Interpretation:

> Initialization scale affects final geometry and transfer through another mechanism.

---

## 25. Caveats

### Midpoint is operational

\[
s_k=(h_A+h_B)/2
\]

does not prove that the system literally stores an arithmetic midpoint.

Therefore the data-driven shared-subspace analysis is mandatory.

### State change is not parameter plasticity

A large \(\Delta d\) could emerge from nonlinear propagation of a parameter update.

That is why gradient and one-step functional-transfer analyses are mandatory.

### Gradient alignment is local

Use actual one-step optimizer-equivalent updates as a complementary functional test.

### No probe-defined subspaces

Mixed, Novel, and NoContext must not enter shared/private subspace estimation.

---

## 26. RED → GREEN tests

### RED 1 — Canonical replay

Noncanonical checkpoint:

```text
FAIL: NONCANONICAL_CHECKPOINT
```

### RED 2 — Frozen evaluation

A/B state extraction must not alter parameters, optimizer state, or RNG.

### RED 3 — Endpoint decomposition identity

Assert:

\[
s_k+d_k=h_{A,k}
\]

\[
s_k-d_k=h_{B,k}
\]

within tolerance.

### RED 4 — Correct temporal update

Ensure \(\Delta s_k(t)\) and \(\Delta d_k(t)\) use adjacent real replay states, not duplicated final checkpoints.

### RED 5 — Training-context labels

Fail if an optimizer update lacks its A/B context label.

### RED 6 — Raw-loss identity

Transfer plotting loss must equal canonical raw sequence loss.

### RED 7 — Gradient isolation

Computing \(g_A\) must not contaminate \(g_B\).

### RED 8 — Counterfactual update restoration

After A-update testing, restore the exact original parameter and optimizer state before B-update testing.

### RED 9 — Familiar-only subspace estimation

Fail if Mixed, Novel, or NoContext enter shared/private subspace estimation.

### RED 10 — Seed-level inference

Bootstrap/resampling must operate on seeds, not sequence updates.

---

## 27. Required report

Create:

```text
PHASE1H_SHARED_PRIVATE_PLASTICITY_REPORT.md
```

Sections:

1. Executive result
2. Canonical replay and provenance
3. Formal shared/private hypothesis
4. Shared sequence structure over training
5. Growth of A/B private structure
6. Shared vs private representational plasticity
7. Task-space plasticity
8. Shared/private sequence decoding
9. Gradient alignment
10. Parameter-block gradient alignment
11. One-step functional A↔B transfer
12. Transfer vs shared/private plasticity
13. High-vs-Low comparison
14. Raw-loss time series
15. Decision-tree classification
16. Implications for the human representational account
17. Remaining uncertainties

---

## 28. Paper-level interpretation if supported

If Outcome A holds:

> A and B can share substantial sequence-specific representational structure while remaining behaviorally insulated because learning-related plasticity is concentrated in context-specific components. Representational sharing and cross-context learning transfer are therefore dissociable: the former supports reuse of common sequence structure, whereas the latter depends on whether learning updates propagate through shared or context-specific directions.

Do not use this conclusion if the plasticity decomposition fails.

---

## 29. Scope discipline

Do not introduce:

- new architectures,
- new probe training,
- new context cues,
- new regularizers,
- new optimization schemes.

Phase 1H is intended to explain the existing result, not produce a new one.

The central question is:

\[
\boxed{
\text{Does the network share sequence content while learning mainly in private/context-specific directions?}
\]
