# PRD: Emergent Retrieval and Context-Inference Program for SRNs
## Testing whether ambiguous/novel expression can emerge from A/B-only learning

## 0. Purpose

The current SRN program has established:

1. High-scale SRNs robustly form context-specific A/B representations and context-yoked RT dynamics.
2. Human-like shared-x geometry improves Mixed/Novel transfer without destroying A/B splitting.
3. Increasing hidden capacity mostly increases specialization and worsens transfer.
4. Increasing explicit context dimensionality does not solve the transfer problem.
5. Therefore the remaining bottleneck is increasingly likely to lie in **how learned context-specific knowledge is retrieved or expressed under ambiguous/novel context**, not in representational capacity alone.

The next experiment must test retrieval/inference mechanisms while preserving a strict emergence criterion:

> **Mixed and Novel must never be training targets, and no model may be optimized to place them between A and B.**

Interpolation is an outcome measure, not a loss.

This PRD compares normative/control mechanisms that explicitly combine A/B outputs with multiple emergent retrieval architectures that are trained only on ordinary A/B experience.

---

# 1. Scientific question

> Can a model trained only on familiar A/B contexts learn a retrieval/inference computation that causes Mixed and/or Novel probes to spontaneously express structured intermediate behavior while preserving the learned A/B split?

The key distinction is:

```text
learning A/B representations
vs
retrieving / expressing those representations
```

---

# 2. Frozen reference environment

Use the current best human-grounded environment:

```text
G2_shared_x
```

Freeze unless explicitly changed by a model variant:

```text
motor-token sequence
P1..P8 ordinal channels
human shared-x geometry
context-specific y geometry
A10/B10 curriculum
170 external training sequences
optimizer
learning rate
next-token objective
monotonic execution readout
RT = 1 - execution speed
probe firewall
```

Primary reference architecture:

```text
8-unit tanh Elman SRN
High initialization scale = 2.0
Low initialization scale = 0.02
```

No capacity sweep in this PRD.

---

# 3. Non-negotiable emergence constraints

Every emergent variant must obey:

## E0.1 — No Mixed/Novel training

```text
Mixed optimizer steps = 0
Novel optimizer steps = 0
```

## E0.2 — No interpolation loss

Forbidden:

```text
loss(Mixed, midpoint(A,B))
loss(Novel, midpoint(A,B))
loss(lambda, 0.5)
```

## E0.3 — No probe-based model selection

Do not select:

```text
hyperparameters
early stopping
architecture
temperature
kernel width
latent size
```

using Mixed or Novel performance.

## E0.4 — No hand-coded Mixed/Novel labels

The model may receive the actual Mixed/Novel cue/layout vectors at test time, but not a symbolic label saying:

```text
"Mixed"
"Novel"
```

## E0.5 — Interpolation is only measured after training

All midpoint/corridor/responsibility measures are post hoc diagnostics.

---

# 4. Model family overview

Test the following variants.

| ID | Model | Is averaging imposed? | Primary role |
|---|---|---:|---|
| V0 | Current G2 SRN baseline | No | Reference |
| C1 | RBF responsibility-weighted output | Yes | Sufficiency / normative control |
| C2 | COIN-lite responsibility + temporal prior | Yes | History-aware sufficiency control |
| E1 | Learned context embedding → ordinary SRN | No | Minimal emergent retrieval model |
| E2 | History-dependent latent-context inference → SRN | No | Emergent contextual inference |
| E3 | Prediction-error-updated latent context | No | Emergent adaptive retrieval |
| E4 | Context-modulated shared recurrent dynamics | No | Emergent factorized architecture |

Controls C1/C2 are not candidate headline mechanisms if they work; they quantify what explicit responsibility weighting could achieve.

Primary theoretical emphasis is E1–E4.

---

# 5. Common A/B training task

All variants train only on familiar A and B contexts.

At each transition k, the environment provides:

\[
x_{c,k}
=
[
	ext{motor token}_k,
P_k,
\mathcal L_c,
C_c
]
\]

where:

```text
c ∈ {A,B} during training only
```

The target remains:

```text
next motor token
```

Primary loss:

\[
L_{	ext{seq}}
=
-\log p(y_{k+1}^{true}).
\]

The current monotonic execution objective/readout remains unchanged unless explicitly stated.

---

# 6. Shared contextual evidence vector

Define one common context-evidence vector for retrieval modules:

\[
z_c
=
[
C_c,
\operatorname{vec}(\mathcal L_c)
].
\]

Use exactly the validated G2 cue and layout representation.

Do not add new contextual dimensions.

For sequence-position-specific models, the ordinal channel P_k remains in the SRN input but is not automatically included in z_c unless explicitly registered.

Primary implementation:

```text
z = context cue + full layout vector
```

constant within a sequence.

---

# 7. V0 — Baseline G2 SRN

Exact current validated model.

\[
h_t
=
	anh(W_xx_t+W_hh_{t-1}+b_h)
\]

\[
p_t
=
\operatorname{softmax}(W_oh_t+b_o).
\]

No new retrieval mechanism.

Purpose:

```text
reference transfer
reference A/B differentiation
reference E3 interaction
```

---

# 8. C1 — RBF responsibility-weighted output control

## 8.1 Purpose

Test whether a simple continuous responsibility mechanism is **sufficient** to produce improved Mixed expression.

This control is intentionally not fully emergent because weighted output combination is built into the architecture.

## 8.2 A/B context prototypes

Estimate from A/B training only:

\[
\mu_A = \mathbb E[z|A]
\]

\[
\mu_B = \mathbb E[z|B].
\]

## 8.3 Learned scale

Use a single positive bandwidth parameter:

\[
\sigma = \operatorname{softplus}(s)+\epsilon.
\]

Train s using A/B context discrimination only.

Loss:

\[
L_{ctx}
=
-\log q_c
\]

for familiar A/B sequences.

No Mixed/Novel data.

## 8.4 Responsibilities

\[
g_c(z)
=
\exp
\left[
-rac{\|z-\mu_c\|^2}{2\sigma^2}
ight].
\]

\[
q_A
=
rac{g_A}{g_A+g_B},
\qquad
q_B=1-q_A.
\]

## 8.5 Output mixture

Run the A- and B-conditioned predictions:

\[
p_A(y|x_k),
\qquad
p_B(y|x_k).
\]

Then:

\[
p(y)
=
q_Ap_A(y)+q_Bp_B(y).
\]

RT is computed from the resulting correct-token confidence using the frozen monotonic execution mapping.

## 8.6 Interpretation

If C1 works:

> explicit responsibility-weighted expression is sufficient.

Do not claim interpolation emerged.

---

# 9. C2 — COIN-lite temporal responsibility control

## 9.1 Purpose

Add a learned temporal prior over familiar context persistence.

Still an explicit weighted-output control.

## 9.2 Learn transition matrix from A/B curriculum

Estimate:

\[
T_{ij}
=
P(c_t=j|c_{t-1}=i)
\]

from familiar A/B training only.

## 9.3 Prior propagation

\[
\pi_t(c)
=
\sum_{c'}
T_{c'c}q_{t-1}(c').
\]

## 9.4 Cue likelihood

Use C1 RBF likelihood:

\[
L_c(z_t)=g_c(z_t).
\]

## 9.5 Posterior responsibility

\[
q_t(c)
\propto
L_c(z_t)\pi_t(c).
\]

Normalize across A/B.

## 9.6 Output

\[
p_t(y)
=
q_t(A)p_A(y)+q_t(B)p_B(y).
\]

Use this only as a history-aware sufficiency control.

---

# 10. E1 — Learned context embedding → ordinary SRN

## 10.1 Purpose

Minimal primary emergent model.

The model learns a continuous context representation from familiar A/B inputs, but no rule says Mixed must interpolate.

## 10.2 Context encoder

Use a small MLP:

```text
z -> Linear(d_z, 4)
   -> tanh
   -> Linear(4, d_r)
```

Primary:

```text
d_r = 2
```

Output:

\[
r_t=g_\phi(z_t).
\]

## 10.3 SRN input

Replace the raw context cue/layout channels entering the SRN with the learned embedding:

\[
	ilde x_t
=
[
	ext{motor token}_t,
P_t,
r_t
].
\]

Then:

\[
h_t
=
	anh
(
W_x	ilde x_t
+
W_hh_{t-1}
+b_h
).
\]

## 10.4 Training

Jointly train:

```text
context encoder
SRN
sequence output head
monotonic execution head
```

using only A/B sequence learning and the existing execution objective.

No explicit context-classification loss in the primary condition.

## 10.5 Diagnostic auxiliary condition

Optional only after primary:

```text
E1_aux_ctx
```

adds weak A/B context-classification loss:

\[
L=L_{	ext{seq}}+\lambda_{ctx}L_{ctx}
\]

with fixed preregistered:

```text
lambda_ctx = 0.1
```

Do not tune using probes.

---

# 11. E2 — History-dependent latent-context inference → SRN

## 11.1 Purpose

Allow context inference to depend on recent context history.

No A/B output mixture is imposed.

## 11.2 Latent inference recurrence

Use:

\[
r_t
=
	anh
(
W_zz_t
+
W_rr_{t-1}
+
b_r
).
\]

Primary latent size:

```text
d_r = 2
```

This is a tiny recurrent inference state separate from the 8-unit motor SRN.

## 11.3 Timescale

Update r:

```text
once per external sequence
```

not once per transition.

The same r is used for all 7 transitions in a sequence.

## 11.4 Motor SRN

\[
h_t
=
f_	heta(
	ext{token}_t,
P_t,
r_{	ext{sequence}},
h_{t-1}
).
\]

## 11.5 Training

Only A/B sequence prediction + existing execution objective.

Do not add a context-state target.

The A10/B10 curriculum itself provides temporal structure from which persistence can be learned.

## 11.6 Probe policies

At frozen test, evaluate:

```text
reset_r
carry_r
```

as separate preregistered probe policies.

No parameter updates.

---

# 12. E3 — Prediction-error-updated latent context

## 12.1 Purpose

Test whether the model can learn to revise its latent contextual state using its own prediction errors.

This is closer to adaptive responsibility inference, but interpolation remains unconstrained.

## 12.2 Initial context state

At sequence start:

\[
r_0=g_\phi(z).
\]

Use the same 2-D feedforward initializer as E1.

## 12.3 Within-sequence error signal

After each transition:

\[
e_t
=
-\log p(y_{t+1}^{true})
\]

Primary error input is scalar NLL.

## 12.4 Context-state update

Use a tiny GRUCell:

\[
r_{t+1}
=
\operatorname{GRUCell}
(
[e_t,z],
r_t
).
\]

Primary:

```text
d_r = 2
```

## 12.5 Motor SRN

\[
h_t
=
f_	heta(
	ext{token}_t,
P_t,
r_t,
h_{t-1}
).
\]

## 12.6 Training

End-to-end differentiable training on A/B only.

The model is never rewarded for moving r toward an A/B midpoint.

It only learns whether prediction error can be used to improve familiar A/B sequence prediction/execution.

## 12.7 Frozen probe

During Mixed/Novel evaluation:

```text
weights frozen
r may update from prediction error
```

This is state inference, not learning.

---

# 13. E4 — Context-modulated shared recurrent dynamics

## 13.1 Purpose

Test whether the model can learn a factorized architecture:

```text
shared recurrent sequence dynamics
+
context-dependent modulation
```

without explicit expert averaging.

## 13.2 Context encoder

Use E1 context embedding:

\[
r=g_\phi(z),
\qquad
d_r=2.
\]

## 13.3 FiLM-style recurrent modulation

Primary implementation:

\[
\gamma(r)=1+W_\gamma r
\]

\[
eta(r)=W_eta r.
\]

Compute shared recurrent preactivation:

\[
a_t
=
W_xx_t+W_hh_{t-1}+b_h.
\]

Then:

\[
h_t
=
	anh
\left[
\gamma(r)\odot a_t+eta(r)
ight].
\]

Thus A and B may learn different effective dynamics, but there are no separately stored A/B expert networks and no explicit interpolation rule.

## 13.4 Low-rank modulation robustness

Optional secondary implementation:

\[
W_h(r)
=
W_h
+
U\operatorname{diag}(Ar)V^	op
\]

with rank:

```text
rank = 2
```

Run only if FiLM model passes training gates but produces no context differentiation.

Do not tune rank on Mixed/Novel probes.

## 13.5 Training

A/B sequence prediction + existing execution objective only.

No context-classification target.

---

# 14. Training protocol

## 14.1 Screening

Run:

```text
5 paired seeds
```

for:

```text
V0
C1
C2
E1
E2
E3
E4
```

High and Low scales where architecture permits.

For controls C1/C2, the underlying SRN can reuse validated G2 baseline weights if implementation is strictly post hoc.

For E1–E4, train end-to-end from matched initialization seeds.

## 14.2 Confirmatory

Any emergent model passing all gates:

```text
20 paired seeds
```

Controls C1/C2:

```text
20 seeds
```

for benchmark comparison.

---

# 15. Hard training gates

A model may not be interpreted on probe behavior unless it passes:

## Gate G1 — familiar sequence learning

At least 4/5 screening seeds:

```text
late A/B NLL < early A/B NLL
```

## Gate G2 — High-scale differentiation

At least 4/5 High seeds:

```text
A/B matched-position hidden distance > corresponding Low seed
```

## Gate G3 — High Cross/Split phenotype

At least 4/5 High seeds:

```text
Cross or Split
```

using the canonical state-space bridge.

## Gate G4 — Low Lump phenotype

At least 4/5 Low seeds:

```text
Lump
```

## Gate G5 — RT grounding

Require:

```text
confidence -> RT slope < 0
```

for every seed.

## Gate G6 — no pathological probe-specific tuning

Training logs must confirm:

```text
0 Mixed batches
0 Novel batches
```

---

# 16. Primary probe outcomes

For frozen A, B, Mixed, Novel probes report:

```text
prediction NLL
correct-token confidence
normalized RT
next-token accuracy
hidden states
latent context state r
```

Mixed and Novel always analyzed separately.

---

# 17. Emergent interpolation metrics

These are **outcomes only**.

## 17.1 Seed-level endpoint canonicalization

For visualization:

```text
Fast = lower mean frozen A/B RT
Slow = higher mean frozen A/B RT
```

Swap whole profiles within seed.

Never use Mixed/Novel to determine labels.

## 17.2 RT interpolation index

For each seed and transition:

\[
I_{s,k}
=
rac
{RT_{probe,s,k}-RT_{Fast,s,k}}
{RT_{Slow,s,k}-RT_{Fast,s,k}}.
\]

Interpret:

```text
0 < I < 1  -> between endpoints
I < 0      -> faster than both
I > 1      -> slower than both
```

Primary summary:

```text
fraction of transitions with 0 < I < 1
```

for Mixed and Novel separately.

## 17.3 Profile interpolation

Fit:

\[
RT_{probe,k}
=
lpha
+
q RT_{A,k}
+
(1-q)RT_{B,k}
\]

post hoc only.

Report:

```text
best-fit q
RMSE
R²
```

This is descriptive and is never used for training.

---

# 18. Representational interpolation diagnostics

For each seed and ordinal position:

\[
\lambda^*
=
rg\min_\lambda
\left\|
h_{probe,k}
-
[(1-\lambda)h_{A,k}+\lambda h_{B,k}]
ight\|^2.
\]

Report:

```text
lambda*
off-corridor distance
inside-segment indicator
distance to A
distance to B
```

Do not require hidden-state interpolation for behavioral success.

---

# 19. Latent-context diagnostics

For E1–E4:

## 19.1 Familiar anchors

Record:

\[
r_A,\qquad r_B.
\]

## 19.2 Probe location

For Mixed/Novel record:

\[
r_M,\qquad r_N.
\]

Compute:

```text
distance to r_A
distance to r_B
closest point on r_A-r_B segment
off-segment distance
```

Again, no latent interpolation target exists.

## 19.3 Dynamics

For E2/E3:

plot r across:

```text
sequence number
preceding context
within-sequence transition
```

where applicable.

---

# 20. History sensitivity test

For E2/E3 and C2, probe the same frozen Mixed/Novel input after:

```text
10 A sequences
10 B sequences
reset state
```

No weight updates.

Measure:

```text
probe NLL
probe RT
latent r
```

Prediction if context inference uses history:

```text
Mixed/Novel expression should shift depending on preceding A vs B
```

Do not define the direction in advance beyond attraction toward the preceding familiar state.

---

# 21. Novel-context fallback diagnostic

Do not require Novel to interpolate.

For each model, compare Novel against:

```text
A
B
Mixed
context-neutral control
```

Define a neutral test vector by removing the context-specific cue component while keeping shared-x geometry and ordinal/task structure intact.

This control is test-only and must not be trained.

Question:

> Does Novel behave more like A/B interpolation or like a shared/default task representation?

---

# 22. Sufficiency vs emergence comparison

Create one explicit comparison table:

| Model | Mixed between-endpoint fraction | Novel between-endpoint fraction | A/B split preserved | Averaging imposed? |
|---|---:|---:|---:|---:|

Primary interpretation:

### If C1/C2 work but E1–E4 fail

> Responsibility-weighted expression is sufficient but does not emerge from the tested A/B-only architectures.

### If E1 works

> A simple learned continuous context embedding is enough for intermediate expression to emerge.

### If only E2/E3 work

> Temporal/contextual inference is required.

### If only E4 works

> Factorized recurrent dynamics are required.

### If none work

> The missing human computation is not captured by these minimal retrieval mechanisms.

---

# 23. Statistical analysis

For confirmatory 20-seed runs, fit separate models for Mixed and Novel.

Primary outcomes:

```text
NLL deficit over worse endpoint
RT deficit over worse endpoint
between-endpoint fraction
off-corridor hidden distance
```

Preferred model:

```text
metric ~ model_variant + (1|seed)
```

For history variants:

```text
metric ~ model_variant * preceding_context + (1|seed)
```

Report:

```text
mean difference
bootstrap 95% CI
standardized effect
```

Do not use significance as the sole criterion.

---

# 24. Primary success criteria

An emergent model is considered a strong candidate if:

## S1 — Familiar phenotype preserved

```text
High remains majority Cross/Split
Low remains majority Lump
```

## S2 — Mixed transfer improves

Relative to V0:

```text
Mixed NLL deficit decreases
Mixed RT deficit decreases
```

## S3 — Mixed becomes more endpoint-intermediate

Relative to V0:

```text
fraction(0 < I_Mixed < 1) increases
```

without training on Mixed.

## S4 — Upstream prediction improves

Improvement cannot exist only in RT.

Require:

```text
NLL improvement
```

## S5 — Novel is interpretable

Novel may either:

```text
move toward A/B corridor
or
move toward neutral/shared-task state
```

but should not merely become arbitrarily worse.

---

# 25. Strong falsification criteria

Reject a variant as explanation if:

### F1

Mixed becomes intermediate only because A/B differentiation collapses.

### F2

RT becomes intermediate while prediction NLL does not improve.

### F3

Interpolation requires tuning a probe-specific temperature/bandwidth on Mixed/Novel.

### F4

The model works only when a direct A/B context classification target is added.

That may still be informative, but it weakens the "emergent from sequence learning" claim.

### F5

Larger latent/context modules are needed after minimal versions fail.

Do not start an unrestricted architecture search within this PRD.

---

# 26. Required ablations

For any successful emergent variant:

## A1 — Remove context inference

Set:

```text
r = 0
```

or remove modulation.

Transfer advantage should disappear.

## A2 — Shuffle cue-to-context mapping at test

Preserve cue magnitude but mismatch A/B mapping.

This should disrupt familiar context expression.

## A3 — Freeze encoder at random initialization

Train only SRN around it.

Tests whether learned contextual inference matters.

## A4 — Remove temporal carry

For E2/E3.

Tests whether history is functionally important.

## A5 — Remove prediction-error input

For E3.

Tests whether error-driven updating is necessary.

---

# 27. TDD requirements

Use RED → GREEN → REFACTOR.

## T1 — Probe firewall

Assert:

```text
num_train_mixed == 0
num_train_novel == 0
```

## T2 — No probe losses

Search training code/config and assert no loss term references:

```text
Mixed
Novel
midpoint
lambda
corridor
```

## T3 — Matched A/B curriculum

Every variant sees byte-identical familiar training sequences.

## T4 — Frozen geometry

All variants use the same G2 shared-x geometry.

## T5 — Fixed RT mapping

Monotonic execution mapping unchanged across variants.

## T6 — Latent size fixed

For E1–E4 primary:

```text
d_r == 2
```

## T7 — History state reset behavior

For E2/E3:

```text
reset condition -> identical initial r
carry condition -> previous r preserved
```

## T8 — Frozen probe weights

Parameter hashes identical before/after Mixed/Novel evaluation.

## T9 — C1/C2 control labeling

Output/report must explicitly mark:

```text
averaging_imposed = True
```

## T10 — Emergent-model labeling

E1–E4:

```text
averaging_imposed = False
```

## T11 — Interpolation metrics post hoc

No interpolation metric can be imported into training modules.

## T12 — Seed pairing

Use same seed IDs across V0/E1/E2/E3/E4.

---

# 28. Required outputs

## Config

```text
retrieval_program_config.json
model_variant_registry.json
```

## Training

```text
retrieval_training_summary.csv
retrieval_state_space.csv
retrieval_hidden_differentiation.csv
retrieval_rt_grounding.csv
```

## Probe behavior

```text
retrieval_probe_metrics.csv
retrieval_interpolation_metrics.csv
retrieval_profile_fit.csv
retrieval_history_effects.csv
```

## Representations

```text
retrieval_hidden_corridor.csv
retrieval_latent_context.csv
retrieval_latent_dynamics.csv
```

## Ablations

```text
retrieval_ablation_results.csv
```

## Report

```text
EMERGENT_RETRIEVAL_PROGRAM_REPORT.md
```

---

# 29. Required figures

## Figure 1 — Architecture schematic

Show all variants side-by-side:

```text
V0
C1
C2
E1
E2
E3
E4
```

Mark explicitly which impose output averaging.

## Figure 2 — Familiar A/B phenotype

For each variant:

```text
High Cross/Split fraction
Low Lump fraction
A/B hidden separation
```

## Figure 3 — Mixed transfer

Plot:

```text
Mixed NLL deficit
Mixed RT deficit
between-endpoint fraction
```

for all variants.

## Figure 4 — Novel transfer

Same metrics, separately.

## Figure 5 — Behavioral interpolation

Canonical Fast/Slow/Mixed/Novel transition profiles.

Do not average A/B labels without within-seed canonicalization.

## Figure 6 — Hidden-state corridor

For Mixed and Novel:

```text
lambda*
off-corridor distance
inside-segment fraction
```

## Figure 7 — Latent-context geometry

For E1–E4:

```text
r_A
r_B
r_Mixed
r_Novel
```

projected only for visualization.

## Figure 8 — History dependence

For C2/E2/E3:

```text
after A history
after B history
reset
```

## Figure 9 — Sufficiency vs emergence summary

One panel comparing:

```text
explicit averaging controls
vs
emergent models
```

---

# 30. Execution order

Run in this order.

## Stage 0

Reproduce V0 on saved seeds.

## Stage 1

Implement C1.

Purpose:

```text
Can explicit responsibility weighting solve the behavioral target at all?
```

If no, stop and reassess the target before building emergent architectures.

## Stage 2

Implement E1.

This is the minimal primary emergent test.

## Stage 3

Implement E2.

Only after E1 is validated.

## Stage 4

Implement E3.

Only after E2 is validated.

## Stage 5

Implement E4.

Architecturally distinct factorization test.

## Stage 6

Implement C2.

History-aware explicit averaging benchmark.

## Stage 7

Run 20-seed confirmatory set for all gate-passing variants.

## Stage 8

Run ablations only for successful emergent variants.

---

# 31. Interpretation matrix

## Result A — C1 works; E1 works

Interpretation:

> Continuous context representations learned from A/B experience are sufficient for intermediate Mixed expression to emerge; explicit responsibility averaging is not necessary.

## Result B — C1 works; E1 fails; E2 works

Interpretation:

> Temporal latent-context inference is needed beyond static context embeddings.

## Result C — C1/C2 work; all E models fail

Interpretation:

> Explicit responsibility-weighted memory expression is sufficient, but it does not arise from the tested minimal neural architectures.

## Result D — E3 uniquely works

Interpretation:

> Error-driven inference is the critical missing computation.

## Result E — E4 uniquely works

Interpretation:

> The key requirement is architectural factorization of shared sequence dynamics and context-specific modulation.

## Result F — Mixed improves but Novel does not

Interpretation:

> Combined-context arbitration and novel-context generalization are distinct problems.

## Result G — Novel moves toward neutral/shared state rather than midpoint

Interpretation:

> Novel context triggers fallback to shared task structure rather than A/B averaging.

## Result H — none work

Interpretation:

> The current hypothesis class is insufficient; do not continue tuning latent size or capacity. Return to the human retrieval analyses for a sharper target computation.

---

# 32. Final scientific criterion

The strongest outcome would be:

> A model trained only on familiar A/B experience learns distinct context-specific sequence states and, when presented with an untrained ambiguous context, spontaneously expresses behavior intermediate between the familiar states without any midpoint target, interpolation loss, or probe-specific tuning.

Formally:

\[
oxed{
	ext{intermediate expression is an emergent consequence of learned context inference}
\]

rather than:

\[
oxed{
	ext{intermediate expression is built into the output rule}.
\]
