# PRD: Episodic Meta-Learning of Context-Generalization Priors in SRNs
## Capacity × Learning Objective × Structured Context Ecology

## 0. Goal

Test whether the failure of ordinary pretraining reflects the **learning objective** rather than the availability of structured experience.

The central hypothesis is:

\[
\boxed{
\text{episodic pressure to generalize within newly sampled task families}
\rightarrow
\text{a learned contextual-generalization prior}
}
\]

The model must learn this prior **without ever seeing the human Mixed or Novel probes during training** and without any loss that rewards midpoint behavior, A/B averaging, corridor occupancy, or human-like RT.

The primary question is:

> Can the same SRN family that learns strong A/B context-specific representations acquire, through episodic meta-learning, a reusable prior that improves zero-shot transfer to unseen members of a new contextual family?

The secondary question is:

> Does this require more capacity than the original 8-unit SRN?

---

# 1. Scientific design

The first experiment should isolate three factors:

1. **Learning objective**
   - ordinary pretraining
   - episodic meta-learning

2. **Ecological structure**
   - structured smooth context→task covariance
   - matched shuffled control

3. **Model capacity**
   - 8
   - 32
   - 64
   - 128 hidden units

The first decisive comparison is:

```text
P2_ordinary
vs
P2_meta
vs
P1_shuffled_meta
```

across:

```text
H ∈ {8, 32, 64, 128}
```

Do not add P3/P4/P5 in this initial experiment.

---

# 2. Fixed architecture family

Use a tanh Elman SRN:

\[
h_t
=
\tanh
(
W_xx_t + W_hh_{t-1}+b_h
).
\]

Output:

\[
p_t
=
\operatorname{softmax}(W_oh_t+b_o).
\]

Hidden sizes:

```text
8
32
64
128
```

Primary candidate:

```text
64 units
```

Roles:

| Hidden size | Purpose |
|---:|---|
| 8 | continuity with existing paper model |
| 32 | moderate-capacity control |
| 64 | primary capacity candidate |
| 128 | ceiling / serious capacity test |

Do not add LSTM/GRU/Transformer in this PRD.

If all SRN capacities fail after the meta-learning gate is implemented correctly, gated architectures may be tested in a later PRD.

---

# 3. Initialization

Retain the validated High-scale initialization concept.

Primary:

```text
scale = 2.0
```

For each hidden width, preserve the repository's existing fan-in-aware initialization rule if one exists.

Log:

```text
input-weight std
recurrent-weight std
output-weight std
recurrent spectral radius
hidden activation norm
```

Do not silently change effective initialization with width.

If the repository does not currently use fan-in scaling, preserve the existing rule and simply report realized scale statistics.

---

# 4. Pretraining ecology

Use the previously defined **P2 smooth contextual covariance** generator.

The key improvement is not a new ecology. It is a new **episodic support/query objective**.

Each synthetic family \(F_e\) contains:

```text
one 8-item sequence family
one latent contextual dimension z
multiple visible context realizations
task realizations that vary smoothly with z
```

The network is never given z directly.

---

# 5. Synthetic family generation

## 5.1 Sequence

Per episode/family:

```text
sequence length = 8
response alphabet = 4
```

Sample a new base sequence subject to:

```text
>= 3 distinct response symbols
not identical to the human target sequence
no trivial all-repeat structure
```

## 5.2 Latent context

Sample:

\[
z_j \in [-1,1]
\]

for 6-8 contexts per family.

Do not always use the same fixed z grid.

Primary:

```text
sample 6 z values uniformly
sort them
require minimum spacing = 0.15
```

## 5.3 Visible context

Generate:

\[
c(z)=f_c(z)+\epsilon_c
\]

and:

\[
L(z)=f_L(z)+\epsilon_L.
\]

Use:

```text
cue features
full 8-position layout
```

with human-like geometry constraints.

Layouts should preserve the general human property:

```text
strong shared left-to-right ordinal scaffold
modest vertical/context-specific variation
```

Do not use the old radically different synthetic 2-D trajectories.

## 5.4 Task realization

Generate task statistics smoothly from z:

\[
\theta(z)
=
\theta_0
+
B\phi(z).
\]

Primary implementation should alter task difficulty/prediction structure without changing sequence length.

Preferred mechanisms:

```text
sparse transition-specific predictive uncertainty
transition-specific execution-independent nuisance statistics
or sparse context-sensitive token-transition probabilities
```

The same latent z must jointly control:

```text
visible context
task realization
```

in the structured condition.

Do not use RT targets.

---

# 6. Structured versus shuffled ecology

## Structured P2

Preserve:

\[
\text{context similarity}
\leftrightarrow
\text{task similarity}.
\]

Nearby z values should induce:

```text
similar cue/layout vectors
similar task realizations
```

## Shuffled P1 control

For each structured family:

1. generate the family once;
2. preserve:
   ```text
   sequence
   z values
   visible contexts
   task realizations
   task difficulty marginals
   support/query sizes
   total optimizer steps
   ```
3. randomly permute task realizations across visible context instances.

This destroys:

\[
P(\text{task similarity}\mid\text{context similarity})
\]

while preserving all marginals.

Use paired random seeds.

---

# 7. Training conditions

For each hidden size run:

```text
A. P2_ordinary
B. P2_meta
C. P1_shuffled_meta
```

Optional reference:

```text
D. P1_shuffled_ordinary
```

only if computationally cheap.

The essential factorial is:

```text
structured vs shuffled
×
ordinary vs meta
×
hidden size
```

but the minimum implementation must include A-C.

---

# 8. Ordinary pretraining condition

`P2_ordinary` uses exactly the structured P2 families but optimizes standard next-token performance on all available training contexts.

No support/query meta-objective.

Within a family:

```text
ordinary SGD/Adam updates
```

Across families:

```text
weights persist
hidden state resets
```

Purpose:

> replicate the previous finding that structured exposure alone does not reliably create a useful prior.

Total number of external training sequences and optimizer updates must be matched as closely as possible to meta-training.

---

# 9. Episodic meta-learning condition

Use **first-order MAML (FOMAML)** for the primary implementation.

Do not implement second-order MAML initially.

Each episode samples a completely new family.

---

# 10. One meta-training episode

## 10.1 Episode sampling

Generate a new family with:

```text
6 contextual members
```

Example latent values:

\[
[-0.88,-0.54,-0.18,0.11,0.47,0.83].
\]

The exact positions vary every episode.

## 10.2 Support selection

Randomly choose:

```text
3 of 6 contexts
```

as support.

Require support to span a meaningful portion of context space.

Primary rule:

```text
at least one support context from lower third
at least one from upper third
third support sampled from remaining contexts
```

This avoids pathological support sets clustered at one endpoint.

## 10.3 Support experience

Per support context:

```text
2 full 8-item sequences
```

Total:

```text
6 support sequences
```

Compute support prediction loss:

\[
L_{\text{support}}.
\]

Perform:

```text
3 inner-loop gradient steps
```

with fixed inner learning rate:

```text
alpha = preregistered constant
```

Primary starting value:

```text
alpha = 0.01
```

Do not tune alpha using human Mixed/Novel performance.

Adapted parameters:

\[
\theta'_e
=
\theta
-
\alpha
\nabla_\theta L_{\text{support}}.
\]

## 10.4 Query selection

Use:

```text
1 familiar/copy query
4 unseen-context queries
```

### Familiar query

Sample one sequence from a support context.

Purpose:

```text
verify that within-episode learning actually retained support information
```

### Unseen queries

Sample from contexts not used for inner-loop adaptation.

Do not always choose a midpoint.

Across episodes include query contexts:

```text
near support
between supports
asymmetric / skewed
farther from support
```

This prevents the network from learning a trivial "midpoint" heuristic.

## 10.5 Query loss

Compute:

\[
L_{\text{query}}
=
L_{\text{unseen}}
+
\lambda_{\text{copy}}L_{\text{copy}}.
\]

Primary:

```text
lambda_copy = 0.25
```

No RT loss.

No interpolation loss.

No human data.

## 10.6 Meta-update

Using FOMAML:

\[
\theta
\leftarrow
\theta
-
\beta
\nabla_{\theta'}
L_{\text{query}}.
\]

Ignore second-order derivatives through the inner update.

Primary outer optimizer:

```text
Adam
```

Primary initial outer LR:

```text
beta = 1e-3
```

Use gradient clipping consistent with repository conventions.

---

# 11. What is meta-learned

Primary:

```text
input weights
recurrent weights
output prediction weights
biases
```

Do **not** meta-train the monotonic RT readout.

Do not include RT in:

```text
support loss
query loss
condition selection
internal validation gates
```

The pretraining stage should install a structural prediction prior, not a speed prior.

---

# 12. RT handling

During synthetic meta-pretraining:

```text
RT readout disabled / excluded from optimization
```

At transition to the human task:

1. retain pretrained sequence-model weights;
2. initialize/reset the validated two-parameter monotonic execution readout;
3. train the human A/B task using the same execution mechanism and objective as the existing successful model.

This isolates:

\[
\text{pretrained structural prior}
\]

from:

\[
\text{RT expression mechanism}.
\]

---

# 13. Pretraining amount

## Engineering smoke test

```text
1 seed
100 episodes
H = 8 and 64
```

Verify:

```text
support adaptation works
query loss backpropagates
structured/shuffled generators are paired correctly
no parameter leaks
```

Do not interpret results.

## Pilot

Run:

```text
5 paired seeds
5000 meta-training episodes
H ∈ {8,32,64,128}
```

for:

```text
P2_ordinary
P2_meta
P1_shuffled_meta
```

Do not stop after H=8 failure.

All four capacities must be tested.

## Confirmatory

Only capacities/conditions passing the synthetic prior gate:

```text
20 paired seeds
50,000 meta-training episodes
```

If computationally feasible, extend the strongest condition to:

```text
100,000 episodes
```

as a robustness check, motivated by the scale of prior meta-learning work.

Do not use human probe results to choose between 50k and 100k.

---

# 14. Synthetic validation battery

The human task remains firewalled until a model demonstrates the intended prior on unseen synthetic families.

Validation families must be generated from new seeds.

No validation query contributes gradients.

---

# 15. V1 — Unseen-family intermediate transfer

For a new family:

Inner-loop adapt on:

```text
z ≈ -0.8
z ≈ -0.2
z ≈ +0.8
```

using exactly the same number of support sequences as training.

Freeze adapted weights.

Probe unseen contexts across the family.

Primary outcomes:

```text
query prediction NLL
query accuracy
deficit relative to nearest familiar/support context
```

Structured-meta should outperform:

```text
structured-ordinary
shuffled-meta
```

---

# 16. V2 — Similarity-gradient test

After adaptation to support contexts, probe:

```text
multiple unseen z values spanning [-1,1]
```

Measure whether transfer quality respects contextual relatedness.

Primary metric:

\[
S_{\text{gradient}}
=
\operatorname{corr}
(
d_{\text{context}},
\text{prediction deficit}
).
\]

Expected:

```text
larger context distance -> larger prediction deficit
```

Use sign convention so higher score means better structure-respecting generalization.

---

# 17. V3 — Non-midpoint query test

Explicitly evaluate unseen queries that are:

```text
25/75 between supports
10/90 between supports
outside the central midpoint
```

Purpose:

> demonstrate that successful meta-learning is not a learned "average the endpoints" trick.

No midpoint-specific success criterion.

---

# 18. V4 — Copy/familiar retention

Probe familiar support contexts after inner-loop adaptation.

Require:

```text
copy performance remains strong
```

A model that generalizes by forgetting the support mappings fails.

---

# 19. V5 — New-family robustness

Evaluate across:

```text
100 unseen synthetic families / seed
```

for confirmatory models.

Report:

```text
mean query NLL
family-level variance
context-distance gradient
copy accuracy
```

---

# 20. Synthetic prior gate

A model may advance to the human task only if:

## G1 — Meta objective matters

\[
P2_{\text{meta}}
>
P2_{\text{ordinary}}
\]

on unseen-family query NLL.

## G2 — Structure matters

\[
P2_{\text{meta}}
>
P1_{\text{shuffled-meta}}
\]

on unseen-family query NLL.

## G3 — Seed consistency

At least:

```text
4/5 pilot seeds
```

show the expected P2_meta advantage over both controls.

## G4 — Similarity prior

V2 gradient is positive and materially larger than shuffled-meta.

## G5 — Not midpoint-specific

V3 performance generalizes across asymmetric queries.

## G6 — Familiar retention

V4 copy performance remains non-pathological.

Do not advance a model based solely on mean query loss.

---

# 21. Capacity interpretation

Compare:

```text
H8
H32
H64
H128
```

Do not normalize away genuine capacity effects.

Report:

```text
synthetic query NLL
similarity-gradient score
copy accuracy
parameter count
training stability
```

Possible outcomes:

### H8 fails, H64/H128 succeed

Interpretation:

> original SRN capacity limits acquisition of the structural prior.

### all capacities fail

Interpretation:

> episodic objective is insufficient under the Elman architecture/ecology.

### H8 succeeds

Interpretation:

> the original small SRN can learn the prior if trained with the correct objective.

### H128 overfits / worsens

Interpretation:

> capacity is not monotonically beneficial; identify the smallest successful model.

Primary model selection principle:

```text
choose the smallest capacity that robustly passes the synthetic prior gate
```

not the model with the best human probe behavior.

---

# 22. Human-task transfer phase

Only run for:

```text
smallest passing P2_meta capacity
one larger passing capacity if available
matched P1_shuffled_meta control
matched P2_ordinary control
P0 random-init baseline
```

This keeps the human experiment compact.

---

# 23. Transition into the human task

For every selected pretrained checkpoint:

```text
reset recurrent hidden state
reset optimizer state
retain pretrained SRN weights
initialize/reset monotonic execution readout
```

Then train identically on:

```text
G2_shared_x / human SVG geometry
same 8-item human sequence
A10/B10 alternation
170 external sequences
same human-task learning rate
same execution objective
same RT mapping
```

No Mixed/Novel training.

---

# 24. Human familiar gates

Before interpreting probes require:

```text
sequence NLL decreases
High-scale A/B hidden separation remains strong
context-yoked RT sawtooth survives
majority Cross/Split state-space winner
confidence -> RT slope < 0
```

Any model that improves probe transfer by becoming Lump fails.

---

# 25. Human probe checkpoints

Run frozen A/B/Mixed/Novel probes at:

```text
20 sequences
50 sequences
90 sequences
170 sequences
```

No optimizer updates during probes.

Purpose:

> determine whether the learned prior is expressed early rather than appearing only after extensive target-task learning.

The primary checkpoint remains:

```text
170 sequences
```

for comparability with the existing model.

The early checkpoints are mechanistic/temporal analyses.

---

# 26. Primary Mixed outcomes

Within each seed canonicalize:

```text
Fast = lower mean frozen A/B RT
Slow = higher mean frozen A/B RT
```

Swap whole profiles only.

Primary:

```text
fraction of Mixed transitions between Fast and Slow
```

Also report:

```text
Mixed NLL deficit over worse endpoint
Mixed RT deficit over worse endpoint
profile-fit q
transition-specific q_k
hidden off-corridor distance
inside A/B corridor fraction
```

All are post hoc.

No q or midpoint metric enters training.

---

# 27. Primary Novel outcomes

Do not require Novel to be a midpoint.

Report:

```text
Novel NLL deficit
Novel RT deficit
next-token accuracy
distance to A/B corridor
distance to shared/context-neutral representation
```

Primary interpretation:

```text
Novel may reuse shared task structure rather than average A/B.
```

---

# 28. Central human-task contrast

The strongest test is:

\[
\boxed{
P2_{\text{meta}}
>
P1_{\text{shuffled-meta}}
}
\]

on:

```text
first-exposure / early Mixed transfer
```

while:

```text
A/B differentiation remains intact
```

Secondary:

\[
P2_{\text{meta}}
>
P2_{\text{ordinary}}.
\]

This separates:

```text
structured experience
```

from:

```text
episodic pressure to extract/use that structure.
```

---

# 29. Statistical analysis

For synthetic validation:

```text
metric ~ training_condition * log2(hidden_size) + (1|seed)
```

Primary metric:

```text
unseen-family query NLL
```

Secondary:

```text
similarity-gradient score
copy accuracy
```

For human transfer:

```text
metric ~ pretraining_condition * checkpoint + (1|seed)
```

Separate models for:

```text
Mixed
Novel
```

Report:

```text
paired mean differences
bootstrap 95% CIs
standardized effects
```

Do not select conclusions from p-values alone.

---

# 30. Mechanistic follow-ups

Run only if P2_meta passes synthetic validation and improves human transfer.

## M1 — Meta-prior retention

Re-run the synthetic V1/V2 battery at:

```text
before human training
after 20 human sequences
after 50
after 90
after 170
```

using disposable frozen copies.

Question:

```text
does A/B training overwrite the meta-learned prior?
```

## M2 — Weight-subsystem ablation

Reinitialize separately:

```text
input weights
recurrent weights
output weights
```

before human-task training.

Identify where the prior is stored.

## M3 — Hidden geometry

Compare:

```text
A/B separation
ordinal subspace alignment
Mixed corridor distance
Novel shared-state distance
```

between meta-trained and controls.

---

# 31. Strong falsification criteria

## F1 — P2_meta does not beat P2_ordinary

Interpretation:

> episodic pressure adds no structural prior beyond ordinary pretraining.

## F2 — P2_meta does not beat shuffled-meta

Interpretation:

> performance reflects generic meta-training rather than learning context–task covariance.

## F3 — only midpoint queries improve

Interpretation:

> model learned a narrow interpolation heuristic rather than general structural inference.

## F4 — synthetic prior is learned but human A/B splitting collapses

Interpretation:

> prior favors generalization at the expense of specialization.

## F5 — synthetic prior is learned but human Mixed/Novel do not improve

Interpretation:

> learned prior does not transfer to the target task geometry.

## F6 — only 128-unit model succeeds

Interpretation:

> result may depend on capacity far beyond the original SRN; treat as existence proof, then test compression separately.

---

# 32. TDD requirements

Use RED -> GREEN -> REFACTOR.

## T1 — Support/query separation

Assert no query sequence is used for inner-loop adaptation.

## T2 — Novel family each episode

Episode family parameters must differ across episodes.

## T3 — Paired structured/shuffled families

Same source family components before permutation.

## T4 — No z input

Latent z must never enter the network input.

## T5 — No RT meta-loss

Synthetic pretraining loss contains only prediction/task terms.

## T6 — No interpolation loss

Fail if training code references:

```text
midpoint
between_endpoint
corridor
q_target
lambda_target
Mixed
Novel
```

## T7 — FOMAML only

Primary run must not accidentally retain second-order graph through inner steps.

## T8 — Query gradient reaches meta-parameters

Unit test with a tiny episode must show nonzero outer gradients.

## T9 — Frozen synthetic validation

Parameter hash identical before/after V1-V5.

## T10 — Human firewall

No human Mixed/Novel files produced before synthetic prior gate passes.

## T11 — Width-specific logging

Log parameter counts and initialization statistics for H8/H32/H64/H128.

## T12 — RT reset before human task

Execution-readout parameters must not carry synthetic-pretraining gradients.

---

# 33. Required outputs

## Config

```text
episodic_meta_config.json
meta_capacity_grid.json
synthetic_family_generator_config.json
```

## Pretraining

```text
meta_training_curves.csv
ordinary_training_curves.csv
meta_episode_manifest.csv
meta_support_query_manifest.csv
```

## Synthetic validation

```text
synthetic_v1_intermediate.csv
synthetic_v2_similarity_gradient.csv
synthetic_v3_asymmetric_queries.csv
synthetic_v4_copy_retention.csv
synthetic_v5_family_robustness.csv
synthetic_prior_gate.csv
capacity_comparison.csv
```

## Human transfer, only if reached

```text
meta_human_training.csv
meta_human_state_space.csv
meta_human_hidden_differentiation.csv
meta_human_probe_metrics.csv
meta_human_checkpoint_metrics.csv
meta_human_interpolation.csv
meta_human_hidden_corridor.csv
```

## Report

```text
EPISODIC_META_LEARNING_CAPACITY_REPORT.md
```

---

# 34. Required figures

## Figure 1 — Episode schematic

Show:

```text
new family
-> support contexts
-> 3 inner updates
-> familiar query + unseen queries
-> outer meta-update
```

## Figure 2 — Structured ordinary vs structured meta vs shuffled meta

Plot unseen-family query NLL by hidden size.

## Figure 3 — Capacity threshold

x:

```text
8,32,64,128
```

y:

```text
meta-generalization advantage
```

## Figure 4 — Similarity gradients

Context distance vs prediction deficit for each condition/capacity.

## Figure 5 — Asymmetric query generalization

Show performance for non-midpoint query locations.

## Figure 6 — Copy vs generalization

Confirm meta-learning does not trade away familiar retention.

## Figure 7 — Human A/B phenotype

Only if synthetic gate passed.

## Figure 8 — Mixed behavior across checkpoints

```text
20 / 50 / 90 / 170
```

## Figure 9 — Novel behavior across checkpoints

## Figure 10 — Differentiation-transfer frontier

x:

```text
A/B hidden separation
```

y:

```text
Mixed between-endpoint fraction or NLL deficit
```

---

# 35. Execution order

## Stage 0 — implementation validation

```text
H8 + H64
1 seed
100 episodes
```

No scientific interpretation.

## Stage 1 — complete capacity pilot

```text
H8/H32/H64/H128
5 seeds
5000 episodes
P2_ordinary
P2_meta
P1_shuffled_meta
```

All capacities run regardless of H8 outcome.

## Stage 2 — synthetic gate

Apply V1-V5.

## Stage 3 — confirmatory meta-training

Passing cells only:

```text
20 seeds
50,000 episodes
```

Optional fixed 100k robustness on strongest passing condition.

## Stage 4 — human task

Advance smallest successful capacity + one larger successful capacity and matched controls.

## Stage 5 — mechanism

Only if human transfer improves.

---

# 36. Decision matrix

## Outcome A — H8 fails, H64 succeeds under P2_meta

Interpretation:

> correct episodic learning pressure is useful, but the original 8-unit network lacks sufficient capacity to acquire the prior.

## Outcome B — all widths improve under P2_meta

Interpretation:

> meta-learning objective, not capacity, was the primary missing ingredient.

## Outcome C — larger widths help ordinary and meta equally

Interpretation:

> capacity, not episodic learning, explains the gain.

## Outcome D — shuffled-meta matches structured-meta

Interpretation:

> generic meta-learning helps, but context→task covariance was not learned as a reusable prior.

## Outcome E — structured-meta wins synthetic tests but not human transfer

Interpretation:

> the prior is learnable but does not transfer to the human contextual geometry.

## Outcome F — structured-meta improves early human Mixed while preserving A/B split

Target result:

\[
\boxed{
\text{episodically meta-learned context-generalization prior}
\rightarrow
\text{immediate human-like ambiguous retrieval}
}
\]

without explicit averaging.

---

# 37. Final scientific criterion

The strongest result is not simply lower loss.

It is:

> After meta-training across unrelated structured contextual families, an SRN adapts to a completely new A/B task, still forms separate context-specific skills, and then spontaneously expresses a previously unseen ambiguous context as a structured reuse of those familiar skills.

This must occur with:

```text
no Mixed/Novel training
no RT pretraining
no midpoint objective
no explicit A/B output averaging
```

The experiment therefore directly tests whether **episodic structural meta-learning plus sufficient capacity** can provide the inductive prior that ordinary SRN training failed to acquire.
