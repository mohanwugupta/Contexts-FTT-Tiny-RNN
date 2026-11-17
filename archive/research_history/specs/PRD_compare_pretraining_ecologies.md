# PRD: Comparing Pretraining Ecologies for Human-Like Context Generalization in SRNs

## 0. Objective

We want to determine whether **prior structured experience** can give the existing SRN a useful inductive bias for the human A/B → Mixed/Novel task.

The central hypothesis is:

\[
\boxed{
\text{prior experience with structured context–task relationships}
\rightarrow
\text{a prior over how new contexts should reuse learned skills}
}
\]

The critical requirement is that human-like probe behavior must be **emergent**. No model may be trained on the human Mixed or Novel probes, and no loss may reward midpoint behavior, interpolation, corridor occupancy, or A/B averaging.

The experiment compares several theoretically motivated pretraining ecologies rather than assuming one mechanism.

---

# 1. Scientific motivation

The candidate pretraining methods are motivated by different literatures:

| Method | Core idea | Literature motivation |
|---|---|---|
| P0 No pretraining | baseline | existing SRN |
| P1 Shuffled matched control | same exposure without stable context→task statistics | causal control |
| P2 Smooth contextual covariance | similar contexts tend to recruit similar versions of a skill | motor structure learning; contextual generalization |
| P3 Hybrid smooth + boundaries | learners should generalize within families but differentiate across strong contextual boundaries | contextual inference / multiple motor memories |
| P4 Compositional cue ecology | independently meaningful contextual components can be reused jointly in new combinations | compositional structure learning / meta-learning |
| P5 Combined structured ecology | smooth contextual covariance + compositional reuse + occasional boundaries | integrated hypothesis |

Conceptual references to motivate the paper-level interpretation:

- Braun et al. — structural learning of motor task distributions
- Krakauer et al. — prior contextual statistics shape motor generalization
- Sarwary et al. — graded expression of context-linked motor memories
- Heald, Lengyel & Wolpert — contextual inference governs motor-memory expression
- Lake & Baroni — compositional inductive biases can arise from task distributions

This PRD does not require implementing any of those published models directly. It tests whether their common ecological principle is sufficient for the existing SRN.

---

# 2. Fixed learner

Keep the post-PNAS SRN architecture fixed across all conditions.

```text
8-unit tanh Elman SRN
High initialization scale = 2.0
same sequence prediction head
same two-parameter monotonic execution/RT readout
same optimizer family
same learning rates
same external training format
```

Do **not** add:

```text
attention
retrieval modules
explicit Bayesian inference
mixture-of-experts
larger hidden state
human RT targets
midpoint losses
probe-specific objectives
```

The purpose of this experiment is to isolate the effect of prior ecology.

---

# 3. Common pretraining world

Each pretraining world is a miniature contextualized sequence-learning problem.

## 3.1 Shared skill

Use:

```text
sequence length = 8
response alphabet = 4
```

Each world has one latent sequence family with a shared ordinal backbone.

## 3.2 Context representation

Each world contains a latent context variable:

\[
z \in [-1,1].
\]

Visible context information has two components:

```text
cue vector
8-position screen layout
```

The layout should be generated from the empirical human-layout family rather than the old horizontal-vs-vertical synthetic extremes.

Recommended implementation:

1. Use the human SVG coordinate templates already extracted.
2. Parameterize a low-dimensional layout manifold from those templates.
3. Generate world-specific layouts by interpolation plus small jitter within the observed human range.

The full 8-position layout remains visible at each step.

## 3.3 Skill realization

The sequence order remains shared.

Context can influence:

```text
local transition difficulty
prediction statistics
or sparse context-sensitive task modifiers
```

depending on the pretraining method.

No condition may alter the basic next-token learning objective.

---

# 4. Conditions

## P0 — No pretraining

Random initialization.

This is the existing High-scale baseline.

No synthetic worlds.

---

## P1 — Matched shuffled pretraining

This is the exposure control.

It receives exactly the same:

```text
number of worlds
number of sequences
cue marginals
layout marginals
task-difficulty marginals
optimizer updates
```

as the corresponding structured condition.

But the relationship between:

```text
context similarity
layout similarity
task similarity
```

is independently shuffled within each world or across worlds.

Thus P1 learns the same amount of material without a stable ecological law.

For every structured world, generate a paired shuffled version using the same random seed and source components.

---

## P2 — Smooth contextual covariance

### Hypothesis

Experience teaches:

\[
\boxed{
\text{similar context} \Rightarrow \text{similar skill realization}
}
\]

### World generation

Sample 5 latent contexts:

\[
z\in\{-1,-0.5,0,0.5,1\}.
\]

Generate cue vectors smoothly from \(z\):

\[
c(z)=f_c(z)+\epsilon_c.
\]

Generate layouts smoothly from the same \(z\):

\[
L(z)=f_L(z)+\epsilon_L.
\]

Generate task variation from the same latent coordinate:

\[
\theta(z)=\theta_0 + B\phi(z).
\]

Primary implementation:

```text
theta(z) changes sparse transition difficulty / local next-token uncertainty
smoothly with z
```

Use a low-rank basis B so nearby z values induce related task realizations.

The SRN is never told z.

### Training exposure

During pretraining worlds, train on all five z values.

### Intended learned prior

Across worlds, the network repeatedly experiences:

```text
nearby cue/layout states -> nearby task states
distant cue/layout states -> more different task states
```

---

## P3 — Hybrid smooth + contextual boundaries

### Hypothesis

Humans do not merely interpolate. They both:

```text
generalize within related contexts
and
split when evidence indicates a context boundary
```

P3 therefore teaches a piecewise ecology.

### World generation

Use the same latent coordinate \(z\), but define two regimes:

\[
z < b \Rightarrow \text{family A}
\]

\[
z \ge b \Rightarrow \text{family B}
\]

where \(b\) varies across worlds.

Within each regime:

\[
\text{context similarity predicts task similarity}.
\]

Across the boundary:

```text
cue/layout similarity can remain moderate
but task mapping changes discontinuously
```

Example:

```text
z = -1,-.5,-.25 -> smooth family 1
z = +.25,+.5,+1 -> smooth family 2
```

This condition is designed to teach:

\[
\text{generalize when evidence supports continuity}
\]

but:

\[
\text{differentiate when evidence supports a new latent context}.
\]

This is the most directly motivated condition for preserving A/B splitting while improving Mixed/Novel expression.

---

## P4 — Compositional cue ecology

### Hypothesis

Prior experience teaches that independently meaningful contextual factors can be reused jointly.

Use two latent binary factors:

\[
c=[c_1,c_2].
\]

Across pretraining worlds expose all four configurations:

```text
[0,0]
[1,0]
[0,1]
[1,1]
```

with structured task rule:

\[
\theta_{11}
=
\theta_{00}
+
\Delta_1
+
\Delta_2.
\]

Use sparse modifiers:

```text
Delta_1 -> P2/P3
Delta_2 -> P5/P6
```

and vary the specific response/task realization across worlds.

This condition gives genuine experience in which conjunctions are useful.

### Critical validation

In **new validation worlds**, train only:

```text
[0,0], [1,0], [0,1]
```

then freeze-test:

```text
[1,1]
```

The network must generalize composition to a new task family.

No conjunction target is provided in the validation family.

---

## P5 — Combined structured ecology

P5 combines the strongest ecological principles:

1. smooth context→task covariance,
2. occasional latent-context boundaries,
3. compositional contextual factors.

The training distribution contains a randomized mixture:

```text
50% smooth-family worlds
25% boundary worlds
25% compositional worlds
```

Do not tune these proportions against the human Mixed/Novel outcome.

The intent is to test whether a richer lifetime-like task distribution installs a more general prior than any single rule.

---

# 5. Pretraining amount

Use the same total exposure for P1–P5.

## Pilot

```text
5 paired seeds
500 synthetic worlds
```

with a fixed total number of external training sequences.

Recommended:

```text
30–50 sequences/world
```

depending on world type.

Normalize total optimizer steps across conditions.

## Confirmatory

Only for conditions that pass the internal validation gate:

```text
20 paired seeds
2000 synthetic worlds
```

Do not run confirmatory human transfer on failed conditions.

---

# 6. Common internal validation battery

The key improvement over previous PRDs is that all pretraining methods are evaluated on the **same unseen-family tests**.

No validation world contributes gradients.

---

## V1 — Intermediate-context transfer

Generate a completely new smooth family.

Train the pretrained network within that world only on:

\[
z=-1,\quad z=+1.
\]

Freeze weights.

Probe:

\[
z=0.
\]

Measure:

```text
prediction NLL
next-token accuracy
normalized RT
hidden-state location relative to endpoint corridor
```

Primary success measure:

```text
prediction deficit relative to the better/worse endpoints
```

and whether the probe lies in the behavioral/representational interval implied by the new family.

The model is not rewarded for midpoint behavior during pretraining evaluation.

---

## V2 — Near-intermediate transfer gradient

In a new family train endpoints:

```text
z=-1,+1
```

and freeze-test:

```text
z=-.75,-.5,-.25,0,.25,.5,.75
```

Test whether transfer varies monotonically with context similarity.

Primary analysis:

\[
\text{probe deficit} \sim |z-z_{\text{nearest endpoint}}|.
\]

This distinguishes a real similarity prior from a one-off midpoint trick.

---

## V3 — Boundary discrimination

Generate a new piecewise family with a latent boundary.

Expose training examples on both sides.

Freeze-test novel contexts:

```text
one within-family interpolation
one cross-boundary near-neighbor
```

A successful prior should:

```text
generalize strongly within a regime
but less strongly across a true boundary
```

This is crucial because indiscriminate averaging is a failure.

---

## V4 — Held-out conjunction

Applicable to all conditions as a common compositional diagnostic.

New family:

Train:

```text
[0,0], [1,0], [0,1]
```

Freeze-test:

```text
[1,1]
```

Score:

```text
overall accuracy
modified-position accuracy
prediction NLL
```

This reveals whether contextual structure learning transfers to compositional reuse even outside P4.

---

# 7. Internal screening metrics

For each condition compute:

## Smooth transfer score

\[
S_{\text{smooth}}
=
-\text{mean intermediate prediction deficit}.
\]

Higher is better.

## Similarity-gradient score

Use correlation:

\[
S_{\text{gradient}}
=
-\operatorname{corr}
(
\text{context distance},
\text{transfer quality}
).
\]

Choose sign convention so higher means better similarity-respecting transfer.

## Boundary selectivity

\[
S_{\text{boundary}}
=
\text{within-family transfer}
-
\text{cross-boundary transfer}.
\]

Higher is better.

## Compositional score

\[
S_{\text{comp}}
=
Acc_{\text{modified}}([1,1]).
\]

---

# 8. Pretraining pass gate

A condition can advance to the human task only if, relative to paired shuffled P1:

1. It improves V1 intermediate transfer at the prediction level.
2. It shows a positive V2 similarity gradient.
3. It does **not** collapse V3 boundary discrimination.
4. Familiar/task learning is not impaired.
5. At least 4/5 pilot seeds show the primary V1 effect in the expected direction.

P4 additionally requires improved V4 modified-position accuracy.

P5 must pass V1 and V3; V4 is secondary.

Do not choose the winner based on human Mixed/Novel results.

---

# 9. Pretraining model selection

After confirmatory runs, rank passing conditions using a preregistered composite:

\[
S
=
z(S_{\text{smooth}})
+
z(S_{\text{gradient}})
+
z(S_{\text{boundary}})
+
0.5z(S_{\text{comp}}).
\]

Use this score only among conditions that pass all hard gates.

Advance:

```text
top 2 structured pretraining conditions
P1 matched shuffled control
P0 no-pretraining baseline
```

to the human task.

This prevents a large human-task fishing expedition.

---

# 10. Human-task transfer phase

Train all selected models identically on the human-matched task.

Use:

```text
G2_shared_x / exact human SVG geometry
same fixed 8-item sequence
A10 / B10 alternation
170 external training sequences
High scale = 2.0
same monotonic execution readout
```

Reset optimizer state before human-task training.

Do not reset learned weights.

No Mixed or Novel exposure.

---

# 11. Familiar A/B gates

Before interpreting probes, require:

```text
successful sequence learning
context-yoked RT learning
A/B hidden-state separation
majority Cross/Split state-space phenotype
negative confidence -> RT relationship
```

If a pretrained condition improves Mixed by collapsing A/B differentiation, it fails.

---

# 12. Primary human outcomes

## H1 — First Mixed probe

Primary metric:

```text
transition-level fraction lying between canonicalized A/B endpoints
```

Also report:

```text
whole-profile q
prediction NLL deficit
RT deficit
hidden-state A/B corridor location
off-corridor distance
```

The key test is **first exposure**.

---

## H2 — Novel probe

Do not require Novel to be exactly at the A/B midpoint.

Primary metrics:

```text
prediction NLL deficit over familiar endpoints
distance to context-neutral/shared representation
behavioral profile relative to A/B/shared states
```

This allows Novel to express a default/shared prior rather than forced interpolation.

---

## H3 — Transition structure

For Mixed and Novel compute:

\[
q_k
=
\frac{RT_{\text{probe},k}-RT_{B,k}}
{RT_{A,k}-RT_{B,k}}
\]

only as a descriptive diagnostic.

Do not use q in training.

Compare:

```text
global q
transition-specific q_k
human transition-profile structure
```

---

# 13. Key confirmatory contrast

The most important human-task contrast is:

\[
\boxed{
P_{\text{structured}}
>
P_{\text{shuffled}}
}
\]

on **first-probe Mixed transfer**, while familiar A/B splitting remains intact.

P0 establishes the raw-initialization baseline.

This is the causal test that the learned prior comes from structured pretraining rather than extra optimization.

---

# 14. Mechanistic follow-ups

Run only if a structured pretraining condition improves human transfer.

## M1 — Weight-subsystem ablation

Reinitialize separately:

```text
input weights
recurrent weights
prediction head
execution readout
```

before human-task training.

Question:

```text
where is the prior stored?
```

---

## M2 — Hidden representational geometry

Before and after human A/B training compare:

```text
context similarity geometry
A/B separation
Mixed/Novel corridor location
shared ordinal subspace
```

---

## M3 — Prior retention over training

Probe pretrained structural-generalization battery:

```text
before human A/B training
after 20 sequences
after 80 sequences
after 170 sequences
```

without updates during probe.

Question:

```text
does the learned prior survive task-specific learning?
```

---

# 15. TDD / RED -> GREEN requirements

## T1 — Matched exposure

P1–P5 must have identical:

```text
total external sequences
optimizer steps
cue dimensionality
layout dimensionality
```

within a run.

## T2 — No human-probe leakage

Fail if pretraining code contains human task identifiers:

```text
Mixed
Novel
Maze
Melody
human participant IDs
```

except in post-pretraining transfer code.

## T3 — No interpolation objective

Fail if any training loss references:

```text
midpoint
between endpoints
corridor
q=.5
A/B average
Mixed RT
Novel RT
```

## T4 — Paired shuffled control

Every structured synthetic world has a reproducible paired shuffled world using the same source components.

## T5 — Frozen validation

Hash all model parameters before and after V1–V4.

They must be identical.

## T6 — Human firewall

No human transfer files may be created before pretraining condition selection is finalized.

## T7 — Human familiar gate

No Mixed/Novel interpretation report may be generated for a condition that fails familiar A/B gates.

## T8 — First-probe isolation

Primary human metrics must use the first probe exposure before any recurrent carry or probe updating can contaminate interpretation.

---

# 16. Required files

```text
pretraining_ecology_config.json
pretraining_world_manifest.csv
pretraining_training_curves.csv
pretraining_validation_v1.csv
pretraining_validation_v2.csv
pretraining_validation_v3.csv
pretraining_validation_v4.csv
pretraining_condition_scores.csv
pretraining_gate_results.csv
human_transfer_training.csv
human_transfer_familiar_gates.csv
human_transfer_first_probe.csv
human_transfer_profiles.csv
human_transfer_hidden_geometry.csv
PRETRAINING_ECOLOGY_COMPARISON_REPORT.md
```

---

# 17. Required figures

## Figure 1 — Pretraining ecology schematic

One panel each for:

```text
P0
P1
P2
P3
P4
P5
```

Show what statistical relationship each condition contains.

## Figure 2 — Unseen-family intermediate transfer

V1 prediction and RT deficit by pretraining condition.

## Figure 3 — Similarity-gradient test

Probe performance across:

```text
z=-.75 ... +.75
```

for each condition.

## Figure 4 — Boundary selectivity

Within-family versus across-boundary transfer.

## Figure 5 — Held-out compositional generalization

Modified-position accuracy by condition.

## Figure 6 — Pretraining selection frontier

x-axis:

```text
smooth transfer
```

y-axis:

```text
boundary selectivity
```

point size:

```text
compositional score
```

## Figure 7 — Human A/B familiar phenotype

Only selected conditions.

## Figure 8 — First Mixed probe

A / B / Mixed transition profiles by condition.

## Figure 9 — Novel probe

A / B / shared/default / Novel profiles.

## Figure 10 — Differentiation-transfer frontier

x-axis:

```text
A/B hidden separation
```

y-axis:

```text
Mixed first-probe transfer quality
```

---

# 18. Decision logic

## Outcome A — None of P2–P5 beat P1 internally

Interpretation:

> Structured prior experience does not become a useful inductive bias in the current fixed-weight 8-unit SRN.

Stop modifying ecology.

Next step should be architecture / learning-rule work.

---

## Outcome B — P2 passes but P3 fails

Interpretation:

> The SRN can learn smooth similarity-based generalization but cannot learn when to suppress it at contextual boundaries.

This predicts a risk of overgeneralization in the human task.

---

## Outcome C — P3 passes

This is the strongest single-method result.

Interpretation:

> Prior experience can teach the SRN both context-sensitive generalization and differentiation.

Advance P3 to human transfer.

---

## Outcome D — P4 passes

Interpretation:

> Explicit experience with reusable contextual components induces zero-shot compositional reuse in new families.

Advance P4 if it also preserves boundary/familiar learning.

---

## Outcome E — P5 beats simpler conditions

Interpretation:

> A richer distribution of prior contextual experiences creates a broader structural prior than any single ecological law.

---

## Outcome F — Structured pretraining passes internally but does not improve human probes

Interpretation:

> The SRN can learn structural priors, but those priors do not transfer into the human A/B representational regime.

This would motivate an explicit contextual-inference / retrieval mechanism.

---

## Outcome G — Structured pretraining improves first Mixed probe while A/B splitting survives

This is the target result.

Interpretation:

\[
\boxed{
\text{human-like ambiguous retrieval can emerge from prior structured experience}
}
\]

without:

```text
Mixed training
Novel training
RT supervision
midpoint objectives
output averaging
```

---

# 19. Recommended execution order

Do not launch all conditions at full scale immediately.

### Phase 1 — engineering smoke test

```text
1 seed
20 worlds
P1–P5
```

Verify generators and frozen validation.

### Phase 2 — preregistered pilot

```text
5 seeds
500 worlds
P1–P5
```

Apply hard gates.

### Phase 3 — confirmatory pretraining

Only passing conditions:

```text
20 seeds
2000 worlds
```

### Phase 4 — human-task transfer

Advance:

```text
P0
P1
top 2 passing structured methods
```

### Phase 5 — mechanism

Only if structured pretraining changes first-probe behavior.

---

# 20. Central scientific test

The experiment is designed to answer:

\[
\boxed{
\text{What kind of prior experience, if any, makes the same SRN that learns separate A/B skills infer sensible behavior in a new ambiguous context?}
}
\]

The candidate explanations are intentionally separated:

```text
P2: similarity-based structural learning
P3: contextual inference with boundaries
P4: compositional reuse
P5: richer lifetime-like ecology
```

The human task is used only after these priors have been independently demonstrated in unseen synthetic task families.
