# PRD: Hierarchical Compositional Pretraining Ecology for Emergent Contextual Recombination
## Testing whether a structural prior learned before the human task enables zero-shot Mixed and Novel transfer

## 0. Purpose

The current SRN program has established a consistent dissociation:

1. High-scale SRNs can learn strongly context-specific A/B sequence representations.
2. The same networks show context-yoked RT dynamics.
3. Human-like spatial geometry improves Mixed/Novel transfer somewhat.
4. Increasing hidden capacity increases A/B specialization but worsens Mixed/Novel transfer.
5. Adding explicit context dimensions does not solve transfer.
6. Explicit responsibility-weighted output averaging is sufficient to place Mixed/Novel between A/B endpoints.
7. Multiple emergent retrieval/context-inference architectures fail to produce robust human-like interpolation while preserving the A/B split.

The remaining hypothesis is that humans enter the experiment with a **pre-existing structural prior**:

> Contexts that share perceptual or structural features often share task structure, and new contexts should reuse relevant prior knowledge rather than be treated as arbitrary new states.

Because human intermediate probe behavior appears very early, this prior plausibly predates the experimental A/B learning phase.

This PRD tests whether the same SRN can acquire such a prior through **structured pretraining across many unrelated task worlds**, and then spontaneously apply it to the human A/B/Mixed/Novel experiment.

The key requirement is:

\[
\boxed{
\text{Mixed and Novel remain completely unseen during both pretraining and human-task training.}
}
\]

Any intermediate expression must emerge from prior structural experience.

---

# 1. Scientific question

> Can prior experience with structured families of related tasks teach an SRN a general inductive bias for contextual reuse, such that it later learns distinct A/B skills while spontaneously recombining them under Mixed cues and falling back toward shared structure under Novel cues?

Target pattern:

\[
\boxed{
\text{A/B differentiation}
+
\text{Mixed recombination}
+
\text{Novel shared-structure fallback}
}
\]

without explicit interpolation targets.

---

# 2. Core hypotheses

## H1 — Structured pretraining improves zero-shot Mixed transfer

Networks pretrained in an ecology where contextual similarity predicts reusable task structure should show more human-like zero-shot Mixed behavior than:

```text
random-init controls
and
structure-shuffled pretraining controls
```

after identical human-task A/B training.

## H2 — Structured pretraining preserves A/B splitting

Improved Mixed transfer must not arise because the network stops differentiating A and B.

Desired regime:

\[
h_A \neq h_B
\]

while:

\[
RT_M
\]

falls more often between the two familiar endpoints.

## H3 — Novel and Mixed rely on different aspects of the prior

Mixed should benefit from **compositional reuse of familiar components**.

Novel should benefit from **hierarchical fallback to shared family/task structure**.

Therefore Mixed and Novel must always be analyzed separately.

## H4 — Generic pretraining is insufficient

If pretraining volume alone helps, both structured and shuffled ecologies should improve.

If the structural prior matters, then:

\[
\boxed{
P2_{\text{structured}}
>
P1_{\text{shuffled}}
}
\]

on zero-shot transfer.

---

# 3. Fixed human-task target

After pretraining, every condition receives the exact same human-grounded task.

Use the validated:

```text
G2_shared_x environment
```

Freeze:

```text
same motor sequence
same P1..P8 ordinal channels
same human-derived shared-x geometry
same context-specific y geometry
same A10/B10 curriculum
170 external A/B training sequences
same optimizer
same learning rate
same monotonic execution readout
same RT definition
same frozen-probe policy
same analysis code
```

Primary architecture:

```text
8-unit tanh Elman SRN
High initialization scale = 2.0
```

Primary experiment should focus on High scale.

Low-scale runs may be retained as a secondary control if computationally cheap, but do not let them expand the scope of the initial implementation.

---

# 4. Pretraining conditions

Run three primary groups.

| ID | Pretraining condition | Purpose |
|---|---|---|
| P0 | None / random initialization | Current reference |
| P1 | Structure-shuffled ecology | Controls for generic pretraining exposure |
| P2 | Hierarchical compositional ecology | Tests learned structural prior |

All P1/P2 networks receive:

```text
same number of pretraining worlds
same number of episodes
same number of optimizer steps
same input dimensionality
same task difficulty distribution
same cue magnitude distribution
same layout distribution
same sequence-length distribution
```

The only difference is whether context structure predicts reusable task structure.

---

# 5. Fundamental emergence constraints

## E0.1 — No human-task leakage during pretraining

The pretraining generator must not contain:

```text
the actual human A layout
the actual human B layout
the actual human Mixed layout
the actual human Novel layout
the actual human cue vectors
the actual human task labels
```

## E0.2 — No Mixed/Novel during human-task training

```text
Mixed optimizer steps = 0
Novel optimizer steps = 0
```

## E0.3 — No interpolation target anywhere

Forbidden losses or labels:

```text
"Mixed should be midpoint"
"Novel should be neutral"
q = 0.5
lambda = 0.5
distance-to-corridor loss
between-endpoint loss
```

## E0.4 — No probe-based hyperparameter selection

Do not select:

```text
number of worlds
latent-factor count
shared-task fraction
learning rate
pretraining duration
checkpoint
```

using human Mixed/Novel results.

## E0.5 — Same architecture across P0/P1/P2

The experiment manipulates **prior experience**, not representational capacity.

---

# 6. Pretraining ecology: conceptual structure

Each pretraining world \(w\) is an unrelated miniature task family.

A world contains:

```text
a base sequence/task backbone
a set of latent contextual factors
a mapping from latent factors -> observable context features
a mapping from latent factors -> task-specific modifications
multiple familiar contexts
held-out context combinations
optional novel family members
```

The stable statistical regularity across worlds is:

\[
\boxed{
\text{shared contextual causes tend to imply shared task structure.}
}
\]

The exact cues, sequences, factor identities, and task mappings change from world to world.

---

# 7. Generative model for one pretraining world

## 7.1 Base task

Sample a new motor sequence:

```text
sequence length = 8
alphabet size = 4
```

Constraints:

```text
not identical to the human target sequence
avoid trivial all-repeat sequences
use at least 3 of 4 response tokens
```

Let the base task be:

\[
\theta_{\text{base}}.
\]

## 7.2 Latent contextual factors

Sample:

```text
J ∈ {2, 3, 4}
```

latent binary factors:

\[
c=[c_1,\dots,c_J],
\qquad c_j\in\{0,1\}.
\]

These factors are not directly provided to the SRN.

## 7.3 Shared task backbone

Every context in the same world shares the base sequence/task backbone.

Context-specific task variation is generated by sparse modifiers:

\[
\theta_c
=
\theta_{\text{base}}
+
\sum_{j=1}^{J}c_j\Delta_j.
\]

Primary implementation:

```text
each factor modifies 1-2 ordinal positions
```

Possible modification types:

```text
local response remapping
local transition-difficulty offset
local execution-cost modifier
```

Do not modify all positions simultaneously.

The point is to create reusable components rather than wholly separate tasks.

---

# 8. Observable context generation

Each latent factor also contributes to the visible context.

Define:

\[
z_c
=
z_{\text{shared}}
+
\sum_j c_j v_j
+
\epsilon.
\]

The observable vector should use the same general input channels available to the human-task SRN:

```text
cue features
layout / coordinate features
```

but with synthetic world-specific values.

Across worlds:

```text
v_j changes
z_shared changes
task modifiers Δ_j change
```

Thus no stable surface cue has a fixed meaning across worlds.

What remains stable is the **relationship**:

```text
shared latent factors -> shared observable structure + shared task structure
```

---

# 9. Ecology components

The P2 structured ecology must contain three kinds of task families.

## 9.1 Shared-structure worlds

Contexts differ perceptually but share most of the task.

Purpose:

```text
different context does not always imply a new skill
```

Generate worlds where approximately:

```text
70-90% of task structure is shared
10-30% is context-specific
```

## 9.2 Context-specific worlds

Some worlds must contain genuinely strong context-specific differences.

Purpose:

```text
the learner cannot solve pretraining by ignoring context
```

Generate worlds where approximately:

```text
40-60% of task-relevant positions differ across contexts
```

These worlds preserve pressure for context-sensitive specialization.

## 9.3 Compositional worlds

Contexts consist of reusable combinations of factors.

Example:

```text
C1 = [1,0,0]
C2 = [0,1,0]
C3 = [1,1,0]
C4 = [0,1,1]
```

Different factors alter different sparse task components.

Hold out some combinations entirely from training.

These held-out combinations are used only as internal pretraining diagnostics.

---

# 10. Hierarchical family structure for Novel-like generalization

Each world also contains a hierarchy:

\[
\theta_c
=
\theta_{\text{family}}
+
\Delta_c.
\]

All contexts in a world share:

```text
sequence identity
ordinal task backbone
family-level action structure
```

while context-specific modifiers vary.

The network should repeatedly encounter situations where a new context belongs to a familiar family but has an unfamiliar specific configuration.

This teaches the useful prior:

\[
\boxed{
\text{new context}
\rightarrow
\text{reuse family structure first}
}
\]

rather than:

\[
\text{new context}
\rightarrow
\text{arbitrary unrelated policy}.
\]

---

# 11. Pretraining held-out tests

Held-out pretraining probes are diagnostic only.

They never update the network.

## 11.1 Held-out factor combination

Train within a world on:

```text
[1,0]
[0,1]
[0,0]
```

Test:

```text
[1,1]
```

The correct task follows from the sampled factor-generating process.

It is **not defined as an arithmetic midpoint** of observed contexts.

## 11.2 New family member

Train on several contexts from one family.

Test a novel context:

```text
new visible configuration
same family backbone
new sparse modifier combination
```

Question:

```text
Does the network preserve shared task structure?
```

## 11.3 Surface-similarity trap

Include occasional cases where two contexts are perceptually somewhat similar but task modifiers differ.

Purpose:

```text
prevent blind nearest-neighbor matching
```

The network should learn statistical reuse, not simple Euclidean copying.

---

# 12. P1 structure-shuffled control

P1 receives the exact same generated worlds as P2 before shuffling.

For each world:

1. Generate the P2 world normally.
2. Preserve all:
   ```text
   sequences
   visible contexts
   factor counts
   task variants
   number of episodes
   difficulty
   ```
3. Randomly permute the mapping between:
   ```text
   observable context configurations
   and
   task modifier configurations
   ```
4. Preserve marginal distributions.

This destroys:

\[
P(\text{task similarity}\mid\text{context similarity})
\]

while keeping generic pretraining exposure matched.

Use one fixed shuffle per:

```text
world × seed
```

and log it.

---

# 13. P0 random-init control

P0 receives no pretraining.

Initialize using the current validated human-task initialization rule:

```text
High scale = 2.0
```

Then train directly on the human A/B experiment.

This reproduces the current baseline.

---

# 14. Pretraining architecture

Use the exact same 8-unit SRN architecture intended for the human task.

Primary recurrence:

\[
h_t
=
\tanh
(
W_xx_t
+
W_hh_{t-1}
+
b
).
\]

Retain the same:

```text
next-token prediction head
monotonic execution readout
```

No retrieval layer.

No attention.

No latent-context module.

No increased hidden size.

The purpose is to test whether experience changes the inductive bias of the existing learner.

---

# 15. Pretraining objective

Use only ordinary task performance.

Primary:

\[
L_{\text{seq}}
=
-\log p(y_{t+1}^{true}).
\]

Retain the current execution objective/readout if the ecology generates sufficient confidence variation.

Total:

\[
L
=
L_{\text{seq}}
+
\lambda_{\text{exec}}L_{\text{exec}}.
\]

Use the same fixed execution weight as the validated current model.

Forbidden:

```text
context classification loss
factor reconstruction loss
similarity loss
contrastive context loss
held-out interpolation loss
```

The prior should arise because structural reuse helps task performance across worlds.

---

# 16. Meta-training regime

Primary starting values:

```text
pretraining worlds = 1000
training contexts per world = 4
held-out probes per world = 2
sequences per observed context = 10
```

Do not tune these on human Mixed/Novel results.

If compute is excessive, run a preregistered pilot:

```text
100 worlds
5 seeds
```

for implementation validation only.

Then freeze the pretraining schedule before the confirmatory experiment.

---

# 17. World reset policy

At each new world:

```text
reset recurrent hidden state
do NOT reset network weights
```

The SRN weights carry learned inductive biases across worlds.

Within a world:

```text
normal recurrent dynamics
```

No world identifier is supplied.

---

# 18. Transition from pretraining to human experiment

After the last pretraining world:

```text
freeze and save checkpoint
```

Then begin the human-task learning phase.

Primary policy:

```text
do not reset weights
reset transient hidden state
reset optimizer moments
```

Use a fresh optimizer state for human-task training.

This prevents optimizer momentum from masquerading as a learned prior.

Secondary robustness:

```text
carry optimizer state
```

only after primary.

---

# 19. Human A/B training

For P0/P1/P2:

```text
same G2_shared_x environment
same 170 external sequences
same A10/B10 schedule
same learning rate
same optimizer
same number of updates
```

No Mixed/Novel exposure.

The primary question is whether pretraining changes how the same model generalizes after familiar A/B learning.

---

# 20. Human-task hard gates

Before interpreting Mixed/Novel, each condition must preserve the familiar phenotype.

## G1 — Sequence learning

At least 80% of seeds:

```text
late A/B NLL < early A/B NLL
```

## G2 — A/B differentiation

P2 must retain substantial matched-position A/B hidden distance.

Primary benchmark:

```text
P2 mean differentiation >= 80% of P0 mean differentiation
```

## G3 — Context-yoked RT structure

Require the familiar A/B sawtooth/switch signature.

## G4 — Canonical state-space phenotype

Majority of P2 seeds must remain:

```text
Cross or Split
```

## G5 — RT grounding

Require:

```text
confidence -> RT slope < 0
```

for all interpretable seeds.

A transfer improvement that collapses A/B differentiation fails.

---

# 21. Frozen human probes

Evaluate:

```text
A
B
Mixed
Novel
```

with:

```text
weights frozen
optimizer frozen
0 parameter updates
```

Report:

```text
prediction NLL
correct-token confidence
normalized RT
next-token accuracy
hidden states
```

Mixed and Novel separately.

---

# 22. Primary Mixed outcome

Canonicalize familiar endpoints within seed:

```text
Fast = lower mean A/B frozen RT
Slow = higher mean A/B frozen RT
```

Swap entire profiles only.

Define:

\[
I_{M,s,k}
=
\frac
{RT_{M,s,k}-RT_{Fast,s,k}}
{RT_{Slow,s,k}-RT_{Fast,s,k}}.
\]

Primary Mixed metric:

```text
fraction of transitions where 0 < I_M < 1
```

Secondary:

```text
Mixed NLL deficit over worse endpoint
Mixed RT deficit over worse endpoint
profile-fit q
profile R²
hidden off-corridor distance
```

No Mixed metric influences training or checkpoint selection.

---

# 23. Primary Novel outcome

Do not require Novel to be a strict A/B midpoint.

Primary Novel metrics:

```text
Novel NLL deficit over worse A/B endpoint
Novel RT deficit over worse A/B endpoint
distance to A/B hidden corridor
distance to context-neutral/shared-task control
next-token accuracy
```

Secondary:

```text
between-endpoint fraction
```

Interpret Novel as potentially expressing:

```text
shared family/task structure
```

rather than literal interpolation.

---

# 24. Early-emergence criterion

Because humans show early intermediate expression, measure the **first frozen Mixed probe** separately from later probes.

Primary question:

```text
Does P2 show improved Mixed interpolation immediately at first exposure?
```

Do not average this away across probe epochs.

Required outputs:

```text
first_probe_between_endpoint_fraction
first_probe_NLL_deficit
first_probe_RT_deficit
```

This is central to the learned-prior hypothesis.

---

# 25. Pretraining diagnostic outcomes

Before human-task transfer, confirm that P2 actually learned the intended prior.

On held-out pretraining probes, compare P1 vs P2 on:

```text
held-out factor-combination NLL
held-out factor-combination accuracy
new-family-member NLL
new-family-member accuracy
```

Expected:

\[
P2 > P1
\]

on structured held-out generalization.

Do not require midpoint geometry.

The target is correct reuse of latent task structure.

---

# 26. Differentiation–transfer analysis

For each seed, plot:

### x-axis

```text
A/B matched-position hidden distance
```

### y-axis

```text
Mixed NLL deficit
```

and separately:

```text
Mixed between-endpoint fraction
Novel NLL deficit
```

The desired P2 shift is:

```text
same/high differentiation
+
lower transfer deficit
+
higher Mixed interpolation
```

not movement toward a Lump solution.

---

# 27. Primary confirmatory contrast

The strongest causal comparison is:

\[
\boxed{
P2_{\text{structured}}
-
P1_{\text{shuffled}}
}
\]

because both conditions receive matched pretraining volume.

Primary confirmatory outcomes:

```text
Mixed first-probe between-endpoint fraction
Mixed NLL deficit
Novel NLL deficit
A/B differentiation
```

P0 remains the no-pretraining reference.

---

# 28. Statistical analysis

Use paired pretraining seeds/world generators wherever possible.

Preferred model:

```text
metric ~ pretraining_condition + (1|seed)
```

for:

```text
P0
P1
P2
```

Primary planned contrasts:

```text
P2 - P1
P2 - P0
P1 - P0
```

Report:

```text
mean difference
bootstrap 95% CI
standardized effect
```

For first-vs-late probe:

```text
metric ~ pretraining_condition * probe_epoch + (1|seed)
```

Do not rely on p-values alone.

---

# 29. Screening and confirmatory workflow

## Stage 0 — Generator validation

Run synthetic checks on:

```text
world generation
factor-task mapping
held-out combination construction
shuffle validity
```

No human task.

## Stage 1 — Small pretraining pilot

Run:

```text
P1
P2
5 seeds
100 worlds
```

Purpose:

```text
verify learning stability
verify P2 > P1 on internal held-out structured generalization
```

Do not inspect human Mixed/Novel for model selection.

## Stage 2 — Freeze ecology schedule

Set:

```text
number of worlds
contexts/world
episodes/context
learning rate
```

before confirmatory human-task testing.

## Stage 3 — Confirmatory pretraining

Run:

```text
20 seeds
P0
P1
P2
```

with paired generators for P1/P2.

## Stage 4 — Human A/B training

Train all conditions identically.

## Stage 5 — Frozen probes

Evaluate A/B/Mixed/Novel.

## Stage 6 — Mechanistic analyses

Only if P2 improves transfer.

---

# 30. Mechanistic follow-ups if P2 succeeds

Do not implement until the primary result is established.

## M1 — Weight-change localization

Compare P1 vs P2 pretraining-induced changes in:

```text
input weights
recurrent weights
output weights
execution readout
```

Question:

```text
where is the prior stored?
```

## M2 — Representation geometry

Test whether P2 produces:

```text
more aligned A/B ordinal subspaces
lower Mixed off-corridor distance
preserved A/B centroid separation
```

## M3 — Causal ablation

Interpolate P2 weights back toward initialization by subsystem.

Determine which pretrained weights are necessary for improved transfer.

## M4 — Ecology decomposition

Only after P2 success, compare:

```text
compositional-only
hierarchical-only
shared-structure-only
full P2
```

Do not start with this larger factorial experiment.

---

# 31. Strong falsification criteria

## F1 — P2 and P1 perform similarly

Interpretation:

> Generic pretraining, not learned structure, explains any benefit.

## F2 — P2 improves Mixed only by reducing A/B differentiation

Interpretation:

> Pretraining weakens specialization rather than teaching flexible reuse.

Reject as target explanation.

## F3 — P2 improves late Mixed but not first Mixed probe

Interpretation:

> Benefit may reflect faster within-task learning rather than a prior available at probe onset.

## F4 — P2 fails internal pretraining held-out generalization

Interpretation:

> Ecology did not successfully teach the intended prior.

Do not interpret human-task transfer.

## F5 — P2 improves only RT, not prediction NLL

Interpretation:

> Downstream execution readout changed without improving task transfer.

Reject as complete explanation.

## F6 — P2 requires increasing hidden size

Do not pursue within this PRD.

The primary hypothesis is about prior experience in the existing architecture.

---

# 32. TDD requirements

Use RED → GREEN → REFACTOR.

## T1 — Human-task leakage firewall

Assert pretraining worlds contain no exact human:

```text
cue vectors
layout arrays
motor sequence
probe labels
```

## T2 — Structured mapping integrity

For P2:

```text
shared latent factor
=> shared observable component
=> corresponding reusable task modifier
```

must match generator specification.

## T3 — Shuffled-control integrity

For P1:

```text
same world marginals as P2
different cue↔task pairing
```

## T4 — Equal pretraining exposure

Assert P1 and P2 have identical:

```text
optimizer-step counts
world counts
sequence counts
context counts
```

## T5 — Same architecture

Parameter shapes must match across P0/P1/P2.

## T6 — No Mixed/Novel training

Assert:

```text
train_mixed_count == 0
train_novel_count == 0
```

## T7 — Frozen probe

Hash parameters before and after probes.

Hashes must match exactly.

## T8 — Held-out combinations are truly held out

No held-out factor combination may appear in a world's training set.

## T9 — Shuffle does not change task difficulty distribution

Compare P1/P2 marginal:

```text
sequence entropy
modifier count
task-change magnitude
```

within tolerance.

## T10 — Optimizer reset

Primary human-task runs must start with fresh optimizer state.

## T11 — Fixed human-task schedule

P0/P1/P2 must receive byte-identical A/B training sequences per paired seed.

## T12 — Post-hoc interpolation only

No interpolation metric may be imported by any training module.

---

# 33. Required configuration files

```text
pretraining_ecology_config.json
pretraining_world_generator_config.json
human_transfer_config.json
```

Log:

```text
seed
world_id
factor_count
context vectors
task modifiers
held-out combinations
shuffle mapping
optimizer steps
```

---

# 34. Required outputs

## Pretraining

```text
pretraining_training_summary.csv
pretraining_heldout_generalization.csv
pretraining_world_manifest.csv
pretraining_shuffle_manifest.csv
```

## Human A/B phase

```text
human_transfer_training_summary.csv
human_transfer_state_space.csv
human_transfer_hidden_differentiation.csv
human_transfer_rt_grounding.csv
```

## Frozen probes

```text
human_transfer_probe_metrics.csv
human_transfer_interpolation.csv
human_transfer_hidden_corridor.csv
human_transfer_first_probe.csv
```

## Report

```text
HIERARCHICAL_COMPOSITIONAL_PRETRAINING_REPORT.md
```

---

# 35. Required figures

## Figure 1 — Ecology schematic

Show one example world:

```text
latent factors
-> visible context structure
-> sparse reusable task modifiers
```

and distinguish:

```text
P2 structured
vs
P1 shuffled
```

## Figure 2 — Internal pretraining generalization

P1 vs P2:

```text
held-out combination NLL
new-family-member NLL
```

## Figure 3 — Human A/B phenotype

P0/P1/P2:

```text
state-space winner fraction
A/B hidden differentiation
context-yoked RT
```

## Figure 4 — First Mixed probe

P0/P1/P2:

```text
Fast
Slow
Mixed
```

transition profiles at first exposure.

## Figure 5 — Mixed interpolation

Plot:

```text
between-endpoint fraction
NLL deficit
RT deficit
```

for P0/P1/P2.

## Figure 6 — Novel transfer

Plot:

```text
NLL deficit
RT deficit
neutral/shared-state distance
```

## Figure 7 — Differentiation–transfer frontier

x:

```text
A/B hidden separation
```

y:

```text
Mixed between-endpoint fraction
```

and separate panel:

```text
Mixed NLL deficit
```

## Figure 8 — Hidden representational corridor

P0/P1/P2 Mixed and Novel:

```text
lambda*
off-corridor distance
inside-segment fraction
```

## Figure 9 — Pretraining effect summary

One compact panel:

```text
No pretraining
Shuffled pretraining
Structured pretraining
```

with the three key outcomes:

```text
A/B differentiation
first-probe Mixed interpolation
Novel transfer
```

---

# 36. Interpretation matrix

## Outcome A — P2 > P1 and P0, with A/B split preserved

Interpretation:

> Structured prior experience teaches the SRN a reusable contextual generalization bias that is applied immediately to a new human-like task.

This is the target result.

## Outcome B — P2 > P0 but P2 ≈ P1

Interpretation:

> Generic task pretraining improves transfer, but the specific hierarchical/compositional structure is not responsible.

## Outcome C — P2 improves Mixed but not Novel

Interpretation:

> Compositional reuse of familiar components is learned, but novel-context fallback requires an additional hierarchical prior.

## Outcome D — P2 improves Novel but not Mixed

Interpretation:

> Shared-family fallback is learned, but simultaneous familiar-cue recombination remains unsolved.

## Outcome E — P2 improves transfer but reduces A/B splitting

Interpretation:

> Pretraining changes the specialization/generalization tradeoff rather than solving both simultaneously.

## Outcome F — P2 improves first-probe Mixed immediately

Interpretation:

> The generalization rule is available as a prior before any experience with the ambiguous probe.

This is especially consistent with the human early-emergence result.

## Outcome G — No improvement

Interpretation:

> The proposed ecological prior is insufficient. Do not escalate immediately to larger networks or richer retrieval heads.

---

# 37. Final scientific criterion

The strongest outcome is:

\[
\boxed{
\text{structured prior experience}
\rightarrow
\text{A/B specialization}
+
\text{immediate zero-shot Mixed recombination}
+
\text{Novel shared-structure transfer}
}
\]

using the same SRN architecture and without any explicit interpolation rule.

The paper-level interpretation would be:

> Representational flexibility is sufficient for context-specific skill formation, but flexible recombination depends on prior experience with the statistical structure of how contexts relate to tasks.

That would provide a principled explanation for why humans generalize immediately while randomly initialized SRNs do not.
