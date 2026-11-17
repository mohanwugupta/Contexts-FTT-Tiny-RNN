# PRD: Human-Task Transfer Test for Context-Controlled Recurrent Models
## Translate Synthetic A/B Latent-Cause Learning into the Human A/B → Mixed/Novel Experiment

## 0. Objective

Test whether the two architectures selected entirely from the synthetic confirmatory stage transfer to the actual human motor-sequence task.

Locked synthetic winners:

```text
Model A = GRU-150
Model B = Elman-128 + context-GRU-8
```

Both architectures passed the synthetic confirmatory split + ambiguity gate and were selected **without using human Mixed/Novel outcomes**.

The human stage is therefore an out-of-distribution test of the hypothesis:

**structured prior experience with latent contextual structure -> better context inference in the human task**

while preserving:

**strong A/B context-specific learning**

The main mechanistic question is:

> Does structured prior experience change how the model infers and expresses context in the human task, rather than simply changing the stored motor sequence?

No further architecture or synthetic-ecology tuning is allowed in this PRD.

---

## 1. Locked scientific specification

Use the exact confirmatory checkpoints and source/config hashes from the independent synthetic confirmatory run.

Primary structured models:

```text
A-S  = GRU-150, structured synthetic prior
B-S  = Elman-128 + context-GRU-8, structured synthetic prior
```

Matched controls:

```text
A-Sh = GRU-150, shuffled synthetic prior
B-Sh = Elman-128 + context-GRU-8, shuffled synthetic prior
```

Additional prespecified controls:

```text
B-S-RP = Model B structured prior with restricted target-task plasticity
H8     = original validated H8 SRN baseline
```

Optional only if already implemented before human results are inspected:

```text
A-Rand = GRU-150 random initialization
B-Rand = Model B random initialization
```

Do not add them later based on results.

---

## 2. Primary causal contrasts

Main within-architecture tests:

### C1 — structured prior effect in GRU

```text
A-S - A-Sh
```

### C2 — structured prior effect in explicit context-state model

```text
B-S - B-Sh
```

These are the strongest causal contrasts because architecture, capacity, training amount, optimizer, synthetic exposure, and update count are matched.

Secondary contrasts:

```text
A-S vs B-S
B-S vs B-S-RP
structured models vs H8
```

Do not use A-vs-B to choose a "winner" based on human outcomes. Both were independently selected before the human stage.

---

## 3. Human target task

Use the exact human target environment already established in the project.

Locked:

```text
same 8-item sequence
same human-derived A/B layouts
same cue inputs
same trial/block structure
same A10/B10 familiar training schedule
same total familiar training exposure
same human probe contexts
same temporal ordering as the human experiment
```

Do not alter geometry or cue salience to help the new architectures.

---

## 4. Match the human experimental timeline exactly

The paper-level comparison should use model checkpoints aligned to actual human measurement points.

Create:

```text
human_model_timeline.csv
```

Columns:

```text
human_phase
human_block
human_sequence_index
probe_name
model_training_sequence_index
weights_frozen_before_probe
```

Include every relevant phase:

```text
familiar A
familiar B
Mixed1
Mixed2
return-to-A/B
final MixedTest
Novel
NoContext/default, if present in the human experiment
```

Use the actual human schedule from the existing human analysis code/data.

Do not substitute generic 20/50/90/170 checkpoints for primary comparisons if the true human probe occurs elsewhere.

Generic checkpoints may remain secondary diagnostics.

---

## 5. Input translation into the human task

### Model A — GRU-150

At transition t, input:

```text
[current sequence input, visible cue/layout information]
```

No explicit latent context label.

The GRU hidden state carries task/sequence state and context inference implicitly.

### Model B — Elman core + context-GRU

Separate context inference from task execution.

Controller input:

```text
u_t = [visible cue/layout evidence, allowed recent observable history]
```

Controller:

```text
c_t = GRU_ctx(u_t, c_{t-1})
```

with:

```text
dim(c_t) = 8
```

Task core:

```text
h_t = tanh(W_x x_t + W_h h_{t-1} + U_c c_t + b)
```

The controller must never receive:

```text
future response
current target token before prediction
human RT
latent A/B label
Mixed/Novel label
```

Only information available at prediction time is allowed.

---

## 6. Human A/B learning phase

After loading the frozen synthetic checkpoint:

```text
reset optimizer state
reset recurrent hidden state
reset target-task execution/RT readout
```

Then train on familiar human A/B only.

Training stream:

```text
A10 -> B10 -> A10 -> B10 -> ...
```

using the exact human schedule.

No Mixed or Novel sequence may update weights.

---

## 7. Primary target-task plasticity condition

For A-S, A-Sh, B-S, and B-Sh:

```text
all recurrent/context/task prediction parameters remain trainable
```

This tests whether the learned prior survives and shapes ordinary learning of a new A/B task.

Use the same target-task optimizer family and learning-rate policy within matched architecture pairs.

Do not tune target learning rates using Mixed/Novel behavior.

Any stability calibration may use only:

```text
A/B training loss
numerical stability
```

and must be completed before human probe files are generated.

---

## 8. Restricted-plasticity control for Model B

Run:

```text
B-S-RP
```

Freeze:

```text
shared Elman recurrent core
```

Allow learning in:

```text
context-GRU
context-to-core weights
prediction head
execution/RT head
```

Purpose:

> Test whether preserving reusable task dynamics while adapting only context-control pathways protects the learned prior.

This is a prespecified mechanistic control, not a rescue.

If B-S-RP succeeds while B-S fails, interpret that as evidence of target-training overwriting.

---

## 9. RT/execution mechanism

Synthetic training did not use RT.

During human A/B training, attach the validated monotonic execution readout already used in the project:

```text
v = sigmoid(softplus(a) * logit(stopgrad(p_correct)) + b)
RT_norm = 1 - v
```

No direct:

```text
context -> RT
controller state -> RT
Mixed label -> RT
```

pathway is allowed.

Mechanistic chain:

```text
context evidence
-> context/task state
-> p_correct
-> vigor
-> RT
```

---

## 10. Human probes are frozen counterfactual tests

At every human probe point:

1. clone the model;
2. freeze all trainable parameters;
3. reset hidden state according to the locked probe convention;
4. present the probe context;
5. record first-exposure outputs;
6. discard the clone.

The main training trajectory continues from the pre-probe model.

Thus model probes do not teach the model how to answer later probes.

---

## 11. Familiar A/B gate

No Mixed or Novel result may be interpreted unless the familiar model passes.

Require:

### F1 — sequence learning
A/B training NLL decreases and settled prediction is above chance.

### F2 — context differentiation
A/B hidden-state distance exceeds the prespecified permutation/null baseline.

### F3 — context-yoked behavior
A/B RT profiles differ in the expected context-sensitive way.

### F4 — state-space phenotype
Use the existing Lump/Cross/Split pipeline.

Primary gate:

```text
majority Cross or Split
```

for each primary model group.

### F5 — confidence/RT relation

```text
corr(p_correct, RT) < 0
```

### F6 — controller separation for Model B

```text
D(c_A, c_B) > null
```

This is additional, not a substitute for behavioral/hidden differentiation.

A model that improves Mixed by remaining Lump fails.

---

## 12. First-exposure Mixed test

Primary Mixed outcome uses the **first frozen Mixed sequence at the exact human probe point**.

Do not average later Mixed exposures for the primary test.

For each transition k, record:

```text
RT_A(k)
RT_B(k)
RT_Mixed(k)
NLL
correct-token confidence
prediction probability
hidden state
```

For Model B additionally record:

```text
context state c_t
```

---

## 13. Canonical familiar endpoints

Within each seed:

```text
Fast = familiar context with lower mean RT
Slow = familiar context with higher mean RT
```

If A/B labels are swapped:

```text
swap full profiles
```

Never relabel individual transitions.

Use the same canonicalization for all endpoint-derived metrics.

---

## 14. Primary Mixed behavioral metrics

### M1 — between-endpoint fraction

For transition k:

```text
I_k = 1 if RT_Fast,k <= RT_Mixed,k <= RT_Slow,k else 0
```

Then:

```text
B_Mixed = mean_k(I_k)
```

### M2 — RT deficit
Distance outside the familiar endpoint interval.

### M3 — NLL deficit

Report:

```text
Mixed NLL - mean familiar NLL
Mixed NLL - worse familiar endpoint NLL
```

### M4 — human profile correspondence

Without fitting to human Mixed:

```text
correlation
RMSE
profile R²
```

between model and human transition-specific Mixed profiles.

Primary causal comparisons:

```text
A-S vs A-Sh
B-S vs B-Sh
```

---

## 15. Mixed representational metrics

For every transition, collect A, B, and Mixed hidden states.

Compute:

### R1 — familiar-axis projection

```text
q_h = ((h_M - h_B)' (h_A - h_B)) / ||h_A - h_B||²
```

### R2 — off-corridor distance
Distance of Mixed from the A/B line/subspace.

### R3 — inside familiar segment
Whether the projected Mixed state lies between familiar endpoints.

### R4 — shared sequence subspace engagement
Measure how much Mixed occupies the sequence-shared subspace learned from A/B.

Do not train on any of these measures.

---

## 16. Model B context-state analysis

At matched transition k, obtain:

```text
c_A(k)
c_B(k)
c_Mixed(k)
```

Compute:

### Ctx1 — controller separation

```text
D(c_A, c_B)
```

### Ctx2 — Mixed controller projection

```text
q_c = ((c_M - c_B)' (c_A - c_B)) / ||c_A - c_B||²
```

### Ctx3 — Mixed controller off-axis distance
Measure whether Mixed occupies the familiar context-state manifold or a distinct state.

### Ctx4 — uncertainty structure
Compare first Mixed controller state with clear A/B and with synthetic ambiguous-evidence controller states.

Do not assume q_c = .5.

Key question:

> Does Mixed immediately alter inferred context state without Mixed-specific learning?

---

## 17. Novel probe

Analyze Novel separately.

Primary Novel outcomes:

```text
prediction NLL deficit
RT deficit
accuracy
hidden distance to A/B manifold
shared-sequence subspace engagement
```

For Model B:

```text
distance of c_Novel from A/B controller manifold
```

Do not require Novel to lie between A and B.

Potential interpretation to test:

```text
Mixed = uncertain inference over familiar contexts
Novel = unfamiliar contextual state with reuse of shared skill structure
```

---

## 18. Probe-timing analyses

Primary analyses use the exact human probe schedule.

Secondary diagnostic probes may use:

```text
20
50
90
170
```

familiar-training sequences if useful.

They may answer:

```text
when does A/B separation emerge?
when does context-state separation emerge?
does the synthetic prior persist or get overwritten?
```

Do not use these checkpoints for model selection.

---

## 19. Prior-retention analysis

At target-training checkpoints, use disposable copies to rerun the original synthetic validation battery.

Recommended checkpoints:

```text
before human A/B training
after early A/B training
before first human Mixed probe
before final MixedTest
end of familiar training
```

Measure:

```text
synthetic split score
synthetic ambiguity NLL
context-belief calibration
```

Question:

> Does human A/B learning preserve, transform, or overwrite the confirmed latent-context prior?

---

## 20. Causal interventions for Model B

Run only after primary human behavioral analysis.

Use frozen copies.

### I1 — context-state swap

Run:

```text
A sequence with c_B
B sequence with c_A
```

while holding sequence input fixed.

Measure:

```text
prediction
confidence
RT
task-core hidden state
```

### I2 — context-state ablation

Set:

```text
c_t = 0
```

or replace with a matched random state.

If A/B behavior collapses, controller state is causally necessary.

### I3 — artificial interpolation

Construct post hoc:

```text
c_avg = 0.5*c_A + 0.5*c_B
```

Compare outputs under c_avg to actual Mixed.

Purpose:

> Determine whether emergent Mixed is equivalent to arithmetic averaging.

Never use this intervention during training.

---

## 21. Causal analyses for Model A

GRU has no explicit controller.

Use frozen hidden-state interventions analogous to the existing SRN patching analyses:

```text
token-matched A->B hidden patch
B->A hidden patch
A/B difference-component removal
projection/interpolation controls
```

Purpose:

> Determine whether the GRU's implicit context representation plays the same causal role as Model B's explicit controller.

---

## 22. Human comparison hierarchy

Interpret results in this order.

### Level 1 — familiar acquisition
Does the model reproduce A/B differentiation?

### Level 2 — first-exposure Mixed behavior
Does structured prior improve Mixed relative to matched shuffled control?

### Level 3 — transition profile
Does the model reproduce the structured human transition-specific Mixed pattern?

### Level 4 — mechanism
Does the context representation/controller state explain behavior causally?

Do not claim mechanistic success if Levels 1–2 fail.

---

## 23. Statistical plan

Primary model:

```text
Mixed metric ~ prior_condition * architecture + (1|seed)
```

Planned within-architecture contrasts:

```text
A-S - A-Sh
B-S - B-Sh
```

For time-matched probe analyses:

```text
metric ~ prior_condition * architecture * probe_phase + (1|seed)
```

For transition-level analyses, optionally add:

```text
(1|transition)
```

if compatible with the existing analysis framework.

Report:

```text
paired mean differences
bootstrap 95% confidence intervals
standardized effects
seed-direction counts
```

Human-profile similarity metrics remain descriptive unless a separate inferential procedure is preregistered.

---

## 24. Seed count

Use:

```text
20 seeds per primary group
```

for:

```text
A-S
A-Sh
B-S
B-Sh
B-S-RP
```

Use the validated H8 seed bank if directly comparable.

Do not reduce below 20 paired structured/shuffled seeds for the primary causal contrasts.

---

## 25. No human-data model selection rule

After human probes are generated, prohibit:

```text
changing architecture
changing context-state dimension
changing target learning rate
changing target optimizer
changing synthetic checkpoint
changing cue scaling
changing human geometry
changing probe schedule
adding a model because it fits Mixed better
```

Unexpected failure should first be treated as a scientific result or implementation bug.

Do not reopen synthetic architecture search based on human Mixed performance.

---

## 26. TDD / RED -> GREEN requirements

### T1 — checkpoint hashes
Verify A/B structured and shuffled models load the exact confirmatory checkpoints.

### T2 — source/config freeze
Verify locked scientific source and config hashes before launch.

### T3 — human timeline mapping
Every human probe phase maps to the correct model training checkpoint.

### T4 — no probe learning
Parameter hashes are identical before and after each frozen Mixed/Novel probe.

### T5 — first-exposure isolation
Primary rows must have:

```text
probe_exposure_index = 1
```

### T6 — no latent labels
Models never receive A/B/Mixed/Novel labels as features.

### T7 — no target leakage into Model B controller
Controller cannot receive current/future prediction targets.

### T8 — matched structured/shuffled target training
Within architecture, assert identical human sequence, geometry, A/B schedule, optimizer, LR, update count, RT objective, and probe schedule.

### T9 — execution-head reset
Human RT readout is freshly initialized at target start.

### T10 — familiar gate enforcement
Mixed/Novel primary inference cannot run for failed familiar groups/seeds under the preregistered gate.

### T11 — whole-profile canonicalization
A/B relabeling swaps the entire transition profile only.

### T12 — intervention isolation
Controller/hidden interventions run only on frozen copies.

---

## 27. Required outputs

### Freeze/config

```text
human_transfer_locked_config.json
human_transfer_checkpoint_hashes.json
human_transfer_source_hashes.json
human_model_timeline.csv
human_transfer_group_registry.csv
```

### Training

```text
human_transfer_training.csv
human_transfer_rt_training.csv
human_transfer_controller_training.csv
```

### Familiar gates

```text
human_transfer_familiar_gates.csv
human_transfer_state_space.csv
human_transfer_hidden_separation.csv
human_transfer_controller_separation.csv
```

### Probes

```text
human_transfer_probe_raw.csv
human_transfer_first_exposure.csv
human_transfer_mixed_metrics.csv
human_transfer_novel_metrics.csv
human_transfer_hidden_geometry.csv
human_transfer_controller_geometry.csv
```

### Prior retention

```text
human_transfer_prior_retention.csv
```

### Causal interventions

```text
human_transfer_context_swap.csv
human_transfer_context_ablation.csv
human_transfer_context_average_control.csv
human_transfer_gru_hidden_interventions.csv
```

### Statistics

```text
human_transfer_planned_contrasts.csv
human_transfer_bootstrap.csv
```

### Main report

```text
HUMAN_CONTEXT_CONTROL_TRANSFER_REPORT.md
```

---

## 28. Required figures

1. Full training logic: synthetic prior -> human A/B learning -> frozen Mixed/Novel.
2. Human/model timeline alignment.
3. Familiar A/B phenotype by group.
4. First Mixed RT profile: Fast / Mixed / Slow.
5. Structured-minus-shuffled Mixed causal contrasts.
6. Human versus model transition-specific Mixed profile.
7. Model B controller geometry: A/B/Mixed/Novel.
8. Shared/context-specific hidden geometry.
9. B-S versus B-S-RP restricted-plasticity comparison.
10. Prior retention over human A/B learning.
11. Controller causal interventions.
12. Human-task differentiation-transfer frontier.

---

## 29. Decision logic

### Outcome A — A-S beats A-Sh and passes familiar gate
A conventional GRU can internalize a transferable latent-context prior without an explicit controller.

### Outcome B — B-S beats B-Sh and passes familiar gate
A low-dimensional context-inference state transfers and improves ambiguous retrieval while preserving context-specific learning.

### Outcome C — both A and B succeed
Strongest broad result. Mechanistic analyses distinguish implicit GRU context coding from an explicit 8D controller state.

### Outcome D — only B succeeds
Explicit separation of context inference from task dynamics appears important.

### Outcome E — only A succeeds
Explicit factorization is unnecessary; GRU gating is sufficient.

### Outcome F — structured models split A/B but do not improve Mixed
The synthetic prior supports specialization but does not transfer to the human ambiguity computation.

### Outcome G — structured models improve Mixed but fail A/B differentiation
Failure of the intended account: the model is overgeneralizing.

### Outcome H — B-S-RP succeeds while full B-S fails
Ordinary target-task plasticity overwrites reusable dynamics; preserving the recurrent core protects the learned context-inference prior.

### Outcome I — structured and shuffled behave similarly
Architecture, rather than learned structured prior experience, explains target-task behavior.

---

## 30. Central success criterion

The strongest result is:

> A model selected entirely from an independent synthetic split-and-ambiguity task enters the actual human A/B motor-learning environment, forms distinct A/B skill representations under ordinary familiar training, and on first exposure to Mixed immediately expresses a structured intermediate/recombined response that is improved specifically by structured prior experience relative to a matched shuffled prior.

For Model B, the key mechanistic chain is:

```text
Mixed cue/layout
-> c_Mixed
-> configured shared sequence dynamics
-> prediction confidence
-> RT
```

with:

```text
no Mixed training
no Novel training
no midpoint loss
no explicit A/B averaging
no human-profile fitting
no human-data-based architecture selection
```

This is the direct target-task test of whether learned contextual inference explains why humans can both split familiar motor skills and flexibly recombine them under ambiguous context.
