# PRD: Human-Task Transfer of a Meta-Learned Context-Generalization Prior
## H128 Causal Transfer Test After Synthetic Structural-Prior Validation

## 0. Objective

Test whether the independently validated H128 meta-learned structural prior transfers into the human-style A/B motor-sequence task.

The core hypothesis is:

\[
\boxed{
\text{episodically meta-learned context→task structure}
\rightarrow
\text{better first-exposure generalization to Mixed/Novel}
}
\]

while preserving:

\[
\boxed{
\text{strong A/B context-specific learning}
}
\]

This is now a **target-task transfer experiment**, not another synthetic-pretraining search.

No additional tuning of the synthetic ecology, meta-learning hyperparameters, or H128 checkpoint is allowed based on human Mixed/Novel performance.

---

# 1. Locked upstream result

Use the already selected H128 synthetic-meta configuration:

```text
hidden size = 128
recurrent initialization = I3_orthogonal
inner alpha = 0.01
inner steps = 10
meta-batch = 16
```

The H128 structured-meta model passed:

```text
copy retention
meta > ordinary
structured-meta > shuffled-meta
seed consistency
similarity-gradient gate
asymmetric-query gate
```

Do not reselect any of these settings using the human task.

---

# 2. Primary scientific question

Does a network that has already learned a reusable contextual-generalization prior:

1. still learn separate A/B skill representations in the target task, and
2. express a previously unseen ambiguous context more like humans?

The decisive result requires both:

\[
A \neq B
\]

and:

\[
Mixed \text{ shows structured reuse of A/B rather than catastrophic OOD behavior.}
\]

A model that improves Mixed only by failing to differentiate A and B does **not** support the hypothesis.

---

# 3. Human-transfer groups

Run five groups.

## G1 — H128 P2_meta

Critical experimental group.

```text
H = 128
structured P2 episodic meta-trained checkpoint
```

## G2 — H128 P1_shuffled_meta

Critical causal control.

Same:

```text
architecture
capacity
number of meta-training episodes
inner-loop procedure
outer-loop optimizer
meta-batch
synthetic exposure
```

but context→task structure was shuffled.

## G3 — H128 P2_ordinary

Controls for structured prior experience without episodic meta-learning.

```text
same structured P2 ecology
ordinary pretraining
H = 128
```

## G4 — H128 random-init

Controls for all pretraining.

Use the target-task initialization convention chosen before this human-transfer run.

Do not initialize from any human-tuned checkpoint.

## G5 — H8 original

Continuity anchor with the original model.

Use the existing validated:

```text
H = 8
High-scale target-task initialization
same human-task code
same RT mechanism
```

This group is descriptive/anchoring rather than part of the primary causal contrast.

---

# 4. Primary causal contrasts

## C1 — Structure-specific meta-learning effect

\[
\boxed{
G1 > G2
}
\]

on first-exposure Mixed transfer while familiar A/B gates remain intact.

This is the strongest contrast.

## C2 — Episodic-learning effect

\[
\boxed{
G1 > G3
}
\]

This asks whether episodic pressure adds value beyond structured experience alone.

## C3 — Generic-pretraining effect

Compare:

```text
G2 vs G4
G3 vs G4
```

to determine whether generic meta-training or ordinary pretraining changes transfer.

## C4 — Continuity with the original SRN

Compare:

```text
G1 vs G5
```

descriptively.

Do not use G5 as the primary causal control because capacity differs.

---

# 5. Target-task architecture

For H128 groups:

\[
h_t=\tanh(W_xx_t+W_hh_{t-1}+b_h)
\]

with the same next-token prediction head.

At transfer:

```text
retain pretrained SRN weights for G1-G3
reset optimizer state
reset recurrent hidden state
freshly initialize execution/RT readout
```

For G4:

```text
fresh H128 SRN
fresh execution readout
```

For G5:

```text
existing H8 target-task initialization
fresh execution readout
```

---

# 6. RT / execution mechanism

Use the previously validated monotonic execution readout:

\[
v=
\sigma
\left[
\operatorname{softplus}(a)
\operatorname{logit}(\operatorname{stopgrad}p)
+b
\right]
\]

with:

\[
RT_{norm}=1-v.
\]

Use the same target-task execution objective as the current successful SRN model.

No RT targets were used during synthetic meta-training.

The execution readout must be newly initialized for every target-task run.

---

# 7. Human-style training environment

Use the locked human-grounded environment:

```text
G2_shared_x / human SVG-derived geometry
same fixed 8-item sequence used in the current paper
A10 / B10 alternating blocks
170 external training sequences
same RSI / sequence timing conventions
same context-input encoding as the current human-matched SRN
```

Do not change the target sequence, geometry, block length, or A/B schedule between groups.

---

# 8. Training schedule

Target training:

```text
A10
B10
A10
B10
...
170 total external sequences
```

Use the same optimizer family and target-task learning rate across all H128 groups.

If H8 historically uses a different optimizer setting because of locked prior implementation, preserve it and report the difference.

Do not tune H128 target learning rate on Mixed/Novel outcomes.

If a target learning-rate sanity check is needed, it may only use:

```text
A/B training loss
numerical stability
```

and must be completed before Mixed/Novel probe files are generated.

---

# 9. Probe checkpoints

Run frozen probes at:

```text
20 sequences
50 sequences
90 sequences
170 sequences
```

Each probe is run on a disposable model copy.

No optimizer updates.

No persistent hidden-state contamination.

Primary manuscript comparison:

```text
170-sequence checkpoint
```

Primary theoretical timing analysis:

```text
first Mixed exposure at each checkpoint
```

The earlier checkpoints test whether the prior is available immediately rather than learned from probe exposure.

---

# 10. Frozen probe battery

At each checkpoint evaluate:

```text
A
B
Mixed
Novel
```

Optionally include:

```text
context-neutral/default
```

only if it already exists in the target-task implementation and is not newly designed after seeing results.

For each context collect:

```text
next-token probabilities
prediction NLL
prediction accuracy
hidden states by transition
correct-token confidence
normalized RT
```

---

# 11. Familiar A/B gate

Mixed and Novel are only interpretable for a run if familiar A/B behavior is valid.

Require:

## F1 — Sequence learning

Target-task NLL decreases over training.

## F2 — Context differentiation

A/B hidden-state distance exceeds the preregistered random/permutation baseline.

## F3 — Context-yoked execution

A/B RT profiles show reliable context differentiation at settled training.

## F4 — State-space phenotype

Majority of seeds favor:

```text
Cross
or
Split
```

over Lump under the existing model-comparison pipeline.

## F5 — Monotonic execution relation

\[
\operatorname{corr}(p_{correct},RT)<0.
\]

## F6 — No gross target-task failure

No:

```text
NaN
exploding NLL
chance-level settled sequence performance
```

If G1 improves Mixed by becoming Lump, the central hypothesis fails.

---

# 12. Canonicalized familiar endpoints

For each seed independently:

```text
Fast = familiar context with lower mean settled RT
Slow = familiar context with higher mean settled RT
```

If A/B labels are swapped:

```text
swap the entire transition profile
```

not individual transitions.

This removes arbitrary A/B label symmetry.

Use the same canonicalization for all probe metrics derived from familiar endpoints.

---

# 13. Primary Mixed outcome

The critical test is the **first frozen Mixed exposure**.

Primary metric:

```text
transition-level between-endpoint fraction
```

For transition \(k\):

\[
I_k
=
\mathbb{1}
[
RT_{Fast,k}
\le
RT_{Mixed,k}
\le
RT_{Slow,k}
].
\]

Then:

\[
B_{Mixed}
=
\frac{1}{K}
\sum_k I_k.
\]

Primary causal test:

\[
\boxed{
B_{Mixed}^{G1}
>
B_{Mixed}^{G2}
}
\]

with G1 also passing familiar A/B gates.

---

# 14. Secondary Mixed outcomes

Report:

## M1 — RT deficit

Distance outside the familiar interval.

## M2 — Prediction NLL deficit

Relative to familiar endpoints.

## M3 — Global retrieval coefficient

Descriptive only:

\[
q
=
\arg\min_q
\|Mixed-(qA+(1-q)B)\|.
\]

## M4 — Transition-specific coefficients

\[
q_k
=
\frac{RT_{Mixed,k}-RT_{Slow,k}}
{RT_{Fast,k}-RT_{Slow,k}}.
\]

Descriptive only.

## M5 — Hidden-state corridor

Project Mixed into the A/B difference subspace and report:

```text
inside-corridor fraction
off-corridor distance
projection coordinate
```

## M6 — Human-profile correspondence

Compare model transition profile with the observed human Mixed profile using:

```text
correlation
RMSE
profile R²
```

Do not fit the model to the human profile.

---

# 15. Novel-context outcomes

Do **not** require Novel to lie exactly between A and B.

Primary Novel metrics:

```text
prediction NLL deficit
RT deficit
accuracy
distance to familiar A/B corridor
distance to shared/context-neutral representation
```

Secondary:

```text
between-endpoint fraction
q
```

Interpret Novel as potentially recruiting shared task structure rather than literal A/B averaging.

---

# 16. First-exposure requirement

Primary probe outcomes must use the first sequence/traversal in each frozen probe context.

Do not average across repeated Mixed/Novel exposures for the primary result.

Repeated frozen exposures may be shown as a secondary stability analysis only if:

```text
no weights update
hidden state resets between repeated probes
```

---

# 17. Seed count

## Pilot / implementation verification

```text
3 seeds / group
```

Purpose:

```text
pipeline validation only
```

Do not make scientific claims.

## Confirmatory human transfer

Run:

```text
20 seeds / group
```

for G1-G4.

For G5, reuse the existing H8 seed bank if methodologically identical; otherwise run 20 matched seeds.

Use paired seeds wherever initialization/pretraining checkpoints are paired.

---

# 18. Statistical analysis

## Primary model

For Mixed between-endpoint fraction:

```text
between_fraction ~ group * checkpoint + (1|seed)
```

Primary planned contrasts:

```text
G1 - G2
G1 - G3
```

at checkpoint 170.

Also report early-checkpoint contrasts.

## Continuous deficits

Use:

```text
metric ~ group * checkpoint + (1|seed)
```

for:

```text
RT deficit
NLL deficit
off-corridor distance
```

Report:

```text
paired mean difference
bootstrap 95% CI
standardized effect
```

Do not rely on p-values alone.

---

# 19. Mechanistic interpretation tests

Run only after primary behavioral analysis is complete.

## H1 — A/B differentiation versus Mixed transfer

Plot:

```text
x = A/B hidden separation
y = Mixed between-endpoint fraction
```

Question:

> Does G1 occupy the desirable region of strong differentiation plus good transfer?

## H2 — Shared versus context-specific subspaces

Decompose hidden activity into:

```text
shared sequence component
A/B difference component
residual
```

Compare Mixed and Novel across G1-G4.

## H3 — Prior retention

Using disposable copies, re-run the synthetic V1/V2 prior battery:

```text
before target training
after 20
after 50
after 90
after 170
```

Question:

> Does human-style A/B training overwrite the meta-learned prior?

---

# 20. Weight-ablation follow-up

Only if G1 improves Mixed.

Before target-task training, create:

```text
G1_input_reset
G1_recurrent_reset
G1_output_reset
```

where one parameter subsystem is reinitialized and the others retain meta-trained weights.

Run the target A/B task identically.

Purpose:

```text
localize where the transferable prior is stored
```

This is a follow-up, not part of the primary test.

---

# 21. Capacity extension: 256H / 512H

Do **not** include H256 or H512 in the primary human-transfer experiment.

H128 has already passed the synthetic structural-prior gate. The next question is transfer, not further capacity search.

A larger-width extension is allowed only under the following preregistered triggers.

## Trigger C1 — Positive H128 human transfer

If G1:

```text
passes familiar A/B gates
and
beats G2 on the primary Mixed measure
```

then optionally run:

```text
H256
H512
```

as a **post hoc scaling/robustness experiment**.

Purpose:

> determine whether the phenomenon strengthens, saturates, or deteriorates with excess capacity.

These larger models must not replace H128 as the primary result.

## Trigger C2 — H128 passes familiar gates but does not improve Mixed

Do **not** automatically escalate to 256/512.

Interpretation should first be:

```text
the learned synthetic prior did not transfer to the target task
```

Larger width would be post hoc model searching.

Only test larger widths if an independent diagnostic demonstrates that H128 is capacity-limited on the target computation.

## Trigger C3 — H128 fails familiar A/B learning

Do not infer anything about Mixed.

First determine whether failure is due to:

```text
catastrophic forgetting
target-task optimization
pretrained initialization incompatibility
```

A larger-width experiment is not the default rescue.

---

# 22. Optional scaling experiment

If Trigger C1 is met, run:

```text
H128
H256
H512
```

for:

```text
P2_meta
P1_shuffled_meta
```

only.

Do not rerun all five groups.

For H256/H512:

1. repeat the **synthetic inner-loop calibration** independently;
2. require copy-learning gate;
3. require structured-meta > shuffled-meta synthetic gate;
4. only then run human transfer.

Do not simply transplant H128 hyperparameters.

This keeps the capacity comparison scientifically valid.

---

# 23. Why 512H is not required for the main claim

The main claim is an existence/causal-transfer claim:

\[
\text{learned structural prior}
\rightarrow
\text{human-like ambiguous-context transfer}.
\]

Since H128 already demonstrates the structural prior synthetically, increasing width before testing transfer does not strengthen the causal design.

It instead changes the question to:

\[
\text{how does the effect scale with capacity?}
\]

That is useful only after establishing the phenomenon.

---

# 24. TDD requirements

Use RED → GREEN → REFACTOR.

## T1 — Checkpoint identity

Verify G1-G3 load the exact locked synthetic checkpoints.

Hash weights.

## T2 — Optimizer reset

Assert no synthetic optimizer state survives target transfer.

## T3 — RT readout reset

Assert execution-head parameters are newly initialized.

## T4 — Human probe firewall

No Mixed/Novel values can influence:

```text
learning rate
checkpoint selection
seed exclusion
training duration
```

## T5 — Frozen probes

Hash all trainable parameters before and after every probe.

They must be identical.

## T6 — Hidden-state reset

Each probe context starts from the same defined reset state.

## T7 — First-exposure isolation

Primary Mixed/Novel rows must be explicitly tagged:

```text
probe_exposure_index = 1
```

## T8 — Whole-profile canonicalization

Unit test A/B swap to ensure all transitions swap together.

## T9 — Familiar-gate enforcement

Mixed/Novel interpretation code must refuse to generate primary inference for failed familiar runs.

## T10 — Cross-group training equality

Assert identical:

```text
A/B schedule
sequence
geometry
number of updates
target optimizer
RT objective
```

for G1-G4.

---

# 25. Required outputs

## Configuration

```text
human_meta_transfer_config.json
human_meta_transfer_group_registry.csv
human_meta_transfer_checkpoint_hashes.json
```

## Training

```text
human_meta_transfer_training.csv
human_meta_transfer_rt_training.csv
```

## Familiar gates

```text
human_meta_transfer_familiar_gates.csv
human_meta_transfer_state_space.csv
human_meta_transfer_hidden_separation.csv
```

## Probes

```text
human_meta_transfer_probe_raw.csv
human_meta_transfer_first_exposure.csv
human_meta_transfer_mixed_metrics.csv
human_meta_transfer_novel_metrics.csv
human_meta_transfer_hidden_geometry.csv
```

## Statistics

```text
human_meta_transfer_planned_contrasts.csv
human_meta_transfer_bootstrap.csv
```

## Report

```text
HUMAN_META_PRIOR_TRANSFER_REPORT.md
```

---

# 26. Required figures

## Figure 1 — Experimental logic

```text
structured meta-pretraining
vs shuffled meta
vs structured ordinary
vs random
-> identical A/B target training
-> frozen Mixed/Novel
```

## Figure 2 — Familiar A/B learning

By group:

```text
training NLL
RT
A/B differentiation
```

## Figure 3 — State-space phenotype

Lump / Cross / Split proportions by group.

## Figure 4 — First Mixed profile

Plot:

```text
Fast
Mixed
Slow
```

for G1-G4.

## Figure 5 — Mixed between-endpoint fraction

By group and checkpoint.

## Figure 6 — Mixed NLL / RT deficit

By group.

## Figure 7 — Novel profile

Familiar endpoints + Novel + shared/default if available.

## Figure 8 — Hidden corridor geometry

A/B line/subspace with Mixed and Novel projections.

## Figure 9 — Differentiation-transfer frontier

```text
x = A/B hidden separation
y = Mixed transfer quality
```

## Figure 10 — Synthetic-prior retention

Prior score over target-training checkpoints.

---

# 27. Decision logic

## Outcome A — G1 preserves A/B split and beats G2 on first Mixed transfer

Primary success.

Interpretation:

\[
\boxed{
\text{meta-learned structural prior transfers into human-like ambiguous retrieval}
}
\]

without Mixed training or explicit averaging.

## Outcome B — G1 beats G3 but not G2

Interpretation:

```text
generic meta-learning matters,
but structured context→task covariance is not the causal ingredient.
```

## Outcome C — G1 beats G2 but A/B separation collapses

Failure of the intended account.

Interpretation:

```text
generalization improved by preventing specialization.
```

## Outcome D — G1 passes synthetic prior test and familiar A/B gates but not Mixed transfer

Interpretation:

```text
the prior is learnable but does not transfer to the target motor-sequence ecology.
```

Do not immediately escalate capacity.

## Outcome E — G1 improves Mixed and Novel

Strongest broad-transfer result.

Then test mechanistic ablations and optional capacity scaling.

## Outcome F — G1 improves Mixed but not Novel

Still theoretically coherent.

Interpretation:

```text
ambiguous reuse of familiar contexts and novel-context fallback are distinct computations.
```

---

# 28. Central claim criterion

The target result is:

> A structurally meta-trained H128 SRN first learns distinct A/B context-specific skills in the human-style task, then immediately expresses an unseen Mixed context as a structured reuse of those skills, outperforming both a shuffled-meta control and a structured ordinary-pretraining control.

This must occur with:

```text
no Mixed training
no Novel training
no human RT pretraining
no midpoint objective
no explicit A/B output averaging
no human-probe-based model selection
```

That is the clean causal test of whether a learned structural prior can explain the gap between the original SRN and human behavior.
