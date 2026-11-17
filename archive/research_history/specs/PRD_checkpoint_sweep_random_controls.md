# PRD: Checkpoint Sweep + No-Pretraining Controls
## Identify When the New Models Best Reproduce Familiar A/B Splitting, Then Test Frozen Mixed/Novel Behavior

## 0. Objective

The latest human-transfer run suggests that the new recurrent architectures learn the familiar sequence too efficiently across contexts. They show substantial cross-context transfer, so A and B do not become as behaviorally separated as in the previously validated H8/human familiar phenotype.

This PRD tests two questions:

1. **Does the desired familiar A/B split appear transiently at an earlier target-training checkpoint and then collapse as cross-context transfer increases?**
2. **Is the excessive cross-context transfer caused by the architecture itself, by generic synthetic pretraining, or specifically by structured pretraining?**

The core comparison is:

```text
A-S     = GRU-150, structured synthetic pretraining
A-Sh    = GRU-150, shuffled synthetic pretraining
A-Rand  = GRU-150, no synthetic pretraining

B-S     = Elman-128 + context-GRU-8, structured synthetic pretraining
B-Sh    = Elman-128 + context-GRU-8, shuffled synthetic pretraining
B-Rand  = Elman-128 + context-GRU-8, no synthetic pretraining

H8      = original validated H8 baseline
```

The key scientific constraint is:

> **Checkpoint selection must use familiar A/B behavior only. Mixed/Novel outcomes cannot influence checkpoint selection.**

After selecting familiar-matched checkpoints, run frozen Mixed/Novel probes.

---

# 1. Primary hypotheses

## H1 — transient split hypothesis

The new models may pass through a regime where:

```text
A/B differentiation is strong
cross-context transfer is incomplete
```

before later training drives:

```text
shared sequence reuse
cross-context transfer
behavioral lumping
```

Predicted trajectory:

```text
early -> undertrained
middle -> split + transferable
late -> over-generalized / lumped
```

## H2 — generic pretraining hypothesis

If:

```text
structured ≈ shuffled > random
```

in cross-context transfer, then synthetic pretraining generically teaches reusable sequence structure.

## H3 — structured-prior hypothesis

If:

```text
structured differs from both shuffled and random
```

then learning the context→latent-task relation specifically changes target-task transfer.

## H4 — architecture-only hypothesis

If:

```text
structured ≈ shuffled ≈ random
```

then the target-task behavior is mainly an architectural property.

---

# 2. Locked architecture groups

## Model A

```text
GRU-150
```

Groups:

```text
A-S
A-Sh
A-Rand
```

## Model B

```text
Elman-128 + context-GRU-8
```

Groups:

```text
B-S
B-Sh
B-Rand
```

## Historical baseline

```text
H8
```

Do not add new architectures.

Do not change:

```text
hidden sizes
context-state dimension
human geometry
human sequence
RT readout
target-task optimizer
target-task learning rate
probe schedule
```

during this experiment.

---

# 3. Constructing the no-pretraining controls

## Preferred implementation

For each seed `s`, use the exact parameter state immediately before synthetic pretraining:

```text
A-Rand_s = initial GRU-150 checkpoint for seed s
B-Rand_s = initial Model-B checkpoint for seed s
```

This is the strongest control because it matches:

```text
architecture
parameter initialization
seed
weight distribution
```

and differs only in:

```text
synthetic updates = 0
```

## If initial checkpoints were not saved

Recreate them deterministically using the frozen:

```text
initialization code
initialization config
seed
architecture definition
```

Then verify against stored metadata from the synthetic run:

```text
parameter count
initialization summary statistics
spectral statistics, if available
seed manifest
```

Do not create a new initialization scheme.

## Hard assertion

For A-Rand/B-Rand:

```text
synthetic_pretraining_updates == 0
```

and no synthetic optimizer state may be loaded.

---

# 4. Human target-task training

All six new-model groups receive identical human-style A/B target training.

Use the locked familiar schedule.

No Mixed/Novel example updates model parameters.

Training must be identical within each architecture across:

```text
structured
shuffled
random
```

except for starting checkpoint.

---

# 5. Checkpoint schedule

Save target-task checkpoints densely enough to resolve the expected transition from splitting to cross-context transfer.

Primary schedule:

```text
0
5
10
15
20
30
40
50
60
80
100
120
```

human familiar A/B updates/sequences.

If checkpoints already exist at finer resolution, use them.

Do not rerun training solely to create more checkpoints unless needed for implementation consistency.

Optional denser exploratory schedule:

```text
every 5 familiar sequences
```

but the primary analysis should use the preregistered schedule above.

---

# 6. Familiar-only metrics at every checkpoint

For every:

```text
group × seed × checkpoint
```

compute the following without presenting Mixed/Novel.

## F1 — sequence-learning performance

```text
familiar NLL
accuracy
correct-token confidence
normalized RT
```

## F2 — A/B hidden-state differentiation

Token-matched:

```text
D_hidden(A,B)
```

using the same hidden-state extraction convention as the current human-transfer pipeline.

## F3 — Model-B controller differentiation

For B groups:

```text
D_context(A,B)
```

using the 8D context controller.

## F4 — familiar RT-profile separation

Compute:

```text
RMSE(A,B)
mean absolute A/B difference
transition-specific A/B profile distance
```

## F5 — state-space phenotype

Run the locked:

```text
Lump
Cross
Split
```

comparison at each checkpoint.

Store:

```text
winner
Cross-vs-Lump evidence
Split-vs-Lump evidence
model score differences using the existing criterion
```

## F6 — switch/relearning cost

At each A→B and B→A transition, compute first-sequence/block cost relative to settled preceding-context performance.

Use prediction and RT versions.

Example:

```text
switch_cost_RT = RT_first_new_context - RT_late_previous_context
switch_cost_NLL = NLL_first_new_context - NLL_late_previous_context
```

## F7 — cross-context transfer index

Primary transfer index should quantify how much learning in the previous context carries into the first exposure to the new context.

Use a normalized metric with intuitive direction:

```text
T = 1 - (cost_at_context_switch / initial_learning_cost)
```

where:

```text
T ≈ 1 -> near-complete transfer / weak split
T ≈ 0 -> little transfer / strong split
```

Implement separately for:

```text
RT
NLL
confidence
```

Also retain all raw numerator/denominator components so the index can be audited.

---

# 7. Human/H8 familiar-match target

Checkpoint selection must be based only on the familiar phenotype.

Use two references.

## Reference 1 — historical H8 familiar phenotype

For each metric, use the validated H8 summary over the corresponding familiar phase.

## Reference 2 — human familiar phenotype

Where directly comparable, include:

```text
switch cost
A/B transition profile
acquisition trajectory
```

Do not use:

```text
Mixed
Novel
NoContext
```

in checkpoint selection.

---

# 8. Familiar-only checkpoint selection score

Create a prespecified score:

```text
S_familiar(checkpoint)
```

with components:

1. familiar acquisition/profile match;
2. switch-cost match;
3. A/B profile-separation match;
4. state-space Cross/Split evidence;
5. A/B hidden differentiation.

Primary implementation:

```text
S_familiar =
  z(profile_match)
+ z(switch_cost_match)
+ z(AB_behavioral_separation_match)
+ z(CrossSplit_evidence)
+ z(hidden_separation)
```

Use equal weights:

```text
w1 = w2 = w3 = w4 = w5 = 1
```

Do not tune weights based on Mixed.

Secondary robustness analysis:

```text
mean rank across the five familiar components
```

This avoids dependence on metric scaling.

---

# 9. Selection granularity

Select one checkpoint per:

```text
architecture × prior condition
```

using the mean familiar score across seeds.

Thus select separately:

```text
A-S
A-Sh
A-Rand
B-S
B-Sh
B-Rand
```

Do not choose a separate primary checkpoint for each seed.

Reason:

> Per-seed checkpoint optimization adds unnecessary flexibility and would make the subsequent Mixed test difficult to interpret.

A secondary exploratory analysis may show each seed's individually best familiar checkpoint.

---

# 10. Familiar-match minimum gate

Before a selected checkpoint enters the primary Mixed analysis, require:

```text
sequence learning = pass
A/B hidden differentiation > null
nonzero switch/relearning cost
Cross or Split preferred over Lump at group level
confidence->RT relationship remains negative
```

For B additionally:

```text
controller A/B separation > null
```

If no checkpoint for a group satisfies this minimum familiar gate:

```text
mark group = no familiar-matched checkpoint
```

and do not include it in the primary Mixed comparison.

Still report its full checkpoint trajectory.

---

# 11. Freeze checkpoint selection before Mixed

Create:

```text
selected_familiar_checkpoints.json
```

containing:

```text
group
selected_checkpoint
S_familiar
gate_status
all component scores
source hashes
```

Hash and freeze this file.

Only after it is written and locked may the Mixed/Novel probe code read the selected checkpoints.

Software barrier:

```text
Mixed probe runner requires selected_familiar_checkpoints.json
```

and checkpoint-selection code must not import/read Mixed or Novel output files.

---

# 12. Primary frozen Mixed test

For each gate-passing selected checkpoint:

1. load selected checkpoint;
2. freeze weights;
3. reset hidden/controller states according to the locked probe convention;
4. present the actual human Mixed context;
5. collect first-exposure output only.

No weight updates.

Measure:

```text
Mixed RT profile
Mixed NLL
Mixed confidence
Mixed hidden state
```

For B:

```text
Mixed controller state
```

---

# 13. Primary Mixed metrics

## M1 — between-endpoint fraction

Use full-profile canonicalized familiar endpoints.

## M2 — RT deficit

Distance outside familiar Fast/Slow interval.

## M3 — NLL deficit

Relative to familiar endpoints.

## M4 — human profile correspondence

Descriptive only:

```text
correlation
RMSE
profile R²
```

No fitting.

## M5 — hidden corridor position

```text
projection coordinate
off-axis distance
inside-segment fraction
```

## M6 — controller geometry for B

```text
q_context
off-axis controller distance
distance to A
distance to B
```

---

# 14. Novel probe

After Mixed evaluation, run frozen Novel using the same selected familiar checkpoints.

Novel is secondary.

Report:

```text
NLL deficit
RT deficit
accuracy
shared-subspace engagement
distance from A/B hidden manifold
```

For B:

```text
distance from A/B controller manifold
```

Do not use Novel to reselect checkpoints.

---

# 15. Full checkpoint-trajectory Mixed analysis

After the primary familiar-based checkpoint selection and Mixed analysis are frozen, run an explicitly exploratory sweep:

```text
Mixed frozen probe at every familiar checkpoint
```

Purpose:

> Determine whether the desired split + Mixed regime ever appears transiently.

For every checkpoint, plot:

```text
A/B differentiation
cross-context transfer
Mixed between-endpoint fraction
Mixed RT deficit
Mixed NLL deficit
```

Color by checkpoint.

Mark this analysis:

```text
analysis_type = exploratory
```

---

# 16. Critical frontier plots

## Plot 1 — cross-context transfer vs A/B differentiation

```text
x = cross-context transfer index
y = A/B hidden or behavioral differentiation
color = checkpoint
```

## Plot 2 — A/B differentiation vs Mixed quality

```text
x = A/B differentiation
y = Mixed transfer quality
color = checkpoint
```

## Plot 3 — cross-context transfer vs Mixed quality

```text
x = cross-context transfer index
y = Mixed between-endpoint fraction or negative RT deficit
color = checkpoint
```

The key question is whether a middle regime exists with:

```text
strong enough split
+
better Mixed flexibility
```

---

# 17. Model-B mechanism across checkpoints

For B-S/B-Sh/B-Rand at every checkpoint, separately track:

```text
D_context(A,B)
D_hidden(A,B)
behavioral A/B separation
cross-context transfer
```

This distinguishes:

## Mechanism 1 — controller collapse

```text
c_A ≈ c_B
```

Interpretation: context inference itself becomes lumped.

## Mechanism 2 — controller remains split, task core collapses

```text
c_A != c_B
h_A ≈ h_B
```

Interpretation: context is inferred but no longer strongly configures separate skill states.

## Mechanism 3 — representation split but output shared

```text
c_A != c_B
h_A != h_B
behavior A ≈ B
```

Interpretation: downstream readout/execution expression collapses despite internal separation.

This decomposition is a central output.

---

# 18. Structured vs shuffled vs random interpretation

Within each architecture:

## Case A

```text
structured ≈ shuffled ≈ random
```

Conclusion: architecture itself drives the transfer/lumping trajectory.

## Case B

```text
structured ≈ shuffled != random
```

Conclusion: generic synthetic pretraining changes target learning independent of useful contextual structure.

## Case C

```text
structured != shuffled ≈ random
```

Conclusion: structured contextual prior specifically changes target learning.

## Case D

```text
structured != shuffled != random
```

Conclusion: both generic pretraining and structured prior contribute.

Primary comparisons:

```text
A-S vs A-Sh vs A-Rand
B-S vs B-Sh vs B-Rand
```

---

# 19. H8 role

Use H8 as a familiar-phenotype anchor, not as a matched pretraining control.

At comparable exposure levels, report:

```text
switch cost
A/B differentiation
Cross/Split evidence
RT profile
cross-context transfer
```

The purpose is to identify the familiar-behavior region historically associated with successful splitting.

---

# 20. Seed count

Use:

```text
20 seeds/group
```

for:

```text
A-S
A-Sh
A-Rand
B-S
B-Sh
B-Rand
```

Pair seeds across structured/shuffled/random within architecture.

Use the same seed IDs where possible.

---

# 21. Statistical analysis

For familiar metrics:

```text
metric ~ prior_condition * checkpoint + (1|seed)
```

within each architecture.

For selected-checkpoint Mixed:

```text
metric ~ prior_condition + (1|seed)
```

within each architecture.

Primary planned contrasts:

```text
structured - shuffled
structured - random
shuffled - random
```

Report:

```text
paired mean difference
bootstrap 95% CI
standardized effect
seed-direction count
```

Checkpoint × condition interactions are important for the trajectory analysis.

---

# 22. No-retuning rule

After this PRD begins, do not change:

```text
target learning rate
optimizer
architecture
hidden size
context-state dimension
human geometry
RT mechanism
checkpoint-selection score weights
minimum familiar gate
Mixed metric definitions
```

based on results.

If a group never reaches the familiar phenotype, report that as the result.

---

# 23. TDD / RED -> GREEN requirements

## T1 — zero-pretraining assertion

For A-Rand/B-Rand:

```text
synthetic_pretraining_updates == 0
```

## T2 — seed-matched initialization

Structured/shuffled/random seed triplets must originate from the same initialization seed where possible.

## T3 — checkpoint completeness

Every required checkpoint exists for every seed or is explicitly marked missing.

## T4 — no Mixed access during selection

Checkpoint-selection code must fail if passed any Mixed/Novel file.

## T5 — selection artifact lock

Mixed runner requires a frozen:

```text
selected_familiar_checkpoints.json
```

## T6 — full-profile canonicalization

A/B relabeling swaps whole profiles only.

## T7 — frozen probes

Parameter hashes identical before/after Mixed and Novel probes.

## T8 — first-exposure primary

Primary Mixed row must satisfy:

```text
probe_exposure_index == 1
```

## T9 — matched target training

Within architecture, structured/shuffled/random groups must share:

```text
human sequence
geometry
schedule
optimizer
learning rate
number of updates
RT objective
```

## T10 — random-control reproducibility

Reconstructed random checkpoint metadata must match frozen initialization metadata.

## T11 — no per-seed primary checkpoint cherry-picking

Primary selected checkpoint is one group-level checkpoint per condition.

## T12 — exploratory sweep labeling

Any all-checkpoint Mixed sweep must be explicitly labeled exploratory.

---

# 24. Required outputs

## Initialization/control audit

```text
no_pretraining_control_manifest.csv
initialization_reconstruction_audit.csv
```

## Familiar checkpoint sweep

```text
checkpoint_familiar_metrics.csv
checkpoint_transfer_indices.csv
checkpoint_state_space.csv
checkpoint_hidden_separation.csv
checkpoint_controller_separation.csv
checkpoint_h8_reference.csv
```

## Selection

```text
familiar_match_component_scores.csv
selected_familiar_checkpoints.json
selected_familiar_checkpoints.csv
```

## Primary probes

```text
selected_checkpoint_mixed_raw.csv
selected_checkpoint_mixed_metrics.csv
selected_checkpoint_novel_raw.csv
selected_checkpoint_novel_metrics.csv
```

## Exploratory trajectory

```text
all_checkpoint_mixed_raw.csv
all_checkpoint_mixed_metrics.csv
checkpoint_frontier_metrics.csv
```

## Statistics

```text
checkpoint_sweep_planned_contrasts.csv
checkpoint_sweep_bootstrap.csv
```

## Report

```text
CHECKPOINT_SWEEP_AND_RANDOM_CONTROL_REPORT.md
```

---

# 25. Required figures

1. Familiar acquisition trajectories: A-S, A-Sh, A-Rand, B-S, B-Sh, B-Rand, H8.
2. Cross-context transfer over training.
3. A/B differentiation over training.
4. State-space phenotype over training.
5. Familiar-match score over training, with selected checkpoint marked.
6. Selected-checkpoint familiar Fast/Slow profiles.
7. Selected-checkpoint first Mixed Fast/Mixed/Slow profiles.
8. Structured/shuffled/random Mixed contrasts within A and B.
9. Checkpoint A/B-differentiation vs Mixed-transfer frontier.
10. Cross-context-transfer vs Mixed-transfer frontier.
11. Model-B controller vs task-core separation over training.
12. H8 familiar region overlaid on checkpoint trajectories.

---

# 26. Decision logic

## Outcome 1 — desired transient regime exists

A model passes through a checkpoint with:

```text
H8/human-like familiar splitting
+
better Mixed transfer than H8
```

Interpretation:

> The desired computation exists transiently but is later eroded by cross-context transfer.

## Outcome 2 — structured beats shuffled and random at familiar-matched checkpoint

Interpretation:

> The learned contextual prior improves flexible retrieval beyond architecture and generic pretraining.

## Outcome 3 — structured ≈ shuffled, both beat random

Interpretation:

> Generic synthetic pretraining, not contextual structure specifically, drives the effect.

## Outcome 4 — all three conditions look similar

Interpretation:

> Architecture dominates; pretraining is not responsible.

## Outcome 5 — random splits more strongly than pretrained models

Interpretation:

> Pretraining actively promotes cross-context reuse and reduces target-task separation.

## Outcome 6 — no new-model checkpoint ever reproduces familiar splitting

Interpretation:

> The architecture/objective combination fundamentally favors transfer over context-specific skill formation in this target task.

## Outcome 7 — Model B controller stays separated while task-core/behavior collapses

Interpretation:

> Context inference survives, but inferred context no longer strongly configures separate task states.

This would motivate revisiting the target objective rather than the context-inference architecture.

---

# 27. Central success criterion

The strongest result would be:

> A structured-pretrained model reaches a familiar-only selected checkpoint where its A/B acquisition and switch behavior resembles the validated split phenotype, yet its frozen first Mixed response is more human-like than matched shuffled, no-pretraining, and H8 controls.

That would show that the new architecture/prior can support both:

```text
context-specific skill formation
and
flexible first-exposure retrieval
```

without using Mixed behavior to choose when to test the model.
