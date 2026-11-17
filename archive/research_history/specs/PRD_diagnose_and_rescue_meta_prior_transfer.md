# PRD: Diagnose and Rescue Meta-Prior Transfer in the Human A/B → Mixed/Novel Task
## Baseline Replay Audit → Exact Meta-Test Adaptation → Conditional Boundary-Aware Meta-Learning

## 0. Objective

The previous H128 transfer experiment produced two problems that must be separated before any further model search:

1. the original H8 continuity baseline did not reproduce the previously validated familiar A/B phenotype;
2. H128 models learned a synthetic structural prior, but the human-target transfer run did not preserve the required A/B specialization.

This PRD therefore implements a strictly gated diagnostic program:

```text
Stage A — replay/audit the original H8 baseline
        ↓ only if reproduced
Stage B — test H128 meta-trained checkpoints using the exact adaptation operator
          they were meta-trained to use
        ↓
        ├─ if structured meta does NOT beat shuffled meta:
        │      stop P2; conclude synthetic P2 prior does not transfer
        │
        └─ if structured meta DOES beat shuffled meta:
               diagnose prior overwriting under long target training
               and proceed to Stage C
               ↓
Stage C — meta-train a boundary-aware ecology that rewards both
          differentiation and generalization
```

Do not test H256/H512 in this PRD. The purpose is diagnosis, not capacity expansion.

# 1. Locked upstream configuration

Use the already selected H128 synthetic meta-learning configuration:

```text
hidden size = 128
recurrent initialization = I3_orthogonal
inner alpha = 0.01
inner steps = 10
meta-batch = 16
```

The upstream structured P2 meta model already passed the synthetic prior battery.

Do not retune hidden size, inner alpha, inner step count, meta-batch, P2 generator, or synthetic gate thresholds using the human task.

# 2. Stage A — exact H8 baseline replay audit

## Goal

Before interpreting H128 target transfer, confirm that the current target-task pipeline reproduces the original H8 familiar phenotype.

The H8 model should reproduce:

```text
sequence learning
A/B hidden differentiation
context-yoked RT
Cross/Split preference
negative confidence→RT relationship
```

If it does not, the transfer pipeline is not comparable to the prior model.

## Exact replay requirement

Recreate the original H8 run from the prior validated implementation using the exact locked:

```text
model class
hidden size = 8
initialization regime
initialization scale
random seeds
target-task sequence
A/B block schedule
human-derived geometry
input encoding
optimizer
learning rate
RT/execution readout
RT objective
checkpoint definitions
hidden-state extraction rule
state-space fitting code
canonicalization logic
```

Do not rewrite these from memory. Prefer importing/reusing the original implementation/config.

## Replay comparison

For every seed, compare old vs replayed outputs:

```text
training NLL trajectory
training RT trajectory
checkpoint A/B hidden distance
frozen A/B transition profiles
state-space model scores
state-space winner
confidence→RT correlation
```

Primary checkpoint: 170 sequences.

Also compare 20, 50, and 90 if old values exist.

## Replay tolerance

For deterministic outputs:

```text
absolute numerical tolerance <= 1e-6
```

For nondeterministic backend behavior:

```text
document source of nondeterminism
require qualitative identity of gate status for >= 19/20 seeds
```

The replay must recover the old familiar gate status.

## Audit order if replay fails

1. Initialization: weight tensors, weight std, spectral radius, seed mapping.
2. Geometry/input encoding: A coordinates, B coordinates, cue vectors, normalization/scaling, token encoding.
3. Training schedule: A/B order, 10-sequence block boundaries, optimizer steps, hidden-state resets, sequence resets.
4. RT mechanism: execution readout initialization, stop-gradient location, vigor calculation, normalized RT.
5. Hidden-state metric: transition state extraction timing, averaging level, distance metric, canonicalization.
6. State-space pipeline: input rows, exclusions, likelihood, parameter bounds, model initialization, winner criterion.

## Stage A gate

PASS only if:

```text
F1 sequence learning = pass
F2 A/B differentiation = pass
F3 context-yoked execution = pass
F4 Cross/Split phenotype = pass at group level
F5 confidence→RT < 0 = pass
```

If Stage A fails: STOP ENTIRE PRD and produce an audit report.

# 3. Stage B — exact meta-test adaptation diagnostic

## Scientific question

The H128 meta-initialization was trained to adapt using:

```text
10 inner SGD steps
alpha = 0.01
```

but the previous human transfer used:

```text
Adam
lr = 0.001
170 target sequences
```

Stage B asks:

> Does the synthetic P2 prior transfer to the human target ecology when the model is adapted using the exact operator it was meta-trained to use?

# 4. Stage B groups

Primary groups:

```text
B1 — H128 P2_meta
B2 — H128 P1_shuffled_meta
```

Optional descriptive controls after the primary analysis:

```text
H128 P2_ordinary
H128 random-init
```

Do not include them in the Stage B primary gate.

# 5. Stage B support construction

## Support contexts

Use only:

```text
A
B
```

No Mixed or Novel examples are allowed in support.

## Support exposure

Primary support set:

```text
A: 5 full sequences
B: 5 full sequences
```

Total: 10 sequences.

Use balanced interleaving:

```text
A B A B A B A B A B
```

Use exactly the same support schedule for B1 and B2.

Do not tune support amount on probe behavior.

## Inner adaptation

Starting from the locked meta checkpoint:

```text
optimizer = plain SGD
alpha = 0.01
inner steps = 10
```

Exactly reproduce the synthetic meta-training inner operator.

If synthetic meta-training used all support examples per inner step, do the same here. If it used minibatches, reproduce those semantics exactly.

# 6. Stage B RT handling

The meta-prior was learned in the prediction model, not the RT head.

Therefore:

```text
do not train RT during the 10-step meta-test adaptation
```

Primary Stage B conclusions use:

```text
prediction NLL
prediction accuracy
correct-token confidence
hidden states
```

For descriptive RT only:

1. attach the locked monotonic confidence→RT mapping after adaptation;
2. do not fit it on Mixed/Novel;
3. if calibration is needed, fit using A/B only under a prespecified procedure.

RT cannot affect Stage B gating.

# 7. Stage B frozen probes

Immediately after the 10th inner update:

```text
freeze weights
reset hidden state
```

Probe:

```text
A
B
Mixed
Novel
```

Use first exposure only for primary results.

Collect per transition:

```text
next-token probability
NLL
accuracy
correct-token confidence
hidden vector
optional derived RT
```

No parameter updates during probes.

# 8. Stage B familiar A/B requirement

Before interpreting Mixed, require that the 10-step adaptation creates measurable A/B differentiation.

Primary criteria:

```text
B-F1 A/B prediction performance above chance
B-F2 A/B hidden-state distance > pre-adaptation distance
B-F3 A/B hidden-state distance > permutation null
B-F4 A/B transition profiles non-identical
```

If native meta-test adaptation cannot create A/B differentiation, the prior cannot solve the target task in the required way.

# 9. Stage B Mixed metrics

Measure:

```text
fraction of Mixed prediction/confidence values between A and B
Mixed NLL - mean(A/B NLL)
Mixed NLL - worse familiar endpoint
hidden A/B projection coordinate
hidden off-axis residual
inside-segment indicator
```

Do not require exact midpoint.

Primary causal contrast:

```text
P2_meta vs P1_shuffled_meta
```

A positive Stage B result should jointly show:

```text
lower Mixed prediction deficit
higher between-endpoint fraction
lower hidden off-corridor distance
```

# 10. Stage B Novel metrics

Novel is not required to interpolate.

Report:

```text
Novel NLL deficit
Novel accuracy
Novel distance to A/B corridor
Novel distance to shared sequence subspace
```

Treat Novel as secondary.

# 11. Stage B decision rule

## Outcome B-A — structured meta does not beat shuffled meta

If B1 fails to outperform B2 on primary Mixed transfer diagnostics:

```text
STOP P2
```

Interpretation:

> The P2 prior is learnable on synthetic families but does not transfer to the human target ecology, even when used with its native adaptation operator.

Do not increase hidden size, test 256/512, retune alpha, retune support count, or redesign Mixed.

## Outcome B-B — structured meta beats shuffled meta

Interpretation:

> The P2 structural prior transfers when used with the adaptation operator it was trained for.

Then the previous long-horizon target-training failure is likely due to prior overwriting, adaptation-operator mismatch, or both.

Proceed to Stage C.

# 12. Stage B seeds and statistics

Engineering pilot:

```text
3 paired seeds
```

Confirmatory:

```text
20 paired seeds
```

Primary paired contrast:

```text
B1 vs B2
```

Report:

```text
mean paired difference
bootstrap 95% CI
standardized effect
paired seed-direction count
```

Call Stage B positive only if at least 16/20 seeds show the expected direction on the primary Mixed deficit metric.

# 13. Stage C — boundary-aware episodic meta-learning
## Run only if Stage B outcome = B-B

## Motivation

P2 teaches:

```text
similar context -> generalize
```

but the human phenomenon requires both:

```text
strong contextual evidence -> split
ambiguous contextual evidence -> reuse/recombine
```

Stage C therefore trains a prior that must learn both within-family generalization and across-boundary differentiation.

# 14. Stage C ecology

Each synthetic family contains a latent contextual axis:

```text
z in [-1,1]
```

and one latent boundary:

```text
b
```

Within a side of the boundary:

```text
context distance predicts task similarity
```

Across the boundary:

```text
task mapping changes discontinuously
```

Visible context may remain relatively smooth across the boundary.

Example:

```text
-0.9 -0.6 -0.3 | +0.2 +0.5 +0.8
                 ^
               boundary
```

# 15. Stage C episode mixture

Use a fixed mixture:

```text
40% smooth interpolation episodes
40% boundary discrimination episodes
20% ambiguous boundary episodes
```

## Smooth interpolation

Support and query are from the same latent regime. Rewards generalization.

## Boundary discrimination

Support spans both regimes; query is near one side. Correct behavior follows that side rather than global averaging.

## Ambiguous boundary

Support establishes both regimes; query lies near the latent boundary. The correct target comes from the ecology's generative task law.

Do not impose:

```text
50/50 midpoint
A/B average
corridor target
```

# 16. Stage C matched shuffled control

Create:

```text
P_boundary_structured_meta
P_boundary_shuffled_meta
```

The shuffled control preserves:

```text
support count
query count
cue marginals
layout marginals
task marginals
boundary frequency
optimizer updates
```

but destroys the consistent relationship between context evidence, latent regime, and task mapping.

# 17. Stage C architecture and optimizer

Use:

```text
H = 128
Elman SRN
I3 orthogonal initialization
inner SGD alpha = 0.01
inner steps = 10
meta-batch = 16
outer optimizer = Adam
```

Do not search width.

Recalibrate only if the new ecology creates diagnosed support-learning instability.

# 18. Stage C synthetic validation

Before human transfer require:

```text
V1 within-regime transfer
V2 cross-boundary selectivity
V3 ambiguous-query transfer
V4 copy retention
V5 structured > shuffled
```

Boundary score:

```text
S_boundary = Transfer_within - Transfer_cross
```

Stage C PASS requires:

```text
copy accuracy >= 0.50
copy NLL improves
within-regime generalization passes
boundary selectivity > 0
structured > shuffled
expected direction in >= 4/5 pilot seeds
```

Then run 20-seed confirmatory before human transfer.

# 19. Stage C human transfer

If the boundary-aware synthetic gate passes, run:

```text
H128 boundary-aware structured meta
H128 boundary-aware shuffled meta
H128 P2_meta
H128 random-init
H8 original
```

Target training:

```text
same A10/B10 human-style schedule
same 170 sequences
same human geometry
same monotonic RT mechanism
```

Primary requirement:

```text
strong A/B split
+
better first-exposure Mixed transfer
```

The boundary-aware model must outperform its shuffled control without collapsing A/B differentiation.

# 20. Prior-overwriting diagnostic

If Stage B succeeds, quantify whether long target training destroys the transferable prior.

Take disposable copies of H128 P2_meta and evaluate after:

```text
native 10-step adaptation
20 target sequences
50 target sequences
90 target sequences
170 target sequences
```

At each checkpoint rerun:

```text
synthetic P2 validation battery
human A/B differentiation
frozen Mixed probe
```

Plot:

```text
synthetic prior score
A/B separation
Mixed transfer quality
```

over target training.

This tests the specialization-versus-prior-retention tradeoff directly.

# 21. No 256/512 rule

Reject hidden sizes >128 for this PRD.

H128 has already demonstrated sufficient capacity to learn the synthetic prior.

The unresolved questions are:

```text
transfer compatibility
adaptation-operator compatibility
specialization/generalization balance
```

not raw capacity.

# 22. TDD / RED -> GREEN requirements

## T1 — exact H8 replay

Load prior baseline config and reproduce seed-level outputs.

## T2 — native adaptation operator

Stage B must assert:

```text
optimizer == SGD
alpha == 0.01
inner_steps == 10
```

for B1/B2.

## T3 — no probe leakage

Mixed/Novel rows must never enter adaptation batches.

## T4 — frozen probe hashes

Parameter hashes identical before and after every probe.

## T5 — paired checkpoint identity

B1/B2 checkpoint manifests match locked upstream checkpoint hashes.

## T6 — no RT gradient in Stage B

RT cannot contribute to inner updates or gating.

## T7 — support/query separation

Query examples never enter adaptation loss.

## T8 — no target-based tuning

Hyperparameter selection code cannot access:

```text
Mixed metrics
Novel metrics
human profile correlation
corridor scores
```

## T9 — Stage gate enforcement

Stage C code refuses to run unless Stage B = PASS.

## T10 — whole-profile canonicalization

Any A/B relabeling swaps entire profiles, never individual transitions.

## T11 — target-training equality

Stage C human groups share identical sequence, geometry, schedule, optimizer, learning rate, execution objective, and probe checkpoints.

## T12 — no capacity escalation

Config validator rejects H > 128.

# 23. Required outputs

## Stage A

```text
h8_replay_config.json
h8_replay_seed_comparison.csv
h8_replay_training_comparison.csv
h8_replay_hidden_comparison.csv
h8_replay_state_space_comparison.csv
h8_replay_gate.csv
H8_REPLAY_AUDIT_REPORT.md
```

## Stage B

```text
native_meta_test_config.json
native_meta_test_support_manifest.csv
native_meta_test_probe_raw.csv
native_meta_test_familiar_metrics.csv
native_meta_test_mixed_metrics.csv
native_meta_test_novel_metrics.csv
native_meta_test_hidden_geometry.csv
native_meta_test_gate.csv
NATIVE_META_TEST_TRANSFER_REPORT.md
```

## Stage C, only if reached

```text
boundary_meta_config.json
boundary_episode_manifest.csv
boundary_meta_training.csv
boundary_synthetic_validation.csv
boundary_synthetic_gate.csv
boundary_human_training.csv
boundary_human_familiar_gates.csv
boundary_human_probe_metrics.csv
boundary_human_hidden_geometry.csv
BOUNDARY_META_TRANSFER_REPORT.md
```

# 24. Required figures

## Stage A

```text
A1 old vs replay H8 learning curves
A2 old vs replay A/B hidden separation
A3 state-space winner replay
```

## Stage B

```text
B1 native meta-test schematic
B2 A/B differentiation after native adaptation
B3 Mixed prediction profile
B4 Mixed hidden corridor geometry
B5 structured-meta vs shuffled-meta paired differences
B6 Novel diagnostics
```

## Stage C

```text
C1 boundary-aware ecology schematic
C2 within vs cross-boundary transfer
C3 ambiguous-query behavior
C4 synthetic structured vs shuffled
C5 human A/B phenotype
C6 first Mixed transfer
C7 differentiation-transfer frontier
C8 prior retention vs specialization
```

# 25. Final decision tree

```text
START
 |
 |-- Stage A: H8 baseline replay?
 |       |
 |       |-- FAIL -> STOP, fix pipeline
 |       |
 |       `-- PASS
 |            |
 |            |-- Stage B: native H128 meta-test
 |                    |
 |                    |-- P2_meta <= shuffled_meta
 |                    |       -> STOP P2
 |                    |       -> synthetic prior does not transfer
 |                    |
 |                    `-- P2_meta > shuffled_meta
 |                            -> prior transfers under native adaptation
 |                            -> diagnose overwriting
 |                            -> Stage C boundary-aware meta-learning
 |
 END
```

# 26. Scientific interpretation

The program distinguishes three hypotheses:

1. **Pipeline mismatch** — if H8 replay fails, the previous transfer run is not comparable to the validated baseline.
2. **Adaptation-operator mismatch / prior overwriting** — if native H128 meta-test succeeds but long Adam target training fails, the prior is real and transferable but not preserved under the previous adaptation regime.
3. **Wrong prior** — if native H128 meta-test fails, P2 learned a useful synthetic prior that does not correspond to the human target ecology.

Only if hypothesis 2 is supported should we proceed to a boundary-aware prior that explicitly learns when to generalize versus split.
