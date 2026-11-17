# PRD: SRN Meta-Learning Calibration Before Structural-Prior Testing
## Width × Initialization × Inner-Loop Stability

## 0. Purpose

The previous episodic meta-learning pilot should **not** be interpreted as a clean failure of structural meta-learning yet.

The main diagnostic pattern was:

- larger SRNs produced dramatically worse unseen-query NLL,
- copy accuracy approached chance for H32–H128,
- meta-training increasingly underperformed ordinary training as width increased,
- the intended similarity prior did not emerge.

Before testing structural generalization again, we need to establish a simpler prerequisite:

\[
\boxed{
\text{Can each SRN width stably adapt to a small support set at all?}
}
\]

This PRD calibrates the **inner-loop learning dynamics** independently of the structural-generalization hypothesis.

The experiment must not use:

```text
human A/B
Mixed
Novel
human RT
context interpolation scores
corridor metrics
q
```

The only goal is to find a stable adaptation regime for each hidden size.

---

# 1. Central scientific question

For each SRN width:

\[
H \in \{8,32,64,128\},
\]

identify whether there exists a combination of:

```text
recurrent initialization regime
inner-loop learning rate
number of inner-loop steps
```

such that support-set adaptation produces:

1. lower support NLL,
2. lower familiar/copy-query NLL,
3. higher copy accuracy,
4. no catastrophic confidence explosion,
5. stable recurrent dynamics.

This calibration must be completed **before** rerunning any structured-vs-shuffled meta-learning comparison.

---

# 2. Fixed architecture

Use the same tanh Elman SRN family:

\[
h_t=\tanh(W_xx_t+W_hh_{t-1}+b_h)
\]

\[
p_t=\operatorname{softmax}(W_oh_t+b_o)
\]

Hidden sizes:

```text
8
32
64
128
```

No:

```text
LSTM
GRU
attention
Transformer
explicit latent-state module
RT head
execution readout
```

This is an optimization calibration, not an architecture comparison.

---

# 3. Synthetic task generator

Use the exact same synthetic family generator used in the prior episodic meta-learning experiment.

Do not redesign the ecology.

Each family contains:

```text
one 8-item sequence family
one response alphabet of size 4
6 contextual members
context-dependent task realizations
visible cue/layout information
```

The calibration should use **structured P2 families only**, because the objective is to test whether the network can adapt to support examples.

The structured-vs-shuffled contrast is irrelevant at this stage.

---

# 4. No outer-loop meta-learning

This is critical.

For this calibration:

```text
DO NOT run FOMAML outer updates.
DO NOT accumulate meta-gradients.
DO NOT update a persistent meta-initialization.
```

Each evaluation episode starts from a fresh model initialization.

Within an episode:

```text
initialize model
measure pre-adaptation performance
adapt only on support set
measure post-adaptation performance
discard model
```

This isolates the inner-loop optimizer.

---

# 5. Experimental factors

## 5.1 Hidden size

```text
H = 8
H = 32
H = 64
H = 128
```

## 5.2 Inner-loop learning rate

Sweep:

\[
\alpha \in
\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}.
\]

Exact values:

```text
0.0001
0.0003
0.001
0.003
0.01
```

## 5.3 Number of inner-loop steps

Sweep:

```text
0
1
3
5
10
```

`0` is the no-adaptation baseline.

## 5.4 Recurrent initialization regime

Test four initialization conditions.

### I0 — Current High-scale baseline

Preserve current implementation:

```text
scale = 2.0
```

### I1 — Moderate scale

```text
scale = 1.0
```

### I2 — Low/moderate scale

```text
scale = 0.5
```

### I3 — Orthogonal recurrent control

Use:

```text
orthogonal W_h
Xavier/Glorot input weights
Xavier/Glorot output weights
```

Then rescale recurrent matrix to target spectral radius:

```text
rho = 0.9
```

This condition tests whether unstable recurrent dynamics, rather than model width itself, cause adaptation failure.

---

# 6. Full pilot grid

The full calibration grid is:

\[
4\text{ widths}
\times
4\text{ initialization regimes}
\times
5\text{ learning rates}
\times
5\text{ step counts}.
\]

Total:

```text
400 cells
```

For each nonzero-step cell:

```text
20 random synthetic episodes
```

for the first pass.

The 0-step condition only needs to be computed once per:

```text
width × initialization × episode
```

and reused for all alpha values.

---

# 7. Episode structure

For each calibration episode:

## 7.1 Sample one new family

Generate:

```text
6 contextual members
```

from the same P2 generator.

## 7.2 Select support contexts

Randomly choose:

```text
3 support contexts
```

with coverage rule:

```text
one lower-third context
one upper-third context
one additional context
```

## 7.3 Support examples

Per support context:

```text
2 full 8-item sequences
```

Total:

```text
6 support sequences
```

## 7.4 Copy/familiar queries

Sample:

```text
1 fresh sequence from each support context
```

Total:

```text
3 copy queries
```

These do not contribute gradients.

## 7.5 Unseen queries

Also sample:

```text
3 unseen-context queries
```

for diagnostic purposes only.

These are not used to select the stable inner-loop regime.

---

# 8. Pre-adaptation measurements

Before any gradient update, record:

```text
support NLL
support accuracy
copy NLL
copy accuracy
unseen-query NLL
unseen-query accuracy
mean correct-token confidence
mean max-softmax confidence
hidden-state mean norm
hidden-state max norm
fraction |h| > 0.95
recurrent spectral radius
gradient norm on support loss
```

---

# 9. Inner-loop adaptation

For each:

```text
alpha
step_count
```

start from the exact same initial weights for that episode.

Perform SGD-style inner adaptation:

\[
\theta^{(k+1)}
=
\theta^{(k)}
-
\alpha
\nabla_\theta L_{\text{support}}.
\]

Primary optimizer:

```text
plain SGD
no momentum
```

Do not use Adam in the inner loop.

Use gradient clipping only in a separate diagnostic condition if needed.

---

# 10. Post-adaptation measurements

After:

```text
1
3
5
10
```

steps, measure the same outputs as pre-adaptation.

Primary adaptation deltas:

\[
\Delta NLL_{support}
=
NLL_{post}
-
NLL_{pre}
\]

\[
\Delta NLL_{copy}
=
NLL_{post}
-
NLL_{pre}
\]

\[
\Delta Acc_{copy}
=
Acc_{post}
-
Acc_{pre}.
\]

Successful adaptation requires negative NLL deltas and positive copy-accuracy delta.

---

# 11. Stability diagnostics

For each step, record:

```text
gradient norm
parameter update norm
update norm / parameter norm
recurrent spectral radius
hidden-state saturation
max softmax confidence
fraction incorrect with confidence > .9
fraction non-finite values
```

Define:

\[
R_{\text{update}}
=
\frac{\|\Delta\theta\|_2}
{\|\theta\|_2}.
\]

Log separately for:

```text
W_x
W_h
W_o
biases
```

---

# 12. Catastrophic adaptation flags

Mark a cell unstable if any of the following occurs in >5% of episodes:

```text
NaN / Inf
query NLL > 20
support NLL increases by > 1.0
copy NLL increases by > 1.0
incorrect max-softmax confidence > .95 for > 50% of query tokens
mean hidden saturation fraction > .75
recurrent spectral radius > 5 after adaptation
```

Do not select unstable cells even if their mean copy accuracy appears high.

---

# 13. Primary calibration criterion

For each:

```text
width × initialization × alpha × step_count
```

compute the mean across episodes.

A cell is **adaptation-valid** if all are true:

## C1 — support learning

\[
\Delta NLL_{support}<0.
\]

## C2 — familiar transfer

\[
\Delta NLL_{copy}<0.
\]

## C3 — copy accuracy gain

\[
\Delta Acc_{copy}>0.
\]

## C4 — meaningful familiar retention

Post-adaptation:

```text
copy accuracy >= 0.50
```

for the pilot mean.

Chance is 0.25.

## C5 — stable optimization

No catastrophic adaptation flag.

## C6 — episode consistency

At least:

```text
80% of episodes
```

show lower copy NLL after adaptation.

---

# 14. Width-specific hyperparameter selection

For each hidden size separately:

1. remove all unstable cells;
2. remove all cells failing C1-C6;
3. among remaining cells, select the cell with lowest:
   ```text
   post-adaptation copy NLL
   ```
4. use support NLL only as a tiebreaker;
5. prefer fewer inner steps if performance is within:
   ```text
   1% relative NLL
   ```
   of the best cell;
6. prefer the simpler/current initialization if within:
   ```text
   1% relative NLL
   ```
   of the best.

This yields:

\[
(\text{init},\alpha,n_{\text{steps}})_H.
\]

Do not select based on unseen-query generalization.

---

# 15. Capacity failure criterion

If a hidden width has **no adaptation-valid cell**, then that capacity is not eligible for meta-learning under the current architecture.

Interpretation:

```text
the inner-loop optimization problem itself is unstable or ineffective at this width
```

before any structural inference claim.

---

# 16. Confirmatory inner-loop validation

For each width with a selected stable regime:

Run:

```text
100 new families
5 random model seeds
```

using the selected:

```text
initialization
alpha
step count
```

Primary confirmatory requirements:

```text
mean copy accuracy >= .50
copy NLL improves in >= 4/5 seeds
support NLL improves in >= 4/5 seeds
no catastrophic instability
```

---

# 17. Optional gradient-clipping rescue

Only if a width fails under all primary conditions.

Run one rescue sweep with:

```text
global gradient norm clipping
clip ∈ {1, 5}
```

using the two best non-clipped learning rates from the failed width.

Do not expand beyond this.

If clipping is required for a width, mark that width as:

```text
stable only with clipping
```

in all later analyses.

---

# 18. Meta-batch calibration

After stable inner-loop settings are identified, test outer-gradient noise **before** rerunning the full structural experiment.

For:

```text
H8
smallest larger stable width
largest stable width
```

run short meta-training jobs with:

```text
meta_batch ∈ {1, 8, 16, 25}
```

using structured P2 only.

Each outer update averages query gradients across independently sampled families.

Run:

```text
500 outer updates
3 seeds
```

Measure:

```text
outer-gradient norm variance
training query NLL
validation copy NLL
validation unseen-query NLL
```

Primary meta-batch selection rule:

```text
choose the smallest batch whose outer-gradient coefficient of variation
is <= 110% of the best batch
```

Expected candidate:

```text
25
```

but do not hard-code 25 as the winner.

---

# 19. Revised meta-learning experiment after calibration

Only after Sections 1-18 pass.

Rerun:

```text
P2_meta
P1_shuffled_meta
P2_ordinary
```

across all adaptation-valid widths.

For each width use its calibrated:

```text
initialization
inner alpha
inner step count
```

and the calibrated common meta-batch size.

Outer optimizer:

```text
Adam
```

Outer LR:

```text
1e-3 initial
```

with gradient clipping if established in calibration.

---

# 20. Revised meta-training pilot

Use:

```text
5 paired seeds
5000 meta-training episodes per condition
```

where one outer update processes:

```text
meta_batch families
```

Count total families, not outer steps, when matching exposure.

For ordinary training:

```text
match total synthetic sequences / gradient-evaluation budget as closely as possible
```

Document any unavoidable mismatch.

---

# 21. Revised copy gate

Replace the previous permissive familiar-retention gate.

A meta-trained cell now requires:

## G_copy_1

\[
Acc_{copy,post}\ge0.50.
\]

## G_copy_2

\[
NLL_{copy,post}<NLL_{copy,pre}.
\]

## G_copy_3

At least:

```text
4/5 seeds
```

show copy NLL improvement.

If these fail, do not interpret structural generalization.

---

# 22. Revised structural-prior gates

Only evaluate these after the copy gate passes.

## G1 — meta beats ordinary

\[
NLL_{P2-meta}
<
NLL_{P2-ordinary}
\]

on unseen-family queries.

## G2 — structure beats shuffled

\[
NLL_{P2-meta}
<
NLL_{P1-shuffled-meta}.
\]

## G3 — seed consistency

At least:

```text
4/5 paired seeds
```

for both G1 and G2.

## G4 — similarity prior

Prediction deficit should increase with context distance.

## G5 — asymmetric-query generalization

Performance must not depend only on midpoint-like queries.

Only models passing all of:

```text
copy gate
G1-G5
```

may advance.

---

# 23. Human firewall

No human transfer should run during this PRD unless the revised structural-prior gate passes.

Explicitly prohibit:

```text
A/B human training
Mixed probes
Novel probes
RT fitting
human corridor analysis
human q analysis
```

during hyperparameter calibration.

---

# 24. Required unit tests

Use RED -> GREEN -> REFACTOR.

## T1 — exact reset per calibration cell

Each alpha/step condition must start from identical weights within an episode.

## T2 — 0-step baseline reused correctly

No redundant gradient update.

## T3 — query examples never enter support loss

Assert disjoint indices.

## T4 — unseen-query metrics cannot influence calibration selection

Selection function accepts only:

```text
support metrics
copy metrics
stability metrics
```

## T5 — width-specific initialization logged

Verify actual:

```text
weight std
spectral radius
parameter norm
```

rather than requested nominal scale only.

## T6 — stepwise measurements

Store metrics after every inner step, not just the final step.

## T7 — no outer optimizer

Calibration mode must hard-fail if an outer optimizer step occurs.

## T8 — no human identifiers

Calibration code should fail if it encounters:

```text
Mixed
Novel
Maze
Melody
participant_id
human_rt
```

## T9 — stability flags

Synthetic tests should deliberately induce an oversized step and verify instability detection.

## T10 — deterministic pairing

Same episode seed + same initialization seed must reproduce identical initial weights and support/query samples.

---

# 25. Required outputs

## Calibration grid

```text
inner_loop_calibration_config.json
inner_loop_calibration_manifest.csv
inner_loop_step_metrics.csv
inner_loop_cell_summary.csv
inner_loop_stability_flags.csv
inner_loop_selected_hyperparameters.csv
```

## Confirmatory validation

```text
inner_loop_confirmatory_metrics.csv
inner_loop_confirmatory_summary.csv
```

## Meta-batch calibration

```text
meta_batch_calibration.csv
meta_batch_gradient_variance.csv
meta_batch_selected.json
```

## Revised meta-learning, only if reached

```text
revised_meta_training_curves.csv
revised_meta_synthetic_validation.csv
revised_meta_gate_results.csv
```

## Main report

```text
SRN_META_LEARNING_CALIBRATION_REPORT.md
```

---

# 26. Required figures

## Figure 1 — Inner-loop learning curves

For each hidden size:

```text
support NLL vs inner step
copy NLL vs inner step
```

faceted by alpha.

## Figure 2 — Width × alpha heatmap

Plot:

```text
post-adaptation copy NLL
```

for each initialization regime.

## Figure 3 — Copy accuracy heatmap

Same grid, but:

```text
post-adaptation copy accuracy
```

## Figure 4 — Stability map

Binary or categorical:

```text
stable
support improves only
copy improves
catastrophic
```

## Figure 5 — Effective update magnitude

Plot:

\[
\|\Delta\theta\|/\|\theta\|
\]

against hidden width.

## Figure 6 — Hidden saturation

Plot:

```text
fraction |h| > .95
```

by width, initialization, alpha.

## Figure 7 — Spectral-radius trajectory

Before and after inner adaptation.

## Figure 8 — Selected stable regime per width

Compact summary:

```text
H
init
alpha
steps
copy accuracy
copy NLL delta
```

## Figure 9 — Meta-batch gradient variance

Outer-gradient CV versus meta-batch size.

## Figure 10 — Revised meta-learning result

Only if reached:

```text
P2_meta
P1_shuffled_meta
P2_ordinary
```

by width.

---

# 27. Decision logic

## Outcome A — all widths find stable adaptation regimes

Interpretation:

> previous failure was substantially confounded by inner-loop optimization.

Proceed to revised meta-learning experiment.

## Outcome B — H8/H32 adapt, H64/H128 cannot

Interpretation:

> increasing capacity destabilizes weight-based inner-loop adaptation under the Elman architecture.

Proceed with stable widths only.

## Outcome C — orthogonal/low-scale initialization rescues large widths

Interpretation:

> recurrent initialization, not capacity itself, caused the previous degradation.

Use width-specific calibrated initialization for the structural test.

## Outcome D — no width can achieve robust copy learning

Interpretation:

> FOMAML-style weight adaptation is poorly matched to this SRN/task setup.

Stop FOMAML.

Next architecture/objective should use **activation-based episodic inference** rather than within-episode gradient updates.

## Outcome E — copy learning succeeds, but revised structured-meta still does not beat shuffled-meta

Interpretation:

> the optimization confound is resolved, and the structural-prior hypothesis genuinely fails under FOMAML SRNs.

This is the first clean point at which we should move beyond the current model.

---

# 28. Central stopping rule

Do not run another ecology sweep until we can answer:

\[
\boxed{
\text{Can the network reliably learn the support set within an episode?}
}
\]

If the answer is no, structural-generalization results are uninterpretable.

If the answer is yes and structured-meta still fails, then the failure is scientifically meaningful.
