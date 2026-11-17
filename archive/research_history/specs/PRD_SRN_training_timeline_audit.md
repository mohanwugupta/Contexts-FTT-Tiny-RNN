# PRD: Audit and Reconstruct the True SRN Training / Probe Loss Timeline

## 1. Purpose

The current raw-loss time-series plots may be misleading because they appear to evaluate every historical A/B sequence using a single already-trained checkpoint rather than the parameter state that existed when that sequence was actually encountered.

This PRD has two goals:

1. **Audit model provenance** to verify that all Phase 1D / 1E analyses are using the canonical SRN:
   - Elman/tanh SRN
   - `hidden_size = 8`
   - High initialization scale = `2.0`
   - Low initialization scale = `0.02`
   - same 170-sequence A/B training schedule used in the current project

2. **Reconstruct the true acquisition timeline** so that raw sequence loss is plotted using the model parameters that actually existed at each point in training, while Mixed / Novel / NoContext probes are evaluated as frozen probes at matched training checkpoints.

The final output must clearly distinguish:

- original SRN training over sequences,
- retrospective evaluation of a fixed trained checkpoint,
- and Phase 1E retrofit optimization over optimizer updates.

Do not change the scientific model or retrofit objective until this audit is complete.

---

# 2. Core hypothesis to test

The likely source of the strange plots is:

> The code loads one final trained checkpoint and repeatedly evaluates the historical A/B sequence schedule without updating the network between sequence indices.

If so, repeated A sequences should have nearly identical loss and repeated B sequences should have nearly identical loss, producing the current step-function appearance.

This is not an acquisition trajectory.

The correct acquisition trajectory should instead be:

\[
\theta_0
\rightarrow
\theta_1
\rightarrow
\dots
\rightarrow
\theta_{170}
\]

with loss at sequence \(s\) computed from the parameter state actually present at sequence \(s\).

---

# 3. Canonical model specification

Before running any scientific analysis, hard-verify the loaded model.

## Required assertions

For the High-scale condition:

```python
assert model.hidden_size == 8
assert init_scale == 2.0
```

For the Low-scale condition:

```python
assert model.hidden_size == 8
assert init_scale == 0.02
```

Also record:

```text
model_class
hidden_size
input_dim
output_dim
activation
initialization_scale
seed
optimizer_class
learning_rate
source_checkpoint_path
source_checkpoint_hash
source_config_path
source_config_hash
training_schedule_hash
git_commit
```

If any High-scale run resolves to the old 6-unit / 0.5 setup, stop immediately and mark the run invalid.

Do not silently convert or reinterpret old checkpoints.

---

# 4. Questions this audit must answer

## Q1. Are the current "raw loss over sequences" plots actually frozen-checkpoint evaluations?

Test explicitly.

For each seed:

1. Load the final trained checkpoint.
2. Evaluate the full historical 170-sequence A/B schedule with **no parameter updates**.
3. Save the raw loss at every sequence.
4. Compare this diagnostic curve to the currently plotted curve.

If they match to numerical tolerance, we have identified the issue.

Expected signature:

\[
L(A_1) \approx L(A_2) \approx \dots
\]

and

\[
L(B_1) \approx L(B_2) \approx \dots
\]

because the same trained network is being shown the same repeated context/sequence combination.

---

## Q2. Are Phase 1D / 1E operating on the correct 8-unit High-scale checkpoints?

Audit every seed used in the current results.

Produce one row per seed:

```text
seed
hidden_size
init_scale
checkpoint_hash
final_training_loss
source_run
valid_reference_checkpoint
```

The analysis must not proceed until all reference checkpoints are validated.

---

## Q3. Can the exact original training trajectory be reconstructed?

Preferred order:

### Option A — replay exactly from saved initialization / RNG state

If the repository stores enough provenance:

- initialization seed,
- exact training schedule,
- optimizer configuration,
- learning rate,
- update ordering,
- RNG state if needed,

then rerun the original training deterministically.

At every external training sequence \(s\), record the raw sequence loss using the model state at that moment.

### Option B — use saved intermediate checkpoints

If sequence-level or block-level checkpoints already exist, reconstruct the trajectory from those.

### Option C — deterministic rerun from canonical configuration

If no exact intermediate states were saved but the training procedure is deterministic from seed, rerun all canonical seeds.

### Option D — if exact replay is impossible

Do not fake the trajectory.

Instead:

1. document which missing provenance prevents exact reconstruction;
2. rerun a fresh canonical seed set using the frozen canonical specification;
3. explicitly label those results as a regenerated reference cohort.

---

# 5. Define the three different time axes

These must never be mixed in one ambiguous figure.

## Timeline A — Original acquisition

x-axis:

```text
external training sequence index, 1...170
```

At sequence \(s\):

1. model has state \(\theta_s\),
2. compute the raw loss on the currently presented A or B sequence,
3. record it,
4. perform the normal training update,
5. advance to \(\theta_{s+1}\).

Primary metric:

```text
raw sequence loss before update
```

Also save post-update loss as a diagnostic if useful, but do not substitute it for the primary curve.

---

## Timeline B — Frozen retrospective evaluation

Optional diagnostic only.

x-axis:

```text
historical schedule position
```

but the network is fixed at:

\[
\theta_{170}.
\]

This plot may be flat within A and B contexts.

It must be titled something like:

> Frozen final-checkpoint evaluation across the historical A/B schedule

Never call this an acquisition curve.

---

## Timeline C — Phase 1E retrofit optimization

x-axis:

```text
accepted or attempted Phase 1E optimizer update
```

At retrofit step \(j\):

\[
\theta^{1E}_j.
\]

Evaluate A, B, Mixed, Novel, and NoContext with the current retrofit state, but do not relabel this as "sequence index."

---

# 6. Probe evaluation during original training

This is a required new analysis.

The goal is to see how Mixed and Novel behavior develops relative to A and B **while the network is actually learning**.

## Registered training checkpoints

Use a fixed set such as:

```python
probe_checkpoints = [
    0, 10, 20, 30, 40, 50, 60, 70, 80,
    90, 100, 110, 120, 130, 140, 150, 160, 170
]
```

Use the project's existing checkpoint convention if one already exists.

At each registered checkpoint \(c\):

1. freeze parameters,
2. evaluate:
   - Context A
   - Context B
   - Mixed
   - Novel
   - NoContextProbe
3. record raw sequence loss,
4. restore/verify the exact training parameter hash,
5. resume training.

Probe evaluation must never update weights, hidden-state carryover, optimizer moments, or RNG state used for training.

---

# 7. Probe firewall

For every frozen probe call:

```python
hash_before = hash_parameters(model)
optimizer_before = hash_optimizer_state(optimizer)
rng_before = capture_rng_state()

evaluate_probes(...)

assert hash_parameters(model) == hash_before
assert hash_optimizer_state(optimizer) == optimizer_before
assert rng_state_equal(capture_rng_state(), rng_before)
```

If recurrent hidden state is normally reset between external sequences, probes must follow the same reset convention.

If probe evaluation modifies persistent hidden state, explicitly save and restore it.

---

# 8. Required raw-loss definitions

Reuse the exact sequence-loss function already used in the canonical SRN training code.

Do not:

- normalize it for visualization,
- convert it to probability,
- divide by a new baseline,
- use the execution proxy,
- use the logit-path residual,
- or mix it with Phase 1E objective terms.

The main plots requested here are **raw model sequence loss**.

Record enough metadata to distinguish:

```text
raw_sequence_loss
mean_token_loss
sum_token_loss
```

If the current code's "raw sequence loss" is a sum across sequence positions, keep that definition as primary.

---

# 9. Required figures

## Figure A — True original acquisition curve

For High-scale 8-unit SRNs.

x-axis:

```text
actual external sequence index, 1...170
```

y-axis:

```text
raw sequence loss
```

Show:

- A training observations,
- B training observations,
- mean across seeds,
- uncertainty band across seeds.

A/B blocks should be visually indicated.

This is the main replacement for the misleading current time-series plot.

---

## Figure B — Original acquisition + frozen probes

Same x-axis:

```text
training sequence index
```

Overlay probe points at registered checkpoints:

- Mixed
- Novel
- NoContextProbe

A and B must remain visible as reference trajectories.

Preferred display:

- A/B as continuous/blockwise training-loss curves,
- probe conditions as distinct markers or dashed checkpoint curves.

This figure should answer:

> At a given stage of A/B learning, where do Mixed and Novel losses sit relative to the familiar A and B losses?

---

## Figure C — Probe-only checkpoint panel

x-axis:

```text
registered training checkpoint
```

y-axis:

```text
raw sequence loss
```

Five curves:

- A
- B
- Mixed
- Novel
- NoContextProbe

This is the cleanest visualization of probe development.

---

## Figure D — Frozen final-checkpoint diagnostic

Use the final trained network \(\theta_{170}\) and evaluate the repeated historical A/B schedule without learning.

This should reproduce the current suspicious step-function if the diagnosis is correct.

Title explicitly:

> Frozen final-checkpoint evaluation — not an acquisition trajectory

This figure is diagnostic/supplemental.

---

## Figure E — Phase 1E retrofit trajectory

x-axis:

```text
Phase 1E optimizer update
```

y-axis:

```text
raw sequence loss
```

Five frozen evaluation curves:

- A
- B
- Mixed
- Novel
- NoContextProbe

This makes explicit that any Mixed improvement during Phase 1E is caused by retrofit optimization, not ordinary A/B acquisition.

---

## Figure F — Before/after provenance comparison

For each seed:

```text
original training final checkpoint
Phase 1D selected checkpoint
Phase 1E starting checkpoint
Phase 1E ending checkpoint
```

Show hashes / distances between states.

This should verify that Phase 1E actually begins from the intended trained network.

---

# 10. Required tables

## Table 1 — Checkpoint provenance

Columns:

```text
seed
condition
hidden_size
init_scale
checkpoint_path
checkpoint_hash
config_hash
training_schedule_hash
valid
```

---

## Table 2 — Replay fidelity

For each seed:

```text
seed
existing_final_checkpoint_hash
replayed_final_checkpoint_hash
parameter_max_abs_diff
parameter_L2_diff
final_loss_existing
final_loss_replayed
exact_match
```

If exact replay is expected but fails, stop and diagnose before interpretation.

---

## Table 3 — Probe losses over learning

Long-format CSV:

```text
seed
training_checkpoint
condition
raw_sequence_loss
mean_token_loss
parameter_hash
```

Conditions:

```text
A
B
Mixed
Novel
NoContextProbe
```

---

# 11. TDD / RED → GREEN tests

## Test 1 — Architecture guard

### RED

Point the loader at an old 6-unit checkpoint.

Expected:

```text
FAIL: hidden_size mismatch
```

### GREEN

Canonical High run loads:

```text
hidden_size = 8
init_scale = 2.0
```

---

## Test 2 — Detect frozen-schedule evaluation

Construct:

```python
losses = evaluate_historical_schedule(
    model=final_checkpoint,
    update=False
)
```

Expected detector:

```python
assert frozen_schedule_detector(losses) == True
```

The detector can check that within-context repeated losses have near-zero variance.

Then run true training replay:

```python
losses = replay_training(...)
```

Expected:

```python
assert frozen_schedule_detector(losses) == False
```

unless a genuine edge case is documented.

---

## Test 3 — Parameter movement during acquisition

For every actual training update:

```python
hash_before != hash_after
```

or, more robustly:

```python
parameter_L2_delta > tolerance
```

for at least the expected large majority of updates.

If every sequence uses the same parameter hash, the curve is invalid.

---

## Test 4 — Probe firewall

Probe evaluation must leave unchanged:

```text
parameters
optimizer state
persistent recurrent state
training RNG state
```

Fail hard if any differ.

---

## Test 5 — Replay fidelity

If deterministic replay is expected:

```python
assert max_abs_param_diff < 1e-8
```

or exact hash equality if serialization is deterministic.

If not, report the precise mismatch.

---

## Test 6 — Sequence ordering

Verify the reconstructed schedule has exactly the canonical number and ordering of A/B training sequences.

Save the schedule hash.

---

## Test 7 — Raw-loss identity

For a fixed model + fixed input sequence:

```python
loss_from_training_function
==
loss_from_plotting_function
```

within numerical tolerance.

This prevents the plotting pipeline from quietly using a different loss.

---

# 12. Debugging branches

## Branch A — Wrong architecture found

If Phase 1D / 1E used 6 units or High scale != 2.0:

1. archive the current outputs,
2. mark them non-canonical,
3. rerun Phase 1D / 1E from the correct 8-unit High-scale checkpoints,
4. do not compare old and new results as if they are the same model family.

---

## Branch B — Correct architecture, wrong plotting logic

If checkpoint provenance is correct but the sequence plot uses a fixed final checkpoint:

1. keep Phase 1D / 1E scientific results provisionally,
2. replace the misleading plots,
3. regenerate acquisition from true training states,
4. regenerate checkpoint probe plots.

---

## Branch C — Exact original replay impossible

If old runs cannot be reconstructed exactly:

1. preserve old final checkpoints for provenance,
2. rerun a fresh canonical seed cohort,
3. add full logging,
4. use the new cohort for all time-resolved learning analyses.

Do not interpolate or invent historical losses.

---

## Branch D — True training curve is unexpectedly flat

If the model genuinely shows no acquisition despite parameter updates:

1. verify loss is measured pre-update,
2. verify the external sequence actually changes the optimizer state,
3. inspect learning rate,
4. inspect whether the network is effectively saturated from High initialization,
5. compare first vs final parameter norms and outputs,
6. verify that the "loss" being plotted is not computed after restoring the final checkpoint.

---

# 13. Logging changes required going forward

Every canonical SRN training run should save:

```text
seed
sequence_index
context
raw_sequence_loss_pre_update
raw_sequence_loss_post_update
parameter_hash_pre
parameter_hash_post
optimizer_state_hash
checkpoint_path
```

At registered probe checkpoints additionally save:

```text
probe_A_loss
probe_B_loss
probe_Mixed_loss
probe_Novel_loss
probe_NoContextProbe_loss
probe_parameter_hash_before
probe_parameter_hash_after
```

This should become standard infrastructure rather than a one-off analysis.

---

# 14. Output directory

Create a new immutable analysis directory, e.g.:

```text
analysis/srn_training_timeline_audit/
```

Suggested outputs:

```text
manifest.json
checkpoint_provenance.csv
replay_fidelity.csv
training_sequence_losses.csv
probe_checkpoint_losses.csv
phase1e_probe_losses.csv

fig_A_true_acquisition.png
fig_B_acquisition_with_probes.png
fig_C_probe_checkpoint_losses.png
fig_D_frozen_checkpoint_diagnostic.png
fig_E_phase1e_retrofit_losses.png
fig_F_checkpoint_provenance.png

AUDIT_REPORT.md
```

Do not overwrite previous Phase 1D / 1E artifacts.

---

# 15. Required AUDIT_REPORT.md structure

## 1. Canonical model verified?

State explicitly:

```text
hidden_size:
High init scale:
Low init scale:
checkpoint family:
```

## 2. Was the old time-series plot a frozen-checkpoint artifact?

Answer:

```text
YES / NO
```

with numerical evidence.

## 3. Could the original training trajectory be exactly replayed?

Answer:

```text
YES / NO
```

and report checkpoint mismatch if any.

## 4. What does the true A/B acquisition curve look like?

Brief descriptive summary only.

## 5. How do Mixed / Novel / NoContext probe losses evolve relative to A/B?

Report the raw losses at registered checkpoints.

Do not make a mechanistic claim beyond what the plots show.

## 6. Did Phase 1E use the correct starting checkpoints?

Answer:

```text
YES / NO
```

## 7. Which previous plots/results are invalidated?

List exact filenames / panels.

## 8. Which previous scientific conclusions remain intact?

Separate plotting errors from actual modeling errors.

---

# 16. Acceptance criteria

This task is complete only when all of the following are true:

- [ ] Every analyzed High-scale checkpoint is verified as 8-unit / scale 2.0.
- [ ] Every analyzed Low-scale checkpoint is verified as 8-unit / scale 0.02.
- [ ] Existing suspicious sequence-loss plot is reproduced or falsified as a frozen-checkpoint diagnostic.
- [ ] A true original training acquisition curve is generated from time-matched model states.
- [ ] Raw A/B losses are visible across the full 170-sequence training schedule.
- [ ] Mixed, Novel, and NoContextProbe losses are shown relative to A and B at registered training checkpoints.
- [ ] Probe evaluation demonstrably makes zero training updates.
- [ ] Phase 1E retrofit curves use optimizer-update index, not sequence index.
- [ ] Replay fidelity is quantified.
- [ ] No historical loss is inferred or fabricated when the exact state is unavailable.
- [ ] Previous outputs are preserved unchanged.
- [ ] `AUDIT_REPORT.md` clearly states whether any previous scientific result must be rerun.

---

# 17. Scientific interpretation rule

Do not interpret an apparent change across "sequence index" as learning unless the plotted point was computed from the parameter state that actually existed at that point in training.

Formally, a valid acquisition curve must satisfy:

\[
L_s =
L(x_s;\theta_s),
\]

not

\[
L_s =
L(x_s;\theta_{170}).
\]

The latter is a retrospective evaluation of a frozen trained model, not a learning trajectory.

Similarly, improvement during Phase 1E should be described as **retrofit optimization**, not further A/B learning.

---

# 18. Priority order

Implement in this order:

1. architecture/checkpoint provenance audit;
2. reproduce current frozen-checkpoint plot deliberately;
3. reconstruct exact original training trajectory;
4. add frozen Mixed / Novel / NoContext probes at registered training checkpoints;
5. regenerate raw-loss figures;
6. verify Phase 1E starting checkpoints;
7. only then revisit scientific interpretation.

Do not tune Phase 1E or change the model architecture while this audit is underway.
