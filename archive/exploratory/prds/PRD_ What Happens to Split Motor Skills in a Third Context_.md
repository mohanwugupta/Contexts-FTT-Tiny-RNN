# PRD: What Happens to Split Motor Skills in a Third Context?

## 1. Project Summary

### Scientific question

Previous work established that, under alternating Maze and Melody contexts, humans behave as though the same motor sequence is represented by **separate context-specific skill states**.

The new question is downstream of that result:

> **Once two context-specific skills have been established, how does the learner determine performance when entering a context that does not uniquely correspond to either existing skill?**

Two new datasets provide complementary tests:

1. **Combined-cue dataset:** during two 10-sequence probe blocks, participants perform the sequence in a context containing cues from both established contexts.
2. **Novel-context dataset:** during two 10-sequence probe blocks, participants perform the sequence in a distinct third context.

Human performance in the probe blocks appears approximately intermediate between performance in the two established contexts.

The goal is **not** simply to reproduce this intermediate RT.

The goal is to determine which computation produces it.

---

# 2. Core Scientific Premise

The previous project has already established the starting architecture:

\[
S_A \neq S_B
\]

where \(S_A\) and \(S_B\) are context-specific latent skill states corresponding to Maze and Melody.

Therefore, the new models should **condition on two established representations**, rather than reopening the lumped-versus-split question.

The scientific problem is instead the mapping:

\[
(S_A,S_B,\text{current cues},\text{history})
\rightarrow
\text{expressed performance}.
\]

The central distinction is between:

- **which existing representation(s) are retrieved,**
- **whether retrieval depends on contextual inference/history,**
- **whether probe experience updates the existing representations,**
- and **whether the probe causes a new representation to form.**

---

# 3. Candidate Mechanisms

The project should discriminate the following mechanisms rather than merely compare arbitrary neural architectures.

## H1. Static weighted retrieval

The probe context simultaneously retrieves the two existing skills:

\[
z_t=qS_{A,t}+(1-q)S_{B,t}.
\]

For a perfectly symmetric mixture:

\[
q=.5.
\]

The weight may instead be estimated from data.

### Predictions

- Probe performance is immediately intermediate.
- Same probe cue produces the same mixture regardless of preceding context.
- Little systematic change in \(q\) across the 10 probe sequences.
- Probe exposure does not alter subsequent A/B performance if retrieval is expression-only.
- The second probe block resembles the first, after accounting for ordinary learning/fatigue.

---

## H2. Cue-compositional retrieval

The established contextual features independently retrieve their associated skills.

Let the input contain Maze and Melody cue evidence:

\[
c_t=[c_A,c_B].
\]

Retrieval is determined by the relative cue evidence.

Pure contexts:

\[
A=[1,0], \qquad B=[0,1].
\]

Combined context:

\[
AB=[1,1]
\]

or, in a normalized implementation,

\[
AB=[.5,.5].
\]

A genuinely novel context contains neither established feature:

\[
C=[0,0].
\]

### Key prediction

**Combined and novel contexts need not behave the same way.**

The combined-cue condition should strongly support joint retrieval because both established cues are explicitly present.

A novel cue has no direct learned association with either representation and therefore requires some different rule: prior, history, fallback, or new-state formation.

This cross-dataset contrast is one of the most important tests in the project.

---

## H3. Contextual inference / responsibility weighting

The learner maintains a latent belief about which existing skill is currently relevant:

\[
q_t=P(A\mid\text{cues, history}).
\]

Behavior is:

\[
z_t=q_tS_{A,t}+(1-q_t)S_{B,t}.
\]

But unlike H1, \(q_t\) evolves:

\[
\operatorname{logit}(q_{t+1})
=
\lambda\operatorname{logit}(q_t)
+\beta e_t,
\]

where \(e_t\) is contextual evidence favoring A versus B.

### Predictions

The same probe context should depend on recent history.

For example:

\[
P(A\mid C,\text{previous A})
>
P(A\mid C,\text{previous B}).
\]

Therefore:

- C following Maze should be more Maze-like.
- C following Melody should be more Melody-like.
- This predecessor dependence should be strongest when current evidence is weak or ambiguous.
- The novel context may show stronger history dependence than the combined-cue context.

This provides a behavioral signature that a static weighted-average model cannot explain.

---

## H4. Responsibility-weighted learning

The probe context not only retrieves the two skills but also updates them.

For example:

\[
\Delta S_A \propto q_t
\]

and

\[
\Delta S_B \propto 1-q_t.
\]

### Predictions

Probe experience should produce **aftereffects**.

After a 50/50 probe block:

- later A performance should differ from what would have been expected without the probe;
- later B performance should also differ;
- the relative size of these effects should correspond to inferred responsibility \(q\).

This distinguishes **mixture during expression** from **mixture during learning**.

---

## H5. Third-state formation

The learner treats the probe context as a new context and creates:

\[
S_C.
\]

Its initial value could inherit information from A/B:

\[
S_{C,0}
=
qS_A+(1-q)S_B,
\]

which means the first probe sequences can still appear intermediate.

The critical distinction is what happens afterward.

### Predictions

- systematic within-probe learning;
- stronger carryover from probe block 1 to probe block 2;
- increasing independence from immediate A/B history;
- little or no corresponding change in A/B performance;
- the second probe block increasingly behaves like retrieval of an established C state rather than recomputation of an A/B mixture.

---

## H6. Default / fallback state

Failure to identify either familiar context may cause retrieval of a context-general/default skill state.

This is particularly relevant to the **novel context**.

The prior NoContext dataset provides an external behavioral benchmark for this possibility.

### Prediction

Novel-context behavior should resemble the NoContext solution more strongly than the combined-cue context does.

This should be treated as a secondary hypothesis because the old and new participants are not identical.

---

# 4. Critical Behavioral Signatures

Do not reduce the data to “probe RT is intermediate.”

For every participant/probe block, estimate the following.

## 4.1 Mixture index

Estimate expected counterfactual A and B performance at the probe time:

\[
\hat y_A(t),\qquad\hat y_B(t).
\]

Then define:

\[
M_t=
\frac{y_{\text{probe},t}-\hat y_B(t)}
{\hat y_A(t)-\hat y_B(t)}.
\]

Under this convention:

- \(M=1\): A-like
- \(M=0\): B-like
- \(M=.5\): midpoint

Compute this on:

- raw RT;
- log RT;
- optionally speed \(1/RT\).

The scale on which averaging occurs is itself theoretically informative.

---

## 4.2 Previous-context pull

Compare:

\[
M_{C\mid prev=A}
\]

versus

\[
M_{C\mid prev=B}.
\]

This is the primary discriminator between static averaging and contextual inference.

---

## 4.3 Within-probe evolution

Estimate:

\[
M_{t=1\ldots10}.
\]

Ask whether the probe:

- stays at a stable mixture;
- drifts toward one existing context;
- moves away from both contexts;
- or systematically improves as a new state develops.

---

## 4.4 Probe-block carryover

Compare probe block 1 with probe block 2.

A new-state account predicts greater probe-specific carryover than an expression-only mixture model.

---

## 4.5 Post-probe A/B aftereffects

For the first return to Maze or Melody after a probe, compare observed RT to the counterfactual continuation predicted from pre-probe A/B learning.

This is the primary discriminator between:

- expression-only mixture;
- responsibility-weighted learning;
- new-state formation.

---

## 4.6 Combined versus novel probe

Directly compare all metrics above between the two datasets.

This contrast should receive at least as much weight as any neural-network analysis.

---

# 5. Modeling Architecture

The analysis should proceed in three layers.

## Layer 1 — Established A/B skill model

Use the previous split state-space model to estimate the two context-specific skill trajectories:

\[
S_A(t),S_B(t).
\]

The probe observations must **not** determine the A/B split itself.

Where possible, fit the base model using Maze/Melody observations only.

The previous fatigue component should be retained as a nuisance process so that within-block slowing and recovery are not mistaken for changes in retrieval.

### Important constraint

Do **not** inherit the previous model's assumption that Mixed/NoContext performance is the average of A and B.

Only the Maze and Melody portions of the model are foundational here.

Probe readout is the quantity being tested.

---

# 6. Explicit Mechanistic Model Ladder

Implement these before the neural model.

## M0 — Equal-mixture benchmark

\[
q=.5.
\]

No free probe retrieval parameter.

This is the strongest simple prediction of literal averaging.

---

## M1 — Static fitted mixture

\[
q=q_{\text{probe}},
\qquad 0\le q\le1.
\]

Fit separately for:

- combined context;
- novel context.

Optional hierarchical extension:

\[
q_i\sim\mathcal N(q_{\text{group}},\sigma_q).
\]

This is primarily a **descriptive ceiling for static averaging**, not a full cognitive mechanism.

---

## M2 — Cue-compositional mixture

Use explicit feature inputs:

\[
[c_A,c_B].
\]

No arbitrary one-hot vector for the third context.

Compare:

### M2a: normalized cue composition

\[
A=[1,0],\;
B=[0,1],\;
AB=[.5,.5],\;
C=[0,0].
\]

### M2b: additive cue composition

\[
AB=[1,1].
\]

This tests whether cue magnitude itself matters.

---

## M3 — Recurrent contextual inference

Maintain one latent responsibility variable \(q_t\).

Minimal version:

\[
l_{t+1}
=
\lambda l_t+\beta(c_A-c_B)+b
\]

\[
q_t=\sigma(l_t).
\]

This is effectively a **one-dimensional recurrent cognitive model**.

The important feature is that when cue evidence is ambiguous:

\[
c_A-c_B\approx0,
\]

recent context can continue influencing \(q_t\).

---

## M4 — Context inference + responsibility-weighted updating

Use M3 for retrieval, but allow probe experience to update A/B states according to responsibility.

This creates testable post-probe aftereffects.

---

## M5 — Third-state model

Add:

\[
S_C.
\]

Estimate:

- initial transfer from A/B;
- C learning rate;
- C retention;
- optional separation between combined and novel contexts.

The model must pay the parameter cost for this additional state.

---

# 7. Primary Neural Discovery Model: Tiny Gate RNN

Do **not** begin with a generic RNN tasked with relearning the eight-item sequence.

Instead implement a tiny recurrent **gating model over the already-established A/B skills**.

## Architecture

Inputs at sequence \(t\):

\[
x_t=
[
c_A,
c_B,
\text{block start},
\text{previous-context features}
].
\]

Primary model:

```text
1-unit GRU
    ↓
sigmoid
    ↓
q_t = responsibility for skill A
```

Behavioral state:

\[
z_t=q_tS_A(t)+(1-q_t)S_B(t).
\]

RT decoder:

\[
\widehat{\log RT_t}
=
\beta_0+
\beta_z z_t+
\beta_F F_t.
\]

Run:

- hidden_dim = 1
- hidden_dim = 2
- hidden_dim = 4 as predictive ceiling

The **one-unit model is preferred unless additional dimensions materially improve held-out prediction**.

---

# 8. Neural Input Ablations

Run the gate RNN under several information conditions.

## G1 — Cue only

Inputs:

\[
[c_A,c_B].
\]

Tests pure cue composition.

---

## G2 — Cue + recurrence

Same cues, but recurrent state persists across contexts.

Tests history-dependent contextual inference.

---

## G3 — Cue + novelty flag

\[
[c_A,c_B,\text{novel}]
\]

Tests whether explicitly recognizing novelty adds predictive value.

If this dramatically improves the novel-context dataset but not the combined-cue dataset, that is evidence against treating the two probes as equivalent ambiguity.

---

## G4 — Non-recurrent gate

Replace the GRU with logistic regression.

This is essential.

If it performs as well as the recurrent version, there is no evidence that a belief/history state is needed.

---

# 9. Neural Interpretation Requirements

A tiny RNN only counts as mechanistically informative if its dynamics can be understood.

## 9.1 1D state-update plot

For the one-unit network, plot:

\[
h_t
\]

against

\[
\Delta h_t
\]

separately for:

- Maze;
- Melody;
- combined cue;
- novel cue.

Look for:

- fixed points;
- attraction toward A/B;
- persistence during ambiguous cues;
- novel-context dynamics.

---

## 9.2 Continuous cue interpolation

At matched hidden states, sweep:

\[
[1,0]
\rightarrow
[.75,.25]
\rightarrow
[.5,.5]
\rightarrow
[.25,.75]
\rightarrow
[0,1].
\]

Measure:

- \(q\);
- predicted RT;
- \(\Delta h\).

This is the analogue of the existing context-perturbation analysis, but performed continuously rather than by only zero/swap/shuffle interventions.

---

## 9.3 State intervention

Set the recurrent state to values corresponding to:

- strongly A-like;
- midpoint;
- strongly B-like.

Present identical probe cues.

Determine whether changing the latent state causally changes probe behavior.

---

## 9.4 Dynamical distillation

Regress learned network updates onto simple candidate equations.

For example:

\[
h_{t+1}
=
a h_t+b_Ac_A+b_Bc_B+b_NN_t+b_0.
\]

If this approximates the neural dynamics well, replace the GRU with the explicit equation and refit it directly to human behavior.

### Interpretability gate

Do not make a cognitive-mechanism claim from the RNN unless:

1. the low-dimensional solution is stable across seeds;
2. its relevant dynamics are reproducible;
3. a simple distilled update rule captures most of its behavior;
4. the distilled model retains most of the held-out predictive advantage.

The RNN is a **model-discovery tool**, not the endpoint.

---

# 10. Training and Evaluation Strategy

## 10.1 Avoid probe leakage

The strongest tests should not train and evaluate on the same probe observations.

Use three levels.

### Level A — True zero-shot

Fit A/B skill dynamics only.

Evaluate predictions implied without fitting probe RTs where a model permits this.

Examples:

- equal mixture;
- cue-compositional model;
- trained gate presented with combined cue values.

### Level B — Probe-block generalization

Fit/calibrate using probe block 1.

Predict probe block 2.

This is particularly important for:

- fitted \(q\);
- third-state models.

### Level C — Participant cross-validation

Use grouped cross-validation so sequences from the same participant do not leak between train and test folds when estimating population parameters.

---

# 11. Outcome Metrics

Primary:

- held-out log-RT RMSE;
- held-out negative log likelihood.

Secondary:

- BIC for explicit parametric models to maintain continuity with the prior state-space analysis;
- MAE;
- calibration of probe mixture index;
- correlation with participant-level probe trajectories.

Always report predictive performance separately for:

1. ordinary A/B trials;
2. probe trials;
3. return-to-A/B trials.

A model that predicts probe RT but destroys A/B predictions is not acceptable.

---

# 12. Scientific Model Recovery

This is mandatory before interpreting the human model comparison.

Generate synthetic datasets using the **actual experimental trial schedules** under:

- static mixture;
- contextual inference;
- responsibility-weighted updating;
- third-state formation.

Add noise approximately matched to human residual RT variability.

Fit all candidate models to every synthetic dataset.

Produce a model-recovery confusion matrix.

## Acceptance criterion

The pipeline must reliably distinguish the mechanisms on the empirical schedule before human-model comparison is treated as mechanistically diagnostic.

Target:

- ≥80% correct recovery for each primary mechanism across simulations at realistic noise;
- or explicitly document which hypotheses are empirically non-identifiable.

If H1 and H3 cannot be distinguished given the existing schedule, the correct scientific result is:

> these datasets cannot distinguish static mixture from contextual inference.

Do not select one based solely on a lower in-sample error.

---

# 13. Human Data Preprocessing

Preserve the previous project's preprocessing wherever applicable.

Normalized schema:

```text
participant_id
dataset
block_id
sequence_in_block
global_sequence
context
probe_type
is_probe
probe_block_number
previous_context
next_context
mean_rt
log_rt
error
break_duration
counterbalance
```

`probe_type`:

```text
None
Both
Novel
```

Context labels must be explicitly validated.

Unknown context labels must raise an error rather than silently mapping to another condition.

---

# 14. First-Sequence Handling

The previous project identified a large generic first-sequence detriment at trial onset.

For the new project:

- retain all sequences in the primary generative model;
- explicitly model sequence-position/fatigue/restart effects;
- report probe sequence 1 separately because it contains potentially important retrieval information;
- repeat key inference excluding sequence 1 as a robustness analysis.

A mechanism should not be inferred solely from a generic trial-start cost.

---

# 15. Existing Code to Reuse

## Reuse directly or refactor

### `pipeline.py`

Reuse/refactor:

- column inference;
- participant identification;
- context-column selection;
- deterministic seeds;
- output-directory management;
- CSV export conventions.

Do not reuse the existing task-training orchestration itself.

---

### `metrics.py`

Reuse conceptual structure for:

- context perturbations;
- hidden-state shifts;
- trajectory extraction;
- shared state-space plotting utilities.

Add new metrics:

```text
mixture_index
previous_context_pull
within_probe_drift
probe_block_carryover
post_probe_aftereffect_A
post_probe_aftereffect_B
cue_interpolation_linearity
gate_entropy
```

---

### `plotting.py`

Reuse plotting/output conventions.

Add probe-specific figures described below.

---

## Do not directly reuse as the primary model

### `ContextSequenceRNN`

The existing model predicts sequence elements and learns by gradient descent during the participant timeline.

That solves a different problem.

The new primary model predicts **human behavior** and should keep learned network parameters fixed while its recurrent state evolves through a participant's experience.

---

### Existing Mixed belief/plasticity model

Do not use its current assumptions as the starting point because it already implements:

- ambiguous cue → approximately 50/50 context representation;
- uncertainty → reduced plasticity.

Those are candidate hypotheses H3/H4.

The new model must be capable of rejecting them.

---

# 16. Proposed Code Structure

```text
probe_modeling/
    __init__.py
    data.py
    skill_states.py
    readout_models.py
    gate_models.py
    tiny_gate_rnn.py
    fit.py
    prediction.py
    metrics.py
    recovery.py
    plotting.py
    pipeline.py

scripts/
    run_probe_analysis.py
    run_model_recovery.py

tests/
    test_data_contract.py
    test_skill_states.py
    test_readout_models.py
    test_context_inference.py
    test_probe_updates.py
    test_third_state.py
    test_tiny_gate_rnn.py
    test_model_recovery.py
    test_no_probe_leakage.py
    test_pipeline_smoke.py
```

---

# 17. TDD Requirement

Development follows strict:

> **RED → GREEN → REFACTOR**

No scientific feature is implemented before the corresponding test exists.

The tests are not merely software tests.

They include **scientific behavioral contracts**.

---

# 18. RED → GREEN Milestone 1: Data Contract

## RED

Write failing tests asserting that the loader:

- identifies participant/context/sequence columns;
- preserves chronological order;
- correctly tags Both versus Novel probes;
- identifies probe block 1 versus 2;
- identifies preceding/following established context;
- rejects unknown labels;
- does not mix participants;
- correctly computes `is_probe`.

Run tests.

They must fail because the new adapter does not exist.

## GREEN

Implement only enough `data.py` to pass these tests.

### Gate

No modeling begins until all data-contract tests are green.

---

# 19. RED → GREEN Milestone 2: Reproduce Previous Split Model

Before extending the science, verify the prior result computationally.

## RED

Write tests requiring the base skill-state implementation to:

- instantiate distinct A and B states;
- reproduce no-cross-talk behavior when \(A_{cross}=0\);
- reproduce coupled behavior when \(A_{cross}>0\);
- separate fatigue from latent skill;
- reproduce the prior dataset's qualitative split-model trajectory.

Where the original state-space implementation is available, test against it.

Otherwise reimplement from the manuscript and verify against saved prior outputs.

## GREEN

Pass all tests.

### Scientific regression test

Future changes to the new probe model must not alter the established A/B split dynamics.

---

# 20. RED → GREEN Milestone 3: Static Mixture Models

## RED

Synthetic test:

```text
A state = 400
B state = 600
q = .25
```

Expected probe state:

```text
550
```

Test:

\[
.25(400)+.75(600)=550.
\]

Additional tests:

- q=1 → exactly A;
- q=0 → exactly B;
- q=.5 → exact midpoint;
- expression-only probe does not change A/B states.

## GREEN

Implement `StaticMixtureReadout`.

---

# 21. RED → GREEN Milestone 4: Contextual Inference

## RED

Tests:

### Pure cue

Repeated A evidence drives:

\[
q\rightarrow1.
\]

Repeated B evidence drives:

\[
q\rightarrow0.
\]

### Ambiguous cue

With zero current evidence:

```text
previous A history → q > .5
previous B history → q < .5
```

### Symmetry

Counterbalancing A↔B must produce mirrored predictions.

## GREEN

Implement minimal one-dimensional inference model.

No GRU yet.

---

# 22. RED → GREEN Milestone 5: Probe Updating

## RED

Expression-only model:

```text
A_before == A_after_probe
B_before == B_after_probe
```

Responsibility-update model with q=.7:

```text
|ΔA| > |ΔB|
```

with relative updating proportional to .7 versus .3.

Third-state model:

```text
C changes during C training
A and B unchanged
```

## GREEN

Implement the three update policies as interchangeable modules.

---

# 23. RED → GREEN Milestone 6: Tiny Gate RNN

## RED

Synthetic curriculum:

```text
AAAA -> ambiguous -> BBBB
```

Required behaviors after fitting:

- pure A produces q near 1;
- pure B produces q near 0;
- ambiguous cue after A is more A-like than ambiguous cue after B.

Second synthetic dataset generated by static mixture:

- recurrent model should not gain meaningful held-out advantage over non-recurrent gate.

## GREEN

Implement:

```python
TinyGateRNN(hidden_dim=1)
```

Then parameterize hidden dimension for 2 and 4.

---

# 24. RED → GREEN Milestone 7: Continuous Interventions

## RED

Construct a deterministic mock gate where cue weight is known.

Sweep:

```text
1.00 / 0.00
0.75 / 0.25
0.50 / 0.50
0.25 / 0.75
0.00 / 1.00
```

Require output ordering to follow the known gate.

## GREEN

Implement `run_cue_interpolation()` returning:

```text
cue_A
cue_B
hidden_state
q
predicted_rt
delta_hidden
```

---

# 25. RED → GREEN Milestone 8: Model Recovery

## RED

Before recovery code exists, write tests that generate low-noise synthetic datasets from each model.

Expected:

```text
Generate H1 → H1 ranks best
Generate H3 → H3 ranks best
Generate H4 → H4 ranks best
Generate H5 → H5 ranks best
```

## GREEN

Implement fitting/recovery pipeline.

Then progressively increase noise to empirical levels.

Produce recovery matrix.

---

# 26. Required Figures

## Figure 1 — Human phenomenon

For both datasets:

- A trajectory;
- B trajectory;
- probe trajectory.

Overlay the counterfactual midpoint:

\[
.5(\hat A+\hat B).
\]

---

## Figure 2 — Mixture index

Probe \(M_t\) over sequences 1–10.

Separate:

- Both;
- Novel;
- previous A;
- previous B.

This may be the single most diagnostically useful behavioral figure.

---

## Figure 3 — Model predictions

Overlay observed probe data with:

- equal mixture;
- static fitted mixture;
- contextual inference;
- responsibility update;
- third-state model.

---

## Figure 4 — Aftereffects

Observed minus predicted A/B performance following probe blocks.

Show separately for:

- return to A;
- return to B;
- Both dataset;
- Novel dataset.

---

## Figure 5 — Tiny gate dynamics

For 1D model:

\[
h_t \text{ vs. } \Delta h_t
\]

under each context.

---

## Figure 6 — Cue morph

Plot:

\[
q
\]

and predicted RT against cue mixture from pure A to pure B.

Show both:

- fixed history;
- A versus B history.

---

## Figure 7 — Model recovery

Confusion matrix:

```text
true generating mechanism × selected mechanism
```

---

## Figure 8 — Model comparison

Held-out performance plus BIC for interpretable models.

Include neural ceiling separately.

Do not mix neural complexity and explicit-model complexity into a single BIC table.

---

# 27. Primary Decision Logic

## Result A

Both and Novel are approximately .5 mixtures.

No predecessor dependence.

No post-probe aftereffects.

No probe-specific learning.

### Interpretation

Static/fallback mixture is sufficient.

Do not invoke contextual inference or new-state formation.

---

## Result B

Probe position depends strongly on preceding context.

### Interpretation

Evidence for history-dependent contextual inference/responsibility.

If the 1D recurrent gate captures this and distills into a simple belief update, that becomes the preferred cognitive mechanism.

---

## Result C

Both cue condition strongly averages A/B, whereas Novel does not.

### Interpretation

Evidence favoring compositional retrieval from the learned contextual features.

---

## Result D

Probe experience changes subsequent A and B.

### Interpretation

Retrieval is accompanied by responsibility-weighted learning/updating.

---

## Result E

Probe performance develops across repetitions and transfers strongly from probe block 1 to 2, while A/B are largely unaffected.

### Interpretation

Evidence for formation of a third context-specific state.

---

## Result F

Several mechanisms fit equally well and model recovery shows they are difficult to distinguish.

### Interpretation

The datasets establish weighted expression but do not identify the underlying computation.

This is an acceptable scientific outcome.

---

# 28. Neural Model Stopping Rule

Do not escalate architecture automatically.

Fit:

```text
nonrecurrent gate
1-unit GRU
2-unit GRU
4-unit GRU
```

Stop when additional dimensions provide negligible held-out improvement.

If 1D is sufficient, use it.

If 2D materially improves prediction, inspect the second dimension.

If the 2D dynamics cannot be consistently interpreted across random seeds, treat the network only as a predictive ceiling rather than a cognitive model.

Do **not** move to DisRNN unless:

1. tiny networks predict substantially better than explicit models;
2. their dynamics remain entangled;
3. this entanglement prevents extraction of a simple cognitive rule.

---

# 29. No Pretraining in the Primary Analysis

Do not pretrain the probe model on next-item sequence prediction.

This would import the representational biases of the previous SRN into the mechanism we are trying to discover.

Optional robustness analysis only:

```text
random initialization
vs.
task-pretrained initialization
```

If both converge to equivalent behavioral dynamics, that strengthens the result.

It is not required for the primary pipeline.

---

# 30. Required Exports

Each run should produce:

```text
clean_probe_data.csv

counterfactual_AB_states.csv
behavioral_probe_metrics.csv

model_fits.csv
model_predictions.csv
model_comparison.csv

gate_states.csv
cue_intervention_results.csv
state_intervention_results.csv

post_probe_aftereffects.csv

model_recovery_results.csv
model_recovery_confusion.csv

run_config.json
```

Every neural export must contain:

```text
seed
hidden_dim
train_participants
test_participants
dataset
model_version
```

---

# 31. Reproducibility

- deterministic global seed;
- participant-level grouped splits saved to disk;
- same splits across competing models;
- ≥20 neural seeds for final reported tiny-RNN results;
- all hyperparameters stored in config;
- no human-data-dependent hyperparameter changes after the final test split is examined.

---

# 32. Minimum Viable Analysis

The quickest scientifically defensible version of this project is:

### Step 1
Reproduce/freeze the established A/B split-state model.

### Step 2
Compute counterfactual A/B trajectories through the probe period.

### Step 3
Quantify:

- mixture index;
- predecessor dependence;
- within-probe drift;
- probe-block carryover;
- post-probe aftereffects.

### Step 4
Fit:

- equal mixture;
- fitted static mixture;
- one-dimensional contextual-inference model;
- third-state model.

### Step 5
Run synthetic model recovery.

### Step 6
Only then fit the 1-unit tiny recurrent gate.

This version should answer most of the scientific question without requiring a large neural-network project.

---

# 33. Definition of Done

The project is complete when:

- [ ] Both datasets pass the same validated data contract.
- [ ] Previous A/B split dynamics are reproduced.
- [ ] Probe observations are excluded from A/B skill estimation.
- [ ] Static mixture, contextual inference, weighted-update, and third-state models are implemented.
- [ ] All models have unit tests written before implementation.
- [ ] Every feature passes RED → GREEN testing.
- [ ] Model recovery is run on the empirical schedules.
- [ ] Identifiability failures are explicitly documented.
- [ ] Held-out model comparison is complete.
- [ ] Previous-context effects are quantified.
- [ ] Within-probe dynamics are quantified.
- [ ] Probe-block carryover is quantified.
- [ ] Post-probe A/B aftereffects are quantified.
- [ ] Combined versus Novel contexts are directly compared.
- [ ] A non-recurrent gate is compared against a 1-unit recurrent gate.
- [ ] 1-, 2-, and 4-unit RNN dimensionality comparisons are complete if the neural analysis is reached.
- [ ] Cue-interpolation interventions are complete.
- [ ] Recurrent dynamics are distilled into an explicit equation if possible.
- [ ] No mechanistic conclusion exceeds what the model-recovery analysis shows is identifiable.

---

# 34. Intended Scientific Contribution

The previous project asks:

> **When does context cause the same motor sequence to split into separate skills?**

The new project asks the natural next question:

> **Once those skills have split, how does the learner decide which skill—or combination of skills—to express when the context is ambiguous, compositional, or novel?**

The strongest possible result is not simply that performance in the probe context is intermediate.

It is identifying the computation that generates that intermediate behavior:

\[
\boxed{
\text{context cues + recent history}
\rightarrow
\text{responsibility over established skills}
\rightarrow
\text{expression and possibly learning}
}
\]

and determining whether that computation is sufficient, or whether exposure to a new context causes a genuinely new skill representation to form.