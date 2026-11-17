Yes. I would treat this as a **replacement/update to the current probe-modeling PRD**, not a separate parallel analysis. The major change is that the model comparison is organized around the actual temporal information in the experiment:

[
\text{Mixed 1}\rightarrow\text{Mixed 2}\rightarrow\text{return to A/B}
\rightarrow\text{extended A/B training}\rightarrow\text{Mixed Test}.
]

The current implementation already preserves the first two Mixed training blocks and later Mixed test as distinct epochs and keeps probe RT out of the foundational A/B fit.  The previous paper also explicitly removed the first sequence because of the large generic trial-start detriment, which is the preprocessing convention we should restore here. 

# PRD v2: Epoch-Structured Modeling of Combined-Context Retrieval

## 1. Project objective

### Scientific question

Previous work established that performance of the same motor sequence under Maze and Melody is best described by separate context-specific skill states.

The present experiment asks:

> **Once A and B have split into separate skills, what happens when both contextual cues are presented simultaneously?**

The Combined/Mixed context occurs only at three points:

1. **Mixed 1:** first 9 analyzable Mixed sequences.
2. **Mixed 2:** immediately following Mixed 1, another 9 analyzable Mixed sequences.
3. **Mixed Test:** 9 analyzable Mixed sequences at the end of the experiment after substantial intervening A/B practice.

The aim is to distinguish:

* fixed weighted retrieval of A and B;
* history-dependent/contextual retrieval;
* adaptation of the retrieval rule during Mixed experience;
* updating of the existing A/B skills;
* formation and retention of a third Mixed-specific skill state.

The central analysis must exploit **when the three Mixed blocks occur**, not pool them into a generic probe category.

---

# 2. Required preprocessing change

## 2.1 Remove sequence 1 from every block

The first sequence of **every trial/block must be removed before any behavioral metric, counterfactual state fit, explicit model fit, or neural model fit**.

The previous human analysis removed this sequence because it exhibits a large generic trial-start performance detriment rather than the transition phenomenon of interest. 

The previous methods also defined sequence RT only for correctly completed sequences. 

### Required cleaned block structure

Each valid block must therefore contain:

[
\boxed{9\text{ correctly performed sequences}}
]

corresponding to original sequences 2–10.

Retain both:

```text
original_sequence_in_block = 2 ... 10
analysis_sequence_in_block = 1 ... 9
```

`analysis_sequence_in_block` should be used for plotting/modeling.

### Data-contract invariant

For every retained participant × block:

```text
n_sequences == 9
original_sequence_in_block does not contain 1
all sequences are correctly completed
```

Do not impute missing sequences to make nine.

If the task-level preprocessing genuinely produces fewer than nine correct sequences for a block, flag that block and apply the predetermined complete-case rule rather than silently padding it.

---

# 3. Experimental epoch labels

The loader must create a categorical variable:

```text
epoch
```

with:

```text
A_or_B
Mixed_1
Mixed_2
Post_Mixed_Return
Mixed_Test
```

For the current Combined schedule:

```text
Mixed_1 = first Mixed Training block
Mixed_2 = second consecutive Mixed Training block
Post_Mixed_Return = first A/B block following Mixed_2
Mixed_Test = final Mixed block
```

The existing implementation already preserves the first two training probe blocks and later test separately; this should now become a first-class part of the model rather than only metadata. 

---

# 4. Scientific interpretation of the three Mixed epochs

The blocks answer different questions and must not be treated as exchangeable observations.

## Mixed 1: initial retrieval

Primary question:

> What happens the first time both established contextual cues are simultaneously present?

This is the cleanest test of how A and B are initially accessed.

Candidate prediction:

[
y_{M1,t}
========

q_t\hat y_{A,t}+
(1-q_t)\hat y_{B,t}.
]

The main question is whether (q):

* equals .5;
* is a fixed non-.5 value;
* depends on history/counterbalance.

---

## Mixed 2: immediate adaptation

Mixed 2 follows directly after nine analyzed Mixed 1 sequences.

Primary question:

> Did one block of Combined-context experience change how the existing skills are expressed?

Compare:

[
q_{M2}
]

against:

[
q_{M1}.
]

A static retrieval mechanism predicts little systematic change.

A learning/adaptation mechanism predicts:

[
q_{M2}\neq q_{M1}
]

or some other systematic change in predicted Mixed performance.

Mixed 2 can establish **adaptation during Mixed experience**, but by itself cannot tell whether that adaptation occurred in:

* the retrieval gate;
* A/B representations;
* or a new C state.

---

## Post-Mixed return: effect on A/B

After 18 analyzed Mixed sequences, participants return to an established context.

Primary question:

> Did Mixed experience modify A or B themselves?

This is the best available discriminator between:

### Expression-only retrieval

[
S_A^{after}=S_A^{before},
\qquad
S_B^{after}=S_B^{before}.
]

versus:

### Responsibility-weighted updating

[
\Delta S_A\propto q,
\qquad
\Delta S_B\propto1-q.
]

This measurement is **secondary evidence**, because there is only one immediate return block and return context is determined by counterbalance.

Do not describe it as repeated evidence for an aftereffect.

---

## Mixed Test: delayed retrieval/retention

The final Mixed block occurs after extensive intervening A/B practice.

Primary question:

> Did anything learned specifically about Combined survive the intervening experience?

This is the strongest test of persistent Mixed-specific learning.

A purely recomputed mixture predicts that final Mixed behavior should mostly follow the **current A and B states**.

A retained third state predicts dependence on earlier Mixed experience:

[
S_C^{test}
\approx
\rho S_C^{training}.
]

A persistent learned gate similarly predicts that the rule established during Mixed 1/2 survives to final test.

---

# 5. Stage 1: foundational A/B model

Retain the basic split state-space logic from the previous paper.

Estimate:

[
S_A(t),S_B(t)
]

using **A/B observations only**.

Probe observations must never fit these foundational trajectories, which is already a safeguard in the current implementation. 

Because sequence 1 is now removed, simplify the nuisance model accordingly.

Retain:

* gradual skill dynamics;
* within-block fatigue across sequences 2–10 if supported;
* between-block recovery.

Do **not** require a special first-sequence/restart parameter in the primary model because that observation has been removed.

---

# 6. Primary dependent variable

Fit human behavior directly on:

[
\log RT.
]

Do not use the per-observation mixture ratio as the primary inferential quantity.

The existing mixture index,

[
M_t=
\frac{y_{Mixed,t}-\hat y_{B,t}}
{\hat y_{A,t}-\hat y_{B,t}},
]

is useful descriptively but unstable when A and B are close.

Retain it as a secondary plot.

### Primary stable quantity

Also compute:

[
D_t=
y_{Mixed,t}
-----------

\frac{
\hat y_{A,t}+\hat y_{B,t}
}{2}.
]

Interpretation:

* (D=0): midpoint;
* sign toward A: A-weighted;
* opposite sign: B-weighted.

Primary mechanistic models should estimate (q) directly from RT rather than derive (q) independently for every observation.

---

# 7. Counterbalance treatment

This is mandatory.

The apparent “previous-context effect” cannot simply be interpreted as a within-subject recency effect because predecessor context is tied to experimental counterbalance.

Therefore:

### Never label the primary effect simply:

```text
previous_context effect
```

Use:

```text
counterbalance/history effect
```

unless a model specifically establishes that immediate predecessor context explains it.

All behavioral summaries and model fits must include:

```text
counterbalance
```

as an explicit variable.

### Required counterbalance tests

1. Fit results separately by counterbalance.
2. Fit combined models with counterbalance represented explicitly.
3. Conduct an A↔B label-mirroring test.
4. Where feasible, fit on one counterbalance and predict the other after appropriate label reversal.

A mechanism that merely memorizes a counterbalance-specific trajectory should not be interpreted as a general contextual-inference mechanism.

---

# 8. Explicit cognitive model ladder

## M0: Equal mixture

[
q=.5.
]

Prediction:

[
y_M=.5S_A+.5S_B.
]

No Mixed learning.

---

## M1: Static weighted retrieval

One fitted population mixture weight:

[
q=q_0.
]

The same (q) applies to:

* Mixed 1;
* Mixed 2;
* Mixed Test.

No history.

No updating.

This is the minimal alternative to equal averaging.

---

## M2: Counterbalance/history-dependent static retrieval

Allow:

[
q=q_{CB}.
]

This determines whether the apparent history effect can be explained purely by a stable counterbalance-specific retrieval bias.

This model is important because it is simpler than recurrent contextual inference.

If M2 explains the data as well as recurrence, do not claim a dynamically evolving belief state.

---

## M3: Recurrent contextual inference

Introduce one latent responsibility variable:

[
l_{t+1}
=======

\lambda l_t+
\beta e_t+b
]

[
q_t=\sigma(l_t).
]

Behavior:

[
y_t
===

q_tS_A(t)+(1-q_t)S_B(t).
]

No changes are made to A or B.

This model tests:

> Does dynamically maintained contextual history improve prediction beyond fixed counterbalance effects?

---

## M4: Mixed-specific gate adaptation

Allow Mixed experience itself to alter the retrieval rule:

[
q_{t+1}
=======

q_t+\alpha_q f(t).
]

But:

[
S_A,S_B
]

remain unchanged.

This creates a critical distinction:

> Mixed 1 → Mixed 2 changes, but post-Mixed A/B does not.

This mechanism should be added because the old model comparison jumps too quickly from contextual inference to A/B updating.

---

## M5: Responsibility-weighted A/B updating

Mixed expression remains:

[
y_t=q_tS_A+(1-q_t)S_B.
]

But Mixed practice alters the existing states:

[
S_{A,t+1}
=========

S_{A,t}+\alpha_Aq_tE_t
]

[
S_{B,t+1}
=========

S_{B,t}+\alpha_B(1-q_t)E_t.
]

Primary signature:

> Changed A/B performance after Mixed 2.

Mixed 1 → Mixed 2 improvement alone is insufficient evidence for this model.

---

## M6: Third-state formation

Introduce:

[
S_C.
]

Initialization:

[
S_{C,0}
=======

qS_A+(1-q)S_B.
]

Learning:

[
S_{C,t+1}
=========

S_{C,t}+\alpha_CE_t.
]

Retention:

[
S_C^{test}
==========

\rho_CS_C^{training}.
]

Primary signatures:

* systematic Mixed 1 → Mixed 2 change;
* retained advantage/pattern at Mixed Test;
* limited post-Mixed A/B aftereffect.

---

# 9. Model fitting strategy

Use grouped participant splits.

Current final analysis uses an 80/20 participant split and should retain that basic principle. 

## Training participants

Fit:

* population A/B dynamic parameters;
* mechanism parameters;
* counterbalance effects.

## Held-out participants

Their own A/B observations may be used to estimate their conditional A/B counterfactual states.

However:

* their Mixed RTs must not estimate population mechanism parameters;
* all Mixed predictions are out-of-participant predictions.

---

# 10. Epoch-specific evaluation

Do not use pooled `probe` RMSE as the headline statistic.

Every model must report separately:

```text
Mixed_1
Mixed_2
Post_Mixed_Return
Mixed_Test
ordinary_AB
overall
```

### Primary comparisons

#### Initial retrieval

[
RMSE_{Mixed1}
]

#### Short-term adaptation

[
RMSE_{Mixed2}
]

and

[
\Delta q=q_{M2}-q_{M1}.
]

#### A/B modification

[
RMSE_{PostMixedReturn}.
]

#### Long-term retention

[
RMSE_{MixedTest}.
]

A mechanism claim must identify **where its predictive advantage occurs**.

For example:

> “Responsibility updating wins primarily on Post-Mixed Return.”

is much more informative than:

> “Responsibility updating had the lowest global RMSE.”

---

# 11. Prospective temporal prediction tests

Add stronger temporal tests.

## Test A: Mixed 1 → Mixed 2

Fit/calibrate the minimal Mixed-specific parameter using Mixed 1 where appropriate.

Predict Mixed 2.

This asks:

> Can the mechanism predict the immediate consequence of nine Mixed experiences?

---

## Test B: Mixed 1/2 → Mixed Test

No final-test refitting.

Use what the model learned/inferred during Mixed 1/2 plus the intervening A/B timeline to predict final Mixed Test.

This is the strongest existing validation of persistent Mixed-specific learning.

---

## Test C: Mixed 1/2 → Post-Mixed A/B

Do not fit the aftereffect observation.

Predict it.

This tests true downstream consequences.

---

# 12. Tiny-RNN discovery model

Keep the neural model, but reposition it as secondary.

The existing implementation already treats the tiny GRU as a predictive/discovery model alongside a non-recurrent gate. 

Fit:

```text
non-recurrent gate
1-unit recurrent gate
2-unit recurrent gate
4-unit recurrent gate
```

The target remains human log RT over frozen A/B counterfactual states.

---

# 13. New neural recurrence ablations

These are especially important given the current results.

## N0: no recurrence

Cue/counterbalance features only.

---

## N1: full recurrence

Hidden state persists through the participant timeline.

---

## N2: block-reset recurrence

Reset hidden state at every block.

If N1 outperforms N2, information carried between blocks matters.

---

## N3: reset-before-Mixed-1

Run the same trained network but set:

[
h=0
]

immediately before Mixed 1.

Measure change in:

* Mixed 1;
* Mixed 2;
* final Mixed Test.

This directly tests whether pre-Mixed history contributes to Mixed expression.

---

## N4: reset-between-Mixed-1-and-Mixed-2

Reset after Mixed 1.

If Mixed 2 changes substantially, the network was carrying information learned during Mixed 1.

---

# 14. Neural interpretability

For the 1D gate:

Plot:

[
h_t
]

across the full experimental timeline.

Mark explicitly:

```text
Mixed 1
Mixed 2
Post-Mixed Return
Mixed Test
```

Also plot:

[
h_t \rightarrow \Delta h_t
]

separately for these epochs.

### Distillation

Fit:

[
h_{t+1}
=======

a h_t+
b_Ac_A+
b_Bc_B+
b_M I_{Mixed}+
b_{CB}CB+
b_0.
]

Then progressively add:

* Mixed experience count;
* time since last Mixed;
* current A/B state difference.

Compare distilled equations by held-out (R^2).

If a simple update equation captures the GRU, use that equation as the candidate cognitive model.

---

# 15. Model recovery v2

Rerun all recovery simulations using the **new nine-sequence-per-block schedule**.

Simulated schedule must exactly match:

```text
17 blocks
9 observations/block
Mixed 1
Mixed 2
intervening A/B
Mixed Test
actual counterbalances
```

Generate data from:

```text
static mixture
counterbalance-specific mixture
recurrent inference
gate adaptation
responsibility update
third state
```

## Important improvement

Do not recover only from one hand-selected parameter setting.

Run recovery at:

1. easy/high-signal parameters;
2. fitted human-like parameters;
3. parameter neighborhoods around the empirical estimates.

The previous pipeline obtained perfect recovery under its configured synthetic conditions, which establishes basic identifiability but not necessarily identifiability near the human solution. 

---

# 16. TDD: strict RED → GREEN → REFACTOR

Every scientific feature must begin with a failing test.

## Milestone 1 — First-sequence removal

### RED

Write tests asserting:

```text
original sequence 1 exists in raw fixture
```

but after preprocessing:

```text
original_sequence_in_block != 1 for all rows
analysis_sequence_in_block == 1..9
n_rows per participant × block == 9
```

Also require:

```text
all retained sequences are correct
```

Run test and confirm failure.

### GREEN

Implement sequence-1 exclusion.

Do not modify modeling code yet.

### REFACTOR

Centralize exclusion into the canonical loader so no downstream pipeline can bypass it.

---

# 17. Milestone 2 — Epoch tagging

### RED

Synthetic 17-block participant must produce exactly:

```text
Mixed_1: 9 rows
Mixed_2: 9 rows
Mixed_Test: 9 rows
```

Post-Mixed Return must point to the first A/B block after Mixed 2.

### GREEN

Implement epoch mapper.

---

# 18. Milestone 3 — Counterfactual A/B leakage test

### RED

Alter all Mixed RTs by +1000 ms.

Expected:

```text
A counterfactual trajectory unchanged
B counterfactual trajectory unchanged
```

### GREEN

Ensure A/B fitting filters all Mixed observations.

This is a scientific regression test and must remain permanently green.

---

# 19. Milestone 4 — Static models

### RED

Known A/B fixture:

```text
A = 400
B = 600
```

Equal mixture must predict:

```text
500
```

Static (q=.25):

```text
550
```

Predictions must be identical across Mixed 1, Mixed 2, and Mixed Test when A/B states are held fixed.

### GREEN

Implement M0/M1.

---

# 20. Milestone 5 — Counterbalance-specific retrieval

### RED

Generate two mirrored counterbalances with different (q).

M1 must fail.

M2 must recover the counterbalance-specific difference.

After A↔B label swap, predictions must mirror.

### GREEN

Implement M2.

---

# 21. Milestone 6 — Recurrent inference

### RED

Synthetic schedule with identical current Mixed cues but different pre-Mixed histories.

Require:

[
q_{Mixed1}^{history=A}
\neq
q_{Mixed1}^{history=B}.
]

Resetting recurrent state immediately before Mixed must eliminate the difference.

### GREEN

Implement M3 and state-reset intervention.

---

# 22. Milestone 7 — Gate adaptation versus A/B updating

### RED

Generate expression-only gate adaptation.

Expected:

```text
Mixed2 differs from Mixed1
PostMixed A/B unchanged
```

Generate responsibility updating.

Expected:

```text
Mixed2 may differ
PostMixed A/B changes
```

### GREEN

Implement M4/M5.

This test is crucial because these mechanisms were insufficiently separated in v1.

---

# 23. Milestone 8 — Third state

### RED

Synthetic third-state data:

```text
Mixed1 learns
Mixed2 begins from Mixed1 endpoint
A/B unchanged
MixedTest retains rho × C state
```

Static mixture and gate adaptation should not reproduce all three signatures simultaneously.

### GREEN

Implement M6.

---

# 24. Milestone 9 — Epoch-specific model evaluation

### RED

Unit test ensures pooled `probe` RMSE is **not** accepted as the only probe evaluation output.

Required regions:

```text
mixed_1
mixed_2
post_mixed_return
mixed_test
ordinary_AB
all
```

### GREEN

Refactor comparison pipeline.

---

# 25. Milestone 10 — Neural history ablations

### RED

On known recurrent synthetic data:

```text
full recurrence < block-reset RMSE
full recurrence < pre-Mixed-reset RMSE on Mixed1
```

On memoryless synthetic data:

```text
recurrence should not materially outperform non-recurrent model
```

### GREEN

Implement reset interventions.

---

# 26. Required outputs

```text
clean_probe_data_v2.csv
counterfactual_AB_states_v2.csv

epoch_behavioral_summary.csv
mixed1_metrics.csv
mixed2_metrics.csv
mixed_test_metrics.csv
post_mixed_aftereffects.csv

model_fits_v2.csv
model_predictions_v2.csv
model_comparison_by_epoch.csv

temporal_prediction_tests.csv
counterbalance_generalization.csv

gate_states_v2.csv
gate_reset_interventions.csv
distilled_dynamics.csv

model_recovery_results_v2.csv
model_recovery_confusion_v2.csv
model_recovery_human_parameter_region.csv

run_config_v2.json
```

---

# 27. Required figures

### Figure 1 — Human timeline

Full trajectory with distinct shading for:

* Mixed 1;
* Mixed 2;
* Mixed Test.

Show nine analyzed observations per block.

---

### Figure 2 — Initial Mixed retrieval

Mixed 1 only.

Overlay:

[
A,\quad B,\quad midpoint,\quad observed.
]

Separate counterbalances.

---

### Figure 3 — Immediate adaptation

Overlay Mixed 1 and Mixed 2.

Plot:

[
Mixed2-Mixed1
]

as the main contrast.

---

### Figure 4 — Delayed Mixed Test

Observed final Mixed performance versus predictions based on:

* static recombination of current A/B;
* recurrent retrieval;
* learned Mixed gate;
* retained third state.

---

### Figure 5 — Post-Mixed A/B consequence

Explicitly title:

> **First A/B block after 18 Mixed sequences**

Do not describe it generically as “aftereffects” without this qualification.

Separate by return context and counterbalance.

---

### Figure 6 — Model performance by epoch

Rows/models × columns:

```text
Mixed1
Mixed2
PostMixedReturn
MixedTest
Overall
```

This replaces the current global bar-chart emphasis.

---

### Figure 7 — One-dimensional recurrent state

Full temporal trajectory plus reset interventions.

---

### Figure 8 — Model recovery

Include human-parameter-region recovery, not just easy synthetic recovery.

---

# 28. Decision rules

### Evidence for static mixture

If:

* one (q) predicts Mixed 1, Mixed 2, and Mixed Test;
* recurrence provides negligible improvement;
* no meaningful post-Mixed A/B consequence.

Then conclude:

> Combined context produces stable weighted retrieval of existing skills.

---

### Evidence for contextual inference

If:

* recurrent history predicts Mixed 1;
* recurrence beats fixed counterbalance effects;
* resetting hidden state removes the relevant effect;
* A/B remain unchanged.

Then conclude:

> Combined-context performance reflects history-sensitive retrieval over existing skills.

---

### Evidence for Mixed-specific gate learning

If:

* Mixed 1 → Mixed 2 changes systematically;
* change persists to Mixed Test;
* A/B return is not altered.

Then conclude:

> Participants learn how to retrieve A/B under the Combined context without substantially modifying the underlying A/B skills.

---

### Evidence for responsibility-weighted A/B updating

Require all of:

1. Mixed experience produces a predicted downstream change in A/B.
2. M5 beats expression-only M3/M4 specifically on those downstream observations.
3. Effect survives counterbalance controls.
4. Synthetic recovery shows M4 versus M5 can be distinguished near empirical parameter values.

Do **not** infer A/B updating merely from lower pooled RMSE/BIC.

---

### Evidence for third-state formation

Require:

* Mixed-specific learning between Mixed 1 and Mixed 2;
* retained Mixed-specific information at final Mixed Test;
* third-state model beats recomputed A/B mixture on final test;
* comparatively weak impact on A/B themselves.

---

# 29. Definition of done

The revised analysis is complete only when:

* [ ] Sequence 1 is removed from every block before all modeling.
* [ ] Every retained valid block contains nine correct analyzed sequences.
* [ ] Mixed 1, Mixed 2, and Mixed Test are modeled as distinct epochs.
* [ ] A/B counterfactual fitting remains entirely probe-free.
* [ ] Counterbalance is explicitly modeled.
* [ ] “Previous context” is not interpreted independently of counterbalance without evidence.
* [ ] Per-observation mixture index is secondary, not primary.
* [ ] Equal/static mixture is tested.
* [ ] Counterbalance-specific static retrieval is tested.
* [ ] Recurrent contextual inference is tested.
* [ ] Mixed-specific gate adaptation is tested.
* [ ] Responsibility-weighted A/B updating is tested.
* [ ] Third-state formation is tested.
* [ ] Mixed 1 → Mixed 2 prediction is evaluated.
* [ ] Mixed 1/2 → final Mixed Test prediction is evaluated without final-test fitting.
* [ ] Post-Mixed A/B prediction is evaluated separately.
* [ ] Model performance is reported separately for each epoch.
* [ ] Neural recurrence-reset interventions are run.
* [ ] 1D dynamics are distilled.
* [ ] Model recovery uses the 9-sequence actual schedule.
* [ ] Recovery is also tested near human-fitted parameter values.
* [ ] All new functionality was implemented through RED → GREEN → REFACTOR testing.

## Bottom-line goal

The revised pipeline should be able to turn the three sparse Mixed observations into a temporal mechanistic question:

[
\boxed{
\text{How is Mixed first interpreted?}
\rightarrow
\text{What changes after 9 Mixed sequences?}
\rightarrow
\text{Did that alter A/B?}
\rightarrow
\text{What survives to the final Mixed test?}
}
]

That is much more diagnostic than asking which model gives the lowest error after pooling the 27 analyzed Mixed observations per participant.
