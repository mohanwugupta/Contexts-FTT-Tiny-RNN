Yes. I would make this a **new mechanistic PRD layered on top of the current pipeline**, rather than another revision of the epoch-model-comparison work. The current work establishes the premise; this one asks the actual computational question.

The clean hypothesis is:

> **Context-dependent skills are retrieved by graded responsibility rather than categorical selection. The current context is compared with the contexts associated with the stored A and B skills, and each skill contributes to behavior in proportion to its relative contextual match.**

That directly extends the previous paper, where context acted as an organizing signal that split the same sequence into context-specific latent states.  The new question is how those already-split states are retrieved when the current context does not uniquely identify either one.

# PRD: Similarity-Weighted Retrieval of Split Motor Skills

## 1. Project summary

### Prior result

Previous work established that the same motor sequence can be represented as separate context-specific skills:

$$
S_A,\qquad S_B.
$$

Humans performing the sequence under alternating Maze and Melody contexts were better explained by split context-specific learning states, and the corresponding SRN work showed that strong context signals combined with sufficient representational flexibility produced separated context-specific hidden representations.  

### New behavioral result

Two subsequent experiments present contexts that do not correspond uniquely to either learned context:

1. **Combined context:** both learned contextual cues are present.
2. **Novel third context:** neither learned context is present; instead a new context is presented.

In both cases, human performance is qualitatively intermediate between the A and B skill trajectories.

Task-trained SRNs do **not** naturally reproduce this behavior.

This suggests that the missing computation may not be part of motor-sequence learning itself, but rather part of the **retrieval mechanism operating over already-separated skills**.

---

# 2. Primary scientific question

> **How does the learner determine which already-learned context-specific skill to express when the current context does not uniquely identify one of them?**

The project should distinguish:

$$
\boxed{\text{What is being averaged?}}
$$

and

$$
\boxed{\text{Why does that averaging occur?}}
$$

The primary hypothesis is that behavior reflects **graded retrieval responsibility over stored context-specific skills**.

---

# 3. Core computational hypothesis: similarity-weighted retrieval

Assume the learner has stored:

$$
(S_A,\mu_A)
$$

and

$$
(S_B,\mu_B),
$$

where:

* \(S_A,S_B\) are the learned skill states;
* \(\mu_A,\mu_B\) are the contextual representations associated with those skills.

The current context is represented by:

$$
x_t.
$$

Compute contextual distance:

$$
d_A(t)=\|x_t-\mu_A\|^2
$$

$$
d_B(t)=\|x_t-\mu_B\|^2.
$$

Convert those distances into retrieval evidence:

$$
e_A(t)=\exp[-\beta d_A(t)]
$$

$$
e_B(t)=\exp[-\beta d_B(t)].
$$

Retrieval responsibility is:

$$
q_A(t)
=
\frac{e_A(t)}
{e_A(t)+e_B(t)}
$$

and

$$
q_B(t)=1-q_A(t).
$$

Expressed performance is then:

$$
\boxed{
S_{\mathrm{expressed}}(t)
=
q_A(t)S_A(t)
+
q_B(t)S_B(t)
}
$$

with the behavioral readout predicting log RT.

### Interpretation

The system does not need to:

* create a third motor representation;
* literally average neural representations;
* choose A or B categorically.

Instead:

> **The context determines responsibility over the two existing skill memories, and motor behavior reflects their responsibility-weighted readout.**

---

# 4. Why Combined and Novel can look similar

Use a minimal task-derived contextual feature representation.

For example:

$$
A=[1,0,0]
$$

$$
B=[0,1,0]
$$

$$
Both=[1,1,0]
$$

$$
Novel=[0,0,1].
$$

The exact feature names should reflect the actual experimental cues, but the encoding must be determined from the **task design**, not from human probe RT.

Then:

### Combined context

$$
d(Both,A)=d(Both,B)
$$

so:

$$
q_A=q_B=.5.
$$

### Novel context

$$
d(Novel,A)=d(Novel,B)
$$

again giving:

$$
q_A=q_B=.5.
$$

Thus both contexts can produce similar **relative responsibility** even though their absolute relationship to the learned contexts is very different.

This motivates separating:

### Relative contextual evidence

$$
R_t=d_B(t)-d_A(t)
$$

which determines **which memory is retrieved**,

from:

### Absolute contextual evidence

$$
E_t=e_A(t)+e_B(t)
$$

which measures **how well the current context matches the learned context set at all**.

The central hypothesis is:

> Mean motor expression primarily depends on **relative evidence**, whereas absolute evidence may affect uncertainty, variability, or an additional novelty cost.

---

# 5. Intended contribution

The conceptual progression from the previous project should be:

### Previous paper

$$
\boxed{
\text{Context determines whether identical motor experience is lumped or split.}
}
$$

Context can organize the same task into distinct context-specific representations. 

### New paper

$$
\boxed{
\text{Once skills are split, contextual evidence determines their graded retrieval.}
}
$$

When the present context does not uniquely identify one learned skill:

$$
\text{context}
\rightarrow
\text{retrieval responsibility}
\rightarrow
\text{weighted expression of existing skills}.
$$

This is the target mechanism.

---

# 6. Primary competing mechanisms

The project should discriminate these mechanisms.

## H1. Continuous responsibility-weighted retrieval

Each probe sequence expresses an actual interpolation:

$$
y_t
=
q_ty_A(t)+(1-q_t)y_B(t).
$$

This is the primary hypothesis.

---

## H2. Categorical probabilistic retrieval

The learner retrieves either A or B on each individual sequence:

$$
z_t\sim Bernoulli(q_t)
$$

and

$$
y_t=
\begin{cases}
y_A(t), & z_t=A\\
y_B(t), & z_t=B.
\end{cases}
$$

Across participants/trials the mean can be intermediate even though **no individual retrieval state is intermediate**.

This distinction is essential.

---

## H3. Fixed context-specific mixture

Each probe context simply has a fitted constant:

$$
q_{Both},
\qquad
q_{Novel}.
$$

No contextual-computation mechanism generates those weights.

This is the descriptive baseline.

---

## H4. History/Bayesian responsibility

The current context provides likelihood evidence, but recent context establishes a prior.

Let:

$$
\pi_t
$$

be prior log-odds favoring A.

Then:

$$
\log\frac{q_A}{q_B}
=
-\beta[d_A-d_B]
+\pi_t.
$$

The prior evolves:

$$
\pi_{t+1}
=
\rho\pi_t+\eta r_t
$$

where:

$$
r_t=
\begin{cases}
+1 & A\\
-1 & B\\
0 & probe.
\end{cases}
$$

This gives a clean interpretation of history:

> history changes the **prior over skill retrieval**, while the present context provides the likelihood.

---

## H5. Absolute-evidence/default retrieval

When:

$$
E_t
$$

is very low, the learner may partly fall back to a context-general/default state:

$$
S_0.
$$

This mechanism is particularly relevant to the Novel condition.

It should be treated as an extension only if the two-skill retrieval model systematically fails on Novel.

---

## H6. Third-state representation

Probe experience creates:

$$
S_C.
$$

Retain this as a falsification model because the existing analyses already suggest it is unlikely to be the primary explanation.

---

# 7. Strong distinction: continuous averaging versus stochastic switching

The mean alone cannot distinguish H1 from H2.

Therefore the primary likelihood comparison should operate on **individual sequence RTs**, not only participant or group means.

## Continuous model

$$
y_t\sim
\mathcal N
\left(
q_ty_A+(1-q_t)y_B,
\sigma^2
\right).
$$

## Categorical model

$$
y_t\sim
q_t\mathcal N(y_A,\sigma^2)
+
(1-q_t)\mathcal N(y_B,\sigma^2).
$$

### Predictions

Continuous retrieval predicts:

* unimodal probe distributions centered between A/B;
* variance broadly comparable to ordinary residual variability.

Categorical retrieval predicts:

* increased variance when \(q\approx.5\);
* potentially bimodal distributions;
* individual sequences clustering around A-like or B-like performance.

This is one of the most important new tests.

---

# 8. Stage 1: estimate the existing skills

Reuse the current split A/B state-space pipeline.

Estimate:

$$
S_A(t)
$$

and

$$
S_B(t)
$$

from A/B data only.

The retrieval model must **never alter these states in the primary analysis**.

This implements the scientific premise:

> A and B are already learned skills; the new question is how they are retrieved.

Reuse:

* sequence preprocessing;
* participant exclusions;
* nine-sequence block structure;
* fatigue modeling;
* A/B-only counterfactual fitting;
* grouped participant splits;
* control participants when added.

Do not reopen lumped-versus-split model selection.

---

# 9. Stage 2: task-derived context representation

Create:

```text
context_features.py
```

The primary context vectors must be determined entirely from the experimental design.

For the minimal implementation:

```text
A:
    A_cue = 1
    B_cue = 0
    novel_cue = 0

B:
    A_cue = 0
    B_cue = 1
    novel_cue = 0

Both:
    A_cue = 1
    B_cue = 1
    novel_cue = 0

Novel:
    A_cue = 0
    B_cue = 0
    novel_cue = 1
```

If the actual experimental context contains multiple cue dimensions, encode those explicitly rather than reducing them prematurely.

### Hard constraint

No feature representation may use:

* probe RT;
* estimated probe \(q\);
* model residuals;
* probe condition performance.

The contextual geometry must be specified **before fitting retrieval behavior**.

---

# 10. Learned context prototypes

The stored prototypes are:

$$
\mu_A
=
E[x\mid A]
$$

$$
\mu_B
=
E[x\mid B].
$$

Because the experimental contexts are stable, these may simply equal the A and B cue vectors.

Do not fit free probe prototypes in the primary model.

An optional later model can learn a low-dimensional transform:

$$
z=W_xx
$$

but \(W_x\) must be trained using A/B behavior or shared across both probe datasets.

It may not receive free condition-specific embeddings that simply memorize Both and Novel.

---

# 11. Explicit model ladder

Implement these in order.

## M0 — Equal mixture

$$
q=.5.
$$

---

## M1 — Free static probe mixture

Fit:

$$
q_{Both}
$$

and:

$$
q_{Novel}.
$$

This is the descriptive benchmark.

---

## M2 — Similarity-weighted retrieval

Shared:

$$
\beta
$$

controls the sharpness of contextual retrieval.

$$
q_A=
softmax(-\beta d_A,-\beta d_B).
$$

No history.

No free Both/Novel retrieval parameter.

This is the primary mechanistic model.

---

## M3 — Similarity retrieval + history prior

Parameters:

$$
\beta,\rho,\eta.
$$

Current cue evidence and prior history jointly determine responsibility.

---

## M4 — Similarity retrieval + absolute evidence

Keep:

$$
q
$$

determined by relative evidence.

Allow absolute evidence:

$$
E
$$

to affect either:

### variance

$$
\log \sigma_t
=
\sigma_0-\kappa E_t
$$

or, secondarily,

### mean

$$
y_t
=
q_Ay_A+q_By_B+
\gamma f(E_t).
$$

Variance modulation should be tested before adding a mean offset.

This model asks whether Novel differs from Both because it is globally less familiar despite having similar A/B responsibility.

---

## M5 — Categorical retrieval

Use the same similarity-generated \(q\), but a mixture likelihood rather than continuous interpolation.

---

## M6 — Default/unknown-state extension

Only implement if M2–M5 systematically fail on Novel.

---

# 12. Joint fitting across both datasets

This is mandatory.

Do **not** fit an independent mechanism to Combined and Novel and then compare parameters afterward.

The central scientific claim is:

> **One retrieval computation explains both datasets.**

Therefore:

### Shared parameters

Share across datasets:

```text
beta
history retention rho
history gain eta
absolute-evidence coefficient kappa
```

where applicable.

Allow dataset-specific:

* residual variance;
* participant intercept nuisance parameters;
* A/B skill trajectories.

---

# 13. Cross-dataset generalization

This is the strongest test of the hypothesis.

## Test A

Fit retrieval mechanism on:

```text
Combined
```

using Combined probe behavior.

Then:

1. estimate A/B skill trajectories for Novel participants;
2. construct the Novel context vector from task design;
3. predict Novel probe behavior **without fitting Novel retrieval parameters**.

---

## Test B

Reverse:

```text
Novel → Combined.
```

### Gate

The shared-retrieval hypothesis is not supported unless the mechanism transfers meaningfully across datasets.

This should be treated as more important than within-dataset BIC differences.

---

# 14. Participant-level structural prediction

A retrieval mechanism makes a strong prediction across participants.

For participant \(i\):

$$
y_{Probe,i}
=
q_i y_{A,i}
+
(1-q_i)y_{B,i}.
$$

Therefore:

$$
y_{Probe,i}
-
\frac{y_{A,i}+y_{B,i}}{2}
=
\left(q_i-\frac12\right)
(y_{A,i}-y_{B,i}).
$$

This means the behavioral consequence of retrieval ambiguity should scale with the participant's own:

$$
A-B
$$

skill separation.

### Critical test

Participants with nearly identical A/B skill states should show little probe displacement even if retrieval is ambiguous.

Participants with large A/B separation should show larger probe displacement.

A common retrieval mechanism should predict these individual differences **without fitting a separate probe weight to every participant**.

This should be a headline analysis.

---

# 15. Relative versus absolute contextual evidence

Joint analysis of Both and Novel should explicitly estimate:

$$
R=d_B-d_A
$$

and:

$$
E=e_A+e_B.
$$

The cleanest theoretical result would be:

### Both

$$
R\approx0,\quad E=\text{high}.
$$

### Novel

$$
R\approx0,\quad E=\text{low}.
$$

while:

$$
q_A\approx.5
$$

in both.

If mean behavior is largely determined by \(R\), this explains why the two conditions look similar.

If \(E\) affects residual variance or novelty cost, the two conditions can differ in secondary ways without requiring different retrieval mechanisms.

---

# 16. Tiny neural gate: discovery model

Do not use the old task-trained SRN as the primary neural model.

Instead implement an explicit **mixture-of-experts retrieval architecture**.

```text
                 Context features
                       ↓
                Tiny gate network
                       ↓
                     q(A)
                    ↙    ↘
             frozen A    frozen B
             skill state skill state
                    ↘    ↙
                weighted readout
                       ↓
                    log RT
```

The experts are:

$$
S_A(t),S_B(t).
$$

They are fixed.

Only the retrieval gate is learned.

---

# 17. Neural gate ladder

Fit:

```text
logistic gate
1D recurrent gate
2D recurrent gate
```

Do not escalate beyond 2D unless clearly necessary.

Inputs:

```text
A cue present
B cue present
novel cue present
optional previous-context state
```

The network must be shared across Both and Novel datasets.

---

# 18. Neural-model purpose

The neural gate answers:

> Is the simplest explicit similarity/Bayesian model missing another low-dimensional computation?

After fitting, distill:

$$
\operatorname{logit}q
=
\beta_RR+
\beta_EE+
\lambda h+
b.
$$

If this captures the neural gate, discard the neural model as the explanation and retain the explicit equation.

### Success criterion

The desired end product is **not a trained neural network**.

It is an interpretable retrieval equation.

---

# 19. Original SRN as a negative control

Retain the original task-trained SRN result as an important control.

The prior SRN architecture learns the sequence and context-conditioned representations but does not produce the human probe averaging.

This supports the claim that:

> **context-dependent representation learning is insufficient; an additional graded retrieval computation is required.**

Do not alter the SRN until it reproduces the desired result.

Its failure is theoretically informative.

---

# 20. Model recovery

Recovery must use the actual Combined and Novel experimental schedules.

Generate from:

```text
continuous_similarity
categorical_similarity
similarity_plus_history
similarity_plus_evidence
free_static_mixture
default_state
```

Fit every model to every synthetic dataset.

Run:

* easy regime;
* human residual-noise regime;
* fitted human-parameter neighborhood.

### Key recovery question

Can the data distinguish:

$$
\text{continuous interpolation}
$$

from:

$$
\text{stochastic A/B switching}?
$$

If not, explicitly limit the mechanistic claim to graded **expected responsibility** rather than within-trial representational averaging.

---

# 21. TDD requirement

All development follows:

$$
\boxed{\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}}
$$

Tests must include both software correctness and scientific contracts.

---

# 22. RED → GREEN 1: context feature contract

## RED

Write tests requiring:

```text
A     = [1,0,0]
B     = [0,1,0]
Both  = [1,1,0]
Novel = [0,0,1]
```

or the finalized task-specific equivalent.

Require:

$$
d(Both,A)=d(Both,B)
$$

and:

$$
d(Novel,A)=d(Novel,B).
$$

Require:

$$
d(Novel,A)>d(Both,A)
$$

for the minimal three-feature encoding.

## GREEN

Implement `context_features.py`.

---

# 23. RED → GREEN 2: similarity retrieval

Use:

$$
\beta>0.
$$

## RED

Require:

```text
A context → qA > .5
B context → qA < .5
symmetric Both → qA == .5
symmetric Novel → qA == .5
```

As:

$$
\beta\rightarrow\infty,
$$

pure A/B retrieval should approach categorical selection.

## GREEN

Implement kernel retrieval.

---

# 24. RED → GREEN 3: symmetry

Swap:

```text
A ↔ B
```

through the entire synthetic dataset.

## RED

Require:

$$
q_A^{swapped}
=
1-q_A^{original}.
$$

Predicted RTs must mirror accordingly.

## GREEN

Fix all context-label assumptions until symmetry passes.

This test remains permanently enabled.

---

# 25. RED → GREEN 4: continuous readout

Synthetic states:

```text
A = 400
B = 600
qA = .25
```

## RED

Require continuous prediction:

```text
550
```

Probe readout must not modify either expert.

## GREEN

Implement continuous retrieval.

---

# 26. RED → GREEN 5: categorical retrieval

Generate:

```text
A = 400
B = 600
qA = .50
sigma = small
```

## RED

Continuous model should predict a unimodal distribution near 500.

Categorical model should predict two modes near 400 and 600.

The fitting procedure must prefer the true generator with sufficient data.

## GREEN

Implement mixture likelihood.

---

# 27. RED → GREEN 6: history as a prior

Create two otherwise identical ambiguous probe trials.

One preceded by A, one by B.

## RED

With:

$$
\eta>0,
$$

require:

$$
q_A(Probe|previousA)
>
q_A(Probe|previousB).
$$

Setting:

$$
\eta=0
$$

must eliminate this difference.

## GREEN

Implement history-prior model.

---

# 28. RED → GREEN 7: relative versus absolute evidence

Construct Both and Novel with:

$$
R_{Both}=R_{Novel}=0
$$

but:

$$
E_{Both}>E_{Novel}.
$$

## RED

Base similarity model:

```text
mean q is equal
```

Evidence-strength model:

```text
predicted residual variance differs
```

## GREEN

Implement total-evidence term.

---

# 29. RED → GREEN 8: participant-level A/B separation

Generate participants with identical \(q\) but differing:

$$
|A-B|.
$$

## RED

Require probe displacement from midpoint to scale linearly with A/B separation.

When:

$$
A=B,
$$

probe displacement must be zero regardless of \(q\).

## GREEN

Implement structural diagnostic.

---

# 30. RED → GREEN 9: cross-dataset generalization

Generate Combined and Novel data using one common retrieval rule.

## RED

Fit on Combined.

Without fitting any Novel retrieval parameter, require accurate Novel prediction.

Reverse direction and repeat.

## GREEN

Implement shared-mechanism fitting interface.

---

# 31. RED → GREEN 10: tiny gate distillation

Generate data from known:

$$
\operatorname{logit}q
=
\beta_RR+\lambda h.
$$

Train 1D gate.

## RED

Distillation must recover:

* correct coefficient signs;
* high held-out \(R^2\);
* no spurious evidence-strength coefficient when none exists.

## GREEN

Implement tiny gate and distillation.

---

# 32. Proposed code structure

```text
retrieval_modeling/
    __init__.py

    data.py
    skill_states.py

    context_features.py
    context_geometry.py

    retrieval_weights.py
    continuous_readout.py
    categorical_readout.py
    history_prior.py
    evidence_strength.py

    explicit_models.py

    tiny_gate.py
    distillation.py

    fitting.py
    cross_dataset.py
    recovery.py
    diagnostics.py
    plotting.py
    pipeline.py

tests/
    test_context_features.py
    test_context_geometry.py
    test_similarity_retrieval.py
    test_symmetry.py
    test_continuous_readout.py
    test_categorical_retrieval.py
    test_history_prior.py
    test_evidence_strength.py
    test_AB_separation_prediction.py
    test_cross_dataset_generalization.py
    test_tiny_gate.py
    test_model_recovery.py
    test_no_probe_leakage.py
```

---

# 33. Existing code to reuse

Reuse from the current PRD2 pipeline:

* cleaned data loader;
* correct-sequence processing;
* participant exclusions;
* A/B skill-state estimation;
* participant splits;
* probe epoch labels;
* counterbalance coding;
* held-out evaluation;
* model-recovery framework;
* plotting infrastructure.

The current analysis already protects foundational A/B fitting from probe leakage and treats counterbalance/history explicitly. 

---

# 34. Existing code to demote

Do not make these the central models anymore:

```text
gate_adaptation
responsibility_update
third_state
```

Retain them as falsification/comparison models.

The new mechanistic competition is primarily:

```text
free static q
vs.
similarity responsibility
vs.
similarity + history
vs.
continuous vs categorical readout
```

---

# 35. Required outputs

```text
context_feature_table.csv
context_geometry.csv

retrieval_model_fits.csv
retrieval_predictions.csv
retrieval_responsibilities.csv

relative_absolute_evidence.csv

continuous_vs_categorical.csv

participant_AB_separation_test.csv

combined_to_novel_transfer.csv
novel_to_combined_transfer.csv

tiny_gate_fits.csv
tiny_gate_states.csv
distilled_retrieval_rule.csv

retrieval_model_recovery.csv
retrieval_recovery_confusion.csv

run_config_retrieval.json
```

---

# 36. Required figures

## Figure 1 — Conceptual model

```text
context
   ↓
similarity to A/B retrieval contexts
   ↓
responsibility qA / qB
   ↓
A skill + B skill
   ↓
expressed RT
```

---

## Figure 2 — Context geometry

Show A, B, Both and Novel in task-derived context-feature space.

Annotate:

$$
R
$$

and:

$$
E.
$$

---

## Figure 3 — Human readout geometry

For Both and Novel:

* current A counterfactual;
* current B counterfactual;
* observed probe behavior;
* similarity-model prediction.

---

## Figure 4 — Participant structural prediction

X-axis:

$$
A-B\text{ skill separation}.
$$

Y-axis:

$$
Probe-Midpoint.
$$

Overlay predicted relationship from common retrieval \(q\).

---

## Figure 5 — Continuous versus categorical retrieval

Observed probe RT distribution versus:

* continuous interpolation likelihood;
* stochastic A/B mixture likelihood.

---

## Figure 6 — Cross-dataset transfer

Show:

```text
fit Combined → predict Novel
fit Novel → predict Combined
```

alongside within-dataset performance.

---

## Figure 7 — Relative versus absolute evidence

Plot Both and Novel on:

$$
R\times E
$$

space.

Show whether mean RT tracks \(R\), while variance/novelty effects track \(E\).

---

## Figure 8 — Tiny-gate distillation

Neural gate output versus explicit similarity/Bayesian rule.

---

# 37. Primary decision logic

## Result A

Similarity model predicts Both and Novel with the same parameters, including cross-dataset transfer.

### Interpretation

> Context-dependent skill retrieval is governed by relative similarity to the contextual conditions under which the skills were learned.

---

## Result B

History-prior model substantially improves transfer.

### Interpretation

> Contextual similarity supplies the likelihood, while recent context supplies a prior over which skill should be retrieved.

This is the strongest Bayesian/latent-cause version of the account.

---

## Result C

Continuous readout beats categorical retrieval.

### Interpretation

> Multiple context-specific skills contribute simultaneously to individual motor expression.

This would justify strong language about **graded recombination**.

---

## Result D

Categorical mixture fits equally well or better.

### Interpretation

> Context determines graded retrieval probabilities, but individual skill expression may remain categorical.

Do not claim simultaneous representational averaging.

---

## Result E

Both and Novel have similar \(R\), but different \(E\), and \(E\) predicts variability rather than mean expression.

### Interpretation

> Relative contextual evidence determines which stored skill contributes to behavior, while absolute familiarity determines retrieval confidence.

This would be an especially clean result.

---

## Result F

A free \(q_{Both},q_{Novel}\) model substantially outperforms the shared similarity rule and fails cross-dataset transfer.

### Interpretation

The proposed common retrieval mechanism is insufficient.

Do not relabel condition-specific weights as “similarity.”

---

# 38. Key falsification criterion

The project should **not** claim a general retrieval mechanism simply because:

$$
Both\approx midpoint
$$

and:

$$
Novel\approx midpoint.
$$

The mechanism passes only if a common rule predicts:

1. individual probe means;
2. differences generated by each participant's A/B skill separation;
3. Combined and Novel data jointly;
4. held-out participants;
5. cross-dataset generalization;
6. the shape of the probe RT distribution.

---

# 39. Definition of done

* [ ] Current A/B skill pipeline is frozen and reused.
* [ ] No probe behavior enters A/B skill estimation.
* [ ] Context features are defined from task structure only.
* [ ] A/B prototypes are learned/defined without probe RT.
* [ ] Equal and free-mixture baselines are fit.
* [ ] Similarity-weighted retrieval is implemented.
* [ ] History-prior retrieval is implemented.
* [ ] Relative and absolute evidence are separately quantified.
* [ ] Continuous and categorical retrieval are compared.
* [ ] Participant A/B separation prediction is tested.
* [ ] Combined and Novel are fit jointly with shared mechanism parameters.
* [ ] Combined→Novel zero-shot transfer is evaluated.
* [ ] Novel→Combined zero-shot transfer is evaluated.
* [ ] Tiny 1D retrieval gate is fit as a discovery model.
* [ ] Neural gate is distilled into an explicit equation.
* [ ] Original task-trained SRN is retained as a negative control.
* [ ] Model recovery is run at realistic noise and human-like parameters.
* [ ] Every feature is implemented through RED → GREEN → REFACTOR.
* [ ] Mechanistic claims are restricted to what model recovery and cross-dataset transfer identify.

# 40. Target paper-level claim

The strongest successful conclusion would be:

> **Context does two computational jobs in motor skill learning. During acquisition, context can partition otherwise identical experience into separate skill representations. During retrieval, the current context assigns graded responsibility across those representations according to its relative match to their learned contextual cues. When contextual evidence fails to uniquely favor one stored skill, behavior reflects a weighted readout of the existing memories.**

Or in the most compact form:

$$
\boxed{
\text{Context splits skills during learning and recombines them through graded retrieval.}
}
$$

That gives you a much cleaner continuation of the original PNAS story than another comparison of behavioral learning-state models.
