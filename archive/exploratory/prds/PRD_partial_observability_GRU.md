# PRD: Emergence of Graded Skill Retrieval Under Partial Observability

## 1. Project objective

### Scientific question

Previous work established that context can cause an otherwise identical motor sequence to become represented as separate context-specific skills.

Subsequent human experiments show that when the context no longer uniquely identifies either learned skill—either because:

- cues from both learned contexts are present, or
- an entirely novel third context is presented—

behavior becomes intermediate between the two established skills.

The present project asks:

> **Why would a learning system that has split experience into context-specific skills subsequently average those skills when contextual identity becomes uncertain?**

The proposed answer is:

> **A recurrent learner operating under partial observability learns to infer the currently active latent context. When it must make a continuous prediction and contextual evidence leaves that state uncertain, minimizing prediction error favors the posterior-weighted expectation over the candidate skill states.**

The project should determine whether this mechanism **emerges spontaneously** in generic recurrent networks rather than being built into their architecture.

---

## 2. Core hypothesis

The environment contains a hidden context:

\[
z_t\in\{A,B\}.
\]

Each context has its own evolving latent skill:

\[
S_A(t),\qquad S_B(t).
\]

The learner does not directly observe \(z_t\).

Instead it receives contextual evidence:

\[
x_t\sim P(x\mid z_t).
\]

Its task is to predict current performance:

\[
y_t=S_{z_t}(t)+\epsilon_t.
\]

Under ambiguous contextual evidence:

\[
0<P(A\mid x_{\le t})<1.
\]

For a deterministic predictor optimized under squared error, the optimal prediction is:

\[
\boxed{
\hat y_t=
P(A\mid x_{\le t})S_A(t)
+
P(B\mid x_{\le t})S_B(t)
}
\]

The central hypothesis is therefore:

\[
\boxed{
\text{partial observability}
\rightarrow
\text{latent-context uncertainty}
\rightarrow
\text{posterior-weighted skill expression}
}
\]

---

## 3. Why this is different from the previous SRN

The previous SRN was optimized to predict the next item of the motor sequence.

Because A and B contained the exact same sequence:

\[
P(k_{t+1}\mid A)=P(k_{t+1}\mid B).
\]

Therefore correctly identifying whether A or B was active provided essentially no benefit for next-key prediction.

The previous architecture could still develop split representations because context served as an organizing feature.

But the objective provided no pressure for the network to solve:

> “Which one of my differently learned A/B skill states should currently determine performance?”

The new task explicitly creates that pressure.

---

## 4. Primary computational claim

### Weak claim

A GRU trained on uncertain contexts can reproduce intermediate outputs.

This is insufficient.

### Strong claim

A generic GRU trained to track independently evolving context-specific states under partial observability spontaneously develops an internal representation equivalent to:

\[
\{S_A,S_B,P(A)\}
\]

and its continuous output approximates:

\[
P(A)S_A+[1-P(A)]S_B.
\]

Furthermore, this algorithm should emerge **because it reduces prediction loss**, not because weighted averaging is wired into the architecture.

---

## 5. Synthetic meta-learning environment

Do not initially train on human Mixed or Novel RTs.

Generate synthetic episodes containing:

1. two latent contexts;
2. two independently evolving skill states;
3. noisy contextual observations;
4. context switches;
5. observed performance feedback.

The exact context features and skill values should vary between episodes so the network cannot memorize fixed A/B values.

---

## 6. Latent skill dynamics

At episode initialization:

\[
S_A(0)\sim P(S_0),\qquad S_B(0)\sim P(S_0).
\]

For the MVP, use:

\[
S_z(t+1)=S_z(t)+\alpha_z[\theta_z-S_z(t)]+\xi_t.
\]

When A is practiced, update \(S_A\) and leave \(S_B\) unchanged; vice versa for B.

Randomize across episodes:

```text
initial skill values
learning rates
asymptotes
A/B skill separation
context order
switch frequency
```

The goal is to create a minimal environment requiring the learner to separately track two context-dependent latent values.

---

## 7. Context-generation process

Each episode has two context prototypes:

\[
\mu_A,\qquad\mu_B.
\]

These should vary across episodes.

For example:

\[
\mu_A,\mu_B\sim\mathcal N(0,I)
\]

subject to a minimum separation.

Context observation:

\[
x_t=\mu_{z_t}+\epsilon_t
\]

with:

\[
\epsilon_t\sim\mathcal N(0,\sigma_c^2I).
\]

This prevents the network from simply learning fixed feature identities for A and B.

---

## 8. Partial observability manipulation

This is the main causal manipulation.

### U0 — Perfect contexts
\[
\sigma_c=0.
\]

### U1 — Low uncertainty
Context observations are mildly noisy.

### U2 — Moderate uncertainty
The A/B likelihood distributions overlap meaningfully.

### U3 — High uncertainty
Context evidence is frequently ambiguous.

### Main prediction

\[
\boxed{
\text{training uncertainty}
\uparrow
\Rightarrow
\text{graded probe interpolation}
\uparrow
}
\]

provided the network still solves the task.

---

## 9. Crucial control: uncertainty must occur during learning

Train one group only on perfectly informative A/B contexts.

Train another group with occasional uncertainty during training.

At test, show both groups the same ambiguous contexts.

If averaging is specifically an adaptation to partial observability:

\[
Interpolation_{uncertainty-trained}
>
Interpolation_{perfect-trained}.
\]

---

## 10. Skill-separation manipulation

Manipulate:

\[
\Delta S=|S_A-S_B|.
\]

Use:

```text
small separation
medium separation
large separation
```

Test:

\[
ContextUncertainty\times SkillSeparation.
\]

The idea is that context inference should matter more when selecting the wrong latent skill state has a larger predictive cost.

---

## 11. Why averaging versus coin-flipping must be experimentally manipulated

A deterministic GRU optimized with MSE is expected to learn the conditional mean.

Therefore the project must include competing output objectives. Otherwise finding averaging would be partly guaranteed by the loss function.

This is mandatory.

---

## 12. Objective O1: deterministic MSE

Architecture:

\[
h_t=GRU(x_t,h_{t-1})
\]

\[
\hat y_t=Wh_t+b.
\]

Loss:

\[
L=(y_t-\hat y_t)^2.
\]

Normative optimum:

\[
\hat y_t=E[y_t\mid x_{\le t}].
\]

Prediction: ambiguity should produce continuous averaging.

---

## 13. Objective O2: Gaussian probabilistic prediction

Network outputs:

\[
\mu_t,\sigma_t.
\]

Optimize:

\[
-\log P(y_t\mid\mu_t,\sigma_t).
\]

Prediction: the mean should approximate the posterior-weighted expectation, while uncertainty may also appear as increased \(\sigma_t\).

---

## 14. Objective O3: latent-state classification

Train a comparable GRU to predict:

\[
P(z_t=A).
\]

This serves as a reference model for the context-inference problem.

Compare its posterior with any belief signal emerging in the performance-prediction GRU.

---

## 15. Objective O4: stochastic/sample-based prediction

Allow the model to sample one latent skill:

\[
z'_t\sim Bernoulli(q_t)
\]

and

\[
\hat y_t=S_{z'_t}.
\]

Compare expected loss with the continuous model.

For two possible states under MSE:

\[
R_{sample}=2q(1-q)\Delta S^2
\]

whereas:

\[
R_{mean}=q(1-q)\Delta S^2.
\]

Thus posterior sampling has approximately double the uncertainty-related squared-error cost.

This directly tests why a learner would average instead of coin-flip.

---

## 16. Objective O5: distributional/log-score prediction

Allow the network to output the full bimodal distribution:

\[
P(y)=qP(y\mid A)+(1-q)P(y\mid B).
\]

This separates:

- **belief representation**: retaining both possible states;
- **behavioral readout**: converting uncertainty into one continuous output.

---

## 17. Core architecture

Primary model:

```text
Context observation x_t
Previous performance/feedback
Optional block/switch signal
            ↓
        Tiny GRU
            ↓
      Linear readout
            ↓
      Predicted state y_t
```

Do not include:

- explicit A/B experts;
- mixture weights;
- posterior variables;
- hard-coded averaging;
- attention over A/B;
- mixture-of-experts architecture.

The network must discover the solution itself.

---

## 18. Inputs

At prediction time \(t\), provide:

```text
current context observation x_t
previous observed performance y_(t-1)
previous action/practice indicator if needed
episode boundary
```

Do not provide:

```text
true latent context z_t
true S_A
true S_B
posterior P(A)
```

Those exist only for evaluation.

---

## 19. Output timing

The network must predict \(y_t\) **before seeing \(y_t\)**.

After prediction, the true \(y_t\) may become available as feedback for the next timestep.

This prevents target leakage.

---

## 20. GRU dimensionality ladder

Fit:

```text
hidden_dim = 1
hidden_dim = 2
hidden_dim = 3
hidden_dim = 4
hidden_dim = 8
hidden_dim = 16 predictive ceiling
```

Use many random seeds.

The theoretically interesting possibility is that approximately three latent quantities suffice:

\[
S_A,\quad S_B,\quad q_A.
\]

Do not assume these dimensions are axis-aligned.

---

## 21. Zero-shot test environments

The key human-like probes must **never occur exactly during training**.

### Clear A
Expected:
\[
\hat y\approx S_A.
\]

### Clear B
Expected:
\[
\hat y\approx S_B.
\]

### Midpoint ambiguity
Generate an observation exactly equally diagnostic of A and B.

Expected:
\[
q_A\approx.5
\]
and
\[
\hat y\approx.5S_A+.5S_B.
\]

### Combined evidence
Construct an OOD observation containing evidence associated with both learned prototypes.

Do not prescribe the expected output during training.

### Novel evidence
Present an observation outside both training distributions.

Ask whether the network averages, extrapolates, selects one, becomes unstable, or expresses uncertainty.

---

## 22. Bayesian ground-truth benchmark

Because the synthetic generative process is known, calculate the exact posterior:

\[
q_t^*=P(z_t=A\mid x_{1:t},y_{1:t-1}).
\]

Also calculate:

\[
y_t^*=q_t^*S_A(t)+(1-q_t^*)S_B(t).
\]

The GRU is never trained on \(q_t^*\). It is only evaluated against it.

---

## 23. Behavioral metrics

For every test observation compute:

### Prediction error
\[
|\hat y-y|.
\]

### Posterior-mean error
\[
|\hat y-y^*|.
\]

### Implied responsibility
When \(S_A\neq S_B\):

\[
\hat q=\frac{\hat y-S_B}{S_A-S_B}.
\]

Compare \(\hat q\) with \(q^*\).

---

## 24. Primary emergence metrics

\[
PosteriorMeanAlignment=R^2(\hat y,y^*)
\]

and:

\[
BeliefAlignment=R^2(\hat q,q^*).
\]

A successful network should show strong alignment without explicit supervision on \(q\).

---

## 25. Internal-state analysis

After training, freeze the network.

Build diagnostic datasets independently manipulating:

```text
S_A
S_B
q*
current observation
previous context
time since context switch
```

Train held-out linear probes on hidden states for:

\[
S_A,\qquad S_B,\qquad q^*.
\]

---

## 26. Representational hypothesis

A successful minimal solution should contain enough information to recover:

\[
S_A,\quad S_B,\quad q.
\]

The strongest result would be:

> Networks trained under uncertainty develop a decodable belief variable, whereas otherwise matched perfect-context networks do not.

---

## 27. Dynamical distillation

For the smallest successful network, approximate its dynamics with explicit equations.

Candidate form:

\[
\hat S_{A,t+1}=f_A(\hat S_{A,t},y_t,x_t)
\]

\[
\hat S_{B,t+1}=f_B(\hat S_{B,t},y_t,x_t)
\]

\[
\operatorname{logit}q_{t+1}
=
\rho\operatorname{logit}q_t
+
\beta\ell(x_t).
\]

Behavior:

\[
\hat y_t
\approx
q_t\hat S_{A,t}
+
(1-q_t)\hat S_{B,t}.
\]

The endpoint is the explicit cognitive algorithm, not the GRU itself.

---

## 28. Causal intervention: belief variable

Identify the direction/subspace associated with \(q^*\).

At matched \(S_A,S_B,x_t\), push the hidden state toward:

```text
A certainty
uncertainty
B certainty
```

Prediction:

\[
S_A
\rightarrow
\text{weighted midpoint}
\rightarrow
S_B.
\]

This is required before claiming the belief signal causally controls expression.

---

## 29. Causal intervention: skill-state variables

Perturb decoded \(S_A\) while holding belief approximately fixed.

Prediction:

\[
\frac{\partial \hat y}{\partial S_A}\approx q_A
\]

and:

\[
\frac{\partial \hat y}{\partial S_B}\approx1-q_A.
\]

This is a strong signature of posterior-weighted readout.

---

## 30. Factorial computational experiment

Primary design:

\[
4\ ContextUncertainty
\times
3\ SkillSeparation
\times
5\ Objective
\times
6\ HiddenDimension.
\]

Run at least:

```text
20 seeds / cell
```

for final inference if computationally feasible.

A reduced pilot may use five seeds.

---

## 31. Main predicted interaction

The central result should not simply be “GRUs average.”

Under deterministic continuous prediction:

\[
ContextUncertainty\uparrow
\Rightarrow
BeliefRepresentation\uparrow
\]

and:

\[
ContextUncertainty\uparrow
\Rightarrow
PosteriorMeanAlignment\uparrow.
\]

This should become stronger when \(|S_A-S_B|\) is behaviorally consequential.

---

## 32. Human-data validation

Only after the synthetic mechanism is established should the human probes enter.

Do not train the synthetic GRU directly on human Mixed/Novel RT.

Instead recreate the human schedule qualitatively:

```text
learn A
learn B
alternate A/B
probe Combined
probe Novel
probe Default
```

Set synthetic A/B skill states to match estimated human A/B separation distributions.

Ask whether networks reproduce:

1. intermediate Combined behavior;
2. intermediate Novel behavior;
3. continuous rather than categorical expression;
4. magnitude of human interpolation;
5. history sensitivity.

---

## 33. Original SRN as critical negative control

Keep the original next-sequence-prediction SRN unchanged.

Prediction:

> It need not develop posterior-weighted skill retrieval because context identity is irrelevant to predicting the identical next keypress.

This creates an informative contrast:

### Sequence-prediction objective
Splitting can emerge, but uncertainty-sensitive averaging need not.

### Latent-performance-prediction objective
Splitting + belief-state inference + averaging can emerge.

---

## 34. Strongest possible mechanistic result

The ideal result would be:

1. Perfect-context GRUs learn separate skill states but do not systematically interpolate OOD.
2. Uncertainty-trained GRUs learn separate skill states **plus a belief-like variable**.
3. Under ambiguous contexts, outputs approximate the exact Bayesian posterior mean.
4. Sampling/categorical models perform worse under continuous prediction loss.
5. Manipulating the belief-like state causally shifts output between A and B.
6. The same networks qualitatively reproduce human Both and Novel behavior without training on those probe types.

Then the explanation becomes:

\[
\boxed{
\text{Averaging is not an arbitrary retrieval heuristic.}
}
\]

It emerges because:

\[
\boxed{
\text{the learner must produce one continuous behavior while uncertain which context-specific state is active.}
}
\]

---

## 35. TDD requirement

All implementation must follow:

\[
\boxed{RED\rightarrow GREEN\rightarrow REFACTOR}
\]

with scientific contracts tested alongside software behavior.

---

## 36. RED → GREEN 1: latent-state independence

### RED

Create episode:

```text
A active
A active
B active
```

Require:

- A practice changes \(S_A\);
- A practice does not change \(S_B\);
- B practice changes \(S_B\);
- true states are reproducible from seed.

### GREEN

Implement `latent_skills.py`.

---

## 37. RED → GREEN 2: context likelihood

### RED

With low noise:

```text
A samples closer to mu_A
B samples closer to mu_B
```

With increasing uncertainty:

```text
classification Bayes accuracy decreases monotonically
```

### GREEN

Implement `context_generator.py`.

---

## 38. RED → GREEN 3: exact Bayesian posterior

### RED

For a symmetric ambiguous observation and equal prior:

\[
q^*=.5.
\]

For clear A evidence:
\[
q^*>.5.
\]

For clear B evidence:
\[
q^*<.5.
\]

### GREEN

Implement `bayes_filter.py`.

---

## 39. RED → GREEN 4: Bayes-risk mean versus sampling

### RED

Known:

\[
S_A=400,\quad S_B=600,\quad q=.5.
\]

Posterior mean:

\[
500.
\]

Expected MSE:

\[
10,000.
\]

Posterior-sampling expected MSE:

\[
20,000.
\]

Require:

\[
R_{sample}=2R_{mean}.
\]

Test additional \(q\) values.

### GREEN

Implement normative-risk utilities.

---

## 40. RED → GREEN 5: no target leakage

### RED

At time \(t\), change \(y_t\) after the input batch has been constructed.

Prediction input for \(t\) must remain unchanged.

### GREEN

Ensure only \(y_{<t}\) is available.

---

## 41. RED → GREEN 6: deterministic GRU can learn clear states

Train on low-noise synthetic data.

### RED acceptance test

Held-out:

\[
RMSE < predetermined baseline
\]

and clear A/B predictions approach their true states.

### GREEN

Implement training pipeline.

---

## 42. RED → GREEN 7: uncertainty produces interpolation

Synthetic generator with ambiguous observations.

### RED

After sufficient uncertainty training:

\[
\hat y_{ambiguous}
\]

must be closer to \(y^*\) than to either categorical endpoint on average.

### GREEN

Implement ambiguity evaluation.

---

## 43. RED → GREEN 8: stochastic control

Train/evaluate sample-based model.

### RED

Under MSE prediction:

\[
Risk_{continuous}<Risk_{sample}.
\]

Compare expected held-out risk over many episodes.

### GREEN

Implement stochastic head.

---

## 44. RED → GREEN 9: objective dependence

### RED

On ambiguous states:

- MSE model predicts conditional mean.
- Distributional model represents posterior mixture.
- Classification model predicts posterior context probability.

Each must recover its normative target on a tractable synthetic environment.

### GREEN

Implement objective family.

---

## 45. RED → GREEN 10: hidden belief decoding

Generate diagnostic held-out data.

### RED

A random/untrained network should not provide strong stable \(q^*\) decoding.

An uncertainty-trained successful GRU should.

### GREEN

Implement representation probes.

---

## 46. RED → GREEN 11: causal belief intervention

Use a synthetic network with a known belief dimension first.

### RED

Intervening:

```text
q low → output toward B
q=.5 → output midpoint
q high → output toward A
```

### GREEN

Implement activation intervention framework, then apply it to learned GRUs.

---

## 47. RED → GREEN 12: zero-shot human-style probes

Training generator must not contain exact:

```text
Both probe
Novel probe
human schedule
```

### RED

Test asserts those patterns are absent from training data.

### GREEN

Implement OOD probe suite.

---

## 48. Proposed code structure

```text
partial_observability_gru/
    __init__.py

    environment/
        latent_skills.py
        context_generator.py
        episode.py
        bayes_filter.py

    models/
        deterministic_gru.py
        gaussian_gru.py
        context_classifier_gru.py
        stochastic_gru.py
        distributional_gru.py

    training/
        losses.py
        trainer.py
        curriculum.py

    analysis/
        bayes_risk.py
        posterior_alignment.py
        hidden_probes.py
        interventions.py
        dimensionality.py
        distillation.py
        human_probes.py

    recovery/
        synthetic_recovery.py

    plotting/
        figures.py

tests/
    test_latent_skills.py
    test_context_generator.py
    test_bayes_filter.py
    test_bayes_risk.py
    test_no_leakage.py
    test_clear_context_learning.py
    test_ambiguity_interpolation.py
    test_stochastic_control.py
    test_objective_dependence.py
    test_belief_decoding.py
    test_belief_intervention.py
    test_zero_shot_probes.py
```

---

## 49. Required exports

```text
training_conditions.csv
network_performance.csv
ambiguity_behavior.csv

bayesian_ground_truth.csv
posterior_alignment.csv

dimensionality_comparison.csv
hidden_state_probes.csv
belief_interventions.csv
skill_interventions.csv

objective_comparison.csv
continuous_vs_sampling.csv

human_style_probe_predictions.csv

distilled_dynamics.csv
run_config.json
```

---

## 50. Required figures

### Figure 1 — Computational problem

```text
hidden context A/B
      ↓
noisy context observations
      ↓
GRU
      ↓
single continuous prediction
```

### Figure 2 — Normative solution

Plot true posterior \(q^*\) against Bayes-optimal continuous output:

\[
q^*S_A+(1-q^*)S_B.
\]

Overlay posterior-sampling outputs.

### Figure 3 — Emergence across uncertainty

X-axis: training context uncertainty.

Y-axis: posterior-mean alignment.

Separate objective conditions.

### Figure 4 — Uncertainty × skill separation

Heatmap of averaging/calibration.

### Figure 5 — Hidden representation

Decode \(S_A\), \(S_B\), and \(P(A)\) across network dimensionalities.

### Figure 6 — Belief causality

Intervention on inferred belief state versus predicted output.

### Figure 7 — Continuous versus sampling objective

Held-out risk under ambiguity.

### Figure 8 — Human-style zero-shot probes

Compare:

```text
Human
Perfect-context GRU
Uncertainty-trained GRU
Bayesian benchmark
```

for Both and Novel.

---

## 51. Decision logic

### Result A: uncertainty causes posterior averaging

If uncertainty-trained networks:

- develop belief-like states;
- track Bayesian posterior means;
- average on unseen ambiguous contexts;
- outperform sampling;

then conclude:

> **Posterior-weighted expression emerges as an efficient solution to tracking context-specific skills under partial observability.**

### Result B: MSE networks average but no belief signal emerges

Then the network may simply approximate a direct conditional-regression mapping.

Do not claim latent-context inference.

### Result C: both perfect and uncertain networks average identically OOD

Then averaging may be an architectural/optimization bias rather than an adaptation to contextual uncertainty.

### Result D: stochastic models match continuous models

Then the continuous-control objective does not explain human averaging.

### Result E: uncertainty-trained GRUs solve synthetic probes but fail human Novel

Then partial observability may explain ambiguity but not extrapolation to genuinely unfamiliar contexts.

---

## 52. Definition of done

- [ ] Synthetic environment contains independently evolving A/B skills.
- [ ] Context identity is latent.
- [ ] Context reliability is experimentally manipulable.
- [ ] Skill separation is experimentally manipulable.
- [ ] Exact Bayesian posterior is available as hidden ground truth.
- [ ] Exact human Both/Novel probes are absent from training.
- [ ] Deterministic MSE GRU implemented.
- [ ] Gaussian prediction GRU implemented.
- [ ] Context classification GRU implemented.
- [ ] Coin-flip/sample-capable model implemented.
- [ ] Distributional model implemented.
- [ ] 1/2/3/4/8/16-unit networks compared.
- [ ] Many random seeds tested.
- [ ] Posterior-mean alignment quantified.
- [ ] Belief alignment quantified.
- [ ] \(S_A\), \(S_B\), and \(q\) decoding quantified.
- [ ] Belief intervention causally alters expression.
- [ ] Skill-state intervention obeys responsibility weighting.
- [ ] Perfect-context versus uncertainty-trained networks compared.
- [ ] Continuous prediction versus posterior sampling compared.
- [ ] Zero-shot Both and Novel probes tested.
- [ ] Human data used only as external validation.
- [ ] Every new feature implemented with RED → GREEN → REFACTOR.

---

## Target paper-level result

> **Why does an uncertain context cause a learner to average separately learned skills? Because once context-specific skills must be tracked under partial observability, a recurrent learner benefits from representing uncertainty over which skill is currently relevant. When the system must convert that uncertainty into one continuous behavioral prediction, the Bayes-optimal solution is to express the posterior-weighted combination of the candidate skills.**

Compactly:

\[
\boxed{
\text{Context splits learning; uncertainty recombines what context split.}
}
\]
