# PRD v1: Proof of Concept — Emergent Averaging Under Partial Observability

## 1. Purpose

This is a deliberately minimal proof-of-concept project.

The goal is **not** to model the human motor task faithfully yet.

The goal is to test one narrow computational claim:

> **If a recurrent network must track two context-specific latent performance states under partial observability, and must make one continuous prediction before knowing which latent context is active, does it spontaneously learn to average between the two states when context is ambiguous?**

If this works, a later v2 will ask whether the same computation can emerge in a more realistic sequence-execution task without explicitly asking the model to predict performance.

---

## 2. Core scientific question

Suppose there are two latent context-specific states:

\[
S_A(t), \qquad S_B(t)
\]

Only one is active on each trial.

The network does **not** directly observe whether A or B is active. It receives noisy contextual evidence.

Its task is to predict the current performance value:

\[
y_t = S_{z_t}(t) + \epsilon_t,
\]

where:

\[
z_t \in \{A,B\}.
\]

The central question is:

> **When the contextual observation is ambiguous, does a generic GRU learn to predict the posterior-weighted average of the two latent states?**

That is:

\[
\hat y_t
\approx
P(A\mid \text{history})S_A(t)
+
P(B\mid \text{history})S_B(t).
\]

No averaging rule, A/B experts, mixture weights, or posterior variable may be hard-coded into the GRU.

---

## 3. Why averaging is expected

Consider:

\[
S_A = 400,\qquad S_B = 600.
\]

Suppose the current observation implies:

\[
P(A)=P(B)=0.5.
\]

A deterministic predictor trained with squared error could predict:

- 400,
- 600,
- or something between them.

Predicting 400 gives expected squared error:

\[
0.5(400-400)^2 + 0.5(600-400)^2 = 20{,}000.
\]

Predicting 500 gives:

\[
0.5(400-500)^2 + 0.5(600-500)^2 = 10{,}000.
\]

Therefore the Bayes-optimal deterministic prediction under MSE is the conditional mean:

\[
\hat y = 0.5S_A + 0.5S_B.
\]

More generally:

\[
\hat y^*
=
qS_A+(1-q)S_B.
\]

The project tests whether a GRU **discovers this computation from experience**.

---

## 4. What previous performance feedback is doing

Previous outcomes are **not** what causes averaging.

They only allow the GRU to estimate the current values of the two evolving latent states.

The task decomposes into two problems:

\[
\boxed{
\text{Past feedback}
\rightarrow
\text{estimate }S_A,S_B
}
\]

and:

\[
\boxed{
\text{Current noisy context}
\rightarrow
\text{estimate which state applies}
}
\]

Averaging is useful only when the second problem is uncertain.

---

## 5. Scope

### In scope

- synthetic latent-state environment;
- two independently evolving latent states;
- noisy contextual observations;
- recurrent prediction with a GRU;
- deterministic MSE prediction;
- exact Bayesian benchmark;
- perfect-context versus uncertainty-trained comparison;
- posterior-mean alignment;
- hidden-state decoding of \(S_A\), \(S_B\), and \(P(A)\);
- basic causal interventions if representation decoding succeeds;
- comparison against simple baselines;
- TDD / RED → GREEN workflow.

### Out of scope for v1

Do **not** implement:

- the full human sequence task;
- human RT fitting;
- Mixed / Both human stimuli;
- Novel human stimuli;
- SRN next-key prediction;
- multi-objective sequence + performance learning;
- distributional output models;
- large architecture sweeps;
- mechanistic claims about human neural implementation.

Those belong in v2 or later.

---

## 6. Environment

Each episode contains two hidden context-specific performance states:

\[
S_A(t),\qquad S_B(t).
\]

Only one is active on each step.

The network observes:

1. the current noisy context cue;
2. the previous observed outcome;
3. an episode-start indicator.

The network predicts:

\[
\hat y_t
\]

before receiving the current true outcome.

After prediction, the true outcome becomes available as feedback for the next timestep.

---

## 7. Latent-state dynamics

For v1, keep the dynamics simple and stable.

At episode start:

\[
S_A(0),S_B(0)
\sim
Uniform(S_{min},S_{max}).
\]

Example default:

```text
S_min = 0.2
S_max = 0.8
```

Each latent state moves slowly toward its own asymptote when active.

For active context \(z_t\):

\[
S_{z_t}(t+1)
=
S_{z_t}(t)
+
\alpha_{z_t}
\left[
\theta_{z_t}-S_{z_t}(t)
\right].
\]

Inactive state remains unchanged:

\[
S_{\neg z_t}(t+1)=S_{\neg z_t}(t).
\]

Randomize across episodes:

```text
S_A(0)
S_B(0)
theta_A
theta_B
alpha_A
alpha_B
context order
context dwell lengths
```

This forces the GRU to infer the current latent values rather than memorize a fixed A/B mapping.

---

## 8. Context process

The true context is:

\[
z_t\in\{A,B\}.
\]

Contexts should occur in runs rather than independently every step.

Use a simple Markov process:

\[
P(z_t=z_{t-1})=\rho_z.
\]

Example:

```text
rho_z = 0.85
```

This makes temporal history useful without making context perfectly predictable.

---

## 9. Context observations

The GRU does not see \(z_t\).

Instead it receives:

\[
x_t.
\]

Use one-dimensional Gaussian observations for the MVP:

\[
x_t \sim
\begin{cases}
\mathcal N(-1,\sigma_c^2) & z_t=A\\
\mathcal N(+1,\sigma_c^2) & z_t=B.
\end{cases}
\]

This creates a known likelihood model.

At low \(\sigma_c\), context is obvious.

At larger \(\sigma_c\), the distributions overlap and context becomes ambiguous.

---

## 10. Main experimental manipulation

Train two primary conditions.

### Condition P — Perfect context

```text
sigma_c = 0
```

The network always knows the active context.

There is no need to represent contextual uncertainty.

### Condition U — Uncertain context

Choose \(\sigma_c\) such that A/B observations overlap substantially but remain learnable.

Pilot target:

```text
Bayes context-classification accuracy ≈ 70–85%
```

This condition creates pressure to represent uncertainty over latent context.

---

## 11. Optional uncertainty ladder

After the MVP works, extend to:

```text
U0 = perfect
U1 = low noise
U2 = moderate noise
U3 = high noise
```

Do not implement the full ladder before the two-condition MVP passes.

---

## 12. Outcome generation

The current target is:

\[
y_t=S_{z_t}(t)+\epsilon_t.
\]

For the MVP:

\[
\epsilon_t\sim\mathcal N(0,\sigma_y^2).
\]

Keep outcome noise modest.

Suggested starting value:

```text
sigma_y = 0.02
```

The network should predict \(y_t\) before observing it.

---

## 13. GRU inputs

At time \(t\):

```text
x_t                  current noisy context observation
y_(t-1)              previous observed performance
previous_outcome_mask
episode_start
```

The previous-outcome mask is necessary because \(y_{t-1}\) is undefined at the first timestep.

Do not provide:

```text
z_t
S_A(t)
S_B(t)
P(A | ...)
current y_t
```

These exist only for evaluation.

---

## 14. GRU architecture

Primary model:

```text
input
  ↓
GRU
  ↓
linear scalar readout
  ↓
predicted performance
```

Equation:

\[
h_t=GRU(x_t,y_{t-1},h_{t-1})
\]

\[
\hat y_t=W_oh_t+b_o.
\]

Primary hidden size:

```text
hidden_dim = 4
```

Use 4 dimensions initially to avoid under-capacity.

Only after the main phenomenon is established should dimensionality be reduced.

---

## 15. Training objective

Primary loss:

\[
L_{MSE}
=
\frac{1}{T}\sum_t
(y_t-\hat y_t)^2.
\]

No auxiliary context-classification loss.

No auxiliary latent-state loss.

No supervision on \(P(A)\).

No explicit mixture loss.

The GRU should discover any useful latent representation purely because it improves performance prediction.

---

## 16. Baselines

Implement three simple baselines.

### B0 — Global mean

Predict the overall training mean.

### B1 — Previous outcome

\[
\hat y_t=y_{t-1}.
\]

### B2 — Oracle Bayesian predictor

This model knows the true generative process and computes the exact posterior.

It provides the normative ceiling.

---

## 17. Exact Bayesian benchmark

Because the environment is known, calculate:

\[
q_t^*
=
P(z_t=A\mid x_{1:t},y_{1:t-1}).
\]

Then:

\[
y_t^*
=
q_t^*S_A(t)
+
(1-q_t^*)S_B(t).
\]

This benchmark is never used as a training target for the GRU.

It exists only for evaluation.

---

## 18. Primary test: ambiguous observations

Create diagnostic test states where:

\[
P(A\mid x_t,\text{history})
\]

takes known values across approximately:

```text
0.05
0.10
0.25
0.50
0.75
0.90
0.95
```

For each test state, hold the latent values known.

Example:

```text
S_A = 0.3
S_B = 0.7
```

The Bayes-optimal prediction is:

\[
y^*(q)=qS_A+(1-q)S_B.
\]

Ask whether the GRU output follows this line.

---

## 19. Main emergence metrics

### 19.1 Held-out performance

```text
test_MSE
test_RMSE
```

### 19.2 Posterior mean alignment

\[
R^2(\hat y,y^*).
\]

This is the primary metric.

### 19.3 Implied responsibility

When:

\[
S_A\neq S_B,
\]

define:

\[
\hat q_t
=
\frac{\hat y_t-S_B(t)}
{S_A(t)-S_B(t)}.
\]

Then compare:

\[
\hat q_t
\]

against:

\[
q_t^*.
\]

Report:

```text
R²
slope
intercept
MAE
```

Do not clip \(\hat q\) for inference.

---

## 20. Main causal comparison

Compare Perfect-context and Uncertain-context networks on the **same ambiguous held-out tests**.

The critical prediction is:

\[
PosteriorMeanAlignment_U
>
PosteriorMeanAlignment_P.
\]

More specifically:

### Perfect-context GRU

May perform well on familiar clear contexts but behave arbitrarily or categorically on ambiguous observations it never needed to interpret.

### Uncertainty-trained GRU

Should produce predictions that track:

\[
qS_A+(1-q)S_B.
\]

This is the core proof-of-concept result.

---

## 21. Coin-flip comparison

This must be included because it directly addresses the alternative strategy.

For each Bayesian diagnostic state:

1. compute the posterior mean prediction;
2. generate posterior-sampling predictions by drawing A or B according to \(q^*\);
3. compare expected MSE.

Analytically:

\[
R_{mean}
=
q(1-q)(S_A-S_B)^2
\]

and:

\[
R_{sample}
=
2q(1-q)(S_A-S_B)^2.
\]

Verify numerically that:

\[
R_{sample}\approx2R_{mean}.
\]

This demonstrates why averaging is advantageous under the current continuous MSE objective.

---

## 22. Hidden-state analysis

After training, freeze the GRU.

Generate a diagnostic dataset spanning:

```text
S_A
S_B
q*
current x
previous context
time since switch
```

Fit linear probes from hidden state \(h_t\) to:

\[
S_A(t)
\]

\[
S_B(t)
\]

\[
q_t^*.
\]

Use held-out episodes for probe evaluation.

Report held-out:

```text
R²_SA
R²_SB
R²_q
```

---

## 23. Representational success criterion

The strongest v1 result is:

1. \(S_A\) is decodable;
2. \(S_B\) is decodable;
3. \(q^*\) is decodable;
4. the output tracks the posterior-weighted mean.

This would suggest that the network has learned something functionally similar to:

\[
\{S_A,S_B,q\}.
\]

Do not require these variables to correspond one-to-one with individual GRU units.

---

## 24. Minimal causal intervention

Only run this if \(q^*\) is reliably decodable.

Estimate a hidden-state direction:

\[
v_q
\]

that predicts \(q^*\).

For matched diagnostic states, perturb:

\[
h'_t=h_t+\lambda v_q.
\]

Sweep:

```text
lambda < 0
lambda = 0
lambda > 0
```

Prediction:

- negative shift moves output toward one latent state;
- positive shift moves output toward the other;
- intermediate values produce intermediate predictions.

This is exploratory in v1.

Do not block the project on intervention success.

---

## 25. Training seeds

MVP:

```text
5 seeds / condition
```

Final v1:

```text
20 seeds / condition
```

Primary comparison:

```text
Perfect-context
vs.
Uncertain-context
```

Do not run a large factorial sweep until the MVP is green.

---

## 26. Recommended MVP order

### Phase 1

Single environment configuration:

```text
Perfect context
Uncertain context
hidden_dim = 4
5 seeds
```

Confirm that both learn the basic prediction task.

### Phase 2

Run ambiguous diagnostic tests.

Check whether uncertainty-trained GRUs show posterior-mean alignment.

### Phase 3

Run hidden-state probes.

Only continue if the behavioral phenomenon exists.

### Phase 4

Increase to 20 seeds.

### Phase 5

Optional uncertainty ladder and hidden-dimension sweep.

---

## 27. TDD requirement

All implementation should follow:

\[
\boxed{
RED\rightarrow GREEN\rightarrow REFACTOR
}
\]

Tests should cover both software correctness and scientific assumptions.

---

## 28. RED → GREEN 1: independent latent states

### RED

Create:

```text
A active
A active
B active
```

Require:

- A practice changes only \(S_A\);
- B practice changes only \(S_B\);
- inactive state remains unchanged;
- results are seed-reproducible.

### GREEN

Implement latent-state environment.

---

## 29. RED → GREEN 2: noisy contexts

### RED

Require:

```text
E[x | A] < E[x | B]
```

with default means \(-1\) and \(+1\).

Require Bayes context-classification accuracy to decrease as \(\sigma_c\) increases.

### GREEN

Implement context observation generator.

---

## 30. RED → GREEN 3: no target leakage

### RED

Construct the input at time \(t\).

Change current target \(y_t\).

Require the model input for time \(t\) to remain unchanged.

### GREEN

Ensure only past outcomes are available.

---

## 31. RED → GREEN 4: exact Bayesian posterior

### RED

With equal prior and \(x=0\) under symmetric likelihoods:

\[
q^*=0.5.
\]

With strongly A-like evidence:

\[
q^*>0.5.
\]

With strongly B-like evidence:

\[
q^*<0.5.
\]

### GREEN

Implement Bayesian filter.

---

## 32. RED → GREEN 5: posterior mean risk

### RED

Use:

\[
S_A=0.3,\quad S_B=0.7,\quad q=0.5.
\]

Require posterior mean:

\[
0.5.
\]

Numerically verify:

\[
Risk_{sample}\approx2Risk_{mean}.
\]

### GREEN

Implement normative-risk utilities.

---

## 33. RED → GREEN 6: GRU learns clear-context prediction

### RED acceptance criterion

On low-noise held-out episodes, trained GRU must beat:

```text
global mean
previous-outcome baseline
```

by a predetermined margin.

### GREEN

Implement training loop.

---

## 34. RED → GREEN 7: uncertain-context interpolation

### RED acceptance criterion

For uncertainty-trained networks on held-out ambiguous observations:

\[
R^2(\hat y,y^*) > 0
\]

and preferably:

```text
PosteriorMeanAlignment > 0.8
```

for the final successful configuration.

Do not hard-code the 0.8 value into early unit tests; use it as a scientific target.

### GREEN

Implement ambiguity evaluator.

---

## 35. RED → GREEN 8: perfect versus uncertain training

### RED

Both models receive the identical ambiguous test set.

Require export of:

```text
PosteriorMeanAlignment_Perfect
PosteriorMeanAlignment_Uncertain
difference
```

### GREEN

Implement condition-comparison pipeline.

---

## 36. RED → GREEN 9: hidden-state probes

### RED

Probe pipeline must:

- split by episode;
- never train/test on states from the same episode;
- report train and held-out R²;
- use identical procedures for \(S_A,S_B,q^*\).

### GREEN

Implement representation probing.

---

## 37. RED → GREEN 10: causal intervention plumbing

### RED

On a toy network with known linear hidden-to-output mapping, hidden perturbations must produce predictable output changes.

### GREEN

Implement intervention utility.

Then apply to the learned GRUs if justified.

---

## 38. Code structure

```text
gru_partial_observability_v1/
    __init__.py

    environment/
        latent_states.py
        contexts.py
        episodes.py
        bayes_filter.py

    models/
        gru_predictor.py

    training/
        trainer.py
        baselines.py

    analysis/
        posterior_mean.py
        ambiguity_tests.py
        hidden_probes.py
        interventions.py
        bayes_risk.py

    plotting/
        figures.py

    pipeline.py

tests/
    test_latent_states.py
    test_contexts.py
    test_no_leakage.py
    test_bayes_filter.py
    test_bayes_risk.py
    test_training.py
    test_ambiguity.py
    test_condition_comparison.py
    test_hidden_probes.py
    test_interventions.py
```

---

## 39. Required outputs

```text
run_config_v1.json

training_conditions.csv
network_performance.csv

ambiguity_behavior.csv
bayesian_ground_truth.csv
posterior_mean_alignment.csv
implied_responsibility.csv

perfect_vs_uncertain.csv
continuous_vs_sampling.csv

hidden_state_probes.csv
belief_interventions.csv

seed_summary.csv
```

---

## 40. Required figures

### Figure 1 — Task schematic

Show:

```text
latent A/B context
      ↓
noisy context observation
      ↓
GRU + previous outcome
      ↓
predict current performance
      ↓
observe true outcome
```

### Figure 2 — Example episode

Plot:

- true \(S_A\);
- true \(S_B\);
- active context;
- observed outcome;
- GRU prediction.

### Figure 3 — Posterior mean behavior

X-axis:

\[
q^*
\]

Y-axis:

\[
\hat y.
\]

Overlay the normative line:

\[
qS_A+(1-q)S_B.
\]

### Figure 4 — Perfect versus uncertain training

Compare posterior-mean alignment on the same ambiguous test set.

### Figure 5 — Coin flip versus averaging

Plot expected MSE of:

- posterior mean;
- posterior sampling.

across \(q\).

### Figure 6 — Hidden-state decoding

Show held-out \(R^2\) for:

```text
S_A
S_B
q*
```

for Perfect and Uncertain models.

### Figure 7 — Optional belief intervention

If successful, plot intervention magnitude versus predicted output.

---

## 41. Primary decision logic

### Result A — Strong support

Uncertainty-trained GRUs:

- learn the prediction task;
- outperform simple baselines;
- track the Bayesian posterior mean under ambiguity;
- outperform Perfect-context GRUs on ambiguous held-out states;
- contain decodable \(S_A,S_B,q^*\).

Interpretation:

> **Posterior-weighted averaging can emerge spontaneously when a recurrent learner must track context-specific latent states under partial observability and produce one continuous prediction.**

Proceed to v2.

### Result B — Averaging emerges, but no belief variable is decodable

Interpretation:

The network may have learned a direct nonlinear regression from observations/history to expected performance.

Do not claim explicit latent-context inference.

Further analysis is needed before v2.

### Result C — Perfect and uncertain models average equally

Interpretation:

Averaging may be due to generic interpolation bias in the network rather than training pressure from partial observability.

This weakens the hypothesis.

### Result D — Uncertainty-trained model does not average

Interpretation:

The proposed learning pressure is insufficient.

Do not move directly to a more realistic v2.

Diagnose whether:

- the GRU fails to track the two states;
- contextual uncertainty is too weak/strong;
- history structure is insufficient;
- architecture/loss prevents the normative solution from being learned.

### Result E — Model coin-flips or behaves categorically

Interpretation:

The deterministic MSE setup did not induce the expected posterior-mean solution.

Investigate optimization or representational constraints.

---

## 42. Definition of done

- [ ] Two independently evolving latent states are implemented.
- [ ] Context identity is hidden from the GRU.
- [ ] Context observations can be perfect or noisy.
- [ ] Previous outcome is available only after prediction.
- [ ] Exact Bayesian posterior is computed independently.
- [ ] Exact posterior mean is computed independently.
- [ ] Deterministic GRU predictor is trained.
- [ ] Perfect-context and uncertain-context conditions are compared.
- [ ] Ambiguous held-out observations are evaluated.
- [ ] PosteriorMeanAlignment is reported.
- [ ] Implied responsibility is compared with Bayesian responsibility.
- [ ] Mean versus posterior-sampling risk is verified.
- [ ] Hidden-state probes for \(S_A,S_B,q^*\) are run.
- [ ] At least five seeds are used for MVP.
- [ ] Twenty seeds are used for final v1 if MVP succeeds.
- [ ] No human Both/Novel data are used in training.
- [ ] No explicit mixture computation is built into the GRU.
- [ ] All core features have RED → GREEN tests.
- [ ] Final report explicitly states whether the proof-of-concept succeeded.

---

## 43. Success criterion for moving to v2

Move to v2 only if the uncertainty-trained GRU shows evidence that:

\[
\boxed{
\hat y_t
\approx
q_t^*S_A(t)
+
(1-q_t^*)S_B(t)
}
\]

on ambiguous observations it was not explicitly trained to map to an average.

The most compelling result would additionally show that:

\[
S_A,\quad S_B,\quad q^*
\]

are recoverable from the hidden state.

v2 can then ask:

> **Can the same computation emerge from a sequence-learning / sequence-execution objective that more closely resembles the human motor task, without explicitly asking the network to predict performance?**
