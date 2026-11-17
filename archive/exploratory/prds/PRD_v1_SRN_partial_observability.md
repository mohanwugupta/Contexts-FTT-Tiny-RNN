# PRD v1: Emergent Averaging Under Partial Observability in a Simple Recurrent Network

## 1. Purpose

This is a deliberately minimal proof-of-concept project.

The goal is **not yet** to reproduce the full human motor-sequence task.

The goal is to test one narrow computational hypothesis:

> **If a simple recurrent network must track two context-specific latent performance states under partial observability, and must produce one continuous prediction before it knows which latent context is active, does it spontaneously learn posterior-weighted averaging between the two states?**

The primary model is an **Elman simple recurrent network (SRN)**.

A GRU is not the main model. It may be used later as an architecture robustness control.

If v1 succeeds, v2 will keep the same basic recurrent architecture but replace explicit performance prediction with a more psychologically realistic sequence-execution objective.

---

## 2. Scientific motivation

Previous sequence-learning work showed that a simple recurrent network can learn context-specific representations of an otherwise identical sequence.

However, the original next-item prediction objective does not create a strong incentive to solve uncertainty about which context-specific skill is currently active, because both contexts predict the same next sequence item.

The present project introduces a new computational pressure while keeping the recurrent architecture simple:

> The learner must track **different context-dependent performance states** and predict current performance when context identity is sometimes uncertain.

The critical question is whether this pressure is sufficient for graded averaging to emerge.

---

## 3. Core computational problem

There are two hidden context-specific performance states:

\[
S_A(t), \qquad S_B(t).
\]

Only one is active on each timestep:

\[
z_t \in \{A,B\}.
\]

The learner does **not** directly observe \(z_t\).

Instead it receives noisy contextual evidence:

\[
x_t \sim P(x \mid z_t).
\]

The target performance is:

\[
y_t = S_{z_t}(t) + \epsilon_t.
\]

The network predicts:

\[
\hat y_t
\]

before observing the current outcome.

Under context uncertainty, the Bayes-optimal deterministic prediction under squared error is:

\[
\hat y_t^*
=
P(A \mid H_t)S_A(t)
+
P(B \mid H_t)S_B(t),
\]

where \(H_t\) denotes the observation/history available before the current outcome.

The project asks whether an SRN learns a function approximating this computation **without the averaging rule being hard-coded**.

---

## 4. Why averaging should be useful

Suppose:

\[
S_A = 0.3,\qquad S_B = 0.7,
\]

and the current evidence implies:

\[
P(A)=P(B)=0.5.
\]

Predicting either endpoint gives expected squared error:

\[
0.5(0)^2 + 0.5(0.4)^2 = 0.08.
\]

Predicting the midpoint gives:

\[
0.5(0.2)^2 + 0.5(0.2)^2 = 0.04.
\]

So the deterministic posterior mean has half the expected uncertainty-related squared error of posterior sampling.

More generally:

\[
R_{\text{mean}}
=
q(1-q)(S_A-S_B)^2
\]

and:

\[
R_{\text{sample}}
=
2q(1-q)(S_A-S_B)^2.
\]

This gives a clean normative reason why averaging can emerge in v1.

---

## 5. What previous outcome feedback is doing

Previous outcome feedback does **not** create averaging.

It serves a different purpose.

The SRN must solve two computational problems:

\[
\boxed{
\text{Past outcome feedback}
\rightarrow
\text{estimate }S_A,S_B
}
\]

and:

\[
\boxed{
\text{Current noisy contextual evidence}
\rightarrow
\text{estimate which state applies}
}
\]

Averaging becomes useful only when the second problem is uncertain.

---

## 6. Scope

### In scope

- simple Elman SRN as the primary recurrent model;
- synthetic latent-state environment;
- two independently evolving context-specific states;
- noisy contextual observations;
- previous-outcome feedback;
- deterministic continuous prediction;
- MSE objective;
- exact Bayesian benchmark;
- perfect-context versus uncertainty-trained SRNs;
- posterior-mean alignment;
- hidden-state decoding;
- small-dimensionality SRN sweep;
- optional causal interventions;
- GRU robustness control only after the SRN result is established;
- TDD / RED → GREEN → REFACTOR.

### Out of scope for v1

Do **not** implement:

- the full motor sequence task;
- direct human RT fitting;
- Both/Mixed human stimuli;
- Novel human stimuli;
- sequence-execution control;
- explicit speed–accuracy tradeoff;
- RL;
- mixture-of-experts architecture;
- explicit A/B experts inside the network;
- explicit posterior input;
- explicit mixture-weight input;
- large transformer/RNN models.

These belong in v2 or later.

---

## 7. Primary architecture: Elman SRN

Use the same broad architectural family as the prior sequence-learning work.

State update:

\[
h_t
=
\tanh(
W_xu_t
+
W_hh_{t-1}
+
b_h
).
\]

Scalar output:

\[
\hat y_t
=
W_oh_t+b_o.
\]

Input vector:

\[
u_t
=
[x_t,\ y_{t-1},\ m_{t-1},\ e_t].
\]

Where:

- \(x_t\): current noisy context observation;
- \(y_{t-1}\): previous observed outcome;
- \(m_{t-1}\): mask indicating whether previous outcome exists;
- \(e_t\): episode-start indicator.

Do not supply the true context label or latent states.

---

## 8. Why SRN is the primary architecture

The scientific question does not require GRU-specific gates.

An Elman SRN already has:

- recurrent memory;
- nonlinear state dynamics;
- sufficient capacity to track recent outcomes;
- sufficient capacity to condition on contextual evidence.

Using the SRN preserves continuity with the previous computational work and makes the causal argument cleaner:

> **If the original SRN does not average but the same general recurrent architecture develops averaging under a different learning pressure, the difference is attributable more naturally to the computational problem rather than to a change in architecture.**

---

## 9. GRU as robustness control

Only after the main SRN result is established, fit a modest GRU under the same environment and objective.

The GRU is not part of the headline claim.

Use it to ask:

> Does the phenomenon generalize across recurrent architectures?

Interpretation:

- SRN + GRU both average → mechanism is not architecture-specific.
- GRU averages but SRN does not → memory/gating capacity may be necessary.
- SRN averages → preferred result because the simpler architecture suffices.

---

## 10. Latent-state environment

Each episode contains two independently evolving hidden performance states:

\[
S_A(t),S_B(t).
\]

Initialize:

\[
S_A(0),S_B(0)
\sim
Uniform(S_{min},S_{max}).
\]

Suggested starting range:

```text
S_min = 0.2
S_max = 0.8
```

Each context also has its own asymptote and learning rate:

\[
\theta_A,\theta_B,\alpha_A,\alpha_B.
\]

When context \(A\) is active:

\[
S_A(t+1)
=
S_A(t)
+
\alpha_A[\theta_A-S_A(t)]
\]

and:

\[
S_B(t+1)=S_B(t).
\]

When \(B\) is active, reverse the roles.

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

The network must infer current values rather than memorize fixed state values.

---

## 11. Context process

The true context follows a simple two-state Markov process.

For example:

\[
P(z_t=z_{t-1})=\rho_z.
\]

Suggested default:

```text
rho_z = 0.85
```

This produces stretches of the same context while allowing switches.

The exact value should be configurable.

---

## 12. Context observation model

The SRN never sees \(z_t\).

For the MVP use a one-dimensional Gaussian cue:

\[
x_t \sim
\begin{cases}
\mathcal N(-1,\sigma_c^2) & z_t=A\\
\mathcal N(+1,\sigma_c^2) & z_t=B.
\end{cases}
\]

This gives an analytically tractable likelihood model.

---

## 13. Main training conditions

### P — Perfect context

\[
\sigma_c=0.
\]

Context identity is fully observable.

The network never needs graded uncertainty over context identity.

### U — Uncertain context

Choose \(\sigma_c\) such that the two cue distributions overlap substantially but remain discriminable.

Pilot target:

```text
Bayes context-classification accuracy ≈ 70–85%
```

This condition creates pressure to infer context probabilistically.

---

## 14. Optional uncertainty ladder

Only after the two-condition MVP succeeds:

```text
U0 = perfect
U1 = low uncertainty
U2 = moderate uncertainty
U3 = high uncertainty
```

The expected pattern is:

\[
ContextUncertainty
\uparrow
\Rightarrow
PosteriorMeanAlignment
\uparrow
\]

up to the point where the task becomes too difficult.

---

## 15. Outcome generation

At each timestep:

\[
y_t
=
S_{z_t}(t)+\epsilon_t.
\]

Use:

\[
\epsilon_t
\sim
\mathcal N(0,\sigma_y^2).
\]

Suggested default:

```text
sigma_y = 0.02
```

The current outcome is revealed only **after** the network predicts.

---

## 16. Target timing

At timestep \(t\):

1. Environment determines \(z_t\).
2. Environment emits current noisy context cue \(x_t\).
3. SRN receives \(x_t\), \(y_{t-1}\), previous-outcome mask, and episode-start flag.
4. SRN predicts \(\hat y_t\).
5. Environment reveals \(y_t\).
6. \(y_t\) becomes available on timestep \(t+1\).

This timing is mandatory.

No current-target leakage.

---

## 17. Training objective

Primary objective:

\[
L_{\text{MSE}}
=
\frac{1}{T}
\sum_t
(y_t-\hat y_t)^2.
\]

Do not add:

- context-classification loss;
- latent-state supervision;
- posterior supervision;
- auxiliary reconstruction losses.

The emergence claim requires the network to discover any useful uncertainty representation purely because it helps predict performance.

---

## 18. Baselines

Implement:

### B0 — Global mean

Predict the overall training mean.

### B1 — Previous outcome

\[
\hat y_t=y_{t-1}.
\]

### B2 — Context-only regression baseline

Predict from current \(x_t\) without recurrence.

This tests whether recurrence/history is actually useful.

### B3 — Oracle Bayesian predictor

Uses the true generative model and hidden latent-state values.

This is the normative ceiling.

---

## 19. Exact Bayesian benchmark

Compute:

\[
q_t^*
=
P(z_t=A\mid x_{1:t},y_{1:t-1}).
\]

Then compute:

\[
y_t^*
=
q_t^*S_A(t)
+
(1-q_t^*)S_B(t).
\]

This benchmark is for evaluation only.

The SRN never receives \(q_t^*\), \(S_A\), or \(S_B\) as training targets.

---

## 20. Primary diagnostic: ambiguous states

Generate held-out diagnostic states with known posterior responsibilities.

Target values:

```text
q* = 0.05
q* = 0.10
q* = 0.25
q* = 0.50
q* = 0.75
q* = 0.90
q* = 0.95
```

Use multiple combinations of:

```text
S_A
S_B
context history
current cue
time since switch
```

For each state:

\[
y_t^*
=
q_t^*S_A(t)
+
(1-q_t^*)S_B(t).
\]

Ask whether \(\hat y_t\) tracks this posterior mean.

---

## 21. Primary metrics

### 21.1 Prediction performance

Report:

```text
MSE
RMSE
MAE
```

on held-out episodes.

### 21.2 Posterior-mean alignment

Primary mechanistic metric:

\[
R^2(\hat y_t,y_t^*).
\]

Also report:

```text
slope
intercept
MAE from y*
```

### 21.3 Implied responsibility

When \(S_A(t)\neq S_B(t)\), calculate:

\[
\hat q_t
=
\frac{
\hat y_t-S_B(t)
}{
S_A(t)-S_B(t)
}.
\]

Compare \(\hat q_t\) with \(q_t^*\).

Report:

```text
R²
slope
intercept
MAE
```

Do not clip \(\hat q_t\) for statistical evaluation.

---

## 22. Critical causal comparison

Evaluate Perfect-context and Uncertain-context SRNs on the **same ambiguous diagnostic states**.

Primary prediction:

\[
PosteriorMeanAlignment_U
>
PosteriorMeanAlignment_P.
\]

### Perfect-context SRN

Can solve familiar clear-context prediction but has never needed to represent graded context uncertainty.

Its OOD ambiguous behavior may be categorical, unstable, or arbitrary.

### Uncertainty-trained SRN

Should learn a smoother uncertainty-sensitive mapping and approximate:

\[
qS_A+(1-q)S_B.
\]

This comparison is the central scientific manipulation.

---

## 23. Original sequence SRN as conceptual negative control

The original next-item sequence-prediction SRN should remain unchanged.

Do not modify it until it produces the desired result.

Its role is theoretical:

> **Sequence prediction alone can support context-specific representational splitting but provides little incentive to learn graded retrieval when A and B imply the same next response.**

The v1 SRN differs primarily in the learning problem, not in requiring a more sophisticated recurrent architecture.

---

## 24. SRN hidden-dimension ladder

After the MVP succeeds with a sufficiently capable model, run:

```text
hidden_dim = 1
hidden_dim = 2
hidden_dim = 3
hidden_dim = 4
hidden_dim = 6
hidden_dim = 12 ceiling
```

Use the same training data distribution and evaluation suite.

Goal:

> Find the smallest recurrent state sufficient to solve the task near ceiling.

A theoretically interesting possibility is that the solution requires information equivalent to:

\[
S_A,\qquad S_B,\qquad q.
\]

Do not assume these correspond to individual units.

---

## 25. Hidden-state analysis

Freeze trained networks.

Generate independent diagnostic episodes spanning:

```text
S_A
S_B
q*
current cue x
previous context
time since switch
```

Fit linear probes from \(h_t\) to:

\[
S_A(t),
\qquad
S_B(t),
\qquad
q_t^*.
\]

Probe train/test split must be by episode.

Report held-out:

```text
R²_SA
R²_SB
R²_q
```

---

## 26. Representational success criterion

The strongest v1 representational result is:

1. \(S_A\) is decodable;
2. \(S_B\) is decodable;
3. \(q^*\) is decodable;
4. output approximates \(qS_A+(1-q)S_B\).

This would support the interpretation that the SRN has developed a functional analogue of:

\[
\{S_A,S_B,q\}.
\]

Decodability alone is not sufficient for a causal claim.

---

## 27. Continuous averaging versus coin flipping

Explicitly compare:

### Posterior mean

\[
\hat y_{\text{mean}}
=
qS_A+(1-q)S_B.
\]

### Posterior sampling

Sample:

\[
z'\sim Bernoulli(q)
\]

and output:

\[
\hat y_{\text{sample}}=S_{z'}.
\]

Verify analytically and numerically:

\[
R_{\text{sample}}
=
2R_{\text{mean}}
\]

for uncertainty-related squared error.

This is a normative analysis explaining why the training objective favors averaging.

---

## 28. Optional causal intervention

Only perform after \(R^2_q\) is reliably above baseline.

Estimate a hidden-state direction or low-dimensional subspace associated with \(q^*\).

Intervene:

\[
h'_t=h_t+\lambda v_q.
\]

Sweep \(\lambda\).

Prediction:

- one direction moves output toward \(S_A\);
- the opposite direction moves output toward \(S_B\);
- intermediate interventions give intermediate outputs.

This analysis is exploratory for v1 and should not block completion.

---

## 29. Random seeds

### MVP

```text
5 seeds / primary training condition
```

Conditions:

```text
Perfect-context SRN
Uncertain-context SRN
```

### Final v1

If the MVP succeeds:

```text
20 seeds / condition
```

Only after that should the uncertainty ladder and dimensionality sweep be run.

---

## 30. MVP execution order

### Phase 1 — Environment validation

Implement latent states, context process, noisy observations, and Bayesian benchmark.

No neural network yet.

### Phase 2 — Single SRN

Train one 4-unit SRN under uncertainty.

Confirm it beats baselines.

### Phase 3 — Main comparison

Train:

```text
5 Perfect-context seeds
5 Uncertain-context seeds
```

Evaluate on identical ambiguous states.

### Phase 4 — Representation analysis

Run hidden-state probes only if behavioral averaging emerges.

### Phase 5 — Final seeds

Increase to:

```text
20 seeds / condition
```

### Phase 6 — Dimensionality ladder

Find minimum sufficient SRN.

### Phase 7 — Optional GRU control

Only after the SRN result is established.

---

## 31. TDD requirement

All implementation must follow:

\[
\boxed{
RED\rightarrow GREEN\rightarrow REFACTOR
}
\]

Scientific contracts should be tested alongside software correctness.

---

## 32. RED → GREEN tests

### 32.1 Independent latent states

Require:
- A steps update only \(S_A\);
- B steps update only \(S_B\);
- inactive state is unchanged;
- simulation is reproducible from seed.

### 32.2 Context observation model

Require:
\[
E[x\mid A]\approx -1
\]
and
\[
E[x\mid B]\approx +1.
\]

Bayes context-classification accuracy must decrease as \(\sigma_c\) increases.

### 32.3 No target leakage

Changing current target \(y_t\) after input construction must not change the input at time \(t\).

### 32.4 Exact Bayesian posterior

Under symmetric likelihoods and equal prior:
\[
x=0 \Rightarrow q^*=0.5.
\]

Strong A evidence gives \(q^*>0.5\); strong B evidence gives \(q^*<0.5\).

### 32.5 Normative averaging advantage

With:
\[
S_A=0.3,\quad S_B=0.7,\quad q=0.5,
\]
require:
\[
y^*=0.5
\]
and verify:
\[
Risk_{sample}\approx2Risk_{mean}.
\]

### 32.6 SRN forward pass

Verify hidden-state and scalar-output tensor shapes and deterministic behavior under fixed seed/weights.

### 32.7 SRN learns basic prediction

On clear/low-noise held-out episodes, trained SRN must outperform:
```text
global mean
previous outcome
context-only baseline
```

### 32.8 Ambiguous evaluation

Diagnostic generator must return:
```text
q*
y*
SRN prediction
implied q
```
across the target posterior range.

### 32.9 Perfect versus Uncertain comparison

Both groups must be evaluated on the exact same diagnostic dataset.

### 32.10 Episode-held-out probes

Probe train/test splits must be by episode, with identical procedures for \(S_A,S_B,q^*\).

### 32.11 Dimensionality sweep

Runner must support:
```text
1
2
3
4
6
12
```
without changing other scientific parameters.

### 32.12 Optional GRU control

GRU must receive identical inputs, train/test episodes, loss, and evaluation metrics.

---

## 33. Proposed code structure

```text
srn_partial_observability_v1/
    __init__.py

    environment/
        latent_states.py
        contexts.py
        episodes.py
        bayes_filter.py

    models/
        elman_srn.py
        gru_control.py

    training/
        trainer.py
        baselines.py

    analysis/
        bayes_risk.py
        ambiguity_tests.py
        posterior_alignment.py
        hidden_probes.py
        interventions.py
        dimensionality.py

    plotting/
        figures.py

    pipeline.py

tests/
    test_latent_states.py
    test_contexts.py
    test_no_leakage.py
    test_bayes_filter.py
    test_bayes_risk.py
    test_elman_srn.py
    test_training.py
    test_ambiguity.py
    test_condition_comparison.py
    test_hidden_probes.py
    test_dimensionality.py
    test_gru_control.py
```

---

## 34. Required outputs

```text
run_config_v1.json

training_conditions.csv
network_performance.csv
baseline_performance.csv

ambiguity_behavior.csv
bayesian_ground_truth.csv
posterior_mean_alignment.csv
implied_responsibility.csv

perfect_vs_uncertain.csv
continuous_vs_sampling.csv

hidden_state_probes.csv
dimensionality_comparison.csv

belief_interventions.csv
gru_robustness_control.csv

seed_summary.csv
```

Outputs skipped because of staged gates should be documented as `not_run` rather than silently omitted.

---

## 35. Required figures

### Figure 1 — Computational task

```text
hidden context A/B
       ↓
noisy context cue
       ↓
     SRN
       ↑
previous outcome
       ↓
predict current performance
       ↓
observe true outcome
```

### Figure 2 — Example episode

Plot:
- \(S_A(t)\);
- \(S_B(t)\);
- active context;
- observed \(y_t\);
- SRN prediction.

### Figure 3 — Emergent averaging

X-axis:
\[
q^*
\]

Y-axis:
\[
\hat y.
\]

Overlay:
\[
y^*=qS_A+(1-q)S_B.
\]

### Figure 4 — Perfect versus Uncertain training

Compare posterior-mean alignment on identical ambiguous states.

### Figure 5 — Implied responsibility

Plot \(q^*\) versus \(\hat q\).

### Figure 6 — Mean versus sampling

Expected MSE as a function of \(q\).

### Figure 7 — Hidden-state representation

Held-out decoding of:
```text
S_A
S_B
q*
```
for Perfect versus Uncertain SRNs.

### Figure 8 — Dimensionality

Performance and posterior alignment across:
```text
1
2
3
4
6
12
```
hidden units.

### Figure 9 — Optional GRU control

SRN versus GRU under identical training conditions.

---

## 36. Main decision logic

### Result A — Strong support

Uncertainty-trained SRNs:
- solve the latent-performance prediction task;
- outperform simple baselines;
- show strong posterior-mean alignment;
- outperform Perfect-context SRNs on ambiguous diagnostic states;
- contain decodable information about \(S_A,S_B,q^*\).

Interpretation:

> **Posterior-weighted averaging can emerge in a simple recurrent learner when context-specific latent states must be tracked under partial observability and converted into one continuous prediction.**

Proceed to v2.

### Result B — Averaging emerges, but \(q^*\) is not decodable

The SRN may have learned a direct nonlinear mapping from history and current evidence to expected performance.

Do not claim an explicit latent belief representation.

### Result C — Perfect and Uncertain SRNs average equally

Averaging may reflect generic interpolation properties of the SRN rather than adaptation to partial observability.

### Result D — Uncertainty-trained SRN fails but GRU succeeds

The computation may require stronger memory/gating mechanisms.

This is informative, but architecture becomes part of the explanation.

### Result E — Neither SRN nor GRU averages

The proposed training pressure is insufficient or optimization does not discover the normative solution.

Diagnose before constructing v2.

---

## 37. Definition of done

- [ ] Primary recurrent model is an Elman SRN.
- [ ] Two independently evolving latent states are implemented.
- [ ] Context identity is hidden.
- [ ] Context cues can be perfect or noisy.
- [ ] Previous outcome is revealed only after the current prediction.
- [ ] Exact Bayesian posterior is available independently.
- [ ] Exact posterior mean is available independently.
- [ ] MSE prediction objective is implemented.
- [ ] Global-mean, previous-outcome, context-only, and Bayesian baselines are implemented.
- [ ] Perfect-context and Uncertain-context SRNs are trained.
- [ ] Both are evaluated on identical ambiguous states.
- [ ] PosteriorMeanAlignment is reported.
- [ ] Implied responsibility is compared with \(q^*\).
- [ ] Mean versus posterior-sampling risk is verified.
- [ ] Hidden-state probes for \(S_A,S_B,q^*\) are run if behavior succeeds.
- [ ] 5 seeds per condition are used for MVP.
- [ ] 20 seeds per condition are used for final v1 if MVP succeeds.
- [ ] SRN hidden-dimension sweep is run after the core result.
- [ ] GRU is treated only as an optional architecture robustness control.
- [ ] No human Mixed/Novel outcomes are used in training.
- [ ] No explicit mixture computation is built into the SRN.
- [ ] Every core feature is developed RED → GREEN → REFACTOR.
- [ ] Final report explicitly states whether the proof of concept succeeded.

---

## 38. Success criterion for moving to v2

Move to v2 only if the uncertainty-trained SRN provides convincing evidence that:

\[
\boxed{
\hat y_t
\approx
q_t^*S_A(t)
+
(1-q_t^*)S_B(t)
}
\]

on ambiguous states.

The preferred additional result is that the hidden state contains information sufficient to recover:

\[
S_A,\qquad S_B,\qquad q^*.
\]

v2 will then preserve the simple SRN architecture but change the **learning objective/environment**.

Instead of explicitly predicting performance, v2 should ask whether graded recombination emerges when an SRN must actually optimize sequence execution under uncertainty about which context-specific skill state is active.

The intended progression is:

\[
\boxed{
\text{Original SRN: sequence prediction}
}
\]

\[
\boxed{
\text{v1 SRN: proof that partial observability can induce averaging}
}
\]

\[
\boxed{
\text{v2 SRN: test whether the same computation emerges from a psychologically stronger motor-control objective}
}
\]

The architecture stays simple.

The scientific manipulation is the **problem the learner is required to solve**.
