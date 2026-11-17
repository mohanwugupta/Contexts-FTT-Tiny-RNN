# PRD v1.2: Environmental Pressures for Emergent Bayes-Optimal Averaging in an 8-Unit Elman SRN

## 1. Purpose

This project asks a narrower and more useful question than the previous “kitchen-sink” PRD:

> **Holding the architecture, training schedule, and amount of training fixed, what environmental pressures cause a simple recurrent network to develop Bayes-optimal posterior averaging?**

The goal is **not** to tune the network until it works.

The goal is to identify which properties of the learning environment make posterior-weighted averaging an adaptive solution.

The primary model remains a plain Elman SRN with:

```text
hidden_dim = 8
```

The training curriculum must match the FTT structure:

```text
10 sequences/trials in Context A
10 sequences/trials in Context B
10 sequences/trials in Context A
10 sequences/trials in Context B
...
```

The number of blocks, training exposure, optimizer, and basic update schedule should remain fixed across all environmental conditions.

Posterior averaging should be expected to emerge **quickly**, within the first few A↔B switches, if the relevant environmental pressure is present.

---

# 2. Central hypothesis

The target computation is:

\[
\hat y_t
\approx
q_t^* S_A(t)
+
(1-q_t^*) S_B(t),
\]

where:

\[
q_t^*
=
P(z_t=A\mid \text{available evidence}).
\]

The network is never given:

```text
q*
S_A
S_B
z_t
posterior mean target
explicit mixture weights
```

The hypothesis is:

> **Posterior averaging emerges when context is informative but imperfect, the A/B latent states differ enough for context assignment to matter, and hard commitment to the wrong state is costly.**

The scientific objective is to determine which environmental ingredients are necessary and sufficient.

---

# 3. Fixed elements

These are **not search dimensions**.

## Architecture

```text
Elman SRN
hidden_dim = 8
tanh recurrence
single scalar output
```

## Training curriculum

Always:

```text
A × 10 trials
B × 10 trials
A × 10 trials
B × 10 trials
...
```

No random block lengths.

No alternate curricula.

No clear→uncertain curriculum.

No fixed→evolving curriculum.

No manipulation of switch frequency.

## Training amount

Use the same total training exposure for every condition.

Do not search:

```text
number of epochs
number of blocks
episode length
training duration
```

Reuse the training horizon and update cadence from the existing FTT/SRN implementation where possible.

## Optimizer

Use one fixed optimizer and learning rate across the environmental sweep.

Optimization hyperparameters should not be tuned condition-by-condition.

If the current implementation is stable, retain its existing optimizer settings.

---

# 4. Important conceptual constraint

If the 10A/10B schedule **perfectly determines which latent state is active**, then after the SRN learns the block structure it can infer the active state by phase/counting alone.

In that environment:

\[
P(z_t=A\mid \text{block phase})
\in\{0,1\}.
\]

There is no normative reason to maintain a graded posterior, and therefore no reason for Bayes-optimal averaging to emerge.

Accordingly:

> **The macro FTT schedule stays fixed, but the mapping from nominal context to the latent performance state must be allowed to be probabilistic in some environmental conditions.**

This is the core environmental manipulation.

---

# 5. Generative structure

Distinguish:

## Nominal task context

\[
c_t\in\{A,B\}.
\]

This follows the exact FTT schedule:

```text
A A A A A A A A A A
B B B B B B B B B B
A A A A A A A A A A
...
```

## Latent performance state

\[
z_t\in\{A,B\}.
\]

This determines which latent skill/performance state actually generates the outcome.

## Context observation

\[
x_t.
\]

This is noisy/graded evidence about the latent state.

## Outcome

\[
y_t = S_{z_t}(t)+\epsilon_t.
\]

The SRN must predict \(y_t\) before observing it.

---

# 6. Primary architecture

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

Output:

\[
\hat y_t
=
W_oh_t+b_o.
\]

Primary input:

\[
u_t=
[x_t,\ y_{t-1},\ m_{t-1},\ e_t].
\]

Where:

- \(x_t\): current contextual evidence;
- \(y_{t-1}\): previous observed performance;
- \(m_{t-1}\): previous-outcome mask;
- \(e_t\): experiment/session-start indicator.

Do not provide the nominal block label \(c_t\) explicitly unless the original FTT SRN already receives an equivalent context feature.

If block phase can be inferred recurrently, that is allowed.

---

# 7. Fixed training objective

Primary loss:

\[
L_{\text{MSE}}
=
\frac{1}{T}
\sum_t
(y_t-\hat y_t)^2.
\]

No auxiliary losses.

No context-classification objective.

No posterior supervision.

No latent-state supervision.

No mixture head.

The network must discover posterior averaging because it improves prediction under the environment.

---

# 8. Environmental pressure 1: context validity

This is the most important manipulation.

Define:

\[
v
=
P(z_t=c_t).
\]

In a nominal A block:

\[
P(z_t=A)=v,
\qquad
P(z_t=B)=1-v.
\]

In a nominal B block:

\[
P(z_t=B)=v,
\qquad
P(z_t=A)=1-v.
\]

Test:

```text
V1 = 1.00
V2 = 0.95
V3 = 0.85
V4 = 0.75
V5 = 0.65
V6 = 0.55
```

Interpretation:

### v = 1.00

Nominal context perfectly identifies the latent state.

A hard A/B selector is sufficient.

### Intermediate validity

Nominal context is informative but imperfect.

The system benefits from representing uncertainty.

### v ≈ .50

Nominal context becomes nearly uninformative.

The network may collapse toward a global average because there is little useful context signal.

### Main prediction

Posterior averaging should be strongest at **intermediate context validity**:

\[
\boxed{
\text{perfect context}
\rightarrow
\text{hard selection}
}
\]

\[
\boxed{
\text{moderately reliable context}
\rightarrow
\text{graded posterior inference}
}
\]

\[
\boxed{
\text{uninformative context}
\rightarrow
\text{global averaging}
}
\]

This predicts a non-monotonic emergence regime rather than “more noise is always better.”

---

# 9. Environmental pressure 2: graded cue evidence

The SRN should receive evidence whose strength varies continuously.

For latent state \(z_t\):

\[
x_t
\sim
\begin{cases}
\mathcal N(-\mu,\sigma_x^2), & z_t=A\\
\mathcal N(+\mu,\sigma_x^2), & z_t=B.
\end{cases}
\]

Keep:

```text
mu = 1
```

and vary cue overlap.

Parameterize conditions by Bayes classification accuracy rather than raw \(\sigma_x\).

Recommended levels:

```text
C1 ≈ 97% accurate
C2 ≈ 90%
C3 ≈ 80%
C4 ≈ 70%
C5 ≈ 60%
```

The cue distributions should naturally generate:

- very A-like evidence;
- weakly A-like evidence;
- ambiguous evidence;
- weakly B-like evidence;
- very B-like evidence.

This is essential because the target computation is **graded**.

If the network only ever sees perfectly clear A/B evidence, it has little reason to learn calibrated intermediate responsibilities.

---

# 10. Environmental pressure 3: A/B state separation

The cost of incorrectly committing to A or B depends on:

\[
\Delta S
=
|S_A-S_B|.
\]

Manipulate:

```text
D1 = 0.20
D2 = 0.40
D3 = 0.60
D4 = 0.80
```

For the simplest fixed-state environment, examples are:

```text
D1: S_A=.40, S_B=.60
D2: S_A=.30, S_B=.70
D3: S_A=.20, S_B=.80
D4: S_A=.10, S_B=.90
```

### Main prediction

The incentive for calibrated posterior inference scales with:

\[
q(1-q)(S_A-S_B)^2.
\]

Therefore:

\[
\boxed{
\Delta S\uparrow
\Rightarrow
\text{greater pressure for posterior averaging}
}
\]

especially under intermediate cue uncertainty.

---

# 11. Environmental pressure 4: outcome reliability

The network uses observed outcomes to learn what the A/B states imply.

Generate:

\[
y_t=S_{z_t}(t)+\epsilon_t
\]

with:

\[
\epsilon_t\sim\mathcal N(0,\sigma_y^2).
\]

Test:

```text
Y0 = 0
Y1 = low
Y2 = moderate
Y3 = high
```

Suggested values:

```text
0
0.01
0.02
0.05
```

Outcome noise should not be the headline manipulation, but it may determine whether latent-state inference is learnable.

Prediction:

> posterior averaging should be easiest when contextual uncertainty exists but outcome feedback remains informative.

---

# 12. Environmental pressure 5: latent-state stability

The network may need a reason to maintain distinct A/B latent values.

Use three environment variants while keeping the exact 10A/10B curriculum.

## S1 — Fixed global states

\[
S_A,S_B
\]

are fixed across all training.

This is the easiest proof that graded posterior mapping can emerge.

## S2 — Fixed per run / participant

Draw:

\[
S_A,S_B
\]

at the start of each simulated learner/run and hold them fixed during that run.

This requires the SRN to infer the two values from experience.

## S3 — Context-specific learning

Restore slowly evolving states:

\[
S_A(t),S_B(t).
\]

Practice updates only the active state.

This is the closest v1 environment to the human learning interpretation.

These are **environmental structures**, not alternative training curricula.

Every variant still uses:

```text
A × 10
B × 10
A × 10
B × 10
...
```

---

# 13. Core factorial search

The primary scientific search should focus on:

\[
ContextValidity
\times
CueUncertainty
\times
StateSeparation.
\]

Start with S1 fixed states and zero/low outcome noise.

Recommended discovery grid:

```text
6 validity levels
×
5 cue-accuracy levels
×
4 state separations
=
120 environmental conditions
```

Use:

```text
5 network seeds / condition
```

This produces a phase diagram of where posterior averaging emerges.

After identifying the relevant region, test outcome noise and S2/S3 only within promising conditions.

---

# 14. Why this is better than changing training duration

The scientific question is:

> What makes averaging useful?

Changing epochs or giving the model progressively easier tasks answers a different question:

> Under what optimization conditions can we force the model to learn it?

That is not the main target.

Accordingly:

- training exposure stays fixed;
- switch schedule stays fixed;
- hidden size stays fixed at 8;
- optimizer stays fixed.

The environment changes.

---

# 15. Emergence speed

Posterior averaging should not require prolonged training.

Checkpoint the network after each completed A/B block pair:

```text
after switch 1
after switch 2
after switch 3
after switch 4
...
```

At each checkpoint evaluate on the same small diagnostic battery.

Export:

```text
switch_number
posterior_mean_R2
implied_q_R2
implied_q_slope
endpoint_error
```

Primary emergence-speed question:

> **After how many A↔B switches does the network begin to approximate the Bayes-optimal mapping?**

A compelling result would show substantial posterior averaging within the first few switches.

---

# 16. Exact Bayesian benchmark

The oracle should match the actual environmental generative model.

The prior comes from nominal context validity:

\[
P(z_t=A\mid c_t).
\]

The likelihood comes from graded cue evidence:

\[
P(x_t\mid z_t).
\]

Thus:

\[
P(z_t=A\mid c_t,x_t)
\propto
P(x_t\mid z_t=A)
P(z_t=A\mid c_t).
\]

When necessary, prior outcome history may also contribute to inference over \(S_A,S_B\).

The exact posterior is:

\[
q_t^*
=
P(z_t=A\mid c_t,x_t,H_{t-1}).
\]

The normative prediction is:

\[
y_t^*
=
q_t^*S_A(t)
+
(1-q_t^*)S_B(t).
\]

The oracle must use every source of information genuinely available in the environment.

Do **not** compute a “Bayesian” target that ignores the predictable 10A/10B block structure.

---

# 17. Counting / block-phase shortcut diagnostic

Because the schedule is exactly:

```text
10 A
10 B
10 A
10 B
```

the SRN may learn block phase.

That is allowed.

However, we need to know whether the model is using:

1. only block phase;
2. only the current cue;
3. or a Bayesian combination of prior and likelihood.

Create held-out cue-conflict diagnostics.

Example:

```text
nominal A block
strong B-like cue
```

and:

```text
nominal B block
strong A-like cue
```

Compare predictions against:

### Phase-only model

Uses:

\[
P(z\mid c)
\]

only.

### Cue-only model

Uses:

\[
P(z\mid x)
\]

only.

### Full Bayes model

Uses:

\[
P(z\mid c,x).
\]

A successful posterior-inference SRN should best align with the full Bayes model.

---

# 18. Ambiguous diagnostic battery

Evaluate:

```text
q* ≈ .05
.10
.20
.35
.50
.65
.80
.90
.95
```

using combinations of nominal block context and cue evidence that generate those posterior values.

For each diagnostic state export:

```text
nominal_context
block_position
x_t
prior_q
likelihood_ratio
q_star
S_A
S_B
bayes_mean
SRN_prediction
implied_q
```

---

# 19. Success metrics

## Posterior-mean alignment

\[
R^2(\hat y,y^*).
\]

## Implied responsibility

\[
\hat q
=
\frac{\hat y-S_B}{S_A-S_B}.
\]

Compare \(\hat q\) with \(q^*\).

Report:

```text
R²
slope
intercept
MAE
```

## Monotonicity

As \(q^*\) increases, \(\hat q\) should increase.

## Endpoint fidelity

At near-certain A:

\[
\hat y\approx S_A.
\]

At near-certain B:

\[
\hat y\approx S_B.
\]

## Full-Bayes advantage

Compare SRN predictions to:

```text
phase-only prediction
cue-only prediction
full-Bayes prediction
global-mean prediction
hard-MAP prediction
```

This is critical.

The goal is not merely “intermediate output.”

The goal is specifically:

> **Does the network behave most like the Bayes-optimal posterior mean?**

---

# 20. Behavioral success gate

A seed passes if:

```text
posterior_mean_R2 >= .80
implied_q_R2 >= .70
implied_q_slope ∈ [.70, 1.30]
endpoint error <= 10% of state separation
monotonicity passes
full-Bayes fit > phase-only fit
full-Bayes fit > cue-only fit
full-Bayes fit > hard-MAP fit
```

Do not count a network that predicts the global midpoint for every ambiguous state as successful.

---

# 21. Discovery and confirmation

## Discovery

Use:

```text
5 seeds / environmental condition
```

A condition is promising if:

```text
>= 4 / 5 seeds
```

pass.

## Confirmation

Freeze the environmental parameters of the best mechanistically interpretable regimes.

Run:

```text
20 fresh seeds
```

without changing:

- architecture;
- training duration;
- learning rate;
- schedule;
- environmental parameters.

A condition is confirmed if:

```text
>= 16 / 20 seeds
```

pass.

---

# 22. Main predicted phase diagram

The central prediction is not simply:

\[
\text{more uncertainty}
\rightarrow
\text{more averaging}.
\]

Instead:

### Perfectly valid context

Hard selection is sufficient.

### Moderately imperfect context + informative graded cues

Posterior inference is valuable.

### Nearly useless context and cues

Global averaging becomes competitive.

Therefore the expected region is:

\[
\boxed{
\text{intermediate uncertainty}
\times
\text{large enough state separation}
\rightarrow
\text{Bayesian averaging}
}
\]

This is the main scientific target.

---

# 23. Hidden-state analysis

Only run after behavioral confirmation.

Probe for:

\[
S_A,
\quad
S_B,
\quad
q^*,
\quad
\log\frac{q^*}{1-q^*}.
\]

Also probe:

```text
nominal block context
block position
current cue x
cue log-likelihood ratio
```

This determines whether the network represents:

- the prior;
- the likelihood;
- the posterior;
- or merely a direct output mapping.

Use held-out runs/seeds.

---

# 24. Causal decomposition

If posterior information is decodable, test whether the hidden state contains separable components corresponding to:

\[
\text{prior}
+
\text{likelihood}
\rightarrow
\text{posterior}.
\]

Potential intervention sequence:

1. hold cue fixed and alter a prior/block-phase direction;
2. hold phase fixed and alter cue-evidence direction;
3. test whether output changes as predicted by Bayes combination.

This would be a stronger mechanistic result than merely decoding \(q^*\).

---

# 25. Environmental ablations

For any confirmed regime, remove one pressure at a time.

## A1 — Restore perfect context validity

Set:

\[
v=1.
\]

Prediction:

posterior averaging should diminish because uncertainty is unnecessary.

## A2 — Collapse state separation

Make:

\[
S_A\approx S_B.
\]

Prediction:

posterior calibration should become less behaviorally valuable.

## A3 — Remove graded cue evidence

Make cues perfectly separable.

Prediction:

hard assignment becomes sufficient.

## A4 — Destroy cue informativeness

Make cues nearly uninformative.

Prediction:

global-average behavior may replace calibrated posterior averaging.

These are the strongest causal tests of the environmental-pressure hypothesis.

---

# 26. What not to vary

Do not include the following in the primary search:

```text
hidden size
number of blocks
number of switches
training epochs
episode length
optimizer
learning-rate schedule
curriculum order
A/B block length
```

The training structure is deliberately held constant.

---

# 27. Required TDD tests

## T1 — Exact FTT block schedule

Assert:

```text
10 A
10 B
10 A
10 B
...
```

for every generated training run.

No condition may alter block length or switch frequency.

## T2 — Fixed training exposure

All conditions must contain the same number of training trials and weight updates.

## T3 — Eight-unit SRN

Primary models must use:

```text
hidden_dim = 8
```

## T4 — Context validity

Empirically verify:

\[
P(z=c)
\]

matches configured \(v\).

## T5 — Cue likelihood

Empirical cue distributions must match specified Gaussian parameters.

## T6 — Cue Bayes accuracy

Observed classification accuracy must match target level within tolerance.

## T7 — State separation

Generated \(S_A,S_B\) must match configured \(\Delta S\).

## T8 — No target leakage

Current outcome \(y_t\) must never enter the timestep-\(t\) input.

## T9 — Exact posterior

For hand-computed test cases, oracle \(q^*\) must match analytic Bayes calculations.

## T10 — Full information oracle

Oracle must include the nominal-context prior implied by the FTT block schedule / validity structure.

## T11 — Cue-conflict tests

Phase-only, cue-only, and full-Bayes predictions must be distinguishable on constructed diagnostic states.

## T12 — Flat midpoint rejection

A constant midpoint predictor must fail the behavioral success gate.

## T13 — Emergence checkpoints

Identical diagnostics must be run after each A/B block pair.

## T14 — Discovery/confirmation separation

No confirmation seed may appear in discovery.

---

# 28. Proposed code structure

```text
srn_environmental_pressures/
    config.py

    environment/
        ftt_schedule.py
        latent_context.py
        cue_evidence.py
        skill_states.py
        outcomes.py
        bayes_oracle.py

    models/
        elman_srn.py

    training/
        trainer.py
        checkpoints.py

    discovery/
        environmental_grid.py
        condition_registry.py
        selection.py

    confirmation/
        confirm_regime.py
        seed_registry.py

    analysis/
        posterior_alignment.py
        implied_responsibility.py
        phase_vs_cue.py
        monotonicity.py
        emergence_speed.py
        hidden_probes.py
        interventions.py
        ablations.py

    plotting/
        phase_diagrams.py
        emergence_figures.py
        mechanism_figures.py

    pipeline.py

tests/
    test_ftt_schedule.py
    test_fixed_exposure.py
    test_hidden_dim.py
    test_context_validity.py
    test_cue_likelihood.py
    test_cue_accuracy.py
    test_state_separation.py
    test_no_leakage.py
    test_bayes_oracle.py
    test_full_information_oracle.py
    test_cue_conflicts.py
    test_midpoint_rejection.py
    test_emergence_checkpoints.py
    test_discovery_confirmation_split.py
```

---

# 29. Required outputs

```text
run_config.json

all_environmental_conditions.csv
all_seed_results.csv
condition_summary.csv

posterior_mean_alignment.csv
implied_responsibility.csv
ambiguity_diagnostics.csv

phase_only_predictions.csv
cue_only_predictions.csv
full_bayes_predictions.csv
model_similarity_comparison.csv

emergence_by_switch.csv

phase_diagram_validity_x_uncertainty.csv
phase_diagram_uncertainty_x_separation.csv

promising_regimes.csv
confirmation_results.csv
confirmed_regimes.csv

environmental_ablation_results.csv

hidden_state_probes.csv
causal_interventions.csv
```

---

# 30. Required figures

## Figure 1 — Fixed FTT training ecology

Show:

```text
A × 10 → B × 10 → A × 10 → B × 10
```

and distinguish:

```text
nominal context c
latent active state z
graded cue evidence x
```

## Figure 2 — Why posterior inference is useful

Show a nominal A trial where:

- block prior favors A;
- current cue partially favors B;
- posterior lands between them;
- output is posterior-weighted between \(S_A,S_B\).

## Figure 3 — Emergence by switch

X:

```text
number of completed A↔B switches
```

Y:

```text
posterior-mean R²
```

Separate environmental regimes.

## Figure 4 — Context validity × cue uncertainty

Heatmap:

```text
posterior-mean R²
```

## Figure 5 — Cue uncertainty × state separation

Heatmap:

```text
posterior-mean R²
```

## Figure 6 — q* calibration

Plot:

\[
q^*
\]

against:

\[
\hat q.
\]

## Figure 7 — Bayes versus shortcut models

Compare SRN similarity to:

```text
full Bayes
phase only
cue only
hard MAP
global midpoint
```

## Figure 8 — Environmental ablations

Show what happens when:

```text
context validity → 1
state separation → small
cue evidence → perfect
cue evidence → useless
```

## Figure 9 — Confirmatory replication

Fresh-seed distributions for the top confirmed regime.

---

# 31. Main decision logic

## Result A — Posterior averaging appears under intermediate context validity

Strong support for the hypothesis:

> **The network learns graded responsibility when contextual cues are useful but not perfectly reliable.**

## Result B — Averaging requires large A/B separation

Interpretation:

> The network only invests in calibrated context inference when choosing the wrong latent state has sufficient behavioral cost.

## Result C — Posterior averaging appears only with graded cue overlap

Interpretation:

> Continuous evidence strength is necessary for learning a continuous responsibility function.

## Result D — Network follows block phase rather than cue evidence

Interpretation:

> The deterministic FTT schedule provides an easier temporal shortcut.

The environment has not created enough uncertainty about the latent state.

## Result E — Network predicts a constant midpoint

Interpretation:

> The environment creates uncertainty but insufficient usable evidence for calibrated inference.

This is global averaging, not Bayesian averaging.

## Result F — Full Bayes behavior emerges within a few switches

This is the preferred result.

It would show that a simple recurrent learner can rapidly acquire:

\[
\text{prior}
+
\text{likelihood}
\rightarrow
\text{posterior-weighted behavior}
\]

under the same switching curriculum used in the FTT.

---

# 32. Definition of done

- [ ] Every training run uses exactly 10 A trials then 10 B trials repeatedly.
- [ ] Training exposure is identical across conditions.
- [ ] Main SRN has 8 hidden units.
- [ ] Training amount is not tuned.
- [ ] Optimizer is fixed across conditions.
- [ ] Context validity is manipulated.
- [ ] Cue evidence is graded and uncertainty is manipulated.
- [ ] A/B state separation is manipulated.
- [ ] Outcome reliability can be manipulated.
- [ ] Fixed, run-specific, and evolving state environments are available.
- [ ] Exact full-information Bayes posterior is computed.
- [ ] Block-phase-only and cue-only shortcut models are computed.
- [ ] Posterior averaging is evaluated after each A/B switch.
- [ ] The main search maps environmental conditions, not optimization conditions.
- [ ] Flat midpoint behavior is explicitly rejected as success.
- [ ] Discovery uses 5 seeds per condition.
- [ ] Promising regimes are confirmed on 20 fresh seeds.
- [ ] Environmental ablations test necessity of each pressure.
- [ ] Final report identifies which environmental pressures are necessary/sufficient for Bayes-optimal averaging.

---

# 33. Desired scientific endpoint

The target result is a statement of the form:

> **A simple recurrent learner does not average merely because its context is noisy. Bayes-optimal averaging emerges when the learner experiences an ecology in which context is predictive but fallible, contextual evidence varies continuously in strength, and the candidate context-specific states differ enough that committing to the wrong state is costly.**

The goal is to explain:

\[
\boxed{
\text{What environmental problem makes posterior averaging worth learning?}
}
\]

—not how much training or architectural tuning is required to force the network to do it.
