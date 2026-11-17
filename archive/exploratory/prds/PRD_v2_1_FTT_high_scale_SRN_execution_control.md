# PRD v2.1: FTT-Like High-Scale SRN With Emergent Execution Control

## 1. Purpose

The proof-of-concept work showed that a simple recurrent network can approach a Bayes-optimal posterior mean under the right environmental pressures.

The next goal is to move back toward the original finger-tapping / motor-sequence task and ask a stronger question:

> **Can graded expression of already-split context-specific skills emerge in a High-Scale Elman SRN that learns the same motor sequence under alternating contexts and is optimized both to predict the next sequence item and to execute efficiently?**

The model should remain as close as possible to the original FTT SRN.

The critical additions are:

1. strict one-hot sequence inputs;
2. a 3-dimensional context basis allowing Pure A, Pure B, Mixed, and Novel probes;
3. mandatory High-Scale initialization, because representational splitting is a prerequisite;
4. a continuous execution-control objective in addition to next-token prediction;
5. Mixed and Novel are **test-only probes**, never training conditions.

The intended mechanistic sequence is:

\[
\boxed{
\text{High Scale + strong context}
\rightarrow
\text{split A/B sequence representations}
}
\]

followed by:

\[
\boxed{
\text{uncertain / unfamiliar context}
\rightarrow
\text{graded expression of those split representations}
}
\]

---

# 2. Core scientific questions

This project asks three sequential questions.

## Q1 — Does the modified SRN still split?

With:

```text
hidden_dim = 8
init_scale = 0.5
strict one-hot sequence inputs
3D context basis
full context strength
```

does alternating A/B training still produce context-specific hidden-state geometry and transition costs?

This is a hard prerequisite.

## Q2 — Can the same network learn context-sensitive execution control?

When the SRN is additionally rewarded for efficient execution, does it learn distinct A/B execution policies grounded in its learned sequence state?

## Q3 — What happens at test under Mixed and Novel context?

After A/B skills are learned and the network is frozen, do Mixed and Novel probes produce execution behavior between the two pure-context policies?

No probe outcome is used for learning.

---

# 3. Architecture

Use a one-layer Elman SRN:

\[
h_t
=
\tanh(W_x x_t + W_h h_{t-1} + b_h).
\]

Primary hidden size:

```text
hidden_dim = 8
```

Two output heads share the same hidden state:

```text
                 ┌→ next-token logits
input → Elman SRN
                 └→ execution-speed signal
```

The shared hidden state is essential.

The goal is to test whether sequence representation and execution control become organized around the same context-specific internal states.

---

# 4. High-Scale initialization is mandatory

The main model must instantiate the same High-Scale principle used in the original splitting work.

Primary configuration:

```text
hidden_dim = 8
init_scale = 0.5
context_alpha = 1.0
learning_rate = reuse validated FTT value
optimizer = Adam
```

All trainable weight matrices are initialized:

\[
W \sim \mathcal N(0, 0.5^2)
\]

unless a parameter is explicitly exempted below.

Biases initialize to zero.

This is the **High-Scale** model.

## Low-Scale falsification control

Use an otherwise identical control:

```text
init_scale = 0.02
```

The Low-Scale model is not the headline model.

Its role is to verify that the new pipeline retains the known lumping-versus-splitting distinction.

---

# 5. Sequence representation

Replace Gaussian-smoothed sequence inputs with strict 4D one-hot vectors.

For key \(k\):

```text
Key 1 = [1,0,0,0]
Key 2 = [0,1,0,0]
Key 3 = [0,0,1,0]
Key 4 = [0,0,0,1]
```

Reuse the exact validated fixed sequence and sequence-indexing logic already present in the FTT repository.

Do not alter the motor sequence itself as part of this PRD.

The next-token target remains the next item in the fixed sequence.

---

# 6. Three-dimensional context basis

Use a 3D binary context-feature basis.

The dimensions correspond to:

```text
dimension 1 = learned Context A cue
dimension 2 = learned Context B cue
dimension 3 = Novel cue
```

## Pure training contexts

```text
A = [1,0,0]
B = [0,1,0]
```

These are strict one-hot context inputs.

## Novel probe

```text
Novel = [0,0,1]
```

Novel is also a strict one-hot vector.

## Mixed probe

Mixed contains both learned context cues simultaneously.

Primary encoding:

\[
Mixed =
\left[
\frac{1}{\sqrt 2},
\frac{1}{\sqrt 2},
0
\right].
\]

This is a norm-matched superposition of the A and B one-hot basis vectors.

The norm match prevents Mixed from receiving a larger raw context-input magnitude merely because two features are active.

### Sensitivity check

Also evaluate:

```text
Mixed_additive = [1,1,0]
```

This corresponds to both cues being fully present.

The primary scientific conclusion must not depend solely on the larger norm of `[1,1,0]`.

---

# 7. Novel-context input initialization

The third context feature is never active during A/B training.

If its input weights were randomly initialized at High Scale and left untouched, Novel behavior would depend arbitrarily on random untrained weights.

Therefore:

```text
W_input[:, novel_context_dimension] = 0
```

at initialization.

Keep this column fixed at zero during A/B training.

Interpretation:

> The novel cue has no learned association with either established context-specific skill.

Thus the Novel probe represents **absence of learned A/B contextual evidence**, not an arbitrary random context association.

## Sensitivity analysis

After the primary analysis, optionally evaluate an untrained-random Novel column initialized at the High-Scale value.

This is diagnostic only.

Do not use it as the main Novel result.

---

# 8. FTT training schedule

The training curriculum must match the validated FTT alternation structure.

Hard contract:

```text
10 sequences in Context A
10 sequences in Context B
10 sequences in Context A
10 sequences in Context B
...
```

Reuse:

- the validated total number of blocks;
- counterbalance logic;
- sequence repetitions;
- online weight-update timing;

from the existing FTT SRN pipeline.

Do not search or tune:

```text
block length
number of blocks
switch frequency
training duration
```

The environmental / objective manipulation is the scientific variable, not additional training exposure.

---

# 9. Pure-context training only

The SRN is trained only on:

```text
A
B
```

Never train on:

```text
Mixed
Novel
```

Mixed and Novel are frozen-model generalization probes.

No gradient updates are permitted during those probes.

This is mandatory for the emergence claim.

---

# 10. Objective 1: next-token prediction

Retain the original sequence-learning objective.

From hidden state \(h_t\):

\[
\ell_t = W_{\text{seq}}h_t+b_{\text{seq}}.
\]

Sequence loss:

\[
L_{\text{seq}}
=
CE(\ell_t,k_{t+1}).
\]

The model must learn the same fixed sequence in both A and B.

---

# 11. Objective 2: continuous execution control

Add a second head:

\[
v_t
=
\sigma(W_vh_t+b_v).
\]

where:

```text
v_t ∈ [0,1]
```

is normalized execution speed / vigor.

Interpretation:

```text
higher v = faster execution
lower v = slower / more cautious execution
```

Define normalized response time:

\[
RT_t^{norm}=1-v_t.
\]

The SRN is **not** given a target RT.

It chooses \(v_t\) to trade speed against execution error.

---

# 12. No externally tracked context-specific competence states

Do **not** maintain external context-specific variables such as:

\[
C_A(t),\qquad C_B(t).
\]

Do not update, store, or provide any hand-built A/B competence trace.

The context itself is supplied directly to the SRN through the 3D context vector:

```text
A     = [1,0,0]
B     = [0,1,0]
Novel = [0,0,1]
Mixed = A+B superposition
```

The model should learn whatever context-specific skill state it needs inside its recurrent hidden representation.

This is a hard mechanistic constraint:

> **Any distinction between “how well I know the sequence in A” and “how well I know the sequence in B” must emerge inside the SRN rather than being represented by an external bookkeeping variable.**

---

# 13. Execution control is grounded in the model's own sequence confidence

The execution head outputs:

\[
v_t
=
\sigma(W_v h_t+b_v),
\]

with:

```text
higher v = faster / more aggressive execution
lower v  = slower / more cautious execution
```

The next-token head produces:

\[
p_t
=
softmax(\ell_t)_{k_{t+1}},
\]

where \(p_t\) is the probability assigned to the correct next sequence item.

Use this **current sequence confidence** as the quantity determining how safely the model can execute quickly.

Do not create a separate target or latent variable for skill.

Thus:

\[
\boxed{
\text{context + sequence history}
\rightarrow
h_t
\rightarrow
\begin{cases}
\text{next-token confidence }p_t\\
\text{execution speed }v_t
\end{cases}
}
\]

If High-Scale context learning creates distinct A/B hidden states, those states can naturally support context-specific sequence confidence, execution policy, or both.

---

# 14. Speed–accuracy environment

Execution speed should have a benefit but become costly when the model is not sufficiently confident in the upcoming response.

Define:

\[
P_{\text{motor-correct},t}
=
\sigma\left[
\kappa
\left(
p_t^{\text{detach}}-v_t-\delta
\right)
\right].
\]

Where:

- \(p_t^{\text{detach}}\) is correct-token probability detached from the execution-loss graph;
- \(v_t\) is chosen execution speed;
- \(\delta\) is an optional safety margin;
- \(\kappa\) controls how sharply excessive speed becomes risky.

**Detach \(p_t\) inside the execution-cost pathway.**

This prevents the network from reducing execution cost by artificially changing sequence confidence instead of learning an appropriate execution-speed policy.

Sequence logits continue to be trained normally by cross-entropy.

---

# 15. Execution objective

Define normalized response time:

\[
RT_t^{norm}=1-v_t.
\]

Execution cost:

\[
L_{\text{exec},t}
=
c_{\text{error}}
\left(
1-P_{\text{motor-correct},t}
\right)
+
c_{\text{time}}
RT_t^{norm}.
\]

Total loss:

\[
\boxed{
L_{\text{total}}
=
L_{\text{seq}}
+
\lambda_{\text{exec}}L_{\text{exec}}.
}
\]

The model is never given:

```text
target RT
target speed
target context-specific skill
C_A
C_B
target Mixed behavior
target Novel behavior
```

The execution policy must emerge from the SRN's internal sequence representation and confidence.

---

# 16. Why this is different from RT regression

Do not train:

\[
(\hat{RT}-RT_{\text{target}})^2.
\]

There are no RT labels such as:

```text
A = fast
B = slow
Mixed = intermediate
Novel = intermediate
```

Instead, the model solves:

> **How quickly can I safely execute the response that my current internal state is preparing?**

The RT-like quantity is:

\[
RT_t^{norm}=1-v_t,
\]

which emerges from optimization of a speed–accuracy tradeoff.

---

# 17. Calibrating the execution environment

The environmental constants:

```text
kappa
delta
c_error
c_time
lambda_exec
```

must be calibrated **before** Mixed or Novel probes are inspected.

Do not tune them to make probe outputs intermediate.

Before SRN training, evaluate the objective over a grid of:

```text
sequence confidence p ∈ [0,1]
execution speed v ∈ [0,1]
```

Required properties:

1. optimal speed increases monotonically with sequence confidence;
2. low-confidence responses have lower optimal speed;
3. high-confidence responses support faster execution;
4. the optimum is not pinned to \(v=0\) or \(v=1\) across the relevant confidence range;
5. the execution loss does not overwhelm sequence learning.

Export:

```text
speed_accuracy_calibration.csv
```

and a calibration figure.

Once selected, freeze these constants for all seeds and conditions.

---

# 18. Training-time execution learning

On each pure A/B training step:

1. provide the current sequence one-hot and current 3D context vector;
2. compute hidden state \(h_t\);
3. compute next-token logits \(\ell_t\);
4. compute correct-token probability \(p_t\);
5. compute execution speed \(v_t\);
6. compute sequence cross-entropy;
7. compute execution cost using **detached** sequence confidence;
8. backpropagate the combined loss.

There is no separate context-specific competence update.

All A/B skill history must be represented by the SRN weights and recurrent hidden dynamics.

Mixed and Novel remain frozen no-update probes.

---

# 19. Stage 1 hard gate: verify representational splitting first

Before interpreting any Mixed or Novel probe, the High-Scale model must show the established splitting phenotype.

Compare High-Scale:

```text
init_scale = 0.5
```

against Low-Scale:

```text
init_scale = 0.02
```

with all other parameters identical.

Required diagnostics:

### G1 — Sequence mastery

Both model types must learn the sequence adequately.

Splitting cannot be interpreted if High-Scale simply fails the task.

### G2 — A/B hidden-state separation

Using shared PCA / the validated geometry pipeline, compute A/B trajectory-centroid distance.

High-Scale must show substantially larger A/B separation than Low-Scale.

### G3 — Transition cost

Measure loss / execution cost immediately around A↔B switches.

High-Scale should show a stronger context-transition signature than Low-Scale.

### G4 — Cross-context transfer

Probe a trained pure-A representation with B context and vice versa.

High-Scale should show reduced cross-context generalization relative to Low-Scale.

### G5 — Context input use

A/B context dimensions should receive meaningful input-to-hidden weight.

The context signal must actually participate in the hidden representation.

---

# 20. Splitting gate decision

Do not proceed to mechanistic interpretation of Mixed/Novel unless the High-Scale cohort reproduces the qualitative splitting signature:

\[
\boxed{
\text{High-Scale A/B separation}
>
\text{Low-Scale A/B separation}
}
\]

and:

\[
\boxed{
\text{High-Scale transition cost}
>
\text{Low-Scale transition cost}.
}
\]

For the final run, use enough random seeds to estimate these reliably.

Recommended:

```text
20 High-Scale seeds
20 Low-Scale seeds
```

A smaller 5-seed pilot is acceptable for debugging.

If the splitting gate fails, stop.

Do not “fix” Mixed/Novel output before the prerequisite split representation exists.

---

# 21. Stage 2 gate: verify that execution speed is grounded in learned sequence confidence

Before Mixed/Novel probing, verify that the execution head has learned the intended speed–accuracy relationship.

For pure A/B training observations, compute:

\[
p_t
=
softmax(\ell_t)_{k_{t+1}}
\]

and:

\[
RT_t^{norm}=1-v_t.
\]

Primary structural prediction:

\[
p_t\uparrow
\Rightarrow
v_t\uparrow
\Rightarrow
RT_t^{norm}\downarrow.
\]

Required checks:

1. execution speed is positively associated with correct-token confidence;
2. this relationship holds within pure A and pure B contexts;
3. the relationship is not explained solely by block number or context label;
4. A/B execution differences occur only insofar as their learned internal sequence states differ.

The model does **not** need A and B to have permanently different RTs.

---

# 22. Pure-context counterfactual readout

At every probe checkpoint, freeze the model and compute four context readouts using the same sequence state:

```text
A
B
Mixed
Novel
```

The pure A/B predictions define the established endpoints:

\[
RT_A,\qquad RT_B.
\]

Probe interpolation is evaluated relative to these within-network endpoints.

---

# 23. Mixed probe

Primary context input:

\[
Mixed =
[1/\sqrt2,\ 1/\sqrt2,\ 0].
\]

No learning occurs.

Measure:

- next-token loss;
- execution speed;
- normalized RT;
- hidden-state trajectory;
- location relative to A/B hidden-state geometry.

Define behavioral interpolation index:

\[
I_{\text{Mixed}}
=
\frac{
RT_{\text{Mixed}}-RT_B
}{
RT_A-RT_B
}
\]

when the denominator is sufficiently large.

Interpretation:

```text
I ≈ 1 → A-like
I ≈ 0 → B-like
0 < I < 1 → between A and B
```

Use midpoint deviation as the primary stable statistic when A/B are close.

---

# 24. Novel probe

Primary context input:

```text
Novel = [0,0,1]
```

with the novel input column neutral / zero-associated.

No learning occurs.

Measure the same quantities as Mixed.

The key question is whether the unassociated context produces:

- A-like expression;
- B-like expression;
- an intermediate/default expression;
- or some other stable policy.

Do not assume in advance that Novel must equal Mixed.

The human result is an external target, not a training constraint.

---

# 25. Mixed versus Novel scientific test

The most interesting successful outcome is:

1. A/B representations are demonstrably split;
2. A/B execution policies differ when competence differs;
3. Mixed falls between the A/B endpoints;
4. Novel also falls between A/B endpoints;
5. neither probe was trained;
6. interpolation scales with each network's own A/B separation.

This would support:

> **An SRN can learn separate context-specific sequence/control states and nevertheless express a graded combination when the current context does not uniquely select one learned state.**

---

# 26. Participant/network-level structural prediction

For network \(i\):

\[
RT_{\text{probe},i}
-
\frac{
RT_{A,i}+RT_{B,i}
}{2}
\]

should be related to its own pure-context separation:

\[
RT_{A,i}-RT_{B,i}.
\]

More generally, infer:

\[
q_i
=
\frac{
RT_{\text{probe},i}-RT_{B,i}
}{
RT_{A,i}-RT_{B,i}
}.
\]

Test whether Mixed and Novel responsibilities are stable across seeds with sufficient A/B separation.

Do not base the main claim only on group means.

---

# 27. Connection to the Bayes-optimal proof of concept

The previous synthetic work showed that simple recurrence can learn an approximately graded posterior-mean mapping under conditions where latent-state identity is uncertain.

v2 does **not** explicitly train a Bayesian posterior.

Instead, it asks whether the same general computation appears when:

- A/B are actual context-specific sequence representations;
- the SRN is High-Scale enough to split;
- the network's own sequence confidence constrains safe execution speed;
- the output is a motor-control variable rather than an RT prediction target.

This is the psychologically stronger test.

---

# 28. Hidden-state geometry analyses

For frozen High-Scale networks collect trajectories under:

```text
A
B
Mixed
Novel
```

Use the same shared PCA / Procrustes-compatible geometry approach as the prior SRN work.

Measure:

```text
A-B centroid distance
Mixed distance to A
Mixed distance to B
Novel distance to A
Novel distance to B
projection onto A-B axis
off-axis distance
```

Do not assume that intermediate behavior requires the hidden state itself to sit exactly on the A-B line.

Behavioral interpolation and representational interpolation are separate hypotheses.

---

# 29. Execution-head mechanistic analysis

Ask whether the execution head is reading out an A/B-combined state.

For each probe:

1. save hidden state \(h\);
2. save execution output \(v\);
3. project \(h\) onto the pure A-B representational axis;
4. test whether execution speed changes monotonically with that projection.

Also inspect:

\[
W_v
\]

relative to the A-B hidden-state difference vector.

This can reveal whether the timing head directly reads the split geometry.

---

# 30. Low-Scale control predictions

Low-Scale networks should remain comparatively lumped.

If so, they may show:

- smaller A/B hidden-state separation;
- smaller A/B execution separation;
- less meaningful notion of “averaging two skills” under Mixed/Novel.

This is useful.

The proposed theory is specifically about:

\[
\boxed{
\text{recombination of already-split representations}.
}
\]

If a Low-Scale network produces the same probe behavior despite never splitting, then similar output can arise from a different mechanism and the mechanistic claim must be narrowed.

---

# 31. Probe timing

Probe the network at human-relevant training checkpoints, especially after multiple A/B switches have already established context-specific representations.

At minimum include:

```text
early after initial A/B exposure
after several A↔B switches
human-matched Mixed/Novel checkpoint
final training checkpoint
```

No probe causes an optimizer step.

---

# 32. Main model conditions

The core comparison is intentionally small.

## H — High-Scale main model

```text
hidden_dim = 8
init_scale = 0.5
context_alpha = 1.0
next-token + execution objective
```

## L — Low-Scale control

```text
hidden_dim = 8
init_scale = 0.02
context_alpha = 1.0
next-token + execution objective
```

## H-seq-only — objective control

Optional but strongly recommended:

```text
hidden_dim = 8
init_scale = 0.5
context_alpha = 1.0
next-token objective only
```

This determines whether the execution objective is necessary for RT-like interpolation beyond representational splitting alone.

---

# 33. Key factorial logic

The most informative comparison is:

| Model | Split representation expected? | Execution-control pressure? |
|---|---:|---:|
| Low-Scale + execution | weak | yes |
| High-Scale sequence-only | yes | no |
| High-Scale + execution | yes | yes |

The target hypothesis predicts the strongest human-like probe behavior specifically in:

```text
High-Scale + execution objective
```

because it has both:

1. separate context-specific states;
2. a reason for those states to control a continuous performance variable.

---

# 34. TDD requirement

All implementation follows:

\[
\boxed{
RED\rightarrow GREEN\rightarrow REFACTOR
}
\]

---

# 35. RED → GREEN tests

## T1 — 4D token one-hot

Each sequence token must be represented by exactly one active element.

No Gaussian smoothing is permitted.

## T2 — 3D context basis

Assert:

```text
A     = [1,0,0]
B     = [0,1,0]
Novel = [0,0,1]
```

## T3 — Mixed norm-matched encoding

Assert:

\[
Mixed=
[1/\sqrt2,1/\sqrt2,0]
\]

and:

\[
\|Mixed\|_2
=
\|A\|_2
=
\|B\|_2
=1.
\]

## T4 — Novel column neutral

Assert the novel context input column initializes to exactly zero and remains zero throughout pure A/B training.

## T5 — High-Scale initialization

All non-exempt trainable weight matrices in the High-Scale model must have initialization SD approximately:

```text
0.5
```

within sampling tolerance.

## T6 — Low-Scale initialization

Low-Scale control initialization SD approximately:

```text
0.02
```

## T7 — Full context strength

Pure context inputs must enter with:

```text
context_alpha = 1.0
```

## T8 — Exact FTT block schedule

Training must follow:

```text
A × 10
B × 10
A × 10
B × 10
...
```

using the existing validated block count / counterbalance.

## T9 — No Mixed/Novel training

Assert optimizer step count does not change during Mixed or Novel probes.

## T10 — No external competence state

Assert that the environment contains no mutable A/B competence traces or equivalent bookkeeping variables.

## T11 — Detached confidence in execution loss

The correct-token probability used by the execution loss must be detached.

The execution objective may train the execution head and shared hidden representation through its own path, but it must not directly increase sequence confidence by backpropagating through the scalar \(p_t\) used as the speed-capacity signal.

## T12 — Speed monotonicity

Before network training, the calibrated execution environment must satisfy:

```text
higher competence → higher optimal speed
```

across the relevant competence range.

## T13 — Interior execution solution

The calibrated execution objective must not produce only speed=0 or speed=1 across the expected competence range.

## T14 — Sequence head works without timing head

Sequence-only High-Scale control must still learn the sequence.

## T15 — Splitting gate plumbing

Known synthetic separated hidden trajectories must pass the geometry gate; identical trajectories must fail it.

## T16 — Probe counterfactual identity

A/B/Mixed/Novel probe passes must use identical network weights and sequence state; only context input may change.

## T17 — Mixed additive sensitivity

The pipeline must separately label:

```text
Mixed_norm
Mixed_additive
```

and never pool them silently.

---

# 36. Code structure

```text
ftt_srn_execution_v2/
    config.py

    inputs/
        token_onehot.py
        context_basis.py

    models/
        elman_srn.py
        sequence_head.py
        execution_head.py
        initialization.py

    environment/
        speed_accuracy.py
        calibration.py

    training/
        ftt_schedule.py
        trainer.py
        checkpoints.py

    diagnostics/
        splitting_gate.py
        transition_costs.py
        transfer_probe.py
        context_weights.py

    probes/
        pure_counterfactuals.py
        mixed_probe.py
        novel_probe.py

    analysis/
        interpolation.py
        structural_prediction.py
        hidden_geometry.py
        execution_geometry.py
        objective_controls.py

    plotting/
        training_figures.py
        splitting_figures.py
        probe_figures.py
        geometry_figures.py

    pipeline.py

tests/
    test_token_onehot.py
    test_context_basis.py
    test_mixed_norm.py
    test_novel_neutral.py
    test_high_scale_init.py
    test_low_scale_init.py
    test_context_strength.py
    test_ftt_schedule.py
    test_no_probe_training.py
    test_competence_isolation.py
    test_no_competence_input.py
    test_speed_monotonicity.py
    test_execution_interior.py
    test_sequence_learning.py
    test_splitting_gate.py
    test_probe_counterfactuals.py
    test_mixed_sensitivity.py
```

---

# 37. Required outputs

```text
run_config_v2.json

speed_accuracy_calibration.csv
training_dynamics.csv
execution_dynamics.csv

sequence_performance.csv
splitting_metrics.csv
transition_costs.csv
transfer_metrics.csv
context_weight_metrics.csv

pure_context_counterfactuals.csv
mixed_probe_results.csv
novel_probe_results.csv

probe_interpolation.csv
participant_structural_prediction.csv

hidden_state_geometry.csv
execution_geometry.csv

low_vs_high_scale.csv
sequence_only_vs_execution.csv

seed_summary.csv
```

---

# 38. Required figures

## Figure 1 — Model architecture

Show:

```text
4D sequence one-hot
+
3D context basis
        ↓
8-unit High-Scale Elman SRN
        ↓
shared hidden state
      ↙         ↘
next token   execution speed
```

## Figure 2 — High-Scale splitting prerequisite

Compare High- versus Low-Scale:

```text
A/B hidden trajectory separation
transition costs
cross-context transfer
```

## Figure 3 — Sequence confidence and execution speed

Plot:

```text
correct-token confidence
execution speed
normalized RT
context A/B
```

across alternating blocks.

## Figure 4 — Speed–accuracy calibration

Show optimal speed versus next-token confidence before neural training.

## Figure 5 — Pure A/B and Mixed probe

Show within-network:

```text
A endpoint
B endpoint
Mixed_norm
Mixed_additive
```

## Figure 6 — Pure A/B and Novel probe

Show:

```text
A endpoint
B endpoint
Novel
```

## Figure 7 — Structural prediction

X:

```text
A-B pure execution separation
```

Y:

```text
probe displacement from A/B midpoint
```

## Figure 8 — Hidden-state geometry

Show A/B/Mixed/Novel trajectories in shared representational space.

## Figure 9 — Scale × objective comparison

Compare:

```text
Low-Scale + execution
High-Scale sequence-only
High-Scale + execution
```

---

# 39. Main decision logic

## Result A — Splitting fails

If High-Scale no longer separates A/B after switching to strict one-hot sequence input / 8 hidden units:

> Stop.

The probe question is uninterpretable until the prerequisite splitting phenotype is restored.

Do not tune Mixed/Novel behavior.

## Result B — Splitting succeeds, but execution head does not track competence

Then the speed–accuracy objective is poorly specified or insufficient.

Fix the execution environment before interpreting probes.

## Result C — Splitting + distinct A/B execution, but Mixed/Novel arbitrary

Then split representations plus a speed–accuracy objective are insufficient to explain human averaging.

This is an informative negative result.

## Result D — Mixed interpolates, Novel does not

Then superposition of learned A/B cues can explain the Mixed result but not the third-context result.

Do not claim a general retrieval mechanism.

## Result E — Novel interpolates, Mixed does not

Then a default/no-learned-association mechanism may explain Novel but not cue conflict.

## Result F — Both Mixed and Novel interpolate

This is the target result.

If:

1. A/B are demonstrably split;
2. execution policies differ;
3. Mixed and Novel both fall between pure A/B endpoints;
4. no probe training occurred;
5. interpolation scales with each network's own A/B separation;

then conclude that a simple High-Scale recurrent learner can:

> **split identical motor sequences into context-specific representations during learning and express graded performance when the current context fails to select a unique learned state.**

---

# 40. Strongest mechanistic result

The strongest possible outcome would be:

\[
\boxed{
\text{High-Scale context learning}
\rightarrow
\text{split A/B hidden states}
}
\]

\[
\boxed{
\text{context-specific execution learning}
\rightarrow
\text{distinct A/B control policies}
}
\]

\[
\boxed{
\text{Mixed or Novel context}
\rightarrow
\text{intermediate execution}
}
\]

with the execution head reading directly from the same hidden geometry that carries the A/B split.

That would connect:

```text
representation
→ retrieval
→ motor expression
```

inside one simple recurrent system.

---

# 41. Definition of done

- [ ] 8-unit Elman SRN implemented.
- [ ] High-Scale initialization is explicitly `0.5`.
- [ ] Low-Scale control initialization is `0.02`.
- [ ] Full context strength is `1.0`.
- [ ] Sequence inputs are strict 4D one-hots.
- [ ] Context basis is 3D.
- [ ] A/B/Novel use strict one-hot context vectors.
- [ ] Mixed uses the preregistered A+B superposition.
- [ ] Novel input column is neutral rather than random.
- [ ] Exact FTT A10/B10 schedule is preserved.
- [ ] Mixed/Novel never receive training updates.
- [ ] Next-token objective is preserved.
- [ ] Continuous execution objective is added.
- [ ] No external A/B competence traces are maintained.
- [ ] Execution speed is grounded in detached next-token confidence rather than RT labels.
- [ ] Execution objective is calibrated before probe analysis.
- [ ] High-Scale splitting gate passes before probe interpretation.
- [ ] Low-Scale control verifies the scale dependence.
- [ ] Pure A/B execution endpoints are measured.
- [ ] Mixed probe is evaluated relative to those endpoints.
- [ ] Novel probe is evaluated relative to those endpoints.
- [ ] Network-level structural predictions are tested.
- [ ] Hidden-state geometry is linked to execution output.
- [ ] Sequence-only High-Scale control is included if computationally feasible.
- [ ] Final report clearly distinguishes representational splitting from behavioral interpolation.

---

# 42. Target paper-level claim

The intended claim is not simply that an RNN can generate intermediate RT.

It is:

> **A high-flexibility recurrent learner first uses context to partition identical motor experience into separate skill representations. When those context-specific states also govern how aggressively the sequence can be executed, ambiguous or unassociated contextual input can produce graded motor expression without retraining or explicitly averaging the learned skills.**

Compactly:

\[
\boxed{
\text{Context splits the skill; uncertain retrieval recombines its expression.}
}
