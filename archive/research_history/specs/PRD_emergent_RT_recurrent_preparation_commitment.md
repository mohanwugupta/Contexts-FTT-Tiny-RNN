# PRD: Emergent Reaction Time from Recurrent Preparation and Learned Commitment

## Goal

Replace the current auxiliary execution-speed head with a more psychologically and computationally grounded mechanism in which **reaction time emerges from how many recurrent preparation cycles the SRN needs before committing to the next action**.

The key causal story should be:

\[
\boxed{
\text{Context}
\rightarrow
\text{configured recurrent sequence state}
\rightarrow
\text{time to commitment}
\rightarrow
\text{RT}
}
\]

The model should never receive a target RT.

The only pressures should be:

1. predict/execute the correct next sequence item;
2. avoid committing while the current recurrent state is insufficiently prepared;
3. avoid waiting unnecessarily.

This is intended to combine the useful intuitions from recurrent decision models, adaptive-computation/halting models, motor-control accounts of time/error tradeoffs, and the existing context-splitting SRN.

The fixed FTT training ecology must remain unchanged.

---

## 1. Hard constraints

Do **not** change:

- 8-unit tanh Elman SRN;
- High-scale initialization = **2.0**;
- existing Low-scale control value;
- 4-item motor sequence;
- A/B contexts;
- 3D context input basis;
- A10 → B10 → A10 → B10 ... blocked curriculum;
- 10 sequences per context block;
- total human-matched training exposure (170 sequences);
- existing counterbalance handling;
- next-token prediction objective;
- no-training status of Mixed and Novel probes.

Do **not**:

- provide RT targets;
- provide target execution speeds;
- provide training sequence number as an input;
- provide an externally calculated skill variable;
- regress against human RT;
- tune on human Mixed/Novel data;
- tune on human transition profiles;
- hard-code RT to fall with practice;
- retain the current unconstrained execution-speed head in the primary model.

The primary model must derive RT from recurrent preparation time.

---

## 2. Context representation

Keep the existing 3D context basis.

Pure training contexts:

\[
A=[1,0,0]
\]

\[
B=[0,1,0].
\]

Frozen probes:

\[
Novel=[0,0,1].
\]

Primary Mixed probe:

\[
Mixed_{norm}=[0.5,0.5,0].
\]

If already supported, retain:

\[
Mixed_{additive}=[1,1,0]
\]

as an exploratory encoding control only.

Mixed and Novel must never be used for training or hyperparameter selection.

---

## 3. Core architectural change

### Previous architecture

Conceptually:

\[
[token_t, context_t]
\rightarrow
h_t
\rightarrow
\begin{cases}
\text{next-token logits}\\
\text{execution-speed head}
\end{cases}
\]

The execution-speed head is removed from the primary model.

### New architecture

For every sequence transition, allow the same recurrent network to perform several **internal preparation cycles** before committing.

For transition \(k\):

\[
[token_k, context]
\rightarrow
h_{k,1}
\rightarrow
h_{k,2}
\rightarrow
...
\rightarrow
h_{k,m}.
\]

At each internal cycle \(m\), the network outputs:

1. **4-way next-token logits**
   \[
   z_{k,m}
   \]

2. **commit hazard**
   \[
   a_{k,m}\in(0,1)
   \]

where \(a_{k,m}\) is the probability of executing now rather than performing another internal recurrent update.

There is no speed output.

The behavioral RT analogue is the amount of internal preparation required before commitment.

---

## 4. Internal preparation dynamics

For a transition from the currently observed token \(x_k\):

\[
h_{k,m}
=
\tanh
\left(
W_x[x_k,c]
+
W_hh_{k,m-1}
+
b
\right).
\]

Use the **same current token and same context input during all preparation cycles for that transition**.

Do not advance the external sequence until the model commits.

The recurrent state at the end of the committed transition becomes the starting recurrent state for the next external sequence item.

This preserves genuine temporal continuity across the motor sequence and across context blocks.

---

## 5. Maximum preparation horizon

Use a small fixed maximum number of internal cycles:

```text
M_max = 8
```

as the initial default.

This is an internal computation cap, not additional FTT training exposure.

The model still receives exactly the same 170 training sequences.

At the final preparation cycle, commitment is forced:

\[
a_{M_{max}}=1.
\]

Run a later robustness check with:

```text
M_max ∈ {6, 10}
```

only after the main mechanism works.

Do not select M_max using human RT.

---

## 6. Differentiable commitment distribution

Use a PonderNet-style hazard formulation rather than high-variance policy-gradient training for the MVP.

For cycle \(m\):

\[
q_m
=
a_m
\prod_{j<m}(1-a_j)
\]

where \(q_m\) is the probability that the response occurs at preparation cycle \(m\).

Force:

\[
a_{M_{max}}=1
\]

so:

\[
\sum_m q_m=1.
\]

This gives an exact differentiable distribution over commitment times.

---

## 7. RT definition

Primary continuous RT analogue:

\[
RT_{cycles}
=
E[m]
=
\sum_m q_m m.
\]

Normalized RT for visualization/state-space fitting:

\[
RT_{norm}
=
\frac{E[m]-1}{M_{max}-1}.
\]

Thus:

- immediate commitment → \(RT_{norm}=0\);
- maximum preparation → \(RT_{norm}=1\).

Also save a discrete evaluation RT:

\[
RT_{hard}
=
\min\{m:a_m\ge .5\}
\]

with forced commitment at \(M_{max}\).

Primary analyses should use expected commitment time because it is stable and continuous.

The hard RT is a robustness diagnostic.

---

## 8. Next-token learning objective

Preserve next-token prediction as a primary learning pressure.

At every preparation cycle compute:

\[
CE_m
=
-\log P(y_k|h_{k,m}).
\]

Primary sequence loss:

\[
L_{seq}
=
\frac{1}{M_{max}}
\sum_m CE_m.
\]

This encourages the recurrent dynamics themselves to refine the action representation over preparation cycles.

Also log the CE at actual/expected commitment.

Do not replace the existing sequence-learning objective with an RT objective.

---

## 9. Commitment objective

The commit policy should trade off:

- committing while the action is unreliable;
- spending extra time preparing.

Use:

\[
L_{commit}
=
\sum_m
q_m
\left[
c_{error}(1-p_{correct,m})
+
c_{time}(m-1)
\right].
\]

where:

\[
p_{correct,m}=P(y_k|h_{k,m}).
\]

Interpretation:

- committing early when the current recurrent state points weakly to the correct action is risky;
- waiting another internal cycle costs time.

### Important gradient rule

For the primary implementation use:

\[
\operatorname{stopgrad}(p_{correct,m})
\]

inside \(L_{commit}\).

Therefore:

- next-token CE trains the recurrent sequence representation;
- commitment learns when to trust the representation;
- the commitment loss cannot improve by artificially manipulating sequence confidence.

Run a joint-gradient version only as a robustness analysis after the primary model succeeds.

---

## 10. Total loss

Primary:

\[
L
=
L_{seq}
+
\lambda_{commit}L_{commit}.
\]

Start with:

```text
lambda_commit = 1
c_error = 1
```

and search only over a small time-cost grid:

```text
c_time ∈ {0.01, 0.025, 0.05, 0.10}
```

Do not run a large kitchen-sink hyperparameter search initially.

The primary question is whether the architecture produces the correct qualitative behavior.

---

## 11. Analytical sanity check before SRN training

Before training any network, calculate the optimal commitment policy for synthetic confidence trajectories.

Create toy preparation trajectories such as:

```text
easy:
p = [.60, .80, .93, .98, ...]

medium:
p = [.35, .55, .72, .84, ...]

hard:
p = [.27, .32, .40, .52, ...]
```

Verify:

- easy trajectories favor earlier commitment;
- hard trajectories favor later commitment;
- increasing `c_time` produces earlier commitment;
- increasing `c_error` produces later commitment.

If these qualitative properties fail, stop before neural training.

---

## 12. TDD implementation

Use RED → GREEN → REFACTOR.

### T1 — commitment probabilities normalize

Require:

\[
\sum_m q_m=1
\]

within numerical tolerance.

### T2 — forced final commitment

Require remaining probability mass to be assigned at `M_max`.

### T3 — RT monotonicity

Synthetic earlier-hazard distributions must yield lower expected RT.

### T4 — time-cost effect

Holding confidence trajectory fixed:

```text
higher c_time -> lower expected RT
```

### T5 — error-cost effect

Holding confidence trajectory fixed:

```text
higher c_error -> higher expected RT
```

when later cycles improve accuracy.

### T6 — no RT supervision

Training batches must contain no:

```text
target_rt
target_speed
human_rt
trial_number input
```

### T7 — external training exposure unchanged

Require exactly:

```text
170 external training sequences
10 sequences per A/B block
```

regardless of internal preparation cycles.

### T8 — probe freezing

Mixed and Novel evaluations must cause:

```text
zero optimizer steps
zero parameter updates
```

### T9 — context vectors

Regression-test:

```text
A
B
Mixed_norm
Novel
```

against the frozen definitions.

---

## 13. Phase 1 — five-seed mechanism screen

Run only:

```text
High scale = 2 + learned commitment
5 seeds
170 sequences
```

for each `c_time`.

Do not run downstream causal analyses yet.

Save per transition and preparation cycle:

```text
seed
training_sequence
context
sequence_position
prep_cycle
hidden_state
p_correct
sequence_CE
commit_hazard
commit_probability
expected_RT
hard_RT
```

---

## 14. Primary behavioral gates

A candidate time cost passes only if it satisfies the following.

### Gate A — learning

Across training:

\[
CE_{late}<CE_{early}.
\]

### Gate B — RT acquisition direction

Across training:

\[
RT_{late}<RT_{early}.
\]

This is the major gate that the previous execution-head model failed.

### Gate C — uncertainty grounding

Within seed:

\[
p_{correct}\uparrow
\Rightarrow
RT\downarrow.
\]

Report within-seed slopes.

Require the correct sign in at least:

```text
4/5 seeds
```

during screening.

### Gate D — preparation dynamics

Within transitions, confidence should generally improve across preparation cycles:

\[
p_{m+1}>p_m
\]

often enough for waiting to have a meaningful computational function.

If recurrent preparation does not improve action readiness, report this explicitly.

### Gate E — nondegenerate RT

Reject settings where:

```text
>90% of responses commit at cycle 1
```

or:

```text
>90% commit at M_max.
```

### Gate F — context structure survives

Execution time should not become completely context-insensitive.

At minimum retain measurable:

```text
A/B difference and/or
switch-related RT effect
```

after several switches.

Do not tune its magnitude to humans.

---

## 15. Critical switch prediction

Define for each context switch:

\[
SwitchCost
=
RT_{first\ sequence\ after\ switch}
-
RT_{late\ preceding\ context}.
\]

Primary prediction:

\[
SwitchCost>0.
\]

Also test whether the cost declines across the first few sequences in the new context.

This is one of the strongest psychologically interpretable diagnostics of the new architecture.

The model does not receive a switch-cost target.

---

## 16. Comparison to deterministic confidence threshold

Add one simple nonlearned control:

Commit at the first preparation cycle where:

\[
p_{correct,m}\ge\theta.
\]

Evaluate:

```text
theta ∈ {.60, .70, .80, .90}
```

without retraining the SRN.

This is not the preferred model.

It answers:

> Is learned commitment doing anything beyond applying a fixed confidence criterion?

Compare:

```text
RT learning curve
switch costs
RT dynamic range
state-space classification
```

Do not choose theta using human data.

---

## 17. Optional explicit WAIT/GO policy control

Only if the differentiable hazard model succeeds, implement a closer WAIT/GO robustness model:

Action space:

```text
WAIT
EXECUTE
```

When EXECUTE is chosen, the 4-way sequence head supplies the key response.

Reward:

```text
+R_correct
-C_error
-C_wait per WAIT cycle
```

Train using actor-critic or REINFORCE with a baseline.

This is a robustness analysis, not MVP.

Do not start here because it adds unnecessary optimization variance.

---

## 18. Phase 2 — confirmatory architecture comparison

Once one commitment configuration passes screening, freeze it.

Run:

```text
20 seeds:
1. High scale = 2 + learned commitment
2. Low scale + learned commitment
3. High scale = 2, sequence only
```

The sequence-only control has no RT output.

For each seed save the final frozen checkpoint.

---

## 19. Confirmatory predictions

### H1 — sensible acquisition

High-scale + commitment:

\[
CE\downarrow
\]

and:

\[
RT\downarrow.
\]

### H2 — context switch slowing

\[
RT_{switch}>RT_{within}.
\]

### H3 — representational flexibility

High-scale networks should show substantially stronger A/B hidden-state separation than low-scale networks.

### H4 — functional state-space bridge

Fit the existing canonical:

```text
Lump
Cross
Split
```

state-space models to emergent RT.

Ideal result:

```text
High scale -> Cross/Split
Low scale -> Lump
```

Do not force this result.

### H5 — sequence-only separation control

The High-scale sequence-only network may still show context-specific hidden/CE structure.

This tests the division of labor:

> high scale creates context-specific organization; commitment makes that organization behaviorally consequential in RT.

---

## 20. Frozen Mixed and Novel probes

After confirmatory training, evaluate without learning.

Use:

```text
A
B
Mixed_norm
Novel
```

and optional:

```text
Mixed_additive
```

Measure:

```text
expected RT
hard RT
sequence CE
p_correct
hidden trajectory
commitment trajectory
```

Primary questions:

1. Does Mixed produce a structured intermediate preparation state?
2. Does Novel produce graded expression despite containing neither trained context?
3. Do Mixed and Novel require different amounts of preparation?
4. Do their hidden states lie in structured relation to A/B states?

Do not require exact 50/50 averaging.

---

## 21. Causal representational analyses

If the final model passes the acquisition gates, rerun the existing causal analyses.

At matched sequence position \(k\):

```text
Mixed <- A component
Mixed <- B component
Novel <- A component
Novel <- B component
```

Primary causal outcome now becomes:

\[
\Delta RT
\]

or equivalently:

\[
\Delta E[m].
\]

Also measure:

```text
change in sequence CE
change in p_correct
change in hard commitment cycle
```

This gives a stronger mechanistic chain:

\[
\boxed{
\text{hidden-state intervention}
\rightarrow
\text{action readiness}
\rightarrow
\text{commitment time}
}
\]

---

## 22. Sequence-preservation requirement

A hidden-state intervention should not count as execution-specific evidence if it simply destroys next-token prediction.

For all causal analyses report jointly:

```text
RT change
next-token accuracy change
CE change
```

The strongest evidence is:

```text
substantial RT shift
with preserved next-token identity/accuracy.
```

---

## 23. Required figures

1. **Architecture:** token + context → recurrent preparation cycles → WAIT/COMMIT; RT = preparation cycles.
2. **Example transition:** confidence, commit hazard, and commit probability across cycles for an easy transition and a switch transition.
3. **Learning curves:** sequence CE, expected RT, hard RT across 170 sequences.
4. **Switch effects:** RT aligned to A/B context switches.
5. **State-space overlay:** Lump/Cross/Split over emergent RT for High and Low scale.
6. **Probe RT:** A, B, Mixed, Novel expected commitment time.
7. **Causal effects:** token-matched hidden-state patching effects on expected RT with sequence preservation.

---

## 24. Required outputs

```text
commitment_screening_summary.csv
commitment_cycle_dynamics.csv
commitment_training_trajectories.csv
commitment_switch_costs.csv

final_commitment_seed_summary.csv
final_state_space_fits.csv

final_probe_commitment.csv
final_probe_hidden_states.csv

final_commitment_patching.csv
final_commitment_controls.csv
```

---

## 25. Required report

Create:

```text
COMMITMENT_RT_REPORT.md
```

Use this structure:

### 1. Did recurrent preparation improve sequence readiness?

Report within-transition confidence dynamics.

### 2. Did RT emerge in the correct learning direction?

Report early→late CE and RT.

### 3. Did context switching slow commitment?

Report switch costs.

### 4. Did High-scale and Low-scale networks differ?

Report hidden separation and state-space classification.

### 5. What happened on Mixed and Novel probes?

Report commitment times and hidden-state positions.

### 6. Did causal hidden-state interventions alter commitment time?

Report RT effect alongside sequence preservation.

### 7. Final interpretation

Choose one:

```text
A. recurrent preparation provides a viable emergent RT mechanism
B. timing works but context-specific state-space structure fails
C. context structure works but acquisition direction fails
D. recurrent preparation itself does not improve action readiness
```

Do not select the interpretation based on the desired manuscript story.

---

## 26. Stop conditions

Stop before expensive downstream analyses if:

```text
late RT >= early RT
```

for most High-scale seeds;

or if:

```text
commitment collapses to cycle 1 / M_max;
```

or if:

```text
confidence does not improve across preparation cycles.
```

If preparation itself does not improve action readiness, the architecture needs reconsideration before more parameter search.

---

## 27. Scientific success criterion

The strongest result would be:

\[
\boxed{
\begin{aligned}
&\text{practice}
\rightarrow
\text{better recurrent sequence state}
\rightarrow
\text{earlier commitment}
\rightarrow
RT\downarrow\\[4pt]
&\text{context switch}
\rightarrow
\text{state reconfiguration}
\rightarrow
\text{later commitment}
\rightarrow
RT\uparrow\\[4pt]
&\text{ambiguous context}
\rightarrow
\text{configured intermediate state}
\rightarrow
\text{graded commitment time}
\end{aligned}
}
\]

with no RT supervision.

That would replace the psychologically opaque claim:

> “the network learned an arbitrary execution-speed scalar”

with the stronger mechanistic claim:

> **reaction time reflects how long the recurrent system needs to configure an action-ready sequence state.**
