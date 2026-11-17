# PRD: Stepwise Structured-Context Ecology for Split-but-Recombinable SRNs

## Goal

Test whether a more human-plausible **context environment** can preserve the previously established A/B splitting result while teaching the SRN that contextual cues are structured, partially shared evidence rather than arbitrary categorical labels.

The central hypothesis is:

\[
\boxed{
\text{shared sequence}
+
\text{partially shared context structure}
\rightarrow
\text{split but potentially recombinable sequence states}
}
\]

The work must proceed **stepwise**.

At every stage:

1. first establish that the model still learns the sequence;
2. then establish that High-scale networks still split A and B;
3. then establish that Low-scale networks remain comparatively lumped;
4. only after those gates pass should we examine Mixed;
5. only after Mixed is characterized should we add contextual uncertainty/salience variation;
6. Novel is last.

Do not tune any stage using human Mixed/Novel RTs.

---

## 1. Existing frozen model to preserve

Use the current successful confirmatory SRN as the reference implementation.

Freeze:

```text
8-unit tanh Elman SRN
High scale = 2.0
Low scale = 0.02
170 external training sequences
17 alternating A10/B10 blocks
same registered fixed sequence used by the repository
same optimizer / learning rate
same next-token objective
same two-parameter monotonic execution readout
RT = 1 - execution_speed
same canonical seed convention
```

Primary monotonic execution readout:

\[
v_t =
\sigma
\left[
\operatorname{softplus}(\alpha)\operatorname{logit}(p_t)+b
\right]
\]

where \(p_t\) is detached correct-next-token confidence.

Do not reopen the RT objective.

---

## 2. Scientific distinction

The current model uses categorical context labels:

```text
A = [1,0,0]
B = [0,1,0]
```

This makes it easy for a High-scale SRN to learn two highly separated trajectories.

The new hypothesis is that the human environment contains important **shared structure**:

- the exact same sequence is visible/performed in both contexts;
- at least one contextual item can be shared between contexts;
- contextual items can covary with specific ordinal parts of the displayed sequence.

The new environment should therefore represent context as **items/features surrounding a shared sequence**, not as an abstract A/B label.

---

## 3. Design principle

For every external sequence position \(k\), input is:

\[
x_k =
[
\underbrace{s_k}_{\text{shared sequence token}},
\underbrace{c_k}_{\text{context item/features}}
].
\]

The sequence component \(s_k\):

- is exactly identical in A and B;
- remains the existing 4D token one-hot;
- is never degraded;
- is never changed by Mixed/Novel manipulations.

Only \(c_k\) changes.

This is critical.

The network must always know that it is performing the **same sequence/task**.

---

## 4. Stage 0 — exact categorical baseline

Before changing context structure, reproduce the current frozen result from the existing checkpoints or rerun if required by code changes.

Baseline context:

```text
A = [1,0,0]
B = [0,1,0]
```

Required baseline observations:

```text
High scale:
    strong A/B hidden separation
    majority Cross/Split state-space classification
    sensible RT learning

Low scale:
    weak A/B hidden separation
    majority Lump classification
```

Use the existing 20-seed confirmatory result as the reference baseline.

Export:

```text
stage0_baseline_summary.csv
```

Do not modify Stage 0 to improve later stages.

---

## 5. Stage 1 — static shared contextual item

The first manipulation should add **one literal overlapping contextual item** while otherwise keeping context simple.

Use three contextual feature channels:

```text
[G, A_specific, B_specific]
```

where:

```text
G = shared item present in both contexts
```

Primary encoding:

\[
A =
\frac{[1,1,0]}{\sqrt{2}}
\]

\[
B =
\frac{[1,0,1]}{\sqrt{2}}.
\]

This ensures:

- A and B have equal context-vector norms;
- one feature is literally identical across contexts;
- one feature distinguishes A;
- one feature distinguishes B.

The sequence token remains appended separately and identical across contexts.

No noise.
No dropout.
No Mixed.
No Novel.

---

## 6. Stage 1 question

Ask only:

> Does a literal shared contextual item preserve the previously established High-scale splitting phenotype?

Run a **5-seed screen** first.

Conditions:

```text
High scale + monotonic RT
Low scale + monotonic RT
High scale + sequence only
```

If Stage 1 passes, rerun 20 seeds.

---

## 7. Stage 1 gates

### G1 — sequence learning

For every condition:

```text
late CE < early CE
```

in at least 4/5 screening seeds.

### G2 — representational scale effect

Require:

```text
High A/B hidden separation > Low A/B hidden separation
```

in at least 4/5 paired seeds.

### G3 — nontrivial High-scale splitting

High-scale A/B separation must remain clearly above the Low-scale regime.

Do not require the exact categorical-baseline magnitude.

Primary comparison is effect size / paired difference relative to Stage 0.

### G4 — state-space behavior

At least 4/5 Low-scale screening seeds should remain Lump or near-Lump.

High-scale should show materially more Cross/Split classifications than Low-scale.

Do not tune the encoding to force a specific winner.

### G5 — RT sanity

For execution conditions:

```text
late RT < early RT
```

and confidence→RT slope must remain negative.

If Stage 1 fails, stop and report which property was lost before adding positional structure.

---

## 8. Stage 2 — sequence-position-covarying context items

Only if Stage 1 passes.

Now make context a structured set of items that covary with **ordinal sequence position**.

Important:

> Do not tie contextual items merely to physical key identity.

Repeated physical transitions occurring at different ordinal positions must remain distinguishable.

Use ordinal transition position.

---

## 9. Stage 2 context bank

Primary context bank:

```text
S   = one shared contextual item

A1
A2
A3

B1
B2
B3
```

Total context feature dimension:

```text
7
```

The 4D sequence token remains separate.

Use a preregistered position-group mapping:

```text
Group 1: transitions 1-2
Group 2: transitions 3-4
Group 3: transitions 5-7
```

This mapping is arbitrary but fixed before seeing results.

Do not optimize grouping based on RT or hidden-state outcomes.

---

## 10. Stage 2 primary encoding

At each transition, activate:

1. the shared item `S`;
2. the context-specific item associated with that ordinal transition group.

Example:

```text
Context A:
T1 -> [S, A1]
T2 -> [S, A1]
T3 -> [S, A2]
T4 -> [S, A2]
T5 -> [S, A3]
T6 -> [S, A3]
T7 -> [S, A3]

Context B:
T1 -> [S, B1]
T2 -> [S, B1]
T3 -> [S, B2]
T4 -> [S, B2]
T5 -> [S, B3]
T6 -> [S, B3]
T7 -> [S, B3]
```

Represent each active pair as equal-amplitude features and L2-normalize the total context vector to unit norm.

Thus context strength is approximately norm-matched to the original categorical context.

No salience noise yet.

---

## 11. Stage 2 alternative shared-item variant

If desired as a preregistered robustness test, implement a second structured encoding where the shared item is active only at specific positions.

Example:

```text
T1-T2 -> shared S in both A and B
T3-T4 -> A2 vs B2
T5-T7 -> A3 vs B3
```

Label:

```text
position_shared_item_control
```

This is secondary.

Do not use it to rescue a failed primary Stage 2.

---

## 12. Stage 2 questions

Before probing Mixed, test:

1. Does High scale still split A/B?
2. Does Low scale remain mostly lumped?
3. Does the network still learn the same sequence?
4. Does RT still decrease with learning?
5. Does A/B context structure appear at the intended ordinal positions?

Only after these pass should Mixed be introduced.

---

## 13. Stage 2 representational analyses

At each transition position \(k\), calculate:

\[
d_k =
\left\|
\bar h_{A,k}
-
\bar h_{B,k}
\right\|.
\]

Also quantify:

### Shared sequence structure

Within the same ordinal position:

\[
sim(h_{A,k},h_{B,k})
\]

relative to cross-position similarities.

### Context displacement

Define:

\[
\delta_k =
h_{A,k}-h_{B,k}.
\]

Ask whether the context-dependent displacement is more consistent across positions than the full hidden state.

### Factorization diagnostic

Test whether a simple decomposition:

\[
h_{c,k}
\approx
s_k+d_{c,k}
\]

reconstructs the hidden states better under structured context than under the categorical baseline.

This is descriptive.

Do not require perfect linear additivity.

---

## 14. Stage 2 splitting gate

A structured-context model passes only if:

```text
High-scale A/B separation remains robust
AND
High > Low separation
AND
sequence learning remains intact
AND
RT learning remains sensible
```

If structured overlap destroys splitting entirely, stop.

Do not introduce Mixed/noise to rescue it.

---

## 15. Stage 3 — frozen Mixed probe under structured context

Only after Stage 2 passes.

Mixed must be constructed from **familiar contextual items**, not a new midpoint vector in an abstract one-hot space.

Primary Mixed encoding:

At each transition group, activate:

```text
S + corresponding A item + corresponding B item
```

Example:

```text
T1-T2 -> [S, A1, B1]
T3-T4 -> [S, A2, B2]
T5-T7 -> [S, A3, B3]
```

Normalize total context-vector norm to match the training contexts.

No training occurs on Mixed.

---

## 16. Stage 3 central test

Ask whether a recombination of familiar contextual items now produces:

\[
RT_M(k)
\]

that lies in a structured relationship to:

\[
RT_A(k), RT_B(k)
\]

more often than under the original categorical-input model.

Do not require exact midpoint behavior.

Primary metrics:

```text
fraction of usable transitions where Mixed lies between A and B
midpoint deviation
global scalar q
transition-specific q_k
distance of Mixed hidden state to A/B segment
distance of Mixed hidden state to nearest trained manifold
```

Compare directly with the Stage 0 categorical baseline.

---

## 17. Hidden-state interpolation diagnostic

For each transition \(k\), define the line segment:

\[
L_k(\lambda)
=
\lambda h_{A,k}
+
(1-\lambda)h_{B,k}
\]

for:

\[
0\le\lambda\le1.
\]

For Mixed hidden state \(h_{M,k}\), compute:

```text
closest lambda
distance to A/B segment
whether projection lies inside segment
```

This is a geometric diagnostic, not a claim that the true computation must be linear.

Question:

> Does structured context make Mixed land closer to the learned A/B representational corridor?

---

## 18. Stage 3 success pattern

The result of interest is:

```text
Stage 0 categorical:
    strong split
    poor / extrapolative Mixed

Stage 2/3 structured overlap:
    strong split retained
    Mixed more often between A/B
    Mixed hidden states closer to A/B corridor
```

This would support:

> shared structured context teaches a representation that remains separable at the learned endpoints but is more compositional under cue recombination.

---

## 19. Stage 4 — contextual salience variability

Only after the deterministic structured-context model has been fully characterized.

Do not add generic noise to sequence inputs.

Apply variability only to contextual items.

Primary manipulation:

For each active context feature weight \(w=1\), sample a multiplicative salience:

\[
\tilde w = w \cdot g
\]

with:

\[
g\sim \text{LogNormal}(\mu,\sigma)
\]

parameterized so:

\[
E[g]=1.
\]

Alternatively, use bounded multiplicative noise if already easier in the repo.

---

## 20. Stage 4 salience levels

Screen:

```text
sigma_context ∈ {0, 0.05, 0.15, 0.30}
```

Keep the sequence token perfect.

No context identity is ever intentionally flipped.

This models variable internal salience/encoding, not physical mislabeling.

---

## 21. Stage 4 hypotheses

Test for a potential regime:

\[
\boxed{
\text{enough context signal to split}
+
\text{enough variability to learn graded contextual evidence}
}
\]

Predictions:

- zero variability may retain strong splitting but weak interpolation;
- small/moderate variability may preserve splitting and improve Mixed interpolation;
- excessive variability may collapse A/B splitting.

Do not assume the dose response must be monotonic.

---

## 22. Stage 4 gates

For each salience level report:

```text
sequence learning
High/Low hidden separation
state-space classification
RT acquisition
Mixed between-endpoint fraction
Mixed hidden distance to A/B segment
```

Do not select the “best” sigma using human RT.

Select only on the mechanistic criterion:

```text
splitting preserved + recombination improved
```

relative to zero-noise structured context.

---

## 23. Stage 5 — Novel context

Novel comes last.

Do not use an entirely untrained single third one-hot axis as the primary Novel probe.

Novel should preserve:

```text
the exact same sequence input
the same shared contextual item S
the same position-to-context-item organization
```

but use new local contextual items:

```text
N1
N2
N3
```

Example:

```text
T1-T2 -> [S, N1]
T3-T4 -> [S, N2]
T5-T7 -> [S, N3]
```

These items are novel, but the task scaffold and shared contextual structure are familiar.

This models:

> familiar task + unfamiliar contextual wrapper.

---

## 24. Novel questions

Ask:

1. Does Novel remain closer to the familiar sequence manifold than the old unused-axis probe?
2. Is Novel slower than pure A/B without becoming arbitrarily off-manifold?
3. Does Novel express a structured relationship to A/B transition states?
4. Does prior training with contextual salience variability change Novel behavior?

Do not require Novel to equal Mixed.

Mixed = conflicting familiar contextual evidence.

Novel = weak/absent match to learned context-specific items within a familiar task scaffold.

---

## 25. Experimental progression

Implement exactly in this order:

```text
Stage 0:
categorical baseline

Stage 1:
one shared static contextual item
NO Mixed

Stage 2:
shared + ordinal-position-covarying context items
NO noise
NO Mixed until splitting gate passes

Stage 3:
frozen Mixed from familiar A/B item recombination

Stage 4:
context-item salience variability
then re-evaluate splitting + Mixed

Stage 5:
structured Novel context
```

Do not skip ahead when a prior stage fails.

---

## 26. Screening and confirmatory seed policy

For every new environmental stage:

```text
5-seed screening
```

If required gates pass:

```text
20-seed confirmatory run
```

Freeze the same seed identities across stages whenever possible so comparisons are paired.

---

## 27. No human tuning

Human data may not be used to choose:

```text
number of context items
position grouping
feature amplitudes
salience noise
Mixed normalization
Novel encoding
```

These must be fixed from environmental logic.

Human results are used only later for external comparison.

---

## 28. Key controls

### Control A — categorical baseline

Current model.

### Control B — shared item only

Tests whether overlap by itself matters.

### Control C — position structure without noise

Tests whether context-to-sequence covariance matters.

### Control D — structured context + salience variability

Tests whether experience with graded contextual evidence matters.

This factorial progression is more informative than comparing one final hand-designed environment against baseline.

---

## 29. Required TDD tests

Use RED → GREEN → REFACTOR.

### T1 — sequence input invariance

For matched transition \(k\):

```text
sequence_token(A,k) == sequence_token(B,k)
```

always.

### T2 — shared context item

Stage 1+:

```text
shared feature identity is exactly identical across A and B
```

### T3 — ordinal mapping

Repeated physical keys at different ordinal positions must not collapse to the same contextual item solely because of key identity.

### T4 — norm matching

Context-vector norms must match the intended stage specification.

### T5 — A/B curriculum

All stages preserve A10/B10 × 17 blocks.

### T6 — no probe training

Mixed/Novel never cause optimizer steps.

### T7 — noise localization

Stage 4 variability applies only to contextual features, never the sequence token.

### T8 — no label flips

Noise cannot replace A-specific item identity with B-specific identity.

### T9 — paired seeds

Same seed initializes matching conditions identically except for the experimental context encoding.

### T10 — stage firewall

The pipeline must not run Stage N+1 automatically when Stage N fails its preregistered gates.

---

## 30. Required outputs

```text
structured_context_config.json

stage0_baseline_summary.csv

stage1_shared_item_training.csv
stage1_shared_item_representation.csv
stage1_shared_item_state_space.csv

stage2_structured_training.csv
stage2_structured_representation.csv
stage2_factorization_diagnostics.csv
stage2_state_space.csv

stage3_mixed_probe.csv
stage3_mixed_geometry.csv
stage3_responsibility.csv

stage4_salience_sweep.csv
stage4_salience_representation.csv
stage4_salience_mixed.csv

stage5_novel_probe.csv
stage5_novel_geometry.csv

structured_context_stage_summary.csv
```

---

## 31. Required figures

### Figure 1 — context ecology schematic

Show:

```text
same sequence token stream
+
A structured context items
vs
B structured context items
```

Highlight literal shared item(s).

### Figure 2 — stepwise splitting

High vs Low A/B hidden separation for:

```text
categorical
shared-item
position-structured
structured + salience variability
```

### Figure 3 — PNAS-style RT/state-space overlay

For each stage that passes:

```text
observed RT
Lump
Cross
Split
```

### Figure 4 — Mixed evolution across stages

For:

```text
categorical baseline
structured overlap
structured + salience variability
```

show A/B endpoints and Mixed.

### Figure 5 — representational corridor

For each transition:

```text
A
B
Mixed
```

plus distance from Mixed to the A/B segment.

### Figure 6 — splitting versus recombination tradeoff

x-axis:

```text
A/B hidden separation
```

y-axis:

```text
Mixed interpolation score
```

Each point = environmental stage / salience level.

This is potentially the most important final figure.

---

## 32. Primary derived metrics

### Splitting score

\[
S =
\frac{1}{K}
\sum_k
\|h_{A,k}-h_{B,k}\|.
\]

### Mixed interpolation score

Primary simple metric:

```text
fraction of usable transitions with Mixed RT between A and B
```

### Representational interpolation score

For each \(k\):

```text
distance of Mixed hidden state to A/B line segment
```

aggregate across positions/seeds.

### Joint success

Do not collapse splitting and recombination into one scalar for model selection.

The desired regime is:

```text
high splitting
+
low Mixed-to-segment distance
+
high between-endpoint fraction
```

---

## 33. Interpretation matrix

### Result A — Stage 1 preserves splitting, but Mixed still extrapolates

Shared context item alone is insufficient.

Proceed to structured covariance.

### Result B — Stage 2 preserves splitting and improves Mixed

Strong evidence that context items tied to sequence structure provide a compositional retrieval ecology.

Stage 4 becomes a robustness/strengthening test.

### Result C — Stage 2 preserves splitting but Mixed remains off-manifold

Proceed to salience variability.

Interpretation:

> structural overlap alone does not teach graded evidence integration.

### Result D — moderate salience variability preserves splitting and improves Mixed

Strong evidence for the environmental-learning hypothesis:

> experience with imperfect contextual evidence teaches a retrieval computation that supports recombination.

### Result E — variability destroys splitting before it improves Mixed

The simple uncertainty ecology is not sufficient.

Do not continue tuning indefinitely.

### Result F — Mixed improves but Novel remains poor

This is acceptable and theoretically informative.

Conflicting familiar evidence and unfamiliar context may require different computations.

---

## 34. Stop conditions

Stop a branch if:

```text
sequence learning fails
```

or:

```text
High-scale A/B separation collapses to Low-scale regime
```

or:

```text
RT acquisition becomes psychologically inverted again
```

Do not compensate by changing:

```text
High scale
hidden size
RT objective
training length
human-matched curriculum
```

Those are frozen.

---

## 35. Final report

Create:

```text
STRUCTURED_CONTEXT_ECOLOGY_REPORT.md
```

Use this structure:

### 1. Baseline replication

Did the categorical model reproduce the frozen result?

### 2. Shared-item context

Did literal overlap preserve splitting?

### 3. Position-structured context

Did context-item/sequence-position covariance preserve splitting?

### 4. Mixed probe

Did recombining familiar contextual items improve interpolation?

### 5. Salience variability

Did graded contextual evidence improve recombination without destroying splitting?

### 6. Novel probe

What happened when the sequence/task scaffold was familiar but contextual items were new?

### 7. Final interpretation

Choose one:

```text
A. shared structured context is sufficient
B. structured context + salience variability is required
C. splitting and recombination remain in tension
D. structured environmental overlap does not solve the problem
```

Do not select based on the desired manuscript story.

---

## 36. Scientific target

The strongest result would be:

\[
\boxed{
\text{shared sequence}
+
\text{partially shared structured context}
+
\text{graded contextual evidence}
\rightarrow
\text{split yet recombinable representations}
}
\]

The key theoretical point would be:

> A learner can maintain distinct context-specific versions of the same sequence while still learning that contextual cues provide graded evidence about how a shared sequence representation should be configured.

This PRD tests the **environmental route** to that computation.

A separate future analysis can test the alternative route in which interpolation is made available directly at the representational/readout level. The two routes should remain conceptually and experimentally distinct.
