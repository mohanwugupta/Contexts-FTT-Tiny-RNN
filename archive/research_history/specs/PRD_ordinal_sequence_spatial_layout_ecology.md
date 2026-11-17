# PRD: Ordinal Sequence Scaffold + Context-Specific Spatial Layout SRN

## Goal

Build a human-faithful SRN environment in which:

- the **entire sequence is visible**;
- each ordinal sequence position is uniquely identifiable;
- the **order is identical across contexts**;
- the **screen locations / spatial layout differ across contexts**;
- broader context identity remains strong enough to preserve the original High-scale context-specific RT phenotype.

Core hypothesis:

\[
\boxed{
\text{same visible ordinal sequence}
+
\text{different context-specific spatial realizations}
\rightarrow
\text{split but potentially recombinable sequence representations}
}
\]

The work must proceed stepwise. The first priority is to recover the original splitting and context-yoked RT dynamics. Mixed probing is blocked until that phenotype is recovered.

---

## 1. Frozen reference model

Use the existing successful confirmatory SRN as the reference implementation.

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
\right].
\]

Do not reopen the RT objective.

---

## 2. Environmental abstraction

Each sequence step should contain three conceptually distinct pieces:

\[
x_{c,k}
=
[
\underbrace{P_k}_{\text{shared ordinal position}},
\underbrace{\mathcal L_c}_{\text{context-specific full layout}},
\underbrace{C_c}_{\text{global context identity}}
].
\]

Where:

- \(k\) is ordinal sequence position;
- \(P_k\) uniquely identifies the current ordinal position;
- \(\mathcal L_c\) encodes the entire visible spatial layout in context \(c\);
- \(C_c\) is the strong A/B contextual cue.

The ordinal sequence is invariant:

\[
P_1\rightarrow P_2\rightarrow\cdots\rightarrow P_8
\]

in both contexts.

The spatial realization differs:

\[
L_{A,k}\neq L_{B,k}.
\]

The same physical/action sequence is used in both contexts.

---

## 3. Ordinal-position representation

Use eight ordinal one-hot channels:

```text
P1 ... P8
```

At sequence position \(k\):

```text
Pk = 1
all other ordinal channels = 0
```

These channels are exactly identical across A and B.

They mean:

> this is the same ordinal item of the same visible sequence.

Repeated physical keys at different ordinal positions must remain distinguishable.

---

## 4. Physical/action token

Preserve the existing action/token input if needed by the current SRN.

The model may therefore receive both:

```text
physical token identity
ordinal position identity
```

These are different variables.

Example:

```text
ordinal position 1 -> token 1
ordinal position 4 -> token 1
```

This is expected.

---

## 5. Full-layout representation

Because the full sequence is visible, the model should receive the full spatial layout on every sequence step.

Primary representation:

For each ordinal position \(k\):

\[
L_{c,k}=[x_{c,k},y_{c,k}].
\]

Concatenate:

\[
\mathcal L_c=
[x_{c,1},y_{c,1},\ldots,x_{c,8},y_{c,8}].
\]

Thus:

```text
layout_dim = 16
```

The same layout vector is present throughout the current A or B block.

The active ordinal position is separately indicated by \(P_k\).

---

## 6. Spatial-layout requirements

A and B must have:

```text
same ordinal sequence
same number of items
same physical response sequence
different screen coordinates
same layout dimensionality
```

Use two fixed layouts selected before training.

The layouts should abstract the human task:

> the same ordinal item can appear at a different height / horizontal location in A versus B, while the sequence order remains unchanged.

Do not choose coordinates based on model outcomes.

---

## 7. Coordinate normalization

Normalize A and B coordinates so layout strength is matched.

Recommended:

1. center each layout around zero;
2. scale both to the same RMS coordinate magnitude.

Save:

```text
layout_A_coordinates
layout_B_coordinates
layout_A_norm
layout_B_norm
```

to config.

---

## 8. Strong global context identity

Keep:

```text
C_A = [1,0]
C_B = [0,1]
```

through the primary branch.

This cue is intentionally strong and stable.

Its purpose is to preserve:

```text
A/B splitting
context-yoked learning
switch-related RT structure
```

Do not add salience noise, dropout, or cue weakening in the primary branch.

---

## 9. Primary input

Primary Stage-2 input:

\[
x_{c,k}
=
[
\text{token}_k,
P_k,
\mathcal L_c,
C_c
].
\]

Approximate dimensions:

```text
token: 4
ordinal position: 8
layout: 16
global context: 2
total: 30
```

If the repository needs another interface, preserve these semantics.

---

# Stepwise experiment

## 10. Stage 0 — categorical reference

Use the frozen categorical confirmatory result as baseline.

Required reference phenotype:

### High scale

```text
strong A/B hidden separation
majority Cross/Split
clear context-yoked RT sawtooth
RT decreases with learning
```

### Low scale

```text
weak A/B hidden separation
majority Lump
```

Do not alter this baseline.

---

## 11. Stage 1 — shared ordinal scaffold only

Before adding spatial layout, test:

\[
x_{c,k}
=
[
\text{token}_k,
P_k,
C_c
].
\]

Question:

> Does explicit shared ordinal-position identity preserve the original High-scale splitting and context-yoked RT phenotype?

Run a 5-seed screen first.

Conditions:

```text
High monotonic
Low monotonic
High sequence-only
```

---

## 12. Stage 1 gates

Require:

### G1 — sequence learning

```text
late CE < early CE
```

for at least 4/5 seeds.

### G2 — High > Low splitting

```text
High A/B hidden separation > Low A/B hidden separation
```

for at least 4/5 paired seeds.

### G3 — High context-specific RT

At least:

```text
4/5 High seeds classified Cross or Split
```

using the canonical state-space models.

### G4 — Low lumping

At least:

```text
4/5 Low seeds classified Lump
```

### G5 — RT sanity

```text
late RT < early RT
confidence-RT slope < 0
```

If Stage 1 fails, stop.

Do not add the full layout until the original phenotype survives explicit ordinal coding.

---

## 13. Stage 2 — add full context-specific layout

Only if Stage 1 passes.

Use:

\[
x_{c,k}
=
[
\text{token}_k,
P_k,
\mathcal L_c,
C_c
].
\]

No noise.
No Mixed.
No Novel.

Question:

> Can the model preserve strong A/B splitting and context-yoked RT while representing the same ordinal sequence under different spatial layouts?

Run a 5-seed screen, then 20-seed confirmation if passed.

---

## 14. Stage 2 hard gates

Require:

```text
sequence learning intact
High > Low A/B hidden separation
High Cross/Split >= 4/5 screening seeds
Low Lump >= 4/5 screening seeds
RT decreases with learning
visible context-yoked sawtooth retained
```

For the 20-seed confirmation, report exact seed fractions.

A branch is rejected if the mean and seed-level RT dynamics are effectively Lump, even if Mixed later looks better.

---

## 15. State-space analysis

Fit the exact existing canonical models:

```text
Lump
Cross
Split
```

Report:

```text
seed-level winner
seed-averaged fit
BIC
RMSE
R²
switch cost
A/B endpoint difference
```

Seed-level classifications are primary.

Also plot the raw A/B-block RT trajectory so loss of context-yoked structure is visually obvious.

---

## 16. Representational analyses

At each ordinal position \(k\):

\[
d_k=
\|h_{A,k}-h_{B,k}\|.
\]

Also calculate:

### Matched-position similarity

\[
\cos(h_{A,k},h_{B,k}).
\]

### Same-position vs different-position similarity

Compare:

```text
same ordinal position across A/B
vs
different ordinal positions across A/B
```

The desired result is:

> corresponding ordinal states remain structurally related across contexts while still being context-specific.

---

## 17. Factorization diagnostic

Test descriptively whether:

\[
h_{c,k}\approx s_k+d_{c,k}
\]

is useful.

Where:

- \(s_k\) is shared ordinal sequence structure;
- \(d_{c,k}\) is context-specific spatial/contextual modulation.

Compare with the categorical baseline.

Do not require exact linear additivity.

---

# Mixed branch

## 18. Mixed firewall

Do not run Mixed unless Stage 2 passes the High-scale Cross/Split behavioral gate.

This is a hard stop condition.

---

## 19. Mixed keeps ordinal identity fixed

Mixed always uses the same:

```text
P1 ... P8
```

and the same physical sequence.

Ordinal identity is never ambiguous.

Only contextual/spatial realization is ambiguous or combined.

---

## 20. Mixed layout representation

The primary Mixed probe should preserve both familiar A and B spatial evidence.

Preferred fixed-dimensional implementation:

```text
layout_A_channels
layout_B_channels
```

Training:

```text
A -> [Layout_A, zeros]
B -> [zeros, Layout_B]
```

Mixed:

```text
[Layout_A, Layout_B]
```

This means:

> both familiar spatial realizations are present.

If this requires an input expansion, the expanded layout-channel architecture must be used from the start of A/B training so Mixed does not activate previously nonexistent dimensions.

Do not use a post hoc adapter trained on Mixed.

---

## 21. Mixed global context cue

Primary:

```text
C_M = [1,1]
```

This represents:

> A evidence is present and B evidence is present.

Do not make `[0.5,0.5]` primary.

Include it only as a norm-matched control.

---

## 22. Mixed outcomes

Measure:

```text
fraction of transitions where Mixed RT lies between A and B
midpoint deviation
global q
transition-specific q_k
next-token accuracy
distance of Mixed hidden state to A/B segment
nearest-manifold distance
```

Do not fit or tune to human Mixed RT.

---

## 23. Representational corridor

For each transition:

\[
L_k(\lambda)=
\lambda h_{A,k}+(1-\lambda)h_{B,k},
\qquad 0\le\lambda\le1.
\]

For \(h_{M,k}\), calculate:

```text
closest lambda
distance to A/B segment
whether projection lies inside the segment
```

Question:

> Does Mixed occupy the corridor between two trained spatial realizations of the same ordinal sequence state?

---

# Follow-up only if necessary

## 24. Sparse shared-item sweep

If Stage 2 preserves Cross/Split but Mixed still extrapolates, add literal visual overlap conservatively.

Possible preregistered conditions:

```text
0 extra shared contextual objects
1 shared object
2 shared objects
larger shared subset
```

Keep:

```text
ordinal scaffold fixed
global A/B cue fixed
layout distinction fixed
no salience noise
```

Reject any condition in which High-scale RT becomes predominantly Lump.

---

## 25. Do not reintroduce salience noise yet

The previous salience branch improved interpolation while weakening context-yoked RT.

Therefore:

```text
no context salience sweep
no global context noise
no dropout
```

in this PRD.

Those can only be revisited after a model simultaneously shows:

```text
robust splitting
Cross/Split RT
sensible Mixed recombination
```

---

# Novel branch

## 26. Novel comes last

Only after Stage 2 passes and Mixed is characterized.

Novel preserves:

```text
same ordinal scaffold
same physical sequence
same whole-layout concept
```

but uses a new layout:

\[
\mathcal L_N.
\]

Primary global cue:

```text
[0,0]
```

unless a separately motivated novel cue representation has been trained in advance.

Do not activate arbitrary untrained dimensions and interpret them as meaningful novelty.

---

# TDD

## 27. Required tests

Use RED → GREEN → REFACTOR.

### T1 — ordinal invariance

```text
Pk_A == Pk_B
```

for every ordinal position.

### T2 — order invariance

Both contexts follow:

```text
P1 -> P2 -> ... -> P8
```

### T3 — spatial difference

Several positions satisfy:

```text
L_A(k) != L_B(k)
```

### T4 — full layout visible

At every step:

```text
full layout vector is available
```

not only current coordinates.

### T5 — repeated-token disambiguation

Repeated physical tokens at different positions have different ordinal channels.

### T6 — stable A/B identity

Primary A/B global cues remain exactly:

```text
[1,0]
[0,1]
```

### T7 — no salience noise

Primary branch uses no context/layout noise.

### T8 — no probe training

Mixed/Novel cause:

```text
0 optimizer steps
0 parameter updates
```

### T9 — stage firewall

Mixed cannot execute if Stage 2 fails the Cross/Split gate.

### T10 — paired seeds

Matched conditions use the same initialization seeds wherever possible.

---

# Outputs

## 28. Required files

```text
ordinal_layout_config.json

stage0_reference_summary.csv

stage1_ordinal_training.csv
stage1_ordinal_representation.csv
stage1_ordinal_state_space.csv

stage2_layout_training.csv
stage2_layout_representation.csv
stage2_layout_factorization.csv
stage2_layout_state_space.csv
stage2_layout_switch_effects.csv

stage3_mixed_probe.csv
stage3_mixed_geometry.csv
stage3_mixed_responsibility.csv

layout_overlap_sweep.csv      # only if invoked
stage5_novel_probe.csv        # only if invoked

ORDINAL_LAYOUT_ECOLOGY_REPORT.md
```

---

# Figures

## 29. Figure 1 — environment schematic

Show:

```text
same ordinal sequence:
P1 P2 P3 ... P8

Context A:
layout A coordinates

Context B:
same ordinal order, different coordinates
```

Make explicit:

> ordinal identity is shared; spatial realization differs.

---

## 30. Figure 2 — raw A/B RT learning

For High and Low:

```text
raw RT trajectory
A/B block shading
switch boundaries
```

This figure is a gating figure, not supplemental.

---

## 31. Figure 3 — Lump/Cross/Split bridge

Show canonical fits plus seed-level winner fractions.

---

## 32. Figure 4 — representation

Compare:

```text
categorical baseline
ordinal scaffold
ordinal + full layout
```

on:

```text
A/B hidden separation
matched-position similarity
```

---

## 33. Figure 5 — Mixed profiles

Only after Stage 2 passes.

Show:

```text
A
B
Mixed
```

across ordinal positions.

---

## 34. Figure 6 — Mixed corridor

Show:

```text
closest lambda on A/B segment
off-corridor distance
```

for each ordinal transition.

---

# Interpretation matrix

## 35. Result A

```text
ordinal scaffold preserves Cross/Split
full layout preserves Cross/Split
Mixed improves
```

Interpretation:

> representing the same visible ordinal sequence under different spatial realizations provides a compositional scaffold for recombination.

## 36. Result B

```text
Cross/Split preserved
Mixed still extrapolates
```

Interpretation:

> shared ordinal structure is insufficient by itself.

Proceed only to the sparse shared-item sweep.

## 37. Result C

```text
hidden states split
but RT becomes Lump
```

Reject the ecology.

Do not proceed to Mixed.

## 38. Result D

```text
full-layout input destroys splitting
```

Stop and diagnose shortcut/input-scaling problems.

Do not weaken the global context post hoc to rescue Mixed.

---

# Final report

## 39. Required report

Create:

```text
ORDINAL_LAYOUT_ECOLOGY_REPORT.md
```

with:

### 1. Reference phenotype
Did the categorical baseline show High Cross/Split and Low Lump?

### 2. Shared ordinal scaffold
Did explicit ordinal identity preserve the phenotype?

### 3. Context-specific full layouts
Did different spatial realizations preserve splitting and context-yoked RT?

### 4. Representation
Did matched ordinal positions remain structurally related across contexts?

### 5. Mixed
Did combining familiar layout/context evidence improve recombination?

### 6. Final interpretation

Choose one:

```text
A. shared ordinal scaffold + spatial realization is sufficient
B. it preserves splitting but not recombination
C. it weakens context-specific behavior too much
D. it does not preserve the original phenotype
```

Do not choose based on manuscript preference.

---

# Scientific target

The strongest result would be:

\[
\boxed{
\text{same visible ordinal sequence}
+
\text{different spatial/contextual realization}
\rightarrow
\text{context-specific skill states that remain compositionally related}
}
\]

The core theoretical claim would be:

> A and B need not themselves be fuzzy or highly overlapping. Their common structure is the same visible ordered sequence; context determines the spatial realization of that shared sequence.
