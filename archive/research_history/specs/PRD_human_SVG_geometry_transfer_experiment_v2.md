# PRD: Human-SVG Geometry Transfer Experiment

## Objective

Test whether the SRN's remaining Mixed/Novel transfer deficit is partly caused by an unrealistic spatial-layout geometry.

The previous SRN environment used strongly different 2-D A/B trajectories. Inspection of the actual human stimuli shows a much more factorized structure:

- the eight sequence items are ordered left-to-right in every supplied human layout;
- normalized horizontal position is nearly invariant across layouts;
- vertical position varies by context, but its range is much smaller than the horizontal sequence span;
- therefore, human participants receive a strong spatial cue to shared ordinal structure in addition to the explicit sequence identity.

The experiment should replace the synthetic spatial layouts with coordinates extracted directly from the supplied human SVGs while freezing the SRN architecture, learning objective, curriculum, and RT readout.

The key question is:

> Does human-faithful spatial geometry preserve A/B splitting while reducing the High-scale cue/layout conjunction sensitivity and improving Mixed/Novel transfer?

---

## 1. Coordinate source and definition

Use the supplied coordinate registry files:

```text
human_svg_target_coordinates.csv
human_svg_target_coordinates.json
```

Coordinates were derived from the center of the visible response-label glyph for each of the eight sequence items in the supplied SVGs.

Items are ordered left-to-right and assigned to:

```text
P1 ... P8
```

Important:

> All layouts must use the same canonical sequence found in M0/M1/S0:

```text
S A F S D A F D
```

The originally supplied `exampleSequence.svg` accidentally displayed a different letter sequence. Retain only its **target x/y geometry** and overwrite its item identities with the canonical sequence above. The different letters in that SVG are not an experimental manipulation and must never enter SRN inputs, targets, or analyses.

---

## 2. Normalization

For each layout separately:

\[
x'_k = \frac{x_k-x_{\text{mid}}}{(x_{\max}-x_{\min})/2}
\]

\[
y'_k = -\frac{y_k-\bar y}{(x_{\max}-x_{\min})/2}.
\]

Thus:

- P1 and P8 span approximately `[-1, +1]` on x;
- SVG y is inverted so positive model y means visually upward;
- **the same horizontal scale is used for x and y**.

Do NOT independently normalize or z-score y.

That would erase the key human property being tested: vertical variation is small relative to the horizontal sequence span.

---

## 3. Derived normalized coordinates

### M0

```text
P1  (-1.0000, -0.0712)
P2  (-0.7075,  0.0996)
P3  (-0.4336,  0.0044)
P4  (-0.1431, -0.0712)
P5  ( 0.1450, -0.0329)
P6  ( 0.4328,  0.0996)
P7  ( 0.7069,  0.0046)
P8  ( 1.0000, -0.0329)
```

### M1

```text
P1  (-1.0000,  0.0261)
P2  (-0.6694, -0.0858)
P3  (-0.3738, -0.0127)
P4  (-0.0612,  0.0282)
P5  ( 0.2143,  0.0646)
P6  ( 0.4687, -0.0780)
P7  ( 0.7639, -0.0094)
P8  ( 1.0000,  0.0670)
```

### S0

```text
P1  (-1.0000, -0.1956)
P2  (-0.7162,  0.0041)
P3  (-0.4350,  0.1926)
P4  (-0.1429, -0.0815)
P5  ( 0.1420,  0.2125)
P6  ( 0.4303,  0.0044)
P7  ( 0.7213, -0.1829)
P8  ( 1.0000,  0.0464)
```

### exampleSequence

```text
P1  (-1.0000, -0.0375)
P2  (-0.7284,  0.0070)
P3  (-0.4241,  0.1107)
P4  (-0.1545, -0.0375)
P5  ( 0.1529, -0.0800)
P6  ( 0.4371,  0.0066)
P7  ( 0.7374,  0.1107)
P8  ( 1.0000, -0.0800)
```

---

## 4. Geometry audit

Before any training, reproduce these descriptive checks.

Expected human-layout geometry:

```text
M0:              y-span/x-span ≈ 0.085
M1:              y-span/x-span ≈ 0.076
S0:              y-span/x-span ≈ 0.204
exampleSequence: y-span/x-span ≈ 0.095
```

Normalized x profiles across every pair of supplied layouts should correlate at:

```text
r > 0.999
```

The coding agent must additionally compute these same metrics for the **old synthetic SRN layouts**:

```text
x-profile correlation across A/B
y-span/x-span
full-layout Euclidean distance
mean matched-position coordinate distance
```

This comparison is part of the scientific result.

---

## 5. Condition mapping

Use the actual human-task mapping from SVG filename to experimental context/probe condition already present in the task source or repository.

Do NOT infer context identity from filename.

If the mapping is not recoverable from the code/configuration:

```text
STOP and report the missing mapping.
```

Do not arbitrarily assign M0/M1/S0/exampleSequence to A/B/Mixed/Novel.

---

## 6. Frozen SRN specification

Freeze the current successful model:

```text
8-unit tanh Elman SRN
High initialization scale = 2.0
Low initialization scale = 0.02
same fixed motor sequence
same ordinal P1..P8 channels
same A10/B10 curriculum
170 external training sequences for primary analysis
same optimizer and learning rate
same next-token objective
same two-parameter monotonic execution readout
RT = 1 - execution_speed
```

Do not alter:

```text
hidden size
initialization scale
training duration
execution objective
RT mapping
probe update policy
```

The only primary manipulation is spatial geometry.

---

## 7. Input representation

At sequence position \(k\), preserve the current semantic structure:

\[
x_{c,k}
=
[
\text{motor token}_k,
P_k,
\mathcal L_c,
C_c
].
\]

Where:

- `motor token` = repository sequence input;
- `P_k` = shared ordinal identity;
- `L_c` = the full 8-item SVG-derived layout, visible at every step;
- `C_c` = existing context cue.

The full layout is:

\[
\mathcal L_c =
[x_1,y_1,\ldots,x_8,y_8].
\]

---

## 8. Experimental conditions

### G0 — Old synthetic geometry

Exact current reference environment.

No changes.

### G1 — Exact SVG-derived geometry

Replace the synthetic layouts with the exact normalized coordinate sets from the relevant human SVGs.

Everything else is frozen.

This is the primary experimental condition.

### G2 — Shared-x factorized control

Construct a single canonical horizontal vector:

\[
\bar x_k
=
\text{mean normalized }x_k
\]

across the supplied human SVG layouts.

For each context:

```text
x_k = pooled human x_k
y_k = that context's exact human y_k
```

This tests the hypothesis that the near-invariant left-to-right scaffold is the critical feature rather than tiny context-specific x deviations.

G2 is confirmatory/mechanistic, not a rescue condition.

Do not tune the pooled x values.

---

## 9. Screening and confirmatory seeds

For G1 and G2:

```text
5 paired seeds first
```

If the A/B splitting gates pass:

```text
20 paired seeds
```

Use the same seed identities as G0.

---

## 10. Hard gate 1 — preserve context differentiation

Better transfer is uninterpretable if it comes from destroying the A/B split.

Require for High scale:

```text
robust A/B hidden-state separation
majority Cross or Split state-space classification
visible context-yoked RT sawtooth
```

Require for Low scale:

```text
substantially lower A/B hidden separation
majority Lump
```

Report seed-level classifications.

Do not proceed to a positive transfer interpretation if High becomes predominantly Lump.

---

## 11. Hard gate 2 — prediction before RT

The recent transfer program showed that the primary deficit already exists in next-token prediction.

Therefore all transfer analyses must report:

```text
prediction NLL
correct-token confidence
normalized RT
```

Interpret NLL as upstream evidence and RT as expressed execution.

A result that improves RT without improving prediction is not sufficient.

---

## 12. Primary transfer analysis

At checkpoint 170, evaluate frozen:

```text
A
B
Mixed
Novel
```

using the same probe/reset policy as the validated transfer program.

For Mixed and Novel compute:

### Absolute probe performance

```text
mean prediction NLL
mean normalized RT
next-token accuracy
```

### Relative deficit

For each seed/transition:

\[
G^{NLL}
=
NLL_{probe}
-
\max(NLL_A,NLL_B)
\]

and analogously for RT:

\[
G^{RT}
=
RT_{probe}
-
\max(RT_A,RT_B).
\]

Also retain the absolute A/B gap.

---

## 13. Critical comparison

Compare:

```text
G0 old synthetic geometry
G1 exact human SVG geometry
G2 shared-x human geometry
```

The desired result is:

```text
Mixed/Novel deficit decreases
WHILE
High-scale A/B differentiation remains strong
```

Do not collapse these into one scalar score.

Plot transfer against differentiation jointly.

---

## 14. Re-run E3 cue/layout decomposition

This is essential.

For G0, G1, and G2, repeat the validated E3 2×2 decomposition:

```text
cue familiar vs altered
layout familiar vs altered
```

for Mixed and Novel.

Report for NLL and RT:

```text
cue main effect
layout main effect
cue × layout interaction
```

Primary mechanistic prediction:

\[
|\text{cue}\times\text{layout interaction}|_{G1/G2}
<
|\text{cue}\times\text{layout interaction}|_{G0}
\]

while High-scale A/B differentiation remains intact.

This would support the idea that the old synthetic geometry encouraged unnecessarily configural cue-layout binding.

---

## 15. Input-space interpolation diagnostic

Repeat the existing A→B interpolation diagnostic separately for:

```text
cue interpolation
layout interpolation
```

under G0/G1/G2.

Question:

> Does human-like geometry make the High-scale prediction function smoother along layout interpolation while preserving endpoint specialization?

Report NLL as the primary outcome.

---

## 16. Representational geometry

At each ordinal position \(k\), compute:

```text
||h_A,k - h_B,k||
cos(h_A,k, h_B,k)
same-position vs different-position similarity
```

For Mixed/Novel compute:

```text
distance to A/B line segment
closest lambda on A/B segment
distance to nearest trained A/B state
```

Do not require hidden-state linear interpolation as a success criterion.

Use it diagnostically.

---

## 17. Human-geometry-specific analysis

Quantify whether the learned hidden geometry tracks the supplied spatial geometry.

For each seed/context:

1. construct the 8×8 pairwise Euclidean distance matrix among human SVG target coordinates;
2. construct the corresponding hidden-state distance matrix;
3. compute rank correlation between spatial and hidden representational distances.

Compare G0/G1/G2.

This is secondary.

It tests whether the model is actually using the human spatial scaffold rather than merely treating the full layout vector as another context ID.

---

## 18. Optional long-training robustness

Only after checkpoint-170 analysis is frozen.

Re-run G0 vs G1 at checkpoint:

```text
680 completed A/B sequences
```

using the same E1 transfer metrics.

Purpose:

> determine whether human geometry changes the specialization–transfer imbalance over extended learning.

Do not use long-training results to select geometry.

---

## 19. Required figures

### Figure 1 — Geometry audit

Show all four supplied normalized human layouts.

Use equal x/y axis scaling.

Also show the old synthetic A/B layouts.

Annotate:

```text
y-span/x-span
x-profile correlation
```

### Figure 2 — A/B learning and state-space bridge

For G0/G1/G2:

```text
High raw RT
Low raw RT
Lump/Cross/Split winner fractions
```

### Figure 3 — Transfer deficit

For Mixed and Novel:

```text
prediction NLL deficit
RT deficit
```

across G0/G1/G2.

### Figure 4 — Differentiation vs transfer

x-axis:

```text
A/B hidden separation or A/B RT gap
```

y-axis:

```text
probe deficit over worse endpoint
```

Each point = seed/geometry.

### Figure 5 — E3 interaction

Compare cue×layout interaction magnitude across G0/G1/G2.

### Figure 6 — Input-space interpolation

Cue and layout interpolation curves for High and Low.

---

## 20. TDD requirements

Use RED → GREEN → REFACTOR.

### T1 — coordinate registry exactness

Loaded normalized coordinates must match `human_svg_target_coordinates.json` within:

```text
1e-6
```

### T2 — ordinal order

For every human layout:

```text
x(P1) < x(P2) < ... < x(P8)
```

### T3 — aspect-ratio preservation

No independent y normalization.

Verify:

```text
computed y-span/x-span == registry value
```

within tolerance.

### T4 — canonical motor sequence invariant

Every geometry condition and every layout must use exactly:

```text
S A F S D A F D
```

at ordinal positions P1..P8.

The accidental letter ordering in the originally supplied `exampleSequence.svg` must be ignored. Only its x/y target coordinates may differ.

### T5 — full layout visibility

The same full 16-D layout vector is available at every step within a context.

### T6 — paired-seed initialization

Matching G0/G1/G2 seeds use identical model initialization except for layout inputs.

### T7 — probe firewall

No Mixed/Novel optimizer steps in frozen transfer analyses.

### T8 — context cue frozen

Do not change context-cue amplitude across geometry conditions.

### T9 — training curriculum frozen

A10/B10 and total sequences remain identical.

### T10 — old synthetic reference reproduced

G0 must reproduce the validated baseline before comparing G1/G2.

---

## 21. Required outputs

```text
geometry_condition_config.json

geometry_audit.csv
geometry_training_summary.csv
geometry_state_space.csv
geometry_transfer_metrics.csv
geometry_E3_decomposition.csv
geometry_interpolation.csv
geometry_hidden_geometry.csv

HUMAN_SVG_GEOMETRY_TRANSFER_REPORT.md
```

---

## 22. Interpretation matrix

### Result A

```text
G1/G2 preserve High splitting
AND reduce Mixed/Novel NLL deficit
AND reduce cue×layout interaction
```

Interpretation:

> unrealistic synthetic spatial geometry was a substantive contributor to excessive configural specialization.

### Result B

```text
G1/G2 improve transfer
BUT A/B differentiation collapses
```

Interpretation:

> human geometry reduces specialization generally; it does not solve the differentiation–transfer problem.

Do not claim successful mechanism.

### Result C

```text
splitting preserved
cue×layout interaction decreases
but transfer does not improve
```

Interpretation:

> synthetic geometry contributed to configural binding, but another expression/retrieval mechanism still limits probe transfer.

### Result D

```text
G1/G2 closely match G0
```

Interpretation:

> spatial geometry mismatch is not a major explanation for the SRN–human transfer difference.

Return focus to retrieval/inference rather than further environment tuning.

### Result E

```text
G2 outperforms G1
```

Interpretation:

> the near-invariant horizontal ordinal scaffold, rather than exact SVG idiosyncrasies, is the important feature.

### Result F

```text
G1 outperforms G2
```

Interpretation:

> context-specific residual x/y geometry carries useful information beyond a perfectly factorized shared-x scaffold.

---

## 23. Scientific rationale

The old model asked the SRN to learn the same sequence under two strongly different 2-D trajectories.

The human stimuli instead provide a striking invariant:

\[
x_{c,k}\approx x_k
\]

for every context, while context variation is concentrated more strongly in vertical placement and scene identity.

Across the supplied SVG layouts, the normalized horizontal profiles correlate above `r=.999`.

Therefore the human environment gives the learner redundant evidence that:

> P3 in one context is the same ordinal P3 in another context.

The experiment tests whether this shared perceptual scaffold allows the SRN to preserve context-specific skills without binding sequence prediction as strongly to an arbitrary cue-layout conjunction.
