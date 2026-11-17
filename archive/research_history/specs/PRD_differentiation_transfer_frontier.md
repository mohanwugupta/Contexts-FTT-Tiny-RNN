# PRD: Differentiation–Transfer Frontier in SRNs
## Shared-vs-Specific Context Structure × Network Capacity

## 0. Purpose

The current human-geometry SRN results establish one useful fact:

> Human-like spatial geometry improves Mixed/Novel transfer while preserving strong A/B context differentiation.

In the current 20-seed run, the shared-x human geometry (`G2_shared_x`) preserved the High-scale Cross/Split phenotype and Low-scale Lump phenotype while reducing High-scale transfer deficits relative to the old synthetic geometry. It therefore identifies a genuine environmental factor that moves transfer in the desired direction without simply erasing the context split.

The next goal is to determine **what controls the tradeoff between context differentiation and cross-context transfer**.

This PRD tests two candidate axes:

1. **Environmental factorization** — how much of the contextual input is shared across A/B versus context-specific?
2. **Representational capacity** — does a larger recurrent state help the network retain both shared and context-specific structure, or does it simply allow stronger specialization?

The scientific target is not to maximize Mixed/Novel transfer at any cost. The target is to identify regimes that simultaneously support:

\[
oxed{
	ext{strong A/B differentiation}
+
	ext{context-yoked RT dynamics}
+
	ext{better Mixed/Novel transfer}
}
\]

---

# 1. Core hypotheses

## H1 — Shared contextual structure improves transfer

Increasing the fraction of contextual features that are literally shared between A and B should improve transfer to Mixed and Novel because the network receives more evidence that both contexts instantiate the same task/sequence.

However, too much shared structure may weaken A/B differentiation.

Expected qualitative tradeoff:

```text
low shared structure   -> strong split, poor transfer
intermediate sharing   -> strong split, improved transfer
high shared structure  -> weak split, strong transfer
```

The scientifically interesting regime is the middle.

## H2 — Capacity changes the differentiation–transfer tradeoff

A larger hidden state could have either of two effects.

### Factorization hypothesis

More capacity allows separate representation of:

```text
shared sequence/task structure
+
context-specific modulation
```

leading to:

```text
strong differentiation
+
better transfer
```

### Specialization hypothesis

More capacity simply allows A and B to be encoded more independently, producing:

```text
stronger differentiation
+
worse transfer
```

The capacity manipulation is therefore theoretically diagnostic and must not be framed as a generic performance optimization.

## H3 — Mixed and Novel may respond differently

Novel may benefit primarily from shared invariant structure:

> "I do not recognize the context, but I recognize the task/sequence."

Mixed may additionally require arbitration between two familiar context signals.

Therefore:

```text
Mixed and Novel must always be analyzed separately.
```

Do not collapse them into a single transfer score.

---

# 2. Frozen reference model

The reference environment is the validated human-like shared-x condition:

```text
G2_shared_x
```

Freeze:

```text
8-unit tanh Elman SRN
High initialization scale = 2.0
Low initialization scale = 0.02
same motor-token sequence
same explicit P1..P8 ordinal channels
same shared-x human geometry
same context-specific y geometry
same A10/B10 training curriculum
170 external A/B training sequences
same optimizer
same learning rate
same next-token objective
same two-parameter monotonic execution readout
RT = 1 - execution speed
same frozen-probe policy
Mixed/Novel receive zero optimizer steps
```

Do not alter the RT mechanism in this experiment.

---

# 3. Primary outcomes

Every condition must be evaluated along **two separate axes**.

## 3.1 Differentiation

Measure:

```text
A/B matched-position hidden-state distance
A/B matched-position cosine similarity
canonical Lump/Cross/Split state-space winner
context-yoked RT sawtooth / switch structure
A/B RT endpoint gap
```

## 3.2 Transfer

For Mixed and Novel separately:

```text
absolute prediction NLL
absolute normalized RT
next-token accuracy
NLL deficit over worse A/B endpoint
RT deficit over worse A/B endpoint
distance to A/B representational corridor
```

Prediction NLL is the upstream primary outcome.

RT is the downstream expression outcome.

---

# 4. Phase A — Shared-vs-specific contextual cue structure

## 4.1 Principle

Keep the human-like G2 shared-x geometry fixed.

Manipulate only the **composition of the contextual cue representation**.

The total cue dimensionality, number of active features, and L2 norm must remain fixed across conditions.

This prevents the manipulation from becoming:

```text
more shared cues = more total signal
```

Instead it tests:

```text
same amount of contextual information
distributed differently between
shared and context-specific features
```

## 4.2 Context cue bank

Replace the current 2-D A/B cue with a fixed 10-D binary feature bank.

For every context:

```text
cue_dim = 10
active_features_per_context = 5
```

Each cue is normalized:

\[
C_c = rac{	ilde C_c}{\|	ilde C_c\|_2}.
\]

Thus every A/B cue has exactly:

```text
5 active entries
L2 norm = 1
```

for every sharing condition.

Unused dimensions remain zero.

## 4.3 Shared-feature conditions

Let:

```text
m = number of active features shared by A and B
```

Test:

| Condition | m shared | Shared fraction of active cues |
|---|---:|---:|
| S0 | 0 | 0% |
| S1 | 1 | 20% |
| S2 | 2 | 40% |
| S3 | 3 | 60% |
| S4 | 4 | 80% |

Each context always has 5 active features.

Example construction:

### S0 — 0% shared

```text
A = [A1 A2 A3 A4 A5]
B = [B1 B2 B3 B4 B5]
```

### S2 — 40% shared

```text
A = [S1 S2 A1 A2 A3]
B = [S1 S2 B1 B2 B3]
```

### S4 — 80% shared

```text
A = [S1 S2 S3 S4 A1]
B = [S1 S2 S3 S4 B1]
```

The exact cue indices must be fixed in config before training.

Do not redraw cue assignments independently for different seeds.

---

# 5. Phase A reference-equivalence control

Because the original G2 model used a 2-D cue and Phase A uses a 10-D bank, first test whether the new `S0` environment reproduces the original G2 phenotype.

Run:

```text
G2_original_2D
S0_10D_disjoint
```

with paired seeds.

Require:

```text
same qualitative High Cross/Split phenotype
same qualitative Low Lump phenotype
comparable A/B hidden separation
comparable Mixed/Novel transfer deficits
```

The 10-D bank does not need to reproduce every numerical value exactly, but if it qualitatively changes the model before cue sharing is introduced, stop and diagnose the representation.

Do not interpret the sharing sweep until this equivalence control passes.

---

# 6. Phase A screening

Run all S0–S4 conditions with:

```text
5 paired seeds
High scale = 2.0
Low scale = 0.02
hidden size = 8
```

Use the same five initialization seeds across every sharing condition.

---

# 7. Phase A hard gates

A condition is eligible for transfer interpretation only if it preserves the core A/B phenomenon.

## Gate A1 — sequence learning

At least 4/5 seeds:

```text
late sequence NLL < early sequence NLL
```

for High and Low.

## Gate A2 — High > Low representational differentiation

At least 4/5 paired seeds:

```text
High A/B hidden separation > Low A/B hidden separation
```

## Gate A3 — High context-yoked state-space phenotype

At least 4/5 High seeds:

```text
Cross or Split
```

## Gate A4 — Low Lump phenotype

At least 4/5 Low seeds:

```text
Lump
```

## Gate A5 — visible High context-yoked RT structure

At least 4/5 High seeds must show a positive registered switch/sawtooth metric.

Do not rescue a condition by altering the gate after seeing transfer results.

---

# 8. Phase A transfer analyses

For every gate-passing condition, compute separately for Mixed and Novel:

\[
G^{NLL}
=
NLL_{probe}
-
\max(NLL_A,NLL_B)
\]

\[
G^{RT}
=
RT_{probe}
-
\max(RT_A,RT_B).
\]

Also report:

```text
absolute probe NLL
absolute RT
next-token accuracy
A/B hidden separation
A/B RT gap
```

The main scientific question is whether increasing shared cue structure moves conditions toward:

```text
lower transfer deficit
without lower differentiation
```

---

# 9. Phase A E3 cue × layout decomposition

For each gate-passing sharing condition, rerun the validated E3 decomposition.

Report for Mixed and Novel:

```text
cue main effect
layout main effect
cue × layout interaction
```

for:

```text
prediction NLL
normalized RT
```

Hypothesis:

```text
more shared cue structure
-> reduced pathological cue×layout conjunction sensitivity
```

but this is not required for transfer improvement.

Do not infer mechanism from transfer metrics alone.

---

# 10. Phase A differentiation–transfer frontier

For every seed and sharing condition, plot:

### x-axis

```text
A/B differentiation
```

Primary:

\[
D_{AB}
=
	ext{mean matched-position hidden distance}
\]

### y-axis

```text
transfer deficit
```

Primary:

\[
G^{NLL}_{Mixed}
\]

and separately:

\[
G^{NLL}_{Novel}.
\]

This is the primary Phase-A visualization.

Do not collapse differentiation and transfer into one arbitrary composite score.

---

# 11. Phase A confirmatory selection

After the five-seed screen, define:

### Baseline condition

```text
S0
```

### Best admissible sharing condition

Among conditions that pass all A/B gates, select the condition with the lowest mean High-scale `Mixed NLL deficit`.

Tie-breaker:

```text
lower Novel NLL deficit
```

This selection rule must be implemented before the 20-seed confirmatory run.

Also retain:

```text
S4
```

as the deliberately high-sharing comparison if it passes the five-seed gates.

Run 20 seeds for:

```text
S0
best admissible sharing condition
S4 if gate-passing
```

If S4 fails the A/B gate, record that failure instead of forcing a confirmatory run.

---

# 12. Phase B — Network capacity

## 12.1 Conditions

Use up to three environmental conditions from Phase A:

```text
E_specific = S0
E_best = best admissible Phase-A condition
E_shared = S4, if it passed
```

Test hidden sizes:

```text
H = 8
H = 16
H = 32
```

Run both:

```text
High scale = 2.0
Low scale = 0.02
```

---

# 13. Capacity initialization rule

Do not accidentally change effective activation scale simply because hidden width changes.

Preserve the repository's existing initialization family.

If the current initialization is:

\[
W \sim \mathcal N(0,\sigma^2)
\]

with explicit scale multiplier, maintain the same **fan-in-aware base initialization** and apply the same scale multiplier:

```text
High = 2.0
Low = 0.02
```

for every hidden size.

Document:

```text
fan-in
base std
scale multiplier
realized weight std
```

for each H.

If the repository currently lacks fan-in normalization, add no new normalization silently. Instead reproduce the existing rule and report realized spectral radius / weight norms by H.

---

# 14. Phase B screening

Run:

```text
5 paired seeds
```

for each:

```text
environment × hidden size × scale
```

Use matched seed identities wherever parameter shapes permit.

---

# 15. Phase B hard gates

Use the same A/B gates as Phase A.

A larger network is not considered better if it obtains lower transfer deficit by becoming Lump.

For each condition report:

```text
High Cross/Split fraction
Low Lump fraction
A/B hidden separation
Mixed NLL deficit
Novel NLL deficit
```

---

# 16. Phase B confirmatory analyses

The key statistical question is the interaction:

\[
	ext{environmental sharing}
	imes
	ext{hidden capacity}
\]

on:

```text
A/B differentiation
Mixed transfer deficit
Novel transfer deficit
```

At minimum report seed-level bootstrap 95% CIs.

Preferred confirmatory model:

```text
metric ~ shared_fraction * log2(hidden_size) + (1|seed)
```

fit separately for:

```text
A/B differentiation
Mixed NLL deficit
Novel NLL deficit
```

Do not interpret p-values without the effect-size plots.

---

# 17. Factorization diagnostics

For each Phase-B condition, test whether larger networks represent shared and specific structure more cleanly.

At each ordinal position \(k\), estimate:

\[
h_{c,k}
pprox
s_k+d_{c,k}.
\]

Report:

```text
shared-component variance
context-specific residual variance
A/B matched-position cosine
A/B matched-position distance
```

Optional low-rank diagnostic:

1. form the 16-state matrix:
   ```text
   A(P1..P8), B(P1..P8)
   ```
2. perform PCA/SVD;
3. quantify how many components explain:
   ```text
   shared ordinal variation
   A/B contextual displacement
   ```

This is descriptive.

Do not force the hidden representation to be linear.

---

# 18. Mixed/Novel corridor diagnostic

For every confirmatory condition, compute for each seed and transition:

```text
closest position on A/B hidden segment
off-corridor distance
inside/outside segment indicator
```

Also compute canonicalized within-seed responsibility:

\[
q_{s,k}
=
rac{RT_{M,s,k}-RT_{Fast,s,k}}
{RT_{Slow,s,k}-RT_{Fast,s,k}}.
\]

Use this diagnostically only.

Do not optimize condition selection using `q`.

---

# 19. Optional endpoint canonicalization

For population visualization only:

```text
Fast = lower mean frozen A/B RT within seed
Slow = higher mean frozen A/B RT within seed
```

Swap the entire A/B profile within seed.

Never flip individual transitions.

Never use Mixed/Novel to determine the swap.

All primary transfer-deficit calculations must remain label-invariant and within-seed.

---

# 20. Stop conditions

Stop the experiment branch if any of the following occur:

### Stop 1

The S0 10-D equivalence control fails to reproduce the original G2 qualitative phenotype.

### Stop 2

All shared-cue conditions that improve transfer also collapse High-scale Cross/Split behavior.

Interpretation:

> cue sharing moves only along the differentiation–transfer tradeoff; it does not create a better regime.

### Stop 3

Capacity monotonically improves transfer only by reducing A/B differentiation.

Interpretation:

> larger hidden size is not solving factorization.

### Stop 4

No sharing or capacity manipulation materially changes NLL transfer deficit.

Interpretation:

> the remaining bottleneck is likely retrieval/arbitration rather than environmental factorization or capacity.

Do not add additional arbitrary knobs after a stop condition.

---

# 21. Required outputs

## Configuration

```text
cue_factorization_config.json
capacity_grid_config.json
```

## Phase A

```text
phaseA_equivalence_control.csv
phaseA_training_summary.csv
phaseA_state_space.csv
phaseA_hidden_differentiation.csv
phaseA_transfer_metrics.csv
phaseA_E3_decomposition.csv
phaseA_corridor_metrics.csv
```

## Phase B

```text
phaseB_training_summary.csv
phaseB_state_space.csv
phaseB_hidden_differentiation.csv
phaseB_transfer_metrics.csv
phaseB_factorization_metrics.csv
phaseB_corridor_metrics.csv
```

## Report

```text
DIFFERENTIATION_TRANSFER_FRONTIER_REPORT.md
```

---

# 22. Required figures

## Figure 1 — Cue-factorization schematic

Illustrate:

```text
S0: 0 shared / 5 unique per context
S1: 1 shared
S2: 2 shared
S3: 3 shared
S4: 4 shared
```

Show constant:

```text
10-D bank
5 active cues/context
unit L2 norm
```

## Figure 2 — Phase-A A/B phenotype

For S0–S4:

```text
High Cross/Split winner fractions
Low Lump winner fractions
mean A/B hidden separation
```

## Figure 3 — Phase-A transfer

Plot separately for Mixed and Novel:

```text
NLL deficit vs shared fraction
RT deficit vs shared fraction
```

Mark conditions failing the A/B gates.

## Figure 4 — Differentiation–transfer frontier

For each seed:

```text
x = A/B hidden differentiation
y = Mixed NLL deficit
```

and separate panel:

```text
y = Novel NLL deficit
```

Color by shared fraction.

## Figure 5 — E3 interaction across shared fraction

Plot:

```text
|cue × layout interaction|
```

for:

```text
Mixed
Novel
```

using NLL as primary.

## Figure 6 — Capacity map

Grid or line plot:

```text
hidden size × environment
```

showing:

```text
A/B differentiation
Mixed transfer
Novel transfer
```

## Figure 7 — Phase diagram

Create a compact 2-D summary:

```text
x = shared cue fraction
y = hidden size
```

Each cell annotated with:

```text
High Cross/Split fraction
Mixed NLL deficit
Novel NLL deficit
```

Do not create a single composite "success score."

---

# 23. TDD requirements

Use RED → GREEN → REFACTOR.

## T1 — fixed cue width

Every sharing condition:

```text
cue.shape == (10,)
```

## T2 — fixed active count

For A and B:

```text
number of nonzero cue entries == 5
```

for every S0–S4.

## T3 — fixed norm

For every cue:

```text
||C||_2 == 1
```

within `1e-8`.

## T4 — exact overlap count

For condition Sm:

```text
count(shared nonzero indices between A and B) == m
```

## T5 — fixed geometry

Human G2 shared-x geometry must be byte-identical across S0–S4.

## T6 — fixed motor sequence

Motor-token sequence and P1..P8 sequence must be unchanged.

## T7 — frozen training curriculum

All conditions use identical:

```text
A10/B10
170 external sequences
```

## T8 — paired seeds

Same seed IDs across sharing conditions.

## T9 — frozen probe

Mixed/Novel:

```text
0 optimizer steps
0 parameter updates
```

## T10 — capacity-only parameter change

Within Phase B, changing H may alter only architecture dimensions required by hidden width.

Environment, cue composition, curriculum, optimizer, RT readout form, and probe policy remain fixed.

## T11 — G2 reproduction

Before the sweep, reproduce the validated G2 shared-x reference within tolerance on saved reference seeds.

## T12 — scale logging

For every hidden size, log:

```text
realized input-weight std
recurrent-weight std
spectral radius
hidden activation norm
```

to detect accidental width-dependent scale changes.

---

# 24. Interpretation matrix

## Outcome A — intermediate sharing improves transfer while preserving split

Interpretation:

> A factorized environment supports both context-specific specialization and reusable shared structure.

This is the main positive theoretical result.

## Outcome B — more sharing smoothly trades differentiation for transfer

Interpretation:

> the model lies on a basic specialization–generalization frontier; cue sharing alone does not reproduce the human regime.

## Outcome C — larger capacity improves both differentiation and transfer

Interpretation:

> sufficient representational capacity allows the SRN to maintain separate context-specific states while preserving a shared reusable substrate.

This supports the factorization hypothesis.

## Outcome D — larger capacity increases differentiation and worsens transfer

Interpretation:

> capacity primarily enables specialization.

This supports the specialization hypothesis.

## Outcome E — capacity matters only at intermediate sharing

Interpretation:

> environmental factorization and representational capacity interact: shared structure must exist before extra capacity can exploit it.

This would be especially theoretically informative.

## Outcome F — Novel improves strongly but Mixed plateaus

Interpretation:

> shared invariant structure helps generalization to a new context, but simultaneous familiar-cue arbitration requires an additional retrieval mechanism.

This is a plausible and scientifically clean endpoint.

---

# 25. Final scientific question

The experiment should answer:

> Under what combinations of environmental shared structure and representational capacity can an SRN learn two strongly context-specific versions of the same sequence while still transferring that knowledge to ambiguous or novel contexts?

The goal is to move from:

```text
"this SRN sometimes averages"
```

to a computational account of:

\[
oxed{
	ext{when context splitting is compatible with generalization}.
}
\]
