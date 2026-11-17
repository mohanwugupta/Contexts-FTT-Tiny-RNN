# PRD: Confirmatory SRN Pipeline for Context-Specific RT and Frozen Probe Analysis

## Goal

Convert the current SRN work from exploratory model development into a **frozen confirmatory pipeline**.

The execution mechanism is now locked:

\[
	ext{context + current sequence state}
ightarrow h_t
ightarrow p_t=P(	ext{correct next item}\mid h_t)
ightarrow v_t
ightarrow RT_t
\]

with the primary execution readout:

\[
v_t=\sigma\left[a\,\operatorname{logit}(p_t)+bight],\qquad a>0,
\]

and normalized RT:

\[
RT_t = 1-v_t.
\]

The purpose of this PRD is not to search for a better RT objective. The purpose is to test, with a fixed model and fixed training ecology:

1. whether representational scale controls context-specific sequence organization;
2. whether that organization produces context-specific RT dynamics;
3. whether frozen Mixed and Novel probes produce structured recombination of the learned A/B states;
4. whether the same computational organization can later be compared directly with human data;
5. whether intervening on the learned A/B representational structure causally shifts model RT.

---

## 1. Scientific hypotheses

### H1 — representational flexibility

High-scale SRNs should develop stronger A/B hidden-state separation than Low-scale SRNs.

### H2 — behavioral expression

High-scale networks should show more context-specific RT dynamics. Canonical Lump/Cross/Split state-space fits should favor a more context-specific solution in High scale than Low scale.

Ideal, but not required:

```text
High scale -> Cross or Split
Low scale  -> Lump
```

### H3 — sequence-only control

High-scale sequence-only networks may still show context-specific hidden-state organization and next-token loss dynamics. This tests whether high scale creates the representation while the execution rule makes it visible in RT.

### H4 — frozen probe recombination

Mixed and Novel contexts should show structured relationships to learned A/B states. Do not require exact midpoint averaging.

### H5 — causal representational control

Moving a probe hidden state toward its matched A or B component should move model RT toward the corresponding A or B RT while preserving next-token identity/accuracy.

---

## 2. Hard freeze

Do not change the following after implementation begins.

### Architecture

```text
8-unit tanh Elman SRN
4-way next-token output
3D context basis
```

### High scale

```text
initialization scale = 2.0
```

Use the exact same scale convention as the existing codebase.

### Low scale

Use the existing Low-scale value from the current implementation. Do not retune it.

### Sequence

Use the existing 8-item FTT sequence:

```text
2-1-4-2-3-1-4-3
```

with seven predicted transitions per sequence.

### Training ecology

Preserve exactly:

```text
10 sequences in Context A
10 sequences in Context B
10 sequences in Context A
10 sequences in Context B
...
```

for the same total:

```text
170 external training sequences
```

Do not alter block lengths, number of switches, sequence exposure, optimizer schedule, or training duration except as required to reproduce the currently frozen working run.

### Context basis

Use:

```text
A          = [1, 0, 0]
B          = [0, 1, 0]
Mixed_norm = [0.5, 0.5, 0]
Novel      = [0, 0, 1]
```

Optional control only:

```text
Mixed_additive = [1, 1, 0]
```

Mixed and Novel are never used for training or model selection.

---

## 3. Frozen execution readout

Primary readout:

\[
v_t=\sigma\left[a\,\operatorname{logit}(p_t)+bight],\qquad a>0
\]

where:

\[
p_t=P(	ext{correct next item}\mid h_t).
\]

Then:

\[
RT_t=1-v_t.
\]

Use a positive parameterization such as:

\[
a=\operatorname{softplus}(lpha)
\]

or equivalent.

The readout may learn only:

```text
a
b
```

It may not read arbitrary hidden units directly.

---

## 4. Control readouts

Retain only as controls.

### C1 — deterministic normative policy

Use the existing deterministic normative confidence-to-speed policy as a normative reference.

### C2 — unrestricted hidden-state execution head

Use only as a negative/flexibility control showing why unconstrained hidden-state-to-RT mappings are less interpretable.

Do not make it the primary model.

---

## 5. Confirmatory model conditions

Run exactly these three primary conditions:

```text
1. High scale = 2 + monotonic execution
2. Low scale + monotonic execution
3. High scale = 2 + sequence only
```

Use:

```text
20 independent seeds per condition
```

Recommended fixed seed list:

```text
42-61
```

or use the repo’s existing canonical 20-seed list. Freeze the list before running.

---

## 6. TDD requirements

Use RED → GREEN → REFACTOR.

### T1 — High-scale regression test

Require:

```text
high_scale == 2.0
```

for all High-scale confirmatory runs.

### T2 — hidden size

Require:

```text
hidden_dim == 8
```

for every confirmatory condition.

### T3 — training curriculum

Require exactly:

```text
A x 10
B x 10
A x 10
B x 10
...
```

for all 170 training sequences.

### T4 — context vectors

Regression-test exact values for:

```text
A
B
Mixed_norm
Novel
```

### T5 — no probe training

Probe evaluation must produce:

```text
0 optimizer steps
0 parameter updates
```

### T6 — monotonic execution

Numerically verify:

```text
p1 < p2  ->  speed(p1) <= speed(p2)
```

for a dense grid over \(p\in(0,1)\).

### T7 — RT transform

Verify:

```text
higher speed -> lower RT
```

### T8 — sequence-only control

Require:

```text
no execution head
no RT loss
no RT output used during training
```

for the sequence-only condition.

### T9 — reproducibility

Running the same seed/config twice must reproduce training loss, final weights, and probe outputs within deterministic numerical tolerance.

---

## 7. Phase 1 — confirmatory training

For every seed and condition, save per external sequence:

```text
condition
seed
training_sequence
block_index
context
within_block_sequence
is_context_switch
sequence_CE
mean_correct_token_confidence
mean_execution_speed
mean_normalized_RT
```

For sequence-only, execution columns may be NA.

Also save per transition:

```text
condition
seed
training_sequence
context
transition_position
current_token
target_token
correct_token_confidence
transition_CE
execution_speed
normalized_RT
hidden_state_1
...
hidden_state_8
```

---

## 8. Training diagnostics

For each condition produce:

### A. Sequence learning

Plot sequence CE across all 170 sequences with 20 individual seeds, mean, 95% CI, and A/B block coloring.

### B. Confidence acquisition

Plot mean correct-token confidence across training.

### C. Emergent RT

For the two execution conditions, plot normalized RT across training with A/B block coloring.

### D. Switch-aligned RT

Align all A↔B switches to:

```text
-3 -2 -1 | 0 +1 +2 +3
```

where 0 is the first sequence in the new context.

Compute:

\[
SwitchCost
=
RT_0
-
rac{RT_{-3}+RT_{-2}+RT_{-1}}{3}.
\]

Report mean, 95% CI, and seed-level distribution for High and Low scale.

---

## 9. Representational separation analysis

Use frozen hidden states.

Primary comparison:

```text
High scale
Low scale
High-scale sequence-only
```

At matched transition position \(k\), calculate:

\[
d_k
=
\left\|
ar h_{A,k}-ar h_{B,k}
ight\|.
\]

Also calculate an aggregate A/B separation score across positions.

Required analyses:

```text
token-matched centroid distance
shared PCA visualization
seed-level separation score
High vs Low comparison
```

Optional only if already stable:

```text
Procrustes / subspace angle
```

Do not make dimensionality reduction the main statistical test.

---

## 10. Canonical state-space bridge

Fit the existing canonical human-model ladder to emergent RT:

```text
Lump
Cross
Split
```

Use the same equations and parameterization already established in the project.

Do not invent a new SRN-specific version.

Fit independently for:

```text
High-scale + monotonic execution
Low-scale + monotonic execution
```

Primary outputs:

```text
BIC
AIC
RMSE
R²
best model
model parameters
```

Save seed-level results as well as seed-averaged overlays.

---

## 11. Required state-space figures

For High and Low scale show:

1. observed emergent RT;
2. Lump fit;
3. Cross fit;
4. Split fit;
5. best-model label;
6. RMSE / R²;
7. BIC comparison.

Also produce:

```text
fraction of seeds classified Lump / Cross / Split
```

Seed-level classification is primary alongside the mean trajectory.

---

## 12. Frozen probe evaluation

At the final frozen checkpoint for every seed, evaluate:

```text
A
B
Mixed_norm
Novel
```

without learning.

Optional:

```text
Mixed_additive
```

For every probe save:

```text
seed
condition
probe_context
transition_position
current_token
target_token
correct_token_confidence
execution_speed
normalized_RT
sequence_CE
hidden_state_1
...
hidden_state_8
```

Use the exact same evaluation protocol for A, B, Mixed, and Novel.

---

## 13. Probe-level behavioral summary

For each seed and transition position calculate:

```text
RT_A(k)
RT_B(k)
RT_Mixed(k)
RT_Novel(k)
```

Define the A/B midpoint:

\[
M_k=rac{RT_A(k)+RT_B(k)}{2}.
\]

Define probe deviations:

\[
D^{Mixed}_k=RT_{Mixed}(k)-M_k
\]

\[
D^{Novel}_k=RT_{Novel}(k)-M_k.
\]

Do not collapse over transition position before saving these values.

---

## 14. Simple responsibility analysis

Before any low-rank analysis, calculate:

\[
q_{C,k}
=
rac{RT_C(k)-RT_B(k)}
{RT_A(k)-RT_B(k)}.
\]

Use only when the A/B denominator is sufficiently nonzero.

Interpretation:

```text
q = 1 -> A-like
q = 0 -> B-like
q = .5 -> midpoint
```

Do not clip q in the primary analysis.

Also fit one global scalar \(q_C\) across all transitions.

This gives the first interpretable comparison:

```text
global scalar responsibility
vs
transition-specific responsibility
```

---

## 15. Transition-specific structure

For every seed define:

\[
\Delta^{AB}_k=RT_A(k)-RT_B(k).
\]

Compare:

```text
Mixed q_k profile
Novel q_k profile
```

against the seven-position A/B transition structure.

Primary descriptive questions:

1. Are probe effects constant across transition positions?
2. Do they systematically depend on A/B separation?
3. Do Mixed and Novel share the same profile?
4. Does High scale show more structured probe modulation than Low scale?

---

## 16. Low-rank bridge analysis

Only after the scalar analysis is complete.

Construct the seven-dimensional transition-profile matrix from A/B-trained states.

Compare:

```text
rank 1
rank 2
rank 3
full rank
```

using held-out reconstruction.

Primary aim:

> determine whether probe recombination is governed by a small number of transition-pattern dimensions.

Do not select rank using Mixed/Novel test performance alone.

Preferred selection:

```text
A/B-only cross-validation
```

then evaluate Mixed/Novel out of sample.

---

## 17. Human-bridge export

Create a standardized SRN table that can later be passed directly into the human comparison code.

Required schema:

```text
source = "SRN"
dataset = "SRN"
condition
seed
context
transition_position
RT
A_RT
B_RT
AB_midpoint
AB_difference
q_unclipped
probe_midpoint_deviation
```

Human code should ultimately generate the same columns.

Do not fit humans in this PRD.

---

## 18. Causal hidden-state analyses

Run only after the confirmatory descriptive pipeline is complete.

Use matched transition position \(k\).

For Mixed and Novel:

```text
Probe <- A matched component
Probe <- B matched component
```

Primary outcome:

```text
normalized RT
```

Secondary outcomes:

```text
correct-token confidence
sequence CE
predicted next-token identity
```

---

## 19. Causal success criterion

A strong causal effect requires:

\[
\Delta RT 
eq 0
\]

in the expected direction while preserving:

```text
same predicted token
high next-token accuracy
small CE disruption
```

The intended claim is:

> changing the context-specific component of the configured sequence state changes execution behavior without simply destroying sequence knowledge.

---

## 20. Specificity controls

For every causal intervention include:

```text
random orthogonal direction
wrong-token A/B component
magnitude-matched random vector
```

Compare signed RT effects.

The matched A/B intervention should outperform controls.

---

## 21. Required primary figures

### Figure 1 — confirmatory training dynamics

Panels:

```text
sequence CE
correct-token confidence
normalized RT
```

for High and Low scale.

### Figure 2 — representational splitting

Panels:

```text
High vs Low A/B hidden separation
token-matched A/B trajectories
```

### Figure 3 — state-space bridge

Canonical Lump/Cross/Split overlays for High and Low emergent RT.

### Figure 4 — frozen probe behavior

Transition profiles for:

```text
A
B
Mixed
Novel
```

### Figure 5 — responsibility profiles

Show:

```text
Mixed q_k
Novel q_k
```

across transition positions, with scalar-q baseline.

### Figure 6 — low-rank structured recombination

Only if supported.

Show observed probe deviation and rank-selected prediction for Mixed and Novel.

### Figure 7 — causal intervention

Show:

```text
natural probe RT
probe <- A
probe <- B
random control
```

with sequence preservation.

---

## 22. Required outputs

```text
confirmatory_config.json

confirmatory_training_sequence_level.csv
confirmatory_training_transition_level.csv
confirmatory_seed_summary.csv

confirmatory_hidden_separation.csv
confirmatory_hidden_states.npz

confirmatory_state_space_fits.csv
confirmatory_state_space_seed_classification.csv

confirmatory_probe_transition_level.csv
confirmatory_probe_responsibility.csv
confirmatory_probe_midpoint_deviation.csv

confirmatory_low_rank_results.csv
confirmatory_low_rank_predictions.csv

confirmatory_causal_patching.csv
confirmatory_causal_controls.csv

srn_human_bridge_table.csv
```

---

## 23. Required report

Create:

```text
CONFIRMATORY_SRN_REPORT.md
```

Use exactly this structure.

### 1. Frozen model specification

Report architecture, hidden size, High/Low scale, training curriculum, seed list, execution readout, and context encodings.

### 2. Did all conditions learn the sequence?

Report early/late CE, confidence, and seed-level consistency.

### 3. Did representational scale control A/B splitting?

Report High vs Low hidden-state separation.

### 4. Did representational scale change emergent RT dynamics?

Report learning-related RT, switch costs, and A/B RT differences.

### 5. Which canonical state-space model described each condition?

Report Lump/Cross/Split, BIC, and seed-level classification.

### 6. What happened on frozen Mixed and Novel probes?

Report A/B endpoints, probe RT, midpoint deviations, and q profiles.

### 7. Is probe recombination scalar or transition structured?

Compare global q, q_k, and low-rank model without overclaiming.

### 8. Do representational interventions causally shift RT?

Report causal patching plus sequence-preservation controls.

### 9. Final interpretation

Choose the best-supported statement:

```text
A. representational scale produces context-specific states that causally organize RT
B. representational splitting exists but does not reliably control RT
C. RT differences exist without strong representational correspondence
D. neither representational nor behavioral context splitting replicates
```

Do not choose based on manuscript preference.

---

## 24. Stop / escalation rules

Do not alter the frozen execution readout merely because a downstream hypothesis fails.

If High scale no longer separates A/B, first diagnose reproducibility/configuration mismatch.

If RT behaves sensibly but state-space classification is not Cross/Split, report it. Do not tune toward Cross/Split.

If Mixed/Novel do not produce intermediate behavior, report it. Do not change probe encodings after seeing results.

If causal patching fails, do not redesign the representation until all specificity and sequence-preservation diagnostics are complete.

---

## 25. Definition of done

This PRD is complete when:

1. all 60 primary networks are trained;
2. all frozen checkpoints are saved;
3. High/Low/sequence-only hidden-state diagnostics are complete;
4. High and Low emergent RT have canonical state-space fits;
5. Mixed and Novel frozen probes are evaluated;
6. scalar and transition-specific responsibility profiles are saved;
7. low-rank analysis is completed only under A/B-only rank selection;
8. causal matched-component interventions and controls are completed;
9. `srn_human_bridge_table.csv` exists;
10. `CONFIRMATORY_SRN_REPORT.md` provides one frozen, reproducible account of the full SRN result.

---

## 26. Intended paper-level logic

If supported, the final computational story should be:

\[
oxed{
egin{aligned}
&	ext{High representational scale}\
&\qquad\downarrow\
&	ext{Context-specific sequence states}\
&\qquad\downarrow\
&	ext{Context-specific action readiness}\
&\qquad\downarrow\
&	ext{Context-specific RT dynamics}\
&\qquad\downarrow\
&	ext{Structured recombination under Mixed/Novel context}
\end{aligned}
}
\]

The human analyses can then ask whether the same interpretable organization is present in behavior.
