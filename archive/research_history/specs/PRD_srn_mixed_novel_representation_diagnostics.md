# PRD: SRN Mixed / Novel Representational Diagnostics
## Diagnose Why Familiar Context Splitting Fails to Generalize Zero-Shot

## 0. Objective

Build a focused white-box analysis pipeline for the existing small SRNs to determine **what actually happens internally when a familiar-splitting network sees Mixed or Novel context**.

The purpose is diagnostic, not to optimize probe performance.

The central scientific question is:

> Why can the SRN learn context-specific familiar A/B representations, yet fail to show human-like zero-shot transfer to Mixed or Novel?

The analysis should distinguish among at least four candidate failure modes:

1. **Input-space failure** — Mixed/Novel enters through a poorly trained or genuinely OOD input direction.
2. **Representational entanglement** — familiar A/B learning reorganizes the sequence representation itself.
3. **Readout failure** — Mixed/Novel preserves sequence information internally but corrupts output-relevant dimensions.
4. **History / attractor failure** — probe behavior is dominated by the preceding recurrent state rather than the current cue.

All analyses must use **frozen trained SRNs**. No Mixed/Novel training and no probe tuning.

---

## 1. Scope

Start with the existing small SRNs already used in the project.

Primary conditions:

```text
High-Scale + High Context
Low-Scale + Low Context
```

Recommended controls:

```text
Low-Scale + High Context
High-Scale + Low Context
```

Use existing checkpoints. Do not change architecture, hidden size, optimizer, training data, context coding, weight scale, or context scale during this diagnostic phase.

Primary analyses should be at the **whole-sequence / trajectory level**. Position-wise hidden states may be used internally, but do not turn this into seven separate behavioral claims.

---

## 2. Frozen evaluation data

For every trained network seed, evaluate:

```text
A familiar
B familiar
Mixed
Novel
```

If available:

```text
NoContextProbe
```

Save for every sequence step:

```text
input
hidden state
output logits
output probabilities
correct-token probability
cross-entropy loss
predicted token
sequence accuracy
```

Required output directory:

```text
outputs/srn_mixed_novel_diagnostics/
```

---

## 3. Analysis A — Familiar-trained sequence decoder

### Question

Does Mixed/Novel preserve the familiar sequence representation?

### Method

Train a linear decoder on **familiar A/B hidden states only** to predict:

```text
sequence position
```

Preferred target:

```text
position 1...8
```

because repeated token identities can make next-token decoding less diagnostic of latent sequence state.

Use:

```text
multinomial logistic regression
train = familiar A/B only
test = held-out familiar A/B + Mixed + Novel
```

Do not fit the decoder on probe hidden states.

### Outputs

Per seed and condition:

```text
decoder accuracy
decoder cross entropy
decoder confidence / margin
```

### Interpretation

Good probe decoding:

```text
sequence representation survives probe context
```

Poor probe decoding:

```text
probe context corrupts the learned sequence state
```

---

## 4. Analysis B — Familiar A/B-axis projection and off-axis residual

### Question

Does Mixed/Novel lie between familiar trajectories or leave the familiar manifold?

For each sequence position k:

```text
d_k = h_B,k - h_A,k
```

For probe P:

```text
q_k =
((h_P,k - h_A,k)^T d_k)
/
(||d_k||^2)
```

Projected point:

```text
h_proj,k = h_A,k + q_k d_k
```

Off-axis residual:

```text
r_k = ||h_P,k - h_proj,k||
```

### Sequence-level summaries

Per network and probe:

```text
mean q
median q
SD q across positions
mean off-axis residual
normalized off-axis residual
fraction positions with 0 <= q <= 1
```

Recommended normalization:

```text
r_norm =
mean_k r_k
/
mean_k ||h_B,k - h_A,k||
```

### Interpretation

```text
0 < q < 1 + small residual -> interpolation
q near 0 or 1 + small residual -> endpoint-like
large residual -> qualitatively new hidden region
q outside [0,1] -> extrapolation
```

---

## 5. Analysis C — Context × sequence factorization

### Question

Does the SRN represent shared sequence structure plus context, or two context-specific sequence solutions?

Decompose familiar hidden activity as:

```text
h_c,k = mu + s_k + u_c + epsilon_c,k
```

where:

```text
s_k = sequence-position component
u_c = context component
epsilon_c,k = context × sequence interaction
```

### Implementation

Use familiar A/B hidden states only.

For each seed:

1. compute grand mean;
2. compute sequence-position means collapsed over context;
3. compute context means collapsed over position;
4. compute interaction residual;
5. compute squared Frobenius norm for each component.

Report:

```text
P_sequence
P_context
P_interaction
```

Primary entanglement measure:

```text
interaction_fraction =
SS_context_x_sequence / SS_total
```

Optional descriptive metric:

```text
factorization_index = 1 - interaction_fraction
```

Always report the raw variance fractions as well.

### Interpretation

Transfer-friendly representation:

```text
large sequence component
nonzero context component
small interaction
```

Entangled representation:

```text
large context × sequence interaction
```

---

## 6. Analysis D — Output-relevant vs output-null context separation

### Question

Does familiar context splitting occur in dimensions that actually change next-token prediction?

For output layer:

```text
z = W_out h + b
```

define:

```text
row space(W_out) = output-relevant hidden subspace
null space(W_out) = output-null hidden subspace
```

For each familiar position:

```text
delta_k = h_B,k - h_A,k
```

Decompose:

```text
delta_relevant,k = P_row delta_k
delta_null,k = P_null delta_k
```

Compute:

```text
relevant_fraction =
sum_k ||delta_relevant,k||^2
/
sum_k ||delta_k||^2
```

and:

```text
null_fraction = 1 - relevant_fraction
```

Also decompose Mixed/Novel off-axis deviations into output-relevant vs output-null components.

### Interpretation

Transfer-friendly split:

```text
A/B separation mostly output-null
```

Potentially harmful entanglement:

```text
A/B separation strongly output-relevant
```

Probe failure caused by sequence corruption:

```text
probe off-axis deviation strongly output-relevant
```

---

## 7. Analysis E — Familiar-trained context decoder

### Question

How does the network internally classify Mixed and Novel as context?

Train a linear binary decoder:

```text
A vs B
```

using familiar hidden states only.

Apply it to Mixed and Novel without refitting.

Use continuous decision score, not only hard class.

Standardize so familiar contexts are approximately:

```text
A ~= -1
B ~= +1
```

Per network / probe report:

```text
mean context score across sequence
SD context score
```

### Joint diagnostic

| Sequence decoder | Context decoder | Interpretation |
|---|---|---|
| good | intermediate | sequence preserved; context interpolates |
| good | endpoint-like | sequence preserved; context selects familiar endpoint |
| good | extreme/OOD | sequence survives despite unusual context state |
| poor | any | context corrupts sequence representation |

---

## 8. Analysis F — Continuous A→B cue sweep

### Question

What function did the SRN learn between familiar context endpoints?

Construct:

```text
c(lambda) = (1-lambda)c_A + lambda c_B
```

for:

```text
lambda = 0.00, 0.05, ..., 1.00
```

Freeze the network.

For every lambda compute:

```text
sequence loss
correct-token confidence
sequence accuracy
sequence-decoder accuracy
context-decoder score
mean A/B-axis q
normalized off-axis residual
```

### Interpretation

Smooth transferable geometry:

```text
performance and geometry vary smoothly between A and B
```

Endpoint-specialized geometry:

```text
good at lambda=0 and 1
large loss / off-axis excursion near intermediate lambda
```

This analysis uses no human probe data and directly measures the learned function between familiar contexts.

---

## 9. Analysis G — Novel input-space audit

### Question

Is Novel interpretable through trained input dimensions?

Inspect context vectors:

```text
c_A
c_B
c_Novel
```

### G1. Familiar-span projection

Project Novel into span(A,B).

Report:

```text
fraction of Novel input norm inside familiar span
orthogonal input residual
```

### G2. Input-to-hidden perturbation

Compute:

```text
v_A = W_x c_A
v_B = W_x c_B
v_N = W_x c_Novel
```

Report:

```text
||v_N||
cos(v_N, v_A)
cos(v_N, v_B)
distance of v_N to span(v_A, v_B)
```

### G3. Training-exposure audit

If Novel uses dedicated input dimensions, determine:

```text
was the dimension active during training?
did its corresponding input weight receive nonzero gradient?
```

If not:

```text
UNTRAINED_NOVEL_INPUT_DIRECTION = TRUE
```

### Interpretation

If Novel enters through an untrained direction, do not interpret its failure as evidence about contextual recombination.

---

## 10. Analysis H — History / recurrent-state dependence

### Question

Does probe behavior depend primarily on current context cue or inherited recurrent state?

Evaluate Mixed and Novel under standardized histories:

```text
actual preceding A history
actual preceding B history
reset hidden state
matched familiar A terminal state
matched familiar B terminal state
```

If possible:

```text
swapped history
```

Keep current probe input identical across history conditions.

Measure:

```text
loss
correct-token confidence
sequence-decoder accuracy
context-decoder score
A/B-axis q
off-axis residual
```

### Interpretation

Strong dependence on prior history:

```text
recurrent attractor / hysteresis explanation
```

Weak dependence:

```text
current cue primarily determines probe representation
```

---

## 11. Analysis I — Causal patching

Run only after descriptive analyses A–H.

### I1. Shared-sequence patch

Estimate familiar shared sequence component `s_k`.

Patch it into Mixed/Novel while preserving probe context component.

Measure whether:

```text
loss falls
correct-token confidence recovers
sequence decoding improves
```

### I2. Context-component patch

Patch familiar context component while preserving probe sequence state.

### I3. Output-relevant patch

Patch only output-relevant component.

### I4. Output-null patch

Patch only output-null component.

### Interpretation

If shared-sequence patch rescues behavior:

```text
probe failure reflects corruption of reusable sequence representation
```

If context patch rescues behavior:

```text
probe failure reflects contextual routing / state selection
```

---

## 12. Primary diagnostic battery

The main report should foreground exactly five analyses:

1. **Sequence decoder generalization**
2. **A/B-axis projection + off-axis residual**
3. **Context × sequence factorization**
4. **Output-relevant vs output-null decomposition**
5. **Continuous A→B cue sweep**

Analyses G–I are follow-ups.

---

## 13. Decision logic

### Case A — Sequence survives and geometry interpolates

Pattern:

```text
probe sequence decoding good
q between endpoints
small off-axis residual
low interaction fraction
context separation mostly output-null
smooth cue sweep
```

Conclusion:

```text
current SRN already has approximately factorized transferable structure
```

Next experiment:

```text
investigate downstream readout / execution mechanism
```

Do not redesign pretraining yet.

### Case B — Sequence representation collapses

Pattern:

```text
poor probe sequence decoding
large off-axis residual
high context × sequence interaction
probe deviation output-relevant
midpoint cue sweep shows performance valley
```

Conclusion:

```text
network learned entangled context-specific sequence solutions
```

Next experiment:

```text
factorization pretraining ecology
```

Target representation:

```text
h_c,k ~= s_k + u_c
```

### Case C — Mixed works, Novel fails input audit

Conclusion:

```text
Mixed and Novel are separate computational problems
```

Next experiment:

```text
pretrain / redesign contextual input representation for Novel
```

### Case D — Strong history dependence

Conclusion:

```text
probe behavior reflects recurrent hysteresis / attractor continuation
```

Next experiment:

```text
analyze state reset, context-update timescale, or recurrent gating
```

### Case E — Context separation mostly output-null but probes still fail

Conclusion:

```text
context splitting itself is not corrupting sequence computation
```

Next experiment:

```text
focus on probe-specific output-relevant deviation / readout
```

---

## 14. Required figures

```text
figure_01_sequence_decoder_generalization.png
figure_02_probe_axis_projection.png
figure_03_factorization_variance.png
figure_04_output_relevant_null_decomposition.png
figure_05_context_decoder_probe_scores.png
figure_06_context_interpolation_sweep_loss.png
figure_07_context_interpolation_sweep_geometry.png
figure_08_novel_input_audit.png
figure_09_history_dependence.png
figure_10_causal_patching.png
```

---

## 15. Required tables

```text
sequence_decoder_results.csv
probe_axis_geometry.csv
factorization_variance.csv
output_subspace_decomposition.csv
context_decoder_scores.csv
context_interpolation_sweep.csv
novel_input_audit.csv
history_dependence.csv
causal_patching.csv
```

---

## 16. Main report

Generate:

```text
SRN_MIXED_NOVEL_REPRESENTATION_DIAGNOSTICS.md
```

Required sections:

```text
1. Executive summary
2. Networks analyzed
3. Sequence information under probes
4. Probe geometry relative to A/B
5. Factorization vs entanglement
6. Output-relevant vs output-null separation
7. Continuous cue interpolation
8. Novel input audit
9. History dependence
10. Causal tests
11. Diagnostic classification
12. Recommended next experiment
```

End with one primary classification:

```text
A: already factorized; downstream failure
B: entangled context-specific sequence representations
C: Novel-specific input/OOD failure
D: recurrent history / attractor failure
E: mixed or inconclusive
```

Secondary flags are allowed.

---

## 17. TDD / RED → GREEN

### T1 — Probe freeze

Assert no gradient updates during Mixed/Novel evaluation.

### T2 — Decoder firewall

Sequence/context decoders train only on familiar A/B hidden states.

### T3 — Axis endpoints

For A:

```text
q = 0
off-axis residual = 0
```

For B:

```text
q = 1
off-axis residual = 0
```

within numerical tolerance.

### T4 — Factorization reconstruction

Verify:

```text
mu + s_k + u_c + epsilon_c,k
```

reconstructs familiar hidden states.

### T5 — Subspace decomposition

Verify:

```text
delta = delta_relevant + delta_null
```

and orthogonality within tolerance.

### T6 — Cue interpolation endpoints

At lambda=0 interpolated cue equals A exactly.

At lambda=1 interpolated cue equals B exactly.

### T7 — Novel audit

Automatically flag any Novel-only untrained input dimension.

### T8 — History manipulation

Probe input must be identical across history conditions.

### T9 — Seed-level metrics

Never compute representational metrics from hidden states averaged across seeds.

### T10 — Human firewall

Human Mixed/Novel outcomes may not select:

```text
decoder type
subspace rank
normalization
checkpoint
analysis hyperparameters
```

---

## 18. Scientific firewall

Do not:

```text
train on Mixed
train on Novel
select networks by human probe similarity
introduce new retrieval modules
change hidden size
run architecture search
```

during this phase.

This is a diagnosis of the existing SRNs.

---

## 19. Minimum implementation order

Implement in this order:

```text
1. hidden-state extraction
2. sequence decoder
3. A/B-axis projection
4. factorization decomposition
5. output-relevant/null decomposition
6. context decoder
7. continuous cue sweep
8. Novel input audit
9. history manipulation
10. causal patching
```

Do not proceed to causal patching until analyses 1–8 pass their tests.

---

## 20. Success criterion

The pipeline succeeds if it answers:

> When a familiar-splitting SRN sees Mixed or Novel, does it preserve reusable sequence structure, interpolate between familiar contexts, leave the familiar manifold, corrupt output-relevant sequence dimensions, or simply inherit the previous recurrent state?

The result should identify the **specific missing computational property** that the next SRN experiment needs to teach.

That property—not similarity to the human probe itself—should determine the next training ecology.
