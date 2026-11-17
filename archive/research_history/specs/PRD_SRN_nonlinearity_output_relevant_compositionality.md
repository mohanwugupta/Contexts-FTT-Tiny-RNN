# PRD: SRN Nonlinearity, Curvature, and Output-Relevant Compositionality
## Formalize Why High-Scale SRNs Split Familiar Contexts but Fail on Mixed

## 0. Purpose

Implement a unified representational analysis of the existing small SRNs that directly tests the mechanistic hypothesis suggested by the current diagnostics:

> Low-scale SRNs preserve a nearly linear shared sequence representation and interpolate smoothly across context cues, but do not split familiar contexts. High-scale SRNs split familiar A/B contexts, but cue interpolation produces curved hidden-state trajectories that leave the familiar A/B line, especially in output-relevant dimensions, causing Mixed performance loss.

This PRD extends the existing diagnostic pipeline and adds the mathematical formalizations needed to distinguish:

1. **Linear shared representation**
2. **Additive factorization**
3. **Structured bilinear/tensor binding**
4. **Output-relevant compositionality**
5. **Nonlinear curvature from tanh saturation**
6. **Recurrent amplification of local curvature**
7. **Unstructured context-specific specialization**
8. **Novel-context input OOD failure**

The central question is no longer simply:

```text
Does Mixed lie between A and B?
```

It is:

> Which parts of the hidden representation are allowed to change nonlinearly with context, and which parts must remain compositionally well behaved for transfer to survive?

---

# 1. Existing evidence to reproduce first

Before adding any new analyses, reproduce the current frozen diagnostics exactly.

Expected qualitative pattern:

## High-Scale + High Context

```text
Mixed A/B-axis q ~ 0.5
Mixed off-axis residual substantial
A->B cue sweep q approximately linear
A->B cue sweep off-axis residual arches upward near lambda=0.5
sequence loss peaks near intermediate lambda
familiar A/B split substantial
context × sequence interaction substantial
patching output-relevant/shared-sequence components rescues loss
```

## Low-Scale + Low Context

```text
Mixed q ~ 0.5
Mixed off-axis residual ~ 0
cue sweep nearly linear
sequence loss flat
familiar context effect negligible
sequence representation almost completely shared
```

Do not proceed until the reproduction tests pass.

---

# 2. Scope

Analyze the existing frozen SRNs only.

Primary conditions:

```text
High-Scale + High Context
Low-Scale + Low Context
```

Add hybrid conditions if checkpoints become available:

```text
High-Scale + Low Context
Low-Scale + High Context
```

Primary analysis target:

```text
Mixed
```

Novel remains secondary because the current Novel encoding is an untrained / zero-weight input direction.

No retraining is allowed in this PRD.

---

# 3. Notation

Let:

```text
k = sequence position
c = context cue vector
lambda = interpolation coefficient between familiar A and B
h_k(c) = hidden state at sequence position k under context c
W_out = hidden-to-output weight matrix
P_out = projector onto row space(W_out)
P_null = I - P_out
```

Define the interpolated cue:

\[
c(\lambda)
=
(1-\lambda)c_A + \lambda c_B,
\qquad
\lambda \in [0,1].
\]

Define the familiar linear hidden interpolation:

\[
h_k^{lin}(\lambda)
=
(1-\lambda)h_{k,A}
+
\lambda h_{k,B}.
\]

Define hidden interpolation residual:

\[
r_k(\lambda)
=
h_k(\lambda)-h_k^{lin}(\lambda).
\]

This residual is the main object of analysis.

---

# 4. Formalization F0 — Shared sequence only

Model:

\[
h_{k,c}
=
\mu + U z_k.
\]

Interpretation:

```text
context does not matter;
all contexts reuse the same sequence trajectory.
```

Expected Low-Scale limit.

Fit using familiar A/B only.

Report:

```text
R²
RMSE
residual energy
```

---

# 5. Formalization F1 — Additive factorization

Model:

\[
h_{k,c}
=
\mu + U z_k + Vc.
\]

Interpretation:

```text
shared sequence geometry
+
context-specific offset
```

Context can separate A and B without changing sequence geometry.

This corresponds approximately to:

\[
h_{k,A}-h_{k,B}
\]

being position-invariant.

Report:

```text
familiar reconstruction R²
context main-effect energy
residual interaction energy
```

---

# 6. Formalization F2 — Structured bilinear / tensor binding

Model:

\[
h_{k,c}
=
\mu
+
Uz_k
+
Vc
+
\sum_{r=1}^{R}
v_r
(a_r^\top z_k)
(b_r^\top c).
\]

Equivalent conceptual form:

\[
h_{k,c}
=
\mu + Uz_k + Vc + W(z_k \otimes c).
\]

Interpretation:

```text
sequence and context interact,
but according to a reusable low-rank algebra.
```

This is structured splitting, not arbitrary memorization.

Evaluate:

```text
R = 0,1,2,3
```

with familiar-only rank selection.

Report:

```text
familiar R²
selected rank
structured interaction energy
unstructured residual energy
```

---

# 7. Formalization F3 — Unconstrained context × sequence specialization

Model:

\[
h_{k,c}
=
\mu + \eta_{k,c}.
\]

where every familiar context × position pair has its own unconstrained vector.

Interpretation:

```text
arbitrary context-specific sequence states
```

Use only as an upper-bound reconstruction model.

Do not use for Mixed prediction.

This formalization defines the "memorized endpoints" limit.

---

# 8. Formalization F4 — Whole-state linear compositionality

Strong compositionality condition:

\[
r_k(\lambda)
=
h_k(\lambda)
-
[(1-\lambda)h_{k,A}+\lambda h_{k,B}]
\approx 0.
\]

Equivalent metric:

\[
E_{whole}(\lambda)
=
\frac{
\sum_k \|r_k(\lambda)\|^2
}{
\sum_k \|h_{k,B}-h_{k,A}\|^2
}.
\]

Interpretation:

```text
the entire hidden trajectory follows the straight line between A and B.
```

Expected to hold approximately for Low-Scale networks.

This is intentionally a strong criterion and may be too restrictive for a human-like solution.

---

# 9. Formalization F5 — Output-relevant compositionality

Primary new hypothesis.

Context may induce nonlinear hidden-state changes as long as those changes do not corrupt the task-relevant computation.

Define:

\[
r_k^{out}(\lambda)
=
P_{out}r_k(\lambda)
\]

and:

\[
r_k^{null}(\lambda)
=
P_{null}r_k(\lambda).
\]

Define normalized energies:

\[
E_{out}(\lambda)
=
\frac{
\sum_k \|r_k^{out}(\lambda)\|^2
}{
\sum_k \|P_{out}(h_{k,B}-h_{k,A})\|^2 + \epsilon
}
\]

and:

\[
E_{null}(\lambda)
=
\frac{
\sum_k \|r_k^{null}(\lambda)\|^2
}{
\sum_k \|P_{null}(h_{k,B}-h_{k,A})\|^2 + \epsilon
}.
\]

Human-like / transfer-friendly condition:

\[
E_{out}(\lambda) \ll E_{null}(\lambda)
\]

for intermediate cues.

Equivalent conceptual statement:

\[
P_{out}h_k(\lambda)
\approx
(1-\lambda)P_{out}h_{k,A}
+
\lambda P_{out}h_{k,B}
\]

even if:

\[
h_k(\lambda)
\]

is nonlinear in the full hidden space.

This is the most important representational formalization in the PRD.

---

# 10. Formalization F6 — Logit-space compositionality

Because behavior is determined by logits:

\[
\ell_k(c)
=
W_{out}h_k(c)+b.
\]

Define:

\[
\ell_k^{lin}(\lambda)
=
(1-\lambda)\ell_{k,A}
+
\lambda\ell_{k,B}.
\]

Define:

\[
r_k^{logit}(\lambda)
=
\ell_k(\lambda)-\ell_k^{lin}(\lambda).
\]

Metric:

\[
E_{logit}(\lambda)
=
\frac{
\sum_k \|r_k^{logit}(\lambda)\|^2
}{
\sum_k \|\ell_{k,B}-\ell_{k,A}\|^2+\epsilon
}.
\]

This is behaviorally closer than hidden-space geometry.

Also compute:

```text
correct-token logit deviation
correct-token probability deviation
cross-entropy deviation
```

Primary question:

> Can full hidden-state curvature be large while logit-space curvature stays small?

If yes, nonlinear context representation can coexist with preserved task computation.

---

# 11. Formalization F7 — Context × sequence variance decomposition

Retain the familiar-state decomposition:

\[
h_{k,c}
=
\mu+s_k+u_c+\epsilon_{k,c}.
\]

Compute familiar-only variance fractions:

```text
P_sequence
P_context
P_interaction
```

Interpretation:

```text
P_sequence = shared sequence structure
P_context = pure context offset
P_interaction = context-specific deformation of sequence geometry
```

Do not call interaction "low" using arbitrary thresholds.

Report it continuously.

Key comparison:

```text
Does P_interaction predict Mixed off-axis residual or Mixed loss across seeds?
```

---

# 12. Formalization F8 — Tanh nonlinearity / saturation mechanism

For each time step:

\[
a_t
=
W_xx_t + W_hh_{t-1}+b
\]

or include the explicit context term if implemented separately:

\[
a_t
=
W_xx_t + W_hh_{t-1}+W_cc+b.
\]

Then:

\[
h_t=\tanh(a_t).
\]

Compute:

## Mean local gain

\[
g_t
=
\frac{1}{H}\sum_j
\left(
1-\tanh^2(a_{t,j})
\right).
\]

## Saturation fraction

For thresholds:

```text
|h| > 0.8
|h| > 0.9
|h| > 0.95
```

compute fraction of hidden units saturated.

## Mean preactivation magnitude

\[
\frac{1}{H}\sum_j|a_{t,j}|.
\]

Primary hypothesis:

```text
High-Scale networks have lower local gain / greater saturation,
especially at intermediate cue values,
and this predicts hidden-state curvature and probe loss.
```

---

# 13. Formalization F9 — Hidden-state curvature across context

For cue step:

\[
\Delta\lambda = 0.05.
\]

Discrete second derivative:

\[
\kappa_k(\lambda)
=
\frac{
h_k(\lambda+\Delta\lambda)
-
2h_k(\lambda)
+
h_k(\lambda-\Delta\lambda)
}{
(\Delta\lambda)^2
}.
\]

Whole-space curvature:

\[
K_{whole}(\lambda)
=
\sqrt{
\sum_k\|\kappa_k(\lambda)\|^2
}.
\]

Output-relevant curvature:

\[
K_{out}(\lambda)
=
\sqrt{
\sum_k\|P_{out}\kappa_k(\lambda)\|^2
}.
\]

Output-null curvature:

\[
K_{null}(\lambda)
=
\sqrt{
\sum_k\|P_{null}\kappa_k(\lambda)\|^2
}.
\]

Logit curvature:

\[
K_{logit}(\lambda)
=
\sqrt{
\sum_k
\|
W_{out}\kappa_k(\lambda)
\|^2
}.
\]

Primary prediction:

```text
High-Scale:
K_whole large
K_out substantial
K_logit substantial
loss peaks where curvature peaks

Low-Scale:
all curvature measures near zero
```

---

# 14. Formalization F10 — Context Jacobian

Estimate how hidden state changes with context interpolation.

Finite-difference first derivative:

\[
J_k^{(\lambda)}
\approx
\frac{
h_k(\lambda+\Delta\lambda)
-
h_k(\lambda-\Delta\lambda)
}{
2\Delta\lambda
}.
\]

Report:

```text
||J||
||P_out J||
||P_null J||
```

Then compute change in Jacobian:

\[
\Delta J_k(\lambda)
=
J_k(\lambda+\Delta)-J_k(\lambda-\Delta).
\]

If the representation were locally affine in context:

```text
J should be approximately constant across lambda.
```

Non-constant J indicates nonlinear cue-to-state mapping.

---

# 15. Formalization F11 — Recurrent amplification

The SRN local hidden-to-hidden Jacobian is:

\[
J_t^{rec}
=
D_tW_h
\]

where:

\[
D_t
=
\operatorname{diag}
\left(
1-\tanh^2(a_t)
\right).
\]

Compute per sequence step:

```text
spectral norm of J_rec
spectral radius of J_rec
largest singular value
```

Also compute cumulative sensitivity:

\[
G_{t:s}
=
J_t^{rec}
J_{t-1}^{rec}
\cdots
J_{s+1}^{rec}.
\]

Metric:

```text
||G_{t:s}||
```

Question:

> Is a small cue-induced deviation at sequence onset amplified by recurrent dynamics?

Primary hypothesis:

```text
Mixed curvature is produced by nonlinear cue transformation
and/or amplified by recurrent dynamics.
```

This analysis distinguishes input nonlinearity from recurrent amplification.

---

# 16. Formalization F12 — Direct nonlinearity decomposition

For each lambda, construct two counterfactual trajectories.

## Counterfactual A — linearized activation

Replace:

\[
\tanh(a)
\]

with first-order linearization around the nearest familiar endpoint or around the midpoint operating point:

\[
\tanh(a_0)
+
D(a_0)(a-a_0).
\]

Run the frozen network forward under this local linearization.

## Counterfactual B — actual tanh

Use original network.

Compare:

```text
off-axis residual
output-relevant residual
sequence loss
logit curvature
```

If linearization removes the Mixed valley:

```text
activation nonlinearity is mechanistically sufficient to explain the failure.
```

Do not retrain.

---

# 17. Formalization F13 — Endpoint-specialization vs structured path

Define three competing geometric descriptions of A/B learning.

## G0 — Lumped

\[
h_{k,A}\approx h_{k,B}.
\]

## G1 — Split but structured

\[
h_{k,c}
\]

varies with context, but:

\[
P_{out}h_k(c)
\]

follows a smooth low-curvature path.

## G2 — Split and endpoint-specialized

A and B are individually functional, but intermediate context produces high:

```text
off-axis deviation
output-relevant curvature
logit curvature
loss
```

This is the candidate High-Scale regime.

Create a seed-level continuous "endpoint specialization index":

\[
ESI
=
\overline{E_{out}(\lambda)}
\times
\overline{K_{out}(\lambda)}
\]

over:

```text
lambda in [0.2, 0.8]
```

Also report each component separately.

Do not rely on the composite alone.

---

# 18. Sequence decoder correction

The current decoder comparison used different familiar and probe training schemes.

Replace with one apples-to-apples decoder protocol.

Primary:

```text
leave-one-seed-out
```

For each held-out seed:

1. train a linear sequence-position decoder on A+B hidden states from the other seeds;
2. test on held-out seed:
   - A
   - B
   - Mixed
   - Novel
   - NoContextProbe
3. repeat for every seed.

Use identical preprocessing and regularization across all test contexts.

Report:

```text
accuracy
cross entropy
margin
```

Do not compare decoder scores produced by different training schemes.

---

# 19. Mixed-specific analysis

Mixed is the primary scientific probe.

For each seed compute:

```text
Mixed q
Mixed whole-space residual
Mixed output-relevant residual
Mixed output-null residual
Mixed logit residual
Mixed sequence loss
Mixed sequence-decoder accuracy
Mixed local gain
Mixed saturation fraction
Mixed whole curvature
Mixed output curvature
Mixed logit curvature
```

Then test seed-level associations:

```text
saturation -> whole curvature
saturation -> output curvature
whole curvature -> output curvature
output curvature -> Mixed loss
P_interaction -> Mixed output curvature
P_interaction -> Mixed loss
```

Use:

```text
Pearson r
Spearman rho
simple OLS slope with bootstrap CI
```

Treat these as mechanistic associations, not causal proof.

---

# 20. Mediation-style descriptive chain

Do not perform strong causal mediation claims.

But quantify the proposed chain descriptively:

\[
\text{weight scale / saturation}
\rightarrow
\text{hidden curvature}
\rightarrow
\text{output-relevant curvature}
\rightarrow
\text{Mixed loss}.
\]

Fit simple nested regressions across seeds:

```text
Model 1:
Mixed loss ~ saturation

Model 2:
Mixed loss ~ saturation + whole curvature

Model 3:
Mixed loss ~ saturation + whole curvature + output curvature
```

Report coefficient attenuation descriptively.

No causal language.

---

# 21. Causal interventions

Retain and extend patching.

## C1 — output-null patch

Patch only:

```text
P_null h
```

Expected:

```text
minimal behavioral rescue
```

## C2 — output-relevant patch

Patch:

```text
P_out h
```

Expected:

```text
large rescue if task-relevant corruption drives failure
```

## C3 — linear-path patch

Replace actual Mixed hidden state with:

\[
h_k^{lin}(0.5)
=
\frac{h_{k,A}+h_{k,B}}{2}.
\]

Measure output loss.

## C4 — output-only linear-path patch

Construct:

\[
h_k^{cf}
=
P_{out}h_k^{lin}(0.5)
+
P_{null}h_{k,Mixed}.
\]

This preserves actual Mixed null-space context geometry while forcing only the task-relevant component onto the compositional path.

This is a critical intervention.

If it rescues performance:

> full hidden-state linearity is unnecessary; output-relevant compositionality is sufficient.

## C5 — null-only linear-path patch

Construct:

\[
h_k^{cf}
=
P_{out}h_{k,Mixed}
+
P_{null}h_k^{lin}(0.5).
\]

Expected:

```text
little rescue
```

if the hypothesis is correct.

---

# 22. Formal comparison of representational models

Fit familiar-only models:

```text
F0 shared sequence
F1 additive context offset
F2 low-rank bilinear interaction
F3 unconstrained context × sequence lookup
```

For each seed report:

```text
familiar R²
held-out familiar R² if feasible
parameter count
AIC/BIC only if likelihood assumptions are explicitly valid
cross-validated reconstruction error preferred
```

Then evaluate how well F0-F2 predict:

```text
Mixed hidden states
Mixed output-relevant hidden states
Mixed logits
```

without fitting Mixed.

Key comparison:

> A formalization can be useful even if it does not predict the entire hidden vector, provided it predicts the output-relevant component.

Therefore report both:

```text
whole-state prediction R²
output-relevant prediction R²
logit prediction R²
```

---

# 23. Novel-context branch

Treat Novel separately from Mixed.

Current known issue:

```text
Novel familiar-span fraction = 0
induced Novel norm = 0
Novel input direction untrained
```

Reproduce this audit.

If true:

```text
NOVEL_INPUT_OOD = TRUE
```

Then Novel results must not be used to adjudicate the compositional-representation hypotheses.

Allowed conclusion:

```text
current Novel encoding is not an interpretable test of novel-context generalization
```

Do not attempt to "fix" Novel in this PRD.

---

# 24. Required figures

Create:

```text
figure_01_decoder_apples_to_apples.png
figure_02_hidden_linear_residual_vs_lambda.png
figure_03_output_relevant_vs_null_residual.png
figure_04_logit_residual_vs_lambda.png
figure_05_tanh_gain_vs_lambda.png
figure_06_saturation_vs_lambda.png
figure_07_hidden_curvature_vs_lambda.png
figure_08_output_curvature_vs_lambda.png
figure_09_logit_curvature_vs_lambda.png
figure_10_recurrent_jacobian_spectrum.png
figure_11_curvature_vs_loss_seed_scatter.png
figure_12_interaction_vs_curvature_seed_scatter.png
figure_13_formal_model_reconstruction.png
figure_14_formal_model_mixed_prediction.png
figure_15_linear_path_patching.png
figure_16_linearized_activation_counterfactual.png
figure_17_summary_tradeoff_map.png
```

---

# 25. Summary tradeoff map

Create a 2D seed-level figure:

```text
x-axis = familiar A/B separation
y-axis = Mixed sequence loss
```

Encode:

```text
point size = output-relevant curvature
point marker or facet = condition
```

The target human-like region is conceptually:

```text
high familiar separation
low Mixed loss
low output-relevant curvature
```

Do not plot human values unless directly comparable.

---

# 26. Required CSV outputs

```text
decoder_loso_results.csv
cue_sweep_hidden_residuals.csv
cue_sweep_output_residuals.csv
cue_sweep_logit_residuals.csv
cue_sweep_tanh_gain.csv
cue_sweep_saturation.csv
cue_sweep_curvature.csv
recurrent_jacobian_metrics.csv
formal_model_familiar_fit.csv
formal_model_mixed_prediction.csv
seed_level_mechanism_table.csv
causal_linear_path_patching.csv
linearized_activation_counterfactual.csv
novel_input_audit.csv
```

---

# 27. Master seed-level table

Generate:

```text
srn_mechanism_master_table.csv
```

Required columns:

```text
condition
seed
AB_hidden_separation
AB_output_relevant_separation
P_sequence
P_context
P_interaction
Mixed_q
Mixed_whole_residual
Mixed_output_residual
Mixed_null_residual
Mixed_logit_residual
Mixed_loss
Mixed_decoder_accuracy
mean_tanh_gain
saturation_08
saturation_09
saturation_095
K_whole_mid
K_out_mid
K_null_mid
K_logit_mid
Jrec_spectral_radius
Jrec_spectral_norm
F0_R2
F1_R2
F2_R2
F2_selected_rank
F2_Mixed_whole_R2
F2_Mixed_output_R2
F2_Mixed_logit_R2
linear_path_patch_delta_loss
output_linear_path_patch_delta_loss
null_linear_path_patch_delta_loss
NOVEL_INPUT_OOD
```

---

# 28. Decision rules

Do not use one automatic classification without showing the continuous evidence.

The report should evaluate the following hypotheses.

## H1 — Pure additive factorization

Supported if:

```text
F1 reconstructs familiar A/B well
P_interaction small
Mixed whole-space residual small
output-relevant residual small
```

## H2 — Structured bilinear splitting

Supported if:

```text
F2 strongly improves over F1
F2 requires low rank
F2 predicts Mixed better than F1
F2 predicts output-relevant Mixed geometry well
```

## H3 — Nonlinear endpoint specialization

Supported if:

```text
High-Scale A/B endpoints are functional
Mixed q ~ 0.5
whole/output curvature peaks at intermediate lambda
sequence loss peaks at intermediate lambda
curvature predicts loss across seeds
```

## H4 — Tanh saturation mechanism

Supported if:

```text
High-Scale has lower gain / more saturation
saturation predicts curvature
local linearization substantially reduces curvature and/or Mixed loss
```

## H5 — Recurrent amplification mechanism

Supported if:

```text
initial cue-induced deviations are modest
but recurrent Jacobian products amplify them across sequence steps
and amplification predicts later curvature/loss
```

## H6 — Output-relevant compositionality is sufficient

Supported if:

```text
forcing only P_out h_Mixed onto the A/B linear path rescues behavior
while forcing only P_null does not
```

This is the most important causal sufficiency test.

## H7 — Full hidden-state compositionality is necessary

Supported only if:

```text
output-only linear path patch is insufficient
but full hidden-state linear path patch rescues performance
```

This is a stronger claim and should not be assumed.

## H8 — Novel failure is primarily input OOD

Supported if:

```text
Novel input direction untrained
Novel induced input-to-hidden effect zero or outside familiar span
```

If supported, Novel must be analyzed separately from Mixed.

---

# 29. Preferred interpretation hierarchy

Use this hierarchy in the report.

### Level 1 — behavioral geometry

```text
Does loss peak between A and B?
```

### Level 2 — representational geometry

```text
Does hidden state curve away from the familiar interpolation path?
```

### Level 3 — computational relevance

```text
Is that curvature in output-relevant dimensions?
```

### Level 4 — local mechanism

```text
Is curvature associated with tanh saturation / low local gain?
```

### Level 5 — dynamical mechanism

```text
Does recurrence amplify the deviation?
```

### Level 6 — causal sufficiency

```text
Does restoring output-relevant compositionality rescue behavior?
```

Do not jump from Level 1 directly to Level 4.

---

# 30. TDD / RED -> GREEN tests

## T1 — frozen checkpoint integrity

Checkpoint SHA-256 must remain unchanged.

## T2 — cue interpolation endpoints

```text
lambda=0 -> exact A cue
lambda=1 -> exact B cue
```

## T3 — hidden interpolation residual endpoints

```text
r(lambda=0)=0
r(lambda=1)=0
```

within tolerance.

## T4 — projector correctness

Verify:

```text
P_out^2 = P_out
P_null^2 = P_null
P_out @ P_null = 0
P_out + P_null = I
```

within numerical tolerance.

## T5 — logit equivalence

Verify:

```text
W_out @ P_null @ h ~= 0
```

within numerical tolerance.

## T6 — curvature sanity

A synthetic linear path must produce:

```text
K_whole = 0
K_out = 0
K_logit = 0
```

within tolerance.

## T7 — tanh derivative

Numerically verify:

```text
d tanh(a) / da = 1 - tanh(a)^2
```

against finite differences.

## T8 — recurrent Jacobian

Numerically verify analytic:

```text
J_rec = D @ W_h
```

against autograd / finite difference for random test states.

## T9 — decoder firewall

LOSO sequence decoder trains on familiar A/B only.

## T10 — decoder comparability

The exact same fitted decoder for a held-out seed must be used to score A, B, Mixed, Novel, and NoContextProbe.

## T11 — formal model firewall

F0-F2 parameter/rank selection cannot access Mixed/Novel states.

## T12 — patch reconstruction

For output-only linear-path patch:

```text
P_out h_cf == P_out h_linear
P_null h_cf == P_null h_actual
```

within tolerance.

## T13 — null-only patch reconstruction

Verify inverse constraints.

## T14 — no human tuning

No human Mixed/Novel behavior may be used to choose thresholds, ranks, checkpoints, or regularization.

---

# 31. Implementation order

Implement in this order:

```text
1. reproduce current diagnostics
2. fix LOSO sequence decoder
3. compute whole/output/null/logit interpolation residuals
4. compute tanh gain and saturation
5. compute hidden/output/logit curvature
6. compute context Jacobian
7. compute recurrent Jacobian and amplification
8. fit F0/F1/F2/F3 familiar representational models
9. evaluate Mixed prediction
10. run linear-path causal patches
11. run local-linearization counterfactual
12. run seed-level association analyses
13. reproduce Novel input audit
14. generate integrated report
```

---

# 32. Main report

Generate:

```text
SRN_NONLINEARITY_AND_COMPOSITIONALITY_REPORT.md
```

Required sections:

```text
1. Executive summary
2. Existing diagnostic reproduction
3. Corrected sequence decoder results
4. Whole-state cue interpolation
5. Output-relevant vs output-null interpolation
6. Logit-space interpolation
7. Familiar factorization and interaction structure
8. Tanh gain and saturation
9. Hidden-state curvature
10. Recurrent amplification
11. Formal model comparison: F0-F3
12. Mixed prediction from formal models
13. Causal linear-path patching
14. Local-linearization counterfactual
15. Novel input audit
16. Integrated mechanistic interpretation
17. Which formalization is currently best supported?
18. Implications for the next training experiment
```

---

# 33. Required conclusion format

The report must answer these questions explicitly:

```text
Q1. Does the High-Scale SRN know that Mixed is between A and B?
Q2. Does Mixed leave the straight A/B hidden-state path?
Q3. Is the deviation primarily output-relevant or output-null?
Q4. Does output-relevant deviation explain the loss increase?
Q5. Is that deviation associated with tanh saturation?
Q6. Does recurrence amplify it?
Q7. Is familiar A/B splitting better described as additive, bilinear, or unstructured?
Q8. Does restoring only output-relevant compositionality rescue Mixed?
Q9. Is full hidden-state linearity actually necessary?
Q10. Should Novel be treated as a separate input-OOD problem?
```

Do not collapse these into a single automatic label.

---

# 34. Decision for the next modeling phase

Only after this analysis should the next training experiment be chosen.

## If F2 / output-relevant compositionality is supported

Next target:

```text
train a vanilla SRN to learn strong context separation
while preserving low-curvature output-relevant sequence geometry
```

Possible next ecology:

```text
stable sequence × context factorization
shared task computation
context-specific surface realization
```

## If tanh saturation is the dominant mechanism

Next target:

```text
test whether training experience can produce splitting
without pushing task-relevant units into strongly nonlinear regimes
```

Do not simply reduce weight scale, because that returns to the Low-Scale lumped solution.

## If recurrent amplification dominates

Next target:

```text
train representations where context-specific deviations are dynamically stable
and confined to task-irrelevant dimensions
```

## If none of the formalizations explain transfer failure

Stop the factorization/compositionality program and reconsider the objective or architecture.

---

# 35. Success criterion

This PRD succeeds if it turns the current qualitative tradeoff:

```text
Low-Scale = transferable but unsplit
High-Scale = split but nontransferable
```

into a precise mechanistic statement of the form:

> Familiar context splitting induces nonlinear curvature in hidden state as the cue moves between A and B. The critical failure is specifically curvature in the output-relevant subspace, which is associated with tanh saturation and/or recurrent amplification. A human-like solution would therefore preserve contextual separation while keeping the task-relevant sequence computation compositionally smooth.

The analysis must also establish whether this statement is actually supported, rather than assuming it in advance.
