# PRD: Learn Strong Context Splitting While Preserving Smooth Task-Relevant Geometry

## 0. Objective

Implement a two-phase experiment based on the current SRN diagnostics.

Current evidence:
- Low-scale SRNs interpolate smoothly across A→B and have low Mixed loss, but barely split A/B.
- High-scale SRNs strongly split A/B, locate Mixed near the midpoint, but show large hidden/output/logit curvature and high Mixed loss.
- Replacing only the output-relevant component of Mixed with the linear A/B midpoint rescues behavior as much as replacing the whole hidden state.
- Null-space replacement does essentially nothing.

Working hypothesis:

> A human-like solution requires strong familiar context separation while preserving smooth/compositional geometry in the task-relevant output subspace. Full hidden-state linearity is not necessary.

This PRD has two gated phases:

1. **Phase 1: Sufficiency** — explicitly train the desired geometry.
2. **Phase 2: Emergence** — test whether structured prior experience makes that geometry arise in a vanilla SRN without the explicit regularizer.

Do not begin Phase 2 unless Phase 1 passes.

---

## 1. Core formalization

Let:

- `h_k(c)` = hidden state at sequence position `k` under context cue `c`
- `W_out` = hidden-to-output weight matrix
- `P_out` = projector onto row space of `W_out`
- `P_null = I - P_out`

For interpolated context:

\[
c(\lambda)=(1-\lambda)c_A+\lambda c_B
\]

define the endpoint-linear hidden prediction:

\[
h_k^{lin}(\lambda)
=(1-\lambda)h_{k,A}+\lambda h_{k,B}.
\]

Define:

\[
r_k(\lambda)=h_k(\lambda)-h_k^{lin}(\lambda).
\]

Do **not** require:

\[
r_k(\lambda)pprox0.
\]

Primary target:

\[
oxed{
P_{out}r_k(\lambda)pprox0
}
\]

while allowing:

\[
P_{null}r_k(\lambda)
eq0.
\]

Equivalent task-space target:

\[
P_{out}h_k(\lambda)
pprox
(1-\lambda)P_{out}h_{k,A}
+\lambda P_{out}h_{k,B}.
\]

This is **output-relevant compositionality**.

---

# PHASE 1 — SUFFICIENCY

## 2. Goal

Test whether explicitly encouraging low output-relevant curvature can produce:

\[
oxed{
	ext{strong A/B separation}
+
	ext{low Mixed loss}
}
\]

without collapsing back into the Low-Scale lumped solution.

Use the same small tanh Elman SRN architecture and familiar A/B training schedule as the existing confirmatory model.

Do not add:
- attention
- retrieval modules
- gates
- tensor-product layers
- larger hidden size

Only change the objective.

---

## 3. Experimental conditions

Train at least:

- `C0`: High-Scale + High Context baseline
- `C1`: High-Scale + High Context + **output/logit curvature regularizer**
- `C2`: High-Scale + High Context + **whole-state curvature regularizer**
- `C3`: High-Scale + High Context + **null-space curvature regularizer**
- `C4`: Low-Scale + Low Context baseline

Use 20 seeds per condition if feasible; minimum 10.

Purpose:
- C0 = reproduce failure
- C1 = primary sufficiency test
- C2 = stronger/full-linearity control
- C3 = negative mechanistic control
- C4 = known transferable-but-unsplit control

---

## 4. Base task loss

Keep original next-token prediction loss:

\[
\mathcal L_{task}=\mathrm{CE}(\hat y_t,y_t).
\]

No Mixed, Novel, or NoContextProbe trials may enter:
- task training
- checkpoint selection
- hyperparameter selection
- early stopping

---

## 5. Output/logit curvature regularizer

Use cue points:

\[
\lambda-\Delta,\lambda,\lambda+\Delta
\]

with default:

`Δ = 0.05` or `0.10`.

For each sequence position:

\[
\kappa_k(\lambda)
=
h_k(\lambda+\Delta)
-2h_k(\lambda)
+h_k(\lambda-\Delta).
\]

Primary regularizer:

\[
oxed{
\mathcal L_{outcurv}
=
\sum_{k,\lambda}
\|W_{out}\kappa_k(\lambda)\|_2^2
}
\]

This directly penalizes logit-space curvature.

Also compute, but do not optimize by default:

\[
\sum_{k,\lambda}
\|P_{out}\kappa_k(\lambda)\|_2^2.
\]

---

## 6. Preserve A/B splitting

Prevent the regularizer from solving the task by collapsing A and B together.

Define familiar separation:

\[
d_{AB}
=
rac{1}{K}
\sum_k
\|h_{k,A}-h_{k,B}\|_2.
\]

Set:

\[
d_{target}
=
\operatorname{median}
(d_{AB})
\]

from the existing successful High-Scale baseline seeds.

Penalty:

\[
\mathcal L_{sep}
=
\max(0,d_{target}-d_{AB})^2.
\]

Log but do not optimize by default:

\[
d_{AB}^{out}
=
rac{1}{K}
\sum_k
\|P_{out}(h_{k,A}-h_{k,B})\|.
\]

---

## 7. Total losses

Primary C1:

\[
oxed{
\mathcal L
=
\mathcal L_{task}
+
\lambda_{curv}\mathcal L_{outcurv}
+
\lambda_{sep}\mathcal L_{sep}
}
\]

Whole-state control C2:

\[
\mathcal L
=
\mathcal L_{task}
+
\lambda_{whole}
\sum_{k,\lambda}
\|\kappa_k(\lambda)\|^2
+
\lambda_{sep}\mathcal L_{sep}.
\]

Null-space control C3:

\[
\mathcal L
=
\mathcal L_{task}
+
\lambda_{null}
\sum_{k,\lambda}
\|P_{null}\kappa_k(\lambda)\|^2
+
\lambda_{sep}\mathcal L_{sep}.
\]

---

## 8. Hyperparameter selection firewall

Allowed selection signals:
- familiar A/B task loss
- familiar A/B separation
- curvature on synthetic A→B interpolation cues

Forbidden:
- human Mixed behavior
- human Novel behavior
- matching SRN Mixed/Novel to human outcomes

Recommended grid:

- `lambda_curv ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`
- `lambda_sep ∈ {0.1, 1, 10}`

Prefer the smallest curvature penalty that:
- keeps familiar loss within tolerance,
- preserves ≥90% of High baseline A/B separation,
- substantially reduces output/logit curvature.

---

## 9. Phase 1 evaluation

Freeze trained networks and evaluate:

- A
- B
- Mixed
- A→B cue sweep

For each seed compute:
- familiar A/B hidden separation
- output-relevant A/B separation
- familiar loss
- Mixed loss
- Mixed `q`
- whole-state residual
- output-relevant residual
- output-null residual
- logit residual
- `K_whole_mid`
- `K_out_mid`
- `K_null_mid`
- `K_logit_mid`
- `P_sequence`
- `P_context`
- `P_interaction`

Novel is excluded from the primary gate because current Novel encoding is OOD/untrained.

---

## 10. Phase 1 success gate

C1 passes if all are satisfied:

1. A/B hidden separation ≥ 90% of High baseline.
2. Familiar sequence loss no worse than High baseline by >10%.
3. Mixed loss substantially lower than High baseline.
4. `K_out_mid` substantially lower than High baseline.
5. `K_logit_mid` substantially lower than High baseline.
6. Mixed `q` remains near 0.5.
7. Solution does not collapse toward Low-scale A/B separation.

Primary comparison:

`C1 vs C0`

Secondary:
- C1 vs C2
- C1 vs C3
- C1 vs C4

---

## 11. Strong causal sufficiency check

Repeat the validated post-hoc interventions:

- baseline actual Mixed
- full linear-path patch
- output-only linear-path patch
- null-only linear-path patch

Expected successful C1 pattern:

- actual Mixed loss is already near the patched loss;
- output-only patch adds little further rescue;
- null-only patch still does nothing.

Interpretation:

> training has moved the actual task-relevant hidden state toward the desired geometry.

---

## 12. Phase 1 failure modes

### Failure A
Curvature drops but A/B separation collapses.

Interpretation:
`regularizer recreated Low-scale lumping`

Stop.

### Failure B
Separation preserved, curvature drops, Mixed remains poor.

Interpretation:
`output-relevant curvature is not sufficient`

Stop.

### Failure C
Only whole-state regularization rescues Mixed.

Interpretation:
`full hidden-state geometry matters more than expected`

Revise the hypothesis before Phase 2.

### Failure D
Output-specific regularization preserves split and rescues Mixed.

Proceed to Phase 2.

---

# PHASE 2 — EMERGENCE FROM STRUCTURED EXPERIENCE

## 13. Goal

Test whether a plain SRN can learn the same geometry **without** explicit curvature or separation penalties.

During Phase 2 pretraining:

- `lambda_curv = 0`
- `lambda_sep = 0`

Question:

> What prior experience makes strong context separation coexist with smooth task-relevant geometry?

---

## 14. Structured ecology

Create:

- latent sequences `S1...SN`
- contexts `C1...CM`

Recommended:
- `N = 8`
- `M = 6`

The same sequence must occur in multiple contexts.
The same context must occur with multiple sequences.

Context changes the **surface realization** of the same latent sequence through a stable transformation.

Conceptually:

\[
x_t=T_c\phi(s_t)
\]

where:
- `s_t` = latent sequence token
- `phi(s_t)` = latent token embedding
- `T_c` = stable context-specific input transformation

Target remains:
`next latent token`

This should make context worth representing without making it useful to relearn the whole sequence separately.

---

## 15. Ecology families

Implement at least two structured families.

### E1 — additive context transform

\[
x_t=\phi(s_t)+Ac.
\]

### E2 — low-rank multiplicative transform

\[
x_t
=
\phi(s_t)
+
\sum_r
(c^	op b_r)M_r\phi(s_t).
\]

E2 is the richer structured-composition ecology.

---

## 16. Shuffled control

Create a matched shuffled ecology with identical:
- sequence counts
- context counts
- token marginals
- context marginals
- sequence marginals
- number of updates
- optimizer schedule
- transform magnitudes

Destroy only the stable relation:

`context -> transformation`

Example:
- structured: C3 always means transformation T3
- shuffled: mapping changes across sequence families

---

## 17. Held-out compositional pairs

Train on a sparse sequence × context matrix.

Every held-out pair must satisfy:
- sequence seen elsewhere,
- context seen elsewhere,
- exact pair never seen.

Evaluate held-out pairs zero-shot.

---

## 18. Phase 2 internal gate

Before human-task transfer, structured pretraining must outperform shuffled control on:

- held-out sequence × context accuracy
- held-out loss
- output-relevant curvature
- logit curvature
- formal-model compositionality

Expected structured pattern:
- lower `K_out`
- lower `K_logit`
- better held-out performance

---

## 19. Formal representational models

Fit on pretraining familiar combinations:

### F0
\[
h=\mu+Uz
\]

### F1
\[
h=\mu+Uz+Vc
\]

### F2
\[
h=\mu+Uz+Vc+W(z\otimes c)
\]

### F3
unconstrained sequence × context lookup

Report:
- familiar R²
- held-out pair R²
- selected bilinear rank
- residual fraction
- output-relevant prediction R²
- logit prediction R²

Key test:

> Does structured experience produce stronger low-rank sequence × context structure and better held-out prediction than shuffled experience?

---

## 20. Transfer to human A/B task

Only if the Phase 2 internal gate passes.

Compare:
- structured-pretrained
- shuffled-pretrained
- random-init High baseline

Reset optimizer state.

Fine-tune all identically on familiar human A/B only.

Do not train on Mixed or Novel.

---

## 21. Human-task success criterion

Structured pretraining succeeds if, after A/B fine-tuning:

1. familiar A/B separation is preserved;
2. Mixed loss is lower than random-init High baseline;
3. `K_out` is lower than random-init High baseline;
4. `K_logit` is lower than random-init High baseline;
5. improvement exceeds shuffled-pretraining control;
6. formal-model compositionality is stronger than shuffled control.

If Mixed improves only because A/B separation weakens, classify as failure.

---

## 22. Novel branch

Do not use the current Novel probe as a decisive test.

Current encoding:
- Novel input direction untrained,
- effective input-to-hidden contribution is zero/OOD.

Novel generalization should be addressed later using a learned context space in which Novel is an interpretable new point.

---

## 23. Required outputs

### Phase 1 CSVs
- `phase1_training_summary.csv`
- `phase1_seed_metrics.csv`
- `phase1_cue_sweep.csv`
- `phase1_curvature_metrics.csv`
- `phase1_patching.csv`
- `phase1_hyperparameter_selection.csv`

### Phase 2 CSVs
- `phase2_pretraining_generalization.csv`
- `phase2_pretraining_compositionality.csv`
- `phase2_human_transfer.csv`
- `phase2_seed_metrics.csv`
- `phase2_formal_model_fit.csv`

### Master
- `output_relevant_compositionality_master.csv`

---

## 24. Required figures

### Phase 1
- `P1_01_AB_separation_by_condition.png`
- `P1_02_Mixed_loss_by_condition.png`
- `P1_03_output_curvature_by_condition.png`
- `P1_04_logit_curvature_by_condition.png`
- `P1_05_AB_separation_vs_Mixed_loss.png`
- `P1_06_AB_separation_vs_output_curvature.png`
- `P1_07_output_curvature_vs_Mixed_loss.png`
- `P1_08_linear_path_patching.png`
- `P1_09_cue_sweep_loss.png`
- `P1_10_cue_sweep_output_residual.png`

### Phase 2
- `P2_01_pretraining_heldout_generalization.png`
- `P2_02_pretraining_output_curvature.png`
- `P2_03_pretraining_formal_model_fit.png`
- `P2_04_human_AB_separation.png`
- `P2_05_human_Mixed_loss.png`
- `P2_06_human_output_curvature.png`
- `P2_07_human_tradeoff_map.png`
- `P2_08_structured_vs_shuffled_seed_scatter.png`

---

## 25. TDD / RED → GREEN

### T1
No Mixed/Novel examples enter task loss.

### T2
`lambda=0` equals exact A cue; `lambda=1` equals exact B cue.

### T3
Synthetic affine hidden path yields zero curvature.

### T4
Verify:
- `P_out^2 = P_out`
- `P_null^2 = P_null`
- `P_out P_null = 0`
- `P_out + P_null = I`

### T5
Verify:
\[
W_{out}P_{null}hpprox0.
\]

### T6
If `d_AB >= d_target`, separation penalty equals zero.

### T7
Null-space regularizer cannot directly alter logits through the projector implementation.

### T8
Human Mixed/Novel outcomes cannot affect hyperparameter selection.

### T9
Structured and shuffled ecology exposures are exactly matched.

### T10
Held-out sequence × context pairs never occur during pretraining.

### T11
Structured/shuffled/random models receive identical human A/B fine-tuning.

### T12
Zero optimizer steps occur on Mixed/Novel during human-task transfer.

---

## 26. Reports

### Phase 1
Generate:

`PHASE1_OUTPUT_RELEVANT_COMPOSITIONALITY_SUFFICIENCY.md`

Must answer:
1. Can output curvature be reduced without collapsing A/B separation?
2. Does reducing output curvature reduce Mixed loss?
3. Is output-specific regularization sufficient?
4. Does whole-state regularization add anything?
5. Does null-space regularization fail as predicted?
6. Does Mixed remain near q≈0.5?
7. Does post-hoc patching become unnecessary after successful training?

End with:

`PHASE1_GATE = PASS / FAIL`

### Phase 2
Generate only if Phase 1 passes:

`PHASE2_STRUCTURED_EXPERIENCE_EMERGENCE.md`

Must answer:
1. Does structured pretraining improve held-out sequence × context generalization?
2. Does it reduce output/logit curvature without an explicit regularizer?
3. Does it produce stronger low-rank compositional structure?
4. Does it outperform shuffled control?
5. After human A/B fine-tuning, is familiar splitting preserved?
6. Is Mixed transfer improved?
7. Is the improvement associated with lower task-relevant curvature rather than weaker splitting?

---

## 27. Implementation order

1. Reproduce current High/Low baseline metrics.
2. Implement output/logit curvature regularizer.
3. Implement A/B separation floor.
4. Train Phase 1 C0–C4.
5. Run frozen A→B sweep diagnostics.
6. Run causal linear-path patching.
7. Evaluate Phase 1 gate.
8. If PASS, build structured/shuffled ecology.
9. Validate held-out pair splits.
10. Pretrain Phase 2 models.
11. Run internal compositionality gate.
12. If PASS, fine-tune on human A/B.
13. Freeze and evaluate Mixed.
14. Generate reports.

---

## 28. Final scientific success criterion

The implementation should answer two separate questions:

### Sufficiency

\[
oxed{
	ext{Can an SRN split familiar contexts while preserving smooth task-relevant geometry?}
}
\]

### Emergence

\[
oxed{
	ext{Can structured experience make that geometry emerge spontaneously in a vanilla SRN?}
\]

The second question should only be pursued if the first is experimentally validated.
