# Final Task-Space Compositionality Report

## 1. Executive conclusion

Strong familiar A/B context splitting can coexist with ambiguous-context transfer when the task-relevant
centered-logit projection follows an approximately compositional path between familiar solutions. Claims A
(behavioral usefulness), B (subset sufficiency), and C (directional specificity) are respectively
**supported**,
**supported**, and
**supported**. No new model was trained.

## 2. Label/selection audit

The Phase 1C report's historical “best stable” label denoted a single seed-cell, whereas the old plot role
denoted an aggregate representative. The audit preserves 5 historical role records with
explicit descriptive qualifiers. The new **best stable aggregate cell** is
`P1_correct_logit_alpha_10_anchor_30`. Seed-specific stable cells were selected without Mixed performance, correct-token
probability, rescue fraction, or remaining patch gap. Full details are in
`audit/PHASE1C_LABEL_AUDIT.csv`.

## 3. Formal task-space definition

For logits $\ell_k(c)=W_{out}h_k(c)+b$, task state is the centered logit vector
$z_k(c)=C\ell_k(c)$, where $C=I-\frac{1}{m}\mathbf{1}\mathbf{1}^\top$.
The compositional target is $z_k^*(\lambda)=(1-\lambda)z_k(A)+\lambda z_k(B)$, and sequence error is
$E_{task}=\frac{1}{7}\sum_k\|z_k(c_\lambda)-z_k^*(\lambda)\|_2^2$.
This constrains only output-relevant geometry; output-null activity need not be linear.

## 4. Baseline split-vs-transfer tension

Frozen High networks had mean A/B separation 2.930 and Mixed sequence loss
2.248. Frozen Low networks had Mixed loss 0.049.
Every High normalized task-path residual equals 1 by construction.

## 5. Acute patching evidence

The output-only endpoint-linear patch reduced baseline High Mixed loss to
0.809 on average. Rescue denominators and fractions were
computed within seed before aggregation; the Low control was never assigned a High denominator.

## 6. Learned path sufficiency evidence

The stable aggregate correct-path cell retained an A/B separation ratio of
0.996, endpoint hidden drift 0.086, and normalized
task-path residual 0.548; its Mixed loss was
1.581. Under the behavior-blind per-seed rule, 4/20
(20.0%) seeds passed all preregistered endpoint, path, rescue, and remaining-gap gates.

## 7. Dose-response

Across the complete P1 sweep, the task-path coefficient predicting Mixed loss was 1.035
(cluster-robust SE 0.294; seed-bootstrap 95% CI [0.367, 2.204]).
Pearson $r=0.400$ and Spearman $\rho=0.400$. Lower path residual therefore
tracked better transfer; this regression is descriptive rather than standalone causal evidence.

## 8. Wrong-path specificity

Across exact seed × alpha × anchor matches, wrong-minus-correct Mixed loss averaged 0.334
(median 0.090; seed-bootstrap 95% CI [0.090, 0.573]);
61.3% of pairs favored the correct path. The aggregate matched wrong-path Mixed
loss was 2.153. Pair duration, source checkpoint, and final-checkpoint rule were
audited exactly.

## 9. Seed-level heterogeneity

The ordered selected-cell distribution is classified descriptively as **continuous frontier**
(largest adjacent-gap fraction 0.195). The 20% full
gate fraction therefore supports sufficiency in a subset, not universality.

## 10. Sequence-level familiar behavior

FINAL_TS01–03 show loss, correct-token probability, and execution proxy over the complete alternating A/B
schedule. Each plotted point aggregates all seven trained transitions, bands are 2.5th–97.5th percentiles
across seeds, and representatives were chosen without held-out Mixed behavior.

## 11. Sequence-level probe behavior

FINAL_TS04–09 retain the complete probe schedule, midpoint-over-fine-tuning trajectory, and dense cue sweep.
The midpoint was held out from training and selection. Every point aggregates exactly seven transitions;
no item-level plot substitutes for these required figures.

## 12. Formal scope of the claim

**Definition.** Task-space compositional interpolation holds when $z_k(c_\lambda)$ approximates the
endpoint-linear task state. **Empirical sufficiency.** Within this SRN family, reducing deviation from that
path can improve zero-shot Mixed performance while familiar endpoints and A/B separation remain approximately
fixed. **Specificity.** Matched orthogonal displacement does not provide equivalent benefit. This does not
establish necessity, universal retrofittability, a known developmental ecology, or literal human neural
implementation.

## 13. Paper-ready result statement

Context splitting and generalization are not opposites. In these split SRNs, held-out ambiguous-context
transfer improved as the task-relevant projection approached a compositional route between familiar A/B
solutions, without requiring the whole hidden state to become linear or unsplit. Acute output-path patching,
learned path alignment, and matched wrong-direction controls jointly support this mechanistic interpretation,
while the 4/20 feasible-seed result limits it to demonstrated subset sufficiency.
