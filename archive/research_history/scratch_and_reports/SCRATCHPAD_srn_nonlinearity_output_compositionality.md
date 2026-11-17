# Scratchpad: SRN nonlinearity and output-relevant compositionality

## Frozen decisions

- Reuse the 20 High/20 Low `confirmatory_srn_final` checkpoints; never load the execution
  policy because this analysis concerns the recurrent state and sequence logits.
- Gate all new analyses on numerical reproduction of the previous Mixed q, off-axis residual,
  and endpoint/midpoint losses plus the prespecified qualitative High/Low pattern.
- Correct sequence decoding with within-condition leave-one-seed-out fitting. One scaler and
  decoder trained on other seeds' familiar A/B states scores all five held-out contexts.
  Because independently initialized hidden units are not aligned, interpret this as a strict
  shared-coordinate decoder, not the sole measure of within-network sequence information.
- Treat F2 as a low-rank approximation to the familiar context×position interaction contrast.
  Select rank 0–3 using familiar-only entry-masked matrix-completion CV. With only two endpoint
  contexts, F0–F2 all predict the midpoint by endpoint-linear interpolation; F2 cannot infer a
  curved Mixed path without probe fitting, which is forbidden.
- Define output row/null projectors from the frozen `sequence_head.weight` SVD and test all
  projector/reconstruction/logit-null identities numerically.
- Use exact manual Elman recurrence to expose preactivations and verify it against `nn.RNN` for
  every forward call. Include both nearest-endpoint and midpoint-operating-point activation
  linearizations; the midpoint scheme necessarily touches the true function at lambda=.5.
- Analyze Novel only as an input audit. It never enters compositionality decisions.
- Report hypothesis evidence continuously; do not emit one automatic classification.

## Progress

- [x] Read PRD and audit reusable frozen-diagnostic code/checkpoints.
- [x] Implement exact recurrence, tanh gain, saturation, curvature, Jacobians, and linearization.
- [x] Implement familiar-only F0–F3 models and F2 rank-selection firewall.
- [x] Implement LOSO decoder and exact shared-decoder audit.
- [x] Implement output/null/full path causal interventions.
- [x] Implement seed associations, descriptive nested regressions, report, 15 required tables, and 17 figures.
- [x] Run RED/GREEN tests and fix float32 fused-kernel and small-sample bootstrap defects.
- [x] Exhaustively audit manual recurrence against all 40 checkpoints × 21 cues; measured
  numerical ceilings were `3.1e-5` hidden and `7.7e-5` logits. Scientific metrics use the
  exact fused outputs; manual recurrence is retained only as a preactivation audit.
- [x] Run the full 40-network frozen analysis and inspect the scientific report.
- [x] Tighten interpretation after results: saturation marks the High-scale regime but is not
  supported as a sufficient cause; nearest-endpoint linearization worsens High Mixed loss.
  Recurrent amplification capacity is strong, while its seed-level explanatory associations
  are reported separately rather than treated as proof.
