# Final Figure Captions

**FINAL_FIG01 — Mechanism triangle.** Each point is one seed × Phase 1C hyperparameter cell. Panel A compares correct-path and norm-matched orthogonal wrong-path sweeps; Panels B–C show that lower centered-logit task-path error need not collapse A/B separation or move familiar endpoints. The Mixed midpoint was held out from training and selection.

**FINAL_FIG02 — Correct-versus-wrong specificity.** Pairs match seed, alpha, anchor strength, training duration, and checkpoint rule exactly. The identity line denotes equal Mixed loss. The histogram shows wrong-minus-correct sequence loss; positive values favor the formal path.

**FINAL_FIG03 — Seed-level sufficiency.** One row of aligned values per seed, ordered by the task-path residual of its behavior-blind stable-cell selection. Dashed lines are preregistered gates. A cell was selected from endpoint drift, familiar loss/separation, and path residual only; Mixed outcomes were not accessed.

**FINAL_FIG04 — Causal chain.** Seed-mean empirical summaries connect the frozen High baseline, acute output-only midpoint patch, and learned correct-path intervention. The patch is a counterfactual state replacement, whereas the learned condition changes recurrent parameters with the output head frozen. This schematic does not imply that humans use identical latent vectors.

**FINAL_TS01 — Familiar sequence loss.** Lines are across-seed means and bands are 2.5th–97.5th seed quantiles for complete seven-transition sequences. A/B shading follows the registered schedule. Representatives were chosen without Mixed behavior.

**FINAL_TS02 — Familiar correct-token probability.** Same sequence-level aggregation, bands, schedule, and selection rule as FINAL_TS01.

**FINAL_TS03 — Familiar execution proxy.** Frozen-policy execution/RT proxy for complete familiar sequences; lines and bands summarize seeds, not transitions.

**FINAL_TS04 — Full probe-schedule loss.** Complete familiar schedule followed by held-out Mixed, NoContextProbe, and Novel sequences. The latter two are OOD under the current encoding. Probes are descriptive and did not select cells.

**FINAL_TS05 — Full probe-schedule probability.** Mean correct-token probability per complete sequence, with across-seed mean and 95% seed-quantile band.

**FINAL_TS06 — Full probe-schedule execution proxy.** Frozen-policy proxy at the complete-sequence level; the probe boundary is marked explicitly.

**FINAL_TS07 — Midpoint during fine-tuning.** Held-out Mixed loss, centered-logit path residual, endpoint drift, and remaining output-patch benefit at fixed 0%–100% checkpoints. These evaluations never affected optimization, early stopping, or selection.

**FINAL_TS08 — Dense cue-sweep loss.** Complete-sequence loss from lambda 0 to 1. The dashed midpoint was excluded from every fine-tuning loss.

**FINAL_TS09 — Dense cue-sweep task-path residual.** Centered-logit compositional error for the same readable representatives. Lines are seed means and bands are seed quantiles; all values aggregate seven transitions.
