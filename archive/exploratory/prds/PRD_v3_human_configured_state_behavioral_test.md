# PRD v3: Human Behavioral Test of Context-Configured Sequence States

## 1. Purpose

The SRN work now supports a more specific behavioral theory than a scalar “context changes overall speed” account.

The refined hypothesis is:

> **Context configures the currently active motor-sequence state. Mixed context behavior is therefore not necessarily a single global average of Context A and Context B, but may preserve token-specific structure from the learned A/B sequence representations.**

This PRD tests that hypothesis directly in the human FTT data.

The goal is to determine whether the computational model is merely an existence proof or whether it generates **new, out-of-sample predictions about human behavior**.

The central comparison is:

\[
\boxed{
\text{scalar A/B skill mixture}
}
\]

versus:

\[
\boxed{
\text{sequence-position-specific configured-state account}.
}
\]

---

# 2. Core scientific questions

## Q1 — Are human A/B differences purely scalar?

Does context simply shift the entire sequence RT profile up or down?

Or does:

\[
RT_{A,k}-RT_{B,k}
\]

depend systematically on sequence position \(k\)?

---

## Q2 — Is Mixed behavior described by one global responsibility weight?

Can human Mixed performance be explained by:

\[
RT_{M,k}
=
q RT_{A,k}
+
(1-q)RT_{B,k}
\]

with one \(q\) applying to every sequence position?

---

## Q3 — Does Mixed preserve token-specific A/B structure?

Does the A/B relationship vary by token such that:

\[
q_k
\]

is required?

---

## Q4 — Does the frozen SRN predict the shape of human Mixed behavior?

Do sequence positions that are more A-like versus B-like in the SRN show the same relative pattern in human behavior?

---

## Q5 — Do Mixed and Novel use the same behavioral mechanism?

Are their token-specific configuration profiles equivalent up to a global scalar, or do they show different structure?

---

# 3. Data scope

Use the existing human FTT / context-switching datasets containing:

```text
Context A
Context B
Mixed / Both context
Novel / Third context
```

Use the existing validated preprocessing pipeline.

Do not change:

```text
sequence definition
trial exclusion rules
first-sequence exclusion
error handling
block labels
participant exclusions
counterbalance coding
```

Reuse the established rule that the first sequence of each block is excluded where applicable.

The purpose of this PRD is **modeling**, not reprocessing the behavioral data from scratch.

---

# 4. Unit of analysis

Primary behavioral unit:

```text
participant × block × sequence repetition × sequence position
```

Use sequence position / transition-level RT rather than block-averaged RT.

Primary DV:

\[
\log RT
\]

unless prior preprocessing requires another stabilized RT measure.

Retain sequence-position identity:

```text
k = 1,...,8
```

or the validated transition indexing if the first item has no meaningful preceding transition.

If transition-based indexing is more theoretically appropriate, use:

```text
transition 1→2
2→3
...
8→1
```

but do not mix position and transition definitions within an analysis.

---

# 5. Stage 1: establish whether humans show context-specific sequence profiles

Fit a hierarchical model to pure A/B data only:

\[
\log RT
\sim
Context
+
Position
+
Context\times Position
+
\text{participant effects}.
\]

Recommended mixed-effects specification:

\[
\log RT_{i,t}
=
\beta_0
+
\beta_C C
+
\beta_k Position_k
+
\beta_{Ck}(C\times Position_k)
+
u_i
+
\epsilon.
\]

Include participant random effects as supported by the data.

At minimum:

```text
(1 | participant)
```

Prefer random context slopes if estimable:

```text
(1 + context | participant)
```

Do not force a maximal random-effects structure if singular.

---

# 6. Stage-1 interpretation

## Scalar-context account

Predicts:

\[
\beta_{Ck}\approx0
\]

for all \(k\).

A and B have similar profile shapes, shifted globally.

## Configured-state account

Predicts:

\[
\exists k:
\beta_{Ck}\neq0.
\]

The A/B difference depends on sequence position.

Primary output:

\[
\Delta_i(k)
=
RT_{A,i,k}-RT_{B,i,k}.
\]

This participant-specific A/B difference profile is the key latent behavioral object for later analyses.

---

# 7. Reliability gate for token-specific A/B profiles

Before interpreting token-specific structure, establish that \(\Delta_i(k)\) is reliable enough to model.

Use split-half or odd/even sequence repetitions within pure A/B blocks.

For each participant:

1. estimate \(\Delta_i^{(half1)}(k)\);
2. estimate \(\Delta_i^{(half2)}(k)\);
3. compute profile correlation / ICC.

Report:

```text
participant-level reliability
group-level mean reliability
position-level uncertainty
```

If individual profiles are too noisy, use hierarchical shrinkage estimates rather than raw participant means.

---

# 8. Pure-context profile estimation

Estimate participant-specific pure-context token profiles:

\[
\mathbf r_{A,i}
=
[r_{A,i,1},...,r_{A,i,K}]
\]

and:

\[
\mathbf r_{B,i}
=
[r_{B,i,1},...,r_{B,i,K}].
\]

Use partial pooling.

Do not estimate these from Mixed or Novel trials.

Define participant midpoint:

\[
m_i(k)
=
\frac{
r_{A,i,k}+r_{B,i,k}
}{2}.
\]

Define participant A/B contrast:

\[
\Delta_i(k)
=
r_{A,i,k}-r_{B,i,k}.
\]

---

# 9. Behavioral model ladder

All models share the independently estimated pure A/B profiles.

Probe data are used only to fit retrieval / configuration parameters.

The primary model ladder is:

```text
M0 scalar-context
M1 scalar-mixture
M2 token-specific mixture
M3 low-rank configured-state
M4 SRN-informed configured-state
```

---

# 10. M0 — Scalar-context baseline

Ignore token-specific A/B structure.

Model each context with a global mean:

\[
RT_A=a_i
\]

\[
RT_B=b_i.
\]

Probe:

\[
RT_M
=
q_i a_i+(1-q_i)b_i.
\]

This is the weakest baseline.

---

# 11. M1 — Token-aware scalar mixture

Use full pure A/B token profiles but only one retrieval responsibility:

\[
RT_{M,i,k}
=
q_i r_{A,i,k}
+
(1-q_i)r_{B,i,k}.
\]

Equivalently:

\[
RT_{M,i,k}
=
m_i(k)
+
\lambda_i \Delta_i(k),
\]

where:

\[
\lambda_i=q_i-\frac12.
\]

One \(q_i\) applies to every sequence position.

This is the primary scalar-mixture account.

---

# 12. M2 — Token-specific mixture

Allow:

\[
q_{i,k}
=
q_i+\delta_k+\delta_{i,k}
\]

with hierarchical shrinkage.

Prediction:

\[
RT_{M,i,k}
=
q_{i,k}r_{A,i,k}
+
(1-q_{i,k})r_{B,i,k}.
\]

Do **not** fit eight fully independent unconstrained weights per participant.

Use:

```text
group token effects
participant random deviations
strong hierarchical shrinkage
```

This asks whether the Mixed responsibility changes systematically with sequence state.

---

# 13. M3 — Low-rank configured-state model

Model A/B differences as:

\[
\Delta_i(k)
=
\Delta_i^{shared}(k)
+
\Delta_i^{specific}(k).
\]

Use a low-rank representation across token positions.

Possible implementation:

1. stack participant A/B difference profiles;
2. perform PCA / factor analysis on training participants only;
3. retain smallest number of components explaining >= 90% variance;
4. model Mixed as recombination in this low-dimensional profile space.

For example:

\[
\mathbf r_{M,i}
=
\mathbf m_i
+
\sum_{j=1}^{J}
\lambda_{i,j}
\mathbf u_j.
\]

This is the closest behavioral analogue of the SRN's shared + token-specific context structure.

---

# 14. M4 — SRN-informed configured-state model

Use the frozen SRN to generate a token-specific Mixed configuration profile.

For SRN seed \(s\):

\[
q_{M,s,k}^{SRN}
=
\frac{
RT_{M,s,k}^{SRN}
-
RT_{B,s,k}^{SRN}
}{
RT_{A,s,k}^{SRN}
-
RT_{B,s,k}^{SRN}
}.
\]

Do not use unstable positions where the SRN A/B denominator is below the preregistered reliability threshold.

Aggregate SRN predictions across seeds using:

```text
mean profile
hierarchical seed distribution
or PCA of seed profiles
```

Primary implementation:

\[
q_{M,k}^{SRN}
=
E_s[q_{M,s,k}^{SRN}].
\]

Then predict human Mixed:

\[
RT_{M,i,k}
=
q_{M,k}^{SRN}r_{A,i,k}
+
(1-q_{M,k}^{SRN})r_{B,i,k}.
\]

Allow at most a small number of calibration parameters.

Recommended:

\[
logit(q_{human,i,k})
=
a_i
+
b \cdot logit(q_{M,k}^{SRN}).
\]

The SRN determines the **shape across tokens**.

The human model may calibrate overall bias / amplitude.

Do not freely refit token weights.

---

# 15. Strict SRN prediction condition

Also evaluate a zero-free-token version:

\[
q_{human,k}
=
q_{M,k}^{SRN}.
\]

No token-specific human fitting.

This is the strongest out-of-sample test.

Report it even if a calibrated SRN model performs better.

---

# 16. Human Mixed configuration profile

Compute:

\[
q_{M,i,k}^{human}
=
\frac{
RT_{M,i,k}
-
RT_{B,i,k}
}{
RT_{A,i,k}
-
RT_{B,i,k}
}.
\]

Use only token/participant cells with sufficiently separated A/B endpoints.

Primary reliability threshold should be defined before analysis.

Do not clip \(q\) to \([0,1]\) for statistical analyses.

Extreme values should remain visible and be handled with robust/hierarchical modeling.

---

# 17. Midpoint-deviation formulation

Because responsibility ratios can be unstable when A/B endpoints are close, the primary stable analysis should also use:

\[
D_{M,i,k}
=
RT_{M,i,k}
-
m_i(k).
\]

Under a strict midpoint model:

\[
D_{M,i,k}=0.
\]

Under scalar mixture:

\[
D_{M,i,k}
=
\lambda_i \Delta_i(k).
\]

Under configured-state models:

\[
D_{M,i,k}
=
\lambda_{i,k}\Delta_i(k).
\]

This is the preferred primary formulation.

Responsibility ratios are secondary/descriptive.

---

# 18. Core structural prediction

The configured-state account predicts that a participant's own pure-context A/B profile should predict the structure of Mixed behavior.

Test:

\[
D_{M,i,k}
\sim
\Delta_i(k).
\]

Then test whether the coefficient depends on token:

\[
D_{M,i,k}
\sim
\Delta_i(k)
+
Position_k
+
\Delta_i(k)\times Position_k.
\]

A scalar mixture predicts one common slope.

A configured-state model predicts systematic token variation.

---

# 19. Cross-participant prediction

For participant \(i\), estimate pure A/B profiles without using their Mixed trials.

Use those profiles to predict Mixed.

Primary procedure:

1. fit pure A/B participant profile;
2. freeze it;
3. fit or apply retrieval model;
4. evaluate held-out Mixed observations.

Do not estimate A/B endpoints jointly with Mixed.

This preserves the representation/retrieval separation used in the computational work.

---

# 20. Within-participant held-out evaluation

Where data volume permits, split Mixed repetitions:

```text
train Mixed epochs
held-out Mixed epochs
```

Fit retrieval/configuration parameters only on train epochs.

Evaluate held-out:

```text
log likelihood
RMSE
MAE
```

Primary metric:

\[
\boxed{
\text{held-out log likelihood}
}
\]

because previous analyses showed that RMSE can be insensitive to continuous versus categorical mechanisms.

---

# 21. Leave-participants-out evaluation

Also perform participant-level cross-validation.

For each fold:

1. learn group token configuration structure from training participants;
2. fit only allowed participant-level scalar calibration on held-out participants;
3. evaluate held-out participant Mixed trials.

This is especially important for M2–M4.

---

# 22. Mixed-specific hypotheses

## H-M1

One scalar \(q\) explains Mixed.

Prediction:

```text
M1 ≈ M2/M3/M4 on held-out likelihood
```

## H-M2

Mixed requires sequence-state-specific configuration.

Prediction:

```text
M2 or M3 > M1
```

## H-M3

SRN captures the human configuration pattern.

Prediction:

```text
M4 > M1
```

and ideally:

```text
M4 ≈ or > M2/M3
```

despite fewer free token parameters.

The strongest result would be that the frozen SRN predicts human token structure better than a generic shrinkage model.

---

# 23. Novel model ladder

Repeat the same logic for Novel.

Use independently estimated pure A/B profiles.

Fit:

```text
N1 scalar mixture
N2 token-specific mixture
N3 low-rank configured state
N4 SRN-informed Novel profile
```

Never pool Mixed and Novel by default.

They may use different mechanisms.

---

# 24. Mixed versus Novel profile comparison

Define group-level configuration vectors:

\[
\mathbf q_M
\]

and:

\[
\mathbf q_N.
\]

Test whether:

\[
\mathbf q_N
\approx
a+b\mathbf q_M.
\]

If yes:

> Mixed and Novel may share a common token-specific configuration pattern with different global bias/amplitude.

If no:

> Mixed and Novel likely use different retrieval/configuration mechanisms.

Compare:

```text
profile correlation
Procrustes / affine fit
held-out likelihood under shared-profile model
held-out likelihood under separate-profile model
```

---

# 25. Shared-profile model

Fit:

\[
q_{probe,k}
=
a_{probe}
+
b_{probe}z_k,
\]

where:

\[
z_k
\]

is one shared token configuration profile.

Mixed and Novel get different:

```text
intercept
amplitude
```

but the same shape.

Compare to separate-profile model.

This is a direct test of whether the same latent retrieval computation explains both conditions.

---

# 26. SRN-to-human prediction

Use frozen SRN probe profiles from the human-matched checkpoint.

Primary SRN outputs:

```text
pure A token execution profile
pure B token execution profile
Mixed_norm token profile
Novel token profile
```

Derive:

\[
\mathbf q_{M}^{SRN}
\]

and:

\[
\mathbf q_{N}^{SRN}.
\]

Then compare with human:

\[
\mathbf q_{M}^{human}
\]

and:

\[
\mathbf q_{N}^{human}.
\]

Primary statistics:

```text
Pearson r
Spearman rho
slope
held-out likelihood gain over scalar mixture
```

Do not use group human data to select which SRN seed looks best.

Use all registered SRN seeds or a prespecified aggregation.

---

# 27. SRN feature ablation predictions

Use the SRN mechanistic decomposition to create competing human predictors.

Compare human fit using:

```text
SRN shared context profile only
SRN token-specific residual only
SRN full token-matched profile
```

This mirrors the neural causal result.

If the human behavior is best predicted by the **full token-matched SRN profile**, that would connect the neural mechanism to behavior more tightly.

---

# 28. Counterbalance handling

Counterbalance must be included explicitly.

Do not allow prior-context / previous-context effects to be confounded with A/B label assignment.

At minimum include:

```text
counterbalance
block order
probe order if applicable
```

as nuisance predictors or stratification variables.

Primary token-configuration conclusions must survive these controls.

---

# 29. Temporal analysis

Probe whether configuration changes across:

```text
Mixed1
Mixed2
MixedTest
```

where available.

Questions:

1. Is token-specific structure already present in Mixed1?
2. Does Mixed2 move toward a stable profile?
3. Does final MixedTest reproduce the early configuration?

This can distinguish:

```text
immediate retrieval
rapid adaptation
late learned probe-specific state
```

Do not collapse all Mixed blocks before testing temporal stability.

---

# 30. Recovery / identifiability simulation

Simulate human-like datasets from:

```text
M1 scalar mixture
M2 token-specific mixture
M3 low-rank configured-state
M4 SRN-informed profile
```

using:

```text
observed participant count
observed RT noise
observed A/B endpoint separation
observed missingness
actual block structure
```

Refit all models.

Report recovery matrix.

Key questions:

```text
Can scalar mixture be distinguished from configured-state models?
Can M2 and M3 be distinguished?
Can M4 be recognized when it generated the data?
```

Do not interpret a model-comparison win strongly if recovery shows poor identifiability.

---

# 31. Primary model-comparison metrics

Report:

```text
held-out log likelihood
ELPD / LOO if Bayesian
RMSE
MAE
parameter count
AIC/BIC for descriptive fits
```

Primary:

\[
\boxed{
\text{held-out predictive likelihood}
}
\]

Secondary:

```text
parsimony
calibration
token-profile recovery
```

---

# 32. Main decision logic

## Result A — No human Context × Position interaction

Then human A/B differences are largely scalar.

The configured-state neural mechanism may be an existence proof without a detectable human token-level analogue.

---

## Result B — A/B token structure exists, but scalar Mixed mixture wins

Then context configures human pure-sequence execution, but Mixed retrieval can still be summarized by one global responsibility weight.

---

## Result C — Token-specific Mixed model beats scalar mixture

Behavioral support for configured-state retrieval.

---

## Result D — Low-rank model wins

Human context configuration is structured but need not match the SRN specifically.

---

## Result E — Frozen SRN profile predicts human Mixed

Strong computational-to-behavioral bridge.

The SRN is no longer merely an existence proof.

---

## Result F — SRN predicts Mixed but not Novel

Mixed and Novel use different mechanisms.

This would align naturally with cue-combination versus unfamiliar-context interpretations.

---

## Result G — SRN predicts both Mixed and Novel

Strongest general account.

---

# 33. TDD requirement

All implementation follows:

\[
\boxed{
RED\rightarrow GREEN\rightarrow REFACTOR
}
\]

---

# 34. RED → GREEN tests

## T1 — Pure A/B isolation

A/B profile estimation must use no Mixed/Novel trials.

## T2 — Probe isolation

Mixed model fitting must not update A/B profiles.

## T3 — Sequence-position preservation

Every behavioral row used in token analyses must retain validated token/transition identity.

## T4 — Counterbalance preservation

Context labels and counterbalance metadata must remain intact.

## T5 — Midpoint identity

For synthetic:

```text
A = 400
B = 600
Mixed = 500
```

require:

\[
D_M=0.
\]

## T6 — Scalar mixture identity

For:

```text
q = .75
A = 400
B = 600
```

require:

\[
M=450.
\]

## T7 — Ratio instability flag

When:

\[
|A-B|<\epsilon
\]

responsibility ratios must be flagged unreliable rather than silently interpreted.

## T8 — No token leakage in CV

Held-out participant/token observations cannot appear in model training where prohibited.

## T9 — SRN predictor freeze

Human outcomes may never alter:

```text
SRN token weights
SRN seed selection
SRN profile shape
```

## T10 — All-seed SRN aggregation

Pipeline must use the preregistered SRN aggregation rather than best-seed selection.

## T11 — Mixed/Novel separation

No pooled probe model unless explicitly labeled as a shared-profile hypothesis.

## T12 — Recovery tests

Each candidate model must recover above chance under at least one identifiable synthetic regime before strong interpretation.

---

# 35. Proposed code structure

```text
human_configured_state/
    config.py

    data/
        load_human.py
        token_index.py
        preprocessing_contract.py

    pure_profiles/
        estimate_ab_profiles.py
        reliability.py
        participant_shrinkage.py

    models/
        scalar_context.py
        scalar_mixture.py
        token_specific_mixture.py
        low_rank_configured.py
        srn_informed.py
        shared_probe_profile.py

    srn/
        load_frozen_profiles.py
        aggregate_seeds.py
        derive_q_profiles.py

    evaluation/
        heldout_epochs.py
        leave_participant_out.py
        likelihoods.py
        model_comparison.py

    analysis/
        context_position_interaction.py
        midpoint_deviation.py
        structural_prediction.py
        mixed_vs_novel.py
        temporal_profiles.py
        counterbalance_controls.py

    recovery/
        simulate.py
        recover.py

    plotting/
        pure_profiles.py
        mixed_profiles.py
        novel_profiles.py
        srn_human_alignment.py
        model_comparison.py
        recovery_matrix.py

    pipeline.py
```

---

# 36. Required outputs

```text
human_run_config.json

pure_ab_token_profiles.csv
pure_ab_profile_reliability.csv
context_position_model.csv

mixed_midpoint_deviation.csv
novel_midpoint_deviation.csv

model_M0_scalar_context.csv
model_M1_scalar_mixture.csv
model_M2_token_specific.csv
model_M3_low_rank.csv
model_M4_srn_informed.csv

novel_model_comparison.csv
mixed_model_comparison.csv

srn_mixed_profile.csv
srn_novel_profile.csv
human_srn_alignment.csv

mixed_vs_novel_profile_comparison.csv
shared_vs_separate_probe_profile.csv

counterbalance_sensitivity.csv
temporal_probe_profiles.csv

model_recovery_matrix.csv
identifiability_summary.csv

final_behavioral_mechanism_summary.csv
```

---

# 37. Required figures

## Figure 1 — Human pure A/B token profiles

Show:

\[
RT_{A,k}
\]

and:

\[
RT_{B,k}.
\]

## Figure 2 — Human A/B difference profile

\[
\Delta(k)=RT_A(k)-RT_B(k).
\]

## Figure 3 — Context × sequence-position interaction

Estimated context effect by token.

## Figure 4 — Mixed midpoint deviation by token

\[
D_M(k).
\]

## Figure 5 — Scalar versus token-specific Mixed predictions

## Figure 6 — Human Mixed q-profile

\[
q_M^{human}(k).
\]

## Figure 7 — Frozen SRN Mixed q-profile

\[
q_M^{SRN}(k).
\]

## Figure 8 — SRN versus human Mixed profile

## Figure 9 — Mixed versus Novel profile shape

## Figure 10 — Held-out model comparison

```text
M0
M1
M2
M3
M4
```

## Figure 11 — Recovery matrix

---

# 38. Definition of done

- [ ] Pure A/B token profiles are estimated independently.
- [ ] Context × Position interaction is tested.
- [ ] Profile reliability is quantified.
- [ ] Midpoint deviation is primary stable probe metric.
- [ ] Scalar mixture model is implemented.
- [ ] Token-specific hierarchical mixture is implemented.
- [ ] Low-rank configured-state model is implemented.
- [ ] Frozen SRN-informed model is implemented.
- [ ] SRN seed aggregation is preregistered and fixed.
- [ ] Mixed is never used to define pure A/B representations.
- [ ] Novel is analyzed separately.
- [ ] Mixed versus Novel shared-profile model is tested.
- [ ] Counterbalance is explicitly controlled.
- [ ] Temporal Mixed blocks are evaluated separately before pooling.
- [ ] Held-out log likelihood is the primary comparison metric.
- [ ] Participant-level cross-validation is included.
- [ ] Model recovery is completed.
- [ ] Final interpretation respects identifiability limits.
- [ ] Final report states whether the SRN provides an out-of-sample prediction of human token-level behavior.

---

# 39. Target paper-level claim

The strongest supported claim would be:

> **Context does not merely alter overall motor speed. It changes the configuration of individual states within the learned sequence. Under ambiguous context, humans recombine these context-specific sequence states in a structured way, and the token-specific pattern predicted by a frozen High-Scale recurrent network generalizes to human behavior.**

Compactly:

\[
\boxed{
\text{Context splits the sequence state-by-state; ambiguous retrieval recombines that structure.}
\]
