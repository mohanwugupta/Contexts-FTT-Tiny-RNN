# PRD: Absolute-Fit Diagnostics and Retrieval-Model Validation

## Goal

Upgrade the Mixed / Novel retrieval pipeline so that model-selection statistics are never interpreted without absolute-fit diagnostics.

The immediate issue is that the current analysis reports a higher predictive likelihood for the categorical endpoint-mixture model (R0) than for the scalar continuous model (R1), while the raw probe RT distributions appear unimodal. The pipeline must therefore determine **why** R0 wins and whether its generative predictions actually resemble the human data.

Core rule:

```text
NO MODEL-SELECTION CONCLUSION
WITHOUT ABSOLUTE-FIT DIAGNOSTICS
```

Do not change the model family yet. First validate the existing models.

---

## 1. Models retained

Keep the current models unchanged for the first diagnostic pass:

```text
R0 = categorical endpoint retrieval
R1 = scalar continuous interpolation
R2 = transition-specific retrieval
R3 = familiar-derived low-rank retrieval
```

The primary diagnostic comparison remains R0 vs R1.

---

## 2. Verify that R0 is a profile-level mixture

This is the highest-priority correctness test.

### Intended categorical hypothesis

For participant i with seven-transition first-Mixed profile y_i:

```text
p(y_i | q)
=
q * PRODUCT_k p(y_ik | A_k)
+
(1-q) * PRODUCT_k p(y_ik | B_k)
```

The latent A/B choice occurs **once for the whole probe profile**.

### Disallowed implementation

Do not implement:

```text
PRODUCT_k [
  q * p(y_ik | A_k)
  +
  (1-q) * p(y_ik | B_k)
]
```

because that permits a participant to retrieve A for one transition and B for another.

### Test

Implement:

```text
test_R0_profile_level_mixture()
```

Construct a synthetic profile and verify that the implemented likelihood equals the whole-profile mixture above. Fail loudly if it does not.

---

## 3. Create a complete long-form fit table

Required file:

```text
full_experiment_fit_long.csv
```

Required columns:

```text
participant
counterbalance
trial
block
sequence
transition
context
probe_type
probe_exposure_index
is_probe
observed_RT
familiar_model
A_anchor
B_anchor
familiar_predicted_RT
R0_expected_RT
R1_predicted_RT
R0_residual_expected
R1_residual
phase
chronological_index
```

This table must contain the complete analyzed experiment and become the source for all mandatory plots.

---

## 4. Mandatory plot A — full familiar-model fit

Create:

```text
figure_A_full_familiar_fit.png
```

Plot every analyzed familiar observation over chronological sequence order.

Show:

```text
observed mean RT
best-fitting familiar state-space prediction
```

Facet or separate by counterbalance.

Mark:

```text
Maze blocks
Melody blocks
probe insertions
```

Purpose: verify that the state-space model generating the frozen anchors actually reproduces the observed familiar learning trajectory.

---

## 5. Mandatory plot B — full probe timeline

Create:

```text
figure_B_full_probe_timeline.png
```

For every probe exposure in chronological order, show:

```text
observed probe RT
A_anchor
B_anchor
R0 expected mean
R1 prediction
```

Separate or distinguish:

```text
Mixed
Novel
NoContextProbe
```

Do not show only first Mixed.

Purpose: verify that the anchors and retrieval predictions remain sensible over the entire experiment.

---

## 6. Mandatory plot C — first-Mixed transition profile

Create:

```text
figure_C_first_mixed_profile_fit.png
```

Across transitions 1...7, show:

```text
A_anchor
B_anchor
observed first Mixed
R0 mixture expectation
R1 predicted profile
```

Include observed uncertainty if participant-level data permit.

Label R0 explicitly as:

```text
R0 mixture expectation
```

because this is only the expected mean of a distributional mixture.

---

## 7. Mandatory plot D — raw first-Mixed RT distributions

Create:

```text
figure_D_first_mixed_raw_distributions.png
```

Include two views.

### D1. Participant mean RT

For each participant:

```text
mean RT across seven first-Mixed transitions
```

Plot the empirical distribution.

### D2. Transition-wise RTs

For each transition:

```text
participant RT distribution
```

Facet transitions 1...7.

Overlay or separately show posterior-predictive distributions from:

```text
R0
R1
```

Purpose: determine whether the fitted categorical model predicts bimodality, endpoint shoulders, broad unimodality, or another shape that resembles the actual data.

---

## 8. Mandatory plot E — participant familiar-axis projections

For each participant's seven-transition first-Mixed profile P_i, compute:

```text
d = B - A

q_proj_i
=
((P_i - A)' d)
/
(d' d)
```

Do not clamp q_proj to [0,1].

Create:

```text
figure_E_participant_projection_distribution.png
```

Plot:

```text
empirical q_proj distribution
R0 posterior-predictive q_proj distribution
R1 posterior-predictive q_proj distribution
```

Purpose: directly test whether whole participant profiles are endpoint-like, intermediate, or broadly distributed.

---

## 9. Posterior-predictive simulation

Implement:

```text
simulate_R0_profiles()
simulate_R1_profiles()
```

Generate at least:

```text
5000 simulated participant profiles
```

per model.

### R0

For each simulated participant:

1. draw one latent endpoint:
   ```text
   z_i ~ Bernoulli(q)
   ```
2. if z_i = A, generate the entire seven-transition profile around A;
3. if z_i = B, generate the entire profile around B.

The latent state must remain constant across all seven transitions.

### R1

Generate the entire profile around:

```text
q*A + (1-q)*B
```

using the same residual/noise assumptions used during fitting.

---

## 10. Mandatory plot F — posterior-predictive geometry

For observed and simulated profiles compute:

```text
q_proj
off-axis distance
mean RT
profile variance
```

Create:

```text
figure_F_posterior_predictive_geometry.png
```

Compare observed distributions against R0 and R1 posterior predictions.

Report at minimum:

```text
observed median q_proj
observed SD q_proj
observed mean off-axis distance
R0 95% predictive intervals
R1 95% predictive intervals
```

---

## 11. Mandatory plot G — residuals over the full experiment

Create:

```text
figure_G_residuals_full_timeline.png
```

Plot:

```text
observed - familiar prediction
observed - R0 mixture expectation
observed - R1 prediction
```

against:

```text
chronological index
probe exposure index
transition
```

Purpose: detect systematic temporal or transition-level failures hidden by aggregate likelihood.

---

## 12. Mandatory plot H — observed vs predicted

Create:

```text
figure_H_observed_vs_predicted.png
```

For R1:

```text
observed RT vs predicted RT
```

For R0:

```text
observed RT vs mixture expectation
```

Label the R0 panel as descriptive only.

Add an identity line.

Report:

```text
RMSE
R²
correlation
```

---

## 13. Explain why R0 or R1 wins

Create:

```text
R0_R1_likelihood_decomposition.csv
```

Required columns:

```text
participant
LL_R0
LL_R1
delta_LL_R1_minus_R0
distance_to_A
distance_to_B
distance_to_R1_mean
q_proj
off_axis_distance
```

Create:

```text
figure_I_participant_delta_LL.png
```

Plot participant-level:

```text
LL_R1 - LL_R0
```

Report:

```text
fraction participants favoring R0
fraction participants favoring R1
median delta LL
mean delta LL
```

Purpose: determine whether the model advantage is broad or driven by a small subset of participants.

---

## 14. Transition-level likelihood diagnostics

Create:

```text
transition_likelihood_diagnostics.csv
```

Required columns:

```text
transition
LL_R0
LL_R1
delta_LL
observed_mean
A_anchor
B_anchor
R1_prediction
```

Do not refit q by transition for this diagnostic.

Because R0's latent state is profile-level, treat this as descriptive decomposition only.

---

## 15. Distribution diagnostics

Create:

```text
first_mixed_distribution_diagnostics.csv
```

For observed first-Mixed data and posterior-predictive draws from R0 and R1, calculate:

```text
mean
SD
skewness
kurtosis
q_proj mean
q_proj SD
q_proj quantiles
off-axis distance mean
off-axis distance SD
```

If robustly available, also report:

```text
Hartigan dip statistic
estimated number of KDE modes
```

These are diagnostics, not binary decision rules.

---

## 16. Familiar-anchor uncertainty must propagate to the model comparison

For every familiar-anchor bootstrap draw:

1. load bootstrap A/B anchors;
2. refit R0;
3. refit R1;
4. compute predictive:
   ```text
   delta_LL = LL_R1 - LL_R0
   ```
5. save q and fit statistics.

Required file:

```text
anchor_uncertainty_R0_R1.csv
```

Required columns:

```text
bootstrap_id
q_R0
q_R1
LL_R0
LL_R1
delta_LL
```

Required summary:

```text
P(delta_LL < 0)
P(delta_LL > 0)
median delta_LL
95% CI delta_LL
```

Create:

```text
figure_J_anchor_uncertainty_delta_LL.png
```

This is more important than plotting uncertainty in q alone.

---

## 17. Participant bootstrap

For every participant-bootstrap sample:

```text
refit R0
refit R1
compute q_R0
compute q_R1
compute delta_LL
```

Save:

```text
participant_bootstrap_R0_R1.csv
```

Report uncertainty for:

```text
q_R0
q_R1
delta_LL
```

Do not report only q uncertainty.

---

## 18. Optional joint uncertainty analysis

If computationally feasible, jointly propagate:

```text
participant resampling
+
familiar-anchor resampling
```

Save:

```text
joint_uncertainty_R0_R1.csv
```

Treat this as robustness analysis.

---

## 19. Absolute-fit gate

Before report generation, require:

```text
FIT_DIAGNOSTICS_COMPLETE = TRUE
```

only if all mandatory figures and CSVs exist.

If any are missing:

```text
stop(
  "Absolute-fit diagnostics incomplete; do not interpret model-selection results."
)
```

---

## 20. Interpretation rules

Do not say:

```text
"categorical retrieval is supported"
```

solely because:

```text
LL_R0 > LL_R1
```

Use staged interpretation.

### Relative fit only

> The endpoint-mixture likelihood exceeded the scalar-interpolation likelihood.

### Relative fit + good posterior-predictive adequacy

> The data are consistent with categorical retrieval of familiar endpoint states.

### R0 wins likelihood but fails absolute fit

> Although the endpoint-mixture model achieved higher likelihood, its generative predictions did not adequately reproduce the observed probe distributions, so the categorical mechanism is not supported.

### Neither model fits adequately

> Neither endpoint retrieval nor scalar interpolation provides an adequate generative description of the probe data.

---

## 21. Update the report

Update:

```text
MIXED_NOVEL_RETRIEVAL_CORRECTED_REPORT.md
```

Add a section before any mechanistic interpretation:

```text
## Absolute model adequacy
```

It must summarize:

```text
full familiar trajectory fit
full probe timeline fit
first-Mixed raw distribution shape
participant q_proj distribution
R0 posterior-predictive adequacy
R1 posterior-predictive adequacy
participant-level likelihood decomposition
anchor-uncertainty delta LL
```

---

## 22. Required output bundle

The final directory must contain:

```text
full_experiment_fit_long.csv

figure_A_full_familiar_fit.png
figure_B_full_probe_timeline.png
figure_C_first_mixed_profile_fit.png
figure_D_first_mixed_raw_distributions.png
figure_E_participant_projection_distribution.png
figure_F_posterior_predictive_geometry.png
figure_G_residuals_full_timeline.png
figure_H_observed_vs_predicted.png
figure_I_participant_delta_LL.png
figure_J_anchor_uncertainty_delta_LL.png

first_mixed_distribution_diagnostics.csv
R0_R1_likelihood_decomposition.csv
transition_likelihood_diagnostics.csv
anchor_uncertainty_R0_R1.csv
participant_bootstrap_R0_R1.csv
joint_uncertainty_R0_R1.csv   # optional

MIXED_NOVEL_RETRIEVAL_CORRECTED_REPORT.md
```

---

## 23. TDD / RED -> GREEN

### T1 — R0 profile-level mixture

Synthetic whole-profile mixture likelihood matches the analytic calculation.

### T2 — constant R0 latent state

A simulated R0 profile cannot switch endpoint identity across transitions.

### T3 — matched R0/R1 data

Assert identical:

```text
participants
transitions
anchors
cross-validation folds
residual family
```

### T4 — posterior simulations use fitted noise

Simulated residual SD matches fitted residual SD within tolerance.

### T5 — complete familiar trajectory

Number of plotted familiar observations equals familiar-analysis row count.

### T6 — complete probe timeline

Every analyzed probe appears exactly once.

### T7 — no probe contamination

No probe observation enters familiar model fitting.

### T8 — frozen anchors

Anchor hashes are unchanged by retrieval diagnostics.

### T9 — uncertainty refits both models

Every anchor-bootstrap draw has valid R0 and R1 fits.

### T10 — interpretation gate

Report generation fails if the absolute-fit bundle is incomplete.

---

## 24. Final decision sequence

After implementation, answer in this order:

1. Is R0 implemented as one latent A/B state per participant profile?
2. Does the familiar model fit the entire familiar trajectory?
3. Are the probe-time A/B anchors sensible over the entire experiment?
4. Does R0 reproduce the raw first-Mixed distribution?
5. Does R1 reproduce it?
6. Does R0 reproduce participant profile projections?
7. Does R1 reproduce them?
8. Why does one model win predictive likelihood?
9. Does the sign of `delta_LL` survive familiar-anchor uncertainty?

Only after these checks should the analysis conclude:

```text
categorical endpoint retrieval
continuous interpolation
or neither model adequately describes the data
```

---

## 25. Success criterion

The implementation succeeds if a reader can determine from the standard output:

```text
what the model predicts at every point in the experiment,
whether those predictions resemble the raw data,
and exactly why one model outperforms another.
```

Relative likelihood by itself is not sufficient.
