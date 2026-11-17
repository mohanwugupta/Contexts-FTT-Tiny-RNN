# PRD: Mixed / Novel Retrieval Modeling from Corrected Familiar A/B State-Space Anchors
## Separate Familiar Learning from Probe Retrieval

## 0. Objective

Build a clean human modeling pipeline that answers a different question from the corrected Lump/Cross/Split state-space analysis.

The corrected familiar state-space models answer:

> How is learning distributed across familiar Maze and Melody contexts?

The Mixed/Novel retrieval models answer:

> Given the familiar A/B states that have already been learned, how are those states expressed when context is ambiguous, combined, novel, or absent?

The key principle is:

```text
Familiar A/B learning and probe retrieval are separate model stages.
```

Mixed, Novel, and NoContextProbe must never update the familiar state-space model.

The primary human retrieval question is:

```text
categorical retrieval of A or B
vs
continuous/intermediate expression between A and B
```

The primary probe is the **first Mixed exposure**.

---

# 1. Inputs

The pipeline should consume outputs from the corrected human state-space Rmd.

Required familiar outputs:

```text
corrected familiar model winner
counterbalance-specific parameter estimates
Maze familiar latent trajectories
Melody familiar latent trajectories
Maze familiar predicted RT trajectories
Melody familiar predicted RT trajectories
trial/block/sequence timing information
bootstrap draws or bootstrap prediction intervals, if available
```

Required probe data:

```text
Mixed
Novel
NoContextProbe, if present
```

Important naming rule:

```text
NoContextTraining
```

means the independent original-paper group trained without context.

```text
NoContextProbe
```

means context is removed after familiar A/B learning.

These must never be conflated.

---

# 2. Familiar model dependency

The retrieval analysis must not hard-code Split.

Instead:

1. Read the corrected familiar model-comparison output.
2. Determine the preregistered familiar winner:
   ```text
   Lump
   Cross
   Split
   ```
3. Use that fitted model to generate the familiar A/B anchors used for retrieval modeling.

If the corrected analysis ultimately selects Cross, use Cross.

If it selects Split, use Split.

If familiar model selection is ambiguous under BIC/recovery, allow a sensitivity analysis using the plausible familiar models, but do not choose among them using Mixed outcomes.

---

# 3. Familiar anchors

For each probe time and transition position, generate two counterfactual familiar predictions:

```text
A_anchor(t, k)
B_anchor(t, k)
```

where:

```text
t = probe time / familiar-learning checkpoint
k = transition position, 1...7
```

Interpretation:

```text
A_anchor = expected familiar performance if the sequence were expressed under Context A
B_anchor = expected familiar performance if the sequence were expressed under Context B
```

These anchors must come only from the corrected familiar model.

Do not define Mixed within the state-space model.

Do not set:

```text
Mixed = (A + B)/2
```

inside familiar-state simulation.

---

# 4. Probe timing

Map every probe to the exact familiar-learning state immediately before that probe.

Create:

```text
probe_timeline.csv
```

with:

```text
probe_type
probe_index
block
sequence
counterbalance
familiar_state_checkpoint
```

Primary probe:

```text
first Mixed exposure
```

Secondary probes:

```text
later Mixed exposures
final MixedTest
Novel
NoContextProbe
```

The primary analysis must never average first and later Mixed exposures together.

---

# 5. Canonical familiar endpoints

Because Maze and Melody can differ in absolute difficulty, canonicalize the familiar endpoints for profile-level analyses.

For each counterbalance / bootstrap draw / relevant grouping:

```text
Fast = familiar endpoint with lower mean RT
Slow = familiar endpoint with higher mean RT
```

If labels are swapped:

```text
swap the entire seven-transition profile
```

Never relabel individual transitions independently.

Retain raw Maze/Melody labels in a parallel output.

---

# 6. Model R0 — categorical retrieval

Scientific hypothesis:

> On each Mixed observation, the learner retrieves either the A state or the B state.

Let:

```text
q = probability of retrieving A
1-q = probability of retrieving B
```

For RT observations under Gaussian residual noise:

```text
p(y_M | q)
=
q * N(y_M | A_anchor, sigma)
+
(1-q) * N(y_M | B_anchor, sigma)
```

For the seven-transition profile, this is a mixture distribution over endpoint profiles, not an arithmetic mean of the predicted RTs.

Primary parameter:

```text
q in [0,1]
```

Use the same residual/noise convention as the continuous comparator.

---

# 7. Model R1 — scalar continuous retrieval

Scientific hypothesis:

> Mixed produces a genuinely intermediate expression of the familiar A/B states.

Prediction:

```text
M_hat(k)
=
q * A_anchor(k)
+
(1-q) * B_anchor(k)
```

with:

```text
q in [0,1]
```

Use one scalar q across all seven transitions.

This is the primary continuous model.

The interpretation of q is:

```text
latent retrieval / expression responsibility
```

not literal Euclidean stimulus similarity.

Do not constrain:

```text
q = 0.5
```

for Mixed.

---

# 8. Primary categorical-vs-continuous comparison

R0 and R1 should be matched as closely as possible in:

```text
number of free retrieval parameters
residual model
familiar anchors
probe observations
```

Primary evidence:

```text
held-out / predictive log likelihood
AICc / BIC if appropriate
RMSE as descriptive only
```

The main conclusion should answer:

```text
Does first Mixed look like stochastic endpoint selection
or
a continuous intermediate state?
```

The comparison must not be based on RMSE alone because mixture and mean models can have similar point predictions but different likelihoods.

---

# 9. Model R2 — transition-specific retrieval

Run only after R1.

Allow:

```text
q_k
```

for transition:

```text
k = 1...7
```

Prediction:

```text
M_hat(k)
=
q_k * A_anchor(k)
+
(1-q_k) * B_anchor(k)
```

Use hierarchical shrinkage toward a common q.

Conceptual purpose:

> Test whether a single retrieval responsibility is insufficient to explain the transition profile.

Do not use seven unregularized q values as the primary model.

---

# 10. Model R3 — low-dimensional familiar-derived retrieval basis

This is the richer structured model.

The basis must be estimated **only from familiar A/B structure**.

Do not use Mixed, Novel, or NoContextProbe to learn the basis.

Let familiar profile variation define basis vectors:

```text
B1(k), B2(k), ...
```

Then predict probe profile as:

```text
M_hat(k)
=
baseline(k)
+
sum_j alpha_j * B_j(k)
```

Primary candidate:

```text
rank 2
```

but rank selection must use familiar-only information or prespecified dimensionality.

Do not claim "rank 2 is the true model" solely because it fits Mixed best.

The purpose is to test whether Mixed recombines low-dimensional structure already present in familiar A/B behavior.

---

# 11. Frozen SRN-informed retrieval model

If the earlier SRN-informed human retrieval model is retained:

```text
its basis / weights / geometry must be frozen before seeing probe data
```

Treat it as:

```text
mechanistic existence proof
```

not the primary behavioral winner unless it actually wins predictive comparison.

Do not tune SRN-informed parameters on Mixed.

---

# 12. Models not included as primary candidates

Do not include the old literal task-geometry / Euclidean cue-similarity model as a primary retrieval model.

Reason:

```text
symmetric Combined cues can mechanically imply q = 0.5
```

which makes the result a consequence of stimulus coding rather than learned retrieval.

It may be retained as a descriptive control only.

---

# 13. First Mixed analysis

This is the primary result.

Use:

```text
probe_type = Mixed
probe_exposure_index = 1
```

Fit:

```text
R0 categorical
R1 scalar continuous
R2 transition-specific
R3 low-dimensional
```

but interpret hierarchically:

1. categorical vs continuous;
2. scalar vs richer structured retrieval;
3. only then low-dimensional shape.

Primary outputs:

```text
q estimate
likelihood
AICc/BIC
RMSE
profile R²
between-endpoint fraction
off-endpoint deficit
```

---

# 14. Between-endpoint metric

For each transition:

```text
between_k = 1
```

if:

```text
Fast(k) <= Probe(k) <= Slow(k)
```

otherwise:

```text
between_k = 0
```

Report:

```text
mean between-endpoint fraction
```

This is descriptive and intuitive.

It is not a substitute for likelihood-based model comparison.

---

# 15. Corridor / projection metrics

For each probe profile P:

Define familiar direction:

```text
d = B_anchor - A_anchor
```

Projection coordinate:

```text
q_proj
=
((P - A_anchor)' d) / (d' d)
```

Off-axis residual:

```text
r_off
=
|| P - [A_anchor + q_proj d] ||
```

Report:

```text
q_proj
off-axis distance
inside-segment indicator
```

Do not force q_proj into [0,1] for descriptive geometry.

For fitting R1, q remains constrained to [0,1].

---

# 16. Novel analysis

Novel must be treated separately from Mixed.

Do not require Novel to be a midpoint.

Primary Novel measures:

```text
projection on familiar A/B axis
off-axis residual
distance to A
distance to B
distance to familiar corridor
prediction deficit
```

Fit R0/R1 only if scientifically meaningful, but do not interpret midpoint performance as required.

Preferred interpretation:

```text
Novel can reuse familiar/shared structure while occupying a distinct contextual state.
```

---

# 17. NoContextProbe analysis

NoContextProbe is a frozen probe after A/B learning.

It is not the original NoContextTraining condition.

Analyze using the same geometric/reporting framework as Novel:

```text
A/B projection
off-axis residual
between-endpoint fraction
prediction deficit
```

Do not automatically set:

```text
q = 0.5
```

and do not average Maze and Melody states by definition.

---

# 18. Later Mixed exposures

After the first-Mixed analysis is frozen, examine:

```text
Mixed2
later Mixed
final MixedTest
```

Questions:

```text
Does q change with probe experience?
Does off-axis residual shrink?
Does the transition profile become more endpoint-like or more intermediate?
```

No new model family should be introduced based on later-probe behavior.

---

# 19. Early-vs-late probe sensitivity

Because previous analyses suggested temporal mismatch, report probe-specific estimates:

```text
q_first
q_late
profile R²_first
profile R²_late
```

Do not add a separate learning model merely to fix early/late mismatch unless a later project explicitly studies probe adaptation.

Primary manuscript result remains first exposure.

---

# 20. Bootstrap uncertainty

Use participant bootstrap if raw participant data are available.

Procedure:

```text
sample participants with replacement
recompute probe means
recompute familiar means / anchors as appropriate
refit retrieval model
store parameters and metrics
```

If familiar state-space anchors are themselves bootstrapped, preserve paired bootstrap identity when possible.

Report:

```text
95% percentile CI
```

for:

```text
q
likelihood differences
between-endpoint fraction
profile R²
off-axis residual
```

Do not bootstrap already-averaged rows as though they were participants.

---

# 21. Familiar-anchor uncertainty

Run two versions:

## Primary

```text
anchors fixed at best corrected familiar fit
```

## Sensitivity

```text
propagate familiar-state bootstrap uncertainty into retrieval estimates
```

The sensitivity analysis should determine whether categorical-vs-continuous conclusions depend on uncertainty in the familiar state-space model.

---

# 22. Familiar-model sensitivity

If corrected familiar model selection is not decisive:

```text
repeat retrieval analysis using each plausible familiar model
```

For example:

```text
Cross-derived anchors
Split-derived anchors
```

Do not choose the familiar model that makes Mixed fit best.

Report whether the retrieval conclusion is robust.

---

# 23. Cross-validation / predictive testing

When there are multiple Mixed sequences:

Primary first-exposure model comparison:

```text
fit q on a prespecified subset / aggregate
evaluate predictive likelihood where possible
```

If sample size does not support meaningful CV, use:

```text
likelihood + complexity penalty + bootstrap
```

and state that limitation.

For transition-specific/low-rank models, prefer leave-one-transition-out diagnostics when feasible.

---

# 24. Statistical hierarchy

Primary confirmatory question:

```text
R0 categorical vs R1 scalar continuous
```

Secondary:

```text
R1 scalar vs R2/R3 structured retrieval
```

Tertiary:

```text
Novel / NoContextProbe geometry
later Mixed evolution
```

Do not let secondary models obscure the primary categorical-vs-continuous result.

---

# 25. Human profile correspondence

For each fitted probe model report:

```text
RMSE
correlation
R²
```

against the observed seven-transition probe profile.

These are descriptive.

No basis, weight, or hyperparameter may be selected using the human profile unless it is explicitly part of the fitted retrieval model and penalized appropriately.

---

# 26. No probe learning in familiar state-space

Hard rule:

```text
Mixed
Novel
NoContextProbe
```

must never enter:

```text
state-space parameter estimation
familiar model selection
familiar latent-state updates
fatigue parameter estimation
A_cross estimation
```

Probe models are readouts from frozen familiar states only.

---

# 27. Software architecture

Implement as a separate script/Rmd:

```text
mixed_novel_retrieval_corrected.Rmd
```

Do not add Mixed logic back into:

```text
human_state_space_corrected.Rmd
```

The familiar state-space script should remain a self-contained familiar-learning analysis.

The retrieval script reads familiar outputs as immutable inputs.

---

# 28. Required input contract

The retrieval Rmd should fail loudly unless it finds:

```text
corrected familiar model-comparison table
corrected familiar parameter estimates
counterbalance-specific familiar predictions
probe data
probe timeline
```

Check source hashes if practical.

---

# 29. TDD / RED -> GREEN tests

## T1 — no probe contamination

Assert no Mixed/Novel/NoContextProbe rows appear in familiar-fit input.

## T2 — frozen anchors

Hash familiar anchor file before and after retrieval fitting.

## T3 — categorical endpoint sanity

If q=1:

```text
R0 reduces to pure A
```

If q=0:

```text
R0 reduces to pure B
```

## T4 — continuous endpoint sanity

If q=1:

```text
R1 prediction == A_anchor
```

If q=0:

```text
R1 prediction == B_anchor
```

## T5 — continuous midpoint sanity

If q=.5:

```text
R1 prediction == arithmetic midpoint
```

This is only a unit test, not an imposed parameter value.

## T6 — full-profile canonicalization

Fast/Slow relabeling swaps the whole profile.

## T7 — first-exposure primary

Primary Mixed rows satisfy:

```text
probe_exposure_index == 1
```

## T8 — no automatic NoContext averaging

NoContextProbe cannot call:

```text
(A+B)/2
```

without fitting / declaring a retrieval rule.

## T9 — familiar-derived low-rank basis

R3 basis-generation code must not access probe rows.

## T10 — matched categorical/continuous data

R0 and R1 must use exactly the same observations and familiar anchors.

---

# 30. Required outputs

## Familiar anchors

```text
probe_familiar_anchors.csv
probe_timeline.csv
```

## First Mixed

```text
first_mixed_model_comparison.csv
first_mixed_parameter_estimates.csv
first_mixed_profile_predictions.csv
first_mixed_geometry.csv
first_mixed_bootstrap.csv
```

## Later Mixed

```text
mixed_all_exposures_parameters.csv
mixed_all_exposures_profiles.csv
```

## Novel

```text
novel_retrieval_metrics.csv
novel_geometry.csv
```

## NoContextProbe

```text
no_context_probe_metrics.csv
no_context_probe_geometry.csv
```

## Sensitivity

```text
familiar_model_sensitivity.csv
anchor_uncertainty_sensitivity.csv
```

## Main report

```text
MIXED_NOVEL_RETRIEVAL_CORRECTED_REPORT.md
```

---

# 31. Required figures

## Figure 1 — modeling schematic

```text
corrected familiar A/B learning
-> frozen familiar anchors
-> Mixed/Novel retrieval model
```

## Figure 2 — first Mixed familiar endpoints

Plot:

```text
Fast
Mixed
Slow
```

across seven transitions.

## Figure 3 — categorical vs continuous likelihood

Show primary model comparison.

## Figure 4 — scalar continuous fit

Observed Mixed profile + R1 prediction.

## Figure 5 — structured retrieval comparison

R1 vs R2 vs R3.

## Figure 6 — q bootstrap

Distribution / CI for first Mixed q.

## Figure 7 — familiar-axis geometry

A, B, Mixed, Novel, NoContextProbe projection and off-axis distance.

## Figure 8 — first vs later Mixed

q and profile metrics over probe exposure.

## Figure 9 — familiar-model sensitivity

Cross-derived vs Split-derived anchors if applicable.

## Figure 10 — uncertainty propagation

Retrieval estimates with fixed vs bootstrapped familiar anchors.

---

# 32. Decision logic

## Outcome A — continuous decisively beats categorical

Interpretation:

> Mixed behavior is better described as a genuinely intermediate expression of familiar states than as stochastic A/B retrieval.

## Outcome B — categorical matches/beats continuous

Interpretation:

> Apparent intermediate averages may arise from endpoint switching across observations/participants.

Do not claim recombination.

## Outcome C — scalar continuous is sufficient

Interpretation:

> A single retrieval-responsibility parameter explains the probe profile.

Prefer the simple model.

## Outcome D — structured low-dimensional model improves substantially

Interpretation:

> Mixed recombination preserves structured transition-level information not captured by a single scalar responsibility.

## Outcome E — Novel lies off the A/B axis

Interpretation:

> Novel context is not simply an A/B mixture and may engage shared familiar structure through a distinct contextual state.

## Outcome F — results depend strongly on familiar Cross vs Split anchors

Interpretation:

> Retrieval conclusions remain contingent on familiar-learning model uncertainty.

Report this rather than selecting the anchor model using probe behavior.

---

# 33. Central success criterion

The strongest result would be:

> Familiar A/B learning is estimated independently with the corrected human state-space model. On the first exposure to Mixed, without updating those familiar states, human behavior is better predicted by a continuous or structured recombination of the familiar representations than by stochastic retrieval of one familiar endpoint.

This directly separates:

```text
how familiar skills are learned
```

from:

```text
how familiar skill states are expressed under ambiguous context
```

and creates the clean human target for the neural-network analysis.
