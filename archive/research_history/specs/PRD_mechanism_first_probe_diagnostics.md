# PRD: Mechanism-First Diagnostics for Mixed and Novel Retrieval

## Goal

Determine **why** the human and SRN probe results look the way they do before changing the network again.

The analysis must separate two quantities that are currently confounded:

\[
\boxed{
\text{probe RT}
=
\text{condition-wide cost}
+
\text{transition-profile expression}
}
\]

The central question is whether Mixed/Combined is slow because it incurs a global ambiguity or out-of-distribution cost, because it fails to retrieve the learned A/B transition structure, or both.

The work must proceed stepwise:

1. inventory and freeze the existing human and SRN inputs;
2. decompose condition level from transition-profile shape in both systems;
3. re-score the existing A/B and low-rank accounts on profile shape alone;
4. test whether human retrieval weights or global costs vary with recent context history, but only if the design makes those effects identifiable;
5. test a genuinely independent shared/default account for Novel, but only if an independent default basis exists;
6. use the joint result to choose the next SRN mechanism.

Do not train a new SRN or redesign the context environment in this PRD.

---

## 1. Scientific findings to preserve

Treat the following as established inputs, not questions to rerun from scratch:

```text
Human data:
    continuous expression beats categorical A-or-B switching
    one global scalar q is insufficient
    transition structure matters
    low-rank retrieval is the strongest predictive account so far
    a compact independently estimated A/B basis captures meaningful structure
    Mixed/Combined and Novel/Third have qualitatively different profiles

SRN data:
    High-scale networks split A and B strongly
    Low-scale networks remain comparatively lumped
    High-scale learning produces context-yoked Cross/Split RT dynamics
    shared ordinal structure can coexist with the A/B split
    Mixed remains mostly slower than both trained endpoints
    Mixed hidden states remain off the A/B segment
    A/B transition-profile orientation is substantially seed-specific
```

The new work must refine the mechanism behind these findings. It must not treat already rejected categorical switching or an unaugmented scalar mixture as the main alternative.

---

## 2. Scientific distinction

For participant or seed \(i\), condition \(c\), and transition \(k\), represent the estimated condition profile as:

\[
M_{i,c,k}=L_{i,c}+Z_{i,c,k},
\]

with:

\[
L_{i,c}=\frac{1}{K}\sum_{k=1}^{K}M_{i,c,k}
\]

and:

\[
Z_{i,c,k}=M_{i,c,k}-L_{i,c},
\qquad
\sum_k Z_{i,c,k}=0.
\]

Here:

```text
L = condition-wide RT level
Z = centered transition-profile shape
```

This distinction is the organizing principle for every analysis and figure.

For a probe condition \(P\), the A/B endpoint model is:

\[
M_{i,P,k}
=
\delta_{i,P}
+
(1-q_{i,P})M_{i,A,k}
+
q_{i,P}M_{i,B,k}
+
\epsilon_{i,P,k}.
\]

Equivalently, the profile-shape fit is:

\[
Z_{i,P,k}
\approx
(1-q_{i,P})Z_{i,A,k}
+
q_{i,P}Z_{i,B,k},
\]

and the condition-wide residual cost is:

\[
\delta_{i,P}
=
L_{i,P}
-
\left[(1-q_{i,P})L_{i,A}+q_{i,P}L_{i,B}\right].
\]

Interpretation:

```text
q describes which A/B transition-profile structure is expressed.
delta describes slowing or speeding not explained by that structure.
```

The two must never be collapsed into one score.

---

## 3. Mechanistic hypotheses

The analysis adjudicates four nonexclusive accounts.

### H1 — global ambiguity or novelty cost

Mixed or Novel has a positive \(\delta\), even when its centered transition profile reuses learned structure.

### H2 — structured continuous retrieval

The centered probe profile is predicted by an A/B or low-rank basis, even if the raw RT curve is globally displaced.

### H3 — history-dependent latent-context inference

Recent A/B experience changes \(q\), rather than only changing \(\delta\).

### H4 — shared/default retrieval for Novel

Novel expresses an independently estimated context-neutral sequence profile rather than an A/B midpoint.

These map onto distinct theoretical families:

```text
cue conflict / uncertainty              -> primarily delta
latent-context inference                -> history-dependent q
low-dimensional compositional retrieval -> centered profile reuse
shared schema / default state           -> independent Novel basis
```

The final report should use this mapping for interpretation. It should not infer a cognitive mechanism from mean RT alone.

---

## 4. Scope

### In scope

```text
existing cleaned human trial-level data
existing participant-level or group-level human model artifacts
existing frozen SRN probe outputs and checkpoints
existing low-rank and A/B-basis predictions
new level-versus-shape analyses
new delta-plus-q fits
new history identifiability audit and gated history models
new independent shared/default comparison for Novel, if identifiable
cross-system diagnostic summary
```

### Out of scope

```text
new SRN training
architecture changes
context-encoding changes
new RT readout fitting
human-target tuning of network parameters
rerunning categorical-versus-continuous retrieval as a primary question
transition-by-transition label flipping
defining the default state as (A + B) / 2
forcing Mixed and Novel into the same model family
```

---

## 5. Frozen analysis conventions

Use the repository's current canonical conventions for:

```text
human exclusions
RT units
error-trial handling
transition numbering
condition naming
participant identifiers
SRN seed identifiers
checkpoint selection
cross-validation folds
low-rank dimensionality and regularization
```

Primary human analyses must use cleaned RT in milliseconds because \(\delta\) is an additive time cost.

If the repository's canonical inferential model uses log RT, reproduce the full analysis on log RT as a labeled robustness analysis. Do not silently switch scales.

Use raw model RT units for within-SRN analyses. Do not compare the numerical magnitude of human and SRN \(\delta\) without an explicitly documented normalization.

---

## 6. Required implementation shape

Add one configuration-driven analysis entry point using the repository's existing language and CLI conventions.

Conceptual interface:

```text
run_probe_diagnostics --config <path>
```

The configuration must declare:

```text
input paths
condition aliases
transition column and valid transition set
participant/seed identifiers
trial, sequence, block, and order columns when present
existing fold assignments
human RT scale
bootstrap or posterior settings
history identifiability thresholds
output directory
random seed
```

Do not hard-code machine-specific absolute paths.

The pipeline must be restartable by stage and must not overwrite canonical input artifacts.

---

## 7. Stage 0 — input and provenance audit

Before fitting anything, locate and record the exact inputs used.

Required inventory:

```text
human cleaned trial-level table
human condition-level/modeling table, if separate
existing categorical-switching comparison summary
existing scalar-q results
existing transition-specific results
existing low-rank basis and held-out predictions
existing A/B-basis results
frozen SRN A/B/Mixed/Novel transition RTs by seed
SRN hidden-state geometry outputs, if available
whole-profile Fast/Slow canonicalization map, if available
cross-validation fold assignments
```

For every input, record:

```text
path
file size
SHA-256 hash
row count
column names and types
condition levels
transition levels
participant or seed count
missingness
```

Export:

```text
input_manifest.csv
schema_audit.json
condition_aliases.json
```

Accepted aliases:

```text
Combined == Mixed
Novel == Third
```

Choose one canonical internal label for each and retain the original label in exported data.

---

## 8. Stage 0 gates

### G0.1 — transition identity

Human and SRN profiles must use the same documented ordinal transition definition before cross-system summaries are produced.

### G0.2 — endpoint availability

Every unit included in an A/B projection must have estimable A and B endpoint profiles.

### G0.3 — probe leakage audit

Confirm whether each existing basis was learned with or without Mixed/Novel rows.

Only bases learned without the evaluated probe condition may be called predictive or independent.

### G0.4 — canonicalization audit

Any Fast/Slow relabeling must operate on the entire A or B profile for a participant/seed and must use A/B data only.

### G0.5 — reproducibility

The same input and configuration must reproduce identical cell estimates and fold assignments.

If a required input cannot be located, stop only the dependent branch and record the missing artifact. Do not fabricate or silently refit a substitute.

---

## 9. Stage 1 — level-versus-shape decomposition

Run this stage for:

```text
human A
human B
human Mixed/Combined
human Novel/Third
SRN A by seed
SRN B by seed
SRN Mixed by seed
SRN Novel by seed
```

For human data, obtain participant × condition × transition estimates using the repository's existing cleaning and adjustment model. If trial-level covariate adjustment is already canonical, use estimated marginal cell means from that model.

Do not introduce a new outlier rule.

For SRNs, use frozen transition RTs directly.

For each unit and condition, export:

```text
level L
centered transition values Z_k
number of contributing observations
standard error or posterior uncertainty when available
missing-cell flag
```

Verify exact reconstruction:

\[
M_{i,c,k}=L_{i,c}+Z_{i,c,k}.
\]

---

## 10. Stage 1 endpoint projection

For Mixed and Novel separately, estimate the unconstrained descriptive A/B coefficient:

\[
\hat q_i
=
\arg\min_q
\sum_k w_{i,k}
\left[
Z_{i,P,k}
-
\{(1-q)Z_{i,A,k}+qZ_{i,B,k}\}
\right]^2.
\]

Use inverse-variance weights only if they are available from the canonical cell-estimation model. Otherwise use equal transition weights.

Primary \(q\) is unconstrained. Also report a constrained \(0\le q\le1\) sensitivity fit.

Then compute:

\[
\hat\delta_i
=
L_{i,P}
-
\left[(1-\hat q_i)L_{i,A}+\hat q_iL_{i,B}\right].
\]

Required per-unit diagnostics:

```text
q unconstrained
q constrained
delta
A/B centered-contrast norm
profile RMSE
profile R-squared
Pearson correlation
cosine similarity
fraction of raw transitions between A and B
fraction of centered transitions between A and B
```

Do not interpret unstable unit-level \(q\) values when the A/B centered contrast is approximately zero. Retain those rows, flag them, and let the hierarchical/population summary carry the inference.

---

## 11. Stage 1 primary questions

For each probe and system, answer:

1. Is \(\delta\) reliably positive?
2. Does the centered probe profile align with the A/B profile family after removing \(\delta\)?
3. How much of the original probe error was level error versus shape error?
4. Does whole-profile Fast/Slow canonicalization change only orientation, or does it materially improve shape agreement?
5. Is the SRN Mixed failure still a failure after condition-wide displacement is removed?

The most important SRN distinction is:

```text
large positive delta + good centered-profile reuse
    -> primarily a global/readout/OOD penalty

large positive delta + poor centered-profile reuse
    -> global penalty plus representational retrieval failure
```

---

## 12. Stage 1 uncertainty

Human population summaries:

```text
resample participants, not individual RT rows
preserve all conditions and transitions within a resampled participant
report point estimate, 95% interval, and full bootstrap/posterior draws
```

SRN population summaries:

```text
resample seeds
preserve all conditions and transitions within a resampled seed
report point estimate, 95% interval, and full bootstrap draws
```

Use at least 2,000 bootstrap draws unless the repository already uses a larger frozen value.

Do not use transition rows as independent replicates.

---

## 13. Stage 2 — re-score existing retrieval models by level and shape

Do not launch a fresh model search.

Re-evaluate only the already established model families:

```text
condition intercept / shared-transition baseline
scalar A/B basis
existing transition-specific or token-specific model
existing low-rank retrieval model
existing independently estimated A/B basis
```

Categorical A-or-B switching may appear as a reference row using its existing result. Do not refit or tune it as a new primary candidate.

For every existing held-out prediction, compute two score families.

### Raw-profile score

Score the original prediction against \(M_{i,P,k}\).

### Centered-shape score

Center prediction and observation within unit × probe condition, then score:

\[
\hat Z_{i,P,k}
=
\hat M_{i,P,k}
-
\frac{1}{K}\sum_k\hat M_{i,P,k}.
\]

Use the same held-out rows and folds as the original comparison.

Required metrics:

```text
held-out RMSE
held-out MAE
held-out log likelihood or ELPD, when available
profile correlation
variance explained in condition level
variance explained in centered shape
```

---

## 14. Stage 2 additive-cost augmentation

Add a condition-wide probe intercept to the existing scalar A/B and low-rank models:

\[
RT_{i,P,k}
=
\delta_{i,P}
+
\widehat{RT}^{\text{structure}}_{i,P,k}
+
\epsilon.
\]

This augmentation must be evaluated out of sample.

If repeated probe sequences are available, estimate \(\delta\) on training repetitions and score held-out repetitions. If only aggregated profiles are available, label the intercept-optimized result **descriptive ceiling**, not predictive evidence.

Do not allow a transition-specific intercept in this stage. That would absorb the profile structure being tested.

---

## 15. Stage 2 decision rule

The low-rank account remains the central retrieval model only if its advantage persists for centered transition shape.

Interpretation:

```text
low-rank wins raw and centered scores
    -> evidence for structured low-dimensional expression

low-rank wins raw but not centered scores
    -> prior advantage was driven mainly by condition level

adding delta markedly improves raw prediction while centered fit is unchanged
    -> ambiguity/novelty cost is separable from retrieval structure

no existing model predicts centered probe shape
    -> current human claim must be weakened before further SRN engineering
```

Do not use a single in-sample \(R^2\) to choose among these interpretations.

---

## 16. Stage 3 — history identifiability audit

Before fitting history-dependent retrieval, audit whether the experiment contains usable within-participant variation.

Candidate history fields, in priority order:

```text
immediately preceding context
most recent unambiguous A/B context
signed A-versus-B recency
number of sequences since last A exposure
number of sequences since last B exposure
probe epoch or probe repetition
position within test block
time since last A/B exposure
test-condition order
```

For each candidate, report:

```text
source column and derivation
missingness
within-participant variation
counts by level or value range
association with probe condition
association with transition and trial order
pairwise predictor correlations
design-matrix rank
variance inflation factor where defined
whether the predictor is experimentally randomized, counterbalanced, or observational
```

Export:

```text
history_identifiability.csv
history_confound_matrix.csv
history_branch_status.json
```

---

## 17. Stage 3 gates

A history predictor is eligible for the primary model only if:

```text
its derivation is unambiguous
it varies within participants
the relevant levels have repeated probe sequences
it is not perfectly determined by test order or probe epoch
the resulting design matrix is full rank
```

Default implementation thresholds, overridable only in the preregistered config:

```text
categorical predictor:
    each contrasted level appears in at least 5 probe sequences
    for at least 50% of included participants

continuous predictor:
    at least 5 distinct within-participant values
    for at least 50% of included participants

collinearity:
    absolute predictor correlation < .80
    VIF < 5 when VIF is defined
```

If no predictor passes, stop Stage 3 and report that history-dependent retrieval is not identifiable from this design.

Do not turn a failed audit into a null psychological result.

---

## 18. Stage 3 — history-dependent model

Fit Mixed/Combined first.

For probe sequence \(t\), transition \(k\), and participant \(i\):

\[
RT_{i,t,k}
=
\delta_{i,t}
+
(1-q_{i,t})A_{i,k}
+
q_{i,t}B_{i,k}
+
\epsilon_{i,t,k},
\]

where:

```text
A_i,k and B_i,k are the participant-specific full endpoint profiles.
The participant random intercept is carried by delta_i,t, so no second
free intercept is added to the observation equation.
```

and:

\[
\delta_{i,t}
=
\delta_0+u^{\delta}_i+\gamma^\top H_{i,t}
\]

and:

\[
\operatorname{logit}(q_{i,t})
=
\beta_0+u^q_i+\beta^\top H_{i,t}.
\]

Use participant-specific A/B endpoint profiles estimated independently of the probe sequence being predicted.

Primary history models:

```text
H0: static delta, static q
Hdelta: history changes delta only
Hq: history changes q only
Hboth: history changes delta and q
```

Fit each eligible history predictor individually first. Fit a joint model only with predictors that are jointly identifiable and show out-of-sample value individually.

Primary \(q\) is constrained to \([0,1]\) here because Stage 3 tests latent responsibility. Preserve the unconstrained Stage 1 projection as a descriptive diagnostic.

---

## 19. Stage 3 evaluation

Use leave-one-probe-sequence-out or leave-one-probe-block-out prediction within participant, depending on the unit of repeated measurement.

Compare models using:

```text
held-out log likelihood or ELPD
held-out RMSE
calibration of predicted condition level
calibration of predicted centered profile
estimated history effect on delta
estimated history effect on q
between-participant heterogeneity
```

Required negative controls:

```text
permute history labels within participant while preserving probe epoch
fit trial/order covariates without contextual history
verify that a simulated delta-only effect is not recovered as q
verify that a simulated q-only effect is not recovered as delta
```

Use false-discovery-rate control for exploratory single-predictor history tests. The multivariable model is confirmatory only if its predictor set was frozen before viewing its coefficients.

Novel history effects are secondary and should be fit only after Stage 4 establishes that an A/B responsibility parameter is meaningful for Novel.

---

## 20. Stage 3 interpretation

```text
history improves Hq but not Hdelta
    -> evidence for dynamic latent-context responsibility

history improves Hdelta but not Hq
    -> evidence for time-varying conflict, uncertainty, or general slowing

history improves Hboth
    -> retrieval selection and global cost both vary over time

history improves neither
    -> static structured retrieval is sufficient at the available resolution

history branch not identifiable
    -> no mechanistic conclusion; recommend a future design manipulation
```

Use causal language only for randomized or counterbalanced history contrasts.

---

## 21. Stage 4 — independent shared/default source audit

Novel must not be defined as a 50/50 A/B blend.

Search for a candidate default basis in this priority order:

```text
1. independent human no-context or single-context control condition
2. independent familiar-sequence baseline collected without A/B probe ambiguity
3. a training-only invariant component from the existing low-rank model,
   learned without Mixed or Novel rows and evaluated on held-out participants
```

For the SRN, candidate independent sources include:

```text
the frozen sequence-only control
a training-only invariant component learned without Mixed/Novel probes
```

For each candidate, record:

```text
source dataset
rows used to estimate it
whether A/B probe data contributed
whether Novel data contributed
normalization
fold assignment
reliability across participants or seeds
```

Reject a candidate if it is merely:

```text
(A + B) / 2
```

or if Novel rows were used to choose or rotate it.

If no independent source exists, stop Stage 4 and state that a shared/default account is not identifiable with the present data.

---

## 22. Stage 4 — Novel model comparison

Center every basis before comparing transition shape.

Let \(S_{i,k}\) be the independently estimated shared/default profile and:

\[
D_{i,k}=Z_{i,B,k}-Z_{i,A,k}.
\]

Compare:

```text
N0: condition level only
N1: shared/default basis only
N2: A/B contrast basis only
N3: shared/default + A/B contrast
N4: existing low-rank basis + additive delta
N5: probe-specific transition profile, descriptive ceiling only
```

The central structured model is:

\[
Z_{i,N,k}
=
\lambda_{S,i}S_{i,k}
+
\lambda_{D,i}D_{i,k}
+
\epsilon_{i,N,k}.
\]

Estimate the Novel level \(\delta_{i,N}\) separately.

Use held-out participants/seeds or the repository's existing folds. A basis learned on one unit may be applied to the held-out unit only after a fold-safe orientation and normalization rule has been fixed.

---

## 23. Stage 4 primary questions

1. Does Novel transition shape align with an independent shared/default basis?
2. Does the shared basis add held-out value beyond the A/B contrast?
3. Is Novel's global cost larger than Mixed's after accounting for structure?
4. Do human and SRN Novel probes recruit the same class of basis?
5. Does the SRN sequence-only control provide a better Novel account than the trained A/B corridor?

Do not require Mixed and Novel to have the same answer.

---

## 24. Stage 5 — cross-system adjudication

Create one summary table with rows:

```text
Human Mixed
Human Novel
SRN Mixed
SRN Novel
```

and columns:

```text
global cost delta
A/B centered-profile reuse
low-rank centered-profile reuse
history effect on delta
history effect on q
shared/default basis evidence
off-manifold hidden-state distance, SRN only
branch identifiability status
```

Do not directly compare raw human milliseconds with normalized SRN RT.

For cross-system visualizations, standardize only the centered profiles within system and preserve the unstandardized \(\delta\) results in separate panels.

---

## 25. SRN decision gate

Use the diagnostic pattern to select one next modeling target.

### Result A — SRN Mixed shape is good, SRN delta is too large

Interpretation:

> The SRN may retrieve useful transition structure, but the confidence-to-RT pathway adds an excessive global OOD penalty.

Next target:

```text
separate global ambiguity/readout cost from transition-specific retrieval expression
```

Do not rebuild the hidden-state retrieval mechanism first.

### Result B — SRN Mixed shape is poor after centering

Interpretation:

> Mixed fails to reuse learned A/B or low-rank transition structure.

Next target:

```text
an explicit retrieval, gating, or compositional state-selection mechanism
```

### Result C — human q is history-dependent

Interpretation:

> A static geometric interpolation target is incomplete.

Next target:

```text
a recurrent latent-context responsibility process driven by recent context evidence
```

### Result D — human delta is history-dependent but q is not

Interpretation:

> Recent experience changes conflict or uncertainty cost more than retrieved content.

Next target:

```text
a separate control/uncertainty signal, leaving retrieval weights static
```

### Result E — human Novel uses an independent shared/default basis

Interpretation:

> Novel should not be implemented as an A/B midpoint or as simultaneous A+B evidence.

Next target:

```text
a context-neutral/shared sequence state with a separate novelty cost
```

### Result F — no existing basis predicts human centered shape

Interpretation:

> The behavioral target is not yet specified well enough to justify more SRN tuning.

Next target:

```text
revise the human descriptive model or collect a discriminating experiment
```

---

## 26. Required TDD tests

Use RED → GREEN → REFACTOR.

### T1 — condition aliases

Mixed/Combined and Novel/Third map deterministically to the configured internal labels.

### T2 — exact decomposition

For every complete profile:

```text
M == L + Z
sum(Z) == 0 within numerical tolerance
```

### T3 — offset invariance

Adding a constant to every transition of a probe profile changes \(\delta\) by that constant and leaves \(q\) unchanged.

### T4 — synthetic q recovery

Recover known \(q\) from noiseless and noisy synthetic A/B/probe profiles.

### T5 — synthetic delta recovery

Recover known additive \(\delta\) without biasing \(q\).

### T6 — weak A/B contrast flag

Near-zero centered A/B contrast produces a flagged, noninterpretable unit estimate rather than an infinite or unstable exported value.

### T7 — whole-profile canonicalization

Fast/Slow relabeling swaps all transitions together and never uses Mixed, Novel, or human-to-SRN similarity.

### T8 — fold firewall

No evaluated probe row contributes to basis estimation, hyperparameter selection, centering constants, or orientation rules for its held-out fold.

### T9 — level/shape scoring

A prediction with perfect shape but a constant offset has zero centered-shape error and nonzero raw error.

### T10 — history derivation

Toy sequences yield the expected previous-context, last-context, recency, and epoch values.

### T11 — history separation

Synthetic delta-only data favor `Hdelta`; synthetic q-only data favor `Hq`.

### T12 — default-source firewall

Novel rows cannot enter default-basis estimation or orientation.

### T13 — deterministic resampling

Fixed seeds reproduce bootstrap draws, fold assignments, and summary tables.

### T14 — source immutability

Input hashes are unchanged after the full pipeline runs.

---

## 27. Required outputs

```text
probe_diagnostics_config.json
input_manifest.csv
schema_audit.json
condition_aliases.json

human_condition_profiles.csv
srn_condition_profiles.csv

human_level_shape_decomposition.csv
srn_level_shape_decomposition.csv

human_probe_q_delta.csv
srn_probe_q_delta.csv
probe_q_delta_population_summary.csv

existing_model_raw_scores.csv
existing_model_centered_scores.csv
additive_cost_model_comparison.csv

history_identifiability.csv
history_confound_matrix.csv
history_branch_status.json
history_model_comparison.csv
history_parameter_summary.csv

default_basis_manifest.csv
novel_default_model_comparison.csv

human_srn_diagnostic_summary.csv
analysis_branch_status.json
```

Every tabular output must be tidy, contain explicit unit and condition columns, and include the config hash and input-manifest hash.

---

## 28. Required figures

### Figure 1 — raw versus centered condition profiles

For humans and SRNs, show A, B, Mixed, and Novel as:

```text
raw transition RT
centered transition shape
```

Do not average across SRN seeds until whole-profile orientation has been handled and individual-seed results are also visible.

### Figure 2 — global cost versus structured reuse

Primary diagnostic figure.

```text
x-axis: additive delta
y-axis: centered-profile variance explained or held-out profile correlation
```

Show separate points/distributions for Human Mixed, Human Novel, SRN Mixed, and SRN Novel.

### Figure 3 — q and delta distributions

Show participant- and seed-level estimates with uncertainty. Flag weak A/B contrast units.

### Figure 4 — model performance by score type

Compare existing models on:

```text
raw prediction
condition level
centered transition shape
```

### Figure 5 — history effects

If Stage 3 passes, show the estimated effect of history separately on \(q\) and \(\delta\), with observed support overlaid.

### Figure 6 — Novel shared/default comparison

If Stage 4 passes, compare the observed centered Novel profile with held-out predictions from the shared/default, A/B, combined, and low-rank models.

Every figure must have a machine-readable source table.

---

## 29. Statistical reporting

For every primary contrast, report:

```text
effect estimate
95% interval
number of participants or seeds
held-out prediction metric when applicable
analysis scale
identifiability status
```

Prefer effect sizes and predictive comparisons over binary significance claims.

Keep descriptive and predictive results labeled separately.

Any post hoc sensitivity analysis must be labeled and may not replace the preregistered primary output.

---

## 30. Stop conditions

Stop only the affected branch if:

```text
required inputs are missing
condition or transition mappings cannot be verified
existing predictions cannot be traced to fold-safe model artifacts
history predictors are not identifiable
no independent shared/default source exists
synthetic recovery tests fail
probe leakage is detected
```

Do not respond by:

```text
tuning thresholds against the desired result
adding transition-specific offsets to improve fit
redefining the default as an A/B midpoint
flipping individual transitions
changing the SRN architecture
training on Mixed or Novel
```

Each stopped branch must produce a status record explaining what was missing and what future data or artifact would make it testable.

---

## 31. Final report

Create:

```text
MECHANISM_FIRST_PROBE_DIAGNOSTICS_REPORT.md
```

Use this structure:

### 1. Provenance and audit

Which human and SRN artifacts were analyzed, and which branches were identifiable?

### 2. Level-versus-shape decomposition

How much of Mixed and Novel behavior is a global cost versus transition-specific structure?

### 3. Existing retrieval models after centering

Does the low-rank/A-B basis advantage survive when condition level is removed?

### 4. History-dependent retrieval

Does recent context change \(q\), \(\delta\), both, or neither?

### 5. Novel shared/default retrieval

Does Novel align with an independently estimated context-neutral basis?

### 6. Human–SRN comparison

Which computational properties match and which fail?

### 7. Next-model decision

Choose the single best-supported next target:

```text
A. separate global cost from the existing SRN readout
B. add a dynamic latent-context responsibility process
C. add an explicit compositional retrieval/gating mechanism
D. add a shared/default sequence state for Novel
E. revise the human target before further network work
```

The choice must be tied to the decision gate in Section 25, not to the preferred manuscript story.

---

## 32. Acceptance criteria

The PRD is complete when:

```text
one documented command reproduces the analysis
all required tests pass
all source artifacts are hash-recorded and unchanged
human and SRN level/shape decompositions are exported
Mixed and Novel have separate q/delta summaries
existing model predictions are scored on raw and centered outcomes
history is either modeled or explicitly ruled non-identifiable
Novel default retrieval is either tested with an independent basis or ruled non-identifiable
all primary figures have source-data tables
the final report selects one next SRN target using the preregistered decision gate
```

---

## 33. Scientific target

The desired outcome is not simply to make Mixed fall between A and B.

It is to determine which of the following is true:

\[
\boxed{
\begin{aligned}
&\text{learned transition structure is retrieved, but a global cost shifts RT;}\\
&\text{retrieval weights change with recent contextual history;}\\
&\text{Novel invokes an independent shared/default state;}\\
&\text{or the current models do not capture probe transition shape.}
\end{aligned}
}
\]

That distinction should determine the next model. Until it is resolved, further environmental or architectural tuning is premature.
