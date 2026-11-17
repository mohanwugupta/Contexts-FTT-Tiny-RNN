# PRD v3.1: Interpreting the Low-Rank Human Configuration Dimensions

## 1. Goal

The v3 human analysis found that the **low-rank configured-state model (M3)** was the best held-out model for both Mixed and Novel behavior. The next goal is to understand what those low-dimensional human configuration axes represent.

Primary question:

> **What low-dimensional property of the learned motor sequence is context reconfiguring?**

This project should determine whether the latent dimensions correspond to interpretable behavioral properties such as:

- generic transition difficulty;
- context sensitivity;
- switch vulnerability;
- learning / practice state;
- formal sequence structure;
- or some other reproducible low-dimensional organization.

Do not tune the SRN to match the human token profile. The SRN exact profile is a failed quantitative prediction; the present goal is to interpret the human structure on its own and then compare the human and SRN systems at a higher abstraction level.

---

## 2. Established inputs

Treat these v3 findings as fixed inputs:

```text
Context × Position effect detected
Pure A/B profile reliability passed
M3 low-rank model won Mixed
M3 low-rank model won Novel
Frozen SRN-informed M4 did not win
Mixed and Novel profiles were moderately correlated
```

Use the existing validated preprocessing, exclusions, transition indexing, counterbalance coding, and log-RT DV.

Do not reopen preprocessing decisions.

---

## 3. Main hypotheses

### H1 — Generic transition difficulty

One component may reflect how difficult each transition is independent of context.

Prediction:

\[
u_j(k) \leftrightarrow ar r(k)
\]

where:

\[
ar r(k)=rac{r_A(k)+r_B(k)}{2}.
\]

### H2 — Context sensitivity

A component may encode how differently the same transition is expressed in A versus B:

\[
\Delta(k)=r_A(k)-r_B(k).
\]

Prediction:

\[
u_j(k) \leftrightarrow \Delta(k).
\]

### H3 — Switch vulnerability

A component may encode which transitions are most disrupted by changing context.

Prediction:

\[
u_j(k) \leftrightarrow SwitchCost(k).
\]

### H4 — Learning state

A component may reflect transition-specific learning dynamics:

```text
early learning slope
late learning slope
initial cost
asymptote
post-switch recovery
```

### H5 — Formal sequence structure

Candidate preregistered sequence features:

```text
transition position
current key
next key
distance since current key last appeared
distance since next key last appeared
cyclic boundary transition
first-half vs second-half position
```

### H6 — Shared Mixed/Novel basis

Mixed and Novel may use the same low-dimensional A/B-derived basis but different coefficients:

\[
D_M = U\lambda_M
\]

\[
D_N = U\lambda_N.
\]

This would mean:

> same representational space, different retrieval location.

---

## 4. Core low-rank representation

For participant \(i\) and transition \(k\), estimate pure-context profiles:

\[
r_{A,i,k}, \qquad r_{B,i,k}.
\]

Define:

\[
\Delta_i(k)=r_{A,i,k}-r_{B,i,k}.
\]

Stack participant profiles into:

\[
X \in \mathbb R^{N	imes K}.
\]

Estimate low-rank structure from **training participants only**.

Primary method:

```text
PCA or probabilistic PCA
```

Robustness method:

```text
factor analysis
```

Do not use nonlinear embeddings as the headline result.

---

## 5. Selecting dimensionality

Evaluate:

```text
1 component
2 components
3 components
full-rank ceiling
```

Use all of:

```text
held-out reconstruction error
held-out Mixed/Novel prediction
parallel analysis
variance explained
```

Primary selection rule:

> Choose the smallest number of components whose held-out performance is within 1 SE of the best solution.

Do not select rank from variance explained alone.

Export:

```text
dimension_selection.csv
```

---

## 6. Component stability

PCA sign/order can change across folds.

Across participant folds and bootstraps:

1. estimate components;
2. match components using absolute cosine similarity + Hungarian assignment;
3. flip signs to match a reference solution;
4. calculate loading stability.

Report:

```text
fold-to-fold cosine
bootstrap mean loading
95% CI
sign consistency
```

Interpret a component only if its median matched cosine is at least:

```text
0.70
```

Otherwise label it unstable/descriptive.

---

## 7. Bootstrap procedure

Bootstrap participants.

For every bootstrap:

```text
re-estimate pure A/B profiles
re-estimate low-rank basis
align to reference basis
save transition loadings
```

Do not bootstrap transition positions as if they were independent participants.

---

## 8. Transition feature table

Create one frozen, preregistered table with one row per transition.

Required columns:

```text
transition_position
current_key
next_key
mean_pure_logRT
A_minus_B
error_rate
early_learning_slope
late_learning_slope
switch_cost
post_switch_recovery
distance_since_current_key
distance_since_next_key
cyclic_boundary
first_half_vs_second_half
```

If a required feature cannot be estimated reliably from the existing data, mark it missing and exclude that specific hypothesis.

Do not silently replace it with a post-hoc proxy.

---

## 9. Component-to-feature associations

For every stable component \(u_j(k)\), test preregistered feature associations.

Because there are few transitions, prefer:

```text
univariate correlations
very small theory-driven regressions
leave-one-transition-out prediction
```

Report:

```text
Pearson r
Spearman rho
LOTO prediction error
bootstrap CI
permutation p
```

Do not fit a high-dimensional regression with more predictors than transition positions can support.

---

## 10. Transition-label permutation tests

For each component-feature association:

1. permute transition labels;
2. recompute association;
3. build the null distribution.

Use:

```text
>= 10,000 permutations
```

where feasible.

Primary inference for transition-level semantic claims should use these permutation tests rather than standard large-sample p-values.

---

## 11. Critical control: generic difficulty

A low-rank component may simply rediscover baseline transition difficulty.

Test:

\[
u_j(k) \sim ar r(k).
\]

Then residualize:

\[
u_j^{resid}(k)
=
u_j(k)
-
\widehat{u_j(k)\mid ar r(k)}.
\]

Ask whether the residual component still predicts:

```text
Mixed midpoint deviation
Novel midpoint deviation
A/B contextual difference
switch cost
```

A strong configured-state interpretation requires structure beyond generic difficulty.

---

## 12. Mixed and Novel projection into the A/B basis

The basis \(U\) must be learned from pure A/B data only.

For Mixed:

\[
D_M(k)
=
RT_M(k)
-
rac{RT_A(k)+RT_B(k)}{2}.
\]

Estimate:

\[
D_M pprox U\lambda_M.
\]

For Novel:

\[
D_N pprox U\lambda_N.
\]

Probe outcomes may estimate only their coefficient vectors.

They may not change the basis.

---

## 13. Shared-basis versus separate-basis test

Compare:

### Shared basis

\[
D_M=U\lambda_M
\]

\[
D_N=U\lambda_N.
\]

### Separate bases

Mixed and Novel each get their own low-rank basis.

### Scalar baseline

One coefficient per probe.

Use participant-held-out predictive likelihood.

Primary question:

> Can one pure-A/B-derived configuration space explain both Mixed and Novel with only different locations in that space?

---

## 14. Compare Mixed and Novel coefficients

If the shared basis wins, compare:

\[
\lambda_M
\]

and:

\[
\lambda_N.
\]

Report:

```text
cosine similarity
component-wise coefficient difference
overall amplitude ratio
bootstrap CI
```

Interpretation:

- high cosine + different magnitude → same structure, different strength;
- different coefficient directions → same space but different configuration;
- separate basis wins → likely distinct mechanisms.

---

## 15. Temporal Mixed analysis

Where available, project separately:

```text
Mixed1
Mixed2
MixedTest
```

into the fixed A/B basis.

Estimate:

\[
\lambda_{M1},
\lambda_{M2},
\lambda_{MTest}.
\]

Test whether:

- the basis explains Mixed immediately;
- coefficients change after the first exposure;
- MixedTest returns to the same configuration.

This distinguishes immediate retrieval from probe-specific adaptation.

---

## 16. Participant-level structural prediction

Estimate participant A/B component scores:

\[
z_{i,j}.
\]

Then estimate participant Mixed / Novel component coefficients.

Test:

\[
\lambda_{M,i,j}
\sim
z_{i,j}
\]

and:

\[
\lambda_{N,i,j}
\sim
z_{i,j}.
\]

Use hierarchical shrinkage.

This asks whether people who express an A/B dimension strongly also show predictable probe behavior along that same dimension.

---

## 17. Switch-cost projection

Estimate participant/transition switch-cost profiles:

\[
SC_i(k).
\]

Project them into the same A/B-derived basis.

Ask whether:

\[
SC_i pprox U\gamma_i.
\]

If the same components explain:

```text
pure A/B differences
switch costs
Mixed/Novel configuration
```

that strongly supports a context-specific motor-state interpretation.

---

## 18. Learning-dynamics projection

Estimate transition-specific:

```text
early learning slope
late learning slope
initial RT
asymptotic RT
post-switch recovery
```

Ask whether these profiles align with the same low-rank basis.

Only assign component labels such as:

```text
learning-state axis
switch-vulnerability axis
```

if the associations are stable and replicate.

---

## 19. Cross-dataset replication

Estimate the pure A/B basis independently in:

```text
Combined / Mixed dataset
Novel / Third-context dataset
```

Before any alignment, fit the bases independently.

Then compare:

```text
component cosine similarity
subspace principal angles
loading correlations
shared-subspace variance
```

Primary question:

> Is the low-dimensional A/B configuration space reproducible across independent human datasets?

---

## 20. Joint cross-dataset model

Only after independent comparison, fit:

```text
shared basis
shared basis + dataset-specific residual
fully separate bases
```

Compare held-out likelihood.

Do not assume a shared basis just because both datasets use the same sequence.

---

## 21. Human–SRN comparison at the correct abstraction level

Do **not** tune the SRN exact token profile to match humans.

Instead compare:

### Effective dimensionality

```text
human rank
SRN A/B subspace rank
```

### Shared versus state-specific variance

How much A/B structure is shared across transitions versus transition-specific?

### Geometry

Compare low-dimensional subspaces using:

```text
principal angles
canonical correlation
variance-partition similarity
```

The question is:

> Did humans and the SRN discover the same *kind* of low-dimensional organization even though they did not learn the same exact token profile?

This is secondary to the human interpretation analyses.

---

## 22. Avoid circular component naming

During all analyses use:

```text
Component 1
Component 2
Component 3
```

Do not name a component “difficulty,” “learning,” or “switch sensitivity” until it satisfies all of:

```text
strong association
bootstrap stability
permutation support
replication where possible
```

If none survives, retain neutral names.

An uninterpreted but reproducible low-rank basis is a valid result.

---

## 23. Semantic recovery simulation

Extend the recovery work.

Simulate realistic datasets where the generating dimensions are:

```text
difficulty only
context sensitivity only
switch vulnerability only
difficulty + context sensitivity
random low-rank structure
```

Use the observed:

```text
participant counts
noise
A/B separation
missingness
transition count
```

Run the full interpretation pipeline.

Questions:

```text
Can true semantic axes be identified?
How often is a random axis falsely labeled?
Can difficulty be distinguished from context sensitivity?
```

Strong component labels are allowed only if semantic recovery is adequate.

---

## 24. Primary success criteria

### S1 — Stable basis

At least one retained component:

```text
median matched cosine >= .70
```

and improves held-out probe prediction.

### S2 — Beyond difficulty

At least one component retains relevant structure after controlling for generic transition difficulty.

### S3 — Interpretable association

At least one preregistered feature association:

```text
survives permutation
is bootstrap-stable
replicates across block/dataset where possible
```

### S4 — Shared Mixed/Novel basis

One A/B-derived basis predicts both probes better than independent scalar accounts.

### S5 — Cross-dataset reproducibility

The human A/B subspace replicates across Combined and Novel datasets.

---

## 25. Decision logic

### Result A — Components mainly encode transition difficulty

Narrow claim:

> M3 captures structured sequence difficulty rather than a distinct context-configuration mechanism.

### Result B — Context sensitivity survives difficulty control

Support:

> context reconfigures particular sequence states beyond their generic difficulty.

### Result C — Components align with switch costs / learning

Strong motor-learning interpretation:

> the latent space reflects how context organizes and destabilizes learned motor states.

### Result D — Stable low-rank basis but no interpretable feature

Still valid:

> context-specific retrieval is low-dimensional, but the dimensions remain behaviorally unidentified.

### Result E — Mixed and Novel share basis but different coefficients

Support:

> same representational space, different retrieval state.

### Result F — Mixed and Novel require different bases

Support distinct retrieval mechanisms.

### Result G — Human and SRN match at dimensionality/geometry level but not exact loadings

Strong abstraction-level computational bridge.

---

## 26. TDD requirements

All implementation follows:

\[
oxed{REDightarrow GREENightarrow REFACTOR}
\]

Required tests:

### T1 — Pure A/B basis isolation

No Mixed/Novel outcomes may enter basis estimation.

### T2 — Participant CV integrity

Held-out participants do not contribute to the basis.

### T3 — Component alignment

Synthetic sign/permutation changes are correctly realigned.

### T4 — Full-rank reconstruction

Full-rank decomposition reconstructs the original profiles within tolerance.

### T5 — Feature determinism

Formal transition features are exactly reproducible.

### T6 — Frozen primary feature table

Primary semantic features are locked before inferential testing.

### T7 — Permutation null

Null synthetic data produce approximately uniform permutation p-values.

### T8 — Difficulty control

A synthetic difficulty-only component loses semantic residual association after correct difficulty residualization.

### T9 — Probe basis freeze

Mixed/Novel can estimate coefficients but cannot alter \(U\).

### T10 — Cross-dataset independence

Each dataset basis is estimated independently before alignment.

### T11 — SRN freeze

Human results may not alter SRN seeds, weights, or token profiles.

### T12 — Semantic recovery gate

Semantic labels are blocked when simulated semantic recovery is inadequate.

---

## 27. Suggested code structure

```text
human_low_rank_interpretation/
    config.py

    data/
        load_v3_outputs.py
        transition_features.py
        learning_features.py
        switch_features.py

    decomposition/
        estimate_basis.py
        dimensionality.py
        bootstrap.py
        align_components.py
        participant_scores.py

    interpretation/
        difficulty.py
        context_sensitivity.py
        switch_vulnerability.py
        learning_dynamics.py
        sequence_structure.py
        permutation_tests.py

    probes/
        shared_basis.py
        mixed_coefficients.py
        novel_coefficients.py
        temporal_mixed.py

    replication/
        cross_dataset_basis.py
        shared_dataset_basis.py
        subspace_similarity.py

    srn/
        dimensionality_comparison.py
        shared_specific_variance.py
        subspace_comparison.py

    recovery/
        simulate_semantic_axes.py
        recover_semantics.py

    pipeline.py
```

---

## 28. Required outputs

```text
low_rank_run_config.json

dimension_selection.csv
component_loadings.csv
component_bootstrap.csv
component_stability.csv
participant_component_scores.csv

transition_feature_table.csv
component_feature_associations.csv
component_feature_permutations.csv
difficulty_residualized_results.csv

mixed_component_coefficients.csv
novel_component_coefficients.csv
shared_vs_separate_basis.csv
temporal_mixed_coefficients.csv

participant_structural_predictions.csv
switch_cost_component_alignment.csv
learning_component_alignment.csv

cross_dataset_component_alignment.csv
cross_dataset_subspace_similarity.csv

human_srn_dimensionality_comparison.csv
human_srn_shared_specific_variance.csv
human_srn_subspace_similarity.csv

semantic_recovery_matrix.csv
final_dimension_interpretation_summary.csv
```

---

## 29. Required figures

1. **Low-rank component loading profiles**
2. **Component bootstrap stability**
3. **Held-out error versus number of dimensions**
4. **Component versus transition difficulty**
5. **Component versus context sensitivity / switch cost**
6. **Difficulty-residualized component profile**
7. **Mixed and Novel coefficients in the shared basis**
8. **Mixed1 / Mixed2 / MixedTest coefficients**
9. **Cross-dataset subspace alignment**
10. **Human versus SRN abstraction-level comparison**
11. **Semantic recovery matrix**

---

## 30. Definition of done

- [ ] Dimensionality selected using held-out prediction.
- [ ] Component stability quantified across folds/bootstraps.
- [ ] Transition feature table frozen before semantic testing.
- [ ] Generic transition difficulty explicitly tested.
- [ ] Difficulty-residualized probe structure tested.
- [ ] Context-sensitivity association tested.
- [ ] Switch-cost association tested.
- [ ] Learning-dynamics association tested.
- [ ] Transition-label permutation tests completed.
- [ ] Mixed and Novel projected into the same pure-A/B-derived basis.
- [ ] Shared versus separate probe bases compared.
- [ ] Mixed temporal blocks projected separately.
- [ ] Participant-level structural predictions tested.
- [ ] Human low-rank structure replicated across datasets.
- [ ] Human–SRN comparison performed at the abstraction level without SRN retuning.
- [ ] Semantic recovery simulation completed.
- [ ] Components remain neutrally named if interpretation is not supported.

---

## 31. Target scientific claim

The strongest supported claim would be:

> **Human context effects on a learned motor sequence are organized in a small number of reproducible dimensions. These dimensions determine how ambiguous and novel contexts reconfigure sequence execution, and at least one dimension maps onto an independently measurable property of motor learning—such as contextual sensitivity or switch vulnerability—rather than generic transition difficulty.**

A stronger cross-system result would be:

> **Humans and High-Scale recurrent networks need not learn the same token-by-token geometry to instantiate the same computational principle: both organize context-specific sequence execution in a low-dimensional space containing shared and state-specific structure.**

Compactly:

\[
oxed{
	ext{What low-dimensional property of the sequence is context reconfiguring?}
}
