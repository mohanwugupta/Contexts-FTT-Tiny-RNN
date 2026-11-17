# PRD: State-Space Bridge from SRN Learning to Human Structured Retrieval

## Status

**Purpose:** implementation PRD for the final paper-level modeling bridge.

**Core principle:**

> Use state-space models to provide the intuitive functional bridge between humans and SRNs, then treat ambiguous/novel retrieval as a separate readout problem, and only then move to the transition-level low-rank/hidden-state analyses.

The intended conceptual ladder is:

\[
\boxed{
\text{scalar state-space learning}
\rightarrow
\text{probe retrieval}
\rightarrow
\text{vector / low-rank state-space}
\rightarrow
\text{SRN hidden-state implementation}
}
\]

or, in words:

1. **What latent skills were learned?**
2. **How does a probe context express those skills?**
3. **What internal structure does each skill contain across the sequence?**
4. **How can a recurrent network implement that structure causally?**

This PRD should make those levels explicit in code, figures, and outputs.

---

# 1. Scientific motivation

The prior PNAS analysis used a simple family of state-space models to bridge human learning curves and SRN behavior:

- **Lump:** one shared latent learning state;
- **Cross:** separate Maze and Melody states with partial cross-context coupling;
- **Split:** separate Maze and Melody states with no cross-talk.

That framework is easy to interpret because it asks a simple functional question:

> Did the learner acquire one shared skill or two context-specific skills?

The current paper asks a second question:

> Once two context-specific states exist, what happens when the retrieval context is Mixed or Novel?

Historical Rmd code attempted to make the state-space model answer both questions at once. That created circular or hand-imposed Mixed readouts. For example, historical implementations included:

- a free `w_Mixed` initialized at `.5` and bounded `[0,1]`;
- an explicit `w_mix = 0.5` rule for the split model;
- a version tying Mixed weighting to `A_cross`;
- inconsistent historical mappings in which one function used `w_mix = A_cross` while another used `w_mix = 0.5 + A_cross/2`.

These files are useful as references for the old latent-state/fatigue machinery, but the final paper pipeline must **not reuse their Mixed/Novel readout rules**.

The clean separation is:

\[
\boxed{
\text{State-space model = learning}
}
\]

and:

\[
\boxed{
\text{Retrieval model = probe expression}
}
\]

Mixed and Novel must never determine whether the underlying A/B learning system is classified as Lump, Cross, or Split.

---

# 2. Primary paper-level model hierarchy

The cleaned analysis should expose four levels.

## Level 1 — Scalar latent learning states

Use the PNAS-compatible state-space family to infer:

\[
x_A(t),\qquad x_B(t)
\]

or one shared state \(x(t)\).

This is fitted using **pure A/B learning observations only**.

Question:

> Does behavior require one shared skill state, partially interacting A/B states, or fully separate A/B states?

---

## Level 2 — Scalar retrieval of frozen states

Given frozen A/B states, ask how a probe is expressed.

Simple scalar retrieval:

\[
\hat y_P(t)
=
q_P \hat y_A(t)
+
(1-q_P)\hat y_B(t)
\]

where \(P\) is Mixed or Novel.

The parameter \(q_P\) belongs to the **retrieval model**, not the state-space learning model.

This level asks:

> Can one global A/B responsibility explain probe behavior?

---

## Level 3 — Vector / low-rank sequence states

At the transition level, define context-specific sequence profiles:

\[
\mathbf x_A =
[x_{A,1},...,x_{A,7}]
\]

\[
\mathbf x_B =
[x_{B,1},...,x_{B,7}].
\]

A scalar mixture predicts:

\[
\mathbf x_P
=
q_P\mathbf x_A+(1-q_P)\mathbf x_B.
\]

The configured-state account instead allows the probe to vary within a low-dimensional sequence space:

\[
\mathbf x_P
=
\mathbf m
+
U\boldsymbol\lambda_P,
\]

where:

\[
\mathbf m = \frac{\mathbf x_A+\mathbf x_B}{2}.
\]

This should be introduced in code/documentation as a **low-rank vector extension of the state-space intuition**, not as unrelated PCA machinery.

---

## Level 4 — SRN representational implementation

For the SRN:

\[
h_{A,k}\neq h_{B,k}
\]

for the same sequence state \(k\).

The existing causal analyses then ask whether replacing the naturally occurring A/B component of:

\[
h_{P,k}
\]

changes execution.

This level asks:

> How could the functional context-specific skill states inferred behaviorally be implemented inside a recurrent network?

---

# 3. Critical non-goals

Do **not**:

- fit Mixed/Novel trials when estimating Lump/Cross/Split learning dynamics;
- add `w_Mixed` to the state-space learning model;
- hard-code `Mixed = 0.5 A + 0.5 B`;
- derive Mixed weighting from `A_cross`;
- use `A_cross` as both a learning-coupling parameter and retrieval parameter;
- let probe RTs update A/B state-space parameters;
- force the corrected High-Scale SRN to select Cross or Split;
- tune the state-space model to agree with hidden-state PCA;
- tune the SRN to human Mixed/Novel behavior;
- tune the SRN to the human low-rank transition profile;
- fit state-space models to SRN sequence CE and interpret that as motor-expression splitting;
- erase the negative result if corrected SRN RT is best fit by Lump.

The state-space bridge is a test, not a target.

---

# 4. Inputs and dependencies

## Human data

Use the validated human pipelines and exclusions already frozen for:

```text
Combined / Mixed dataset
Novel / Third-context dataset
```

Preserve:

```text
participant exclusions
warmup exclusions
first-sequence exclusion
counterbalance labels
chronological block structure
transition indexing
log-RT primary DV for transition-level models
```

For the scalar PNAS-style state-space bridge, use the same sequence-level RT definition used in the prior PNAS state-space analysis unless there is a documented reason it cannot be reproduced.

Do not silently replace the old sequence-level dependent variable with the newer transition-level DV.

---

## SRN data

The bridge analysis must use the **corrected SRN rerun**, not the current scale `.5` run.

Required canonical conditions:

```text
High scale = 2.0 + execution
Low scale + execution
High scale = 2.0, sequence only
```

Primary state-space modeling applies to:

```text
High scale = 2.0 + execution normalized RT
Low scale + execution normalized RT
```

The sequence-only condition has no emergent RT and therefore is not a primary RT state-space condition.

The corrected execution model must first satisfy the scientific sanity check:

```text
sequence CE decreases with learning
correct-token confidence increases
execution speed increases
normalized RT decreases
```

Transient switch-related RT increases are allowed.

If the corrected execution-control run fails this gate, stop and report rather than fitting the state-space bridge.

---

# 5. Stage 0 — Audit the historical state-space code

Before implementing the final bridge, create:

```text
docs/HISTORICAL_STATE_SPACE_AUDIT.md
```

Audit at minimum:

```text
state-space-trial-V4-5-generative-fits.Rmd
state-space-trial-V5-adjusted.Rmd
```

Document:

### Reusable machinery

Identify the final/most coherent historical implementations of:

```text
latent performance-state simulation
fatigue accumulation
fatigue recovery
linear decoder
counterbalance handling
RMSE objective
Gaussian likelihood
BIC
Lump/Cross/Split topology
```

### Historical Mixed logic to reject

Explicitly identify every occurrence of:

```text
w_Mixed
w_mix
use_cross_for_mixed
A_cross-derived Mixed weighting
strict 50/50 Mixed weighting
NoContext/Mixed shared weighting
```

Include a table:

| File | Function | Historical behavior | Final status |
|---|---|---|---|
| ... | ... | free `w_Mixed` | deprecated |
| ... | ... | `w_mix=.5` | deprecated |
| ... | ... | `w_mix=A_cross` | deprecated |
| ... | ... | `w_mix=.5+A_cross/2` | inconsistent/deprecated |

The historical code is not itself a scientific result.

Do not modify or delete these Rmds.

---

# 6. Stage 1 — Implement one canonical scalar state-space engine

Create one implementation shared by human and SRN data.

Suggested module:

```text
src/state_space/
    dynamics.py
    decoders.py
    fit.py
    comparison.py
    simulation.py
```

The implementation should reproduce the **final PNAS topology and nuisance structure** as closely as possible.

Use the prior PNAS model as the canonical specification, not the exploratory Mixed modifications in the two Rmds.

---

# 7. Common state-space model family

## M-SS1 — Lump

One shared performance state:

\[
x(t).
\]

All pure A/B practice contributes to the same latent learning process.

Use the same fatigue/recovery and decoder conventions as the PNAS model.

Interpretation:

> A and B are behaviorally one shared skill.

---

## M-SS2 — Cross

Two latent states:

\[
\mathbf x(t)=
\begin{bmatrix}
x_A(t)\\
x_B(t)
\end{bmatrix}
\]

with coupling matrix:

\[
A =
\begin{bmatrix}
A_{\mathrm{diag}} & A_{\mathrm{cross}}\\
A_{\mathrm{cross}} & A_{\mathrm{diag}}
\end{bmatrix}.
\]

Reuse the exact final PNAS update rule surrounding this matrix.

Interpretation:

> A and B are distinguishable latent skill states but practice partially transfers between them.

As:

\[
A_{\mathrm{cross}}\rightarrow 0,
\]

the model approaches the Split solution.

---

## M-SS3 — Split

Two context-specific latent performance states with:

\[
A_{\mathrm{cross}}=0.
\]

A practice updates the A stream according to the canonical PNAS dynamics; B practice updates B.

Interpretation:

> A and B behave as separate latent skills.

---

# 8. Fatigue and decoder

Preserve the PNAS distinction between:

```text
latent skill/performance change
within-block fatigue / recovery
```

The canonical implementation should include:

- context-specific fatigue accumulation where supported;
- exponential recovery across breaks;
- linear decoding from latent skill + fatigue to observed output;
- separate A/B decoder terms where that is part of the final PNAS implementation.

For each candidate dynamic parameter setting:

1. simulate latent state trajectories;
2. fit the linear decoder by OLS;
3. generate predictions;
4. calculate residual error / likelihood;
5. calculate BIC.

Do not give Mixed or Novel a decoder during Stage 1.

---

# 9. Stage 1 data contract: pure A/B only

This is a hard scientific contract.

The scalar state-space fitter receives only observations whose true training context is:

```text
A
B
```

Probe outcomes:

```text
Mixed
Novel
Default / No Context
```

must never contribute to:

```text
dynamic parameter fitting
decoder fitting
BIC
state-space model selection
```

## Primary human fit

To avoid assumptions about how probe practice changes latent skills, the headline learning-model fit should use the **pure A/B training period before first probe exposure** where the schedule permits.

This provides the cleanest estimate of what was learned before ambiguous retrieval.

## Secondary robustness fit

Optionally fit all pure A/B observations while:

- treating probe observations as missing outcomes;
- applying **no probe-driven A/B skill update**;
- preserving elapsed breaks/time for fatigue recovery only if the canonical dynamics require it.

Label this explicitly as a robustness analysis because “no probe learning” is an assumption.

Do not allow the secondary analysis to replace the clean pre-probe fit if they disagree.

---

# 10. Same model code for humans and SRNs

Human and SRN fits must use the same state-space engine and model topology.

Only the observed-output mapping may differ in scale.

Human observable:

```text
sequence RT
```

SRN observable:

```text
emergent normalized RT = 1 - execution_speed
```

Do not fit state-space models to SRN CE as the primary bridge.

Sequence CE remains a learning diagnostic.

This common model family is the functional bridge.

---

# 11. Primary state-space outputs

For each dataset/condition export:

```text
state_space_model_fits.csv
state_space_parameters.csv
state_space_predictions.csv
state_space_latent_states.csv
state_space_decoders.csv
state_space_bic_comparison.csv
```

Required columns include:

```text
agent_type          # human / SRN
dataset
condition
counterbalance
seed_or_group
model               # Lump / Cross / Split
parameter
value
BIC
delta_BIC
RMSE
R2
n_observations
fit_scope            # pre_probe_AB / all_AB_robustness
```

---

# 12. State-space recovery simulation

Before interpretation, prove that the current schedule can identify the three learning models.

Generate synthetic data from:

```text
Lump
Cross
Split
```

using:

```text
actual A/B block schedule
realistic noise
realistic fatigue
realistic decoder coefficients
parameter ranges matching observed data
```

Fit all three models.

Run at least:

```text
100 simulations per generating model
```

if computationally reasonable.

Primary recovery metrics:

```text
confusion matrix
selection accuracy
A_cross recovery
parameter bias
```

Interpret a model winner strongly only if generating-model recovery is acceptable.

Suggested criterion:

```text
>= 80% correct class recovery
```

for the relevant empirical regime.

---

# 13. Stage 1 figure: PNAS-style functional bridge

Create one paper-ready figure with matched human/SRN visual logic.

## Panel A — Human A/B learning

For each counterbalance:

```text
observed sequence RT
Lump fit
Cross fit
Split fit
```

Overlay predictions on the actual learning trajectory.

Visually distinguish A and B blocks.

## Panel B — Corrected High-scale SRN emergent RT

Same visual grammar:

```text
observed normalized RT
Lump
Cross
Split
```

## Panel C — Low-scale SRN emergent RT

Same.

## Panel D — BIC comparison

Heatmap/table:

```text
Human
High-scale + execution
Low-scale + execution
```

columns:

```text
Lump
Cross
Split
```

show:

```text
Delta BIC relative to winner
```

## Panel E — fitted cross-talk

Where Cross wins or remains competitive, show:

\[
A_{\mathrm{cross}}.
\]

Do not interpret a cross-talk parameter from a model that is decisively rejected.

---

# 14. Decision logic for Stage 1

## Ideal convergence

If:

```text
Humans -> Cross/Split
High-scale SRN -> Cross/Split
Low-scale SRN -> Lump
```

then state:

> The same functional latent-state model that describes human context-specific learning also emerges in the execution dynamics of the flexible SRN.

## SRN RT remains Lump despite hidden splitting

Do **not** fix this.

Interpret:

> Hidden representational splitting in the SRN is not fully expressed in ordinary A/B execution trajectories and becomes behaviorally visible primarily through ambiguous probes or interventions.

This is scientifically meaningful.

## Human state-space fit differs from previous PNAS result

Stop and audit before proceeding.

Possible causes:

```text
different participants
different preprocessing
different time window
different sequence exclusion
different decoder
different state-space implementation
```

Do not silently overwrite the old interpretation.

---

# 15. Stage 2 — Freeze the latent A/B endpoints

After state-space model selection, freeze the A/B learning model.

For every probe time \(t\), generate counterfactual pure-context predictions:

\[
\hat y_A(t)
\]

and:

\[
\hat y_B(t).
\]

These represent:

> What would performance look like if the currently estimated A-specific versus B-specific skill were expressed?

The counterfactual endpoints must be generated without using the observed probe RT.

Export:

```text
frozen_AB_counterfactuals.csv
```

with:

```text
participant_or_seed
probe_epoch
sequence
A_prediction
B_prediction
A_state
B_state
A_fatigue
B_fatigue
endpoint_separation
reliable_endpoint
```

Use the established reliability rule for ill-conditioned A/B endpoints or define a preregistered equivalent.

---

# 16. Stage 2 — Probe retrieval is a separate model family

Do not modify the Stage-1 state-space dynamics.

Use the frozen A/B endpoints as inputs to retrieval models.

The primary conceptual ladder is:

## R0 — Categorical selection

Each probe sequence comes from either the A or B endpoint.

This is the winner-take-all hypothesis.

---

## R1 — Scalar mixture

One global responsibility:

\[
\hat y_P(t)
=
q_P\hat y_A(t)
+
(1-q_P)\hat y_B(t).
\]

For participant-level models, allow hierarchical participant variation if already established.

One \(q\) applies across the sequence.

---

## R2 — Transition-specific mixture

At transition \(k\):

\[
q_{P,k}.
\]

Use hierarchical shrinkage.

This is a flexibility control, not the preferred theoretical model.

---

## R3 — Low-rank configured-state retrieval

Use frozen pure A/B transition profiles.

Define:

\[
\mathbf m =
\frac{\mathbf x_A+\mathbf x_B}{2}.
\]

Model probe structure as:

\[
\mathbf x_P
=
\mathbf m
+
U\boldsymbol\lambda_P.
\]

Use the already validated low-rank pipeline and rank-selection logic.

The current frozen human result selected rank 2; the implementation must reproduce this result from the frozen data without hard-coding rank 2 into the estimator.

---

## R4 — Frozen SRN-informed retrieval

Use the frozen transition profile from the corrected final SRN run.

Human outcomes may calibrate only the prespecified scalar terms.

Do not select SRN seeds or profile shape using human behavior.

---

# 17. Retrieval evaluation

Use:

```text
held-out log likelihood
RMSE
MAE
```

Primary:

```text
held-out log likelihood
```

Preserve:

```text
participant train/test separation
A/B endpoint freezing
no human->SRN tuning
model recovery
```

The state-space bridge does not replace the existing human configured-state comparison.

It reorganizes it conceptually.

---

# 18. Stage 2 figure: from latent states to probe readout

Create a simple conceptual + empirical figure.

## Panel A — Scalar latent-state cartoon

Show:

```text
x_A
x_B
```

as frozen learned endpoints.

Then:

```text
Mixed / Novel observation
```

relative to them.

No PCA yet.

## Panel B — Categorical vs continuous

Show the existing continuous-versus-categorical evidence in intuitive form.

## Panel C — Scalar mixture overlay

Observed probe RT versus:

\[
qA+(1-q)B.
\]

## Panel D — failure of one global scalar

Show transition-level residuals / midpoint deviations demonstrating that one q does not fully explain probe behavior.

This figure is the bridge from classic state-space intuition into vector states.

---

# 19. Stage 3 — Reframe transition modeling as vector state-space

Documentation and figures should explicitly teach this progression.

## Scalar skill

Old model:

\[
x_A,\quad x_B.
\]

Each context-specific skill is represented by one latent performance variable.

## Vector skill

New observation:

\[
\mathbf x_A=
[x_{A,1},...,x_{A,7}]
\]

\[
\mathbf x_B=
[x_{B,1},...,x_{B,7}].
\]

The same key/transition identity at different ordinal sequence positions remains position-specific.

Do not collapse repeated physical transitions across ordinal positions.

## Low-rank vector skill

Rather than estimating seven unrelated dimensions:

\[
\mathbf x
\approx
\boldsymbol\mu+
U\boldsymbol\lambda.
\]

The low-rank result means that the internal transition-level structure can be summarized with a small number of coordinates.

The code comments and README should use this interpretation.

---

# 20. Stage 3 human data contract

Use the validated transition-level observations:

```text
participant
context
block
sequence repetition
ordinal transition position
log RT
```

Preserve:

```text
transition_position = 1...7
```

independently of physical key identity.

For example, repeated physical transitions that occur at different positions remain separate sequence states.

Pure A/B transition profiles must be estimated without Mixed/Novel outcomes.

---

# 21. Stage 3 outputs

Export:

```text
human_vector_AB_profiles.csv
human_probe_midpoint_deviation.csv
human_low_rank_basis.csv
human_low_rank_scores.csv
human_low_rank_predictions.csv
human_low_rank_rank_selection.csv
human_low_rank_bootstrap_stability.csv
```

Preserve current cross-dataset replication analyses.

Do not expand the semantic-component analysis further unless required to reproduce the existing result.

The paper-level claim is:

> the structure is stable and low-dimensional,

not:

> Component 1 or 2 has a uniquely identified psychological meaning.

---

# 22. Stage 3 figure: scalar state -> vector state -> low-rank state

Create one didactic figure.

## Panel A

Scalar state-space:

\[
x_A,\ x_B.
\]

## Panel B

Transition profile:

\[
\mathbf x_A,\ \mathbf x_B.
\]

## Panel C

Scalar probe mixture:

\[
q\mathbf x_A+(1-q)\mathbf x_B.
\]

## Panel D

Observed probe departures from scalar mixture.

## Panel E

Low-rank reconstruction:

\[
\mathbf m+\lambda_1u_1+\lambda_2u_2.
\]

The figure should make the low-rank analysis feel like a natural extension of the state-space model rather than a separate analysis family.

---

# 23. Stage 4 — Link functional vector states to SRN hidden states

Do not refit the SRN for this stage.

Use the corrected frozen SRN.

The conceptual correspondence is:

Behavioral transition state:

\[
x_{A,k},x_{B,k}
\]

Network hidden state:

\[
h_{A,k},h_{B,k}.
\]

The SRN analyses should include:

```text
A/B hidden separation
shared + token-specific A/B structure
natural token-matched patching
Mixed<-A / Mixed<-B
correct-token vs wrong-token patch
random orthogonal controls
sequence-preservation diagnostics
```

This is an implementation-level analysis.

It should not be used to fit the human state-space model.

---

# 24. Human–SRN bridge analyses

Implement three explicitly distinct comparisons.

## Bridge A — Functional learning architecture

Compare state-space model selection:

```text
Human Lump/Cross/Split
SRN Lump/Cross/Split
```

This is the most intuitive bridge.

---

## Bridge B — Probe computation

Compare qualitative/functional properties:

```text
continuous vs categorical expression
global scalar vs structured probe expression
```

Do not require matching exact q values.

---

## Bridge C — Representational organization

Compare:

```text
effective dimensionality
shared vs state-specific structure
```

Retain the current negative result that exact human and SRN transition profiles do not match.

Do not combine A, B, and C into one “match score.”

---

# 25. Paper-level composite figure

Create an optional manuscript figure called:

```text
figure_bridge_state_space_to_geometry.png
```

Suggested vertical layout:

### Top — Functional learning

```text
A/B observations
      ↓
Lump / Cross / Split state-space
      ↓
frozen x_A, x_B
```

### Middle — Retrieval

```text
Mixed / Novel
      ↓
scalar mixture fails
      ↓
low-rank vector configuration
```

### Bottom — Network implementation

```text
h_A,k / h_B,k
      ↓
context-configured hidden state
      ↓
execution
```

Label the three levels:

```text
Functional state
Retrieval computation
Representational implementation
```

---

# 26. PNAS-style model overlays

The final repository should contain PNAS-style observed/predicted overlays for both model layers.

## Learning overlays

For Lump/Cross/Split:

```text
observed A/B sequence RT
predicted A/B sequence RT
```

for humans and corrected SRNs.

## Retrieval overlays

For:

```text
R1 scalar
R2 token-specific
R3 low-rank
R4 frozen SRN
```

show:

```text
observed transition-level probe RT
held-out predicted transition-level RT
```

for Mixed and Novel.

Also make observed-versus-predicted scatter plots.

This is intended to make the model comparison visually understandable before readers encounter BIC/held-out likelihood tables.

---

# 27. Important analysis separation

Use different names for the two model families.

Do **not** call both of them M1/M2/M3 in code.

Recommended:

```text
SS_LUMP
SS_CROSS
SS_SPLIT
```

for state-space learning.

And:

```text
RET_CATEGORICAL
RET_SCALAR
RET_TOKEN
RET_LOWRANK
RET_SRN
```

for probe retrieval.

This prevents the current manuscript/model nomenclature from becoming confusing.

---

# 28. Reproducibility tests

Follow:

\[
\boxed{
RED \rightarrow GREEN \rightarrow REFACTOR
}
\]

## T1 — Historical Mixed logic excluded

Search canonical state-space code for:

```text
w_Mixed
w_mix
use_cross_for_mixed
```

The new state-space module should contain none of these.

---

## T2 — Probe leakage

Fit state-space models.

Then arbitrarily permute all Mixed/Novel RTs.

Require exactly identical:

```text
state-space parameters
A/B latent trajectories
decoders
BIC
model winner
```

---

## T3 — Probe removal

Delete all probe rows entirely.

Pure A/B state-space results must remain identical to the fit that retained probes as ignored rows for the same primary fitting window.

---

## T4 — Synthetic model recovery

Generate Lump/Cross/Split data.

Require model recovery above configured threshold.

---

## T5 — Cross-talk boundary

Synthetic:

```text
A_cross = 0
```

must behave as Split within numerical tolerance.

Increasing positive coupling must increase cross-context influence.

---

## T6 — Same engine human/SRN

Human and SRN state-space fitting must call the same core simulation/fitting implementation.

No duplicated “SRN version” of the model.

---

## T7 — Corrected SRN only

The state-space bridge must reject a final analysis config pointing to the old High-scale `.5` run.

Require metadata:

```text
high_scale = 2.0
execution_sanity_gate = passed
```

---

## T8 — RT semantics

For SRN:

```text
normalized_RT == 1 - execution_speed
```

within numerical tolerance.

---

## T9 — Transition-position preservation

The vector analysis must treat ordinal transition positions separately even when physical key transitions repeat.

---

## T10 — A/B vector leakage

Changing Mixed/Novel outcomes cannot change:

```text
human A profile
human B profile
low-rank basis learned from pure A/B
```

except where an explicitly probe-trained basis is being tested as a labeled non-primary analysis.

---

## T11 — Rank regression

On the frozen current human dataset/config, the cleaned low-rank implementation should reproduce the current selected rank:

```text
rank = 2
```

This is a regression test, not a hard-coded selection rule.

---

## T12 — Frozen SRN human comparison

Changing human outcomes may not alter:

```text
SRN seeds
SRN checkpoint
SRN weights
SRN transition profile
```

---

# 29. Recovery for retrieval models

Preserve or rerun recovery for:

```text
RET_CATEGORICAL
RET_SCALAR
RET_TOKEN
RET_LOWRANK
RET_SRN
```

using the actual empirical schedule/noise.

At minimum ensure that scalar and low-rank generating regimes are distinguishable under realistic conditions.

The state-space recovery and retrieval recovery are separate matrices.

Export both.

---

# 30. Suggested repository structure

```text
src/
    state_space/
        dynamics.py
        fatigue.py
        decoder.py
        fit.py
        recovery.py
        plotting.py

    retrieval/
        endpoints.py
        categorical.py
        scalar.py
        token_specific.py
        low_rank.py
        srn_informed.py
        comparison.py
        recovery.py

    bridge/
        human_sequence_level.py
        srn_sequence_level.py
        vector_extension.py
        cross_system.py

scripts/
    audit_historical_state_space.py
    run_state_space_bridge.py
    run_retrieval_bridge.py
    run_full_bridge.py
    make_bridge_figures.py

configs/
    state_space_bridge_final.json
    retrieval_bridge_final.json

outputs/
    bridge_final/
        state_space/
        retrieval/
        vector/
        srn/
        figures/

docs/
    HISTORICAL_STATE_SPACE_AUDIT.md
    STATE_SPACE_BRIDGE_README.md
    BRIDGE_RESULT_MAP.md
    BRIDGE_REPORT.md
```

Do not force this structure if the repository already has a better package organization.

The scientific separation between `state_space/` and `retrieval/` is mandatory.

---

# 31. Required final tables

```text
historical_state_space_audit.csv

state_space_model_fits.csv
state_space_parameters.csv
state_space_bic_comparison.csv
state_space_predictions.csv
state_space_latent_states.csv
state_space_recovery.csv

frozen_AB_counterfactuals.csv

retrieval_model_comparison.csv
retrieval_predictions.csv
retrieval_recovery.csv

human_vector_AB_profiles.csv
human_low_rank_rank_selection.csv
human_low_rank_basis.csv
human_low_rank_predictions.csv

cross_system_bridge_summary.csv
```

---

# 32. Required final figures

## Figure SSB1 — PNAS-style state-space learning overlays

Human + High-scale corrected SRN + Low-scale corrected SRN.

---

## Figure SSB2 — State-space BIC bridge

Lump/Cross/Split Delta-BIC heatmap plus cross-talk parameter where appropriate.

---

## Figure SSB3 — Frozen A/B states and probe observations

Simple scalar state-space intuition.

---

## Figure SSB4 — Retrieval model overlays

Mixed and Novel:

```text
scalar
token-specific
low-rank
frozen SRN
```

Observed + held-out predictions.

---

## Figure SSB5 — Scalar-to-vector-state explanation

Didactic transition from:

```text
x_A, x_B
```

to:

```text
vector x_A, vector x_B
```

to rank-2 configuration.

---

## Figure SSB6 — Functional-to-representational bridge

State-space functional level -> low-rank retrieval -> SRN hidden-state intervention.

---

# 33. Required reports

## `HISTORICAL_STATE_SPACE_AUDIT.md`

Explain what was inherited and what was rejected from the two Rmds.

---

## `STATE_SPACE_BRIDGE_README.md`

Explain the model in plain language.

The first section should answer:

> What is a “state” here?

Then:

> Why is Mixed not part of the state-space learning model?

Then:

> Why does the low-rank analysis count as a vector extension of the same intuition?

---

## `BRIDGE_RESULT_MAP.md`

For every manuscript claim list:

```text
claim
analysis level
script
input
output
figure
statistic
interpretive boundary
```

---

## `BRIDGE_REPORT.md`

Final concise result summary.

Use the structure:

### 1. Functional learning architecture

What did Lump/Cross/Split say for:

```text
humans
High-scale corrected SRN
Low-scale corrected SRN
```

### 2. Probe retrieval

What did categorical/scalar/low-rank models say?

### 3. Vector-state structure

Did rank-2 replicate?

### 4. SRN implementation

What causal hidden-state result remains supported after the corrected scale-2 rerun?

### 5. Cross-system conclusion

Choose one:

```text
A. strong functional + representational convergence
B. functional convergence, representational divergence
C. retrieval convergence without matching learning dynamics
D. limited cross-system convergence
```

Do not choose based on desired paper story.

---

# 34. Paper-level decision matrix

## Result A

```text
Human: Split/Cross
High-scale SRN: Split/Cross
Low-scale SRN: Lump
Human probe: Low-rank
SRN probe: structured / causal
```

Interpretation:

> Strong bridge. Flexible SRNs and humans share context-specific functional learning dynamics, while the SRN exposes a plausible representational implementation.

---

## Result B

```text
Human: Split/Cross
High-scale SRN RT: Lump
SRN hidden states: strongly split
```

Interpretation:

> Representational splitting can be behaviorally silent during ordinary A/B execution. The probe/intervention analyses reveal structure not recoverable from scalar learning curves.

This is still publishable and conceptually interesting.

---

## Result C

```text
Human and SRN learning architecture differ
Probe computation converges
```

Interpretation:

> The model is an existence proof for retrieval computation, not a full process model of human acquisition.

---

## Result D

Low-rank human result fails to reproduce after integration.

Stop and audit.

Do not continue manuscript integration until discrepancy is resolved.

---

# 35. Definition of done

- [ ] Historical Rmds audited.
- [ ] Old Mixed weighting logic documented and excluded from canonical code.
- [ ] Corrected High-scale=2 SRN output is the only SRN used by final bridge.
- [ ] Execution sanity gate passes before state-space fitting.
- [ ] One canonical Lump/Cross/Split engine is implemented.
- [ ] Human and SRN state-space models use the same engine.
- [ ] Primary state-space fits use pure A/B pre-probe data only.
- [ ] Mixed/Novel outcomes cannot affect learning-model parameters or BIC.
- [ ] State-space recovery passes or limitations are reported.
- [ ] PNAS-style human/SRN state-space overlays are generated.
- [ ] Frozen A/B counterfactual endpoints are exported.
- [ ] Probe retrieval models are implemented as a separate family.
- [ ] No `w_Mixed` exists in the state-space learning module.
- [ ] Current low-rank human result is reproduced without hard-coding rank.
- [ ] Transition positions remain ordinal-context-specific sequence states.
- [ ] Retrieval model overlays are generated.
- [ ] SRN causal analyses consume the corrected frozen network.
- [ ] Exact human–SRN mismatch remains visible if it remains present.
- [ ] State-space and retrieval recovery matrices are separate.
- [ ] Final bridge report states the strongest supported convergence level without forcing the paper story.

---

# 36. Target scientific narrative

The analysis should make it possible to write the paper in this order:

### Step 1 — Functional learning

> Both humans and SRNs can be described in the intuitive language of latent skill states. The state-space model asks whether A and B behave as one shared skill or as context-specific skills.

### Step 2 — Retrieval

> Once A/B skill states are estimated and frozen, Mixed and Novel ask a separate question: how are those learned states expressed when context is ambiguous or unfamiliar?

### Step 3 — Structured sequence state

> A single scalar A/B responsibility does not fully explain human probe behavior. Context configures a structured transition-level state, but that structure is low-dimensional.

### Step 4 — Implementation

> In the SRN we can open the system and observe a candidate implementation: context changes the recurrent representation of the currently active sequence state, and causally transplanting those naturally learned components changes execution.

The final conceptual summary is:

\[
\boxed{
\underbrace{\text{State-space model}}_{\text{What skills were learned?}}
\rightarrow
\underbrace{\text{Retrieval model}}_{\text{How are they expressed?}}
\rightarrow
\underbrace{\text{Low-rank vector state}}_{\text{What structure is expressed?}}
\rightarrow
\underbrace{\text{SRN geometry}}_{\text{How can it be implemented?}}
}
\]

That hierarchy—not a hand-imposed Mixed blend—is the intended bridge between the human and SRN results.
