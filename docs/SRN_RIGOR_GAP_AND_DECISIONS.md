# SRN rigor gap analysis and decisions needed

This document compares the paper-facing SRN workflow in this repository with
the SRN workflow in `../SRN/CogSci`. It is a design audit, not an implementation
change. No SRN model, configuration, checkpoint, or frozen result has been
modified.

## Bottom line

The CogSci directory should be treated as a source of useful robustness
analyses, not as a model specification to copy verbatim. The present repository
is already stronger in several areas that matter for reproducibility: it has a
frozen configuration contract, 20 explicitly indexed seeds per condition,
checkpoint hashes, no-probe-update checks, causal negative controls, automated
tests, and bitwise verification of final CSVs. The CogSci workflow is broader
in the questions it asks about the networks, especially performance matching,
training to asymptote, hybrid initialization controls, and hyperparameter
sweeps.

The defensible next step is therefore to keep the current confirmatory model
frozen and add a separately registered robustness layer. Replacing the current
architecture with the CogSci architecture would alter the scientific question
and invalidate direct comparison with the existing causal and human analyses.

## What the two workflows actually model

| Feature | Current paper SRN | CogSci SRN | Consequence |
|---|---|---|---|
| Token input | Strict 4-D one-hot | Gaussian-smoothed token code, sigma=.7 | Changing this is a representational intervention, not a minor refactor. |
| Context input | 3-D A/B/Novel basis; Novel column fixed at zero | 2-D Maze/Melody basis | The current design uniquely supports a neutral third-context probe. |
| Hidden layer | 8-unit tanh Elman RNN | 6-unit tanh Elman RNN | Hidden size should be a sensitivity factor, not silently replaced. |
| Outputs | Next-token head plus execution-speed head | Next-token head only | CogSci cannot substitute for the current execution mechanism. |
| Main initialization contrast | SD .50 versus .02 | SD 2.0 versus .02 | The labels High/Rich are not numerically equivalent. |
| Training schedule | Fixed A10/B10 sequence blocks, 170 updates | Participant-specific empirical timelines, then 500 additional passes | These answer exposure-controlled versus participant-history questions. |
| Replication | 20 seeds in every primary condition | One seed per participant; same seed paired across conditions | CogSci jointly varies initialization and participant history across participants. |
| Checkpoint comparisons | Fixed exposure at sequences 20/40/80/170 | Fixed steps, CE/accuracy thresholds, and asymptote | CogSci better separates exposure from performance. |
| Causal analysis | Local A/B patches, ablations, wrong-position and random controls | Context perturbation/ablation plus geometry diagnostics | The current causal specificity tests are more tailored to the paper claim. |
| Reproducibility | Tests, frozen manifests, checkpoint hashes, final-output checksum verification | Deterministic seed rule and tidy outputs | The current repository has the stronger executable audit trail. |

## High-value analyses to port

### 1. Performance-matched comparisons

At fixed exposure, High and Low networks can differ in task performance as well
as geometry. Add snapshots at common next-token cross-entropy and accuracy
thresholds, then compare A/B separation, probe interpolation, and causal patch
effects at matched performance. A threshold should only be analyzed when a
network actually crosses it; do not force threshold attainment by redefining
the observation.

Recommended status: high priority. This is the cleanest control for the claim
that initialization scale changes representational organization rather than
merely learning speed.

### 2. Fixed-exposure, performance-matched, and asymptotic analysis frames

Retain sequence 40 as the fixed-exposure primary checkpoint only if its
"human-matched" interpretation can be justified from the experimental design.
Add performance-matched results as the main robustness control and a fixed-
budget asymptotic continuation as a boundary test. These frames should never be
pooled into a single estimate.

Recommended status: high priority.

### 3. Hybrid initialization controls

CogSci independently changes the initialization scale of context-input columns
and all remaining weights. Port that logic to the present model with four
conditions: High/High-context, Low/Low-context, Low/High-context, and
High/Low-context. Preserve the zeroed Novel-context column in every condition.
This tests whether strong context-input weights are sufficient or necessary for
splitting while holding the rest of the initialization regime apart.

Recommended status: high priority, but it is a new experiment rather than a
reinterpretation of the frozen three-condition result.

### 4. Replicated sensitivity sweeps

Sweep hidden dimension, learning rate, and initialization scale around the
registered values. Use multiple seeds per cell and report the seed distribution
or a hierarchical response surface. The CogSci sweep files contain one trained
network per hyperparameter cell, so their dense grids show topology but do not
provide cell-level uncertainty. Copying that one-run-per-cell design would not
meet the intended rigor standard.

A practical staged design is:

1. one-dimensional coarse sweeps with 20 paired seeds per value;
2. a smaller two-dimensional initialization-scale by learning-rate grid near
   the transition region;
3. confirmation of selected boundary cells with new held-out seeds.

Recommended status: high priority after the analysis frames are decided.

### 5. Representational and optimization diagnostics

Add the following CogSci-style measures at the same frozen checkpoints:

- context and sequence-position decodability evaluated out of seed/sample;
- effective dimensionality and within-context trajectory dimensionality;
- within- versus between-context trajectory distance;
- context-input, token-input, recurrent, and output weight norms;
- context, recurrent, and output gradient norms during training;
- recurrent Jacobian spectral radius or gain;
- output divergence after context swap, zeroing, or interpolation;
- transition costs separated by switch direction.

These measures are explanatory diagnostics. They should not be promoted to new
success gates after viewing the current results.

Recommended status: medium priority. Predeclare a small primary subset to avoid
a large garden of correlated measures.

### 6. Participant-timeline simulations, only if the modeling target changes

Training one network on each participant's actual context order and exposure
history is useful if the goal is to predict variation induced by human
training schedules. It is not necessary for a mechanistic existence proof with
a common controlled curriculum.

If this analysis is added, initialization seed and participant timeline should
not be perfectly confounded. Use either multiple seeds per observed timeline or
a crossed sample of seeds and timelines. Pair conditions within the same
seed-timeline combination. Otherwise a between-participant network distribution
cannot distinguish schedule heterogeneity from random initialization.

Recommended status: conditional on the intended scientific claim.

## What should not be copied silently

- Do not change one-hot token encoding to Gaussian token encoding in the primary
  model. Add it as a named sensitivity analysis if theoretically useful.
- Do not change hidden dimension 8 to 6 or High initialization SD .50 to 2.0 in
  the primary model. These can be grid points in robustness experiments.
- Do not remove the execution head. The current paper's causal outcome depends
  on it. Instead retain `High_seq_only` and consider an execution-head ablation
  from a paired initialization.
- Do not treat 500 additional training passes as statistical replications.
  Epochs and transition positions are repeated observations within a network.
- Do not use one network per sweep cell as evidence that a hyperparameter effect
  generalizes across initialization seeds.
- Do not add new gates based on results already inspected. New diagnostics
  should be labeled exploratory or registered before a fresh-seed confirmation.

## Proposed registered robustness workflow

After the decisions below are resolved, the implementation can be separated
from the frozen pipeline under a new package and output namespace. A suitable
order is:

1. write a versioned design/config contract, including primary estimands and
   seed sets;
2. add unit tests for schedules, paired initialization, snapshot timing,
   threshold crossing, probe non-learning, and output schemas;
3. reproduce the frozen primary conditions without changing their results;
4. add performance-matched and asymptotic checkpoints;
5. add hybrid initialization controls;
6. run replicated one-dimensional sweeps and then the targeted two-dimensional
   sweep;
7. run the predeclared geometry and optimization diagnostics;
8. confirm any newly discovered boundary with untouched seeds;
9. write checksummed tidy CSVs with one explicit independent unit per row or
   identifiers sufficient for correct aggregation.

The main paper result should continue to use the frozen 20 seeds. Robustness
experiments should use a separate seed registry so they cannot retroactively
select the original result.

## Decisions required before implementation

1. Is the intended SRN claim a controlled mechanistic existence proof, or must
   the network also reproduce participant-specific exposure histories and
   predict variation across people?
2. What empirical fact makes sequence 40 "human matched"? Is it the number of
   A/B sequence repetitions completed before the human probe, or was it chosen
   for another reason?
3. Should the current architecture, one-hot inputs, SD=.50/.02 contrast, and
   execution objective remain frozen as the primary specification, with all
   CogSci-style additions treated as robustness analyses?
4. For fresh confirmation, is there a computational limit on the number of
   networks that can be trained, or should the design be chosen purely from the
   needed seed precision and then costed?

