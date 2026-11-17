# Methods draft: context-configured motor-sequence learning

This draft is reconstructed from the executable final pipelines and frozen
configuration files. It deliberately distinguishes the trained SRN experiment,
the post-training causal analyses, the human behavioral tests, and the later
low-rank interpretation analysis. Bracketed manuscript details such as ethics,
recruitment, compensation, apparatus, and preregistration URLs are not present
in this repository and must be supplied from the study records.

## Analysis strategy

The analyses followed a fixed SRN-first, human-second order. The SRN experiment
tested whether an otherwise identical motor sequence could acquire
context-specific recurrent states with consequences for an execution-speed
readout. All Mixed and Novel evaluations were frozen counterfactual probes:
they caused no optimizer steps and did not update the network. Causal analyses
were then applied to checksum-validated frozen networks. Human data were
analyzed only after the SRN predictors had been frozen, and no human outcome was
used to select an SRN seed, alter an SRN weight, or change an SRN transition
profile. The SRNs are therefore a mechanistic existence proof rather than a
fitted model of the exact human transition profile.

All primary SRN analyses used 20 random seeds. The human analyses used two
independent source experiments: a Mixed-context experiment (Combined dataset)
and a third-context experiment (Novel dataset). Analyses were performed on the
natural-log reaction-time scale. The common feature space consisted of seven
response transitions, corresponding to response positions 1->2 through 7->8
of the fixed eight-item sequence.

## SRN experiment

### Sequence, inputs, and architecture

Every network learned the fixed sequence

`0, 1, 2, 0, 3, 1, 2, 3`.

The network received the first seven tokens and predicted each of the next
seven tokens. Tokens were encoded as strict four-dimensional one-hot vectors.
A three-dimensional context vector was concatenated to the token input at
every recurrent step. The pure contexts were A=`[1,0,0]` and B=`[0,1,0]`.
The frozen probe contexts were normalized Mixed=`[1/sqrt(2),1/sqrt(2),0]`,
additive Mixed=`[1,1,0]`, and Novel=`[0,0,1]`.

The model was an Elman simple recurrent network with eight tanh hidden units.
The seven-dimensional token-plus-context input entered a PyTorch `nn.RNN`.
Two linear readout heads shared the recurrent representation: a four-category
next-token head and a one-unit execution head. A sigmoid transformed the
execution-head output to an execution speed in `[0,1]`; normalized model RT was
defined as `1 - speed`. Biases were initialized to zero and all nonexempt
weights were initialized from a zero-mean normal distribution with the
condition-specific standard deviation described below. The input-weight column
for Novel context was initialized to exactly zero, its gradient was masked, and
the column was explicitly reset to zero after every optimizer step.

The recurrent state was initialized afresh at the beginning of every sequence
repetition. Backpropagation through time therefore covered the seven
within-sequence prediction steps; recurrent state was not carried from one
sequence repetition or experimental block to the next. Learning across blocks
occurred through weight updates.

### Training conditions and exposure schedule

Three prespecified model conditions were trained:

1. High scale with execution objective: weight SD=.50 and both prediction and
   execution losses active.
2. Low scale with execution objective: weight SD=.02 and both losses active.
3. High scale, sequence only: weight SD=.50 and only next-token prediction loss
   active.

For each condition, seeds 0-19 initialized 20 networks, yielding 60 trained
networks. Each network received 170 online training repetitions arranged as 17
alternating blocks of 10 sequences. Training began with A and alternated A10,
B10, ..., ending with A. Only A and B were presented during training. Each full
sequence repetition produced one optimizer step.

Networks were optimized with Adam at learning rate .005. Next-token loss was
the mean categorical cross-entropy over the seven transitions. Global gradient
norm was clipped at 1.0. Four checkpoints were saved after sequences 20, 40,
80, and 170 and labeled `early_after_initial_ab`, `human_matched`,
`several_switches`, and `final`, respectively. The primary post-training causal
checkpoint was the sequence-40 human-matched checkpoint; the v2.1 splitting
gate was evaluated at the final checkpoint.

### Execution objective

For execution-objective conditions, the correct next-token softmax probability
at each transition was treated as sequence confidence, `c`. Motor-correct
probability was

`p(correct_motor) = logistic[kappa * (c_detached - speed - delta)]`,

where `kappa=12` and `delta=.05`. Confidence was detached in this expression,
so execution loss did not directly optimize the sequence-confidence output.
The differentiable execution cost was

`C_exec = 1 * (1 - p(correct_motor)) + .25 * (1 - speed)`.

The total loss in execution-objective conditions was

`cross_entropy + .20 * mean(C_exec)`.

Although confidence was detached inside the motor-correct function, the
execution loss updated the execution head and the recurrent representation
through the speed term. Before any network probes, the speed-accuracy ecology
was audited on a 21-point confidence grid and a 1,001-point speed grid. The
registered calibration checks required optimal speed to be nondecreasing with
confidence and required at least one confidence value to have an interior
speed optimum.

### Representational-splitting prerequisite

At every checkpoint, A/B hidden separation was the Euclidean distance between
the centroids of the seven A and seven B recurrent states. Block-transition
cost was the absolute change in sequence cross-entropy from the final sequence
of one block to the first sequence of the next, averaged over available A/B
boundaries. Cross-context transfer was measured by generating the recurrent
prefix under one pure context, inserting that recurrent state into the same
current token under the other context, and measuring the resulting change in
next-token cross-entropy relative to the matched target-context state.
Context-input use was quantified from the norm of the learned A and B input
columns.

The cohort-level splitting gate used final-checkpoint means and comprised:

- G1: mean final next-token accuracy was at least .80 in both High and Low;
- G2: High separation exceeded both `1.25 * Low separation` and
  `Low separation + .05` (implemented as exceeding their maximum);
- G3: High block-transition cost exceeded Low block-transition cost;
- G4: High absolute cross-context transfer disruption exceeded Low disruption;
- G5: the High A/B context-input weight norm exceeded .05.

G4 was exported as a diagnostic but was not part of the hard pass calculation;
the implemented hard gate required G1, G2, G3, and G5.

### Execution-grounding prerequisite

Execution grounding was evaluated in High-scale execution networks over the
last eight training blocks (sequences 91-170). Within each seed, Pearson
correlations were calculated between correct-token confidence and execution
speed overall and separately in A and B; finite seed correlations were then
averaged. A pooled least-squares model regressed speed on confidence,
standardized block index, a B-context indicator, and seed dummy variables. The
gate required positive mean overall, within-A, and within-B correlations; a
positive controlled confidence coefficient; monotone pre-probe calibration;
and at least one interior calibration optimum. These criteria were directional
greater-than-zero checks rather than null-hypothesis tests.

### Frozen A/B, Mixed, and Novel probes

At each checkpoint, the same fixed token sequence was passed through A, B,
normalized Mixed, additive Mixed, and Novel contexts with model weights frozen.
For each seed, probe normalized RT was compared with that seed's A and B
endpoints. An endpoint pair was considered reliable when
`abs(RT_A - RT_B) >= .02`. The interpolation index was

`q = (RT_probe - RT_B) / (RT_A - RT_B)`,

so q=0 denoted B, q=1 denoted A, and q=.5 denoted the endpoint midpoint.
Between-endpoint status was assessed within seed. The v2.1 result code was
determined at the final checkpoint from the proportion of endpoint-reliable
High seeds whose primary Mixed and Novel probes fell between their endpoints;
a probe passed when that proportion was at least .50. The number of reliable
seeds must be reported with this proportion because endpoint reliability was
not universal, especially at the final checkpoint. Additive Mixed was a scale
sensitivity analysis rather than the primary Mixed definition.

## Causal analysis of frozen SRN states

All causal analyses loaded only checksum-validated High-scale checkpoints,
disabled gradients on every parameter, constructed no optimizer, and verified
that model-weight digests were identical before and after intervention. The
sequence-40 human-matched checkpoint was primary; the other three checkpoints
were temporal diagnostics. Separate deterministic forward passes were used to
construct A/B components and evaluate interventions. These were separate code
paths, not independent stochastic samples: because the frozen network and
inputs were deterministic, corresponding natural states were numerically the
same.

### Full A/B subspace analysis (v2.2)

For each seed, checkpoint, and transition position, the A/B direction was
`h_A - h_B`, with the midpoint `(h_A + h_B)/2`. Singular-value decomposition of
the seven-by-eight matrix of position-specific difference vectors defined a
within-network A/B subspace. The smallest number of singular directions
accounting for at least 90% of uncentered squared singular-value energy was
retained.

Sufficiency was tested by adding `alpha * (h_A - h_B)` to natural A, B, Mixed,
and Novel states. The instantaneous alpha grid was -2 to 2 in .5 increments;
the primary slope used `abs(alpha) <= 1`. A secondary recurrent rollout used
alpha=-1, 0, and 1. Partial necessity was tested by removing the projection of
each state, relative to its position-matched midpoint, onto the retained A/B
subspace. Mixed and Novel pathway use was tested by replacing the recipient's
A/B-subspace component with the component from the position-matched A or B
donor while retaining the recipient's orthogonal residual.

Negative controls were norm-matched random directions orthogonal to the A/B
subspace, dimension-matched random subspaces, and sequence-position directions.
Every intervention exported next-token identity and cross-entropy, hidden-state
norm, nearest-natural-state distance, and regularized Mahalanobis distance. The
regularization ridge was 1e-5.

At the seed level, M1 required nonzero natural A/B geometry and the passed
source split gate. M2 required correctly signed steering, improvement toward
both endpoints, a steering slope larger than the absolute random-control slope,
and at least .90 next-token identity preservation. M3 required reliable A/B RT
endpoints, at least .50 collapse of the A/B execution difference after
ablation, and at least .90 preservation of next-token accuracy. M4 required
successful Mixed patching with greater endpoint-proximity improvement than the
random-subspace patch. A corresponding Novel patching criterion was reported
separately. A group criterion passed when at least 80% (16/20) of seeds passed.
The complete v2.2 intervention chain did not pass and is retained as the strict
necessity boundary.

### Position-matched context-component analysis (v2.3)

The primary causal specificity analysis decomposed each position-matched pair
exactly as

`h_A,k = mu_k + d_k` and `h_B,k = mu_k - d_k`,

where `mu_k=(h_A,k+h_B,k)/2` and `d_k=(h_A,k-h_B,k)/2`. SVD of the seven local
half-differences defined a shared PC1 context component. Each `d_k` was
separated into its projection on this shared axis and an orthogonal
position-specific residual.

Mixed and Novel recipient states were patched toward A and B using the full
local component, the shared component alone, or the position-specific component
alone while preserving the recipient residual orthogonal to the patched
component. For each target position, every one of the other six positions
served as a wrong-position control after its component was norm-matched to the
target component. Orthogonal random and adjacent-sequence-position components
were additional controls. A patch contrast was the A-donor minus B-donor
execution speed; its sign was aligned with the natural A-B execution difference
at that position.

Four capacity-controlled execution regressions were fit only to natural A and
B rows: sequence-position only, continuous context coordinate only, their
additive combination, and a Context × Position interaction. The continuous
coordinate was the projection of the natural state, relative to its local
midpoint, onto the shared context axis. Models used a ridge of 1e-8 (the
intercept was unpenalized) and leave-one-seed-out evaluation. Mixed and Novel
states were generalization targets only and never contributed to coefficient
estimation.

M1 required a positive full patch effect and endpoint-proximity improvement.
M2 required the matched full effect to exceed the exhaustively averaged
wrong-position effect. M3 required a positive across-position correspondence
between natural A-B execution differences and patch contrasts. M4 required the
full effect to exceed the shared-only effect. M5 required full ablation to
collapse more A-B execution separation than shared-only ablation, a positive
specific-component collapse, and at least .90 sequence-identity preservation.
M6 required the interaction regression to beat the additive regression on both
natural A/B held-out seeds and Mixed generalization. Novel generalization was
reported separately.

The implemented group rules were not identical across criteria. M1 and M5 used
the .80 seed threshold. M2-M4 required more than half the seeds plus the
corresponding favorable group mean. M6 was the cohort-level model-comparison
result. The orthogonal-random specificity control required more than half the
seeds and a favorable group mean. The primary result passed M1-M4 and the
random-direction control but failed M5 and M6; it therefore supports
position-matched causal specificity, not the complete necessity-and-predictive-
interaction chain.

## Human behavioral data

### Participants and source experiments

The final analysis read `data/mixedBreak_clean.csv` as the Combined/Mixed
experiment and `data/threeContextsBreak_clean.csv` as the Novel/third-context
experiment. After the complete-case sequence and block rules below, the
analyzed samples contained 142 Combined participants and 65 Novel participants.
These are the analysis Ns; broader counts in the three-context data dictionary
refer to a less restrictive sequence-level descriptive contract and should not
be substituted for the final Ns.

[Add recruitment source, eligibility criteria, demographic information,
compensation, ethics approval, informed-consent procedure, and any a priori
sample-size rationale here.]

### Preprocessing and transition construction

Warm-up blocks were excluded. Experimental labels were mapped explicitly to A,
B, Mixed (`Both`), Novel/Space, or No Context (`Default`); unknown labels raised
an error rather than being silently pooled. Participants were retained only if
they belonged to the target-probe experiment. Counterbalance, exposure cohort,
probe phase, probe-entry context, and the nearest preceding established A/B
context were preserved.

The preprocessing contract first aggregated raw keypresses to sequence units.
Only finite positive RTs on correct keypresses contributed to sequence mean RT.
A sequence was considered correct only when it contained at least eight correct
keypresses. The first raw sequence in every block was excluded. Each remaining
block was required to contain exactly the nine original sequence repetitions
2-10 and all nine sequences had to be correct; if any block failed this rule,
the participant was excluded. Participants whose number of retained blocks did
not equal the modal block count for that experiment were also excluded. This
rule conservatively removed duplicated full sessions recorded under a reused
participant identifier.

The validated sequence units were then joined back to correct, finite,
positive keypress RTs. The Maze task's doubled keypress encoding (0,2,...,14)
was mapped to the same ordinal response positions as the other tasks (0,...,7).
The first response position was excluded, leaving ordinal keypress values 1-7,
which were labeled as transitions 1->2 through 7->8. When multiple source
attempts mapped to the same participant, block, sequence, and transition, their
RTs were averaged. Every retained sequence was required to contain all seven
transitions. The resulting token-level tables contained 151,214 Combined and
69,433 Novel transition observations. RT was natural-log transformed.

Odd analysis sequence repetitions were labeled the within-participant
calibration split and even repetitions the evaluation split. This odd/even split
was used only for held-out participants' probe calibration and scoring; group
structure was learned from other participants. All pure A/B data remained
available to estimate a held-out participant's A/B endpoints because those
endpoints contain no Mixed or Novel outcomes.

### Pure A/B transition profiles

For each participant, context, and transition, mean log RT was computed from
pure A or B observations. These cell means were shrunk toward the corresponding
dataset-by-context-by-transition group mean with weight

`w = n / (n + 8)`.

The frozen participant endpoints were the shrunken values `r_A,i,k` and
`r_B,i,k`. The participant's A/B contrast was

`delta_i,k = r_A,i,k - r_B,i,k`,

and the endpoint midpoint was `(r_A,i,k + r_B,i,k)/2`. No Mixed, Novel, or
Default observation contributed to these profiles.

Profile reliability was estimated by recomputing A-B log-RT contrasts from odd
and even pure sequence repetitions. For each participant, the two
seven-transition profiles were correlated and corrected with the
Spearman-Brown formula `2r/(1+r)`. The reliability gate required the mean of
finite participant reliabilities to be at least .30.

### Context × Position analysis

Pure A/B log RTs were first averaged within participant, counterbalance, block,
context, and transition. Separate ordinary least-squares models were fit in the
Combined and Novel datasets:

`logRT ~ Context * categorical(Position) + participant fixed effects +`
` categorical(counterbalance) + block_order`.

Standard errors were clustered by participant. A six-df Wald test jointly
tested all Context × Position coefficients. Probe outcomes were not used in
this analysis. Both dataset-specific tests should be reported; the pipeline's
single final flag was true if either dataset test was significant.

### Probe outcomes and endpoint coordinates

Mixed and Novel were analyzed separately. For every probe observation, the
midpoint deviation was

`d_i,k,t = logRT_i,k,t - (r_A,i,k + r_B,i,k)/2`.

For positions with reliable human endpoints,
`abs(r_A,i,k-r_B,i,k) >= .03`, a responsibility/interpolation coordinate was

`q_i,k,t = (logRT_i,k,t - r_B,i,k) / (r_A,i,k - r_B,i,k)`.

The q coordinate was not clipped and could lie outside `[0,1]`; it was set to
missing when the endpoint difference was below .03. Midpoint deviation, which
does not divide by endpoint separation, was retained for all joined probe rows.

### Participant-held-out model comparison

Participants were assigned deterministically to five folds by sorting
participant identifiers and cycling through fold labels. For each fold, all
group-level structure was learned from the other participants. Within each
held-out participant, odd probe repetitions calibrated the participant-specific
free coefficients and even probe repetitions were scored. Mixed and Novel
model ladders were fit independently.

Five primary models were compared:

- M0, scalar context: one participant q mixed global A and B endpoints averaged
  across calibration rows, yielding a position-invariant prediction.
- M1, scalar mixture: one participant coefficient `lambda_i` predicted
  `midpoint_i,k + lambda_i * delta_i,k`.
- M2, token specific: reliable training-participant q values supplied a
  group transition effect, expressed as q-.5 and shrunk with
  `n/(n+12)`. A held-out participant scalar was added to this transition effect
  and multiplied by the participant's delta.
- M3, low-rank human: within each fold, PCA was applied to the training
  participants' seven-element pure A/B delta profiles. The smallest dimension
  explaining at least 90% of their variance was retained. The training-group
  probe midpoint-deviation profile was projected into that subspace. A held-out
  participant intercept and amplitude were fit on odd repetitions and applied
  to the frozen shape on even repetitions. In the final v3 run this 90% rule
  selected five dimensions in every fold; this v3 M3 should not be conflated
  with the separately selected rank-2 v3.1 basis below.
- M4, SRN informed: the sequence-40 SRN execution q profile was averaged over
  all endpoint-reliable registered seeds at each transition; no best seed was
  selected. The SRN transition shape was fixed before human modeling. A
  group-level slope relating logit SRN q to reliable training-human q was
  estimated in each fold, and a held-out participant intercept was calibrated
  on odd repetitions. The primary M4 was this calibrated execution-q model.
  Zero-parameter strict SRN predictions and full/shared/specific component-q
  variants were secondary outputs.

For each fold, the Gaussian residual SD was estimated from training-participant
probe deviations around each participant-by-transition mean and was floored at
.05 log-RT units. The primary metric was the sum of pointwise held-out Gaussian
log likelihood; mean log likelihood, RMSE, and MAE were also exported. Because
the primary sum gives every held-out observation equal weight, participants
with more probe blocks contribute more to the registered comparison. AIC and
BIC were descriptive full-data summaries and were not the model-selection
criterion.

Observed-scale identifiability was assessed with a simplified template
recovery analysis, not by rerunning the complete cross-validated model-fitting
code. M0-M4 transition templates were simulated using the observed
participant-by-transition count matrix, observed endpoint magnitudes, an
effect multiplier of 1.5, and observed within-participant noise. Each candidate
template was fit with an intercept and, when nonconstant, an amplitude; BIC
selected the winner. Twenty replicates were run per generating template, so
reported recovery probabilities have resolution .05. Recovery above the
five-model chance value of .20 was required for the corresponding gate.

## Low-rank human interpretation analysis (v3.1)

### Rank selection

The v3.1 analysis reloaded the frozen pure A/B profiles and probe tables. It
considered ranks 1, 2, 3, and 7. Within each dataset and participant fold, PCA
and Factor Analysis bases were fit only to training participants' pure A/B
delta profiles. Held-out pure-profile reconstruction error was recorded.
Separately, a held-out participant's odd probe repetitions estimated component
coefficients in the training-only basis and even repetitions supplied Gaussian
probe log likelihood.

PCA probe likelihood was the primary rank-selection metric. For each candidate
rank, the fold scores were pooled across the five Combined and five Novel
folds. The selected rank was the smallest rank whose mean score was within one
standard error of the best rank's mean score. This rule selected rank 2.
Factor Analysis, held-out reconstruction error, and a 1,000-replicate parallel
analysis obtained by independently permuting each transition column across
participants were robustness diagnostics rather than the primary selector.

The primary rank-2 basis was then refit using only the 142 Combined
participants' pure A/B profiles. The Novel basis used for replication was fit
independently and aligned only after estimation.

### Component stability and scores

Stability of the Combined rank-2 basis was evaluated with 1,000 participant
bootstrap resamples and the five participant-training folds. Candidate
components were matched to the full-sample reference by maximizing absolute
cosine similarity using the Hungarian assignment algorithm, then sign aligned.
A component was stable when its median matched cosine across the combined
bootstrap/fold audit was at least .70. Percentile 95% bootstrap intervals were
calculated for every transition loading.

Participant component scores were obtained by projecting centered pure A/B
profiles into the frozen basis and shrinking the scores toward zero with weight
`n/(n+8)`, where n was the participant's mean pure A/B cell count. Mixed and
Novel midpoint-deviation profiles were projected into the fixed basis; they
could estimate coefficients but could not update the basis.

### Semantic features and inference

The seven transition features included formal sequence properties (current and
next key, lag since prior key occurrence, and sequence half), group pure
difficulty and A-B effects, raw error rate, early and late learning slopes,
initial and asymptotic log RT, switch cost, and post-switch recovery. Learning
and switch quantities were first calculated within participant and transition
and then aggregated. The resulting transition-feature table was frozen and
hashed before component labels were assigned.

Each stable component loading was associated with each estimable transition
feature across the seven transition positions. Numeric associations used
Pearson correlation as the permutation statistic; categorical associations
used eta squared. P values enumerated all label permutations for the available
positions, up to 7!=5,040 permutations, with a plus-one correction.
Leave-one-transition-out error and loading-bootstrap intervals were also
exported. The implemented association gate used unadjusted p<.05 across the
many feature tests plus a bootstrap interval excluding zero; these semantic
associations should therefore be presented as constrained interpretation or
exploratory evidence rather than a familywise-confirmatory test.

To assess generic transition difficulty, component loadings were residualized
against mean pure log RT and retested against A-B sensitivity, switch cost, and
Mixed/Novel profiles with exact permutations. Certain labels, including
context sensitivity and switch vulnerability/recovery, were allowed only if
this beyond-difficulty test passed.

An observed-scale semantic-recovery simulation generated difficulty, context-
sensitivity, switch-vulnerability, combined two-axis, and random profiles. Two
hundred replicates per generator preserved the participant count and estimated
profile noise. A semantic label was allowed only if minimum diagonal recovery
was at least .60 and the random-axis false-label probability was at most .05.
Final label assignment additionally required cross-dataset mean matched cosine
of at least .70. Components remained neutrally named when the applicable gates
did not all pass.

### Shared versus dataset-specific human bases

The Combined and Novel pure A/B bases were fit independently at rank 2 and
aligned post hoc. Replication was declared when the solution was lower than
full rank and mean matched-component cosine was at least .70. Principal angles,
matched cosines, and the fraction of one subspace captured by the other were
reported descriptively.

A separate participant-held-out comparison evaluated a scalar baseline, a
rank-2 basis shared across dataset-centered Combined and Novel pure profiles,
and rank-2 bases estimated separately within each dataset. Fold identities were
defined from the participant identifier with the dataset prefix removed so a
reused identity could not enter the other dataset's training basis. Held-out
participants again used odd probe repetitions for coefficients and even
repetitions for scoring. The pipeline labeled separate bases as preferred when
their mean fold log-likelihood gain over the shared basis exceeded one standard
error in at least one dataset; this was a one-SE decision rule, not a formal
two-sided hypothesis test.

### Frozen human-SRN comparison

For the human system, SVD was applied to the Combined participants' pure A/B
delta profiles. For the SRN system, SVD was applied to the 20 seeds' frozen
sequence-40 A-B normalized-RT profiles. Rank explaining 90% of variance,
participation-ratio effective dimensionality, PC1 variance, principal angles,
and canonical correlations were compared without modifying either system.

A human-SRN abstraction match required all of the following: selected human
rank below seven, mean principal angle at most 30 degrees, maximum principal
angle at most 45 degrees, minimum canonical correlation at least .50, and
absolute PC1 variance-partition difference at most .20. The frozen comparison
did not pass. This negative result limits the claim to a shared qualitative
configured-state principle; it does not support an exact human-SRN transition
geometry match.

## Reproducibility and software

The validated runtime was Python 3.11 with package versions frozen in
`requirements.txt`. The complete canonical workflow is

```bash
python scripts/run_all_final.py
```

or, using checksum-validated existing v2.1 checkpoints,

```bash
python scripts/run_all_final.py --reuse-execution
```

The repository stores SHA-256 hashes for all 240 serialized checkpoints and
for the registered final CSVs and figure-source copies. Frozen analyses verify
checkpoint hashes before loading and verify that weights remain unchanged.
The canonical test suite covers sequence/context definitions, zero-update
probes, human transition indexing, participant holdout, causal controls, output
schemas, and baseline hashes.

## Manuscript reporting boundaries

- Report the SRN as an existence proof, not as a quantitatively fitted account
  of human data.
- Report the number of reliable SRN endpoints whenever quoting a fraction of
  probes between endpoints.
- Describe v2.3 as evidence for position-matched causal specificity. Do not
  claim that the complete necessity or interaction-model chain passed.
- Distinguish the v3 M3 model (five dimensions under its foldwise 90%-variance
  rule) from the v3.1 rank-2 solution selected by pooled held-out probe
  likelihood and the one-SE rule.
- Treat transition-feature semantics cautiously: only seven transition labels
  are available, and the primary feature screen is not multiplicity adjusted.
- Report both dataset-specific Context × Position tests, not only the minimum
  p-value or the Boolean `any()` summary.
- State that the registered human likelihood weights observations, whereas a
  participant-equal figure or sensitivity analysis changes that weighting.
- Retain the failed exact human-SRN subspace match as an explicit negative
  result.

