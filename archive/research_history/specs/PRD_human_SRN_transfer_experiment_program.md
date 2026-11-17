# PRD: Experiments to Explain Partial Transfer in Humans and SRNs

## Goal

Implement a coordinated program that distinguishes why the current High-scale SRNs show partial transfer to Mixed and Novel contexts while the human results suggest stronger transfer. Test acquisition, recurrent state and probe experience, cue composition, action visibility, and prior structural experience separately before proposing another architecture.

Build all five experiments and the initial human measurement audit. Run the computational experiments where repository inputs are available. For experiments requiring new human observations, deliver a working local task, counterbalanced schedules, data contracts, analysis scripts, and power simulations. Report implementation completion separately from empirical completion.

The scientific target is the joint pattern of familiar-context learning, A/B differentiation, and transfer. An intervention can improve transfer while weakening A/B differentiation; that is an informative tradeoff, not automatically a successful account of the humans.

**Agent instruction:** Work through the stages below, preserving the reference implementation. A scientifically negative result does not stop the other experiments. Stop only the affected comparison when its inputs or identification requirements fail, record why, and continue independent work. Do not tune interventions to human probe RTs.

## 1. Evidence this program must respect

| Established observation | What it supports | What remains unresolved |
| --- | --- | --- |
| High-scale Mixed and Novel prediction and RT improve during A/B training. | Partial transfer is present. | Whether its magnitude and time course match humans under comparable measurement. |
| Same-checkpoint High probes increasingly exceed the worse A/B endpoint. At checkpoint 170, mean paired RT excess is 0.223 for Mixed and 0.242 for Novel. | The late relative deficit survives time matching and sequence-level aggregation. | Which computation or task assumption produces it. |
| Checkpoint-170 mean paired NLL excess is 1.438 for Mixed and 1.931 for Novel. | The deficit already exists in next-token predictions. | Its specific representational cause. |
| Applying the checkpoint-20 readout to unchanged checkpoint-170 predictions leaves RT excess of 0.198 and 0.204. | Readout parameter changes amplify part of the expressed deficit. | Whether a prediction-based RT model captures human execution. This substitution does not answer that. |
| Low-scale networks have nearly overlapping A/B/probe performance and substantially better late prediction. | Strong generalization is possible in this implementation. | Acquisition differences confound a direct attribution to splitting. Near-identical A/B endpoints cannot establish intermediacy. |
| Human continuous-expression models outperform the tested categorical-switching model; structured A/B models capture meaningful variation. | Human expression contains structured continuous variation. | A unique retrieval mechanism, or a calibrated estimate of first-encounter transfer. |
| Human probes and pooled A/B profiles may sample different learning times. | Human transfer estimates require an event-level audit. | Whether the apparent human–SRN difference persists with local references and matched probe experience. |

Numerical values above come from `PROBE_PREDICTION_VS_EXECUTION_REPORT.md`. Timing and replay verification come from `TIME_MATCHED_SEQUENCE_PROBE_REPORT.md`. Treat these as reference results to reproduce, not thresholds every intervention must reach.

Do not describe the current network as showing “no transfer.” Do not infer that separated hidden states cause poor transfer solely because both occur in High-scale networks. Do not treat human negative additive offsets relative to pooled training as demonstrated facilitation.

## 2. Work packages and completion states

| ID | Scientific question | Required implementation |
| --- | --- | --- |
| H0 | Are human and network probes measuring comparable learning and exposure? | Existing-human-data timing/first-exposure audit and local A/B comparison. |
| E1 | Does the deficit persist at comparable familiar-context acquisition? | Matched-acquisition analysis, bounded training extension, transfer and splitting trajectories. |
| E2 | Does recurrent history or learning during probe exposure change transfer? | Frozen/learning × reset/carry experiment, retention tests, and a human history/practice task configuration. |
| E3 | Which part of the probe input produces the deficit? | Cue/layout factorial intervention, graded generalization diagnostics, and a human cue-factorial configuration. |
| E4 | Do humans transfer an execution skill that a prediction-only readout misses? | Full-preview versus one-target human experiment, analysis and power code, and an explicit model-information audit. |
| E5 | Does prior experience with structure improve later transfer? | Cross-layout pretraining versus exposure-matched context-bound practice and no-pretraining controls. |

Use two status fields: `implementation_status` and `evidence_status`. Examples: `validated` with `awaiting_human_data`; `validated` with `completed`; `partial` with `missing_source_data`; `validated` with `no_common_support`. Never turn unavailable data into a null result.

## 3. Frozen reference and permitted changes

Locate and preserve the actual repository implementation:

- Eight-unit tanh Elman SRN; High initialization scale 2.0 and Low 0.02.
- The repository's fixed eight-item motor sequence in both A and B.
- Alternating ten-sequence A/B blocks; reference endpoint after 170 external sequences.
- Physical-token, shared ordinal-position, full-layout, and global-context input channels.
- Existing next-token objective, optimizer, learning rates, precision, reset policy, and execution objective.
- Monotonic readout: `a = softplus(alpha)`, `v = sigmoid(a * logit(p) + b)`, normalized RT `1 - v`.
- Existing handling of probability clipping, detached prediction confidence in the execution objective, and sequence boundaries.

The implementation and audited configuration take precedence over approximate prose descriptions. Record discrepancies explicitly; do not silently repair the reference to fit this document.

Reference diagnostics use independent, zero-state, post-update A/B/Mixed/Novel branches. Each scores the same seven transitions, with no parameter update and no extra preparation cycles. Keep the distinction between pre-update acquisition observations and post-update frozen evaluations.

Changes are confined to the named manipulation: training duration in E1; state/update policy in E2; evaluation inputs in E3; human visibility in E4; prior training in E5. Keep the baseline as a separate immutable run. Existing High sequence-only checkpoints may support prediction/geometry controls, but do not invent an execution RT for them.

## 4. Source discovery and repository integration

Use repository conventions and existing modules rather than rebuilding the model. Known starting points, subject to verification:

| Purpose | Reported repository location |
| --- | --- |
| Reference run | `outputs/ordinal_layout_ecology_final` |
| Training | `ordinal_layout_ecology/training.py::train_layout_condition` |
| Frozen evaluation | `ordinal_layout_ecology/probes.py::evaluate_frozen_layouts` |
| Reference plotting | `ordinal_layout_ecology/plotting.py::figure_raw_rt` |
| Time-matched replay | `scripts/run_time_matched_sequence_probe_overlay.py` |
| Prediction/readout diagnostic | `scripts/run_probe_prediction_vs_execution.py` |
| Human Mixed data | `data/mixedBreak_clean.csv` |
| Human Novel data | `data/threeContextsBreak_clean.csv` |

Recover schedules and raw event records where needed. Cleaned human tables contain fields such as `ID`, `trialNumber`, `sequenceNumber`, `keypressNumber`, `trial_type`, `trialType`, `counterBalance`, `correct`, and `rt`; their semantics must be traced to source preprocessing. Do not assume a sequence counter is globally chronological or that trial labels identify first-ever exposure.

Namespace participant IDs by dataset. Preserve source condition labels alongside canonical A/B/Mixed/Novel labels. Recover actual seed lists from manifests; do not assume they are 0–19. Record code revision, source paths, hashes, versions, and all derived exclusions.

If only these reports are supplied to the coding agent, implement adapters and synthetic validation, and list the precise missing repository/data artifacts. Reports and figures cannot substitute for checkpoints or event-level observations.

## 5. Shared measurement contract

For seed `s`, checkpoint `t`, context `c`, and matched seven-transition sequence, compute:

\[
L_{s,t,c}=\frac{1}{7}\sum_k-\log p_{s,t,c,k},\quad
P_{s,t,c}=\frac{1}{7}\sum_kp_{s,t,c,k},\quad
R_{s,t,c}=\frac{1}{7}\sum_k f_{s,t}(p_{s,t,c,k}).
\]

Apply the readout per transition before averaging. Mean NLL is not `-log(mean p)`, and mean correct-token probability is not classification accuracy. Export all three plus token accuracy, entropy, and probability margin when available.

For `Y` equal to NLL or RT, define the within-seed relative deficit:

\[
G^Y_{s,t,c}=Y_{s,t,c}-\max(Y_{s,t,A},Y_{s,t,B}).
\]

Positive values mean worse than both familiar endpoints. Also retain both signed contrasts `Y_c - Y_A` and `Y_c - Y_B`; a single max can obscure asymmetry. Calculate contrasts before averaging across seeds.

Unless explicitly labeled otherwise, all three values use the same parameters, state-initialization policy, input sequence, and training checkpoint. Section 10 defines separate reference types once adaptation branches have diverged.

Measure absolute improvement separately:

\[
B^Y_{s,t,c}=Y_{s,t_0,c}-Y_{s,t,c}.
\]

Use initialized checkpoint 0 when available and checkpoint 20 for continuity with prior figures. In E5, distinguish improvement relative to original initialization from improvement relative to the post-pretraining/pre-target checkpoint. A probe can improve absolutely while its relative deficit grows.

Report the A/B gap `D^R = abs(R_A - R_B)`. Retain the reference 0.010 normalized-RT tolerance for interpretable endpoint separation. Use sensitivity values 0.005 and 0.020, without choosing the one that favors a result. If `D^R <= tolerance`, label intermediacy `endpoints_too_close`; still report absolute performance and deficits. Keep raw A/B identities. Optional Fast/Slow displays must use one frozen whole-profile A/B-only mapping, never per-transition relabeling.

Sequence means are primary. Transition-level records support integrity checks and secondary mechanism diagnostics; pooled item profiles do not replace time-matched sequence comparisons. Human milliseconds, log RT, SRN normalized RT, and NLL remain distinct scales.

## 6. Splitting, uncertainty, and comparison rules

Reuse the established representational and behavioral splitting measures, recording their definitions. Export raw matched-position hidden-state A/B distance, any existing scale-normalized version, A/B sequence RT contrast, and existing state-space classifications. Fit visualization transforms using familiar training states only; hold them fixed for probes. Calculate geometry within seed before summarizing. These measurements describe organization and do not establish a causal mechanism by themselves.

Use the existing 20 seeds for reference reproduction. For new computational experiments, use a common prespecified set of 20 additional seeds, disjoint from previously inspected seeds, with a deterministic derivation recorded before viewing their probe outcomes. Pair manipulations within seed, task, layout bank, and parent checkpoint. Report original and new cohorts separately, then a clearly labeled combined estimate. Smoke runs may use two seeds but cannot support conclusions.

Use 2,000 whole-seed or whole-participant bootstrap draws and 95% intervals for primary contrasts. Keep repeated checkpoints, transitions, and branches together when resampling. Human population models must account for participants and datasets; do not pool every keypress as an independent observation. Predefine a small primary contrast set per experiment. Label other contrasts exploratory; if significance tests are included, adjust within the declared experiment family.

Show continuous effects and uncertainty. A null interval crossing zero does not establish equivalence. Do not import the prior report's arbitrary centered-R² success threshold as a mechanistic gate.

## 7. Stage 0 — baseline reproduction and provenance

Before interpreting new manipulations:

1. Resolve the reference run, seed list, schedules, and parent checkpoints.
2. Reproduce A/B/Mixed/Novel NLL, confidence, and RT at 20, 100, and 170; reuse all available 20, 30, …, 170 checkpoints.
3. Verify the time-matched final contrasts in Section 1 within the reference pipeline's numerical precision. Use unrounded tables when available.
4. Verify state and RNG isolation, and that running diagnostics leaves subsequent training unchanged.
5. Save the resolved configuration and a machine-readable audit.

If intermediate snapshots are unavailable, reuse the previously verified deterministic replay and save them once. Cache by source/configuration hash. Do not repeatedly replay training for each plot. A mismatch blocks interpretation of descendants of that run, while human task implementation and independent audits continue.

## 8. H0 — human timing and first-exposure audit

**Question:** Is the human comparison using a first encounter at comparable acquisition, or behavior after additional sequence/probe experience?

Build an event table preserving absolute order, context, block, repetition within block, cumulative A/B exposure, cumulative exposure to each probe configuration, preceding context, elapsed time/rest where observed, correctness, and the existing cleaning decision. Derive separately:

- First-ever sequence in each probe configuration, including its individual transitions.
- Later sequences in that first probe block.
- Later probe blocks and cumulative probe exposure.
- First occurrence of any probe and first occurrence of each probe family.

Derive exposure counts from the complete event stream before RT exclusions. If the first sequence contains errors, preserve its identity rather than calling the first retained correct sequence the first encounter. Report missingness and error rates separately. Use the existing RT cleaning contract; do not choose a new cutoff based on whether probes look intermediate.

Retain the number and identity of valid transitions in every sequence. Do not impute excluded intervals as correct performance. If the existing contract permits incomplete sequences, retain that primary contract and add an all-seven-valid sensitivity; report differential missingness across contexts and never replace an excluded first encounter with a later one.

**Local familiar references:** estimate A and B performance at each probe event using A/B observations only. Primary implementation: a regularized hierarchical model of sequence log RT with participant/context effects and a low-complexity learning-time term, initially a spline with three degrees of freedom. Include observed rest and switch indicators only when identifiable. Select regularization using held-out A/B blocks, without consulting probe outcomes. Freeze the model before predicting probes' familiar references.

Validate reference prediction by masking A/B blocks as pseudo-probe events with comparable temporal gaps. Export prediction error and coverage. Mark estimates unsupported if the event is outside both contexts' observed time range by more than one scheduled A/B cycle, or if available data cannot distinguish time from context. Use a causal sensitivity based only on preceding A/B observations and a simple nearest-block descriptive sensitivity. References using post-probe A/B trials must be labeled as such because probe experience can affect those trials.

Plot observed sequences in actual order with probe points and local A/B estimates. Give first-encounter and later-exposure summaries separately. Do not claim human same-snapshot counterfactual endpoints were directly observed. Do not refit the old history-dependent `q` branch; an unidentifiable schedule remains unidentifiable.

**Deliverables:** event/exposure audit, pseudo-event validation, local reference table with uncertainty, first/later probe plots, and a statement of which human–SRN comparisons are now defensible. If timing is unrecoverable, retain the raw descriptive comparison and state that its temporal confound remains unresolved.

## 9. E1 — acquisition matching and extended training

**Question:** Does High transfer remain weaker after familiar-context performance is made comparable?

Keep the 170-sequence reference. Continue both High and Low runs, preserving the alternating schedule and optimizer state, to 340 and 680 total sequences. No Mixed/Novel training. Save frozen evaluations every ten sequences and at 170, 340, and 680. Preserve the exact schedule phase; do not restart A/B ordering at the extension boundary.

Implement two complementary comparisons:

1. **Acquisition trajectories:** plot probe NLL/RT and relative deficits against familiar acquisition, using both mean and worse-endpoint A/B NLL as x-axes. Show the original time axis alongside them. Mark first attainment of prespecified worse-endpoint NLL thresholds 2.0, 1.0, 0.5, and 0.1 nats. Unreached thresholds are censored, not discarded.
2. **Strict matching:** within each paired High/Low seed, use only A/B results to match each High anchor at 170, 340, and 680 to a saved Low checkpoint. Match preceding-context phase and require each labeled A/B NLL to differ by at most 0.05 nats. Require each context's token accuracy to differ by no more than one of seven responses. Choose the smallest A/B-vector distance; break ties by earlier checkpoint. Preserve unmatched anchors. Sensitivities use NLL calipers 0.025 and 0.10, reported separately.

Export the complete candidate/match table, A/B confidence distributions, transition-loss dispersion, exposure counts, and residual imbalance. Matching a single mean can conceal one poorly learned context. Do not describe a threshold comparison as strict matching if companion endpoints remain imbalanced. If strict common support is absent, say so; do not extrapolate a matched result or select only favorable seeds.

At every anchor, track splitting and behavioral A/B differentiation. Report paired changes from 170 to 340 and 680 in probe absolute benefit, relative deficit, familiar performance, and splitting.

**Interpretation:** a deficit that shrinks at comparable acquisition supports a learning-stage contribution. A persistent deficit in valid matches argues against acquisition alone. Improvement accompanied by collapsing differentiation identifies a tradeoff. Continued absence of common support leaves the causal attribution unresolved. Extended training is a duration intervention; it does not isolate initialization's representational effect from all other effects of initialization.

## 10. E2 — recurrent state versus learning during probe exposure

**Question:** Is the present first-sequence result sensitive to recurrent history, or do probes require additional parameter learning?

Use reference checkpoints 20, 100, and 170. At each, independently clone the full checkpoint, including optimizer and readout state. Run eight repetitions of A, B, Mixed, or Novel under the following factorial design:

| Weight policy | Sequence-boundary state policy | What can change across repetitions? |
| --- | --- | --- |
| Frozen | Reset | Neither learned parameters nor inherited sequence state. |
| Frozen | Carry | Recurrent state only. |
| Learn | Reset | Parameters using the ordinary training update. |
| Learn | Carry | Parameters and recurrent state. |

Run each cell after either one familiar A or one familiar B priming sequence. Priming is frozen in every cell and begins at zero state. Retain a zero-state/no-primer branch as the exact reference. For carry, the first probe starts from the primer's terminal state, and later repetitions inherit the preceding terminal state. For reset, start each repetition from zero, including the first probe. Score each sequence before its update. Learning uses the existing sequence and execution objectives and update cadence; frozen cells perform no optimizer step. Detach carried hidden state at sequence boundaries; do not introduce cross-sequence backpropagation.

The primer is a real external sequence with normal inputs. Add no fictitious repeated settling cycles. Reset/carry manipulations outside the training protocol are explicitly intervention tests, not corrected versions of the baseline.

For every Mixed/Novel branch, run A and B control branches from the same parent and primer under the same factorial policy. During learning cells, two endpoint references are required:

- Parallel A/B-practice branches: comparison to equal amounts of familiar-context practice.
- Frozen A/B evaluations from the probe-adapted parameters after each repetition, using zero state: retained familiar skill after probe learning.

These answer different questions; never label the parallel branches as same-parameter endpoints after their training has diverged. Keep diagnostics isolated from continuing adaptation. Also evaluate the adapted probe with zero state to separate a retained parameter change from a carried-state effect.

## 11. E2 — outcomes, retention, and human implementation

Primary outcomes are first-repetition deficit, change from repetition 1 to mean repetitions 6–8, and the update × carry interaction. Report primer A versus B effects and whether they persist across repetitions. Assess NLL and RT separately.

After eight repetitions, evaluate zero-state A/B/probe performance with weights frozen. In separate descendants, resume one ordinary ten-sequence A block or B block and evaluate again. This measures retention and interference without contaminating the source adaptation branch. Do not infer long-term retention from this short computational test.

Expected diagnostic distinctions:

- Frozen carry helps but reset removes the benefit: sensitivity to recurrent history.
- Learning helps and improvement survives zero-state evaluation: parameter adaptation contributes.
- Both help: both processes contribute, with interaction estimated directly.
- A/B performance worsens after probe learning: transfer improvement carries an interference cost.
- No effect with valid controls: these manipulations did not explain the deficit within the tested range.

A deterministic frozen/reset repeated sequence must return identical predictions. Any apparent learning there is an implementation error.

Build a human history/practice configuration using the shared task engine: counterbalance first probe identity and preceding A/B context, equate the pre-probe pause, present eight probe repetitions, then familiar return blocks. Log every event and analyze first versus later performance. Humans cannot be assigned “weights frozen” or literal hidden-state reset; the human implementation tests behavioral history and practice predictions rather than claiming those internal manipulations were reproduced.

## 12. E3 — disaggregate the contextual input

**Question:** Is the deficit driven by the global cue, the spatial layout, or their conjunction?

At checkpoints 20, 100, and 170, use frozen independent zero-state branches. Hold physical motor sequence, ordinal inputs, transition scoring, and all parameters fixed. For each familiar anchor `j ∈ {A, B}`, evaluate:

| Cell | Global cue | Layout |
| --- | --- | --- |
| 00 | Familiar j | Familiar j |
| 10 | Combined A+B | Familiar j |
| 01 | Familiar j | Combined A+B |
| 11 | Combined A+B | Combined A+B |

For Novel, run the analogous four cells replacing “Combined A+B” with the repository's held-out novel cue or layout. Preserve the original full Mixed and Novel encodings as reference cells. Reuse identical shared cells rather than treating duplicates as independent observations.

Compute the cue effect, layout effect, and interaction:

\[
I_Y=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

Report them by anchor, scale, checkpoint, and probe family for NLL and RT. Hidden-state displacement and prediction changes are secondary localization measures. A significant interaction can identify sensitivity to a conjunction without proving a particular configural-learning theory.

Audit exactly what “combined layout” means in the actual encoder: channel concatenation, superposition, occupancy, or another operation. Do not silently replace it with coordinate averaging. Preserve input width and familiar endpoints. If global cue and layout cannot be independently varied without changing the encoding, record the confound and implement the smallest explicit factorial adapter in a separate branch, with a familiar-endpoint equivalence test.

## 13. E3 — intensity controls, gradients, and human configuration

Keep the original `[1,1]` and existing `[0.5,0.5]` combined-cue variants distinct. Do not call half amplitude unit-norm matching unless the actual encoded norm supports that label. Export total and channel-wise norms, active-feature counts, overlap, and clipping. Include a familiar-cue/layout duplication or gain control where the representation permits it, so adding familiar features can be separated from adding A/B-conflicting identity information.

Implement a prespecified five-point generalization grid `λ = 0, 0.25, 0.5, 0.75, 1` for each independently manipulable channel. For numeric cue vectors, a convex A→B path is a graded cue control, not the simultaneous Mixed condition. For layouts, use a generator that preserves valid displays and motor demands; invalid interpolated layouts must not be used. If no psychologically interpretable path exists, export the input-space diagnostic under that name and leave the human graded-layout test uninstantiated. Include Mixed and Novel as distinct off-path points.

Run the fixed grid on the original cohort for diagnosis. Any subsequent selected channel or refined range must be frozen before the additional seed cohort is examined. Do not turn this into a broad salience/noise search; the earlier overlap/noise intervention has already exposed a splitting–transfer tradeoff.

Build a human cue-factorial task configuration using familiar-anchor, cue-only, layout-only, and combined changes. Assign the first test manipulation between participants so exposure to one compound does not contaminate a supposedly naive response to another. Counterbalance anchor and preceding context. Keep motor sequence, target availability, timing, and response mapping constant. Match visual complexity where feasible and log the remaining differences. Prepare both Mixed and Novel families; do not force them to share an effect.

**Interpretation:** a channel-specific deficit prioritizes how that information is encoded or learned. A conjunction-specific deficit motivates a configural account. A pure intensity effect identifies an input-magnitude contribution. Comparable human sensitivity strengthens a shared-computation interpretation; a human–network difference localizes a task-representation mismatch. None of these outcomes alone selects a new recurrent architecture.

## 14. E4 — action visibility and human execution

**Question:** Does human transfer depend on externally available action information in a way that the SRN's prediction-based RT readout does not capture?

Build a human experiment with two visibility conditions:

- **Full preview:** the upcoming eight-item sequence is visible using the original task display.
- **One target:** only the currently required response target is revealed; the next target is revealed after the preceding response. Participants can still use whatever sequence knowledge they have learned.

Use visibility as a between-participant factor in the primary design, during both A/B acquisition and testing. This avoids treating participants who have just viewed the entire sequence as naive to it under masking. It does not remove remembered sequence knowledge. Keep the same required responses, context cues, layout geometry, and timing rules. Mask future action information while preserving contextual display structure as far as the original display permits. Export stimulus screenshots showing exactly what information remains available.

After the original A/B acquisition schedule, independently assign the first test context to A, B, Mixed, or Novel. Counterbalance terminal training context A/B and familiar layout assignments. This produces `2 visibility × 4 first-test contexts × 2 preceding contexts`, with layout/response counterbalances layered onto those cells. First-test context is between participants for the primary first-encounter inference. Continue eight test repetitions, followed by separate A and B return blocks in counterbalanced order; later observations are secondary practice/retention outcomes.

Use the human task's original acquisition dose when recoverable. Do not substitute 170 simply because that is the SRN dose. Record actual human schedule and exposure counts. The SRN schedule is a separately labeled comparison.

The A/B first-test arms control for the measurement boundary, pause, and familiar-context switch. Obtain additional local within-person A/B acquisition references using H0, but do not pretend an individual's first Mixed test has directly observed simultaneous A and B controls.

Primary sequence outcome: mean execution interval across the seven scored transitions. Analyze initiation latency, errors, and completion time separately. A common increase in all RTs with restricted preview is insufficient; the key test is whether preview changes the probe disadvantage relative to familiar contexts.

For each probe `c`, estimate the interaction on log RT:

\[
J_c = [Y_{c,one}-Y_{F,one}]-[Y_{c,full}-Y_{F,full}],
\]

where `F` is the counterbalanced, model-estimated mean of familiar A and B test arms at the same boundary. Export separate A and B contrasts too. Use this factorial contrast for inference rather than taking a noisy maximum across different participants. Report local worst-endpoint comparisons only as separately labeled descriptive estimates.

Control for acquisition differences with the same familiar-only methods used in H0, and show the unadjusted randomized contrast. Adjustment is a sensitivity analysis: visibility can itself change learning, so conditioning on acquisition changes the estimand. Check speed–accuracy tradeoffs and whether effects concern initiation, execution, or both.

**Interpretation:** a larger probe disadvantage under restricted preview supports a contribution from available action information/planning. A stable disadvantage across preview conditions suggests this manipulation is insufficient to explain the difference. Neither outcome uniquely identifies a motor mechanism; masking also changes perceptual and planning demands.

## 15. E4 — model-information audit and task engineering

Create an explicit information-timing table for humans and the SRN: what is visible before an action, what enters recurrent prediction, when the target becomes known, what the execution readout receives, and when learning occurs. The present SRN already has ordinal and layout information. Do not assume it is an entirely unguided memory task.

The fixed-readout result addresses readout parameters within the model; it does not validate the mapping from next-token confidence to human execution time. Conversely, feeding the correct target directly into the network before scoring that same target's prediction would make the test tautological. Do not implement that as a transfer rescue. Deliver the information audit and model predictions for the human visibility contrast; a new execution architecture is outside this experiment's required scope.

Reuse an existing human experiment framework if present. Otherwise implement a self-contained local browser task with a deterministic schedule generator and downloadable event data. The common engine must support E2 history/practice, E3 cue-factorial, and E4 visibility configurations without duplicating timing logic.

Required capabilities:

- Deterministic participant assignment from a stored counterbalance code; separate randomization from response-dependent behavior.
- Original response mapping and error progression rules, recovered from task source where available. Explicitly document any new default.
- Monotonic response timestamps, actual stimulus-onset/frame timestamps, key identity, repeated-key suppression, focus changes, and aborted trials.
- Explicit context onset, sequence onset, response, error, sequence end, rest, and block boundary events.
- Separate future-target masking and context-layer rendering; tests must verify no hidden action labels leak into one-target displays.
- Stable export schema, resume/abort handling, and a simulated-participant mode clearly labeled synthetic.

Deliver locally runnable studies and screenshots, not a deployed recruitment service. Actual participant collection is a separate research operation. Do not label synthetic validation as human evidence.

## 16. E5 — prior structural experience

**Question:** Does prior experience that the same sequence can occur across layouts improve transfer while retaining later A/B differentiation?

Compare three arms within each initialization scale:

| Arm | Prior experience | Target-task phase |
| --- | --- | --- |
| P0 | None | Original A/B training |
| P1 | Each prior sequence practiced across multiple layouts | Identical original A/B training |
| P2 | Exposure-matched practice with each prior sequence bound to one layout | Identical original A/B training |

Use eight held-out eight-item prior sequences and eight prior layouts. P1 uses every sequence × layout pairing ten times: 640 sequences. P2 assigns one layout per sequence with a balanced bijection and repeats each pairing 80 times: 640 sequences. Thus each sequence and each layout has identical marginal exposure in P1/P2, and only their pairing structure differs. Counterbalance the bijection across seeds. Hold the target sequence, final A/B layouts, and final Mixed/Novel configurations out of pretraining.

Use the original token vocabulary and input width. Generate valid alternative sequences/layouts through the existing environment, preserving ordinal-channel semantics. Shared component tokens and transitions may be unavoidable; document exact sequence, transition, and geometric overlaps instead of claiming every component is unseen. Exclude exact target sequences and task-defined equivalent rotations/reversals when those are equivalent in the actual task.

Preserve the same physical-token sequence stream across P1/P2 where possible, and balance layout marginals, global-cue exposure, and switch counts. Do not create new cue dimensions for P1 alone. Existing global cue values may be reused with balanced exposure in both arms; the held-out claim then concerns sequences/layouts, not those cue values. Export the pairing matrix and all marginal counts.

If the current encoder cannot represent multiple sequences/layouts without changing dimensions or revealing the next target, mark that version unsupported. Implement an explicit adapter with equal dimensions in all arms only if its target-task P0 branch reproduces the original. Do not silently redefine the prediction task.

## 17. E5 — training boundary, matching, and interpretation

Pretrain with the existing next-token objective and reset cadence. For the primary experiment, carry prediction/input/recurrent/output weights into target training, reset the execution readout to its paired original initialization, and reset optimizer buffers in every arm. Train all arms on the identical target A/B schedule for 170 sequences. This isolates experience in the predictive system from a head start in the two-parameter RT readout. Record the reset policy and verify it tensor by tensor.

P0 receives no pretraining; it is the reference for the benefit of any prior practice. P1 versus P2 is the exposure-matched structural contrast. Their learning difficulty can differ despite equal counts, so report pretraining losses and target acquisition, rather than attributing every difference to abstraction automatically.

Capture pre-target and target checkpoints 20, 30, …, 170 and frozen A/B/Mixed/Novel evaluations. Reuse E1's acquisition-matching implementation across P1/P2/P0, with the same support checks. If an arm has no matching support by 170, continue that arm to the already specified 340/680 cap with identical rules, reporting the extension separately. No hyperparameter search against probe performance.

Primary comparison: P1 versus P2 change in checkpoint-170 probe NLL deficit, separately for Mixed and Novel. Secondary comparisons: RT deficit, first-encounter absolute probe performance, target learning speed, and A/B splitting. Report all seeds, plus continuous plots of transfer change against differentiation change. Do not select only seeds that happen to preserve splitting.

Specifically, the primary paired effect is `G_NLL(P1, 170) - G_NLL(P2, 170)`; negative values favor structured prior experience. It is an arm contrast, not a subtraction of two differently pretrained baseline performances.

**Interpretation:** P1 outperforming P2 at comparable acquisition supports a role for prior cross-layout experience. Similar P1/P2 benefits support a more general practice/optimization contribution. Transfer improvement only when A/B differentiation collapses reproduces the earlier tradeoff. No benefit at this dose rejects this intervention, not every possible prior-experience account.

## 18. Human analysis and power simulations

Provide analysis functions that accept either observed or explicitly tagged simulated human event tables. For first-encounter analyses, aggregate the prespecified seven transitions within participant before group inference; preserve transition records for secondary analyses and error auditing. Repeated-exposure models must include within-participant dependence and the true exposure index.

Primary planned contrasts:

| Human configuration | Primary contrast |
| --- | --- |
| H0 | First-ever probe disadvantage versus later probe exposure, with local-reference uncertainty |
| E2 | Preceding A/B context effect on first probe, and probe repetition effect |
| E3 | Cue × layout interaction at first exposure, separately for Mixed and Novel |
| E4 | Probe × visibility interaction at first exposure, separately for Mixed and Novel |

Fit log-RT models for inference and report marginal effects in the original units with the transformation stated. Retain error-rate analyses and initiation/execution separation. For new experiments, identify the randomized contrast separately from adjusted and repeated-exposure results. Inspect design rank before fitting interactions; do not report coefficients for confounded schedule factors.

Build Monte Carlo power/precision simulations using the actual counterbalance cells, event counts, exclusion rules, and intended model. Use existing human data for nuisance variance and within-participant correlation when defensible. Do not estimate the new manipulation's effect from an unrelated SRN RT scale. Simulate standardized interaction sizes 0, 0.10, 0.20, and 0.35 residual SD, including conservative missingness and error scenarios.

For E4, report total-N options 64, 128, 256, 384, and 512, balanced over its 16 core cells. For E2/E3, use feasible multiples of their actual cell counts over a comparable range. Report type-I error, power, interval width, and estimable participants per cell. Use 200 simulation draws for implementation checks and 1,000 for final planning estimates, including Monte Carlo uncertainty. No sample size is declared sufficient solely because it appeared in this PRD. Deliver the planning table and a configuration field for the chosen recruitment target.

## 19. Required data contracts

Keep raw transition/event records, sequence aggregates, contrasts, and summaries separate. Use Parquet or CSV according to repository conventions, with a schema manifest.

| Table | Minimum additional fields beyond the source metrics |
| --- | --- |
| `run_manifest` | experiment, cohort, seed, parent checkpoint/hash, config/code hash, initialization, training dose, input layout/sequence IDs, manipulation values |
| `transition_results` | branch ID, completed training sequences, pre/post-update status, context/cue/layout IDs, repetition, primer, weight/state policy, transition, target, prediction, probability, NLL, RT, hidden state |
| `sequence_results` | sequence ID, scored-transition count, means, errors, exposure counts, update count, readout parameters, parent/descendant relationship |
| `paired_contrasts` | exact endpoint branch IDs, comparison type, probe, A/B gap, absolute benefit reference, NLL/RT deficits, uncertainty cohort, interpretable flag |
| `acquisition_matches` | candidate/selected checkpoints, matching variables, calipers, distance, balance, support/rejection reason |
| `human_events` | dataset/participant, assignment, absolute event order, onset/response times, visibility, cue/layout/context, sequence and block IDs, full-stream exposure count, accuracy/exclusion flags |
| `human_local_references` | probe event, source A/B rows, fitting split, model version, reference values/intervals, temporal distances, extrapolation and validation flags |
| `branch_status` | implementation status, evidence status, prerequisites, outputs, failed checks, scientific interpretation limit |

Never average branches with different exposure histories under a shared “frozen probe” label. Keep synthetic rows in separate files and mark them in every derived output.

## 20. Runner, configuration, and execution order

Implement one top-level driver, adapting names to repository conventions:

```bash
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage audit
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage human_audit
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage e1
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage e2
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage e3
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage e4
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage e5
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage report
```

Provide `--stage all`, `--dry-run`, `--smoke`, and resumable execution. `--dry-run` reports source availability, exact branch counts, seed/checkpoint lists, and estimated runtime from a small timing run. Stages declare prerequisites; the E4 stage builds and validates the task, runs power simulations, and analyzes real data only when explicitly provided. It must not wait indefinitely for participants.

The resolved configuration must expose all budgets and choices specified above: 170/340/680 training caps; checkpoint sets; eight probe repetitions; primer and retention schedules; cue/layout grids; 640-sequence pretraining; original/additional seed IDs; bootstrap/simulation seeds; human design cells; and numerical tolerances. Save generated schedules before running outcomes.

Execution order is audit, H0, E1, E2, E3, E4, E5, then synthesis. E4 engineering and E5 preparation can proceed while other calculations run, but outcome-dependent refinement must respect the cohort separation. Reuse E1 measurement utilities throughout. Do not add an architecture search when a manipulation fails.

## 21. Required figures

Produce standalone PNG plus vector PDF/SVG figures and the exact plotted data. Use shared units and axis limits for panels intended for direct comparison; label any necessary scale difference. Error intervals must name their unit and construction. Show individual seeds/participants where heterogeneity affects interpretation.

1. **Reference and human measurement audit:** actual-order human sequences with first/later probe events and local A/B references; reference SRN overlays at corresponding completed training counts.
2. **Acquisition versus transfer:** probe NLL and RT against both training time and familiar acquisition, with strict matches and unmatched cases visible.
3. **Transfer and differentiation:** absolute benefit and relative deficit alongside A/B behavioral and hidden-state separation; avoid a single success score.
4. **Probe experience:** the four state/update cells across repetitions, faceted by primer and probe; zero-state retention and A/B return results.
5. **Cue decomposition:** cue/layout factorial means and interactions, with input-norm controls and fixed generalization curves.
6. **Action visibility:** task screenshots and first/later probe × visibility effects when real data exist; power plots separately labeled as simulations beforehand.
7. **Prior experience:** pretraining pairing matrices, target acquisition, matched transfer, and change in A/B differentiation for P0/P1/P2.
8. **Evidence synthesis:** a compact table of each hypothesis, its discriminating contrast, observed result/uncertainty, and remaining alternative explanations.

Preserve sequence-time overlays as the main performance figures. Transition geometry can explain a result but must not displace the primary event-level comparison.

## 22. Required verification

Write targeted tests for risks that could reverse the scientific conclusion:

- Baseline replay reproduces stored learning and probe metrics within justified precision; cached runs resolve to the correct parent hashes.
- Frozen evaluation changes no model/readout/optimizer/RNG state and does not change later acquisition.
- Every sequence is scored before its authorized update; learning/reset/carry/primer policies match the manifest.
- Frozen/reset repetitions are identical, carry is actually carried, and resetting removes inherited state.
- Adaptation branches are isolated; parallel-practice endpoints and probe-adapted same-parameter endpoints cannot be confused.
- Per-transition NLL and RT reconstruct sequence means; paired endpoint max is taken within seed before aggregation.
- Changing cues/layouts leaves the physical target sequence and scored transitions unchanged; familiar factorial cells reproduce the baseline encoder.
- Acquisition selection cannot access probe columns. Failed matches remain visible. E1 continuation preserves optimizer and schedule phase.
- Pretraining has the exact marginal exposure counts, target holdouts, paired initializations, and intended boundary resets.
- Human chronology and first exposure are reconstructed before exclusions; excluded early events cannot become nonexistent exposure.
- Human task schedules are balanced and reproducible; one-target displays hide future action information; timestamps, key handling, focus loss, and resume behavior are exercised end to end.
- Bootstrap resampling preserves whole participant/seed clusters; synthetic outcomes cannot enter empirical reports.

Use small constructed counterexamples for chronology, matching leakage, paired aggregation, and branch contamination. Do not expand testing into a redundant suite of assertions that merely restate function implementations.

## 23. Interpretation matrix and stopping rules

| Pattern | Supported conclusion | Claim to avoid |
| --- | --- | --- |
| Human first-encounter advantage shrinks after timing/exposure audit | Measurement or probe experience contributed to the apparent mismatch | Humans and SRNs are therefore identical |
| E1 deficit shrinks with acquisition and valid balance | Acquisition stage contributes | Initialization has no representational effects |
| E2 frozen carry improves probes | Performance depends on recurrent history | The SRN implements Bayesian context inference |
| E2 updates improve zero-state retained performance | Additional learning contributes | Humans require the same optimizer-like mechanism |
| E3 one channel or conjunction drives the deficit | Probe encoding/association is a useful intervention target | Every off-segment hidden state is mechanistically wrong |
| E4 preview changes the human probe disadvantage | Available action information contributes to transfer expression | Prediction plays no role in human skill |
| E5 structured prior practice improves matched transfer with differentiation retained | Prior cross-layout experience is a plausible missing ingredient | The tested pretraining recreates human lifetime experience |
| Transfer improves while A/B differentiation disappears | The intervention exposes a differentiation–transfer tradeoff | The joint human target is solved |
| No valid acquisition match, confounded human design, or missing chronology | The affected contrast is inconclusive | Evidence for absence of the proposed mechanism |

Technical integrity failures stop affected descendants until fixed. Scientific negative results are reported and do not block other packages. Training stops at the stated caps; no extra seeds, epochs, cue strengths, or pretraining doses are added because an effect is disappointing. Optional follow-up variants must be proposed explicitly in the report rather than silently folded into this program.

## 24. Deliverables and definition of done

Deliver the reusable runner/configuration, repository-integrated experiment modules, human task engine/configurations, schemas, tests, source audit, generated schedules, computational result tables/checkpoints, figures, human analysis/power code, and a final `TRANSFER_EXPERIMENT_PROGRAM_REPORT.md`.

For each work package, the report must state: question; exact intervention; controls; implementation/evidence status; sample and exclusions; primary contrast with uncertainty; acquisition and A/B differentiation checks; result; remaining explanation; and a justified next decision. Include exact reproduction commands and machine-readable branch status.

Implementation is complete when all five experiments and H0 have runnable, validated code and the specified outputs, and new-human tasks have verified analysis/power pipelines. Computational execution is complete only when the specified runs with available sources have finished. Empirical completion of new-human comparisons requires actual observations. If missing model source/checkpoints prevent real computational execution, mark that package blocked even if its synthetic integration test passes. A human-data dependency is not a reason to leave unrelated computational work unfinished.

No requirement exists to obtain intermediate probes or a particular mechanism. A clear demonstration that a plausible explanation fails is a successful experiment.

## 25. Literature motivating the manipulations

These studies motivate discriminating experiments; they do not establish that this sequence task uses the same mechanisms.

- **Acquisition and differentiation:** [Holton et al., *Humans and neural networks show similar patterns of transfer and interference during continual learning*](https://www.nature.com/articles/s41562-025-02318-y), online 2025, journal issue 2026. Human and network differences in representational reuse can accompany transfer–interference tradeoffs. This motivates evaluating acquisition and differentiation jointly, not assuming High/Low labels identify a universal learning regime.
- **History and memory expression:** [Heald, Lengyel, and Wolpert, *Contextual inference underlies the learning of sensorimotor repertoires*](https://www.nature.com/articles/s41586-021-04129-3), 2021. The distinction between memory expression and updating motivates E2; a carry effect alone would not validate their inference model.
- **Compound cues:** [Avraham et al., *Contextual effects in sensorimotor adaptation adhere to associative learning rules*](https://elifesciences.org/articles/75801), 2022. Contextual conditioning and compound-cue effects motivate separating cue components and their interaction in E3.
- **Generalization gradients:** [Sarwary et al., *Generalization and transfer of contextual cues in motor learning*](https://pubmed.ncbi.nlm.nih.gov/26156381/), 2015. Graded contextual generalization motivates testing a curve rather than judging only one combined or novel point.
- **Prediction and execution:** [Kashefi, Diedrichsen, and Pruszynski, *Motor Sequence Learning Involves Better Prediction of the Next Action and Optimization of Movement Trajectories*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12352532/), 2025. Their reaching experiments distinguish benefits of upcoming-action information from continued improvement in movement execution, motivating E4 without assuming reaching and finger tapping are equivalent.
- **Prior experience:** [Flesch et al., *Comparing continual task learning in minds and machines*](https://www.pnas.org/doi/10.1073/pnas.1800755115), 2018. Prior representational experience can alter later task learning; E5 tests a specific controlled version in the present sequence environment.

## 26. Final scientific target

Determine which measured differences arise from acquisition, probe history, cue interpretation, action availability, or prior experience, and which remain after those manipulations. Preserve the distinction between demonstrated partial transfer, its remaining deficit, and the stronger claim of a mechanistic explanation of human behavior.
