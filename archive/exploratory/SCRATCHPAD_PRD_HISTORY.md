# Probe-modeling implementation scratch pad

## Scope

- Implement the PRD for the combined-cue (`mixedBreak_clean.csv`) and novel/third-context (`threeContextsBreak_clean.csv`) schedules.
- Reuse the starter repository's deterministic seed, column-inference, export, metric, and plotting conventions, but do not reuse its sequence-prediction RNN as the behavioral model.
- Follow RED -> GREEN -> REFACTOR, with tests serving as scientific contracts.

## Initial repository audit (2026-08-24)

- The directory is not currently a Git working tree, so changes cannot be summarized with `git diff`/`git status`.
- `data/mixedBreak_clean.csv` contains 363,108 keypress-level rows, 223 participants, and explicit Maze, Melody, Mixed, and No Context labels.
- The combined-cue experimental schedules are counterbalances 2 and 3. They include two contiguous `Mixed Training` blocks (10 sequences each) and a later `Mixed Test` block.
- Counterbalances 1 and 6 are A/B-only controls with a later Mixed test; 4 and 5 contain No Context training/test and are an optional fallback benchmark, not the primary combined-cue cohort.
- `data/threeContextsBreak_clean.csv` contains only its header (zero observations). The implementation can support and validate third-context data, but a human-data third-context run cannot be completed until observations are present.

## Decisions

1. Aggregate raw keypress rows into one row per participant/block/sequence. Mean RT uses correct, finite, positive keypress RTs; sequence error is the proportion of incorrect keypresses.
   Preserve the starter analysis's complete-case convention by retaining the modal normalized schedule length within each cohort. In the combined data this excludes two participants (`MG272655`, `MG810155`) who completed concatenated 34-block schedules while the modal schedule is 17 blocks.
2. Exclude warmup labels from the modeling timeline. Validate every remaining context label explicitly; unknown labels raise an error.
3. Canonical A/B labels are Maze/Melody. `Mixed` maps to probe type `Both`; labels containing `Novel` or `Third Context` map to `Novel`. `No Context` is accepted only as the external fallback benchmark and maps to `Default` internally.
4. Preserve all probe-context blocks chronologically. The first two `* Training` probe blocks are the primary block-1/block-2 comparison; later `* Test` probes remain identifiable for secondary prediction/carryover analyses.
5. Fit counterfactual A/B trajectories from A/B rows only. Probe RTs never enter A/B skill estimation.
6. Model log RT by default, while exporting mixture indices on raw RT, log RT, and speed scales.
7. Implement explicit mechanisms before the neural gate: equal/static mixture, one-dimensional contextual inference, responsibility-weighted updating, and a third-state model.
8. Treat the tiny GRU as a predictive ceiling/discovery model. A non-recurrent logistic gate is always fit alongside hidden dimensions 1, 2, and 4.
9. Write large CSVs through a local temporary file and atomically replace the destination. Direct row-wise writes into the Box-synced directory produced long stalls and could leave partial exports after interruption.
10. Mixture indices are undefined when the estimated A/B counterfactual RTs differ by less than 2% (`|log A - log B| < log(1.02)`). Export a reliability flag and mask all three ratio scales in those rows rather than displaying unstable extreme ratios.
11. Recovery acceptance requires both >=80% accuracy and at least 20 simulations per generating mechanism. A one-replicate smoke run is labeled `verification_only`, even if its confusion matrix is diagonal.
12. Expose cue composition explicitly: M2a uses relative normalized evidence with a 50/50 fallback at [0,0]; M2b represents Both as [1,1] and fits a cue-magnitude effect while keeping relative responsibility at 50/50.
13. The Combined analysis is allowed to run independently while the Novel file is pending. Use `--combined-only`; this omits Novel from the run record instead of treating its empty file as part of the analysis.

## Progress log

- [x] Read PRD and starter-code interfaces.
- [x] Audited raw data schemas and actual mixed-condition schedule.
- [x] RED/GREEN 1: normalized data contract.
- [x] RED/GREEN 2: leakage-free A/B skill trajectories.
- [x] RED/GREEN 3-5: explicit retrieval/update mechanisms.
- [x] RED/GREEN 6-7: tiny gate and interventions.
- [x] RED/GREEN 8: model recovery and acceptance gate.
- [x] End-to-end exports, figures, and smoke tests.
- [x] Run the available mixed human data and document the missing third-context observations.

## Validation and handoff

- `35 passed` in the complete test suite after the final M2 cue-composition coverage audit.
- Verification output: `outputs/probe_analysis_verification/`.
- Verification configuration: one neural seed, 60 epochs, one recovery simulation per mechanism; `analysis_tier=verification_only`.
- Combined data: 12,920 normalized rows, 76 participants, 2,280 probe sequences; 20 probe rows have an ill-conditioned A/B denominator and are flagged/masked.
- Novel data: skipped explicitly because `threeContextsBreak_clean.csv` has zero observations.
- All eight PNG figures were rendered and visually inspected.
- The one-replicate recovery smoke matrix is diagonal, but `model_recovery_identifiability.csv` correctly marks every mechanism non-identifiable because the minimum simulation count was not met.
- Preliminary combined-only verification results must not be treated as final inference. Responsibility updating had the lowest explicit-model BIC in this smoke run, but held-out differences were small and the third-context contrast is unavailable.

## Combined-only final run

- User authorized moving forward without Novel/Third data. Added `--combined-only`, which omits Novel from the input and run record rather than logging it as an empty source.
- Completed output: `outputs/probe_analysis_combined_final/` (`analysis_tier=final_config`).
- Configuration: 20 neural seeds, 300-epoch ceiling with early stopping, and 20 recovery simulations per generating mechanism.
- Artifact integrity passed: 76 participants, 12,920 sequence rows, 90,440 explicit predictions, 306,000 held-out gate-state rows, 80 recovery fits, and all eight figures.
- Recovery at noise SD 0.05 was 100% for equal mixture, contextual inference, responsibility updating, and third-state generation (20/20 each), so all four meet the configured identifiability criterion on the Combined empirical schedule.
- Responsibility-weighted updating had the best explicit-model BIC (`-24695.75`), followed by third state (`-24648.84`) and static mixture (`-24620.20`).
- Held-out overall RMSEs among the leading explicit models were extremely close: responsibility update `0.23163`, static mixture `0.23176`, contextual inference `0.23179`. Responsibility updating improved return-to-A/B RMSE (`0.26078` versus `0.27144`) but not probe-only RMSE.
- The recurrent cue gate was stable across seeds and improved over the non-recurrent cue gate (mean RMSE `0.22920` versus `0.23305`). The 1D distilled dynamics captured a mean `R^2=0.961`; 2D/4D gains were small (`0.22898`/`0.22865`).
- Behavioral mixture indices showed a large predecessor effect in both primary probe blocks: approximately `0.67` difference in block 1 and `0.68` in block 2 between previous-A and previous-B schedules.
- Interpretation: the Combined dataset contains history dependence and return-to-A/B aftereffects consistent with responsibility-weighted retrieval/updating, but the leading explicit models are nearly tied in held-out overall prediction. Treat the updating account as favored, not decisively proven, until robustness/model assumptions are reviewed.

## PRD2 replacement analysis

### Initial decisions

1. Preserve v1 code/artifacts for provenance, but make the canonical analysis entry point use the epoch-structured v2 pipeline.
2. Raw `sequenceNumber` is zero-based (0–9). Export `original_sequence_in_block=sequenceNumber+1`, remove original sequence 1, and model `analysis_sequence_in_block=1..9` (original 2–10).
3. In the raw task, every retained sequence has at least eight correct keypress records, although many contain intervening incorrect attempts. Define `sequence_correct` as a completed sequence with at least eight correct, finite, positive keypress RTs; compute RT from those correct keypresses only. Requiring zero erroneous attempts would remove every participant and is inconsistent with the existing sequence-level export.
4. A v2 participant is complete only when every retained block contains exactly nine completed sequences and the participant has the modal 17-block schedule. Never pad an invalid block.
5. `Post_Mixed_Return` is one block—the first established A/B block after Mixed 2—not a recurring generic aftereffect category.
6. The primary behavioral deviation is stable log-RT distance from the A/B midpoint. Per-observation mixture ratios remain secondary and retain the 2% counterfactual-separation mask.
7. Mechanism parameters are fit without Post-Mixed Return or Mixed Test. Prospective Test A calibrates through Mixed 1; Tests B/C calibrate through Mixed 2. Held-out participants never fit population mechanism parameters.
8. Counterbalance is represented both by its raw code and a mirrored `A_before_Mixed`/`B_before_Mixed` history code. Claims use “counterbalance/history,” not an unqualified “previous-context effect.”

### PRD2 progress

- [x] Milestone 1: first-sequence removal and correct nine-sequence blocks.
- [x] Milestone 2: epoch tagging.
- [x] Milestone 3: restart-free, probe-free A/B counterfactuals.
- [x] Milestones 4–8: explicit M0–M6 and mirroring.
- [x] Milestone 9: epoch-specific/prospective evaluation.
- [x] Milestone 10: neural reset interventions and progressive held-out distillation.
- [x] Multi-regime v2 recovery on a 17×9 representative schedule retaining both counterbalances.
- [x] Required v2 exports and figures implemented.

### Implementation notes

- M0–M6 are schedule-driven generative predictors: updates never consume a held-out participant's observed Mixed RT. Mixed experience alters either the retrieval logit (M4), the A/B offsets (M5), or an independent C offset (M6), yielding distinct downstream signatures.
- Main explicit fits use training participants' Mixed 1/2 observations. Mixed Test and the first post-Mixed A/B block are prospective. Counterbalance-specific diagnostic fits are exported separately from the combined fit.
- A↔B equivariance is encoded through `label_mirror` and tested at prediction precision. Cross-counterbalance evaluation fits one history and predicts the other after explicit label reversal.
- Neural N2–N4 interventions reuse the exact fitted N1 weights; only hidden state is reset. This isolates carried state from parameter differences.
- Recovery uses one empirical representative schedule from each counterbalance to keep final multi-regime simulations tractable while preserving the exact 17 blocks, 9 observations per block, epoch placement, and both histories.
- The v2 smoke pipeline passed with all 18 required exports. It took about 2 minutes on the four-participant fixture because it exercises all explicit prospective fits and six-mechanism recovery.
- Added flushed, timestamped terminal progress reporting after the first full run appeared idle. Reporting covers load/preprocessing, every five A/B participant fits, each explicit/temporal/counterbalance model fit, every neural seed/model, every recovery simulation and candidate fit, individual output writes, and figure rendering. `--quiet` disables it. A real-data startup check confirmed messages through participant 10/76 before the check was intentionally interrupted.

## PRD3 similarity-weighted retrieval analysis

### Isolation and data decisions

1. PRD3 is additive: all new implementation lives under `retrieval_modeling/` with a separate `scripts/run_retrieval_analysis.py`. The currently running PRD2 modules and CLI remain untouched.
2. PRD3 imports PRD2 preprocessing and A/B skill fitting as frozen dependencies; it does not copy or alter their optimization logic.
3. The pending third-context source may contain both Novel/Third and No Context probes. The new adapter treats these as distinct retrieval targets in one dataset: `Novel=[0,0,1]` and task-absent `Default=[0,0,0]`. A/B and Combined/Both retain the preregistered vectors.
4. The Novel adapter supports mixed Novel and No Context labels without changing the older loader, which intentionally accepts only one probe family at a time.
5. Joint Combined+Novel parameters are mandatory for the primary fit. A partial-data mode may produce development artifacts, but it must mark cross-dataset transfer and the paper-level joint claim unavailable.
6. Participant nuisance intercepts are estimated from A/B residuals only and frozen before retrieval fitting. They cannot absorb probe behavior.
7. M4 tests absolute evidence through residual variance before any mean-cost extension. M6/default state is implemented as a conditional falsification/recovery model because No Context is an explicit target, but it is not promoted over M2–M5 without Novel-model failure.
8. Model likelihoods operate on individual sequence log RT. Continuous models use a Gaussian interpolation likelihood; categorical retrieval uses an A/B Gaussian-mixture likelihood.

### PRD3 progress

- [x] RED/GREEN context features, geometry, symmetry, and readouts.
- [x] Novel + No Context data contract and frozen A/B wrappers.
- [x] Explicit joint model ladder and individual-sequence likelihoods.
- [x] Cross-dataset transfer and participant A/B-separation diagnostic.
- [x] Shared tiny-gate ladder and distillation.
- [x] Multi-regime recovery, outputs, figures, and standalone CLI.
- [x] Complete synthetic test suite and visual artifact inspection.

### PRD3 validation and handoff

- New code is isolated in `retrieval_modeling/` and `scripts/run_retrieval_analysis.py`; no PRD2 model/fitting module was edited for this PRD.
- Seventeen focused PRD3 tests pass: task-only feature provenance, Both/Novel/Default geometry, kernel limits, A/B symmetry, continuous and categorical sequence likelihoods, history priors, evidence-scaled variance, participant structural predictions, joint/shared parameters, bidirectional transfer, No Context loading, probe leakage, tiny gates, distillation, high-signal model recovery, and the complete artifact contract.
- Synthetic end-to-end testing produced all 15 required CSVs, the required run-config JSON, five supporting audit tables, and eight figures. Figures were rendered and visually inspected; the geometry plot was revised to a true 3D cue space so Novel and Default do not collapse visually.
- Full repository regression result: `67 passed, 1 pre-existing pandas FutureWarning` in 162.30 seconds. This includes all PRD1/PRD2 tests, confirming the additive implementation did not regress the running analysis.
- At initial handoff no real Combined/Novel retrieval analysis was launched because `threeContextsBreak_clean.csv` was header-only. The populated-file audit and corrections below supersede that initial data-status note.

### 2026-08-28 populated three-context audit

- Audited every literal `trialType`, participant schedule, and counterbalance in the now-populated `threeContextsBreak_clean.csv`.
- Corrected the actual third-context contract: `Space Training` and `test spaceA` are `Novel`; No Context remains `Default`; all three warmup labels are excluded.
- The literal `test spaceA` spelling revealed why CB1/CB6 were initially missed: a regex required a word boundary after `space`, but the export concatenates `spaceA`. The selector now recognizes that exact label.
- Cohorts are inferred from actual exposure rather than assumed from counterbalance: CB2/3 `Novel_training`, CB4/5 `Default_training`, and CB1/6 `Novel_test_only_control`. The mixed file is labeled analogously as `Both_training`, `Default_training`, and `Both_test_only_control`.
- Post-exclusion validation: Combined has 218 participants/33,354 nine-sequence rows (76 Both-trained, 76 Default-trained, 66 Both test-only); the three-context file has 92 participants/14,076 rows (35 Novel-trained, 27 Default-trained, 30 Novel test-only). Every retained participant has 17 complete blocks and every block has original sequence positions 2–10.
- Conservatively retained the existing complete-case policy for reused IDs. `MG578955`, `MG600555`, and `MG783355` each contain two complete third-context sessions under one ID and are excluded rather than treated as independent participants.
- Added explicit phase, exposure-cohort, control, and probe-entry labels to audit/prediction outputs. Added run-specific label-dictionary and cohort-summary exports and cohort-specific model-comparison/transfer metrics.
- The changes remain isolated to PRD3 code and tests; no PRD2 fitting code was changed.

## Partial-observability GRU PRD

### Implementation decisions

1. Added a standalone `partial_observability_gru/` package and
   `scripts/run_partial_observability_gru.py`. The original task SRN and all
   PRD2/PRD3 model-fitting code remain unchanged.
2. Context prototypes are resampled around a random center and direction in
   every episode. This varies cue identity while preserving a known exact
   midpoint. Both is the vector sum, Novel is an orthogonal extrapolation from
   the midpoint, and Default is the zero vector, so these OOD probes remain
   distinct rather than collapsing to the same observation.
3. Randomly relabeled A/B prototypes are otherwise exchangeable. The network
   therefore receives the PRD-permitted *previous* practice indicator as two
   lagged action features. It never receives current `z_t`; current context
   can switch after the lagged action. This anchors A/B labels without target
   leakage.
4. Input order is current context observation, lagged A/B practice indicator,
   `y_(t-1)`, feedback-available flag, and episode-boundary flag. Batch
   construction copies lagged feedback, so changing `y_t` afterward cannot
   change input or prediction at time `t`.
5. The exact Bayesian filter first applies the Markov transition, then current
   context likelihood, and only after recording `q*_t`/`y*_t` incorporates
   observed `y_t` for the next timestep. Known latent skills are used only by
   this benchmark and evaluation diagnostics.
6. O1/O2 use direct deterministic/Gaussian GRU readouts. O3 is the supervised
   context-inference reference. O4 and O5 use stochastic and distributional
   two-state heads only as mandatory objective controls; they are not the
   primary mechanistic architecture.
7. Smoke, pilot, and full profiles separate software validation from final
   inference. Pilot runs the entire 4×3×5×6 grid with five seeds; full uses 20
   seeds. No large factorial run is launched automatically.
8. Human-style Midpoint/Both/Novel/Default observations are constructed only
   after training. Optional human RT input is merged into validation outputs
   and plots but never passed to the optimizer.

### TDD record

- RED: new environment/model/analysis/pipeline tests initially failed because
  `partial_observability_gru` did not exist.
- GREEN: focused tests cover independent seeded skill updates,
  uncertainty-dependent context accuracy, exact symmetric posterior, analytic
  Bayes risk, lagged-feedback leakage prevention, all objective-head losses,
  same-time target invariance, distributed hidden decoding, posterior/belief
  alignment, and known-direction intervention.
- GREEN: a reduced all-objective end-to-end smoke test writes all required
  exports, `run_config.json`, and eight figures.

### Validation

- Thirteen focused partial-observability tests pass, including a tractable
  trained deterministic/classifier acceptance test and zero-shot probe
  contract.
- A real CLI smoke run completed 10 perfect/moderate × five-objective
  conditions, wrote 14 CSVs (13 required plus training history), the run
  configuration, and all eight figures. Artifact schemas and zero-shot flags
  were checked; the human-style and belief-intervention figures were visually
  inspected.
- Byte compilation passed with a writable external pycache.
- Full repository regression: `82 passed, 1 pre-existing pandas
  FutureWarning` in 240.24 seconds. No legacy SRN, PRD2, or PRD3 test regressed.
- No pilot or full factorial experiment was launched; those profiles are left
  for the user because they span 1,800 and 7,200 trained networks,
  respectively.

## PRD v1 Elman-SRN partial-observability proof of concept

### Direction and isolation

1. The newer `PRD_v1_SRN_partial_observability.md` supersedes the interrupted
   GRU-v1 direction. The implementation is isolated in
   `srn_partial_observability_v1/` with a new CLI; no original sequence-SRN,
   PRD2, PRD3, broad partial-observability GRU, or existing model-fit module was
   changed.
2. The earlier `gru_partial_observability_v1/` development package remains
   untouched for provenance, but it is not imported by or used in this SRN
   analysis.
3. Fixed cue identities (`A=-1`, `B=+1`) make A/B states identifiable without
   a lagged action input. The exact network input is `[x_t, y_(t-1),
   previous_outcome_mask, episode_start]` as required.

### Scientific implementation decisions

1. The primary model is `torch.nn.RNN(..., nonlinearity="tanh")` with four
   hidden units and a scalar linear readout. It has no gate, expert head,
   explicit posterior, auxiliary loss, or built-in averaging operation.
2. The two latent states receive independent initial values, rates, and
   asymptotes. Only the state active on trial `t` updates after that trial.
3. Prediction and the exact posterior mean are recorded before `y_t` is
   observed. Outcome evidence is incorporated into the exact filter only for
   the next timestep, and `y_t` first appears in the SRN input at `t+1`.
4. The Uncertain cue SD is 1.1, yielding approximately 81.8% equal-prior Bayes
   context accuracy. Perfect cues use SD 0.
5. Global mean and context-only OLS are fit on generated training episodes.
   Previous outcome uses only the lagged input, and the oracle Bayesian
   predictor is evaluation-only.
6. Ambiguity histories are clear and identical across conditions. Only the
   final cue is inverted from the exact filtering prior to attain requested
   `q*` values `.05, .10, .25, .50, .75, .90, .95`. This isolates the causal
   Perfect-versus-Uncertain comparison from an entire noisy out-of-distribution
   history.
7. Implied responsibility is exported without clipping and is undefined when
   `|S_A-S_B|<.03`. Both posterior-mean and responsibility summaries include
   R², slope, intercept, and MAE.
8. Hidden probes split whole episodes, never timesteps. Belief intervention is
   gated on held-out `q*` decoding. The analytic posterior-sampling risk is
   checked against the exact two-times posterior-mean risk and Monte Carlo.
9. Hidden probes, dimensionality, and GRU controls obey staged gates. Every
   skipped output and figure is still emitted with `status=not_run` and a
   reason. `force_staged_analyses` exists only for testing every software path.
10. The five-seed MVP and twenty-seed final profile are separate so a costly
    final run cannot start implicitly before the MVP result is reviewed. The
    final CLI additionally requires the MVP `run_config_v1.json` and refuses
    to start unless its behavioral stage gate passed.

### TDD progress

- [x] RED: the new SRN package imports failed.
- [x] GREEN: independent states, cue calibration, no current-target leakage,
  exact filtering, Elman architecture, matched GRU interface, ambiguity grid,
  risk identity, held-out probes, trained baseline acceptance, and complete
  staged pipeline contracts.
- [x] Nine core environment/model/training tests pass.
- [x] Three end-to-end/staging tests pass, including forced dimensionality,
  matched GRU paths, and the enforced MVP-to-final gate (12 focused tests in
  total after final refactoring).
- [x] Real CLI smoke wrote all 14 CSVs, the JSON decision record, explicit
  report, and nine figures; schemas, exact q grid, risk ratio, staged status,
  and figures were inspected.
- [x] Full repository regression: 101 passed with one pre-existing pandas
  FutureWarning in 199.45 seconds.
- [ ] Five-seed MVP and twenty-seed final scientific analyses are intentionally
  left unrun until software validation and MVP staging are complete.

## PRD v1.2 SRN environmental pressures

### Scope and isolation

1. Implemented this direction as a new `srn_environmental_pressures/` package
   and `scripts/run_srn_environmental_pressures.py`. Existing sequence-SRN,
   v1 partial-observability SRN, PRD2, PRD3, and GRU fitting code was not
   modified.
2. No human data is loaded. Every training and diagnostic outcome is generated
   by the v1.2 ecology.
3. The scientific discovery profile fixes an eight-unit tanh Elman SRN, Adam
   at learning rate `.005`, 16 blocks, exactly 10 trials per block, one weight
   update per trial, and a batch of 128 simulated runs. None of those settings
   appears in the environmental grid.
4. The smoke profile shortens the ecology only for software verification. Its
   outputs and report are explicitly non-inferential; settings remain identical
   across all smoke conditions.

### Generative and Bayesian decisions

1. Nominal context follows deterministic `A×10 → B×10` pairs. The active latent
   state is sampled trialwise with `P(z=c)=context_validity`; the network never
   receives nominal or latent labels.
2. Cue SD is calculated from requested equal-prior Bayes accuracy using the
   inverse Normal CDF with fixed means `-1/+1`. This makes `.97/.90/.80/.70/.60`
   accuracy the registered environmental axis.
3. The full oracle combines the nominal-context validity prior with the current
   Gaussian cue likelihood. Exact validity `v=1` retains structural probability
   zero/one under arbitrary finite cue conflict; clipping cannot manufacture an
   impossible latent state.
4. S1 uses fixed global states centered at `.5`; S2 draws an exactly separated
   pair per simulated run; S3 begins at the configured separation and updates
   only the active state toward independent asymptotes.
5. The current target is generated before state updating and first enters the
   SRN at `t+1` through the lagged outcome column. Inputs remain exactly
   `[x_t, y_(t-1), previous_outcome_mask, session_start]`.
6. Online training carries the hidden state through the entire session while
   detaching it between fixed trialwise Adam updates. Snapshots are recorded
   after each completed 20-trial A/B pair.

### Diagnostics and staging

1. Each diagnostic crosses `.05/.10/.20/.35/.50/.65/.80/.90/.95` requested
   posterior levels with both nominal contexts and matched held-out histories.
   It exports the phase prior, cue likelihood ratio, cue-only posterior, full
   posterior, all candidate predictions, SRN prediction, and unclipped implied
   responsibility.
2. The success gate implements every numeric PRD criterion and additionally
   requires explicit rejection of a constant midpoint. Full Bayes must be
   closer to the SRN than phase-only, cue-only, and hard-MAP predictions.
3. Discovery uses seeds `0..4`; confirmation uses fresh seeds `1000..1019`,
   with overlap rejected before training. Conditions need four of five and then
   sixteen of twenty passing seeds.
4. Confirmation, S1/S2/S3 and outcome-noise follow-ups, four environmental
   ablations, held-out-run probes, and prior/likelihood interventions are gated.
   Skipped stages always produce `status=not_run` artifacts and figures.
5. `force_staged_analyses` exists only for the smoke profile so all software
   paths can be tested without weakening scientific gates.

### TDD and validation progress

- [x] RED: all new imports initially failed because the package did not exist.
- [x] GREEN: exact schedule/exposure, eight-unit architecture, validity, cue
  distribution/accuracy, state separation, leakage timing, full oracle,
  cue-conflict shortcuts, midpoint rejection, checkpoint cadence, 120-condition
  registry, and seed separation.
- [x] GREEN: ordinary smoke and forced-stage pipeline contracts, including all
  confirmation, ablation, probe, and intervention paths.
- [x] Real CLI smoke wrote and validated all 19 CSV schemas, fixed-exposure
  audit fields, JSON, report, exact diagnostic grid, shared checkpoint battery,
  and nine figures.
- [x] Forced-stage CLI smoke validated fresh confirmation seeds, 11
  ablation/follow-up conditions, eight held-out probe targets, and separable
  prior/cue intervention outputs. Staged figures were visually inspected.
- [x] A full 16-block, 128-run one-condition check completed and exercised the
  optimized checkpoint battery at scientific horizon without altering metrics.
- [x] Full repository regression: 114 passed with one pre-existing pandas
  FutureWarning in 191.73 seconds.
- [ ] Do not launch the 600-network discovery run automatically.
## PRD v2.1 FTT High-Scale SRN with execution control

### Scope and isolation

1. Added a standalone `ftt_srn_execution_v2/` package and
   `scripts/run_ftt_srn_execution_v2.py`. The existing task SRN,
   environmental-pressure SRN, PRD2/PRD3, and partial-observability fit code
   were not changed.
2. This experiment is fully synthetic and therefore has no dependency on the
   pending third-context human dataset.
3. Kept the fixed task sequence `(0,1,2,0,3,1,2,3)`, strict 4D token one-hot,
   three-dimensional context basis, eight tanh hidden units, full context
   strength, `.005` Adam learning rate, 17 alternating A10/B10 blocks, and one
   optimizer update per completed sequence repetition.

### Scientific decisions

1. Registered conditions are exactly High `.5` + execution, Low `.02` +
   execution, and High `.5` sequence-only. There is no condition-specific
   tuning.
2. The Novel input-to-hidden column is initialized to exact zero, protected by
   a gradient hook, and re-zeroed after each training update. Training schedule
   validation accepts only pure A/B; all five readouts are frozen and record
   zero optimizer updates.
3. The execution ecology uses `kappa=12`, `delta=.05`, `c_error=1`,
   `c_time=.25`, and `lambda_exec=.20`. The time cost was selected from the
   pre-probe confidence×speed grid because it yields a globally monotone policy
   with interior optima; it was not selected from Mixed or Novel behavior.
4. Correct-token probability is detached only in the execution-cost branch.
   Sequence cross-entropy still trains the shared recurrent network and token
   head normally.
5. Cross-context transfer changes the recurrent prefix state while holding the
   current token, target context, network weights, and target fixed. This
   isolates disruption from importing an A state into B and vice versa.
6. Behavioral interpolation is defined within each seed/checkpoint relative to
   that network's own pure A/B normalized-RT endpoints. Midpoint deviation is
   always exported; the ratio index is masked when endpoint separation is less
   than `.02`.
7. The stage-1 and stage-2 gates never erase probe output. Failed-gate probes
   are exported for auditability with `interpretation_allowed=False`, and the
   report does not make a mechanistic interpolation claim.
   Execution grounding is assessed over the final eight complete A/B blocks,
   after sequence mastery is established; using the full acquisition period
   would confound the learned within-state confidence/speed relationship with
   the execution head's early adaptation from random High-Scale weights.
8. Checkpoints are 20, 40, 80, and 170 sequence updates for final/pilot runs.
   Smoke uses 20 and 40; automatic labels preserve a final snapshot for reduced
   test configurations.
9. Results are written directly to the requested output directory. Only the
   Matplotlib cache uses the system temporary directory; no model artifact is
   hidden in a temporary results folder.

### TDD and progress

- [x] RED/GREEN strict inputs, initialization scale, and Novel neutrality.
- [x] RED/GREEN calibration monotonicity, interior optimum, detached confidence,
  exact FTT schedule, and sequence-only acquisition.
- [x] RED/GREEN cohort splitting gate and no-update counterfactual identity.
- [x] Stage-1 switch, transfer, context-weight, and geometry diagnostics.
- [x] Stage-2 overall/within-context/controlled execution-grounding diagnostics.
- [x] Within-network interpolation, seed-level structural prediction, hidden
  geometry, and execution-head alignment.
- [x] All 18 CSVs, run audit, decision report, nine figures, profiles, and
  timestamped terminal progress.

### Validation record

- Focused v2.1 suite: `15 passed`, including the complete three-condition
  pipeline smoke. It produced all 18 required CSVs, JSON/report, and exactly
  nine PNG figures.
- Byte compilation and CLI help/startup passed.
- Visually inspected the architecture, learning/execution dynamics, and shared
  hidden-trajectory figures from the smoke artifact set; labels, geometry, and
  layout render correctly.
- Ran the registered five-seed/17-block pilot path without figures in a system
  temporary validation directory. Both scientific gates passed and reliable
  endpoint networks yielded a preliminary Result F. The report correctly marks
  `final_inference_ready=False`; this pilot is validation, not final inference.
- Final full-repository regression: `129 passed, 1 pre-existing pandas
  FutureWarning` in 188.54 seconds. No legacy SRN, PRD2, PRD3, GRU, or
  environmental-pressure test regressed.
- The 20-seed final profile was intentionally not launched. It is available
  through the documented CLI for the user to run in their terminal.
## PRD v2.2 causal A/B representation and execution

### Scope and checkpoint provenance

1. Added an isolated `ftt_srn_causal_representation/` package and
   `scripts/run_ftt_srn_causal_v22.py`. No v2.1 architecture, objective,
   schedule, optimizer, or analysis definition was changed.
2. The completed v2.1 run saved behavioral/geometry tables but not model state.
   v2.1 now serializes every trained checkpoint plus a SHA-256 manifest during
   its normal run. This is serialization only; causal code never imports the
   trainer or recreates a model through optimization.
3. The pre-existing final directory cannot support causal intervention because
   weights cannot be recovered from its CSVs. Documentation therefore uses a
   new `final_checkpointed` source directory rather than overwriting the user's
   completed result.

### Causal implementation decisions

1. Human-matched is primary; early A/B, several-switch, and final are complete
   secondary suites. Existing splitting and execution-grounding gates are
   loaded from the source audit and enforced before any intervention.
2. Every seed/checkpoint gets distinct estimation and evaluation forward-pass
   identifiers. Estimation uses pure A/B only. Directions are `h_A(k)-h_B(k)`
   at exactly matched token and target positions, so positive always means A.
3. Uncentered SVD is applied to the seven token-specific difference vectors.
   The smallest subspace reaching 90% difference energy is used for ablation
   and patching. Token-specific full differences remain the primary steering
   direction.
4. Instantaneous readout interventions are primary. Recurrent rollouts use the
   modified state as the next `nn.RNN` initial state and export every horizon;
   the original evaluation trajectory remains untouched.
5. A/B ablation removes only `U U' (h-mu)`. Probe patching replaces only that
   component with matched A or B while preserving the recipient's orthogonal
   residual numerically.
6. Deterministic random controls are norm matched and orthogonal to the A/B
   subspace whenever complement dimension permits. Random-subspace ablation
   and patching accompany random steering. Sequence-position steering uses the
   next matched position contrast, norm matched to the A/B difference.
7. The natural manifold is the within-network evaluation cloud spanning pure
   A/B, Mixed_norm, and Novel. Ridge-regularized Mahalanobis distance,
   nearest-state Euclidean distance, and hidden norm accompany every nonzero
   intervention.
8. Targets never change. Prediction identity is recorded separately because a
   large intervention can legitimately alter the predicted token. Model hashes
   are checked before/after every causal suite; audit counters for optimizers,
   backward calls, and parameter mutations are fixed at zero.
9. All causal effects are calculated within seed before group aggregation.
   The registered strong group threshold is 80% (16/20 in the final run).
   Mixed and Novel receive separate patching decisions.

### TDD progress

- [x] RED: imports failed because the v2.2 package/checkpoint bundle was absent.
- [x] GREEN: token matching, A orientation, projection and patch residual math.
- [x] GREEN: automatic multidimensional subspace selection.
- [x] GREEN: norm-matched orthogonal controls, zero identity, readout isolation,
  recurrent propagation, fixed targets, and off-manifold distances.
- [x] GREEN: checkpoint loading, frozen-weight audit, all 15 tables, report,
  and ten-figure end-to-end smoke contract.
- [x] Focused v2.1 + v2.2 integration suite: `23 passed`.
- [x] Five-seed frozen integration validation completed 20 causal
  seed/checkpoint suites in approximately four seconds without plots. It
  yielded a deliberately preliminary Result C: M1 100%, M2 20%, M3 100%,
  Mixed M4 80%, and Novel pathway 100% at human-matched. Correctly signed raw
  steering was 100%, but only 20% also satisfied the stricter A/B-over-random,
  both-endpoint, and sequence-preservation sufficiency conjunction.
- [x] Byte compilation and causal CLI help passed.
- [x] All ten figures rendered. Visually inspected the causal schematic,
  token-direction/cosine matrix, steering curve, ablation dissociation, Mixed
  patching, and checkpoint-dependence figures; corrected token labels and
  preregistered checkpoint order during visual QA.
- [x] Full repository regression: `137 passed, 1 pre-existing pandas
  FutureWarning` in 193.19 seconds. No legacy SRN, PRD2, PRD3, GRU,
  environmental-pressure, or v2.1 behavior regressed.
- [x] The 20-seed causal final was not launched because the user's existing
  v2.1 final run predates checkpoint persistence. The documented one-time
  checkpoint-bearing v2.1 rerun is required before frozen final interventions.

## PRD v2.3 natural context × sequence interaction

### Scope and frozen provenance

1. Added an isolated `ftt_srn_context_sequence_interaction/` package and
   `scripts/run_ftt_srn_context_sequence_v23.py`. The v2.1 trainer/objective and
   the v2.2 intervention pipeline were not changed.
2. The source is the checkpoint-bearing v2.1 High-Scale run. Source splitting
   and execution-grounding gates are enforced before analysis; every exported
   mechanistic row repeats both gate statuses.
3. Models are loaded through the checksum-validating v2.2 loader, put in eval
   mode, and have gradients disabled. SHA-256 digests before and after every
   seed/checkpoint suite must match. Optimizer, backward, and mutation counts
   are explicitly zero.

### Representational and causal decisions

1. Local context components are half-differences, `d_k=(h_A-h_B)/2`, so
   `mu_k+d_k` and `mu_k-d_k` exactly reconstruct natural A and B. Estimation and
   evaluation forward passes have distinct identifiers and targets are matched
   at the same token position.
2. The shared component is the projection of each `d_k` onto within-network
   uncentered PC1. PC1 is oriented toward the mean B→A difference only for
   stable reporting; projections are sign invariant. The specific residual is
   checked for exact reconstruction and numerical orthogonality.
3. Full/shared/specific patches replace only the named component and preserve
   the recipient residual orthogonal to it. Natural A←A and B←B identity
   patches are asserted before Mixed/Novel interventions.
4. Wrong-token controls use every `j != k`, norm-match `d_j` to `d_k`, and keep
   the target token's original full-component residual fixed. No convenient
   wrong token is selected.
5. Random directions are norm matched and orthogonal to the within-network
   span of natural A/B token differences. Sequence controls use the adjacent
   midpoint contrast and are norm matched to the local context component.
6. Full/shared/specific ablations remove only the corresponding projection.
   Every intervention reports execution, RT, CE, correct-token probability,
   predicted identity, target invariance, and natural-manifold diagnostics.
7. Statistical models use aligned token-position indicators for
   `f_seq(mu_k)`. Raw hidden units cannot be pooled across independently
   initialized seeds because their coordinate systems are not aligned. The
   within-network shared-axis projection supplies `z_context`; the interaction
   model adds token × context terms.
8. All execution models are fit exclusively on natural A/B outcomes with
   leave-one-seed-out evaluation. Mixed and Novel states supply features only
   after fitting and are reported separately. AIC/BIC are descriptive full-A/B
   fits; held-out MSE, R², and Gaussian log likelihood are cross-validated.
9. Seed summaries preserve token-level effects and repeat within-seed
   natural-versus-causal correlations. Mechanism decisions are aggregated only
   after within-seed calculation. Result A additionally requires the mandated
   A/B-over-random specificity control.

### TDD and validation progress

- [x] RED: all v2.3 imports failed before the new package existed.
- [x] GREEN: exact local reconstruction, shared+specific reconstruction,
  residual orthogonality, residual-preserving and identity patches.
- [x] GREEN: exhaustive wrong-token exclusion/norm matching, frozen targets,
  norm-matched random controls, and complete off-manifold coverage.
- [x] GREEN: A/B-only model fitting, no Mixed/Novel leakage, and strict
  leave-one-seed-out splits.
- [x] GREEN: all required tables, audit/config/report, and ten-figure smoke
  contract (`8 passed`).
- [x] A real two-seed × four-checkpoint smoke against
  `outputs/ftt_srn_execution_v2_final_checkpointed` completed in under five
  seconds without plots and wrote every nonempty table. Its off-manifold table
  contained exactly one row for each of the 2,800 synthetic states. This is
  software validation, not final inference.
- [x] The registered 20-seed × four-checkpoint frozen validation completed all
  80 suites in approximately ten seconds without plots. All checkpoint hashes
  remained unchanged, all 25 CSVs were nonempty, and 28,000/28,000 synthetic
  states had off-manifold diagnostics. `final_inference_ready=True`.
- [x] At `human_matched`, the registered outcome was Result D for both Mixed
  and Novel. Mixed passed M1 (100% of seeds), M2 (85%), M3 (100%), M4 (85%),
  and the random-direction specificity control (90%), but failed M5 (15%) and
  M6. The interaction model had held-out MSE `.047007` versus `.046763` for
  additive and also did not improve Mixed generalization (`.045708` versus
  `.045596`). This supports correct-token causal specificity but not the full
  M1–M6 configured-state claim.
- [x] Rendered all ten figures on the final validation data and visually
  inspected the hypothesis diagram, decomposition, natural patching,
  correct/wrong-token contrast, token correspondence, shared/full effects,
  necessity, model comparison, and both generalization panels. Labels and
  layouts are readable and match the registered figure roles.
- [x] Byte compilation and CLI help passed. Full repository regression:
  `145 passed, 1 pre-existing pandas FutureWarning` in 214.20 seconds.

## PRD v3 human configured-state behavioral test

### Scope and preprocessing decisions

1. Added an isolated `human_configured_state/` package and
   `scripts/run_human_configured_state_v3.py`. The v2.1–v2.3 training and causal
   analysis code was not changed.
2. Reused `retrieval_modeling.data.load_retrieval_data`, so the v3 token rows
   inherit the prior participant/block completeness checks, error exclusions,
   duplicate-session handling, and exclusion of raw sequence zero.
3. Validated the raw keypress exports directly. Maze encodes positions as
   `0,2,...,14`; Melody, Mixed, and third context encode them as `0,...,7`.
   Encoding is inferred within the declared block label and converted to a
   common ordinal coordinate.
4. The analysis uses seven response transitions, `1→2` through `7→8`, matching
   the seven frozen SRN next-token states. Response position one is excluded
   because it has no within-sequence predecessor; this definition is repeated
   in the config and preprocessing audit.
5. The third-context export's `Space Training` and `test spaceA` labels map to
   Novel while their untouched raw labels remain available. Mixed and Novel are
   never pooled except in the explicitly labeled shared-profile hypothesis.
6. Correct positive-RT attempts are averaged within participant × block ×
   sequence × transition. Odd analyzed repetitions are participant calibration
   data and even repetitions are held-out evaluation data.

### Modeling and inference decisions

1. Pure A/B token profiles are estimated with shrinkage and frozen before any
   probe fit. Midpoint deviation is primary; responsibility q is exported
   unclipped and flagged unreliable below `.03` log-RT endpoint separation.
2. Stage 1 uses participant fixed effects with participant-clustered standard
   errors, plus counterbalance and block-order terms. This is the supported
   non-singular participant-effects implementation for the current stack.
3. M0–M4 are evaluated with held-out Gaussian log likelihood as primary, plus
   RMSE, MAE, parameter count, and descriptive AIC/BIC. Training participants
   alone define M2 group token effects, M3 PCA components, and M4 group slope;
   held-out participants receive only their allowed scalar calibration.
4. M3 retains the smallest training-only PCA basis explaining at least 90% of
   pure-profile variance. M4 uses the frozen all-seed mean execution-q shape
   and also reports zero-free-token strict and full/shared/specific variants.
5. Every registered human-matched SRN seed is loaded through the checksum
   manifest. Twenty before/after weight hashes match; optimizer steps, backward
   calls, parameter changes, human seed selection, and human profile-shape
   fitting are all zero.
6. Recovery uses actual participant-by-position counts, endpoint separation,
   residual noise, and the registered effect scale. The null template receives
   the correct one-parameter BIC penalty; nonconstant templates receive two.

### TDD and validation progress

- [x] RED: v3 package imports initially failed.
- [x] GREEN: pure/probe isolation, sequence and counterbalance preservation,
  midpoint and scalar-mixture identities, denominator reliability flags,
  participant/observation CV isolation, all-seed aggregation, and 5×5 recovery.
- [x] GREEN: end-to-end synthetic Mixed plus `Space Training`/`test spaceA`
  pipeline test verifies all 22 required CSVs, 11 figures, frozen hashes,
  all-seed use, separate probe ladders, and leakage audit fields (`8 passed`
  across the focused v3 suite).
- [x] Real 12+12 smoke completed against the existing human CSVs and all 20
  checkpointed seeds. This exposed and fixed the doubled Maze keypress encoding.
- [x] Registered full run completed with 142 Mixed and 65 Novel participants,
  all 20 human-matched seeds, five participant CV folds, 20 recovery replicates,
  every required nonempty table, and 11 rendered figures.
- [x] Full-run result is D: M3 low-rank predicts both Mixed and Novel best.
  Mean split-half reliability is `.441`; the Context × Position test passes.
  Frozen M4 does not outperform M1 and its group q correlation is negative for
  both probes, so the behavioral result supports low-rank human token structure
  but not the frozen SRN profile as an out-of-sample human predictor.
- [x] All figures were visually inspected. Axes, labels, legends, unclipped q
  values, model ordering, and recovery annotations render correctly.
- [x] Byte compilation and CLI help passed. Full repository regression:
  `153 passed, 1 pre-existing pandas FutureWarning` in 313.66 seconds. No
  legacy retrieval, GRU, SRN v1/v2.1/v2.2/v2.3, or earlier behavioral analysis
  test regressed.

## PRD v3.1 low-rank human-dimension interpretation

### Scope and frozen-data decisions

1. Added the isolated `human_low_rank_interpretation/` package and
   `scripts/run_human_low_rank_interpretation_v31.py`. No v3 behavioral-model
   or earlier SRN/retrieval code was edited.
2. The primary reference basis is estimated only from the larger Combined
   dataset's frozen pure A/B participant profiles. Mixed and Novel outcomes
   estimate coefficient locations but contribute zero rows to basis fitting.
3. Novel pure A/B profiles receive an independent basis and are aligned to the
   Combined reference only after fitting. This prevents replication from being
   manufactured by pooling. Shared-versus-separate CV centers pooled training
   profiles within dataset and holds out the raw human identity across both
   datasets, so an identity cannot leak through the other dataset's basis.
4. The analysis preserves the seven SRN-aligned transitions `1→2` through
   `7→8`. The formal sequence is `(0, 1, 2, 0, 3, 1, 2, 3)`; the cyclic-boundary
   feature is correctly constant/structurally inestimable because transition
   `8→1` is outside the validated response set.
5. The frozen feature table includes pure mean log RT, A−B, raw-attempt error
   rates joined only to validated units, early/late within-block slopes,
   first-versus-sequences-2–3 switch cost, first-versus-sequences-7–9 recovery,
   initial/asymptotic 20% summaries, key recurrence distances, and sequence-half
   indicators. Its final SHA-256 is
   `7cebf6ed16a540878fd1510cf48519ad1c251c384230d214a5cd0fa4ac54ff49`.

### Estimation and inference decisions

1. PCA is primary; factor analysis is the robustness method. Ranks 1, 2, 3,
   and full rank 7 are compared using participant-held-out reconstruction and
   held-out probe likelihood. The one-standard-error rule chooses the smallest
   adequate rank; parallel analysis and explained variance are also exported.
2. Participant folds and 1,000 participant bootstraps are matched to the
   Combined reference with Hungarian assignment plus sign alignment. A retained
   component is stable at median matched cosine `>= .70`.
3. Transition-level inference is exact rather than Monte Carlo: all estimable
   label permutations are enumerated, up to `7! = 5,040`. Categorical features
   use exact eta-squared tests; numeric features report Pearson, Spearman,
   leave-one-transition-out error, bootstrap intervals, and exact permutation p.
4. Generic difficulty is explicitly regressed from each component before
   testing A−B, switch, Mixed, and Novel profiles. Component labels require the
   association, stability, semantic-recovery, and independent-replication gates;
   difficulty/context/switch labels additionally require their registered
   difficulty-control conditions.
5. Mixed, Novel, and Mixed1/Mixed2/MixedTest coefficient projections use the
   frozen Combined A/B basis. Shared, separate, and scalar probe models are
   compared with participant-held-out Gaussian likelihood. Separate-versus-
   shared decisions compare the mean paired fold gain with its fold-level SE;
   total gains are also preserved for audit.
6. Semantic recovery simulates difficulty, context, switch, combined, and
   random axes at observed participant count and split-half A/B noise. The
   difficulty+context generator is scored by subspace coverage because PCA may
   rotate the two generating axes. Random false-label rate must be `<= .05`.
7. Human/SRN comparison remains secondary and frozen: selected human dimension,
   variance partition, principal angles, and canonical correlations are tested
   without seed selection, optimization, backward calls, or weight changes.

### TDD, full run, and interpretation

- [x] RED: imports for decomposition, alignment, features, probes, replication,
  SRN abstraction, and recovery failed before the v3.1 package existed.
- [x] GREEN: T1–T12 cover recovery of known rank, no probe-to-basis leakage,
  held-out identity isolation, component permutation/sign correction, exact
  permutation inference, difficulty removal, coefficient recovery, shared/
  separate behavior, cross-dataset matching, frozen SRN use, and semantic gates.
- [x] The synthetic end-to-end test uses the actual Novel labels `Space Training`
  and `test spaceA`, checks all required nonempty outputs and 11 figures, and
  verifies frozen-basis/SRN audit fields. Focused suite: `13 passed`.
- [x] Registered final run completed with 142 Combined and 65 validated Novel
  participants, five CV folds, 1,000 bootstraps, 1,000 parallel-analysis draws,
  200 semantic-recovery draws per generator, exact transition permutations,
  and all 20 frozen SRN seeds.
- [x] The selected rank is 2; both components are stable (all-resample median
  cosines `.997` and `.932`). Component 2 is labeled `late learning` after its
  exact association (`p=.00298`), stability, recovery, and replication gates;
  Component 1 remains neutral. The explicit beyond-difficulty outcome is false.
- [x] Independent Combined/Novel bases replicate at mean matched component
  cosine `.847` and shared-subspace variance `.758`. Shared A/B beats the scalar
  probe model in both datasets, but Novel's separate A/B basis improves over the
  shared basis by mean fold log likelihood `3.002` versus paired SE `2.038`,
  yielding probe Result F.
- [x] Human/SRN abstraction matching fails (mean angle `46.63°`, maximum
  `86.17°`, minimum canonical correlation `.067`), and no human outcome modifies
  an SRN. Semantic recovery passes with minimum target recovery `.80` and random
  false-label rate `0`.
- [x] All 23 required CSVs, config, report, additional audit tables, and 11
  figures were produced and audited. Every figure was visually inspected after
  correcting the coefficient aggregation/error bars and single-feature layout.
- [x] Byte compilation and CLI help passed. Full repository regression:
  `166 passed, 1 pre-existing pandas FutureWarning` in 317.53 seconds. No prior
  retrieval, GRU, SRN, causal, interaction, or human v3 test regressed.
