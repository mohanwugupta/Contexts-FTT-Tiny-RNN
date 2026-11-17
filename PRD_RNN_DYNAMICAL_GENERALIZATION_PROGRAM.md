# PRD — Dynamical Accessibility of Shared Computations After Context Splitting

**Status:** New, prospective extension to the frozen four-experiment SRN paper analysis.  
**Audience:** Coding agent implementing and auditing the RNN experiments.  
**Scope:** RNNs only. No human model fitting, RT readout, or new human-data analysis.  
**Implementation mode:** Test-driven development (RED → GREEN → REFACTOR), with a persistent scratchpad, source provenance, and decision-gated execution.  
**Main scientific goal:** Determine whether generalization after context-specific sequence learning is limited by output-compatible state availability, cue-driven access, recurrent trajectory viability, and the information preserved in context representations.

## 0. Read this first: scientific thread and immovable constraints

The learned A and B tasks use the **same eight-item sequence and the same next-item targets**, but context-specific SRN trajectories can separate. A midpoint of A/B hidden states produces the midpoint of their output logits because the output head is linear. For a common correct label, cross-entropy convexity gives:

\[
\mathrm{CE}((\ell_t^A+\ell_t^B)/2,y_t)\leq (\mathrm{CE}(\ell_t^A,y_t)+\mathrm{CE}(\ell_t^B,y_t))/2.
\]

**This is an algebraic expectation, not evidence that the SRN has learned to generate the midpoint.** The program must isolate the more interesting question: whether the recurrent computation can *reach and follow* an output-compatible trajectory under changed contextual inputs.

Four work packages, in order:

1. **E1 — Dynamical accessibility / closure audit:** output-compatibility vs one-step closure vs propagation of error; existing checkpoints only.
2. **E2 — Context-information grid:** vary independent cue coordinates and geometry controls; existing checkpoints only.
3. **E3 — Causal separation × dynamics training intervention:** small, prospectively defined factorial training experiment, gated on A/B-only feasibility and interpretability.
4. **E4 — Replication across novel fixed sequences:** repeat the fixed design without changing it after seeing results; new training only if E3 implementation is technically valid.

**Do not** retrain or alter any historical checkpoint, revise Novel30/60 based on results, build a new `rho` pathway, fit probes during training, or search architectures to obtain the preferred outcome. The old paper freeze and its stop decision remain intact. These experiments have new preregistered identities and must never be relabeled as part of the original confirmatory freeze.

### Required reference artifacts

Read and index before implementation:

- `FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md` (canonical frozen outcomes, cohort definitions, Novel registry hash, metric contract).
- `FINAL_EXPERIMENT_INFRASTRUCTURE.md`, `FINAL_EXPERIMENT_INVENTORY.json` (source paths and provenance).
- `PRD_phase1k_cue_only_composite_generation(1).md` and Phase 1K report (matched-prestate cue branches).
- `PRD_phase1l_contrast_bottleneck_trajectory_closure(1).md` and Phase 1L report (context projection and neutral closure).
- Phase 1I, 1J, 1F reports and code as needed.

If actual repository code or checkpoint bytes are missing, do not fabricate values or retrain substitute checkpoints. Mark the affected job `blocked_missing_source` and report the exact missing path/hash. PRDs describe intended methods; source code and frozen checkpoint artifacts govern faithful reproduction.

## 1. Mandatory agent workflow and scratchpad

Create `scratchpads/dynamical_generalization_program.md` **before making substantive code changes**. Update it after every test group, important code discovery, failed experiment, or scientific decision.

Use this format:

```markdown
# Dynamical generalization program — live scratchpad
## 0. Snapshot
- datetime / git commit / environment / input artifact root
- source reports read, hashes, missing material
- frozen Novel registry SHA256, config SHA256
## 1. Current scientific question
- hypothesis, null, intended causal contrast, competing interpretation
## 2. Assumptions to verify in source
- source file:function:line, status [unknown|verified|incorrect]
## 3. Decisions (append-only)
- ID, decision, date, A/B-only justification, probes inspected? [yes/no]
## 4. RED tests
- test name, why it must initially fail, observed failure
## 5. GREEN implementation
- smallest fix, changed files, rerun tests, result
## 6. REFACTOR / regression
- invariants checked, historical parity, hashes
## 7. Run register
- experiment, config hash, seed, sequence, checkpoint, status, output path
## 8. Results, including negative results
- effect, uncertainty, failure mode, prohibited interpretation
## 9. Open blockers and next decision
- blocked vs completed; next authorized work package
## 10. Final claim-to-evidence map
- each claim -> table/figure/test/source identity
```

Maintain `scratchpads/decisions.jsonl` as an append-only machine-readable decision trail. Each `decision` must have `id`, `timestamp`, `scope`, `rationale`, `data_allowed_for_decision`, `files_changed`, and `changes_scientific_contract`.

### Implementation discipline

For each feature: **(1) write minimal failing test → (2) run and record RED failure → (3) implement smallest change → (4) run GREEN tests → (5) refactor without changing results → (6) regression-test old frozen behavior → (7) document.** No scientific execution until all relevant gates pass.

Do not use scientific outcome magnitude as a software acceptance criterion. A clean null result is completion; malformed data, false provenance, incorrect indexing, or failed regression is not.

## 2. Shared SRN and metric contract

### 2.1 Existing-model baseline

Canonical model: eight-unit tanh Elman SRN, four token inputs, three context inputs, hidden recurrence, linear next-token head, fixed eight-item sequence `(0,1,2,0,3,1,2,3)`, seven scored transitions, A10/B10 alternating training, 170 online updates, Adam 0.005, gradient clipping 1.0, High/Low initialization scales 2.0 / 0.02 as applicable. **Verify these values in source** and record any discrepancy rather than silently imposing this document.

Natural evaluation: fixed zero initial hidden state, same sequence tokens under teacher forcing, frozen weights, no optimizer/backprop. Raw next-token CE in nats is the behavioral outcome. Report per-transition CE and sequence mean (7 transitions); keep familiar A/B losses separately. Do not treat teacher-forced prediction as independent autonomous sequence production.

Cohorts must not be merged:
- canonical High/Low *monotonic* checkpoints: the Experiment 1–3 cohort;
- Phase 1L C0–C3 *sequence-only* checkpoints: the constructive intervention cohort. C0 tensors match the separately saved High sequence-only model, **not** canonical High monotonic.

Probes: A `(1,0,0)`, B `(0,1,0)`, Mixed `(.5,.5,0)`, NoContext `(0,0,0)`, frozen Novel30/60 A-side and B-side from the final novelty registry. Keep historical LegacyNovel cues in a separate optional sensitivity table; do not mix them into NovelMean.

### 2.2 Output and geometric measures

Let `h_t(c)` be the post-update hidden state and `ell_t(c) = W_o h_t(c)+b_o` the logits. Use centered logits `z_t(c)=P ell_t(c)`, where `P = I - 11ᵀ/K` for K=4 output classes. Familiar midpoint `m_t=(h_t^A+h_t^B)/2`, task midpoint `z_t^m=(z_t^A+z_t^B)/2`.

Retain all of the following rather than collapsing to one normalized residual:

- per-token and mean natural CE;
- hidden A/B separation and centered-logit A/B separation;
- absolute hidden midpoint error `||h_t(c)-m_t||`;
- absolute task midpoint error `||z_t(c)-z_t^m||`;
- normalized residual `||z_t(c)-z_t^m||/(||z_t^A-z_t^B||+eps)` as sensitivity only;
- correct-token probability and complete output probabilities.

The inferential unit is a trained network seed, with sequence-level clustering when multiple sequences are introduced. Do not pseudoreplicate transitions, cues, injection positions, or paired variants as independent network replicates.

### 2.3 Statistics

Use paired seed effects within each fixed sequence; fixed bootstrap seed `20260925`, 10,000 whole-seed bootstrap draws where feasible, saved resample indices. For new-sequence replication, show **each sequence separately** and hierarchical descriptive summaries; 3–4 sequences are not enough for strong population-level claims about all sequences. Report means, paired effect estimates and intervals, raw n, any exclusions, and failure/quality flags. Do not replace effect sizes with p-value gates.

### 2.4 Strict probe firewall

Do not use any Mixed/NoContext/Novel30/Novel60 data for checkpoint selection, hyperparameters, training loss, regularizer calibration, augmentation choice, or model family selection. The known *definitions* of probes may be used to predefine evaluation and expected mathematical identities. The required training-design pilot for E3 must use only A/B familiar data.

## 3. E1 — Dynamical closure and error-propagation audit (no new training)

### 3.1 Scientific question

**Do frozen SRNs generalize poorly because output-compatible midpoint states are unavailable, because the recurrent dynamics do not carry those states to their next useful position, or because deviations from the shared trajectory grow?**

This directly tests the gap between a good static output state and a *viable recurrent trajectory*.

### 3.2 Definition and exact decomposition

For each seed, model, cue `c`, and scored transition `t`, compute natural familiar `h_t^A`, `h_t^B` and midpoint `m_t`. Let `m_0=h_0` (same frozen zero initial state). With `f_theta` the actual model's recurrent step:

\[
 b_t(c)=f_\theta(m_{t-1},x_t,c)-m_t \quad\text{(one-step closure defect)}.
\]

For natural probe state `h_t^c`, let `e_t(c)=h_t^c-m_t`. Then the **exact** identity is

\[
 e_t=b_t+[f_\theta(m_{t-1}+e_{t-1},x_t,c)-f_\theta(m_{t-1},x_t,c)].
\]

Calculate the analytical Jacobian of the frozen recurrent step at the midpoint prestate:

\[
 J_t(c)=\operatorname{diag}[1-\tanh^2(a_t(m_{t-1},c))]W_h
\]

for the verified tanh recurrence, including whichever model-specific cue pathway is actually used. The **first-order approximation** is `e_t ≈ b_t + J_t e_{t-1}`. Save the exact nonlinear propagation term and the Taylor remainder; do **not** label the approximation exact.

Output-relevant closure defect: `||P W_o b_t||`. Output-relevant error: `||P W_o e_t||`. Keep hidden norms separately; hidden displacement may be irrelevant to behavior. Save `||J_t||_2` and a finite-horizon product diagnostic for local amplification, but **do not interpret a singular value >1 as proof of a long-run attractor or instability**.

### 3.3 Static mathematical baseline

Recompute the CE of averaged A/B logits at each transition. Check numerical equality to CE from the patched midpoint hidden state:

\[
 \mathrm{CE}(W_o m_t+b_o,y_t)=\mathrm{CE}((\ell_t^A+\ell_t^B)/2,y_t).
\]

Check convexity inequality against mean familiar A/B endpoint CE for every row. Compare this *algebraically expected* repeated-patch endpoint with (i) natural probe rollout and (ii) single-injection recurrent continuation. Do not claim repeated midpoint success itself reveals natural retrieval.

### 3.4 Evaluations and competing accounts

Use canonical High (and Low as floor/geometry control) and Phase 1L C0–C3, **each analyzed within its own cohort**. Cues A, B, Mixed, NoContext, Novel30-A/B, Novel60-A/B.

Prespecified descriptive questions:

1. How much of midpoint success is exactly explained by linear-logit averaging?
2. Is the desired midpoint dynamically closed under each cue (`b_t≈0`)?
3. Which of closure defect, exact propagated displacement, and Jacobian amplification are output-relevant?
4. Do single-injection lag profiles reproduce the decomposition's predicted divergence?
5. Do C2/C3 change closure defect compared with their matched C0/C1 conditions, and do those changes accompany CE changes?

For an *additional predictive* comparison, define a small grouped-by-seed cross-validation:
- **Static baseline:** familiar mean CE, current cue `alpha`, cue norm, and A/B output separation;
- **Dynamics-augmented:** same predictors plus prespecified mean output-relevant closure defect and finite-horizon local amplification statistic.

Use the same held-out seed folds and regularization protocol for both; save held-out predictions. Interpret improved prediction as **incremental predictive value**, not causal mediation. The exact recurrent identity cannot be used as a supposedly independent predictor of its own target.

### 3.5 RED → GREEN tests

- `E1_T01`: recorded midpoint logits equal arithmetic mean of A/B logits for randomly generated legal tensors and actual checkpoints;
- `E1_T02`: midpoint CE obeys Jensen/convexity rowwise when A/B share target;
- `E1_T03`: exact error recurrence reconstructs natural `e_t` to dtype-declared tolerance, including `t=1`;
- `E1_T04`: analytic Jacobian matches finite differences at multiple non-saturated prestates (tolerance frozen before model evaluation);
- `E1_T05`: `||P W_o e||` is invariant to adding a constant to every logit;
- `E1_T06`: first-order Taylor error is separately reported and is not silently treated as exact;
- `E1_T07`: A/B training and probe teacher-forcing step indices align (7 predictions; no off-by-one);
- `E1_T08`: original Phase 1I/J/K/L CE, state hashes, and midpoint references reproduce within evaluator tolerance;
- `E1_T09`: no optimizer, no checkpoint mutation, seed-paired aggregation;
- `E1_T10`: held-out seed validation does not leak transitions of the same seed into train and test folds.

### 3.6 Deliverables and completion gate

`e1_transition_dynamics.csv`, `e1_seed_summary.csv`, `e1_static_convexity_audit.csv`, `e1_single_patch_replay.csv`, `e1_grouped_cv_predictions.csv`, plots of closure versus behavioral persistence and C0–C3 matched changes, `E1_RESULTS.md`.

**Done** once all identities/regressions pass and all registered rows are reported. Negative predictive value is a scientifically valid result and must not initiate a new objective search.

## 4. E2 — Context-information and cue-geometry grid (no new training)

### 4.1 Scientific question

**Which components of contextual input affect natural sequence predictions and shared-state accessibility, beyond the learned A↔B contrast?** Can the effect be explained by input magnitude or extrapolation rather than an abstract novelty operation?

Do not infer that a raw cue coordinate is itself a representation of psychological familiarity.

### 4.2 Independent cue coordinates

Use only the two active cue channels, keeping the historical third channel at zero. Define:

\[
 \alpha=x-y,\qquad \beta=x+y,\qquad c(\alpha,\beta)=((\alpha+\beta)/2,(\beta-\alpha)/2,0).
\]

A = `(alpha=1,beta=1)`; B = `(-1,1)`; Mixed = `(0,1)`; NoContext = `(0,0)`. Frozen Novel30/60 A-side have identical alpha `+(sqrt(3)+1)/2` and beta `±(sqrt(3)-1)/2`; B-side have negative alpha with the same paired beta magnitudes/signs. **Verify from frozen cue registry, not from labels.**

Predeclare and hash a finite evaluation grid before evaluating it:

```yaml
alpha: [-1.366025403784, -1.0, -0.5, 0.0, 0.5, 1.0, 1.366025403784]
beta:  [-1.0, -0.366025403784, 0.0, 0.366025403784, 1.0]
third_channel: 0.0
```

35 grid points/model/seed; retain exact frozen Novel vectors (not rounded grid approximations) as separately registered anchors. Also predeclare norm-matched comparisons at fixed alpha (e.g., beta = ±0.366..., same Euclidean norm), cue-magnitude rays `scale × A/B` with `scale ∈ {0.5,1.0,1.5}`, and physical Mixed/NoContext anchors. The primary controls are **same alpha, different beta** and **same norm, different direction**. Grid is designed for measurement, not to optimize probes.

### 4.3 Frozen-model experiment

Evaluate canonical High/Low and Phase 1L C0–C3 checkpoints with identical natural, zero-initial-state rollouts. Compute CE, logit geometry, closure defect from E1, cue norm, angular distance from A/B, alpha, beta, and raw context drive to hidden layer. Keep different model cohorts separate.

Primary within-seed contrasts:

1. `beta(+0.366) − beta(−0.366)` at `alpha=+1.366` and `alpha=−1.366` (Novel30/60 same-alpha alias pairs);
2. Mixed `alpha=0,beta=1` versus NoContext `alpha=0,beta=0`;
3. same-alpha matched-norm beta-sign pairs, which isolate an otherwise discarded direction without norm change;
4. same raw norm, different alpha/beta direction pairs;
5. gradient/sensitivity to beta at fixed alpha for original versus bottleneck models (descriptive finite differences).

**Expected mathematical invariant:** for C1/C3, same alpha must give identical effective context drive and identical natural trajectories/CE, up to numerical precision; the unconstrained models need not show this equality. This is a structural check, **not** a required scientific outcome for C0/C2.

### 4.4 RED → GREEN tests

- `E2_T01`: alpha/beta inverse transform round-trips raw two-channel cues;
- `E2_T02`: A/B/Mixed/NoContext anchor identities exact;
- `E2_T03`: frozen Novel30/60 alpha identities verified to tolerance;
- `E2_T04`: every grid cue hash matches the preregistered JSON before model eval;
- `E2_T05`: C1/C3 invariant to beta at fixed alpha, including full transition logits and CE;
- `E2_T06`: paired beta-sign comparisons have identical norm when specified;
- `E2_T07`: context third channel always zero; historical non-active Novel column never interpreted as trained;
- `E2_T08`: no labels or probe names special-cased in `alpha(c)` projection;
- `E2_T09`: checkpoint hashes unchanged and forbidden optimizer calls trapped;
- `E2_T10`: pairing/aggregation does not turn 35 grid points into 35 independent seeds.

### 4.5 Deliverables and completion gate

`configs/context_grid_v1.json`, registry SHA256, `e2_grid_transition.csv`, `e2_grid_seed.csv`, `e2_matched_geometry_controls.csv`, `e2_projection_invariants.csv`, `E2_RESULTS.md`, plots of CE/closure versus `(alpha,beta)` (separate panels for families). Exact cue geometry and out-of-support distance must be distinguished in captions.

**Done** when the complete preregistered grid and controls are evaluated and all invariants pass. Do not introduce a learned beta/rho pathway as a follow-up to a negative result.

## 5. E3 — Causal separation × recurrent-closure factorial (new training; gated)

### 5.1 Scientific question

**Can representational separation and recurrent trajectory viability be manipulated separately, so that their causal contributions to generalization can be estimated at comparable familiar task competence?**

The old High/Low comparison cannot isolate splitting because familiar A/B learning differs. E3 is not a repetition of High-versus-Low. It manipulates **training objectives** under the same base architecture and initialization family.

### 5.2 Eligibility and A/B-only pilot

Proceed to E3 only if E1/E2 completed their integrity audits, the standard 8-unit recurrence/gradient can be faithfully replayed, and source checkpoints/training stream are available. Results from E1/E2 may motivate reporting but **may not tune the E3 training grid after probe results have been inspected**. Lock the candidate design and calibration rules here; do not change them based on probe outcomes.

Run a small **A/B-only** pilot with reserved pilot seeds `1000..1004` on the original sequence. All pilot evaluations are familiar A/B only. Fix selection criteria in advance: (a) model learns familiar A/B below the documented A/B-only reference competence bound, (b) strong-versus-weak regimes differ in hidden A/B separation, (c) both closure arms are numerically stable and show nontrivial gradients, (d) no collapse to degenerate constant output. Calibrate objective strengths exclusively from A/B loss/geometry/gradient norms. Record all attempted settings, including failures, and freeze selected settings before any Mixed/NoContext/Novel model evaluation. If no settings work, set `evidence_status=insufficient_familiar_support`; do not widen search using probes.

### 5.3 Matched 2 × 2 training conditions

All conditions: **8-unit tanh SRN, original unrestricted context-input architecture**, High initialization family (`scale=2.0`), same A10/B10 stream, 170 updates, Adam 0.005, gradient clipping 1.0, same seed and initialization where parameter shapes match. Novel probes never train. No rank-one bottleneck: its inclusion would confound the splitting/closure factorial.

| Condition | Separation pressure | Shared trajectory closure |
|---|---|---|
| F00 | Weak/overlap pressure | Off |
| F01 | Weak/overlap pressure | On |
| F10 | Strong separation pressure | Off |
| F11 | Strong separation pressure | On |

A/B-only hidden trajectory references are generated from familiar rollouts of the current model. Define:

\[
D_h=\frac1T\sum_t \|h_t^A-h_t^B\|_2^2/H,\qquad H=8.
\]

Weak-split loss: `L_weak = D_h`. Strong-split loss: `L_strong = (1/T) Σ_t relu(m_h − ||h_t^A−h_t^B||_2)^2 / H` with a single prospectively fixed `m_h` set in A/B-only pilot. **Interpret these as direct regularization of hidden-state separation, not proof that separate memories have formed.** Always measure output/task-space separation as an independent outcome to detect irrelevant hidden splitting.

Shared midpoint `s_t* = stopgrad((h_t^A+h_t^B)/2)`. Neutral rollout `q_0=h_0`, `q_t=f(q_{t-1},x_t,cbar)` with `cbar=(cA+cB)/2` and a genuinely recursively inherited state. Use one closure objective in all four cells:

\[
L_{\rm close}=\frac{1}{TH}\sum_t \|q_t-s_t^*\|_2^2.
\]

**Unlike historical Phase 1L**, this E3 closure loss uses a **fixed hidden-size normalization rather than the current A/B separation as denominator**, because the denominator would change systematically with the separation treatment. Do not directly equate its numeric λ with Phase 1L λ=1. Freeze `λ_close` from the familiar-only calibration protocol and record this as a distinct new experiment.

Objective:

\[
L_{ij}=L_{\rm CE}^{A/B}+\lambda_{\rm sep}L_{\rm sep,i}+j\lambda_{\rm close}L_{\rm close}.
\]

Calibration candidate strengths are frozen to `{0.03, 0.1, 0.3, 1.0}` for `λ_sep` and `λ_close`, with `m_h` candidates `{1.5,2.5,3.5}` (hidden Euclidean units); use **only pilot A/B familiar competence and separation diagnostics** to choose a feasible *common* configuration. The agent must commit a single selected configuration or mark it infeasible before running confirmatory seeds.

### 5.4 Prespecified feasibility rules and causal contrasts

Confirmatory seeds `0..19` on the original fixed sequence (pilot seeds never pooled). Produce all F00/F01/F10/F11 conditions even if the scientific effect disappoints. Save all checkpoints and 170 true pre-update training records.

Before probe interpretation, report:

- familiar A/B CE and learning curves for every cell;
- A/B hidden and centered-logit separation;
- target-margin attainment;
- whether weak versus strong pressure actually produced nonoverlapping separation regimes;
- familiar CE imbalance, measured before viewing protected probes;
- divergence/stability/NaN counts.

Define **strongly supported competence comparability** prospectively as both familiar A and B paired CE gaps between weak and strong conditions at fixed closure arm having absolute mean ≤0.30 nats, with direction/seed variability reported. This threshold is a gate for *strong claims of separation at approximately comparable competence*, not a deletion or tuning criterion. If this fails, retain the intended-treatment estimates but **do not infer separation-only causality**; label `competence_confound`. Do not exclude seeds based on probe CE.

Primary registered causal estimands (raw CE, per probe):

- closure main effect at weak splitting: `F01 − F00`;
- closure main effect at strong splitting: `F11 − F10`;
- separation-pressure effect with closure off: `F10 − F00`;
- separation-pressure effect with closure on: `F11 − F01`;
- interaction: `(F11−F10) − (F01−F00)`.

Report these separately for Mixed, NoContext, Novel30Mean and Novel60Mean, with familiar CE and both separation measures alongside. **Do not** treat an interaction as mediation proof. A strong inference requires the intended separation/closure manipulations to change their target process without grossly changing familiar competence. If the manipulation fails, result is `failed_target_engagement`, not evidence against the theoretical mechanism.

### 5.5 RED → GREEN tests

- `E3_T01`: matched initial parameters and exact A/B training stream across all four arms for same seed;
- `E3_T02`: weak regularizer has correct gradient toward A/B convergence on synthetic states;
- `E3_T03`: strong hinge loss decreases once `||hA−hB||>=m_h` and gives correct gradient below it;
- `E3_T04`: no NaN/gradient singularity at zero A/B separation;
- `E3_T05`: stop-gradient closure target receives no target-branch gradient (verify actual implementation, not just detached display);
- `E3_T06`: closure `q_t` inherits its own previous `q_{t−1}` rather than resetting to midpoint;
- `E3_T07`: neutral cue during closure is derived only from A/B; report the algebraic identity with physical Mixed cue; **do not falsely assert Mixed physical cue never appears**;
- `E3_T08`: objective decomposition sum and coefficient selections match frozen config; new fixed-H loss cannot silently revert to Phase 1L denominator;
- `E3_T09`: 170 optimizer steps and genuine training updates with changing parameter hashes;
- `E3_T10`: protected probes never enter training, calibration, hyperparameter selection, or checkpoint selection;
- `E3_T11`: pilot seeds excluded from confirmatory inference; failed pilot settings logged;
- `E3_T12`: all four arms produce predictions using exactly the same evaluator and per-transition target;
- `E3_T13`: competence and separation gates computed from familiar-only rows and never trigger hidden seed exclusion.

### 5.6 Deliverables and stopping rule

`configs/factorial_v1.json`, `e3_pilot_ab_only.csv`, `e3_selected_config_manifest.json`, training curves, `e3_familiar_feasibility.csv`, `e3_probe_transition.csv`, `e3_seed_contrasts.csv`, `e3_interaction_bootstrap.csv`, `E3_RESULTS.md`, and a figure showing familiar competence, realized separation, closure defect and probe CE across the 2×2.

**Stop after the registered confirmatory 20 seeds.** If familiar-only feasibility cannot be achieved with the frozen candidate grid, label E3 inconclusive. No wider post-result search, additional architecture, special beta/rho head, or retraining because Mixed remains poor.

## 6. E4 — Replication across independently generated sequences (new training; fixed design)

### 6.1 Scientific question

**Is the relationship among splitting, dynamic accessibility and generalization specific to one eight-item sequence, or does it replicate across different sequence structures?**

The point is sequence-level replication, not a larger hidden-size or model-family sweep.

### 6.2 Frozen sequence registry

Retain the original `(0,1,2,0,3,1,2,3)` as a reference. Freeze three additional eight-item sequences, each with four symbols occurring twice and no adjacent repeats, before looking at any new probe behavior. Proposed exact list:

```text
S1 = (0,1,0,1,2,3,2,3)
S2 = (0,1,0,2,1,3,2,3)
S3 = (0,1,2,0,3,2,1,3)
```

Verify with tests: eight tokens, each symbol twice, no adjacent repeats, not identical to original or each other, seven next-token predictions, and no duplicate equivalence classes under symbol relabeling plus circular rotation. The proposed S1/S2/S3 are deliberately chosen to occupy different equivalence classes from the original and one another; **verify algorithmically** before running. If equivalence is nonetheless detected, record it and replace the candidate **before any model run** using a deterministic generator, then hash the replacement. Do not select based on behavior.

### 6.3 Replication protocol

When E3 training is verified, use the **exact same frozen E3 objective, parameter values, context registry and evaluation protocol** for new sequences: four factorial arms, paired seeds `0..19` per new sequence if compute permits. Run all sequences with the same declared count; if limited to 10, freeze the first ten seed identities in advance, do not stop early based on effects. Pilot hyperparameters from E3 remain fixed. No calibration per sequence.

Primary outcomes: target engagement (hidden/task separation and output-relevant closure defect), familiar CE, natural Mixed/NoContext/Novel30/Novel60 CE, and the same closure × separation interaction. Estimate paired effects per sequence. Summarize sign consistency and heterogeneity; do not treat all seed × sequence observations as independent evidence for a universal law.

If E3 fails its code correctness or feasibility gate, do not substitute another model. E4 becomes `blocked_by_e3` and may instead produce a **design-only** report for future work; never quietly convert to C0–C3 because the factorial treatment is not viable.

### 6.4 RED → GREEN tests

- `E4_T01`: all sequences satisfy constraints and frozen hashes;
- `E4_T02`: sequence is identical in A and B within every training run;
- `E4_T03`: no cross-sequence checkpoint or token-state leakage;
- `E4_T04`: same training objective, parameters, seeds, and number of steps for each registered sequence;
- `E4_T05`: no sequence-dependent coefficient search;
- `E4_T06`: aggregate first within seed×sequence, then report per-sequence inference;
- `E4_T07`: tests detect accidental pooling of multiple sequences as independent transitions or identical dataset;
- `E4_T08`: no probe-based sequence pruning or success-only reporting.

### 6.5 Deliverables

`configs/sequence_registry_v1.json`, registry hash, `e4_sequence_validation.csv`, `e4_sequence_seed_outcomes.csv`, `e4_sequence_effects.csv`, `e4_heterogeneity.csv`, `E4_RESULTS.md`, visual replication grid. State whether replication covers multiple sequences only or also broader architecture families (this PRD does **not** test GRUs/other architectures).

## 7. Optional future work — explicitly outside scope

Do **not** run in this program: GRU/LSTM replication, larger hidden sizes, new context encoder, true semantic context-category pretraining, added beta/rho learning pathway, transfer across differing action mappings, probe-targeted optimization, human-fitting objective, new human data collection, model-informed post hoc novel-cue definitions, or a larger sequence sweep. These can be proposed later as separate, separately justified programs.

## 8. Integration tests and scientific integrity gates

Before any E1/E2 evaluation:

1. Hash every loaded source checkpoint **before and after** replay; require identity.
2. Confirm the frozen Novel registry hash matches the existing paper freeze (`2b28095562ea69a33d96c521817dcd0743f3d66658385ac425212f6951e00727`) or explicitly stop and audit why source files differ.
3. Confirm common evaluator reproduces historical CE and logits within declared numerical tolerance; preserve `numpy_explicit_recurrence_v1` provenance if using it.
4. Verify no new optimizer creation/step, backward call, probe-triggered training, or source-output overwrite occurs in E1/E2.
5. Verify CE in nats, seven transitions, frozen sequence, proper hidden-state timeline, correct cues.
6. Verify both absolute and normalized geometry are retained and normalizer changes do not masquerade as hidden geometry effects.
7. Ensure error messages explicitly name missing checkpoint, source module, registry, or failed invariant.
8. All tests run in CI-friendly offline mode with synthetic fixtures; real-checkpoint regression tests run only when backing files are present and emit `blocked_missing_source` when absent.

Before E3/E4 training: pass synthetic gradient tests, model recreation tests, reproducibility tests, protected-probe firewall, exact run-matrix tests, and A/B-only calibration freeze. Existing source checkpoint files are immutable.

Use separate statuses:

```text
implementation_status: not_started | red_tests_written | validated | blocked | failed_integrity
scientific_status: not_run | completed_positive | completed_mixed | completed_null | insufficient_familiar_support | failed_target_engagement | competence_confound | inconclusive
```

Do not translate null scientific findings into software failures.

## 9. Repository layout / runner interface

Implement the smallest reusable package without dangerous cleanup of historical code:

```text
configs/dynamical_generalization_v1.json
configs/context_grid_v1.json
configs/factorial_v1.json
configs/sequence_registry_v1.json
scratchpads/dynamical_generalization_program.md
scratchpads/decisions.jsonl
src/context_sequence/dynamics/{jacobian,closure,propagation}.py
src/context_sequence/evaluation/{cue_grid,rollout}.py
src/context_sequence/training/{separation_regularizers,factorial}.py
src/context_sequence/analysis/{aggregation,contrasts,figures}.py
scripts/audit_dynamical_generalization_inputs.py
scripts/run_dynamical_generalization.py
tests/unit/test_e1_*.py ... test_e4_*.py
tests/regression/test_historical_parity.py
results/dynamical_generalization/<freeze_id>/
    manifests/
    tables/
    figures/
    audit/
    E1_RESULTS.md
    E2_RESULTS.md
    E3_RESULTS.md
    E4_RESULTS.md
    FINAL_DYNAMICAL_GENERALIZATION_REPORT.md
```

Expected CLI, adaptable to repository norms:

```bash
pytest tests/unit tests/regression
python scripts/audit_dynamical_generalization_inputs.py --config configs/dynamical_generalization_v1.json
python scripts/run_dynamical_generalization.py --stage e1 --mode eval_only
python scripts/run_dynamical_generalization.py --stage e2 --mode eval_only
python scripts/run_dynamical_generalization.py --stage e3 --mode ab_pilot
python scripts/run_dynamical_generalization.py --stage e3 --mode confirmatory
python scripts/run_dynamical_generalization.py --stage e4 --mode confirmatory
```

No implicit fallback from missing checkpoint to training. Every command records `git commit`, environment, source hashes, config hash, seed registry, and output manifest. Run results are immutable and versioned.

## 10. Claim-to-test decision table

| Claim we want to evaluate | What would support it | What would weaken it | What must not be claimed |
|---|---|---|---|
| Useful states exist but are dynamically inaccessible | Midpoint CE algebraically low; natural error high; one-step closure/propagation explains the gap; single patch decays | Natural failures explained almost entirely by static cue/input geometry; closure adds little | Midpoint patch alone discovers a naturally retrieved memory |
| Closure matters independently of separation | E3 closure manipulation changes closure defect and probe CE under both split regimes, with familiar CE comparable | Closure manipulation fails target engagement or changes only familiar mastery | Correlation between closure and CE proves causal mediation |
| Separation and transfer can dissociate | E3 strongly separated conditions can generalize when closure works, with familiar A/B controlled | Only less-separated or better-learned models improve | High vs Low alone isolates splitting causally |
| Context information controls types of generalization | Same-alpha/different-beta cues differ in C0/C2; bottleneck identities hold in C1/C3 | Differences disappear after norm and competence controls | Raw beta is proven psychological familiarity |
| Mechanism is sequence-general | Target engagement and core contrasts repeat across fixed independent sequences | Effects depend sharply on sequence and are not replicable | Three sequences establish universality across recurrent architectures |

## 11. Final stop rules

Stop E1/E2 after preregistered complete analyses. Stop E3 after the fixed A/B-only calibration grid and registered confirmatory seeds. Stop E4 after the registered sequences and seeds. **Do not** add conditions, change cue axes, change loss terms, expand λ grids, modify output heads, tune to Mixed/Novel, or switch network architecture to fix an inconvenient result. A null/inconsistent outcome is a completed scientific result.

If any foundational arithmetic/provenance/indexing test fails, pause scientific evaluation **of that dependency**, repair implementation under RED → GREEN, document the correction, and rerun the affected preregistered analysis. Do not redesign the scientific hypothesis during a bugfix.

## 12. Final deliverable and definition of done

Create `results/dynamical_generalization/<freeze_id>/FINAL_DYNAMICAL_GENERALIZATION_REPORT.md` with:

1. Exact scientific question and four experiment decisions.
2. A compact A/B acquisition and competence table for each cohort.
3. Midpoint-convexity proof and numeric audit.
4. E1 exact recurrence decomposition, Jacobian/approximation error, and genuine predictive results.
5. E2 cue-coordinate geometry and same-alpha/different-beta comparisons.
6. E3 A/B-only pilot provenance, realized splitting/closure target engagement, all factorial contrasts, competence-gate interpretation.
7. E4 results for each new sequence and heterogeneity.
8. All negative/null/blocked analyses and exact reasons; no imputed missing results.
9. Claim-to-source and figure-to-table map, hashes, and reproduction commands.
10. Plain-language answers: **What do we now know about shared output-compatible knowledge, recurrent accessibility, representational separation, and contextual information? What remains unresolved?**

**Definition of done:** RED/GREEN logs preserved; all unit, regression, and integration tests pass; immutable historical artifacts verified; frozen registries/hyperparameters and probe firewall audited; per-seed/per-transition data retained; inference respects clustering; figures are regenerated from tables; scratchpad/decisions updated; clean run instructions work; the final report is written. **Then stop.**
