# PRD — Why Does Mixing Contexts Disrupt the Same Learned Sequence?

**Project:** Mixed & Third FTT — mechanism of failed contextual generalization in simple recurrent networks  
**Version:** 1.0 · 12 November 2026  
**Audience:** coding agent and researcher  
**Status:** prospective scientific extension, **not** an amendment to the previous frozen four-experiment, E1–E4, P0–P4, or S0–S4 programs  
**Development contract:** strict test-driven development (**RED → GREEN → REFACTOR**), append-only scratchpad, immutable source evidence, frozen analysis decisions  
**Goal:** explain why a network that predicts an identical sequence under A and B performs poorly under their physical mixture, then test whether training for *within-sequence context continuity* prevents that failure without merely erasing A/B separation.

---

## 0. One-page scientific contract (read before coding)

### The phenomenon

The fixed eight-item sequence is identical under context A and B. Yet some trained eight-unit Elman SRNs develop separated A/B trajectories and perform much worse with the physical Mixed cue, even though Mixed is a simple combination of learned cue channels. In the canonical High cohort, previous means were approximately A 0.917, B 1.456, and Mixed 2.248 nats of next-token cross-entropy (CE). High and Low networks differ in familiar competence, so **do not** interpret that contrast as evidence that splitting caused poor transfer.

Earlier frozen work established that repeated A/B midpoint placement can supply useful outputs, that alternative output-compatible hidden trajectories can sometimes outperform the midpoint, and that future-informed cue or state interventions can occasionally produce viable continuations. Those findings establish computational capacity, **not** why the network normally fails.

### Central question

> **When familiar contexts A and B support exactly the same sequence, why does their mixture disrupt performance? What information or computation fails, and can a specific learning experience prevent that failure while leaving A/B representations separated?**

### Three explicit candidate explanations

- **H1 — Nonlinear contextual mixing:** even from the same initial state, the recurrent nonlinearity converts a mixed cue into a state unlike the average A/B states; this mismatch is output-relevant and can propagate through the sequence.
- **H2 — Sequence-information transfer:** Mixed may disrupt the coding of sequence progress, especially when the current token alone does not determine the next item. Alternatively, progress information may remain recoverable but poorly expressed by the learned output head.
- **H3 — Missing context-continuity learning:** learning A and B in separate, context-stable sequences does not require the network to preserve sequence progress *through* a context change. Training on familiar within-sequence switches may improve held-out Mixed performance without necessarily merging A/B representations.

These are not mutually exclusive. Do not force all seeds into one mechanism or treat correlational diagnostics as causal proof.

### Minimal experimental program

1. **D0 — Source and numerical audit:** faithfully reproduce frozen A/B/Mixed trajectories and earlier CE; freeze cohort, geometry, and indexing.
2. **D1 — Exact mixing-failure decomposition:** identify the first mismatch between natural Mixed and averaged A/B trajectories; separate new nonlinear mixing mismatch from propagation of inherited mismatch. Evaluate output relevance and selected causal state-reset controls.
3. **D2 — Sequence-information diagnostic:** compare Mixed excess error for token-ambiguous versus token-diagnostic transitions; test carefully constrained A/B-trained readouts on natural Mixed states, subject to familiar-transfer sanity checks.
4. **D3 — Causal learning test:** from the *same already-trained High checkpoints*, compare matched additional practice under blocked, whole-sequence interleaved, and within-sequence A/B-switch schedules. Evaluate Mixed, NoContext and Novel only **after** all training configurations are frozen. Check familiar competence and A/B separation.
5. **D4 — Synthesis:** three figures, one claim/alternative ledger, complete methods, full source mapping, one researcher teach-back.

### Scope

- **D1–D2: evaluation only.** Do not update existing weights or fit trainable model components using probe targets.
- **D3: new training authorized only in three predeclared arms** using familiar A/B cues and the original token sequence; do not expose training to Mixed, NoContext, or Novel, including through a closure target at a Mixed cue.
- No new architectures, hidden units, loss terms, oracle state searches, cue-optimization searches, Jacobian/eigenvalue surveys, human-data fitting, or broad hyperparameter sweeps.
- **Do not reproduce the failed earlier E3 separation×closure factorial.** This is a curriculum intervention with the original loss, not an auxiliary-loss calibration problem.

---

## 1. Source-of-truth and version boundary

The agent must inspect the actual repository, code and checkpoint artifacts before implementing any scientific computation. The reports below are navigation/provenance aids; the **executed source implementation is authoritative** about states, optimizer, indexing and training objective.

### Required source references (names are expected; locate actual paths)

- `FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md` and `FINAL_EXPERIMENT_INFRASTRUCTURE.md`.
- `E1_RESULTS.md` and `FINAL_DYNAMICAL_GENERALIZATION_REPORT.md` (exact recurrence analysis).
- `FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md` (output compatibility).
- `FINAL_CONTEXT_CUE_ACCESS_REPORT.md`, `CLAIM_LEDGER.md` (cue intervention caveats).
- Canonical source model and training code, cue encoder, checkpoint loader, A/B and Mixed rollout code, frozen CE evaluator, source training logs.
- If historical state tables exist, verify hashes against saved manifests and reuse. If a needed optional historical table is an unavailable cloud placeholder, use **read-only faithful replays** from existing frozen checkpoints and retain a missing-input audit. Never invent matching file hashes.

### Known expected task; verify rather than silently hardcode

- Eight-unit `tanh` Elman network; four token channels; three raw context channels, of which the third is inactive in the target cohort; four next-token output logits.
- Sequence **`(0,1,2,0,3,1,2,3)`**; seven teacher-forced transitions, indexed `t=1…7`; `h_0=0` at sequence start unless the source demonstrates otherwise.
- A cue `(1,0,0)`, B cue `(0,1,0)`, physical Mixed cue **`(0.5,0.5,0)`**; NoContext `(0,0,0)`. Import exact Novel30/60 A- and B-side encodings from the frozen registry. Do **not** substitute earlier Novel1–4 cues.
- Natural High canonical checkpoints at final original training update **170**, seeds 0–19, for the primary analysis. If the historical family includes a monotonic execution head, **score only the next-token head** in D1–D4; do not confuse it with the C0 sequence-only cohort.
- Historical original training: 170 online sequences with A10/B10 blocks, Adam about `.005`, gradient clipping about `1.0`, High initialization scale about `2.0`. Verify objective, online-sequence update granularity, optimizer state availability, sequence reset and source tensor implementation before D3. No assumptions about execution-head training may be invented.

### Integrity contracts

- Create a new package, e.g. `experiments/context_mixing_sequence_continuity/`, and run root `results/context_mixing_sequence_continuity/v1/`.
- Hash original code, cue registry, source references and all 20 checkpoints before/after every stage. No edits to original training functions or saved weights; a thin documented adapter is preferable.
- Store run manifests with `seed`, `cohort`, `condition`, `token_index`, `target_index`, `checkpoint_sha`, `source_sha`, `config_sha`, and `source_data_sha`.
- Run new training only in D3, writing new checkpoints to a **separate** output directory; never overwrite original checkpoint paths.
- Unit of independent inference: **trained seed**, not timestep, intervention, token or candidate decoder. Preserve paired-seed identities.

**Stop condition:** if frozen natural A/B/Mixed next-token CE cannot be faithfully reproduced to the source-precision tolerance established at D0, stop with `blocked_source_fidelity`. Do not silently change the cue geometry, source objective, or numerical precision to make it match.

---

## 2. Shared definitions and metrics

Let the source recurrence and readout be

\[
h_t^c = f(h_{t-1}^c,x_t,c) = \tanh(W_h h_{t-1}^c + W_x x_t + W_c c + b_h),
\]

\[
\ell_t^c=W_o h_t^c+b_o,\qquad p_t^c=\operatorname{softmax}(\ell_t^c),\qquad CE_t^c=-\log p_t^c(y_t).
\]

- `c∈{A,B,Mixed,NoContext,Novel30_A,Novel30_B,Novel60_A,Novel60_B}` during **evaluation**.
- Use **physical Mixed** exactly `(c_A+c_B)/2`. A/B and Mixed have identical token sequence and correct next-item target at matched position.
- For each seed, save natural `h`, preactivations, logits, four probabilities, raw CE and correct-target logit margin at each of seven positions.
- Familiar comparator at each position: `CE_fam_t=(CE_A_t+CE_B_t)/2`, not the CE of the average probability; those are different nonlinear quantities.
- **Primary Mixed excess error:** `excess_t=CE_Mixed_t−CE_fam_t`; primary overall result is the mean across seven transitions, computed **within seed**, with a paired whole-seed bootstrap (10,000 draws, fixed seed 20260925) only after all test decisions are frozen.
- Report CE in **nats**, not RT; source's execution head and human RT are not used in this program.
- Report both **natural Mixed** and **predictive averaging baselines**: probability average `(p_A+p_B)/2`, logit average `(ell_A+ell_B)/2`, and hidden midpoint readout `ell((h_A+h_B)/2)` (the last two must agree exactly up to numerical precision). These baselines are *externally constructed*, not natural recurrent trajectories. Do not claim that the model naturally averages.
- Per-seed and per-transition tables must preserve individual A/B and Novel directions before any means.

### Structural sequence-history classification (preregistered)

Current token → future target labels in the seven scored transitions of `(0,1,2,0,3,1,2,3)`:

| Scored t | Current token | Correct next token | Same current token has another scored successor? | Category |
|---|---:|---:|---|---|
| 1 | 0 | 1 | Yes: token 0 at t4 goes to 3 | history-dependent |
| 2 | 1 | 2 | No: token 1 at t6 also goes to 2 | token-diagnostic |
| 3 | 2 | 0 | Yes: token 2 at t7 goes to 3 | history-dependent |
| 4 | 0 | 3 | Yes: token 0 at t1 goes to 1 | history-dependent |
| 5 | 3 | 1 | No other **scored** occurrence with different successor | token-diagnostic |
| 6 | 1 | 2 | No | token-diagnostic |
| 7 | 2 | 3 | Yes: token 2 at t3 goes to 0 | history-dependent |

Position 8 is terminal and has no scored next-item prediction. **Do not** use the label “history-independent” to imply that the entire recurrent network need not use history; it means only that current token identity uniquely determines the next item among the registered seven scored positions.

---

## 3. D0 — Audit and stage gates (no training)

### Question

Can we exactly replay the frozen natural trajectories, with correct state and transition indexing, before making new mechanistic claims?

### Required work

1. Find the executed canonical SRN class and replay function. Document layer sizes, activation, bias, input concatenation, initial hidden state, output timing, and numerical dtype.
2. Recover 20 High checkpoints and expected train-end update 170. Verify that original frozen parameter hashes remain unchanged.
3. Reproduce mean natural CE for A/B/Mixed, then one selected seed's seven per-position logits and all four softmax probabilities against prior saved evidence. Use tolerances justified by source float32/double operations rather than arbitrary exact equality.
4. Distinguish canonical High from the historical C0 sequence-only checkpoint; do not silently load the wrong family.
5. Inspect original training objective: token-only versus token plus auxiliary execution loss; inspect optimizer and reset behavior. Write a **D3 objective contract** before any training. If an exact continuation cannot be defined, D3 is blocked and D1/D2 remain valid.
6. Generate `audit/source_contract.md` with full paths, hashes, exact equations, tensor shapes and timeline; produce `d0_native_replay.csv` and `d0_manifest.json`.

### Required RED → GREEN tests

- `test_d0_expected_checkpoints_and_shas`: missing/changed checkpoint fails.
- `test_d0_token_target_alignment`: independently written toy sequence expects exactly seven targets and correct repeated-token transitions; off-by-one implementation fails.
- `test_d0_physical_mixed_identity`: Mixed equals the literal A/B raw-cue midpoint, third channel unchanged.
- `test_d0_native_source_replay`: reference frozen source logits/CE agree within registered precision tolerance.
- `test_d0_no_weight_change`: before/after tensor hashes identical; a deliberately modified fixture fails.
- `test_d0_cohort_identity`: canonical versus C0 must not alias silently.

**Gate:** proceed to D1 only after all D0 tests are GREEN. If D3 optimizer/source continuation has an unresolved contract, mark D3 blocked rather than guessing; D1/D2 may proceed.

---

## 4. D1 — Where does contextual mixing first break the computation? (frozen)

### Main question

If A and B both predict the same sequence, **what makes the natural Mixed trajectory depart from their averaged computation?** Does the mismatch originate in nonlinear contextual integration, carry forward through recurrence, and affect the next-item probabilities?

### Exact decomposition (central contribution of D1)

Define each familiar preactivation before `tanh`:

\[
a_t^A = W_hh_{t-1}^A+W_xx_t+W_cc_A+b_h,\qquad
 a_t^B=W_hh_{t-1}^B+W_xx_t+W_cc_B+b_h.
\]

Define familiar state midpoint `m_t=(h_t^A+h_t^B)/2`, its preactivation midpoint `\bar a_t=(a_t^A+a_t^B)/2`, and natural Mixed deviation `e_t=h_t^M-m_t`.

Since `c_M=(c_A+c_B)/2` and `x_t` is shared, the actual Mixed preactivation is exactly

\[
a_t^M=\bar a_t+W_he_{t-1}.
\]

Consequently, the **exact** difference between natural Mixed and the familiar midpoint splits into:

\[
\boxed{
 e_t = \underbrace{\left[\tanh(\bar a_t)-\frac{\tanh(a_t^A)+\tanh(a_t^B)}{2}\right]}_{j_t:\ \text{new nonlinear mixing mismatch}}
 +\underbrace{\left[\tanh(\bar a_t+W_he_{t-1})-\tanh(\bar a_t)\right]}_{r_t:\ \text{propagation of previous deviation}}.
}
\]

This is an algebraic identity, **not** a first-order approximation or a claim about attractor stability. Because all contexts start from the same `h_0=0`, `e_0=0`, so at **t=1** `r_1=0` and any hidden-state mismatch is due to `j_1` alone.

### Output relevance

Let `P=I−11ᵀ/4` center output logits. Map each hidden-state component into the task-relevant output contrast space:

\[
J_t^{out}=PW_oj_t,\quad R_t^{out}=PW_or_t,\quad E_t^{out}=PW_oe_t=J_t^{out}+R_t^{out}.
\]

Record norms, signed projections onto the correct-versus-competitor logit margin, and actual **CE for the natural Mixed rollout**. Show per-position output-relevant changes and resulting errors. Do **not** add component norms or assign a “percentage of CE explained”; CE is nonlinear and different component directions may cancel.

### Fixed counterfactuals (no new search)

- **Natural A, B, Mixed**: primary trajectories.
- **Midpoint readout**: state `m_t` supplied to output head at each position; *not* a recurrent rollout. This is a reference baseline.
- **First-position midpoint correction**: at `t=1` replace `h_1^M` with `m_1`, then let the frozen network update under **physical Mixed** for `t=2…7` without another replacement. Compare future per-step CE and raw/centered-logit deviation with natural Mixed. Label current t1 as exogenous assistance.
- **Later position-specific correction (secondary):** perform one midpoint placement at each registered `k∈{2,3,4,5,6}` then follow natural Mixed. Use the old E1/P1 single-patch outputs when exactly equivalent, otherwise source replay with explicitly matched schedule. Treat these as intervention diagnostics, not evidence that midpoint is necessary.
- **One-step no-prior-deviation counterfactual:** evaluate `f(m_(t−1),x_t,c_M)` solely to measure `j_t` at each position. This is NOT a self-consistent natural Mixed rollout; label it as a local diagnostic.

### Primary outcomes and comparisons

1. Seven-step plot for `||E_t^{out}||`, `||J_t^{out}||`, `||R_t^{out}||`, and natural `excess_t`, with whole-seed uncertainty.
2. Exact identity checks for every seed/transition (hidden and centered-logit space).
3. Effect of **first-position correction** on **remaining six-step CE**, paired within seeds, with natural Mixed future as reference; fixed single intervention.
4. Sequence-level comparison of natural Mixed to probability ensemble and midpoint readout; do not claim the baselines are naturally generated or that their success demonstrates compositional learning.
5. Negative examples where large hidden-state mismatch is output-irrelevant or correction fails; do not discard them.

### How to interpret

- If nonlinearity produces task-relevant deviation at `t=1`, this identifies the earliest computational source of mismatch; it does **not** prove it causes all seven-step failure.
- If correcting `t=1` improves subsequent CE, that is a causal test of the contribution of the initial state mismatch under otherwise unchanged Mixed continuation.
- If correcting `t=1` does little, later contextual mixing and recurrent-state dependence may dominate.
- If large hidden mismatches coexist with small output-relevant mismatches, avoid calling geometric divergence alone an information failure.

### D1 RED → GREEN suite

- `test_d1_midpoint_preactivation_identity`: synthetic linear affine source and actual frozen model.
- `test_d1_exact_jensen_propagation`: `e_t=j_t+r_t` for all 7 steps; deliberately incorrect sign fails.
- `test_d1_first_step_propagation_zero`: `r_1=0` for common initial states.
- `test_d1_linear_activation_control`: if tanh is replaced by identity in a synthetic fixture, `j_t=0` exactly.
- `test_d1_output_projection`: `PW_oe=PW_oj+PW_or`; mean-logit invariance.
- `test_d1_single_correction_schedule`: exactly one patch, future context remains Mixed, no weight update.
- `test_d1_ensemble_identity`: `(pA+pB)/2` probability simplex; midpoint logits agree with average logits; probability average and averaged logits are **not** asserted equivalent.
- `test_d1_seed_bootstrap`: 20 seed clusters, never 140 “independent” transitions.

**D1 gate:** export audited D1 output regardless of whether the intended mechanism is supported. No outcome-dependent changes to decomposition or intervention schedule.

---

## 5. D2 — Which sequence information is disrupted under Mixed? (frozen)

### Main question

Does Mixed disproportionately harm predictions that require knowing *where the system is in the sequence*, rather than just recognizing the current token? And is sequence position still recoverable from its naturally generated hidden state?

### D2A — Task-defined diagnostic (primary; no fitted decoder)

The task-defined history-dependent positions are **`{1,3,4,7}`** and token-diagnostic positions **`{2,5,6}`** (see §2). This classification uses only the **fixed sequence grammar**, never observed losses.

Within each seed compute:

\[
D_s = \operatorname{mean}_{t\in\{1,3,4,7\}}[CE^M_{s,t}-CE^{fam}_{s,t}]
 -\operatorname{mean}_{t\in\{2,5,6\}}[CE^M_{s,t}-CE^{fam}_{s,t}].
\]

Primary inference is paired seed-bootstrap CI for the mean `D_s` and the actual per-position excess CE plot. Report all raw A/B/Mixed per-transition CE and relative CE after source precision checks.

**Major confound:** history-dependent and token-diagnostic positions also differ in token identity, sequence position and difficulty. A positive `D_s` is **consistent with** difficulty maintaining sequence progress, not definitive proof that the network loses sequence position. Do not regress away position with the same seven fixed items and then claim unique identification. This analysis is a motivated signature, not a causal mediation test.

### D2B — Conservative familiar-only position/readout diagnostic (secondary)

Provide two predeclared decoders per trained seed, trained using **only familiar A/B natural hidden trajectories and position labels**, never Mixed/Novel labels or errors.

1. **Prototype transfer (no fit):** label the 14 familiar state prototypes `{h^A_t,h^B_t}` by sequence position `t=1…7`. For each natural Mixed state, use Euclidean and optionally whitened/regularized familiar-distance nearest-position assignment. Compute position accuracy and true-position distance. This is a geometric *familiar-prototype transfer* test, not a measure of how much abstract sequence information exists.
2. **Fixed-ridge linear position decoder (diagnostic):** fit a seven-class ridge regression from 8D hidden activity to position one-hot labels using the 14 A/B states for each model. Choose one fixed ridge coefficient (default `λ=1.0` after scaling each training feature by A/B-only pooled feature SD with a prespecified nonzero floor). No probe-based selection, no tuning on Mixed/Novel, no nonlinear decoder, no architecture search. Record numerical conditioning and training accuracy.

**Sanity gate for diagnostic interpretation:** Also train A-only and test on B-only, and vice versa, with the same preprocessing and fixed ridge. If familiar cross-context transfer is poor, the decoder is not validated as a shared position readout; classify Mixed decoding as `nonidentifiable_shared_decoder` rather than using failure as evidence of missing information. A/B-pool training accuracy alone is not sufficient because 14 training states give extremely little independent support. Do not exclude seeds from the primary D2A analysis.

**Optional diagnostic extension, only if part of the frozen protocol at D0:** compare the network's own output distribution with a *familiar-only learned* alternative linear next-token readout trained on A/B hidden states. It must obey the same small-data and transfer-sanity caveats and cannot enter primary inference or be described as a demonstrated human-like retrieval mechanism.

### D2 RED → GREEN suite

- `test_d2_sequence_repeated_token_map`: expected successor mapping and position labels exact.
- `test_d2_history_mask_fixed`: mask equals `[1,0,1,1,0,0,1]` (for seven scored transitions).
- `test_d2_excess_ce_familiar_reference`: exact per-position comparator and no leakage.
- `test_d2_ambiguous_contrast_seed_unit`: within-seed aggregation before inference.
- `test_d2_decoder_training_firewall`: probe states/targets never influence fitted parameters, normalizer or hyperparameters.
- `test_d2_familiar_cross_context_sanity`: synthetic known-shared/split examples correctly pass/fail transfer flag.
- `test_d2_underdetermined_decoder_flag`: low sample size, degenerate covariance, or failed familiar transfer marked as non-identifiable; never silently impute.
- `test_d2_no_probe_choice_of_lambda`: coefficient is frozen once before reading probe decoding.

**D2 gate:** produce the D2 findings regardless of whether decoder evidence is positive. The human-relevant statement is restricted to what these natural SRN trajectories show. Avoid equating decodability with naturally used information.

---

## 6. D3 — Causal learning experiment: can context continuity be learned?

### Core question

If a trained SRN has separated A/B representations, can a small dose of *familiar* within-sequence context switching teach it to preserve the same sequence computation across a context change? Crucially, does this improve **held-out Mixed** without simply eliminating A/B separation or improving familiar competence more than controls?

### Why continued training, not training from scratch?

Primary design starts all arms from the **same frozen High checkpoints at update 170**. This asks whether the learned context dependence is modifiable after it has developed. New independent training from initialization answers a different question and is **not** part of v1.

### Source objective and optimizer contract (mandatory pretraining audit)

- Inspect historical actual training code and recover exact differentiable loss. All three arms must use **identical source training loss** and optimizer settings; only the **cue schedule** differs. If the saved High model has an auxiliary execution head that received gradient updates, preserve that component *identically across arms* or use a documented objective-matched continuation; do **not** silently switch to token-only loss and claim curriculum-only manipulation.
- When checkpoint optimizer moments exist, restore the same moments into each paired arm. If moments are unavailable, restart a **fresh but identical Adam state** for all three arms and label runs `fine_tune_from_weights_optimizer_reset`. Original training history is then a starting-weight prior, not exact optimizer continuation.
- Use one original update unit per teacher-forced sequence, as source confirms; reset hidden state between sequences exactly as original source training did. No probe cue, probe CE, hidden oracle state, or midpoint-closure loss participates in training.
- The prospective continuation budget is **160 sequence updates** per seed×arm (16 full familiar blocks of 10); no checkpoint/epoch selection based on probe behavior. Retain interim logs for scientific transparency but primary evaluation is the fixed final step `170+160`.
- If original objective cannot be reproduced or a backward pass updates unexpected tensors, block D3 with `blocked_training_objective_fidelity` and report D1/D2 normally.

### Three arms (20 matched seeds; no per-probe tuning)

**Arm B — Blocked A/B practice (baseline):** 160 complete sequences with pure A for a block of 10, then pure B for a block of 10, repeated for 8 A/B pairs; **80 A-only and 80 B-only sequences**.

**Arm I — Interleaved A/B whole-sequence practice (schedule control):** alternate whole-sequence A and B for 160 sequences (80 A-only, 80 B-only). Context never changes **within** a sequence.

**Arm S — Within-sequence switch practice (critical intervention):** 160 identical token sequences, each with exactly one familiar A↔B cue change between scored positions 3 and 4. Alternate **A→B** and **B→A** for 80 trials each. At every scored position, familiar A and B each occur exactly 80 times across the arm. There is no exposure to Mixed, NoContext, or Novel; the third cue channel remains zero.

For all arms, the exact seven token inputs and targets are identical on every training sequence; only cue schedules differ. All arms see **80 A and 80 B cue occurrences at each scored position**, matched overall cue and token exposure; all receive 160 gradient updates. S has more within-sequence context transitions and therefore differs in the *location of contextual changes*, which is the intended causal manipulation. B and I help separate this from general continued learning and overall context interleaving. Do **not** claim that total contextual switch events are exactly matched across arms when they are not.

Freeze the **order and trial schedule** deterministically before looking at any protected probes. Use the same fixed schedule template for all 20 seeds, but use seed-specific stochastic optimizer randomness according to the source. Record actual A/B marginal counts by position and every within-sequence switch time as a manifest.

### Pure training A/B distribution checks (before any protected probe evaluation)

- All three arms processed identical counts of A and B at each scored transition and identical target counts, updates, optimizer settings and initial weights.
- No nonfinite loss/gradient/weights; all training updates complete. Training stability failures remain in outcome tables; no seed replacements.
- Evaluate **familiar A/B competence only** on the final checkpoints and record raw CE per A and B, mean, worst endpoint, difference from seed's starting checkpoint, token-level accuracy, and loss during continuation. Do not tune training length from these results.
- Measure A/B hidden and centered-logit separation at matched positions. Include raw norms and optionally normalized forms with denominator checks. An increase in performance with disappearance of separation is still a result, but is **not** evidence that split representations can coexist with generalization.

### Protected held-out evaluation (after objective, schedule and budget are sealed)

For each seed×arm final checkpoint, execute exactly the frozen natural evaluation on:

- **Primary:** Mixed `(0.5,0.5,0)`; seven-transition natural CE.
- **Secondary:** NoContext; Novel30_A/B, Novel60_A/B (exact frozen cue registry); also familiar A/B as necessary for comparisons.
- Never pick a best epoch/model/seed because of a Mixed or Novel result. Every seed retained.

### Registered effects

**Primary factorial contrast (within seed):**

\[
\Delta_s^{Mix}=CE_{s,S,Mixed}-CE_{s,I,Mixed},
\]

with a paired-seed 95% percentile bootstrap (10,000 draws, fixed seed) and raw per-seed plot. Negative values favor within-sequence switching over whole-sequence interleaving.

**Secondary:** `S−B` and `I−B` on Mixed, and `S−I` on NoContext / individual Novel cue sides; within-seed averaged Novel30/60; across-arm starting checkpoint change; per-position Mixed excess CE and history-dependent contrast from D2.

### Interpretation controls — report, do not retroactively select

Define *before probe inspection* two descriptive checks to qualify a successful transfer claim:

1. **Familiar competence comparability:** `|CE_familiar_mean(S)−CE_familiar_mean(I)| ≤ 0.20 nats` **at the cohort mean**, and report both A and B separately with paired seed intervals. This margin is a pragmatic preregistered interpretation check, not a statistical equivalence test. If it fails, report `familiar_competence_confounded`, not a pure invariance mechanism.
2. **Retention of splitting:** S retains at least **80% of I's across-position raw hidden A/B separation at the cohort mean**, and the absolute separation is non-negligible compared with frozen pre-continuation High separation. Report both raw hidden and output-task separation, not just ratios. If the apparent repair is accompanied by strong loss of A/B separation, label `repair_via_merging_possible`.

Do not drop nonmatching seeds or adjust these margins post hoc. **Do not gate performing the protected evaluation on achieving these margins:** all trained models should be evaluated, with the margins determining the strength of interpretation. The earlier E3 pilot's infeasibility problem must not reappear as a concealed stop or threshold-relaxation exercise.

### Causal inference boundary

Because paired starting weights and training objectives are fixed, the curriculum assignment identifies the **total effect of training schedule** within this model family. If S improves Mixed, it does not alone establish that improved sequence-position preservation *mediated* the effect. To argue mediation, use D1/D2 diagnostics measured on S/I final weights as **converging evidence**; do not claim a formal causal mediation effect unless independently manipulated.

### D3 RED → GREEN suite (write these before implementing training)

- `test_d3_identical_initial_parameters`: same seed has bit-identical starting weights in B/I/S.
- `test_d3_optimizer_contract`: only intended source parameters receive gradients; matched state across arms; wrong loss component fails.
- `test_d3_all_arms_same_update_count`: exactly 160 sequence optimizer updates, no hidden variation in mini-batch size.
- `test_d3_exposure_balance_by_position`: every arm has 80 A and 80 B cue occurrences at each of 7 scored steps; labels and tokens identical across arms.
- `test_d3_blocked_and_interleaved_schedule`: B has 10-sequence familiar blocks; I alternates whole sequences and has no within-sequence switch.
- `test_d3_switch_schedule`: S has exactly one switch at boundary `3→4` per sequence and contains both A→B and B→A orientations, 80 each.
- `test_d3_no_protected_cue_training`: no Mixed, NoContext, Novel or physical midpoint used anywhere in D3 training, auxiliary losses, schedule selection or gradients.
- `test_d3_source_training_step_fidelity`: known toy/source fixture produces matching gradients/parameter updates within numeric tolerance.
- `test_d3_frozen_initial_checkpoint_unchanged`: before/after original checkpoint hashes match even if new D3 output is produced.
- `test_d3_no_probe_checkpoint_selection`: primary final step is fixed, with no code path reading probe losses during training or choice.
- `test_d3_numerical_failure_retention`: nonfinite/loss anomalies never trigger seed replacement or changed learning rate.
- `test_d3_bootstrap_clusters`: paired comparison resamples 20 seed IDs, not 60 trained arms independently.
- `test_d3_split_measurement`: two identical frozen A/B trajectories yield zero separation, known divergent fixtures nonzero; output-centered projection correct.

### What each result would mean

| Observed outcome | Interpretation |
|---|---|
| S improves Mixed versus I; A/B competence comparable; separation retained; D1/D2 signatures improve | Strong evidence for learnable context-robust sequence continuation **after** splitting. Supports but does not prove the proposed information-preservation mechanism. |
| S improves Mixed but splits collapse | Repair may be due to merging/ignoring contexts, not robustness of separated representations. |
| S improves Mixed but familiar A/B also improves substantially more | Transfer may be due to better task acquisition; causal schedule effect remains but mechanism ambiguous. |
| I works as well as S | Varied whole-sequence practice is sufficient under this protocol; within-sequence continuity not specifically required. |
| S improves Mixed but not Novel | Interpolation in familiar cue plane and extrapolation to novel inputs are separate generalization problems. |
| No detectable S benefit | Training switches at this frequency/budget are insufficient; does not disprove all continuity-learning mechanisms. |

---

## 7. D4 — Synthesis, figures, and decision tree

### Figure 1 — Where Mixed computation fails

- **A:** A/B/Mixed natural CE across seven transitions (one line per context, seed-bootstrap uncertainty).
- **B:** Exact decomposition of natural Mixed−A/B-midpoint hidden difference into new nonlinear `j_t` and propagated `r_t` **after output projection**; show signed task-relevant effects or separate norms; never stack component norms as if additive CE.
- **C:** Single first-position midpoint correction versus untouched Mixed continuation and external midpoint reference; line plot by lag. Label teacher forcing and intervention.
- Caption conclusion must separate a mathematical identity from causal patch evidence.

### Figure 2 — What sequence information is present

- **A:** natural Mixed excess CE by position, colored by registered history-dependent versus token-diagnostic transition type.
- **B:** seed-level `D_s` contrast with bootstrap interval and all 20 points.
- **C (only if shared familiar decoding sanity succeeds):** A/B-trained prototype and fixed-ridge position decoder performance on A/B and natural Mixed; labels indicate tiny sample and diagnostic status.
- If familiar sanity fails, replace C with an explicit `diagnostic_nonidentifiable` panel; don't suppress the failure.

### Figure 3 — Can training preserve the computation through contextual change?

- **A:** three familiar-only continuation schedules with identical token targets and matched cue counts.
- **B:** paired-seed natural Mixed CE for B/I/S; primary S−I paired effect and CI.
- **C:** A/B familiar competence and raw A/B hidden/task separation in the same three arms; preserve failed/nonfinite seeds.
- **D (optional):** change in D1/D2 signatures for I versus S; descriptive, not mediation.

### Final evidence and manuscript outputs

Write:

- `FINAL_CONTEXT_MIXING_SEQUENCE_REPORT.md`: ~2-page plain-language story followed by exact methods, equations, statistics, failures and alternative explanations.
- `D0_SOURCE_AUDIT.md`, `D1_MIXING_BREAKDOWN.md`, `D2_SEQUENCE_INFORMATION.md`, `D3_CONTINUITY_TRAINING.md`.
- `CLAIM_LEDGER.md`: claim / tested alternative / evidence rows / failure status / allowed statement.
- `RESEARCHER_HANDOFF.md`: algorithm, objectives, parameters, assumptions, tests, negative cases, 60–90-second teach-back.
- `scratchpad.md`: append-only RED/GREEN log, run history, code decisions, errors, recovery steps, results and interpretation limits.
- `tables/` with long-form seed×cue×step data and paired-seed summaries. `figures/` in PNG+SVG; if PDF requested, use vector exports with valid embedded text.
- `manifests/` with source and output hashes, environment, commands, checkpoints, objective contract, seeding, code revision, exact cue encodings, exclusions/invalid statuses, and test results.

### Scientific decision tree

1. **D1:** Is there an output-relevant early Mixed mismatch? If yes, did first-position correction improve later CE? If no, inspect later mismatch; do not assume static geometry causes behavior.
2. **D2:** Are errors disproportionately history-sensitive? If yes, that's consistent with sequence-progress disruption; if no, broader output disruption remains. Decoder transfer success or failure refines but does not identify missing information unambiguously.
3. **D3:** Does switch-practice improve held-out Mixed relative to whole-sequence interleaving? If yes, is familiar competence comparable and splitting retained? If not, report schedule effect without strong mechanistic claim.
4. **Synthesis:** Only claim a learning-based explanation when training manipulation, computation diagnostics and competence/separation checks converge. Do not conflate any prior oracle cue/hidden optimization with naturally learned retrieval.

---

## 8. Required TDD workflow — RED → GREEN → REFACTOR

Implement exactly one stage at a time. **Do not write scientific implementation and tests in one unobservable batch.** For every unit:

1. **RED:** write a test specifying the source rule, algebraic identity, schedule invariant or result firewall. Run it and save the actual failure log. A failure caused solely by a missing import is acceptable initially, but add an independent behavior-failing fixture when appropriate (e.g. wrong indexing, wrong midpoint, probe leakage, incorrect exposure balance).
2. **GREEN:** implement the smallest code needed for tests to pass. Save command, exit status and test report. Verify source replay and unchanged checkpoint hashes.
3. **REFACTOR:** remove duplication, add type hints and validation, rerun the focused and historical read-only regression suites. Never call old tests that train scientific models without a disposable sandbox and explicit safe gating.
4. **SCRATCHPAD:** append the hypothesis, what was expected, what failed, exact code/file changes, numerical evidence, scientific interpretation, remaining uncertainties and next gate.
5. **FREEZE:** write stage manifest/hash before starting next stage. Do not overwrite earlier stage outputs or rerun with new thresholds without a separately versioned protocol.

### Required synthetic fixtures beyond stage tests

- Linear-recurrence identity: synthetic `tanh→identity` yields zero Jensen mismatch for physical Mixed.
- Nonlinear saturated-recurrence fixture: nonzero Jensen mismatch on the very first step, while recurrence propagation is zero at the first step.
- Constant-output head fixture: large hidden deviations but zero centered-logit deviation.
- Repeated-token fixture with exactly four history-dependent scored positions.
- Familiar-only decoder fixture whose A/B codes transfer, and a separate fixture whose A/B codes do not; correct `nonidentifiable` status.
- Three-arm cue-schedule fixture proving per-position cue exposures and input-target equality.
- Synthetic separation test with full merging versus intact A/B divergence.
- Probe-leakage trap: deliberately inserting Mixed into training data must fail a test.
- Resampling trap: intentionally resampling 7 positions as independent must fail.
- Missing/changed original checkpoint trap: stop, do not create a new scientific checkpoint.

### Computational limits and operational safety

- D0–D2 must run without optimizer steps. Only D2 diagnostic ridge uses closed-form/fixed regularized fitting on A/B states; its coefficients are **not model weights**.
- D3: at most `20 seeds × 3 arms × 160 online sequence updates = 9,600` planned continuation updates. The source may count each sequence's seven token transitions internally; preserve that definition explicitly.
- No search over switching locations, switch rates, optimization coefficients, architectures, probe cues, epochs or seeds. The first prospective switch location is fixed `3→4`; alternative schedules require a new PRD/version.
- Maintain a source/test-first fast smoke run on **two reserved seeds** solely to verify code and numerical stability, not select parameters. Once tests pass, execute the full 20-seed cohort. If smoke models reveal a serious implementation bug, log/fix under versioned test evidence before any protected outcomes are computed.
- Save checkpoints for all three arms and every seed at fixed final continuation update; even adverse/failing seeds remain in registries.
- If tests pass but the expected scientific effect does not appear, the agent must **report the negative result** rather than modifying the protocol.

---

## 9. Explicit prohibitions

**Do not:**

- tune any parameter based on Mixed/Novel loss, use probe labels in training, pretrain on physical Mixed cue, or silently train with a neutral closure rollout at `(A+B)/2`;
- change initial model architecture, hidden dimension, output head, source cue vectors, or historical checkpoint weights;
- reintroduce hidden-state oracle placements/cue optimization in D3;
- fit a high-capacity decoder to seven trajectory positions and call it evidence of information content;
- use a failed familiar cross-context decoder to conclude sequence information is absent;
- claim an averaged probability is the same as the softmax of averaged logits;
- declare a fraction of CE “caused by” `j_t` from norms alone;
- report threshold-based equivalence as a formal equivalence test;
- pool old C0 sequence-only and canonical High monotonic cohorts;
- infer human retrieval mechanism, biological hidden directions, generalization across architectures, or autonomous token generation from these source SRNs;
- silently drop seeds, merge Novel sides into pseudo-seeds, or replace missing data with zeros;
- train another model because the intended result was not obtained.

---

## 10. Acceptance criteria and completion contract

The program is complete only when:

- [ ] D0 reports exact source identity, full cohort, training objective, indexing, and unchanged frozen hashes.
- [ ] D1 exact mixing decomposition passes all source and synthetic tests; its numerical counterfactuals are labeled correctly.
- [ ] D2 primary token-history contrast includes all 20 seeds; decoder sanity and interpretability statuses are explicit.
- [ ] D3 has all 60 planned paired-arm model records, or an explicit blocked objective/implementation status; no protected evaluation leaks into training.
- [ ] D3 schedules are matched by token, position-level A/B exposure and 160 update count, with exact same source objective.
- [ ] Familiar competence and A/B hidden/task splitting are presented with the primary Mixed contrast.
- [ ] Per-probe effects use seed-level paired uncertainty; no transitions treated as independent.
- [ ] At least one adverse example and one illustrative example are retained, regardless of expected sign.
- [ ] All RED and GREEN tests are logged, plus refactor regression tests; stage outputs have immutable hash manifests.
- [ ] At least 3 main figures and concise lab-meeting presenter summaries are produced, even if the hypothesized effect is null or the diagnostic decoder is not interpretable.
- [ ] Final report states **what failed**, **what information remains uncertain**, and **what observation would change the conclusion**.

### Suggested code organization (adapt to repository structure; do not create duplicate source model classes)

```text
experiments/context_mixing_sequence_continuity/
  __init__.py
  source_adapter.py
  replay.py
  mixing_decomposition.py
  history_diagnostic.py
  familiar_decoder.py
  curriculum.py
  training.py
  analysis.py
  plots.py
  cli.py
  verify.py

tests/context_mixing_sequence_continuity/
  test_d0_source.py
  test_d1_mixing.py
  test_d2_information.py
  test_d3_curriculum.py
  test_d4_reporting.py

results/context_mixing_sequence_continuity/v1/
  scratchpad.md
  D0_SOURCE_AUDIT.md
  D1_MIXING_BREAKDOWN.md
  D2_SEQUENCE_INFORMATION.md
  D3_CONTINUITY_TRAINING.md
  FINAL_CONTEXT_MIXING_SEQUENCE_REPORT.md
  CLAIM_LEDGER.md
  RESEARCHER_HANDOFF.md
  tables/
  figures/
  checkpoints_new/
  audit/
  manifests/
```

### Final researcher teach-back, required in `RESEARCHER_HANDOFF.md`

In 60–90 seconds, explain:

> “The network predicts the same sequence in A and B but often performs worse when their cues are mixed. We traced the Mixed computation and separated the *new nonlinear mismatch* caused by averaging inputs from the *propagation of earlier mismatch* through recurrence. We tested whether errors are concentrated on transitions that require sequence history and whether familiar-trained position information transfers to the natural Mixed hidden states. Finally, we restarted training from the same split checkpoints under matched blocked, interleaved and within-sequence-switch schedules. Mixed and Novel cues were never used for training. If within-sequence switching improves held-out Mixed performance while familiar competence and A/B separation remain comparable, that supports learning of context-robust sequence continuation. If splitting collapses, or familiar competence differs, the explanation is narrower. The model is teacher-forced, and these results do not by themselves establish a human mechanism.”

**Do not substitute a summary of the code for a demonstration that these statements are supported by source measurements.**
