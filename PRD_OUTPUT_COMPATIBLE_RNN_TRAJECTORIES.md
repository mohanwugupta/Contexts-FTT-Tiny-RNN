# PRD — Are Output-Compatible States Dynamically Usable?

**Project:** Mixed & Third FTT — SRN mechanistic extension  
**Document version:** v1.0, 2026-11-11  
**Status:** Prospective diagnostic program; do not amend the previous frozen experiment reports  
**Owner:** Researcher + coding agent  
**Implementation:** Test-driven development (RED → GREEN → REFACTOR), append-only scratchpad, immutable checkpoint inputs  
**Data scope:** Existing trained SRNs and teacher-forced next-token tasks. **No human modeling, no execution-RT head fitting, no new network training.**

---

## 0. Executive brief — the scientific problem

The same eight-item sequence is learned in familiar contexts A and B. Some recurrent networks acquire very different A/B hidden-state trajectories yet perform poorly under Mixed, NoContext, or out-of-support Novel contexts. A midpoint constructed from the A/B states can produce useful next-token predictions. But placing the network at that state once does not reproduce the benefit of repeatedly forcing the midpoint at every sequence position.

**Central question:** Does natural generalization fail because the recurrent network cannot follow the **specific A/B midpoint trajectory**, or because it cannot **reach and sustain any output-compatible trajectory** under the changed context?

We need to test two related accounts:

- **H-midpoint (specific closure):** the A/B midpoint (or its local neighborhood) is the behaviorally relevant shared trajectory; a failure to track this trajectory explains natural transfer errors.
- **H-viability (set-based dynamics):** the midpoint is one convenient output-compatible state, but other states—possibly far from it—could support equally good or better unpatched recurrent continuation. Natural generalization depends on entering and remaining within *some* viable trajectory, not necessarily the midpoint.

H-midpoint is a **special case** of H-viability. Finding a good, output-equivalent, non-midpoint recurrent trajectory refutes midpoint *necessity* for that context and horizon. Failing to find one does **not** establish necessity, because the search is finite and nonconvex.

**Deliverable to lab:** Two primary figures showing (1) what happens after a single state patch, and (2) whether output-identical initial states can have different future outcomes; one optional third figure diagnosing cue-only access. A final claim ledger must distinguish algebra, causal state intervention, oracle existence, and natural cue access.

### Minimal scientific conclusions we aim to adjudicate

| Candidate conclusion | Required evidence | Prohibited shortcut |
|---|---|---|
| Output-compatible states exist | Linear-head/Jensen audit + direct CE | Calling this natural retrieval |
| State information invisible to present output affects future behavior | Matched immediate softmax, manipulated hidden state, divergent unpatched future CE | Comparing states whose immediate predictions differ |
| The midpoint is not necessary for successful continuation | Prespecified **far-from-midpoint** state with fixed immediate output and successful unpatched future CE | Calling a one-step or repeatedly patched result successful |
| A viable trajectory exists but the natural cue does not access it | Viable state-only patch + failed native or cue-only continuation from fixed sources | Equating oracle discovery with natural access |
| Failed access is due to cue generation versus subsequent maintenance | Same-prestate cue branches and post-branch time course | Claiming global unreachability from finite tested branches |

### Program stages

1. **P0 — Reproduction and data contract.** Lock checkpoints, exact indexing, cue registry, and old transition results.
2. **P1 — Single-patch lag audit.** Reanalyze/replay existing interventions; make drift-versus-CE time course the starting point.
3. **P2 — Output-equivalent nullspace intervention.** Manipulate hidden information without changing the current softmax; measure future consequences.
4. **P3 — Viable-trajectory search.** Find (or fail to find) alternative output-equivalent starting states under frozen recurrence; distinguish a free hidden-state oracle from constrained familiar-near states.
5. **P4 — Cue-only access.** Test whether identical natural prestates and alternative context cues generate states that support the same useful future behavior.
6. **P5 — Synthesis and talk figures.** Explicit hypothesis adjudication, limitations, and practical next experiment.

**Execution gates:** P0 is mandatory. P1/P2 may run in parallel after P0. P3 requires a passing P2 mathematical intervention implementation, not a positive behavioral result. P4 requires P3 technical validity and at least one valid reference trajectory; if no trajectory is found, report the cue branch results without oracle-distance interpretations. Negative/null outcomes complete a stage. Implementation failures stop only the affected stage.

---

## 1. Source of truth and integrity rules

Before coding, read and index these files from the repository or attached project evidence:

- `FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md`
- `FINAL_EXPERIMENT_INFRASTRUCTURE.md` and `FINAL_EXPERIMENT_INVENTORY.json`
- `E1_RESULTS.md` and `FINAL_DYNAMICAL_GENERALIZATION_REPORT.md`
- `PHASE1I_CONTEXT_COMPOSITION_REPORT.md` (if present; otherwise its checked-in corresponding source and output report)
- `PHASE1J_SINGLE_INJECTION_CONTEXT_STABILITY_REPORT.md`
- `PHASE1K_CUE_GENERATION_REPORT.md`
- `PHASE1L_CONSTRUCTIVE_RECOVERY_REPORT.md`
- The actual checked-in model/recurrent-step, output-head, probe-cue and state-replay implementations.

Expected historical source families to discover, **not assume**:

- `confirmatory_srn/` and its checkpoint/provenance files
- `phase1i_context_composition/`, `analysis/phase1i_context_composition/`
- `phase1j_context_stability/`, `analysis/phase1j_single_injection_context_stability/`
- `phase1k_cue_generation/`, `analysis/phase1k/`
- `phase1l_contrast_closure/`, `analysis/phase1l/`
- `analysis/dynamical_generalization/` or the previously documented `dg_v1_20261103` results.

**Actual code and checkpoint bytes outrank narrative PRDs.** Do not trust a path merely because it appears in a report; find it, hash it, and record the actual function and state-dict key. If any key checkpoint is absent, report `blocked_missing_source` and the exact missing identity. **Never silently retrain it.**

### Cohort contract

- **Primary:** 20 canonical High-scale checkpoints, frozen model weights, seed identities unchanged. The High *monotonic* historical cohort may contain an execution head, but this PRD only evaluates **the next-token readout**. Preserving next-token softmax does **not** imply preserving an execution-speed readout.
- **Secondary:** C0–C3 Phase 1L **sequence-only** cohort, 20 paired seeds per condition. C0 is not interchangeable with the canonical High monotonic checkpoint. Treat these cohorts separately.
- **Optional control:** canonical Low-scale checkpoints; familiar CE is near floor and rescue contrasts can be unstable. Do not merge with High for the primary effect.
- No new training, fine-tuning, checkpoint selection, optimizer steps on model weights, changes to output heads, or synthetic replacement checkpoints.

### Fixed task / probe registry

Verify in code: eight tanh hidden units, four next-token output classes, sequence `(0,1,2,0,3,1,2,3)`, **seven scored transitions**, four one-hot token inputs, three cue channels, frozen A and B contexts, teacher-forced externally supplied tokens. Canonical cues include A `(1,0,0)`, B `(0,1,0)`, Mixed `(0.5,0.5,0)`, NoContext `(0,0,0)`, and previously frozen 30°/60° Novel directions on both A and B sides.

**Never reconstruct Novel cue vectors from prose.** Import the actual frozen registry and audit its SHA-256 hash. Historical Novel1–4 are a separate cue definition and must never be pooled with Novel30/60. The third context channel may be structurally inactive in the canonical model: verify and report.

### Frozen evaluation / firewall

- No updating trained model weights, no redesigning the cue registry, and no choosing checkpoint seeds based on probe errors.
- P2's random directions, counts, radii and outcome definitions must be frozen **before** viewing new P2 behavioral outcomes.
- P3 is an explicitly labeled **oracle**. It may optimize an **initial hidden state** using the correct future sequence targets under a specific probe. This is *not* learned retrieval, and it is forbidden to use oracle outcomes to retune the experiment or alter the checkpoint. The optimization budget and thresholds are fixed in this PRD.
- P4 is evaluation-only cue branching. No optimization or future-target information is allowed in the cue-only generation step; oracle states may be used **afterward** as diagnostic reference points.
- Preserve the previous E1–E4 reports as immutable historical outputs. New artifacts go under a new root.

---

## 2. Mandatory RED → GREEN → REFACTOR workflow

Create the scratchpad specified in §13 **before making substantive code changes**.

For **each** implementation unit:

1. **RED:** Write a minimal deterministic test of the contract, run it, and record the failing assertion and its intended reason. If the code already exists and the test passes, record `baseline_pass`, then create a mutation/negative fixture that demonstrates the test would fail for the relevant bug; do not manufacture a RED failure by damaging production code.
2. **GREEN:** Implement the smallest change that passes the test. Run the narrow test first, then the full relevant suite.
3. **REFACTOR:** Remove duplication; re-run all tests; ensure inputs, checkpoint hashes, and old frozen result aggregates remain unchanged.
4. **LOG:** Append test output, evidence path, decision, and any failure to `scratchpads/output_compatible_trajectories.md`.

**Do not proceed from tests to scientific analysis unless source fidelity, state indexing, softmax invariance, no-weight-update, and manifest tests pass.** Statistical significance is never a software acceptance test: clean nulls are valid outcomes.

### Required repository layout

```text
experiments/output_compatible_trajectories/
    config.py             # frozen registry, budgets, rules
    source_audit.py       # checkpoint discovery, hashes, replay parity
    state_geometry.py     # centered output map, nullspace, valid-state bounds
    continuation.py       # teacher-forced post-state forward runner
    lag_audit.py          # P1
    nullspace.py          # P2
    viability.py          # P3 constrained oracle
    cue_access.py         # P4
    analysis.py           # seed-level statistics, aggregation
    figures.py            # two primary panels + supplemental
    cli.py                # explicit evaluation-only stages

tests/output_compatible_trajectories/
    test_source_and_replay.py
    test_state_index_and_teacher_forcing.py
    test_output_nullspace.py
    test_valid_hidden_state.py
    test_patch_and_continuation.py
    test_oracle_and_constraints.py
    test_cue_branches.py
    test_analysis_and_export.py

scratchpads/output_compatible_trajectories.md
scratchpads/output_compatible_decisions.jsonl
results/output_compatible_trajectories/v1_20261111/
    manifests/
    audit/
    tables/
    figures/
    reports/
```

Use repository conventions where existing utilities already cover a function. Avoid copying trained-model code. Expose pure tested helpers for geometry, state stepping and scoring. Every stage must have a `--smoke` and `--eval-only` path. No implicitly invoked training pipeline.

---

## 3. Shared definitions and exact indexing

### 3.1 Recurrence and present output

Use the **actual** model implementation to determine indexing. The intended convention in this PRD is:

\[
 h_t = f_\theta(h_{t-1},x_t,c_t),\qquad
 \ell_t=W_o h_t+b_o,\qquad
 p_t=\operatorname{softmax}(\ell_t),
\]

where `t = 1,…,7` indexes scored next-item predictions, and the output at `t` predicts a known next sequence token `y_t`. If the repository uses different zero-based positions, implement an explicit mapping and tests. Natural states begin at the canonical reset state `h_0`, with exactly the existing externally supplied token order.

**State patch at position k:** first compute a valid reference post-update state `h_k`, replace it with `h'_k`, score its **present** output using `W_o`, then continue the unchanged recurrence from `h'_k` to `h_{k+1}` under exactly the same future tokens and cues. There is **one** state placement, and no more patches after `k` unless a separately labeled repeated-patch control is run.

Define future loss for a patch at `k`:

\[
 L_{>k}(h'_k,c)=\frac{1}{7-k}\sum_{t=k+1}^{7}
 \mathrm{CE}(W_o h_t+b_o,y_t).
\]

This explicitly **excludes the present (identical) output at k**. No inference for k=7 (zero future transitions). Primary injection positions: `k = 1, 3, 5` (six, four and two future transitions respectively). Primary full-horizon endpoint: `k=1`; k=3/5 are sensitivity for position dependence. Use all positions for the descriptive existing P1 lag audit where valid.

### 3.2 Familiar states and reference midpoint

For each seed and scored position, run the same externally supplied sequence under A and B from canonical `h_0`:

\[
 m_t=(h_t^A+h_t^B)/2,\qquad d_t=h_t^A-h_t^B.
\]

The A/B midpoint is a **reference**, not a claim that the human or network naturally computes an average. Record `||d_t||`, familiar CE, centered logit A/B separation, and the present midpoint CE.

### 3.3 Exact output equivalence

For `K=4` output logits, define a centering matrix

\[
 P=I_K-\frac{1}{K}\mathbf 1\mathbf 1^{\!\top},\qquad M=PW_o.
\]

Let the columns of `N` form an orthonormal basis of `ker(M)` (full-right SVD; rank tolerance recorded). For any `a`, construct

\[
 h'_k=m_k+Na. 
\]

Then `P(W_oh'_k+b_o)=P(W_om_k+b_o)` and `softmax(ell(h'_k))=softmax(ell(m_k))` **up to floating-point error**. For 8 hidden units and 4 logits, `rank(M) ≤ 3`, so nullity is at least 5. Assert actual dimensions instead of hardcoding the numerical rank.

**An immediate output-equivalent pair has the same four-class probabilities and CE, not necessarily the same output logits up to a shared additive constant.** Preserve centered logits and softmax, not raw-logit equality.

**Secondary anchor:** repeat key tests around the natural post-update probe state `h_k^c` (instead of `m_k`) to show results do not depend solely on constructing the midpoint. Keep the two anchor families separate.

### 3.4 Valid states and distances

Tanh post-update hidden activations lie in `[-1,1]^8`. All state interventions must stay within this box (use a fixed interior margin of `1e-6` for oracle feasibility). **Never elementwise-clip an output-equivalent state:** clipping typically violates `M delta = 0`.

Report absolute Euclidean distance to midpoint, relative distance

\[
 r(h)=\frac{\|h-m_k\|_2}{\|h_k^A-h_k^B\|_2+10^{-8}},
\]

and output-space distance to the midpoint. A `far` candidate is prospectively defined as `r(h) ≥ 0.25`. This is a reporting threshold, not an intrinsic state-space boundary; report continuous distances and fraction of positions where such a candidate is geometrically feasible.

For each valid intervention state, record position-specific and horizon-wide CE, worst-step CE, hidden norm, output-equivalence residual, box-bound margin, and number of future transitions.

### 3.5 Familiar-derived success criterion

For each seed and injection position, define the familiar reference threshold from **A/B only**:

\[
 B_{s,k}=\frac{1}{7-k}\sum_{t=k+1}^7 \frac{\mathrm{CE}_{A,t}+\mathrm{CE}_{B,t}}{2}.
\]

Primary `viable_continuation` criterion: `L_>k ≤ B_(s,k) + 0.10 nat`, with an explicit sensitivity score using the worse-endpoint average `mean_t max(CE_A,t,CE_B,t)+0.10`. Report a stricter zero-tolerance criterion `L_>k ≤ B_(s,k)` without choosing whichever favors a hypothesis. Always show actual absolute CE and each future transition; these thresholds are operational definitions, not discoveries about cognition.

The key falsification pattern is a **far, output-equivalent, single-placement state** that meets the primary viability criterion while the original midpoint under the identical future cue schedule fails it. A more demanding secondary pattern is `L_>k(h') ≤ L_>k(m) − 0.20 nat`, predefined as a practical effect threshold.

---

## 4. P0 — Source audit and reproduction (mandatory)

### Question

Can we reproduce the exact task, midpoint, one-patch branch and known frozen losses without training or changing any weights?

### Implementation

1. Resolve source files, checkpoints and cue vectors; record model family, seed ID, state-dict checksum, model code version, train/eval mode, dtype and device.
2. Recompute A/B and probe natural state trajectories under teacher forcing. Compare seed-level and condition-level CE with existing frozen tables at the actual saved precision.
3. Test the midpoint affine identity and Jensen bound. Reproduce a known repeated-patch result **only** as a parity check, never as a new scientific outcome.
4. Find original P1-relevant lag tables (Phase 1J / E1) and confirm injection-position, patch count and lag alignment. Mark missing columns rather than inventing them.
5. Write a single manifest containing every source, frozen config, cue registry and output hash.

### RED → GREEN tests

- **T00:** Synthetic sequence of distinct token IDs fails if next-token label shifted by one or reset state changed.
- **T01:** Replaying checkpoint reproduces known seed-level natural CE and state/logit traces within dtype-dependent tolerance; expected failures on a deliberately permuted checkpoint/seed fixture.
- **T02:** All model-weight tensors and checkpoint hashes unchanged before and after each evaluation stage.
- **T03:** Midpoint logits obey affine identity and Jensen bound within numerical tolerance; negative fixture uses wrong token label.
- **T04:** Single patch alters exactly one post-update state; future updates inherit it and no later step is replaced. Negative fixture applies patch at wrong position.
- **T05:** Every source row is uniquely identified by cohort, condition, seed, cue ID, injection position, branch, transition/lag and repeat ID.
- **T06:** A/B familiar CE / separation / Novel registry parity; C0 never silently mapped to High monotonic.

**Gate:** If these tests cannot pass because inputs are missing, stop with `blocked_missing_source`. If parity fails, stop with `blocked_replay_mismatch` and record numerical evidence; do not override expected results.

---

## 5. P1 — Reanalyze single-patch continuation by lag

### Scientific question

When the RNN is placed once at the midpoint, does prediction performance decay **as a function of recurrent continuation**, and does increasing distance from the midpoint necessarily imply worsening CE?

### Reuse before rerun

Use Phase 1J's `phase1j_transition_metrics.csv` and E1's `e1_single_patch_replay.csv` if they already preserve patch position, lag, raw CE, original unpatched CE and the reference midpoint. Recompute only missing compatible metrics using frozen checkpoints. Do not rerun acquisition or overwrite prior analyses.

### Registered comparisons

For every available seed, probe, injection position `k`, and future lag `l≥0`, compute:

- `CE_patched(k,l)` and `CE_natural(k,l)` under the **same** subsequent context; raw rescue `CE_natural−CE_patched`.
- `d_hidden(k,l)=||h^patched_(k+l)−m_(k+l)||`.
- `d_task(k,l)=||P W_o (h^patched_(k+l)−m_(k+l))||`.
- `same_output_now`: both immediate CE and four-class probability vector at l=0 match target parity.
- Following context D0 native, D1 zero and D2 physical Mixed, where available; preserve existing aliases (Mixed D0=D2; NoContext D0=D1) rather than double counting.

The primary story is the **paired lag-specific raw CE rescue and output-space drift**. Lag zero is an **exogenous patch** and must never be described as naturally generated. At future lag `l`, only injection positions with `k+l ≤ 7` contribute. Aggregate lag effects within seed and cue *before* seed-level inference; report the number of available positions per lag and avoid interpreting non-identical lag composition as pure temporal decay. Show fixed-k time courses (especially k=1) as a composition-safe companion.

### Decision patterns (descriptive, not tests of human theory)

| Pattern | Interpretation |
|---|---|
| Distance rises and CE rises | Consistent with midpoint departure impairing prediction; does not prove causation |
| Distance rises and CE stays low | Midpoint proximity is **not necessary** for low CE at those observed positions |
| Distance stays low and CE rises | Proximity alone is insufficient for good output under actual target/indexing |
| Both remain low | Midpoint may be locally viable for that cue and horizon |

### RED → GREEN tests

- **T10:** Lag zero equals the patched state's output and never the natural native state unless actually identical.
- **T11:** Patches at k=1,3,5 have exactly 6,4,2 future scored transitions.
- **T12:** Same-seed raw rescue equals unpatched CE minus patched CE row by row, no normalized-denominator sign mistakes.
- **T13:** Mixed D0/D2 and NoContext D0/D1 alias checks, where those branches are reused.
- **T14:** Per-lag aggregation weights injection positions correctly; no lost or multiplied `Novel30_A/B` or `Novel60_A/B` rows.
- **T15:** Identity fixtures show that output-space distance and hidden distance need not correlate with CE monotonically.

**Outputs:** `p1_lag_long.csv`, `p1_lag_seed.csv`, `p1_lag_bootstrap.csv`, `p1_fixed_k.csv`, `fig_p1_single_patch_timecourse.svg/png/pdf`, `P1_RESULTS.md`.

---

## 6. P2 — Output-equivalent hidden-state intervention

### Scientific question

Can two hidden states that make the **same present next-token prediction** produce **different future predictions** when the recurrent network receives identical subsequent inputs?

This establishes whether present-output equivalence conceals behaviorally consequential hidden-state information. Such divergence is expected in a recurrent model but must be quantified and related to *viable continuation*, not presented as a surprising fact by itself.

### Manipulation and fixed parameters

**Primary anchor:** familiar midpoint `m_k`. **Secondary anchor:** natural probe post-update state `h_k^c`. All following evaluations use frozen trained weights.

For each primary High seed, `k ∈ {1,3,5}`, and cue `c ∈ {Mixed, NoContext, Novel30_A, Novel30_B, Novel60_A, Novel60_B}`:

1. Compute `M = P W_o` in float64 and an orthonormal kernel `N` by full-right SVD. Save rank, singular values, tolerance (`max(M.shape)*eps_machine*sigma_max` or a justified documented conservative bound), and orthogonality residuals.
2. Generate **24 fixed RNG directions** in kernel coordinates per seed/k; evaluate both `+` and `−` signs. Generate two **A/B-calibrated target radii**, `0.10 ||d_k||` and `0.30 ||d_k||`, for 96 planned interventions per seed/position/cue. RNG index must be deterministic based on the stored seed/position/direction (not Python's randomized `hash()`).
3. Calculate each direction's largest geometrically feasible step within `[-1+1e-6,1−1e-6]^8`. Set realized radius to `min(target_radius, 0.90*feasible_max_radius)`. This may reduce radius; **do not clip components**. Preserve target and realized radii, direction, and feasibility flag. Zero-radius cases are flagged, not silently dropped.
4. Construct `h'_k=anchor+delta`, verify `||M delta||` is numerically negligible and the full four-class softmax matches the anchor. Apply the one-time post-state patch at k and score *only future* CE under the same fixed cue and teacher-forced tokens.
5. For each `delta`, generate a deterministic **output-visible row-space control** of equal realized norm, similarly bounded by the box; if a norm-matched control is infeasible, flag it rather than silently change its scale. Row-space controls are a software/geometry sensitivity analysis, not an output-equivalent comparison.
6. Record the unperturbed midpoint, the natural probe continuation, A and B familiar endpoint continuations, and repeated midpoint patches **as differently labeled references**. Do not use repeated-patch CE as the same outcome as single-placement future CE.

For the secondary natural-state anchor, repeat with **12 directions × 2 signs × one radius `0.10||d_k||`** on the same seeds/positions/cues. Keep outcomes separate from the midpoint-anchor primary set.

### Primary effect summaries

- Per-seed/cue/k distribution of `L_>k(h') − L_>k(anchor)` versus realized perturbation radius and sign; fraction improved, worsened, or indistinguishable at a fixed `0.01 nat` reporting tolerance.
- Fraction reaching the familiar-derived viability criterion (§3.5) and the fraction qualifying as `far` (§3.4).
- Immediate output equivalence residual and the distribution of future softmax/CE differences.
- Time-to-first loss of the familiar-derived per-transition criterion, where defined; report the full loss trajectory because horizon lengths differ.
- Pairwise comparisons use matched seed/position/cue/direction families. Bootstrap seeds, never directions or transitions as independent replicates.

**Interpretation:** A significant future difference after an output-identical immediate state is a **causal state-intervention effect**. It does not by itself show a naturally reachable alternative trajectory or rule out special midpoint closure. A distant, viable state is stronger evidence against midpoint necessity.

### RED → GREEN tests

- **T20:** `N.T @ N=I`, `M @ N≈0`, and `dim(N)=hidden_dim−rank(M)` on hand-built low- and full-rank synthetic heads.
- **T21:** Perturbed midpoint and anchor have identical centered logits, softmax probabilities, correct-token CE, and argmax within tolerance. Negative fixture uses a row-space direction.
- **T22:** Both states remain inside the valid hidden box; clipping is explicitly disallowed/tested as an invariance-breaking operation.
- **T23:** The **same** external future x/c/y and **same** frozen weights are used in both continuations; only one state differs.
- **T24:** Changes in future CE are zero in a synthetic SRN whose recurrence discards all hidden state, and nonzero in a synthetic SRN that carries a nullspace coordinate into an output-visible coordinate one step later.
- **T25:** Opposite signs and both radii are retained, all 20 seeds tracked, no pseudoreplication, and all intervention counts reconcile with manifest (including flagged zero-radius rows).
- **T26:** Visible-control perturbations actually change centered logits generically, while nullspace perturbations do not.
- **T27:** Present-output similarity checks run **before** computing future score, and malformed states never enter analysis.

**Outputs:** `p2_nullspace_states.csv`, `p2_future_transition.csv`, `p2_seed_summary.csv`, `p2_invariance_audit.csv`, `p2_effect_bootstrap.csv`, `fig_p2_identical_now_different_later.svg/png/pdf`, `P2_RESULTS.md`.

---

## 7. P3 — Does a different viable unpatched trajectory exist?

### Scientific question

If the midpoint performs poorly under recurrence, can another **output-equivalent** initial state—potentially far from the midpoint—support low CE for the **whole remaining sequence** under the same frozen cue and recurrent weights?

This is a **trajectory existence** assay. An oracle is allowed to use future target tokens to choose the starting state; that ability is never attributed to the SRN's natural retrieval mechanism.

### Parameterization and objective

With `N` from P2, parameterize candidate post-update states at position k by

\[
 h_k(a)=m_k+Na.
\]

Optimizing `a` leaves present softmax and CE invariant **by construction**. Optimize only `a`; **freeze** `W_h,W_x,W_c,W_o,b_*` and all other model parameters.

\[
 a^*=\arg\min_a L_{>k}(m_k+Na,c)
\quad\text{subject to}\quad
 -1+10^{-6}\le m_k+Na\le 1-10^{-6}.
\]

Analyze two **separate feasibility sets**:

- **Free-output-equivalent oracle:** any state in the intersection of the affine output-equivalence subspace with the tanh hidden-state box.
- **Familiar-near oracle:** same constraints and `||Na||₂ ≤ 0.5 ||h_k^A−h_k^B||₂`. This asks whether a more plausible neighborhood suffices; it is not a true learned-state manifold.

Keep “A/B-near” separate from “reachable by natural task history”; closeness is **not** reachability.

### Fixed optimization protocol

- Primary cohorts: 20 canonical High seeds, same `k={1,3,5}` and all six registered probe cues from P2. Same cue held constant for all subsequent steps (native future probe cue).
- For each seed/k/cue/constraint set: **8 random kernel-space initializations plus the exact midpoint**. All starts from fixed RNG seed schedule, constructed from A/B-only reference radii, with feasible backoff; no run-dependent restart selection.
- Use up to **150 gradient steps** per start with a declared optimizer such as Adam (initial LR `0.02`), with projection/backtracking **in kernel coordinates**, rejecting steps that leave the box or radius constraint. Do **not** clamp hidden coordinates. Before full execution, gradient-check the pure objective and verify exact output equivalence after every accepted step. A fixed step schedule is preferable to probe-dependent optimization tuning.
- Save **every restart** and its loss trajectory, constraint residual, converged/stalled status, hidden-state distance, and checkpoint ID. Best-of-restarts is allowed for **descriptive existence** but not as an unbiased natural-performance estimate. Save midpoint, random-initialization (unoptimized) distribution, and matching-budget oracle results.
- Technical smoke test: four predetermined seeds, k=1/3, Mixed/NoContext, two restarts, up to 10 optimization steps. This smoke test checks only gradient validity, feasibility, reproducibility, and runtime; **do not change the scientific contract based on whether probe loss improves**.
- After smoke tests pass, run the registered full design. If the method fails numerically, report failure with exact logs and apply a clearly documented numerical correction that does not alter the intended scientific comparison; otherwise retain the fixed design.

### Evidence and interpretation

For every condition and seed, report:

1. `L_>k(m)` versus best **free** and **familiar-near** oracle `L_>k`, and versus distribution of random unoptimized output-equivalent states.
2. Whether the best candidate is `far` (`r≥0.25`) **and** viable by the A/B-derived criterion.
3. Whether the midpoint fails while the far alternative succeeds on exactly the same future tokens/cue. This is the strongest falsifying case for midpoint necessity.
4. Future-transition losses and full four-class output probabilities, not just one average.
5. Hidden state validity, nullspace residual, and distance to familiar A/B state trajectories. Off-manifold scores are descriptive because familiar trajectories do not exhaust reachable states.
6. Free vs near oracle difference: does viability require moving beyond the predefined familiar neighborhood?

**Crucial scope:** A free oracle's success shows the *frozen recurrence admits* a good continuation from some admissible state. It does **not** show that natural cues can reach that state, that an optimizer-like mechanism exists in the network, or that the learned SRN stores the oracle state. Failure to find viability is `search_limited_inconclusive`, not proof of impossibility.

### RED → GREEN tests

- **T30:** Optimization gradients flow to `a` and recurrent computation but not to any frozen model parameter. Check pre/post checkpoint tensor hashes.
- **T31:** Every iterate preserves `||M(h−m)|| < tolerance` and lies within box (and radius cap for near oracle); a test rejects naïve clipping.
- **T32:** Analytic/autograd `∂L_>k/∂a` agrees with finite differences on a tiny synthetic recurrent network.
- **T33:** A synthetic case with a nullspace coordinate that determines future outputs has a known lower-loss solution: algorithm finds or closely approaches it within fixed budget.
- **T34:** A synthetic case with no future influence from nullspace coordinates yields no artificial oracle improvement.
- **T35:** Optimization excludes the patched present output at k, scores exactly `7−k` future positions, and contains no repeated patches.
- **T36:** Multiple restarts and best-of-restarts selection are exported, never counted as independent seeds or used to select a network checkpoint.
- **T37:** Viability and distance thresholds are calculated from source-frozen A/B values (not probe outcomes).
- **T38:** A no-feasible-state, no-improvement, or stalled-search result is represented honestly, never silently dropped.

**Outputs:** `p3_oracle_restart_long.csv`, `p3_oracle_best_seed.csv`, `p3_future_transition.csv`, `p3_viability_counts.csv`, `p3_gradient_feasibility_audit.csv`, `fig_p3_midpoint_vs_alternative.svg/png/pdf`, `P3_RESULTS.md`.

---

## 8. P4 — Can the context cue generate a useful state?

### Scientific question

If a viable trajectory exists after a state-only oracle patch, can a **physical cue change** generate an equally viable continuation from a natural prestate, without hidden-state placement or future-target optimization?

This separates **existence under frozen dynamics** from **finite-horizon accessibility by the specified cue family**. It cannot establish global state-space reachability.

### Same-prestate cue branch experiment

Re-use Phase 1K's matched-prestate branching implementation and provenance checks. For each seed and primary position `k={1,3,5}`:

1. Collect **bit-identical pre-update hidden states** from natural A and B trajectories and each natural probe trajectory at `k−1`; label the source `source_context` unambiguously.
2. Apply the exact same current token `x_k`, changing **only the cue** at that update: native, A, B, physical Mixed, zero, or the registered target Novel cue (one registered branch per target).
3. Record generated post-update state `h_k^branch`, immediate output CE, centered logits, future CE under a fixed **target probe cue** for all remaining transitions, and the corresponding fully native continuation from the same prestate.
4. Separate **one-step pulse** (only cue at k changed, thereafter target native context) from **persistent takeover** (branch cue used from k through transition 7). Do not conflate the two manipulations.
5. Record raw future rescue, distance to familiar midpoint, minimum distance to the **set of viable states found by P3** (diagnostic, not a mechanism proof), and whether the generated state itself meets the same future-viability criterion.
6. Compare with a state-only placement of the best P3 oracle state at k, followed by the *same* target future cue: this is a sufficiency reference, not an ordinary cue branch.

If P3 found no viable state, continue the predefined cue-only future-CE comparison but mark all “distance-to-viable-set” columns `not_established` rather than manufacturing a reference.

### Registered inference and interpretation

- **State exists; cue branch fails:** evidence for **failure of the tested cue branches to generate a viable starting state**, not proof that all possible contextual histories fail.
- **Cue branch initially does well, then worsens:** evidence for **limited continued viability** under the tested future schedule. Compare step-specific CE to avoid a purely mean-level artifact.
- **Cue branches differ by identical prestate source:** evidence that inherited state/history matters for context-dependent continuation in this trained network.
- **Cue branch and oracle both succeed:** evidence that some viable behavior is attainable through the tested physical cue route.

All branch outcomes are teacher-forced across **seven scored transitions**, not autonomous sequence-generation dynamics.

### RED → GREEN tests

- **T40:** Cue branch pairs have exactly identical bit-level prestate, token, weights, and future targets; only specified cue branch differs at k.
- **T41:** Native branch exactly reproduces its historical unmodified rollout; same-cue pulse equals native when there is no actual cue change.
- **T42:** Pulse/takeover differ only in the documented post-k cue schedule; no hidden-state patch in either cue-only arm.
- **T43:** A/B, Mixed, zero and Novel cue vectors match frozen source registry by exact values/hash.
- **T44:** Oracle-reference branch is kept in a separate intervention type and never counted as natural cue retrieval.
- **T45:** Seed, source, branch, cue, position and lag weights are preserved; no averaging branches as independent networks.
- **T46:** Missing viable-set results yield explicit NA/status rather than zero distances or fabricated target trajectories.

**Outputs:** `p4_same_prestate_branch_long.csv`, `p4_cue_access_seed.csv`, `p4_oracle_vs_cue.csv`, `p4_branch_bootstrap.csv`, `fig_p4_generation_vs_maintenance.svg/png/pdf`, `P4_RESULTS.md`.

---

## 9. Analysis, statistics, and scientific decision table

### Statistical unit and aggregation

The primary independent unit is the **trained seed** within one model/cohort. Cue families, Novel A/B variants, injection positions, future transitions, random perturbation directions and optimizer restarts are repeated measures on the same seed. Report counts for each but infer over seeds. For Novel30/60, retain A-side and B-side outcomes and average within seed only when a combined `Novel30Mean`/`Novel60Mean` is explicitly named.

Use paired seed effects with a fixed RNG bootstrap seed, 10,000 seed-bootstrap draws where computationally feasible, resample complete seed clusters, and save resample indices. Primary model: canonical High. C0–C3 descriptive replication is supplementary; separate cohort tables. Do not pool C0 with High canonical or High with Low. Report full effect distributions and intervals; do not infer that a descriptive improvement is a statistically supported mechanism.

### Model-independent causal contrast

For each output-equivalent perturbation, the manipulated hidden information is causal **within the frozen SRN** because its current softmax, next-token target, input sequence and recurrent weights are held constant. Future changes after this intervention are the causal effect of the post-state perturbation. This does **not** causally identify the origin of the *natural* Mixed/Novel deficit, and the viability oracle uses future information unavailable to ordinary retrieval.

### Hypothesis adjudication

| Observed pattern | Supportable conclusion | Must not claim |
|---|---|---|
| Future differences among present-output-equivalent states | Output state alone does not determine future behavior | The midpoint is unimportant |
| **Far** state succeeds, midpoint fails under same native cue | Exact midpoint is **not necessary** for a viable recurrent continuation in this model/context/horizon | All distant states succeed; humans do this |
| Far oracle succeeds, natural cue branches fail | A viable recurrent path exists but tested cue routes fail to reach/use it | It is globally unreachable |
| Natural cue branch succeeds and persists | A tested physical cue can generate a viable unpatched continuation | A unique latent-cause inference mechanism exists |
| All viable found states near midpoint | Consistent with midpoint neighborhood importance | The midpoint is uniquely necessary |
| No oracle succeeds | Search-limited evidence; report precise bound and optimization diagnostics | No viable trajectory exists |
| Midpoint distance increases without CE cost | Midpoint tracking is not needed for those *observed successful steps* | Long-horizon generalization is solved |
| All P2 effects essentially zero | No material future divergence detected at tested radii/horizons | All output-equivalent states are dynamically identical |

### Primary comparison order (freeze before analyzing)

1. P1: fixed-k lag traces of natural versus single-midpoint continuation, and CE versus task-space distance.
2. P2: within-seed future CE effects at matched immediate output and their viability/distance distribution.
3. P3: frequency of **far+viable** alternatives, especially cases where the midpoint itself fails; free versus familiar-near oracle.
4. P4: fraction of source/cue branches that meet viability and what happens at each later transition.

Do **not** switch the headline outcome to a secondary normalized score after seeing the results.

---

## 10. Figure plan for lab meeting and paper

**Figure A — “What happens when we stop helping?”**

- Natural, once-patched-midpoint, and repeated-patch references for each probe (clearly distinguish direct patch at lag 0).
- Main panel: **raw CE by post-patch lag** for k=1 and 3 (not a single pooled mean that changes composition).
- Adjacent panel: centered-logit distance to midpoint by lag, on its own y axis.
- Add paired-seed intervals; show one gray line per seed if legible.
- Interpretation caption: “A useful present output need not imply useful recurrent continuation. Descriptive drift is not a causal mediation test.”

**Figure B — “Same prediction now, different future?”**

- At left, clear schematic: `h=m` and `h'=m+Na`, equal present softmax, identical future inputs.
- At center, paired scatter/violin of `L_>k` for midpoint vs output-equivalent random perturbations, with actual realized perturbation radius.
- At right, **best bounded oracle** vs midpoint vs randomly sampled controls, showing `far` and viability status by seed. Do not present best-of-9 as an unbiased natural model outcome.
- Explicitly label the chosen position k and remaining scored steps.

**Figure C (conditional) — “Can the cue access a viable region?”**

- Same-prestate native/Mixed/A/B/zero/Novel cue branches.
- Show percent viable future continuations and per-lag CE, separately for pulse, persistent takeover, and state-only oracle.
- Show source-context A/B differences and clear tooltips/caption explaining the oracle is not natural retrieval.

**Supplement:** nullspace rank and numerical invariance; CE Jensen audit; all per-seed distributions; box/bounds feasibility; restarts and convergence; alternative radius; natural-state anchor; C0–C3 replication; legacy cue registry if relevant.

---

## 11. Deliverables and reproducibility contract

### Required Markdown reports

- `P0_SOURCE_AUDIT.md`
- `P1_RESULTS.md`
- `P2_RESULTS.md`
- `P3_RESULTS.md`
- `P4_RESULTS.md` (or an explicit status file if blocked)
- `FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md`
- `RESEARCHER_HANDOFF.md`

### `RESEARCHER_HANDOFF.md` must answer (without marketing language)

1. **Question:** What would distinguish exact midpoint closure from existence of other viable paths?
2. **Algorithm:** How were nullspace directions, norm/box constraints and matched output verified?
3. **Objective:** What was optimized, by whom, with what future information, and which weights were frozen?
4. **Evidence:** Which exact table rows and figures show an output-identical initial state with a better or worse future?
5. **Assumptions:** Why does the linear output head make a nullspace possible? What is not guaranteed by that fact?
6. **Causal scope:** Which results are state interventions, which are oracle existence results, which are natural cue interventions, and which are only prediction/correlation?
7. **Teach-back:** 60–90-second explanation of the result; one worked numeric example; one explanation of why the conclusion could be wrong.
8. **Next experiment:** Only if the observed pattern makes it necessary. No reflexive proposal to train a new network.

### Required machine-readable evidence

- `manifest.json` with hashes of code, configs, checkpoint states, cue registry, each output, executed command, version metadata and completion status.
- `claim_source_map.csv`: claim ID, exact metric, cohort, seed count, table, figure, test, caveat.
- `test_evidence.jsonl`: RED failure (or baseline_pass + failing negative fixture), GREEN result, REFACTOR suite status, git commit or diff hash.
- `run_registry.csv`: stage, cohort, seed, cue, k, branch, restart/direction index, status, output path.
- `decisions.jsonl`: append-only scientific contract changes with pre-/post-probe exposure.
- All report figures as editable-source PDF/SVG plus quick-view PNG. Data are preserved separately in CSV/Parquet.

### Required researcher-facing summary format

At the end of each stage, write **no more than one page** in plain language:

> We asked __. We changed __ while fixing __. We found __ (with seed-level uncertainty). This supports __. It does **not** show __. The remaining question is __.

Keep the detailed method, negative results, tests and provenance in appendices or audit files.

---

## 12. Stop rules and red lines

**Stop a stage and report rather than improvise if:**

- Checkpoint or cue identity is missing, corrupted or changed.
- Source replay and previously frozen CE cannot be reconciled.
- A purported output-equivalent state has changed current softmax/CE beyond declared numerical tolerance.
- P2/P3 accidentally uses a different future cue/token schedule between comparison states.
- P3 optimizer modifies any network parameter, changes model training mode, or violates box/nullspace/radius constraints.
- A report interprets an oracle state chosen with future targets as naturally accessible.
- Synthetic checks show incorrect patch position or double patching.
- An optimization result is cherry-picked by variable restart counts, altered thresholds, excluded failing seeds, or probe-specific retuning.
- E3/E4 from the prior program is silently resumed or its failed calibration gate overwritten.

**Scientific negative results are not failures to stop:** no better output-equivalent state, no oracle viability, no cue-only access, or no effect at fixed radii are reportable outcomes. Preserve their limitations and do not expand a search ad hoc to secure a positive result.

**At final handoff:** Do not claim unique midpoint necessity from non-discovery; do not call teacher-forced continuation autonomous generation; do not infer human hidden-state implementation. The scope is the trained SRN under the registered probes.

---

## 13. Scratchpad template — create and maintain live

Create `scratchpads/output_compatible_trajectories.md` with exactly these initial sections. Append entries; do not replace inconvenient decisions or unsuccessful runs.

```markdown
# Output-compatible trajectories — live scratchpad

## A. Snapshot and scientific claim
- Current git commit:
- Date/time / Python / torch / device / dtype:
- Source checkpoint root:
- Frozen model IDs / seeds:
- Frozen cue registry hash:
- Current question:
- Hypothesis that could be falsified:

## B. Source and implementation audit
| claim or assumption | source file:function:line | verified? | discrepancy |
|---|---|---|---|

## C. Active stage and gate
- Stage P0/P1/P2/P3/P4/P5:
- Prerequisites passed:
- Known blockers:
- Is any future-target oracle optimization allowed here? why?

## D. RED → GREEN → REFACTOR log (append-only)
| ID | proposed contract | RED failure / negative fixture | GREEN command+result | refactor regression | evidence file |
|---|---|---|---|---|---|

## E. Decisions (append-only)
- ID / UTC timestamp:
- Why needed:
- Information available at decision:
- Probe outcome already inspected? yes/no
- Changed methods or only implementation detail?
- Affected files and hashes:

## F. Run ledger
| stage | cohort | seed | cue | k | direction/restart | status | hash/output |
|---|---|---|---|---|---|---|---|

## G. Findings, including negative and blocked results
- Numeric result and unit:
- Uncertainty and independent unit:
- Strongest alternative explanation:
- What this result cannot establish:

## H. Researcher teach-back
- One-paragraph algorithm description:
- Why same current softmax? Explain using P W_o N = 0:
- Why can future differ? Explain using recurrent state update:
- What did an oracle know that the natural network did not?:
- One experimental falsifier:

## I. Open questions and next permitted action
- Current gate status:
- Next permitted task:
- Conditions that require researcher review:
```

---

## 14. First instruction to the coding agent

> **Implement this PRD with strict TDD. Do P0 first, create the scratchpad before altering scientific code, and show the RED → GREEN proof for source replay, state indexing and the `P W_o N = 0` nullspace test. Do not train any model. Reuse existing Phase 1J/1K and E1 analyses before recomputing. Proceed in stage order, keeping all config values fixed and recording full failures, negative outcomes, and hashes. Treat P3 as future-target-informed oracle state optimization, never as evidence of natural retrieval. Complete the report and figures even if P3 finds no viable trajectory. Produce `RESEARCHER_HANDOFF.md` with a teach-back, a claim-to-evidence map, and a clear answer to whether a non-midpoint viable continuation was found.**

