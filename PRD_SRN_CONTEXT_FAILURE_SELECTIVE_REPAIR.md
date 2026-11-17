# PRD — Why the same learned sequence fails under Mixed: state, update, and selective repair

**Status:** Implementation-ready v1 · **Study:** canonical High-scale monotonic SRNs, seeds 0–19 · **Primary probe:** physical Mixed · **Implementation:** frozen evaluation only · **Training:** gated follow-up, **not** authorized in v1

## 0. The scientific thread

**Central question.** Why does the physical mixture of two familiar contexts disrupt the network's prediction of the *same* learned sequence?

**Target explanation.** “The SRN fails under Mixed because **[an identified contextual operation]** disrupts **[a task-relevant recurrent computation]**. Selectively preserving that computation restores sequence predictions, without replacing the entire hidden trajectory or modifying learned weights.”

The goal of v1 is to make the first two blanks testable. We are not trying to find another helpful hidden state or context cue. We are testing whether the *specific changes naturally produced by Mixed* cause its prediction failures, and whether repairing only those changes restores performance.

Three results already constrain the work:

1. The High SRNs predict the same sequence under familiar A/B but fail under natural Mixed (mean next-token CE: A **0.917**, B **1.456**, Mixed **2.248** nats). Familiar competence differs from Low; splitting alone has not been isolated as a cause.
2. External midpoint states and some non-midpoint states support useful predictions. P2 shows that two states with identical current softmax probabilities can generate different futures. These establish **capacity**, not the natural cause of failure.
3. P4, Phase 1K, S2, and D1 have already manipulated prestates, cues, and early state deviations. Reuse these records before running any new rollout. D1 identified initial nonlinear mismatch, but one-time midpoint correction did not clearly rescue future performance. D2 did not identify lost sequence-position information; D3's switch curriculum did not provide a clean training repair.

**What success looks like.** A preselected direction or subspace, derived without Mixed outcome labels, explains part of the natural Mixed error. Correcting only that component yields better **unassisted future predictions** than equally large control interventions; adding the disruptive component to a useful state impairs future predictions. The original source weights remain fixed throughout.

**What counts as a useful negative result.** The factorial may show that Mixed cue updates fail even from useful familiar prestates; a proposed future-sensitive subspace may not selectively rescue performance; exact current-output-preserving repair may have no effect. Retain and report these outcomes. Do **not** optimize new hidden states or cues until a positive result appears.

## 1. Hard scope and anti-leakage rules

- **Primary cohort:** exactly the 20 original `High_monotonic` checkpoints, final original update **170**, seeds **0–19**. Eight tanh hidden units; four one-hot sequence inputs; four next-token logits; three raw context channels, with third channel inactive in canonical checkpoints. Use original float32 source recurrence for all claimed outcomes.
- **Do not pool** Low, sequence-only High, C0–C3, retrained D3 endpoints, or ordinal-layout models into the primary estimates. Low is an optional *predeclared descriptive control* only if the same code path/checkpoints can be verified.
- **Canonical source convention:** `h_t = tanh(W_h h_(t-1) + W_x x_t + W_c c_t + b_h)`, `z_t = W_o h_t + b_o`. The seven teacher-forced inputs `(0,1,2,0,3,1,2)` predict `(1,2,0,3,1,2,3)`; `h_0=0`. Position `k` means the output after consuming input `x_k`. Check the actual source indices before analysis.
- **Primary target** physical Mixed; **secondary** NoContext, Novel30_A/B, Novel60_A/B, only after methods and thresholds are fixed. Do not infer across different physical cue definitions in old studies.
- **Primary intervention positions:** `k=3` (four subsequent predictions). `k=5` is a secondary replication (two subsequent predictions). `k=1` is **not** a history test because all histories begin at the same zero state.
- No training, new loss, optimizer updates, cue optimization, P3 target-informed oracle selection, or hidden-state optimization in required stages R0–R3. P3 solutions may appear only as separately labeled historic references.
- All future evaluations receive **identical teacher-forced tokens** and the **same Mixed cue from k+1 onward**. Changing future cues changes the question; persistent takeover rows must not be mixed with pulse rows.
- Use the original source next-token head for scoring. The separate monotonic execution policy is irrelevant to frozen prediction but must not be mistaken for the output head. No autonomous-generation claim.
- Keep full-cohort results, all failed references, failed feasibility checks, and negative signs. The trained seed, **not** a position, direction, probe, or restart, is the independent unit.
- Evaluation labels may calculate CE and viability **after the intervention set is frozen**. They must not choose the subspace, direction, reference state, intervention amplitude, or control for primary inference.

## 2. Read-only predecessor audit: R0

**Question:** Which parts of the proposed state × cue test already exist?

**Read first:**

- `FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md`, `P4_RESULTS.md`; P4's natural-prestate × cue **pulse** branch data and source manifest.
- `PHASE1K_CUE_GENERATION_REPORT(1).md` and source `phase1k_local_cue_branches.csv` / pulse tables.
- `FINAL_CONTEXT_CUE_ACCESS_REPORT.md`, `S2_CUE_HISTORY_RESULTS.md`, `S3_LOCAL_CONTROL_RESULTS.md` for common-future contracts and derivative fixtures.
- `FINAL_CONTEXT_MIXING_SEQUENCE_REPORT.md`, `D0_SOURCE_AUDIT.md`, `D1_MIXING_BREAKDOWN.md` for source/indexing and the exact `e=j+r` audit.
- `P2_RESULTS.md` and `FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md` for the centered output-kernel/SVD implementation.
- Actual source loader and frozen checkpoint hashes. Treat report filenames as leads; discover the real repository paths, never invent them.

**Output:** `audit/R0_SOURCE_AND_REUSE_MAP.md` and `tables/r0_reuse_registry.csv`, one row per required factorial cell, with source path, source hash, seed, k, inherited context, current cue, pulse-vs-persistent flag, future cue, token indexing, available fields, and status `exact_reuse / recompute_required / incompatible / unavailable`.

**Strict reuse rule:** A legacy result is reusable only if the checkpoint, natural prestate, current cue, input token, output-step indexing, future schedule, and output-head convention match. Verify at least one full state/logit/probability replay per branch type and **every reused natural baseline**. A Phase 1K geometric “closer to midpoint” score is not a substitute for next-token CE. Recompute only genuinely missing or incompatible cells through the source, without changing old artifacts.

**Gate:** Do not start R1 until 20 checkpoint identities, natural A/B/Mixed losses, precise pulse/continuation semantics, and old-vs-new replay tolerances are verified. Missing original evidence must be an explicit blocker or recompute reason, never silently treated as a zero.

## 3. R1 — Is the problem the inherited state, current cue update, or continued Mixed recurrence?

### 3.1 Plain-language question

If we give the network the same next token and the same current Mixed cue, does it matter whether the state came from A, B, or Mixed? If a familiar history makes the next prediction better, does that advantage persist during ordinary Mixed continuation?

### 3.2 Predeclared 3 × 3 experiment

For each seed and each `k in {3,5}`:

1. Generate three *natural*, unpatched prestates `h_(k-1)^A`, `h_(k-1)^B`, and `h_(k-1)^Mixed` by running the same prior tokens under constant A, B, or Mixed, respectively.
2. From each identical-within-row prestate, deliver **one** current cue `c_k in {A,B,Mixed}` with the common actual token `x_k`. Record the resulting `h_k` and four-class `p_k`.
3. Continue each branch for positions `k+1,...,7`, with the **physical Mixed cue** at every future step and identical teacher-forced tokens; apply no additional patch.
4. Primary output is one 3 × 3 matrix per seed/k. Preserve all 20 seeds and all nine cells, whether favorable or adverse.

The native Mixed row is `(prestate=Mixed, cue=Mixed)` and must exactly reproduce the natural Mixed source. Familiar-history × Mixed cells test state history with the current cue held fixed. Within a row, changing the cue tests the immediate cue-update effect with prestate fixed. The 3 × 3 interaction tests whether cue effects depend on inherited state.

**Important:** `(A prestate,A cue)` followed by Mixed *future* cues is not an ordinary full-A rollout. Its **current** output matches familiar A; its future is deliberately a counterfactual Mixed continuation. Do not mislabel it as natural A performance.

### 3.3 Outcomes and comparisons

- Per-step correct-target CE, correct-target margin, complete four-class probabilities, centered logits, and hidden states.
- `current_ce = CE(p_k,y_k)` and `future_ce = mean(CE_(k+1):CE_7)` separately. Report `full_remaining_ce` (`k..7`) only as a clearly labeled secondary endpoint.
- Import the *unchanged* familiar-reference viability function from `experiments/output_compatible_trajectories/continuation.py:future_summary`: mean future CE ≤ position-specific mean familiar A/B future CE + 0.10 nats. Re-audit the exact actual function, threshold and indexing in R0. Also report continuous CE, worst-step CE, and zero-slack sensitivity; do not turn a one-step accuracy improvement into a viable trajectory claim.
- Pairwise within-seed **state-history effect** under identical current Mixed cue: `(A prestate,M cue)−(M prestate,M cue)` and analogous B, for current and future CE.
- Pairwise **cue-update effect** for a given inherited state: `(source prestate,M cue)−(source prestate,source cue)` (same common Mixed future). Also report A/B swapped cue cells; do not collapse different branches.
- State × cue interaction using within-seed contrasts. Treat a 3 × 3 heatmap as descriptive; use seed-cluster intervals on predeclared contrasts.
- Per-step success/failure onset: define `margin_t = z_correct − max(z_other)` and show both continuous margin and CE after the cue. **Do not infer latent state memory from the first error time alone.**
- For the k3 primary analysis, report whether useful familiar-history × Mixed branches (if any) (i) are already successful at k, (ii) remain successful for k+1..7, or (iii) deteriorate. Counts are seed-level and nonexclusive where both A and B qualify.

### 3.4 Interpretive decision table

| Pattern | Inference supported | Not established |
|---|---|---|
| A/B prestate + Mixed cue gives better current and future CE than M prestate + M cue | Inherited-state access can matter | That A/B state is normally retrieved under Mixed |
| Same prestate, switching current A/B cue to Mixed immediately worsens output | The current context update has a causal effect | Which hidden coordinate causes the effect |
| Good immediate prediction, poor later future under common Mixed cues | Single successful output does not ensure future maintenance | That a particular attractor or sequence memory is lost |
| State and cue have a strong interaction | Input consequences depend on inherited state | A unique underlying mediation pathway |
| No familiar prestate is useful | The proposed familiar-history route is insufficient | That no viable state exists (P3 found other routes) |

**R1 deliverables:** `R1_STATE_X_CUE_REPORT.md`, `tables/r1_branch_rollouts.csv`, `tables/r1_seed_contrasts.csv`, `figures/fig_1_state_by_cue.png`, `figures/fig_2_error_by_lag.png`, and source fidelity evidence. Explicitly state how many cells were reused versus newly evaluated.

## 4. R2 — What part of the natural Mixed state change affects future sequence predictions?

### 4.1 The key measurement problem

Hidden-state distance alone is not explanatory. Some changes alter the *current* output; others are invisible to that output but can influence later steps. P2 already showed the latter possibility for random interventions. Here we must test the **actual natural Mixed-to-familiar-reference difference**, not a random displacement.

We will use two complementary, **predefined and label-free** descriptions of the hidden space:

1. **Current-output geometry.** Which coordinates can affect the current four-class softmax?
2. **Future sensitivity.** Which currently output-invisible coordinates can affect future logits under familiar continuation?

No sequence-position decoder is required. D2's previous shared decoder was nonidentifiable for most seeds, so do not retry it with tuned regularization or interpret its failures as absent information.

### 4.2 Current-output-visible and output-silent geometry

Let `P = I_4 − (1/4)11ᵀ`, `O = P W_o` for each frozen seed. Use the **full** right SVD of `O` to obtain orthonormal bases:

- `V` spans the row space of `O` (audited rank 3, nominal shape 8 × 3);
- `N` spans `ker(O)` (audited nullity 5, nominal shape 8 × 5).

For any natural post-update Mixed state `h_k^M` and a prespecified familiar-derived reference `h_k^R`, define `d = h_k^R − h_k^M`, and project:

- `d_visible = V Vᵀ d`,
- `d_silent = N Nᵀ d`.

Check `d_visible+d_silent=d` at the declared double-precision tolerance, and `P W_o d_silent=0`.

**Crucial implication:** adding `d_silent` to `h_k^M` leaves its **entire current softmax distribution** identical in exact arithmetic, but may alter the future. Float32 replay can differ by roundoff; retain the existing P2 roundoff audit rather than asserting bitwise equality.

**Reference policy (fixed before testing outcomes):** For each k/seed evaluate **both** `(A prestate, Mixed current cue)` and `(B prestate, Mixed current cue)` from R1, with neither chosen by performance on Mixed. Also export natural full-A/full-B states as separate diagnostic references. Do **not** choose whichever reference yields lower Mixed CE for the primary analysis. Report reference viability afterward, and any analyses restricted to viable references as explicitly conditional and secondary. No P3 future-label oracle state is a primary reference.

### 4.3 Familiar-derived future-sensitive directions within the silent space

Use the frozen recurrence to calculate a finite-horizon **Jacobian of future centered logits**, not an attractor or fixed-point analysis.

For each seed and k, start from natural familiar post-k state `h_k^A` or `h_k^B`; continue with that familiar cue and the fixed future token inputs. For each familiar context `g`, define:

`G_g(k) = ∂ vec(P z_(k+1)^g, ..., P z_7^g) / ∂ h_k^g`, shape `(4H) × 8`, with `H=7−k`.

Stack `G_A` and `G_B`: `G_fam = [G_A; G_B]`. Derivatives use **weights and known input schedule only**, not the correct next-token labels, Mixed losses, or optimizer-based search. Source autodiff and a finite-difference fixture must agree.

Now analyze `G_fam N`, which describes the future influence of currently output-silent directions. Its right singular vectors give a fixed ranking of directions within the 5D output kernel. In v1 define the primary **two-dimensional** future-sensitive subspace `U = N Q[:,:2]`; its orthogonal complement inside `ker(O)` is `W = N Q[:,2:]`. Do not select `r` after seeing Mixed performance. Export full singular values, rank, and near-degeneracy flags; when the top-two basis is not numerically identifiable, label it and refrain from orientation-specific claims. The full 5D silent-space repair remains interpretable separately.

Compare the *natural* reference gap `d` with `V`, `N`, `U`, and `W`. Report raw component norms, proportions of `||d||`, and the familiar-derivative prediction `||G_fam d||`. Do not translate a projected norm into a percent of **CE caused**. Nonlinear future rollouts, not Jacobian norms, determine behavioral claims.

**No “information lost” claim from SVD alone.** The measured quantity is sensitivity of future predictions to a hidden-state change, not an independently validated representation of psychological sequence position.

### 4.4 R2 outputs

- `tables/r2_output_kernel_audit.csv`: seed/rank/singular values/roundoff status.
- `tables/r2_future_sensitivity.csv`: seed/k, familiar-source A/B derivatives, singular values, local ranks, near-degeneracy and finite-difference error.
- `tables/r2_natural_gap_components.csv`: all predefined A/B references, current-hidden gap, visible/silent/future-sensitive components, source identities; **no outcome-based reference selection**.
- `R2_TASK_RELEVANT_STATE_REPORT.md` and `figures/fig_3_gap_components.png`.

## 5. R3 — Can selective repair restore Mixed predictions, and can the reverse change recreate failure?

### 5.1 Why this is the decisive new experiment

Full-state replacement is already known to help in some cases. It cannot tell us *which computation needed fixing*. R3 changes **only predetermined components of the actual Mixed-to-reference state difference**. All weights, sequence tokens and future Mixed cues remain unchanged.

### 5.2 Intervention contract

At primary k=3 (secondary k=5), compute natural post-update `h=h_k^M`, reference `r=h_k^R`, and `d=r−h`. For each original seed and **each** reference A/B, run the following unpatched *future* from the post-k intervention state:

| Arm | State at k after a single intervention | Key question |
|---|---|---|
| Natural | `h` | How badly does the network perform naturally? |
| Full reference | `r` | Is this particular familiar-derived state a useful upper-bound reference under future Mixed? |
| Visible only | `h + a V Vᵀ d` | Does correcting the current-output-visible part help later? |
| Silent only | `h + a N Nᵀ d` | Can future performance improve **without changing current probabilities**? |
| Future-sensitive silent | `h + a U Uᵀ d` | Are the familiar-derived, future-sensitive silent directions selectively helpful? |
| Silent control, complementary | `h + a q_W` | Does an equally large change in less-sensitive silent directions help as much? |
| Silent control, random | `h + a q_rand` | Does an equally large arbitrary silent change help? |

Here `q_W` lies in `span(W)` and `q_rand` lies in `span(N)`, with norms **matched to `||U Uᵀ d||`**. Construct `q_W` by projecting `d` into W and rescaling; if that projection is numerically zero, use a predetermined hash-seeded unit vector in W. Use a predetermined hash-seeded nullspace random vector for `q_rand`, with its seed recorded. They may be generated using frozen geometry, never Mixed labels. Use the same random vectors across repeat source replays.

**Amplitude and feasibility:** Select the **largest jointly feasible scale** `a ≤ 1` based *only* on the post-k tanh-interior box and the three matched U/W/random direction vectors; multiply by 0.95 to avoid strict-boundary roundoff. Use the same `a` in the primary matched comparison, and record realized displacement norms. The full-reference arm uses its exact interior state and is not scaled. Visible/silent full projections receive separately logged feasibility checks and the same geometrically determined scale where valid. The source evaluator must **not clip** hidden coordinates: clipping changes the tested direction. Mark infeasible comparisons instead of substituting a different direction or seed. Zero-norm or rank-degenerate directions produce `not_identifiable`, not zero benefit.

**Primary behavioral contrasts (all specified before inspecting intervention outcomes):**

1. Difference in remaining mean CE: `future-sensitive silent − natural` (negative = rescue).
2. Matched specificity: `(future-sensitive silent − natural) − (complementary silent control − natural)` and analogous random control, with identical physical future.
3. Full-silent, visible, and full-reference effects as separately reported mechanistic bounds.
4. Fraction of all 20 seeds crossing the previously registered future-viability criterion under each intervention, with nonidentifiable/infeasible cells retained as such and denominator explicitly specified.

When output-silent repairs are mathematically valid, verify **all four current probabilities** remain unchanged within predeclared double and float32 tolerances. If present predictions change beyond roundoff, label the repair invalid. A good current prediction is not a substitute for sustained future behavior. Report per-step CE, worst-step CE, and full-sequence costs separately.

**Interpretation:** A U repair more successful than norm-matched W/random controls, without current-output change, would support a specific **future-relevant component** of natural failure. Merely showing full-state transplant works repeats earlier work and is not sufficient.

### 5.3 Reverse perturbation: test whether the candidate component is harmful when introduced

For each originally viable reference branch identified **without selecting the best**:

- Start from its actual source state `r` at k, then inject the **opposite of the rescue**: `r − a U Uᵀ d` (toward the natural Mixed state). Continue under the same Mixed future cues. Compare to matched W/random control displacements from the same reference, with norm and box feasibility matched.
- Record the change relative to the same unperturbed reference. If U repair rescued the natural state and the reverse U displacement specifically damages a viable reference, the bidirectional pattern strengthens causal interpretation.
- Separately apply the fixed U direction (same norm; both signs predeclared) to **natural A and B poststates** and continue under A or B. These tests ask whether the candidate directions are needed for familiar competence; they are *supporting controls*, not proof that human sequence-position information is stored in U.

The reverse comparison is a fixed test, not a search for the sign that produces more errors. Do not select favorable signs, reference branches, k values, or perturbation amplitudes after seeing outcomes.

### 5.4 Negative results and claim gates

- **R1 gives no viable familiar-history reference:** still execute and export all arms, but do not claim the reference is a usable trajectory; target-subspace success remains an empirical question. Report the count of viable references separately.
- **Only full-state transplant rescues:** the useful change is distributed or the predefined subspaces do not isolate it. This fails the selective-mechanism claim.
- **Visible repair helps, silent repair does not:** the error is at least partly tied to current-output-visible coordinates; do not force a “hidden information inaccessible to readout” explanation.
- **Silent repair helps but matched controls help equally:** supports nonspecific hidden-state perturbation rather than the proposed future-sensitive directions.
- **Selective repair helps but reverse does not damage:** evidence for sufficiency of a corrective intervention, not a demonstrated necessary mechanism.
- **Repair helps only in a selected oracle-positive subgroup:** report exactly which subgroup and denominator, without generalizing to all 20 seeds.
- **Familiar A/B performance is harmed by the same modification:** interpret the repair as a trade-off, not context-invariant skill preservation.

**Primary claim gate:** Report a selective mechanism only if the prespecified mean paired advantage of U over **both** matched silent controls is supported by 95% seed-cluster intervals excluding zero in the favorable direction, with transparent coverage/feasibility and no current-softmax violation. The reverse direction is a separate required piece for a **bidirectional causal mechanism** claim. This is a stringent reporting gate, not permission to search more dimensions if it fails. Because multiple probe/position contrasts are secondary, label unadjusted confidence intervals descriptive; primary claim concerns Mixed at k3.

**R3 deliverables:** `R3_SELECTIVE_REPAIR_REPORT.md`, `tables/r3_interventions.csv`, `tables/r3_seed_contrasts.csv`, `tables/r3_reverse.csv`, `figures/fig_4_selective_repair.png`, `figures/fig_5_reverse_controls.png`.

## 6. R4 — Learning explanation, **design only in v1**

The previous context-switch curriculum (D3) did **not** improve Mixed reliably and substantially worsened familiar A/B performance. Do not rerun it or increase training budgets to obtain a positive result.

At completion of R3, write `R4_LEARNING_FOLLOWUP_DECISION.md`:

- If R3 identifies a selective disruptive component, propose one minimal, familiar-only learning constraint that directly preserves that component through A/B cue changes. The proposed constraint must be derived from R3, not from optimizing Mixed probe CE. State its equation, how the subspace is frozen/updated, and a matched control objective.
- Include preregistered checks for familiar A/B competence, retained raw and output-relevant A/B separation, and held-out Mixed/NoContext/Novel transfer. Keep the same original full loss and policy components unless the *new* objective is explicitly declared as an additional intervention. Mixed and Novel cues must never enter training or model selection.
- If R3 fails to isolate a component, recommend **no new training experiment**. Describe the failure mode and the smallest theory-revising diagnostic instead.

**No R4 scientific weights may be updated in this PRD.** A mechanism-specific training study needs a separate fixed protocol after inspecting R3. Do not silently branch into another regularization sweep.

## 7. Analysis and inference contract

- Primary descriptive unit: seed-specific means over predeclared A and B references; retain per-reference observations and *paired within-seed* contrasts. If neither reference is estimable, that seed is missing for continuous paired CE and remains in the denominator for prespecified discovery/failure-status tables. Never substitute an oracle-optimized reference.
- `n=20` independent trained High seeds. Use 10,000 whole-seed paired bootstrap draws with RNG seed **20260925** (consistent with predecessor studies). Do not resample 3 × 3 cells, individual hidden directions, steps, or Novel variants as independent replicates.
- Primary R1 contrasts: k3, physical Mixed, A/B-history source versus native M, same current cue, future CE and current CE reported separately. R3 primary: k3 Mixed, U repair versus two same-norm silent controls. k5 and novel probe families are secondary.
- Verify all declared source logits/probabilities under float32 source; exact vector and projection identities in float64. Separate numerical identity from actual CE effect.
- Report signed effects, 95% seed intervals, individual seed dots, viable counts, all adverse cases, and denominator coverage. Avoid interpreting a CI including zero as evidence of no effect. No p-values needed for claims beyond the registered intervals.
- Separate `original_loss`, `familiar_reference_loss`, and `probe_loss`. Natural A/B and Mixed curves must never be conflated with externally patched states.
- No generative-autonomy, infinite-horizon stability, human-brain mechanism, or unique-latent-information claim. The measured task is seven teacher-forced next-token transitions.

## 8. Test-driven implementation: RED → GREEN → REFACTOR

**Make tests before each stage's implementation.** For every test family, preserve the failing RED log, the minimally passing GREEN log, and a REFACTOR regression check. A test that merely imports a missing module is not enough: include an independent *behavioral negative fixture* before the stage is considered complete.

### R0 required tests

1. Wrong checkpoint family/seed/update rejected; checksum change rejected.
2. Wrong cue registry/third-channel activation rejected.
3. Off-by-one token/target indexing rejected.
4. Pulse rows with non-Mixed future, or persistent takeover, rejected from the R1 reuse pool.
5. Old-row reuse is bitwise/source-tolerance consistent for native state/logit/probability; incompatible rows labeled `recompute_required`.
6. Reused natural A/B/M matches saved original source within predeclared tolerance, without optimizer calls.

### R1 required tests

7. All 9 factorial cells exist for each of 20 seeds at k3 and k5; no duplicated/mislabeled branches.
8. Holding prestate fixed truly changes only current cue; holding cue fixed truly changes only prestate.
9. All futures have the same Mixed cue, token schedule and scoring window; future differs only because post-k state differs.
10. `(M,M)` is identical to natural Mixed; `(A,A)` and `(B,B)` at k match familiar *current* predictions.
11. At k1, all inherited-state labels share zero prestate (negative fixture against false history effect).
12. Immediate loss and future loss are computed from different position sets; mistaken inclusion of k in future fails.
13. Source criterion equality at the threshold is handled exactly as imported (strict-boundary fixture).
14. Bootstrap unit is seed, not branch × position.

### R2 required tests

15. Full SVD recovers 3D current-output row space and 5D nullspace for each verified checkpoint; synthetic rank-deficient case handled without guessing dimensions.
16. `VᵀN=0`, `VVᵀ+NNᵀ=I`, `ON=0`, and gap decomposition pass strict numeric tolerance.
17. Nullspace perturbations preserve *complete* current softmax, with float32 roundoff separately bounded.
18. Future Jacobian dimensions/chain rule match actual source recurrence and autodiff; predeclared finite-difference scales agree on nonsaturated fixtures.
19. `G_fam` uses familiar-only states/cues and correct token inputs but **no future labels or Mixed CE**. Attempted label-driven basis selection fails a guard test.
20. Top-two `U` and complementary `W` are orthonormal and contained in the current-output kernel; near-degenerate/zero rank yields explicit status.
21. A/B references are both retained; selecting minimum probe CE as the primary reference fails a negative fixture.

### R3 required tests

22. No model parameters change; each patched state has the same token/future cue continuation and original source scoring path.
23. Silent/U/W/random arms preserve all current probabilities within declared tolerance; visible arm need not.
24. U/W/random primary perturbations have matched realized Euclidean norms and same geometry-only amplitude, or `not_comparable` status.
25. No hidden clipping; infeasible projected changes recorded; impossible state should not be silently repaired by clipping.
26. A synthetic model with only an output-visible error shows **no** false positive output-silent mechanism.
27. A synthetic model with a delayed output-nullspace effect shows current-softmax equality but changed future CE.
28. A synthetic model with nonspecific random improvement does **not** pass the selective-U reporting gate.
29. Reverse perturbation sign and starting reference are tested independently; reverse tests cannot choose the more harmful sign from outcomes.
30. Familiar A/B perturbations keep A/B future cues; Mixed reverse tests keep Mixed future cues.
31. Seed-specific missing/degenerate cases are never dropped from coverage summaries or filled with zero CE.
32. No outcome-driven change to subspace rank, amplitudes, references, windows, or optimizer (none allowed).

### R4/reporting required tests

33. No scientific training occurs; R4 produces a proposal and an evidence-linked `go/no-go` recommendation only.
34. Claim ledger rejects “natural retrieval,” “lost sequence information,” “global unreachability,” “human mechanism,” or “attractor” labels unsupported by experiments.
35. Exact per-seed differences recompute from source tables, and all CI drawing units are seeds.
36. All figures regenerate from saved tables without rerunning experiments or changing hashes.
37. Attempting to overwrite an old sealed predecessor program fails, and missing/extra scientific bytes fail final manifest verification.

Pass the complete existing regression suites for P2/P4/Phase 1K/D1/S3 where available. If dependencies cannot load, record a clear blocker rather than rewriting inherited scientific code to make tests pass.

## 9. Execution stages and immutable outputs

| Stage | Action | New scientific evaluation allowed? | Gate |
|---|---|---|---|
| **R0** | Discover old data, verify source, produce reuse map | Only source-replay fidelity checks | Verified canonical inputs, full required cell registry |
| **R1** | Reuse/recompute 3 × 3 state × cue, immediate/future tests | Only missing frozen pulse cases | All 20 seeds, source-consistent controls |
| **R2** | Compute output row/null geometry and label-free future sensitivities | Frozen algebra/autodiff only | Projection and source-derivative tests pass |
| **R3** | Targeted repair and matched reverse perturbation | Frozen hidden-state patch rollouts | Prespecified contrasts, complete adverse/null/feasibility records |
| **R4** | Mechanism-specific learning decision memo | **No training** | Report locked conclusions; no post-hoc search |
| **R5** | Final synthesis, figures, independent teach-back, hash seal | No scientific model calls required | Complete verified report and reproducible tables |

Suggested new namespace: `experiments/context_failure_selective_repair/`, `tests/context_failure_selective_repair/`, `results/context_failure_selective_repair/v1/`. These are **requested output names**, not assertions that those paths currently exist. Use clean adapters to the original source rather than copying or changing the source recurrence.

All stage output directories and their manifests are immutable when complete. If correcting a scientific implementation error, preserve original logs, register an explicit version change, rerun only affected stages in a **new** directory, and label old claims superseded without deleting them.

## 10. Final deliverables

1. `FINAL_CONTEXT_FAILURE_SELECTIVE_REPAIR_REPORT.md` — one-paragraph answer, methods, seed-level tables, exact intervention interpretation, negative findings, what remains unknown.
2. `CLAIM_LEDGER.md` — each claim, source table/hash, evidence, counterexample, alternative explanation, verdict.
3. `R0_SOURCE_AND_REUSE_MAP.md`, `R1_STATE_X_CUE_REPORT.md`, `R2_TASK_RELEVANT_STATE_REPORT.md`, `R3_SELECTIVE_REPAIR_REPORT.md`, and `R4_LEARNING_FOLLOWUP_DECISION.md`.
4. At least five figures: 3 × 3 state/cue map, immediate-vs-future/lag figure, silent-versus-visible natural-gap geometry, selective repair versus matched controls, reverse perturbation results. Source-level seed dots and coverage counts; no color-only legend required for comprehension.
5. Source tables: one row per seed × k × prestate × cue, derivative basis audit, patch outcome, reverse outcome, per-transition CE, bootstrap summary; both raw A/B reference sides retained.
6. `RESEARCHER_HANDOFF.md` — exact recurrence, subspace definitions, patch formulas, parameters, code paths, what was held fixed, numerical tests, limits, a 90-second teach-back and two adverse cases.
7. Append-only `scratchpad.md`, RED/GREEN test logs, checksums for original source/checkpoints and new outputs, a read-only regeneration command, final independent verification command.

### Required conclusion format (fill with evidence; never pre-fill the answer)

> **What failed?** Under the physical Mixed cue, the network [specific observed R1 pattern].
>
> **What was disrupted?** The natural Mixed state changed [task-relevant state component, if identified].
>
> **What caused the deficit?** Selectively correcting [component] changed future CE by [effect, seed CI] relative to [same-norm controls], with [current-output preservation status]. Reversing it [did/did not] recreate the failure.
>
> **What can be learned from this?** [Minimal follow-up that targets this actual mechanism, or no justified learning intervention yet].

If R3 fails its selective gates, the final answer must say: **“We localized effects of state history and context on future performance, but did not identify a unique disruption whose correction explains the natural Mixed deficit.”** This is a valid scientific outcome.

---

## 11. Agent instructions

Read this PRD and the supplied scratchpad, then locate and audit the actual source files. Work in stage order. Use test-driven development (RED → GREEN → REFACTOR), and stop at the explicit gates. Make the fewest new model calls necessary: reuse P4/Phase 1K rollouts wherever exact branch definitions allow. Never choose favorable directions, references, or intervention amplitudes based on Mixed losses. Do not train a new network in v1. Deliver the full reports, numeric tables, figures, claim ledger, hash-indexed evidence, and researcher teach-back in the same run.
