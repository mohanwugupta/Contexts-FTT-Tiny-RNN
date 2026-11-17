# PRD — When Can Context Cues Access a Viable Recurrent Computation?

**Project:** Mixed & Third FTT — frozen SRN mechanistic program  
**Version:** 1.0 · 12 November 2026  
**Audience:** coding agent and researcher  
**Status:** prospective extension of the completed P0–P4 output-compatible-trajectory program  
**Development:** test-driven development (**RED → GREEN → REFACTOR**), append-only scratchpad, scientific stop rules  
**Scope:** frozen RNNs; context-cue interventions and **finite-horizon control sensitivity only**. **Do not implement fixed-point searches, attractor analysis, Lyapunov exponents, eigenspectrum surveys of autonomous dynamics, or full dynamical-systems modeling.**

---

## 0. Executive brief — read before coding

### The observation

The same eight-item sequence is learned in contexts A and B. Some High-scale SRNs develop separated internal trajectories and make poor predictions when presented with Mixed, absent, or extrapolated context cues. The completed P3 experiment showed that, in *some* seeds and probes, a hidden state away from the A/B midpoint can support a viable unpatched future when a midpoint state cannot. P4 showed that some ordinary cues also yield viable futures, but the tested fixed cue menu did not recover the strict output-equivalent oracle state.

### The next question

> **Can the network reach useful recurrent computations through its context input, and does success require a single cue or a short history of cues?**

We want the minimum experiment that adjudicates this *within the existing frozen networks*. First map the effect of one cue. Then compare one versus two preparatory cue updates **at exactly the same probe position**, holding the physical target cue and all later inputs fixed. Finally, use local **control-theory sensitivities** to explain any observed benefit of cue history. No new model training.

### Three candidate explanations

1. **Wrong cue:** an alternative single cue reaches a viable continuation; the ordinary target cue does not.
2. **History-dependent access:** two contextual updates reach viable behavior that no tested one-step update achieves under the registered comparison.
3. **Input-access limit:** a direct hidden-state oracle succeeds, but no bounded tested cue schedule does. This is *finite-search evidence*, never proof of global impossibility.

These mechanisms may differ by seed and probe. Do not force a single label on every network.

### Minimal deliverable

- **Figure 1:** a two-dimensional **single-pulse cue map** of mean remaining-sequence CE for the frozen networks, showing ordinary target cues, discovered viable cue regions and input-boundary labels.
- **Figure 2:** **same-final-probe**, matched-budget comparison of natural baseline, one preparatory cue, and repeated two-step preparation (same two cue parameters). Show success rates and paired CE; independently parameterized two-step preparation is secondary.
- **Figure 3 (only if informative):** compact finite-horizon **control-sensitivity** analysis explaining why cue history changes which hidden-state directions can be influenced.
- **One-page claim ledger:** each proposed inference, exact evidence, alternative explanation and uncertainty.

---

## 1. Source-of-truth audit and nonnegotiable contracts

**Inspect actual repository code and files before writing analysis code.** Relevant prior results:

- `FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md`
- `P0_SOURCE_AUDIT.md`, `P1_RESULTS.md`, `P2_RESULTS.md`, `P3_RESULTS.md`, `P4_RESULTS.md`
- `RESEARCHER_HANDOFF.md`
- `FINAL_DYNAMICAL_GENERALIZATION_REPORT.md`, `E2_RESULTS.md`
- the existing canonical network, cue registry, state-extraction, source replay, and test packages.

Expected, to be verified (do not hardcode unverified keys):

- eight-unit `tanh` Elman SRN, four next-token logits, four one-hot token inputs, three cue channels with the third input channel effectively inactive;
- fixed sequence `(0, 1, 2, 0, 3, 1, 2, 3)` with **seven teacher-forced predictions**;
- canonical A `(1,0,0)`, B `(0,1,0)`, Mixed `(.5,.5,0)`, NoContext `(0,0,0)`, and the exact frozen Novel30/Novel60 A- and B-side cue vectors **imported from code**;
- the existing recurrent update `h_t = tanh(W_h h_(t-1) + W_x x_t + W_c c_t + b_h)` and next-token CE metric, including true state and token indices.

**Primary cohort:** the same 20 canonical High-scale frozen checkpoints used in P3/P4. This includes the historical monotonic head if present, but **only next-token prediction** is measured here. Do not mix with C0–C3 sequence-only checkpoints. Other cohorts are out of scope for primary inference; report them separately only if time remains *after* completing the registered primary program.

**Primary probes:** Mixed, NoContext, Novel30_A, Novel30_B, Novel60_A, Novel60_B. Report individual Novel sides before within-seed means. Never combine the older Novel1–4 cue definition with angular Novel30/60 probes.

**Freeze:** model weights, output heads, sequence tokens, training checkpoints, exact cue-encoding function, original P3 viability criterion and pilot labels. Hash checkpoint and relevant source/config bytes before and after each stage. No silent re-training or reinitialization; no hidden-state patch in this program. All new outputs go under `results/context_cue_access/v1/` with no overwrite of old reports.

### The earlier P3 evidence is a reference, not a training target

P3 found viable alternative trajectories under *future-label-informed hidden-state optimization*. Use its accepted state IDs and source manifests to identify cases where an oracle path exists. **Never** use a P3 oracle hidden state as the objective of cue optimization in the primary experiments. We optimize the remaining-sequence performance of the *cue-driven* rollout itself. Distance to an oracle state can be explored afterward as a descriptive diagnostic.

**Observed prior scope to preserve:** P3 `k=1` far+viable+midpoint-failed examples were rare (Mixed 1/20; NoContext 4/20; Novel aggregates smaller). P4's tested pulse routes sometimes produced viable futures; at `k=1`, all source labels share the same zero prestate. We are asking about **access under a broader, controlled input family**, not claiming no physical cue has ever worked.

---

## 2. Exact mathematical and temporal definitions

### State and future evaluation

Use the canonical one-step recurrent function:

\[
 h_t = f_\theta(h_{t-1},x_t,c_t)
 =\tanh(W_h h_{t-1}+W_xx_t+W_cc_t+b_h),
\]

\[
 p_t = \operatorname{softmax}(W_o h_t+b_o).
\]

For an intervention completed at scored position `k`, the primary outcome is **mean future CE over the subsequent scored positions only**:

\[
 L_{>k}(h_k;c_{k+1:7})
 =\frac{1}{7-k}\sum_{t=k+1}^{7}-\log p_t(y_t), \qquad k<7.
\]

- At `k=1`: six future predictions.
- At `k=3`: four future predictions.
- At `k=5`: two future predictions.
- The **current** prediction at `k` is measured separately, but excluded from future viability. This prevents a cue intervention from looking successful only because it fixes the immediately patched/current position.
- **Future sequence tokens are teacher forced** and held identical across paired comparisons. This is not autonomous token generation.
- For all cue interventions, record current full softmax, future per-step CE, mean future CE, CE over the full sequence, and future cue schedule.

**Viability:** import and reuse *the exact viability function, reference familiar CE and tolerance (`+0.10` nats) used by the completed P3 program*. During P0, print the code location, the threshold formula and whether the reference is position-specific. If the old function cannot be faithfully reconstructed, stop with `blocked_metric_contract` instead of quietly defining a new threshold. A stricter zero-slack and worst-step diagnostic can be reported as *secondary*, never used to replace the primary viability result.

### Cue parameterization and input domains

Write every candidate context as `c=(u,v,0)`. The third channel is fixed at zero, preserving the trained network's effective context input subspace.

Registered domains:

- **Familiar line segment:** `u >= 0`, `v >= 0`, `u+v=1` (convex A/B mixtures).
- **Familiar square:** `u,v in [0,1]` (contains Mixed and NoContext).
- **Extended mathematical input box:** `u,v in [-1.5,1.5]`, includes signed extrapolations. These are valid *numeric model inputs*, not necessarily physically or psychologically natural cues.

Primary grid: **13×13 = 169 values**, each coordinate drawn from `[-1.5,-1.25,...,1.25,1.5]`. Always append all exact registered cue vectors as additional named anchors even if a grid point is numerically close. Do not round or silently replace exact radical Novel cues. Retain normalized cue norm, A/B contrast `alpha=u-v`, sum coordinate `beta=u+v`, and familiar-domain membership as independent columns.

### Paired unit

All inference and bootstrap intervals treat **trained seed** as the independent unit. Cue grid points, cue schedules, model positions and optimization restarts are repeated measurements. Report full-cohort rates alongside prospectively defined oracle-positive / native-failing subsets; never rely solely on hand-picked favorable seeds.

---

## 3. Execution plan and scientific gates

| Stage | Purpose | Training? | Decision |
|---|---|---|---|
| **S0** | Source, indexing, checkpoint and viability audit | No | Pass before computing outcomes |
| **S1** | Single-cue input map and bounded one-cue oracle | No (cue only) | Determine whether different single cues enable viable continuation |
| **S2** | Same-final-cue one- versus two-preparation-step comparison | No (cue schedule only) | Test whether cue history adds access without changing the probe |
| **S3** | Local finite-horizon controllability of context inputs | No | Explain the intervention result; no standalone causal claim |
| **S4** | Results, figures, falsification and researcher handoff | No | Finish with supported, unsupported and unresolved claims |

Technical test failure stops the affected stage. A negative scientific result **does not** permit search expansion or new model training. If S1 finds strong single-cue access, **still execute S2 at the fixed budget** to compare single and history routes; do not only run S2 for unsuccessful seeds, which biases the comparison.

---

## 4. S0 — Source audit, indexing and frozen replay

1. Locate canonical frozen checkpoints and exact previously saved P3/P4 checkpoints, cue tables and viability implementation.
2. Check hashes and prove no model parameter requires gradient updates in a cue oracle run.
3. Reproduce native all-target-cue futures for the registered positions `k=1`, `k=3` and `k=5` from the original source recurrence. Compare with P4 native branches and P3 reference where the estimands coincide.
4. Test two history anchors:
   - For the S2 `k=3` comparison, start from the **same** `h_0` for every arm (generally zero).
   - For S2 `k=5`, start from the same `h_2` obtained by a single prespecified natural-target-cue replay; **do not** choose this starting state based on subsequent CE.
5. Record precisely which scored positions use which sequence input token and correct next-token target. Create a one-page indexing diagram and verify that future CE excludes all preparatory and final-cue steps.
6. Reproduce the complete current softmax and next-token logits in native branches; exact cue-replay identity should hold to source numerical tolerance.

**Mandatory outputs:** `S0_SOURCE_AUDIT.md`, `manifests/s0.json`, `tables/native_replay.csv`, `tables/input_registry.csv`, `tables/index_contract.csv`.

---

## 5. S1 — Does any single context cue produce viable continuation?

### Scientific question

> Starting from the **same natural prestate**, can changing just the current cue move the frozen network onto a useful future trajectory?

### Manipulation

For seed `s`, target probe `q`, and placement position `k`, take the frozen natural prestate `h_(k-1)^q` from a target-probe rollout (at `k=1`, this is zero). For each candidate `c_k=(u,v,0)`:

1. Hold prestate `h_(k-1)^q` and token `x_k` fixed.
2. Apply the single candidate context cue at position `k`.
3. For `t>k`, restore the **registered target probe cue** `c_q` at every step.
4. Run the unchanged recurrent network on the same remaining tokens.
5. Evaluate `L_>k` and the primary P3 viability gate. Record the full present softmax, but do **not** require equality with the midpoint's softmax; that requirement was specific to the P2/P3 nullspace intervention.

Primary position `k=1` (six future predictions); `k=3` and `k=5` are prespecified secondary positions. Probe conditions: all six registered cues. Native target cue must be an explicit baseline row. The earlier finite P4 cue menu must be a labeled subset for historical reproduction.

### S1a. Fixed finite map (primary observational intervention)

Evaluate all 169 grid points plus exact cue anchors. Publish the surface for each seed (machine-readable) and across-seed summaries. Predeclare no smoothing or extrapolation when assigning viability. Report success separately inside the familiar line, familiar square, and extended signed box. A successful input from the signed box is **mathematical context-pathway capacity**, not necessarily a plausible real-world stimulus.

### S1b. Bounded cue oracle (explicitly future-target informed)

Optimize the **two cue values** `u,v` only; no hidden-state parameter and no network-weight parameter may change. Use future CE as the oracle objective and report the search and best-of-grid separately.

**Fixed budget:** 6 starts × 80 optimizer updates per seed/probe/position; Adam with learning rate `0.04`, input values parameterized into the `[-1.5,1.5]` box (e.g., smooth `tanh` coordinates). Initialize starts deterministically at A, B, Mixed, zero, one registered target, and the best fixed-grid candidate (a declared two-stage oracle). Replay each selected cue using the exact original frozen source recurrence. Do not restart search or increase the budget following a null result. An infeasible, saturated or non-finite run is kept as a recorded failure.

**Report both:** (a) fixed grid discovery and (b) oracle discovery. They are not natural retrieval. The test is a causal input intervention on a frozen network; *selection* of the cue uses future answers.

### Primary contrasts

- Native target cue vs **best prespecified fixed-grid cue**, on same prestate and same future cue schedule.
- Native target cue vs **bounded oracle cue**, separately labeled target-informed.
- Viability counts/rates for full 20-seed cohort and for the previously declared `P3-oracle-positive AND native-failing` subgroup; preserve denominators and inability to adjudicate when prior P3 reference is unavailable for that k.
- Within familiar line, within familiar square and across entire extended box, reported separately.
- Cue-coordinate effects `alpha`, `beta` and norm are descriptive, not latent psychological categories.

### Interpretation gate

If a different cue succeeds while the target cue fails, **the input pathway can generate viable behavior with a different signal**. This does **not** show the network naturally identifies or selects that cue. If no tested cue succeeds, conclude only that the **registered bounded single-cue search failed**, not that the network is globally unreachable from its context input.

---

## 6. S2 — Is a short cue history more effective than a single cue?

### Scientific question

> With the *same final physical target cue* and the *same future sequence*, does one additional preparatory context update enable viable behavior that a single preparatory update does not?

This is a different estimand from S1. **Do not combine S1 pulse rates with S2 preparatory-cue rates.**

### Crucial matched-time design

For `k=3` (primary) or `k=5` (secondary), start every arm from the *same* prestate `h_(k-3)` (defined in S0). Process **exactly the same three sequence tokens** `x_(k-2),x_(k-1),x_k`. Manipulate only the first two context cues. At the final step `k`, every arm receives the **same physical target probe cue** `c_q`. All later cues `c_(k+1:7)` are also `c_q`. Then measure only `L_>k`.

| Arm | Cue at `k−2` | Cue at `k−1` | Cue at `k` | Free cue parameters |
|---|---|---|---|---:|
| **N0: Natural** | target `q` | target `q` | target `q` | 0 |
| **E: Early-only** | variable `u` | target `q` | target `q` | 2 |
| **L: Late-only** | target `q` | variable `u` | target `q` | 2 |
| **R: Repeated two-step** | **same** variable `u` | **same** `u` | target `q` | **2** |
| **I: Independent two-step** | variable `u_1` | variable `u_2` | target `q` | 4 |

`E`, `L`, and `R` each have **two free cue scalars**. This makes `R` vs `L` the cleanest test of temporal accumulation *at a matched parameter count*. `I` is a capacity ceiling with more free parameters; its superiority alone cannot show a benefit of temporal integration independently of parameter count. `E` tests whether earlier cue exposure is sufficient by itself.

**Do not allow a comparison where the two-step arm simply has more post-intervention future predictions or a different final cue.** All conditions always process the same number of tokens before scoring the same future positions.

### Search and budget controls

For each arm, use the same frozen input box `[-1.5,1.5]` on both active cue values. The optimizer may update the **cue values only**, with gradients through—but not updates to—the frozen SRN.

- S2 fixed-candidate checks: natural target, A, B, Mixed, zero, and the prespecified 169-grid cue candidates for each two-parameter arm. For independent `I`, use a deterministic, preregistered **512-pair Sobol candidate list** plus nested control candidates; do **not** enumerate 169² combinations.
- S2 oracle search: `6 deterministic starts × 80 Adam updates` per arm per seed/probe/k, LR `0.04`, same smooth bounded transform as S1. Starting values include natural, familiar A/B, Mixed, zero and fixed-grid/Sobol best. Keep selection logs, all failures and actual source replays.
- Count **the same number of objective evaluations/backprop steps** per arm and disclose that `I` has more parameters. Do not use more starts or adapt the budget for the two-step arm if it fails.
- For the fixed candidate comparison, include all nested baseline/equivalent sequences as a safeguard: `I`'s candidate registry must include `(target,u)` and `(u,u)`. Theoretical set inclusion alone is **not** evidence of temporal access; behavioral differences must be reported using realized, fixed-budget searches and the equal-parameter `R` vs `L` contrast.

**Primary:** paired `R` versus `L` future CE and viability (both two free parameters), with `N0` as reference. **Secondary:** `E` vs `L`, `I` vs `R`, full cue trajectory and per-step CE. Analyze `k=3` separately from `k=5`, since their remaining horizons differ.

### Outcome patterns and permitted claims

| Finding | Interpretation | What it does *not* prove |
|---|---|---|
| An alternative current single pulse in S1 is viable | Different current cue can access useful behavior | That ordinary target cue retrieves it |
| `R` succeeds while `L` does not under the frozen budget | Repeated preparation can produce an advantage with the **same number of free parameters** | No untested single cue could ever work |
| Only independent `I` succeeds | Extra independently controlled steps can help within this search | Whether the advantage comes from history or more free parameters |
| Some target contexts succeed after prep, others do not | Accessibility depends on cue context and inherited state | One universal cognitive retrieval rule |
| P3 hidden-state oracle works, all registered cue arms fail | Finite bounded input-access gap | That no possible cue history can access a viable trajectory |

The final physical context is held fixed by design, making a viable `R`/`I` result stronger evidence for *prepared-state access* than merely optimizing the current probe cue.

---

## 7. S3 — Minimal control-theory analysis (not full dynamical systems)

### Scientific question

> Does an additional context-input step increase the hidden-state directions the cue pathway can influence, and do those directions align with improvements in future performance?

This is a **local finite-horizon input sensitivity** calculation. It is not a claim about global reachability, attractors or autonomous stability.

### Exact derivatives

For each state update along the same **baseline** or declared successful experimental cue schedule:

\[
 D_t=\mathrm{diag}(1-h_t^2),\qquad
 B_t=\frac{\partial h_t}{\partial (u_t,v_t)}=D_t W_{c,\mathrm{active}}\in\mathbb R^{8\times2},
\]

\[
 J_t=\frac{\partial h_t}{\partial h_{t-1}}=D_tW_h\in\mathbb R^{8\times8}.
\]

With the final physical target cue at `k` fixed, the two preceding context steps influence `h_k` through:

\[
 R_{\mathrm{late}}=J_k B_{k-1}\in\mathbb R^{8\times2},
\]

\[
 R_{\mathrm{early}}=J_kJ_{k-1}B_{k-2}\in\mathbb R^{8\times2},
\]

\[
 R_{\mathrm{two}}=\left[\,R_{\mathrm{early}}\;R_{\mathrm{late}}\,\right]\in\mathbb R^{8\times4}.
\]

At one step, input sensitivity has rank at most two. Across two independent steps the **linearized** rank has an upper bound of four, but can be lower because of saturation and redundant directions. For repeated-cue `R`, both steps share the same `u`, so its sensitivity is **`R_early + R_late` (8×2)**, not the full 8×4 independent-step matrix. This is essential to avoid overstating controllability.

### Required metrics (small)

1. Singular values and effective rank of `R_late`, `R_early+R_late` (repeated schedule) and `R_two` (independent schedule). Predeclare rank using both an absolute threshold `1e-8` and a relative threshold `1e-4 × largest singular value`; record raw singular values, not only ranks.
2. A simple output-relevant control direction: gradient of the **fixed future CE** with respect to `h_k` computed with frozen weights. Measure the fraction of this gradient norm lying in the column span of the relevant sensitivity matrix. Define zero-gradient cases explicitly as undefined. **This diagnostic uses future labels** and is not a natural selection mechanism.
3. Finite-difference checks on actual source recurrence for each input coordinate and both prep positions, at step sizes `1e-3` and `1e-4`. Compare to analytic derivatives, and flag saturation/boundary cases where the linear approximation is poor.
4. For successful/failed paired schedules, plot *one* primary control figure: future CE improvement versus singular-value support or gradient alignment. Compare **within seed** and **do not** interpret association as causal mediation.

**Stop rule:** If S2 shows no substantive history advantage, still compute a concise sensitivity audit on the registered baseline and report it as a negative/diagnostic result. Do not expand into attractors, global controllability proofs, architecture searches or task retraining.

---

## 8. TDD — RED → GREEN → REFACTOR, mandatory and auditable

### Development protocol

For *every* stage:

1. **RED:** write the smallest test that captures a scientific invariant or failure mode. Run it; save failing output to `tests/red_green_log.jsonl` and scratchpad. A test that accidentally passes before implementation does **not** count as RED; write a stronger test that demonstrably fails because the required behavior is missing.
2. **GREEN:** implement the minimum function needed to pass, rerun the focused test, then the stage test suite. Save command, exit code, and change summary.
3. **REFACTOR:** consolidate code without changing outcomes; rerun the focused tests plus golden replay fixtures. Record any numerical changes.
4. Only after all required tests are green may the stage produce scientific result tables. Stop on test failures; never edit tests or tolerances simply because results are unfavorable.

### Test registry

**S0 — provenance and indexing**

- **T01:** missing canonical checkpoint produces `blocked_missing_source`, never trained fallback.
- **T02:** checkpoint and source hashes are identical before and after evaluation.
- **T03:** only two active cue coordinates affect the canonical recurrence; the third remains fixed at zero.
- **T04:** all seven token/target/hidden-state indices reproduce the source native rollout.
- **T05:** native cue replay matches P4 where conditions and positions coincide.
- **T06:** score excludes current or preparatory steps and includes exactly `7−k` future predictions.
- **T07:** imported P3 viability function matches frozen acceptance/rejection fixtures, including strict boundary tests.
- **T08:** no hidden state, output head or recurrent weight is trainable under the cue optimizer.

**S1 — one cue**

- **T09:** 13×13 grid has 169 unique coordinate pairs in the frozen range; exact anchors retain identity.
- **T10:** cue vectors are three-dimensional with final coordinate exactly zero; all bounds and domain labels correct.
- **T11:** identical prestates, tokens and future target cues across all candidate cue comparisons.
- **T12:** changing `c_k` alone changes no other input or model parameter.
- **T13:** native target cue reproduces the untouched native continuation.
- **T14:** current prediction is excluded from primary future CE and recorded separately.
- **T15:** selected oracle cue respects bounds and updates only cue parameters; no source state or weights change.
- **T16:** source-float32 replay, not differentiable surrogate loss, determines viability and printed outcomes.
- **T17:** missing/infeasible optimization starts remain explicit and do not trigger added starts.

**S2 — history and fair comparisons**

- **T18:** `N0`, `E`, `L`, `R`, `I` share bit-identical initial `h_(k-3)`, token sequence, final `c_k` and all later cues.
- **T19:** `E`,`L`,`R` each expose **exactly two free cue scalars**; `I` exposes four.
- **T20:** `R` ties both preparatory cues to the same parameters; `I` has independent inputs.
- **T21:** future scoring windows and endpoint k are identical for every arm; k=3 and k=5 not pooled as equal horizons.
- **T22:** `N0` exactly reproduces the native physical target-cue rollout.
- **T23:** `I` fixed menu includes every prespecified `(target,u)` and `(u,u)` nesting-control pair.
- **T24:** all oracle arms receive identical starts/updates/evaluation budget, with a documented parameter-count exception for I.
- **T25:** deliberately shuffled prep-cue schedules change only the declared context inputs, never sequence token identity.
- **T26:** E/L/R/I optimize only cues, respect the numerical domain, and replay through the untouched frozen recurrence.

**S3 — control theory**

- **T27:** analytic `B_t` agrees with autodiff and two finite-difference step sizes on synthetic nonsaturated fixtures.
- **T28:** analytic `J_t` agrees with finite differences on independent hidden-state perturbations.
- **T29:** products `J_k B_(k−1)` and `J_k J_(k−1) B_(k−2)` match finite differences of final state with respect to individual preparatory cues.
- **T30:** the tied two-step derivative equals `R_early + R_late`, **not** the 8×4 independent-input matrix.
- **T31:** synthetic duplicated control directions yield deficient rank despite two-step history; no automatic rank-growth claim.
- **T32:** a null future-CE gradient returns undefined alignment, not a fabricated zero or one.

**S4 — analysis and integrity**

- **T33:** bootstraps resample seeds as clusters, not cue points, restarts or transitions.
- **T34:** Novel direction means are built within seed and do not count as independent network observations.
- **T35:** report includes all seeds, no-success cases, infeasible starts and exploratory domain partitions.
- **T36:** oracle-selected cues and physically presented native cues are labeled separately everywhere.
- **T37:** no success claim is inferred from an optimizing surrogate unless source recurrence confirms it.
- **T38:** deterministic rerun re-creates key tables/plots from sealed manifests without mutation.

Add tests for unexpected source variants uncovered in S0. **Do not delete or weaken the registered tests to obtain GREEN.**

---

## 9. Analysis, uncertainty and decision logic

### Primary outcomes

- Seed-level binary **any viable cue** from S1 finite grid; separately oracle-assisted S1.
- Paired **mean future CE** and viability from S2 `R−L` at k=3; k=5 secondary.
- Full-cohort and `P3 oracle-positive & native-failing` subgroup denominators. The subgroup is a hypothesis-driven diagnostic, not the only analysis.
- Source-fidelity and finite-horizon sensitivity audits. Use bootstrap intervals over 20 seeds, report counts as counts, and do not present seed×cue as independent sample size.

### Decision table

| S1 result | S2 result | Conclusion allowed | Next issue |
|---|---|---|---|
| Grid or oracle single cue succeeds where native fails | Any | **Alternative cue has input-pathway access capacity** for that model/context/horizon | Why ordinary cue does not select it |
| No found S1 rescue | Repeated `R` beats late-only `L` and crosses viability in paired cases | **Cue history enables access under tested budgets with matched cue-parameter count** | Local sensitivity and nonlinear control limits |
| S1 only extended-box cue succeeds | Any | **Mathematical extrapolative input capacity** | Whether natural cue space supports it |
| No S1 rescue | Only independent `I` succeeds | Extra cue control can help; **parameter count is confounded** | Need matched-capacity follow-up before broad claim |
| P3 viable oracle exists; tested cue families all fail | None | **Finite bounded cue-access gap** | Cannot infer global unreachability |
| S1 and S2 both fail or effects inconsistent | None | No decisive access mechanism found under registered protocol | Reconsider viable-set definition or input control, not training by default |

### Interpreting success

- Do **not** claim an optimized context cue was naturally selected. It was found using correct future answers.
- Do **not** require any cue-generated state to match midpoint/current-oracle softmax. Those equality constraints answered a different earlier question.
- Do **not** interpret increased rank as proving a beneficial history effect. The rank is a local *capability measure*; behavior is assessed by actual frozen nonlinear rollouts.
- Do **not** call the eight-unit SRN a two-dimensional dynamical system. Only the effective context input has two dimensions.
- Do **not** overstate a finite failure as proof of inaccessible states, whether under a bounded grid or nonconvex oracle search.

---

## 10. Figures and reports (lab-meeting ready)

### Figure A — The single-cue accessibility map

- Per-target 2D context grid, axes `alpha=u−v`, `beta=u+v`; color = mean **future** CE.
- Mark A, B, physical target cue and any grid-discovered viable cue cells. Label extrapolative candidates. Plot raw input coordinates in the machine-readable source table.
- One concise per-probe bar panel: `native`, `best within familiar line`, `best within square`, `best extended-grid`, `bounded input oracle` (do not show these as five naturally generated policies).
- If aggregate heatmaps obscure seed heterogeneity, include one *predeclared* seed exemplar with clear selection criterion (e.g. lowest numeric seed exhibiting the specified S1 rescue) and cohort-wide rates.

### Figure B — Is one extra cue update useful?

- S2 k=3 matched comparisons: full-cohort dot/line plot of future CE for `N0`, `E`, `L`, `R`, `I`; group arms visually by **two-parameter** vs **four-parameter** search.
- Viability counts (n/20) for each cue family and arm. Paired `R−L` with seed-bootstrap 95% interval. Show k=5 separately or in appendix.
- Essential small schematic: `same h0 → prep cue 1 → prep cue 2 → fixed target cue → identical future inputs`.

### Figure C — Local control mechanism (one modest figure)

- If S2 history benefit: singular values/effective ranks of `R_late`, `R_early+R_late` and `R_two`, and output-relevant gradient alignment. Tie to observed within-seed improvement without causal mediation language.
- If no S2 benefit: keep figure in supplement and report the null/ambiguous conclusion directly.

### Required outputs

```
results/context_cue_access/v1/
├── S0_SOURCE_AUDIT.md
├── S1_SINGLE_CUE_RESULTS.md
├── S2_CUE_HISTORY_RESULTS.md
├── S3_LOCAL_CONTROL_RESULTS.md
├── FINAL_CONTEXT_CUE_ACCESS_REPORT.md
├── RESEARCHER_HANDOFF.md
├── scratchpad.md
├── tables/
│   ├── input_registry.csv
│   ├── native_replay.csv
│   ├── s1_grid_transition.csv
│   ├── s1_grid_seed.csv
│   ├── s1_oracle_attempts.csv
│   ├── s1_best_cues.csv
│   ├── s2_candidate_schedules.csv
│   ├── s2_oracle_attempts.csv
│   ├── s2_seed_summary.csv
│   ├── s2_paired_effects.csv
│   ├── s3_sensitivities.csv
│   ├── s3_finite_difference_audit.csv
│   └── claim_source_map.csv
├── figures/
│   ├── fig_A_single_cue_accessibility.{png,pdf}
│   ├── fig_B_cue_history.{png,pdf}
│   └── fig_C_local_control.{png,pdf}
├── audit/
│   ├── test_evidence.jsonl
│   ├── run_registry.csv
│   ├── source_replay.json
│   └── final_test_result.json
└── manifests/
    ├── s0.json
    ├── s1.json
    ├── s2.json
    ├── s3.json
    └── sealed_manifest.json
```

Every figure must be regenerable from exported tables, not manually constructed. Report uncertainty at seed level and record every failed run. If a table is intentionally empty, include a reason/status field; no fake zeros.

### Researcher handoff — required sections

1. **Question and hypothesis:** two sentences.
2. **Algorithm:** recurrent source step, cue intervention schedule, loss, parameterization, search budgets and control sensitivities.
3. **What changed / what stayed fixed:** cues changed; tokens, weights and eventual target cue/future inputs fixed as specified.
4. **Evidence:** primary full-cohort comparison, one meaningful worked example and a negative example; exact values and test results.
5. **Assumptions:** teacher forcing, bounded numeric cues, optimization informed by future labels, short horizon, canonical High-only sample.
6. **Researcher teach-back:** 60–90 second explanation of what was tested and why its outcome answers—or fails to answer—the accessibility question.
7. **Decisive uncertainty:** best alternative explanation and smallest permitted next experiment.

---

## 11. Scratchpad protocol

Create `results/context_cue_access/v1/scratchpad.md` **before substantive implementation** from `CONTEXT_CUE_ACCESS_SCRATCHPAD_TEMPLATE.md`. Keep it append-only, with permanent sections for:

- source/checkpoint manifest and exact indexing;
- scientific claim in plain language, one-sentence falsification criterion;
- stage gate/status;
- **RED** test command, observed failing assertion and file;
- **GREEN** implementation, passing command, test count;
- **REFACTOR** changes and replay equality;
- fixed budget/domain decisions before outcome inspection;
- attempted runs, nonfinite/infeasible cases and source hashes;
- discrepancies and deviations (do not silently repair experimental design);
- interpretation ledger: what result proves, what it does not and alternative explanation;
- teach-back updates and questions for lab discussion.

A task is not complete until the scratchpad records the tests and actual evidence that support it.

---

## 12. Stop rules and prohibited substitutions

1. **No SRN training, new architectures, checkpoint replacement or hidden-state patching.** Optimization of context *inputs* is allowed and clearly tagged as oracle selection.
2. **No ad hoc expansion of `[-1.5,1.5]`, grid density, restarts, steps or learning rate** based on whether a preferred result appears.
3. **No collapsing current-cue and preparatory-history effects into one estimand.** Preserve which cue steps changed and which future cues remained fixed.
4. **No post hoc selection of only oracle-positive seeds** as the primary analysis; retain all 20 with clear subgroup denominators.
5. **No declaring input controllability from the number of cue coordinates alone.** Calculate actual sensitivities and their finite differences.
6. **No attraction/fixed-point/long-run stability analyses.** The relevant system is a short forced sequence.
7. **No claim that future-informed oracle cues are naturally retrieved** or biologically plausible contextual stimuli.
8. **No fabricated comparison to missing P3/P4 artifacts.** Record the missing file or mismatch, then use source replay where fully specified.
9. **No overwriting the prior P0–P4 or E1–E4 freezes.** Any correction is a separately versioned output with a reason and hash.
10. **No promises of a desired result.** Null, negative and blocked findings are final outcomes when tests pass.

---

## 13. Implementation order — coding-agent checklist

- [ ] Initialize output root and scratchpad; freeze source and checkpoint hashes.
- [ ] S0 write RED tests, audit model/cues/time indexing/viability, implement GREEN replay, REFACTOR.
- [ ] S1 write RED tests, compute fixed grid, add bounded cue oracle, run seed-group summaries, seal outputs.
- [ ] S2 write RED tests, implement matched final-target comparison and equal-parameter R/L arms, then independent arm, freeze budget, evaluate all seeds.
- [ ] S3 write RED derivative tests, implement minimal finite-horizon sensitivity matrices, compare to actual nonlinear intervention results.
- [ ] S4 assemble figures/claim ledger, run all tests, verify checksums, create final reports and researcher teach-back.

### First prompt for the coding agent

> Implement this PRD in order, with strict RED → GREEN → REFACTOR development and an append-only scratchpad. Start by reading `FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md`, `P0_SOURCE_AUDIT.md`, `P3_RESULTS.md`, `P4_RESULTS.md`, and the actual frozen model/cue code. Verify the P3 viability function and input/state indices before any analysis. Never update model weights. Run S1 on the frozen 20 High-scale seeds, then S2 with the **same physical final target cue and future tokens for all arms**, then S3 local finite-horizon control sensitivities. Optimize cue values only and label future-target-informed cue searches as oracles. Preserve null and blocked results. Finish with `FINAL_CONTEXT_CUE_ACCESS_REPORT.md`, three reproducible figures, a researcher handoff, hashed manifests and a complete test log.
