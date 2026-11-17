# PRD — Phase 1E: Relinearized Manifold-Following Constrained Retrofit

## 1. Objective

Determine whether the low Phase 1D retrofit success rate arises because the desired compositional solution lies along a **curved endpoint-preserving parameter manifold** that cannot be traversed reliably using the Phase 1D constrained optimizer.

Implement an adaptive relinearized constrained optimizer that:

- repeatedly recomputes the local constraint geometry;
- moves toward the previously defined correct compositional path;
- keeps familiar A/B behavior fixed;
- uses no Mixed, Novel, or NoContextProbe outcomes for optimization or hyperparameter selection;
- evaluates the exact same held-out success gate used in Phase 1C/1D.

The critical comparison is:

\[
\text{Phase 1C soft penalty}
\quad\text{vs}\quad
\text{Phase 1D constrained}
\quad\text{vs}\quad
\text{Phase 1E manifold-following}.
\]

Do **not** alter the model architecture or add a new learning signal.

---

## 2. Primary scientific question

Phase 1D showed:

\[
\text{acute rescue} \gg 0
\]

and

\[
Q_{\mathrm{local}} \text{ is high},
\]

but

\[
P(\text{successful constrained retrofit}) = 3/20.
\]

There are at least two interpretations.

### H1 — optimization/geometric bottleneck

A good endpoint-preserving direction exists locally, but the relevant tangent space rotates as parameters move.

A first-order direction at \(\theta_0\) therefore becomes invalid at:

\[
\theta_0+\Delta\theta.
\]

Repeated local relinearization should allow the optimizer to follow the curved solution manifold.

### H2 — genuine global accessibility bottleneck

The acute state manipulation is effective, and an initial local direction exists, but there is no sufficiently long endpoint-preserving parameter trajectory connecting the trained network to the desired computation.

Then even a properly relinearized optimizer should fail in many seeds.

**Phase 1E discriminates H1 from H2.**

---

## 3. Hard scientific firewall

These restrictions are mandatory.

### Allowed during optimization / model selection

Use only:

- familiar A/B training behavior;
- registered non-midpoint compositional training cues from Phase 1D;
- endpoint preservation constraints;
- correct-path residual on those training cues;
- numerical stability;
- trust-region diagnostics.

### Forbidden

Never use:

- held-out \(\lambda=.5\) Mixed performance;
- Mixed probe outcomes;
- Novel probe outcomes;
- NoContextProbe outcomes;
- human behavior;
- success/failure on held-out probes;
- final success rate.

for:

- learning-rate selection;
- trust-region tuning;
- stopping;
- checkpoint selection;
- damping selection;
- restart selection.

Probe conditions remain **read-only**.

Add runtime assertions that fail if any probe-labelled row enters an optimization or model-selection tensor.

---

## 4. Phase 0 — audit the Phase 1D optimizer first

Before implementing anything, determine exactly what Phase 1D currently does.

Produce:

`PHASE1E_OPTIMIZER_AUDIT.md`

For every optimization step, document whether Phase 1D recomputes:

- endpoint constraint Jacobian;
- task-path Jacobian;
- tangent/null space;
- damping;
- trust-region scale;
- equality correction.

Specifically answer:

> Are \(J_C(\theta)\) and \(J_P(\theta)\) recomputed after every accepted parameter update?

If **no**, Phase 1E's main manipulation is full relinearization.

If **yes**, do not simply reproduce Phase 1D. Implement the adaptive manifold-following procedure below, including acceptance/rejection, adaptive trust region, nonlinear constraint restoration, and recomputation after every accepted step.

This audit is a required deliverable.

---

## 5. Mathematical formulation

Let the trainable retrofit parameters be:

\[
\theta.
\]

The original trained checkpoint is:

\[
\theta_0.
\]

The original output/readout remains **bitwise frozen**, as in Phase 1D.

### 5.1 Familiar endpoint constraint

Construct the exact same familiar endpoint representation used in Phase 1D.

Define:

\[
F(\theta)
\]

as the vector containing all registered familiar A/B quantities that must remain invariant.

Store:

\[
F_0 = F(\theta_0).
\]

Define constraint residual:

\[
c(\theta)=D_F^{-1}[F(\theta)-F_0],
\]

where \(D_F\) is the same source-relative scaling used previously.

The feasible manifold is approximately:

\[
\mathcal M=\{\theta:c(\theta)=0\}.
\]

Do not weaken the existing Phase 1C/1D endpoint gates.

---

## 6. Correct compositional path objective

Use the **identical correct-path target from Phase 1D**.

For registered training cue values:

\[
\lambda\in\Lambda_{\text{train}},
\]

with held-out \(\lambda=.5\) excluded, calculate:

\[
r(\theta)=D_P^{-1}[P(\theta)-P^*].
\]

Objective:

\[
L_{\text{path}}(\theta)=\frac12 \|r(\theta)\|_2^2.
\]

No Mixed probe loss appears in this objective.

No Novel loss appears in this objective.

---

## 7. Phase 1E optimizer

Use a **relinearized constrained trust-region optimizer**.

This should behave like a compact SQP/manifold-following algorithm rather than introducing another penalty objective.

At accepted optimization step \(t\):

\[
\theta_t.
\]

Recompute from scratch:

\[
c_t=c(\theta_t)
\]

\[
r_t=r(\theta_t)
\]

\[
J_{C,t}=\frac{\partial c}{\partial\theta}
\]

and

\[
J_{P,t}=\frac{\partial r}{\partial\theta}.
\]

Do not reuse a stale Jacobian after an accepted update.

---

## 8. Split each update into normal + tangent movement

This gives the experiment a clean geometric interpretation.

### 8.1 Normal correction

First calculate the minimum-norm movement toward the familiar constraint manifold:

\[
\Delta\theta_N=-J_C^\top(J_CJ_C^\top+\gamma I)^{-1}c.
\]

This component exists only to restore familiar endpoint behavior.

### 8.2 Tangent projector

Calculate the local projector onto directions that, to first order, leave the familiar endpoints unchanged:

\[
\Pi_T=I-J_C^\top(J_CJ_C^\top+\gamma I)^{-1}J_C.
\]

Then the locally allowable task Jacobian is:

\[
\widetilde J_P=J_P\Pi_T.
\]

### 8.3 Tangential compositional step

Solve:

\[
\Delta\theta_T=\arg\min_{\Delta\theta}
\left\|r+J_P\Pi_T\Delta\theta\right\|^2
+\eta\|\Delta\theta\|^2
\]

subject to:

\[
\|\Delta\theta\|_{\text{scaled}}\leq\rho_t.
\]

Then:

\[
\Delta\theta=\Delta\theta_N+\Pi_T\Delta\theta_T.
\]

The distinction matters:

- \(\Delta\theta_N\): restores the familiar solution;
- \(\Delta\theta_T\): moves along the currently estimated feasible manifold toward compositionality.

---

## 9. Nonlinear feasibility restoration

The linear tangent condition

\[
J_C\Delta\theta=0
\]

does **not** guarantee:

\[
c(\theta+\Delta\theta)=0.
\]

After proposing a step:

\[
\theta'=\theta+\Delta\theta,
\]

evaluate the **actual nonlinear** endpoint residual.

If violation exceeds the allowed tolerance, perform one or more restoration steps:

\[
\theta'\leftarrow
\theta'-J_C(\theta')^\top[J_C(\theta')J_C(\theta')^\top+\gamma I]^{-1}c(\theta').
\]

Recompute the Jacobian after each restoration iteration.

Cap the number of restoration iterations.

If feasibility cannot be restored, reject the step.

---

## 10. Adaptive trust region

This is probably the most important addition.

The optimizer should automatically determine how far the current local linear approximation can be trusted.

For each proposed step calculate predicted path improvement:

\[
\Delta L_{\text{pred}}
\]

from the linearized model and actual improvement:

\[
\Delta L_{\text{actual}}=L_{\text{path}}(\theta_t)-L_{\text{path}}(\theta').
\]

Define:

\[
R_t=\frac{\Delta L_{\text{actual}}}{\Delta L_{\text{pred}}+\epsilon}.
\]

Use fixed preregistered rules such as:

- \(R_t<0.25\): reject and shrink \(\rho\);
- \(0.25\leq R_t<0.75\): accept if feasible, leave \(\rho\) unchanged;
- \(R_t\geq0.75\): accept and optionally enlarge \(\rho\).

Exact thresholds may be conventional defaults, but they must be fixed **before seeing held-out Mixed performance**.

Also reject any step that breaks familiar guardrails.

---

## 11. Recompute geometry after every accepted step

This is non-negotiable.

After:

\[
\theta_t\rightarrow\theta_{t+1},
\]

recalculate:

\[
J_C(\theta_{t+1}),
\]

\[
J_P(\theta_{t+1}),
\]

\[
\Pi_T(\theta_{t+1}),
\]

and all residuals.

The experiment is specifically testing whether **following the changing tangent space** improves realizability.

---

## 12. Checkpoint selection

Checkpoint selection must use only:

1. training-cue correct-path residual;
2. familiar endpoint feasibility;
3. familiar raw loss;
4. A/B separation;
5. numerical stability.

Among feasible checkpoints choose the one with the lowest registered training-path residual.

Do not select the checkpoint with the best Mixed loss.

---

## 13. Hyperparameter selection

Keep this very small.

Primary parameters:

- initial trust radius \(\rho_0\);
- damping \(\gamma\);
- tangent regularization \(\eta\).

Prefer deterministic/adaptive rules over a large grid.

If a grid is unavoidable, select globally across seeds using only training/familiar diagnostics.

Absolutely no per-seed tuning using probe results.

---

## 14. Baselines

Every figure/table should include:

- **Baseline High**
- **Baseline Low**, where relevant
- **Phase 1C soft penalty**
- **Phase 1D constrained correct path**
- **Phase 1E manifold-following correct path**

For specificity, additionally run:

- **Phase 1E matched wrong path**

using exactly the same optimizer and compute budget.

---

## 15. Correct-vs-wrong control

Retain the Phase 1D orthogonal, norm-matched target construction.

For each seed:

\[
P_{\text{wrong}}\perp P_{\text{correct}}
\]

under the registered metric before nonlinear optimization.

Run the exact same Phase 1E procedure toward both targets.

Primary specificity statistic:

\[
\Delta L_{\text{Mixed}}=L_{\text{wrong}}-L_{\text{correct}}.
\]

Report:

- mean;
- median;
- bootstrap CI;
- fraction \(>0\).

Do not require exact equality of final achieved displacement magnitudes if nonlinear constraints prevent it. Report achieved displacement explicitly.

---

## 16. Required raw-loss time-series diagnostics

This changes the plotting standard from the previous normalized diagnostics.

The **main time-series figure should show the actual model loss.**

Do not make normalized loss ratio the primary visualization.

### Figure A — Raw familiar loss over every sequence

For every sequence in the complete familiar schedule, plot:

\[
L_s=-\frac{1}{T_s}\sum_t\log p_\theta(y_{s,t}\mid y_{s,<t},c_s)
\]

using the exact sequence loss already used by the model.

x-axis: **sequence index**

y-axis: **raw sequence loss**

Plot separate trajectories for:

- original Baseline High;
- Phase 1C;
- Phase 1D;
- Phase 1E correct;
- Phase 1E wrong.

Shade A and B blocks as before.

### Important

Do **not** divide by baseline.

Do **not** normalize first value to one.

Do **not** convert to a residual.

The reader should be able to see directly:

> “The original model lost X nats/cross-entropy here, and the retrofitted model lost Y.”

Keep normalized plots only as supplementary diagnostics if useful.

---

## 17. Required probe-vs-familiar loss figure

We need to see where Mixed and Novel sit relative to the already learned Context A and Context B solutions.

At every registered probe checkpoint, perform **read-only frozen evaluation** of:

\[
L_A,\quad L_B,\quad L_{\text{Mixed}},\quad L_{\text{Novel}}
\]

at the **same checkpoint**.

If NoContextProbe exists, also evaluate:

\[
L_{\text{NoContext}}.
\]

Do not infer A/B reference values from distant blocks when direct frozen evaluation at the same checkpoint is possible.

### Figure B — Probe losses relative to A/B

For each probe occurrence or probe phase, plot raw loss for:

- Context A;
- Context B;
- Mixed;
- Novel;
- optionally NoContextProbe.

Use the same y-axis in raw model-loss units.

This directly shows whether Mixed/Novel are:

- A-like;
- B-like;
- intermediate;
- worse than both;
- better than both.

---

## 18. Full sequence timeline including probes

Also produce one manuscript-style figure containing the **entire sequence schedule**.

For each actual sequence:

- A training sequences: raw A loss;
- B training sequences: raw B loss;
- Mixed probe sequences: raw Mixed loss;
- Novel probe sequences: raw Novel loss;
- NoContext probe sequences if present.

Do not connect a line across fundamentally different probe types unless visually clear.

Recommended:

- familiar A/B = continuous lines;
- Mixed = distinct point marker;
- Novel = another marker;
- NoContext = another marker.

The x position must reflect the **actual chronological sequence index**.

This plot should answer at a glance:

> What loss did the model have on A and B, and where did the unseen probe conditions land relative to them when they occurred?

---

## 19. Matched-checkpoint reference plot

Because chronological A/B observations can differ from the exact probe checkpoint, also generate a stricter diagnostic.

For each probe checkpoint \(j\), compute frozen:

\[
L_A^{(j)},L_B^{(j)},L_M^{(j)},L_N^{(j)}.
\]

Make a connected/raw-loss plot or faceted plot by probe checkpoint.

Do this separately for:

- baseline;
- Phase 1D;
- Phase 1E.

This eliminates time/fatigue/order ambiguity.

---

## 20. Probe reference derived quantities

These can accompany the raw plots but **cannot replace them**.

For Mixed:

\[
q_M=\frac{L_M-L_A}{L_B-L_A}
\]

only when \(L_A\neq L_B\).

Interpret descriptively:

- \(q=0\): same raw loss as A;
- \(q=1\): same raw loss as B;
- \(0<q<1\): between familiar losses.

Do the analogous measure for Novel.

But report raw \(L_A,L_B,L_M,L_N\) first.

---

## 21. Required optimizer diagnostics

Produce per-seed trajectories for:

### 21.1 Raw training-path objective

\[
L_{\text{path}}(\theta_t)
\]

### 21.2 Actual familiar raw loss

Not ratio.

### 21.3 Constraint violation

\[
\|c(\theta_t)\|.
\]

### 21.4 Trust radius

\[
\rho_t.
\]

### 21.5 Predicted-vs-actual improvement ratio

\[
R_t.
\]

### 21.6 Step decomposition

Plot:

\[
\|\Delta\theta_N\|
\]

and

\[
\|\Delta\theta_T\|.
\]

### 21.7 Tangent-space rotation

Between accepted steps measure similarity of consecutive feasible subspaces, e.g. principal angles between:

\[
\mathcal T_t
\]

and

\[
\mathcal T_{t+1}.
\]

At minimum report:

\[
\|\Pi_{T,t+1}-\Pi_{T,t}\|_F.
\]

If this is substantial in seeds that Phase 1D fails but Phase 1E rescues, that is strong mechanistic evidence for the curvature account.

---

## 22. Primary outcome

Use the exact existing complete Phase 1C-comparable success gate.

Report:

\[
\frac{\#\text{successful seeds}}{20}
\]

for:

- Phase 1C;
- Phase 1D;
- Phase 1E correct;
- Phase 1E wrong.

Do not redefine success after seeing results.

---

## 23. More informative continuous outcome

The binary success gate should not be the only output.

For every seed calculate:

\[
\Delta L_i=L_{\text{baseline},i}^{Mixed}-L_{\text{Phase1E},i}^{Mixed}.
\]

Show a paired plot from Baseline High to Phase 1E for all 20 seeds.

Also report:

- mean;
- median;
- bootstrap 95% CI;
- fraction improved;
- effect versus Phase 1D.

This distinguishes “almost passed” from genuine failures.

---

## 24. Seed-level failure decomposition

For every seed assign exactly one terminal category:

1. **No local access**
2. **Optimization stagnation**
3. **Constraint-limited**
4. **Numerical failure**
5. **Improved but below complete gate**
6. **Complete success**

Classification rules must be computed mechanically.

Especially distinguish:

\[
\text{path objective does not fall}
\]

from

\[
\text{path objective falls but familiar constraints prevent further movement}.
\]

---

## 25. Critical comparison: Phase 1D versus 1E

Produce a scatter:

\[
x=L_{\text{Mixed}}^{1D}
\]

\[
y=L_{\text{Mixed}}^{1E}.
\]

Identity line:

\[
y=x.
\]

Also produce:

\[
x=Q_{\text{local}}
\]

\[
y=L_{\text{Mixed}}^{1D}-L_{\text{Mixed}}^{1E}.
\]

And:

\[
x=\text{tangent-space rotation}
\]

\[
y=L_{\text{Mixed}}^{1D}-L_{\text{Mixed}}^{1E}.
\]

The latter is the most direct test of the curvature hypothesis.

---

## 26. Decision rule

### Outcome A — Phase 1E substantially raises success

For example, success becomes common rather than exceptional and correct-path improvement remains specific relative to the wrong path.

Interpretation:

> The compositional solution was broadly parameter-accessible, but reaching it required following a curved endpoint-preserving manifold. Phase 1D underestimated learnability because its optimization procedure was inadequate.

This gives a strong sufficiency/accessibility story.

### Outcome B — continuous improvement rises but strict success remains modest

Interpretation:

> Manifold curvature accounts for part, but not all, of the acquisition bottleneck.

### Outcome C — Phase 1E remains around 3–4/20

Interpretation:

> The compositional state is causally sufficient and locally accessible, but local accessibility generally does not extend into a finite endpoint-preserving path.

That is scientifically informative and argues against treating acute causal sufficiency as equivalent to learnability.

---

## 27. RED → GREEN tests

Before running 20 seeds, write failing tests.

### RED 1 — probe leakage

Optimizer intentionally receives a Mixed/Novel row.

Expected:

`FAIL: PROBE_DATA_ENTERED_OPTIMIZER`

### RED 2 — held-out \(\lambda=.5\)

Attempt to include \(.5\) in training cue set.

Expected failure.

### RED 3 — stale Jacobian

Mock two accepted steps while reusing the same constraint Jacobian.

Expected failure:

`FAIL: JACOBIAN_NOT_RELINEARIZED`

### RED 4 — frozen readout

Mutate output/readout weights.

Expected failure.

### RED 5 — nonlinear endpoint violation

Accept a step whose linear constraint is satisfied but true nonlinear endpoint gate fails.

Expected rejection.

### RED 6 — raw-loss plotting

Feed known losses:

- A = 0.8
- B = 1.5
- Mixed = 1.1
- Novel = 2.0

Assert plotted/exported values remain exactly:

`0.8, 1.5, 1.1, 2.0`

No ratio or normalization transformation is allowed in primary raw-loss figures.

### RED 7 — chronological probe placement

Provide synthetic schedule:

`A A B B Mixed A B Novel`

Assert Mixed and Novel appear at their true sequence positions.

### RED 8 — matched checkpoint evaluation

Assert A/B/Mixed/Novel reference losses are generated from the **same model checkpoint** for each probe comparison.

Then make all tests GREEN before running the real experiment.

---

## 28. Required outputs

At minimum:

```text
PHASE1E_MANIFOLD_FOLLOWING_REPORT.md
PHASE1E_seed_metrics.csv
PHASE1E_optimizer_trace.csv
PHASE1E_probe_raw_losses.csv
PHASE1E_familiar_raw_losses.csv
PHASE1E_manifest.json
```

Figures:

```text
PHASE1E_FIG01_success_rate.png
PHASE1E_FIG02_1D_vs_1E_mixed_loss.png
PHASE1E_FIG03_correct_vs_wrong.png
PHASE1E_FIG04_failure_modes.png
PHASE1E_FIG05_curvature_vs_rescue.png

PHASE1E_TS01_raw_familiar_loss_all_sequences.png
PHASE1E_TS02_raw_full_timeline_with_probes.png
PHASE1E_TS03_probe_loss_vs_A_B.png
PHASE1E_TS04_matched_checkpoint_A_B_Mixed_Novel.png
PHASE1E_TS05_optimizer_path_loss.png
PHASE1E_TS06_constraint_violation.png
PHASE1E_TS07_trust_radius.png
PHASE1E_TS08_tangent_space_rotation.png
```

**TS02 and TS03 are mandatory report figures**, not supplemental debugging figures. They answer the central descriptive question:

> What is the model actually losing on Mixed and Novel compared with the two familiar contexts?

---

## 29. Report language

The code agent should not automatically conclude “optimization solved the problem.”

The report should mechanically choose among:

> **Curvature-supported:** Relinearization materially improves finite retrofit while familiar endpoints remain preserved.

> **Partial curvature contribution:** Relinearization improves the continuous outcome but complete success remains uncommon.

> **Global accessibility bottleneck:** Acute and local sufficiency remain strong, but relinearized manifold following does not make finite retrofit broadly successful.

The report must separately state:

- acute state-level sufficiency;
- local parameter accessibility;
- finite learnability;
- endpoint preservation;
- correct-vs-wrong specificity.

That separation is the conceptual payoff.

---

## 30. Scope constraint

Do not let the new optimizer become another large ML optimization project.

Implement:

- one principled manifold-following algorithm;
- one matched wrong-path control;
- the same 20 seeds;
- the same registered gates;
- the raw-loss plots specified above.

The purpose is to determine whether the 3/20 Phase 1D result reflects an optimization artifact or a real property of the trained networks.
