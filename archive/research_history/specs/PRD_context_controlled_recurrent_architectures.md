# PRD: Context-Controlled Recurrent Architectures for Split + Ambiguous Retrieval
## GRU Control vs Low-Dimensional Context State vs Soft Context Gating

## 0. Objective

Test whether the missing ingredient is **architectural factorization between shared task dynamics and context control**.

The computational target is:

- learn distinct context-specific states when evidence supports separation;
- preserve a shared sequence computation;
- reuse/recombine those states when contextual evidence is ambiguous.

The central hypothesis is:

**slow shared recurrent dynamics + fast context inference/control -> strong A/B splitting + graded ambiguous retrieval**

Compare three architectures under the same synthetic latent-cause ecology:

1. **A: GRU-150**
2. **B: Elman-128 + 8D context GRU**
3. **C: Elman-128 + 8D context GRU + soft multiplicative gating**

No model may be trained on actual human Mixed/Novel probe outcomes. Do not test H256/H512 in this PRD.

---

## 1. Primary scientific question

Can a recurrent system learn:

- clear A evidence -> express A-specific skill state;
- clear B evidence -> express B-specific skill state;
- ambiguous evidence -> uncertainty-sensitive reuse of familiar structure;

while preserving both strong A/B differentiation and a reusable shared sequence backbone?

The primary manipulation is architecture, not model size.

---

## 2. Architecture A: GRU-150

Use a single-layer GRU:

- hidden size = 150;
- same task inputs as the current model;
- same next-token prediction target;
- no explicit context label;
- no RT loss in synthetic training.

Hypothesis: ordinary GRU gating is sufficient for latent-context inference and context-dependent retrieval.

Human transfer later uses the same monotonic confidence-to-RT mechanism already validated in the project.

---

## 3. Architecture B: Elman-128 + 8D context state

Use a shared Elman task core plus a low-dimensional context controller.

Context controller:

`c_t = GRU_ctx(u_t, c_{t-1})`

with:

- context-state dimension = 8;
- controller input `u_t` includes observable context cues/layout plus recent task history;
- no latent A/B label is provided.

Task core:

`h_t = tanh(W_x x_t + W_h h_{t-1} + U_c c_t + b_h)`

with:

- task hidden dimension = 128;
- context state additive to the shared recurrent computation.

Hypothesis: separating fast context inference from shared task dynamics is sufficient.

---

## 4. Architecture C: Elman-128 + 8D context state + soft gate

Use the same controller as Model B.

Controller:

`c_t = GRU_ctx(u_t, c_{t-1})`

Soft gate:

`g_t = sigmoid(W_g c_t + b_g)`

where `g_t` has dimension 128.

Primary recurrent implementation:

`h_t = tanh(W_x x_t + W_h (g_t * h_{t-1}) + U_c c_t + b_h)`

where `*` is element-wise multiplication.

Hypothesis: contextual inference must **multiplicatively configure** shared recurrent dynamics.

Optional secondary implementation, only after the primary experiment, is FiLM-style gain/bias modulation. Do not choose between gating forms using human Mixed/Novel results.

---

## 5. Fixed task components

Across all architectures preserve:

- sequence length = 8;
- response alphabet = 4;
- same next-token prediction objective;
- same downstream human target sequence;
- same human-derived target geometry;
- same A10/B10 target schedule;
- same monotonic execution/RT mapping at human transfer;
- same frozen-probe analysis pipeline.

The intended manipulation is recurrent/context-control architecture.

---

## 6. Synthetic latent-cause ecology

Each synthetic world instantiates the same abstract problem.

### 6.1 Latent causes

Each world has two latent contextual causes:

`C in {A, B}`

These are anonymous synthetic causes, not the human task's actual contexts.

### 6.2 Shared skill backbone

Each world contains one shared 8-item ordinal sequence.

Represent task structure as:

`theta_A = theta_shared + Delta_A`

`theta_B = theta_shared + Delta_B`

The shared component should dominate overall sequence identity.

Context-specific modifiers should alter only a subset of task/transition statistics.

Recommended sparse structure:

- shared-heavy transitions: P1, P4, P7;
- context-sensitive transitions: P2/P3 and P5/P6.

If the existing generator has a cleaner sparse-modifier implementation, reuse it.

---

## 7. Observable context evidence

The model never receives the latent cause label directly.

Generate observable context:

`x_ctx ~ p(x | C)`

from:

- cue vector;
- layout realization;
- recent task/history information.

Each world independently samples:

- cue means;
- cue covariance;
- layout transforms;
- cue reliability;
- task modifiers;
- sequence identity.

This prevents memorizing fixed surface-to-task mappings.

---

## 8. Context evidence regimes

Every world includes:

### R1: clear A
Observation strongly favors latent cause A.

### R2: clear B
Observation strongly favors latent cause B.

### R3: ambiguous
Observation lies in the overlap of the A/B cue/layout distributions.

Every training sequence still has a true latent cause A or B.

**There is no explicit Mixed class in synthetic training.**

The statistically useful computation is latent-cause inference from uncertain evidence, not learning a midpoint label.

---

## 9. Synthetic world stream

Primary world schedule:

- 10 clear-A sequences;
- 10 clear-B sequences;
- 10 alternating clear A/B sequences;
- 6 ambiguous-evidence sequences.

Total = 36 sequences/world.

Ambiguous sequences:

- 3 have true latent cause A;
- 3 have true latent cause B.

Their observable evidence is uncertain, but their task realization follows the true latent cause.

This creates pressure to integrate cue evidence and recent history instead of averaging outputs.

---

## 10. Training objective

Primary synthetic loss:

`L = L_next-token`

Do not include:

- RT loss;
- context-label loss;
- gate loss;
- hidden-distance loss;
- midpoint loss;
- corridor loss;
- q target;
- human profile loss.

The controller/gates must become useful because they improve sequence prediction.

---

## 11. Optional supervised-context diagnostic

Only if all primary models fail to infer context.

Secondary control:

`L = L_next-token + lambda_ctx * L_context`

with a fixed small `lambda_ctx`.

This tests whether the bottleneck is:

- discovering latent context;
- versus using an available context representation.

Do not mix this diagnostic into the primary claims, and do not advance it to human transfer without labeling it as supervised.

---

## 12. Matched shuffled control

For each architecture create a paired shuffled condition.

Preserve exactly:

- sequences;
- cue marginals;
- layout marginals;
- latent-cause frequency;
- task-modifier marginals;
- ambiguous-evidence frequency;
- number of updates.

Shuffle the mapping between observable context evidence and latent task regime within world.

This destroys the stable `context evidence -> task regime` relationship while preserving exposure.

Run:

- A_structured;
- A_shuffled;
- B_structured;
- B_shuffled;
- C_structured;
- C_shuffled.

---

## 13. Training scale

### Smoke test
- 2 seeds;
- 100 worlds;
- all six cells;
- engineering only.

### Pilot
- 5 paired seeds;
- 5,000 synthetic worlds/cell.

### Confirmatory
Only passing architectures:
- 20 paired seeds;
- 50,000 worlds/cell.

Do not inspect human Mixed/Novel to select an architecture.

---

## 14. Synthetic validation battery

All validation worlds use unseen seeds and make zero optimizer updates.

### V1: clear-context differentiation

Evaluate clear A and B.

Measure:

- A/B hidden-state distance;
- prediction-profile distance;
- context-state distance for B/C;
- gate-vector distance for C.

Primary split score:

`S_split = D(h_A, h_B)`

Require meaningful differentiation.

### V2: shared sequence preservation

Measure whether A/B still share sequence structure.

Use:

- token-matched hidden similarity;
- shared PCA/low-rank basis;
- Procrustes alignment;
- shared-subspace variance explained.

A successful model cannot solve splitting by learning two unrelated systems.

### V3: ambiguous-evidence robustness

Evaluate ambiguous evidence.

Measure:

- NLL;
- accuracy;
- correct-token confidence.

Structured must outperform matched shuffled while clear-context differentiation remains intact.

### V4: context-belief decoding

Because synthetic latent causes are known, compute oracle:

`P(C=A | x, history)`

Do not train on it.

Decode this posterior from:

- GRU hidden state for A;
- `c_t` for B/C.

Report:

- correlation with oracle posterior;
- Brier score;
- calibration curve.

Diagnostic only.

### V5: switch adaptation

Generate:

`AAAA -> BBBB`

and the reverse.

Measure:

- prediction recovery time;
- context-state transition latency;
- gate transition latency;
- hidden-state transition latency.

### V6: uncertainty continuum

Vary cue reliability over:

`r in {0, .2, .4, .6, .8, 1}`

where 0 is maximally ambiguous and 1 maximally diagnostic.

Measure:

- decoded context belief;
- controller state;
- gate state;
- prediction profile.

Look for graded internal context representation; outputs may be nonlinear.

### V7: conflicting-evidence control

Create cases where current surface evidence leans A but history/task evidence supports B, and vice versa.

Test whether the model uses:

- current surface cue only;
- or integrated context evidence/history.

---

## 15. Two-dimensional synthetic pass frontier

Do not use one scalar as the primary gate.

Plot:

- x = clear-context differentiation;
- y = ambiguous-context transfer quality.

A model only passes if it is high on both.

This explicitly rejects:

- strong split + catastrophic ambiguity;
- good ambiguity + no split.

---

## 16. Hard synthetic gates

G1. Clear A/B prediction is above chance and numerically stable.

G2. Structured has greater A/B differentiation than its shuffled control.

G3. Shared token/sequence structure remains nontrivial.

G4. Structured has lower ambiguous-query NLL than shuffled.

G5. Internal context representation tracks cue reliability/oracle context evidence.

G6. Model reconfigures after latent-context switches.

G7. At least 4/5 pilot seeds show the expected structured > shuffled direction on both:
- differentiation;
- ambiguous-query performance.

Only passing architectures advance to confirmatory.

---

## 17. Architecture-selection rule

After confirmatory:

1. discard any architecture failing a hard gate;
2. prefer the simplest passing architecture;
3. if models are close, prefer:
   - fewer parameters;
   - more interpretable context representation;
   - stronger shared-subspace preservation;
4. advance at most two architectures to human transfer.

Do not use human probe behavior for selection.

---

## 18. Human transfer

Only after synthetic selection.

Use:
- winning structured architecture;
- its matched shuffled control;
- second-best structured architecture if it independently passed;
- original H8 baseline;
- optional random-init control for the winning architecture.

Human target task is locked:

- same sequence;
- G2_shared_x / human geometry;
- A10/B10 alternating blocks;
- 170 external sequences;
- same target optimizer conventions;
- same monotonic confidence-to-RT readout.

No Mixed/Novel training.

---

## 19. Plasticity during human target training

### Primary transfer
Allow ordinary target-task learning in all trainable modules.

### Restricted-plasticity control for Models B/C
Prespecify a secondary condition:

Freeze:
- shared recurrent core.

Train only:
- context controller;
- context-to-core control weights;
- prediction head.

This tests the reusable-core hypothesis directly.

This is a planned mechanistic control, not a post hoc rescue.

---

## 20. Human familiar gates

Before interpreting Mixed/Novel require:

- sequence learning;
- A/B hidden differentiation;
- context-yoked RT;
- Cross/Split or equivalent context-specific phenotype;
- negative confidence-to-RT relation.

For Models B/C also report:
- A/B controller-state separation;
- A/B gate separation for C.

A model that gets better Mixed behavior only by becoming Lump fails.

---

## 21. Human Mixed probe

Primary analysis uses the **first frozen Mixed exposure**.

Collect:
- NLL;
- accuracy;
- confidence;
- normalized RT;
- recurrent hidden state;
- context-controller state;
- gate vector.

Behavioral metrics:
- transition-level between-endpoint fraction;
- RT deficit;
- NLL deficit;
- profile RMSE/correlation/R² to human data, unfitted.

Representational metrics:
- A/B projection coordinate;
- off-corridor distance;
- shared-subspace engagement;
- controller-space position;
- gate overlap with A and B.

Do not require exact midpoint.

---

## 22. Human Novel probe

Analyze separately from Mixed.

Report:
- NLL deficit;
- RT deficit;
- accuracy;
- distance from A/B controller manifold;
- shared-subspace engagement;
- gate entropy/sparsity for C.

Do not force Novel to behave like Mixed.

---

## 23. Key mechanistic predictions

### Model A
Ambiguity appears as graded movement within the GRU hidden state.

### Model B
Expected pattern:
- `c_A` and `c_B` separated;
- `c_Mixed` in an uncertain/intermediate region;
- shared sequence dynamics retained in `h`.

### Model C
Expected pattern:
- `g_A` and `g_B` differ;
- `g_Mixed` overlaps partially with both;
- shared recurrent sequence geometry remains stable.

These are predictions, not training constraints.

---

## 24. Causal interventions

Run only after a model passes human transfer.

### I1: context-state swap
For B/C:
- run A with B controller state;
- run B with A controller state.

Measure prediction, RT, hidden geometry.

### I2: gate swap
For C:
- apply `g_A` during B;
- apply `g_B` during A.

### I3: artificial arithmetic interpolation
Post hoc only:
- `0.5*c_A + 0.5*c_B`;
- or `0.5*g_A + 0.5*g_B`.

Compare with actual Mixed.

Purpose: determine whether emergent Mixed is equivalent to arithmetic averaging.

### I4: controller ablation
Zero/randomize controller state on frozen copies.

If A/B differentiation collapses, controller is causally necessary.

---

## 25. Statistics

Synthetic:

`metric ~ architecture * structure_condition + (1|seed)`

Primary within-architecture contrasts:
- structured vs shuffled for split score;
- structured vs shuffled for ambiguous-query NLL.

Human:

`metric ~ architecture * context * checkpoint + (1|seed)`

Report:
- paired mean differences;
- bootstrap 95% CIs;
- standardized effects;
- seed-direction counts.

Do not select conclusions from p-values alone.

---

## 26. TDD / RED -> GREEN requirements

T1. Latent cause label is never directly passed into the primary model.

T2. No synthetic training row has an explicit `Mixed` class.

T3. Ambiguous evidence is produced by overlapping A/B observation distributions while retaining a true latent cause.

T4. Structured/shuffled worlds are paired before permutation.

T5. No RT supervision exists in synthetic training.

T6. Model B/C primary context dimension is exactly 8.

T7. Primary size registry accepts only:
- GRU = 150;
- Elman core = 128.

T8. Synthetic validation is frozen; parameter hashes must match before/after.

T9. Human Mixed/Novel metrics cannot be accessed by synthetic selection code.

T10. Human transfer uses identical sequence, geometry, A/B schedule, update count, probe checkpoints, and RT objective across comparable groups.

T11. Primary probes explicitly mark `probe_exposure_index = 1`.

T12. Causal interventions run only on frozen copies.

---

## 27. Required outputs

Synthetic:
- `context_control_architecture_config.json`
- `latent_cause_ecology_config.json`
- `architecture_registry.csv`
- `synthetic_training_curves.csv`
- `synthetic_world_manifest.csv`
- `synthetic_context_evidence.csv`
- `synthetic_split_metrics.csv`
- `synthetic_shared_structure.csv`
- `synthetic_ambiguity_metrics.csv`
- `synthetic_context_belief_decode.csv`
- `synthetic_switch_recovery.csv`
- `synthetic_uncertainty_continuum.csv`
- `synthetic_gate_results.csv`

Human transfer:
- `context_control_human_training.csv`
- `context_control_human_familiar_gates.csv`
- `context_control_human_probe_raw.csv`
- `context_control_human_mixed_metrics.csv`
- `context_control_human_novel_metrics.csv`
- `context_control_human_hidden_geometry.csv`
- `context_control_human_context_state.csv`
- `context_control_human_gate_vectors.csv`

Interventions:
- `context_state_swap.csv`
- `gate_swap.csv`
- `controller_ablation.csv`
- `artificial_interpolation_control.csv`

Main report:
- `CONTEXT_CONTROLLED_RECURRENT_ARCHITECTURES_REPORT.md`

---

## 28. Required figures

1. Architecture schematic: A/B/C.
2. Latent-cause ecology schematic.
3. Synthetic split-versus-ambiguity frontier.
4. Uncertainty continuum.
5. Switch recovery.
6. Shared versus context-specific sequence geometry.
7. Human familiar A/B phenotype.
8. First Mixed profile.
9. Controller-state geometry for A/B/Mixed/Novel.
10. Gate overlap for Model C.
11. A/B differentiation versus Mixed-transfer frontier.
12. Causal intervention results.

---

## 29. Decision logic

### Outcome A: GRU-150 passes synthetic and human transfer
Preferred simplicity result.

Interpretation: standard gated recurrence is sufficient.

### Outcome B: GRU fails, additive context-state model passes
Interpretation: explicit separation of context inference from task dynamics is necessary.

### Outcome C: only soft-gated context model passes
Strong mechanistic result.

Interpretation: context inference must multiplicatively configure shared recurrent dynamics.

### Outcome D: structured and shuffled are similar
Architecture helps, but learned context-task structure is not the cause.

### Outcome E: strong split, poor ambiguity
The model specializes but lacks uncertainty-sensitive retrieval.

### Outcome F: good ambiguity, weak split
The model overgeneralizes and fails the core requirement.

### Outcome G: synthetic pass, human fail
The computation is learnable, but the synthetic ecology still misses target-relevant structure. Do not respond by increasing width first.

---

## 30. Central success criterion

The target result is:

> A context-controlled recurrent model learns distinct A/B skill representations under clear contextual evidence, preserves a reusable shared sequence computation, and on first exposure to an unseen ambiguous context expresses structured reuse of those familiar states without being explicitly trained to average them.

This must occur with:
- no human Mixed training;
- no human Novel training;
- no midpoint loss;
- no explicit A/B output averaging;
- no target-profile fitting;
- no human-probe-based architecture selection.

The decisive mechanistic signature is:

**clear context -> differentiated controller state/gate**

**ambiguous context -> graded controller state/gate**

while **shared sequence dynamics remain reusable across contexts**.
