# PRD — Final SRN Paper Freeze

## Graded Out-of-Support Novelty + Four-Experiment Mechanistic Program + Repository Reorganization

## Status

**Final implementation and paper-freeze program.**

This PRD supersedes the earlier paper-freeze plan **only where explicitly changed here**, principally the definition and evaluation of Novel contexts.

It does **not** reopen model development.

The program has three purposes:

1. replace the historical “Novel1–4” primary probe with a prospectively frozen form of genuine graded contextual novelty;
2. regenerate the four final computational experiments and eight paper analyses under one consistent analysis contract;
3. reorganize the repository into a reproducible paper-facing implementation while preserving all historical evidence and provenance.

The intended endpoint is the manuscript.

No Phase 1M should follow this program.

---

# 1. Goal

The final computational section should answer:

> **When identical sequence knowledge becomes differentiated across familiar contexts, what determines whether that knowledge remains accessible when contextual evidence becomes ambiguous, absent, or genuinely unfamiliar?**

The scientific program has four sequential questions.

### Experiment 1 — Phenomenon

> **Does stronger contextual differentiation accompany poorer expression of the learned sequence when contextual evidence departs from the trained A/B contexts?**

### Experiment 2 — Availability

> **After the network differentiates A and B, does a behaviorally useful shared sequence state still exist inside the network?**

### Experiment 3 — Access and maintenance

> **If a useful shared state exists, why does the network fail to reach or sustain it naturally under changed contextual input?**

### Experiment 4 — Constructive sufficiency

> **If context routing and recurrent stabilization contribute to the deficit, can modifying those mechanisms improve natural generalization while preserving strong A/B differentiation?**

The final result may be full recovery, graded recovery, selective recovery, or a clean failure. No specific scientific outcome is required.

---

# 2. Agent instruction

Work through this PRD sequentially.

Preserve the current repository and all historical outputs before modifying package structure.

### Critical rules

* **Tests before scientific evaluation.**
* Use RED → GREEN implementation.
* Freeze the new Novel registry **before evaluating any checkpoint on those cues**.
* Do not inspect Novel performance and then alter the Novel cue definitions.
* Do not retrain the canonical High/Low checkpoints.
* Do not retrain Phase 1L C0–C3.
* Do not tune an intervention, cue, closure weight, checkpoint, seed set, metric, or plot selection based on the new Novel results.
* Never overwrite historical Phase 1I–1L outputs.
* Write regenerated outputs to a new versioned paper-freeze directory.
* Preserve original historical labels while adding clean paper-facing labels.
* Separate `implementation_status` from `evidence_status`.
* A clean negative result is a completed experiment.
* A missing artifact blocks only the affected analysis.
* Technical failures must be fixed before descendants run.
* Scientific failures must be reported and do **not** trigger additional modeling.

Maintain a running implementation log / TODO document. Record decisions before implementing them. Do not silently resolve scientific ambiguities in code.

---

# 3. Existing evidence this program must respect

The repository already contains the core evidence needed for Experiments 1–4.

### Canonical cohort

Experiments 1–3 use the existing canonical High/Low SRNs:

* eight-unit tanh Elman SRN;
* High initialization scale 2.0;
* Low initialization scale 0.02;
* seeds 0–19;
* A/B familiar training;
* primary final checkpoint at update 170;
* seven scored transitions;
* teacher-forced sequence execution;
* zero initial hidden state for natural probe rollouts.

Phase 1I, 1J, and 1K reuse the same 40 canonical final checkpoints.

### Constructive cohort

Experiment 4 uses the existing Phase 1L sequence-only cohort:

* C0 baseline;
* C1 familiar-contrast bottleneck;
* C2 neutral/shared trajectory closure;
* C3 both mechanisms;
* seeds 0–19;
* 80 total checkpoints.

C0 is bit-identical to the saved High sequence-only cohort and must remain separate from the canonical High monotonic cohort.

### Current mechanistic evidence

The existing work supports:

1. High SRNs develop much stronger A/B differentiation than Low SRNs and generalize less well outside A/B.
2. An A/B-derived composite hidden state can causally improve Mixed, NoContext, and historical held-out cue behavior.
3. Single placement of the useful shared state does not guarantee that it remains stable.
4. Physical cue input does not reliably generate the useful midpoint state from naturally occurring prestates.
5. Phase 1L shows that changing context routing and recurrent closure can improve natural generalization while preserving strong A/B differentiation, but the full historical three-probe phenotype was not recovered.

These are reference findings to reproduce or update under the new Novel definition, not thresholds that new analyses must satisfy.

---

# 4. Critical correction: historical Novel is not the primary Novel test

The existing `Novel1–4` cues must remain immutable for provenance.

However, they should no longer be described as the primary test of genuinely novel contextual input.

The canonical network's third context channel is inactive. Historical Novel behavior therefore arises from how the probe vectors engage the familiar active cue channels.

Rename these only in the new paper-facing registry:

* `LegacyNovel1`
* `LegacyNovel2`
* `LegacyNovel3`
* `LegacyNovel4`
* `LegacyNovelMean`

Do not alter their historical source files, tables, manifests, or original labels.

They may appear in supplementary sensitivity analyses.

They must **not** be pooled with the new Novel conditions.

---

# 5. New primary Novel manipulation

## 5.1 Operational definition

For this paper, Novel means:

> **An unseen continuous contextual state outside the familiar A/B training support and outside the convex interpolation region between A and B.**

This is a graded **out-of-support contextual generalization** manipulation.

It is not claimed to represent a completely new semantic context dimension.

That stronger problem would require the model to have learned a more general context representation and would constitute a new modeling program.

---

# 6. Frozen Novel registry

Use the existing active two-dimensional familiar cue plane.

Let:

$$
A=(1,0,0)
$$

and

$$
B=(0,1,0).
$$

Interpret these as angles:

$$
\theta_A=0^\circ
$$

and

$$
\theta_B=90^\circ.
$$

The familiar angular sector is therefore:

$$
[0^\circ,90^\circ].
$$

Mixed remains:

$$
Mixed=(0.5,0.5,0).
$$

NoContext remains:

$$
NoContext=(0,0,0).
$$

## New Novel contexts

Define four unit-length context vectors outside the familiar angular sector.

### Novel30-A

30° beyond A:

$$
\theta=-30^\circ
$$

$$
c_{N30A}
=
(\cos -30^\circ,\sin -30^\circ,0)
=
(\sqrt{3}/2,-1/2,0).
$$

### Novel60-A

60° beyond A:

$$
\theta=-60^\circ
$$

$$
c_{N60A}
=
(1/2,-\sqrt{3}/2,0).
$$

### Novel30-B

30° beyond B:

$$
\theta=120^\circ
$$

$$
c_{N30B}
=
(-1/2,\sqrt{3}/2,0).
$$

### Novel60-B

60° beyond B:

$$
\theta=150^\circ
$$

$$
c_{N60B}
=
(-\sqrt{3}/2,1/2,0).
$$

Paper-facing labels:

```text
Novel30_A
Novel30_B
Novel60_A
Novel60_B
Novel30Mean
Novel60Mean
NovelMean
```

`Novel30Mean` averages the two 30° conditions within seed.

`Novel60Mean` averages the two 60° conditions within seed.

`NovelMean` averages all four raw Novel variants within seed.

Never calculate NovelMean by averaging `Novel30Mean`, `Novel60Mean`, and their constituent conditions together.

---

# 7. Why this qualifies as novelty

Each new Novel cue is:

* absent from training;
* not the physical Mixed cue;
* not NoContext;
* outside the angular interval occupied by A/B;
* outside the convex A/B interpolation segment;
* unit norm;
* separated from the nearest familiar cue by a prespecified angular distance;
* defined without reference to behavioral results.

The primary novelty variable is:

$$
\delta(c)
=
\min
\left[
\angle(c,A),
\angle(c,B)
\right].
$$

Therefore:

$$
\delta(Novel30)=30^\circ
$$

and

$$
\delta(Novel60)=60^\circ.
$$

Primary graded contrast:

$$
Novel60Mean-Novel30Mean.
$$

Do not require monotonic degradation as an acceptance criterion.

It is a scientific result.

---

# 8. Novel-registry freeze

Before running **any** new-model evaluation:

create:

```text
configs/novelty_registry_v1.json
```

containing:

* exact cue names;
* exact vectors at full numerical precision;
* angle;
* nearest familiar context;
* angular novelty;
* norm;
* convex-hull membership;
* registry version;
* registry SHA256;
* creation date;
* git/repository state if available.

Also create:

```text
docs/novelty_registry.md
```

explaining the scientific rationale.

Once any Novel30/Novel60 behavioral result has been inspected, the registry is frozen.

Any later alternative cue definition would constitute a separate future study.

---

# 9. Important prospective Phase 1L property

The Phase 1L familiar-contrast coordinate is still defined exclusively from A and B:

$$
\bar c=(c_A+c_B)/2
$$

$$
d_c=(c_A-c_B)/2
$$

$$
\alpha(c)
=
\frac{(c-\bar c)^Td_c}
{\|d_c\|^2+\epsilon}.
$$

Do **not** special-case the new Novel cues.

Their effective \(\alpha\) values must be computed automatically.

For the proposed frozen cues, an interesting geometric property follows prospectively:

$$
|\alpha(Novel30_A)|
=
|\alpha(Novel60_A)|
$$

and symmetrically on the B side.

Thus the one-dimensional Phase 1L contrast representation cannot fully preserve the raw novelty-distance distinction.

This is not a bug.

It is a testable representational consequence.

Save both:

* `cue_alpha`
* the residual component discarded by the A/B contrast projection.

The discarded component may be analyzed descriptively.

Do **not** feed it into the model.

Do **not** implement a second `rho` pathway.

---

# 10. Paper analysis contract

Create:

```text
configs/paper_freeze_v2.json
```

and freeze it before paper evaluation.

It must identify:

* novelty-registry hash;
* cohort;
* model family;
* training objective;
* seed;
* checkpoint;
* source checkpoint hash;
* source-code hash;
* sequence;
* cue;
* intervention;
* transition;
* evaluator;
* numerical dtype;
* bootstrap seed;
* aggregation rules;
* output freeze ID.

---

# 11. Primary behavioral outcome

Primary outcome:

**raw transition cross-entropy in natural-log units.**

Sequence CE:

$$
CE_{seq}
=
\frac{1}{7}
\sum_{t=1}^{7} CE_t.
$$

Familiar performance:

$$
CE_{fam}
=
\frac{CE_A+CE_B}{2}.
$$

Also calculate a clearly named descriptive transfer deficit:

$$
D_{probe}
=
CE_{probe}-CE_{fam}.
$$

Raw CE remains primary.

The deficit exists to help distinguish general task-learning differences from relative generalization differences.

Never substitute:

* RT proxy;
* accuracy;
* normalized rescue fraction;
* probability;
* transition-pooled inference

for the primary behavioral outcome.

---

# 12. Primary representational outcome

Task space is centered logits:

$$
z = logits-\overline{logits}.
$$

At each transition:

$$
m_z=(z_A+z_B)/2.
$$

Absolute midpoint distance:

$$
d=
\|z_{probe}-m_z\|.
$$

A/B task separation:

$$
s=
\|z_A-z_B\|.
$$

Normalized midpoint residual:

$$
r=
\frac{d}{s+\epsilon}.
$$

Calculate transitionwise before averaging.

Save all three:

* absolute distance;
* A/B denominator;
* normalized residual.

Hidden-space geometry is secondary.

---

# 13. Aggregation contract

The hierarchy is:

```text
transition
→ probe variant
→ within seed
→ novelty level
→ paired condition contrast
→ across-seed inference
```

Novel aggregation:

```text
Novel30_A + Novel30_B → Novel30Mean
Novel60_A + Novel60_B → Novel60Mean
all four raw variants → NovelMean
```

Novel variants are not independent samples.

Transitions are not independent samples.

Intervention positions are not independent samples.

Seeds are the model-level inferential unit.

---

# 14. Inference

Use paired seed inference whenever conditions share seeds.

Default:

* 10,000 paired bootstrap resamples;
* fixed bootstrap seed `20260925`;
* 95% percentile confidence interval;
* report mean effect;
* median effect where informative;
* report all 20 seeds where present.

Save the actual bootstrap index matrix.

Do not regenerate different bootstrap samples panel by panel.

Do not select analyses based on whether intervals cross zero.

---

# 15. The four experiments

---

# Experiment 1 — Differentiation and generalization

## Scientific question

> **Do networks that develop stronger A/B context-specific representations show poorer expression of the learned sequence when contextual input is ambiguous, absent, or increasingly outside the familiar context distribution?**

This establishes the phenomenon.

No causal claim about splitting is permitted from Experiment 1 alone.

---

## A1 — Canonical phenotype

### Question

> **What is the relationship between A/B differentiation and natural generalization in canonical High and Low SRNs?**

### Use

Canonical High/Low checkpoints, seeds 0–19.

Reuse genuine acquisition logs.

Do not reconstruct acquisition by repeatedly evaluating the final checkpoint.

### Evaluate naturally

For every checkpoint/seed required by the final analysis:

* A
* B
* Mixed
* NoContext
* Novel30_A
* Novel30_B
* Novel60_A
* Novel60_B

Also retain legacy Novel results separately if useful.

### Required outputs

For final checkpoint:

* A CE;
* B CE;
* familiar mean CE;
* Mixed CE;
* NoContext CE;
* Novel30Mean CE;
* Novel60Mean CE;
* NovelMean CE;
* each transfer deficit;
* hidden A/B separation;
* task A/B separation;
* absolute and normalized probe midpoint geometry.

### Planned novelty contrasts

Within each family:

$$
Novel30Mean-CE_{fam}
$$

$$
Novel60Mean-CE_{fam}
$$

$$
Novel60Mean-Novel30Mean.
$$

### Interpretation

A1 can support:

> Strong contextual differentiation co-occurs with reduced generalization beyond the trained context distribution.

A1 cannot establish:

> Splitting causes the deficit.

---

# A2 — Acquisition robustness

## Question

> **Does the differentiation/generalization relationship remain when familiar A/B acquisition is more closely matched?**

### First attempt: canonical same-model matching

Use existing canonical training states from update 0–170.

Freeze a familiar-only matching rule **before** examining probe outcomes.

Matching variables may use:

* familiar A/B CE;
* familiar A/B accuracy if already available;
* training stage/context as necessary.

They may not use:

* Mixed;
* NoContext;
* Novel30;
* Novel60;
* legacy Novel.

Evaluate the frozen new Novel cues only after matching is complete.

### If support is inadequate

Report:

```text
evidence_status = insufficient_common_support
```

Do not:

* widen the caliper after inspecting probes;
* extend training;
* add seeds;
* change matching variables based on Novel behavior.

### Historical analysis

The ordinal/layout matched-acquisition experiment remains a labeled supplementary robustness result.

Do not present it as the same canonical cohort.

Do not use its historical Novel cue as evidence for the new Novel manipulation.

---

# Experiment 2 — A useful shared solution remains available

## Scientific question

> **Has contextual differentiation destroyed the common sequence solution, or does a useful shared state remain causally accessible inside the split network?**

---

# A3 — Composite-state sufficiency

## Question

> **Is an A/B-derived shared state sufficient to support behavior under Mixed, NoContext, and genuinely out-of-support Novel cues?**

Use Phase 1I mechanics with existing frozen checkpoints.

No retraining.

The shared target must remain derived exclusively from familiar A/B states.

Primary target:

$$
h_{shared,t}
=
(h_{A,t}+h_{B,t})/2.
$$

### Re-evaluate with new Novel registry

Run existing intervention logic for:

* Mixed;
* NoContext;
* Novel30_A;
* Novel30_B;
* Novel60_A;
* Novel60_B.

### Interventions

Retain:

* natural / no patch;
* repeated correct composite state;
* acute single composite placement;
* wrong-position control;
* orthogonal control;
* A endpoint;
* B endpoint.

The repeated intervention is the sufficiency ceiling.

At least one acute intervention result must appear in the paper or supplement so that repeated state replacement is not the only causal evidence.

### Primary questions

1. Does the composite state reduce raw CE?
2. Does rescue occur for both Novel30 and Novel60?
3. Does rescue change with novelty distance?
4. Is the correct state more effective than wrong-position and orthogonal controls?
5. Does rescue occur without requiring the natural Novel cue to generate the state?

### Interpretation

Positive evidence supports:

> Strong contextual differentiation does not eliminate a behaviorally useful shared sequence state.

Do not claim:

* the midpoint occurs naturally;
* the midpoint is necessary;
* the midpoint uniquely explains behavior;
* the cue naturally accesses the midpoint.

---

# Experiment 3 — Generation and maintenance

## Scientific question

> **If a useful shared state exists, is poor generalization caused by failure to generate it, failure to maintain it, or both?**

Experiment 3 combines the scientific logic of Phase 1J and 1K.

---

# A4 — Maintenance after placement

## Question

> **Once the shared state is placed into the network, how does subsequent contextual input affect its persistence and behavioral benefit?**

Use existing Phase 1J single-injection mechanics.

No training.

Run new forward evaluations for:

* Mixed;
* NoContext;
* Novel30_A/B;
* Novel60_A/B.

### Continuation conditions

Retain:

* `D0`: native context continuation;
* `D1`: zero-cue continuation;
* `D2`: physical Mixed continuation.

Known invariants remain:

* Mixed D0 = D2;
* NoContext D0 = D1.

For Novel conditions, D0 is the corresponding Novel cue and should generally be distinct from both D1 and D2.

### Retain

* injection position;
* transition;
* lag;
* raw CE;
* raw rescue;
* absolute task midpoint distance;
* normalized task residual;
* hidden geometry.

### Primary summaries

Per seed and probe family:

* lag-zero rescue;
* lag-specific CE rescue;
* lag-specific task-space drift;
* mean remaining-sequence rescue;
* mean post-injection task distance.

Aggregate injection positions within seed before inference.

### Novelty analysis

Compare native Novel continuation for:

$$
Novel60-Novel30.
$$

This asks whether increasingly unfamiliar context input destabilizes or changes the shared state differently.

Do not assume the direction in advance.

---

# A5 — Cue-driven generation

## Question

> **From the same prestate, how does changing only the contextual cue move the network relative to the useful shared state?**

Use Phase 1K identical-prestate branching mechanics.

### New evaluation

Add branches for:

* Novel30_A;
* Novel30_B;
* Novel60_A;
* Novel60_B.

Preserve:

* A;
* B;
* Mixed;
* NoContext;
* endpoint controls.

### Required local analysis

From identical prestates, save:

* post-update hidden state;
* centered logits;
* movement toward midpoint;
* movement along A/B task axis;
* off-axis movement;
* raw CE;
* correct-token probability.

### Persistent takeover

Evaluate the new Novel cues with the existing persistent-takeover mechanism.

Retain takeover position and lag.

Compare:

* local one-step cue effect;
* persistent cue takeover;
* pulse cue effect where currently implemented.

### Key scientific distinction

A cue may:

* help behavior without recreating the exact midpoint;
* move geometry without improving behavior;
* have different immediate and persistent effects.

Preserve these possibilities.

Do not reduce A5 to a binary “cue works / cue fails” outcome.

---

# Experiment 4 — Constructive test

## Scientific question

> **Can changing the mechanisms implicated by Experiments 2–3 improve natural generalization without eliminating the A/B split?**

Use the already-trained Phase 1L C0–C3 models.

There are no hidden-state interventions in the primary Experiment 4 phenotype.

No retraining.

---

# A6 — C0–C3 constructive comparison

## Question

> **Do the familiar-contrast bottleneck and neutral-trajectory closure improve natural generalization under Mixed, NoContext, and graded Novel contexts while preserving strong A/B differentiation?**

Evaluate all 80 existing C0–C3 checkpoints on:

* A;
* B;
* Mixed;
* NoContext;
* Novel30_A;
* Novel30_B;
* Novel60_A;
* Novel60_B.

### Required outputs

Per condition and seed:

* familiar mean CE;
* A/B hidden separation;
* A/B task separation;
* Mixed CE;
* NoContext CE;
* Novel30Mean CE;
* Novel60Mean CE;
* NovelMean CE;
* transfer deficits;
* absolute task midpoint distance;
* normalized midpoint residual;
* cue alpha;
* off-axis cue component;
* neutral closure measures.

### Primary paired comparisons

Within seed:

* C1 − C0;
* C2 − C0;
* C3 − C0;
* C3 − C1;
* C3 − C2.

Primary constructive contrast:

$$
C3-C0.
$$

### Scientific success is multidimensional

Do not define success as “all CE values become Low-like.”

Report separately:

1. familiar performance;
2. A/B differentiation;
3. Mixed generalization;
4. NoContext generalization;
5. Novel30 generalization;
6. Novel60 generalization.

A selective improvement is a scientific result.

---

# A7 — Geometry–behavior relationship

## Question

> **When the constructive intervention changes shared-state geometry, does behavioral performance change with it?**

For each seed calculate C3 − C0:

$$
x=
\Delta task\ midpoint\ distance
$$

$$
y=
\Delta CE.
$$

Primary facets:

* Mixed;
* NoContext;
* Novel30Mean;
* Novel60Mean.

Also provide NovelMean as a summary if useful.

Lower-left means:

* closer to shared geometry;
* lower CE.

Include:

* zero lines;
* one point per seed;
* absolute-distance version;
* normalized-residual sensitivity.

Do not require a fitted correlation.

If a correlation is shown, label it exploratory.

The principal question is whether geometry and behavior can dissociate across probe families.

---

# A8 — Effective-cue / information-loss audit

## Question

> **What information about ambiguous, absent, and Novel contexts survives the familiar A/B contrast bottleneck, and what information is discarded?**

For every cue save:

* raw vector;
* norm;
* raw angle;
* nearest familiar cue;
* novelty angle;
* `alpha`;
* projection magnitude;
* orthogonal residual norm;
* effective context drive.

### Explicitly test

* A maps to +1;
* B maps to −1;
* Mixed maps to 0;
* NoContext maps according to the frozen formula;
* Novel values are computed, never assigned.

For the new Novel30/Novel60 design, verify the expected geometric aliasing produced by the one-dimensional projection.

This is particularly important if two contexts at different raw novelty distances receive the same effective `alpha`.

### Interpretation

A8 may support:

> The one-dimensional familiar-context representation preserves only information relevant to the learned A-vs-B contrast and can collapse context states that differ along other dimensions.

The discarded component may motivate a future richer representation.

Do not implement that representation here.

---

# 16. Updated four-figure manuscript plan

## Figure 1 — Differentiation and graded generalization

Panels:

A. genuine High/Low acquisition;

B. A/B hidden and task-space differentiation;

C. natural raw CE for:

* familiar;
* Mixed;
* NoContext;
* Novel30;
* Novel60;

D. transfer deficit or novelty-gradient panel.

A2 matching is primarily supplementary unless canonical support is strong.

---

## Figure 2 — A shared solution remains causally available

Panels:

A. schematic of A/B-derived composite state;

B. natural versus repeated composite-state CE;

C. acute composite versus wrong-position / orthogonal controls;

D. rescue by probe family and novelty level.

---

## Figure 3 — Generation versus maintenance

Panels:

A. single-state placement and subsequent drift;

B. task-distance / behavioral rescue over lag;

C. identical-prestate cue-generation analysis;

D. pulse versus persistent cue takeover if needed.

Novel30 and Novel60 should remain distinguishable.

---

## Figure 4 — Constructive recovery and boundary

Panels:

A. C0–C3 mechanism schematic;

B. natural CE plus preserved A/B separation;

C. geometry–behavior C3−C0 scatter by probe family;

D. effective-cue projection showing raw context geometry versus the one-dimensional `alpha` representation.

---

# 17. Supplementary analyses

Retain:

* legacy Novel1–4 as explicitly historical held-out-combination sensitivity;
* canonical acquisition matching;
* ordinal/layout historical matching;
* all individual Novel30/60 directions;
* Low-network intervention controls;
* full injection positions;
* full lag curves;
* endpoint controls;
* absolute versus normalized geometry;
* C1/C2 component conditions;
* neutral closure audit;
* numerical evaluator comparison;
* Phase 1G only if scientifically useful and carefully qualified.

Phase 1H may remain supplementary or separate.

Phases 1B–1E remain research history.

---

# 18. Repository target

Create a paper-facing package approximately:

```text
README.md
pyproject.toml
requirements.lock

configs/
  paper_freeze_v2.json
  novelty_registry_v1.json
  experiments/

src/context_sequence/
  task/
    sequence.py
    cues.py
    novelty.py
    schedule.py

  models/
    canonical.py
    contrast.py

  training/
    objectives.py
    reproduction.py

  evaluation/
    rollout.py
    interventions.py
    branching.py
    maintenance.py

  metrics/
    behavior.py
    geometry.py
    novelty.py

  statistics/
    aggregation.py
    bootstrap.py

  provenance/
    registry.py
    hashing.py
    validation.py

  experiments/
    phenotype.py
    state_sufficiency.py
    generation_maintenance.py
    constructive_test.py

  human/
    ...

  paper/
    analyses.py
    figures.py
    captions.py
    manifests.py

scripts/
  audit_paper_inputs.py
  freeze_novelty_registry.py
  run_paper_freeze.py
  reproduce_models.py

tests/
  unit/
  regression/
  integration/

data/
  README.md
  human/

artifacts/
  manifests/
  checkpoints/

results/
  paper/<freeze_id>/
    tables/
    figures/
    captions/
    audit/
    manifest.json
    FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md

docs/
  methods.md
  analysis_contract.md
  novelty_registry.md
  experiment_map.md
  decisions.md
  provenance.md

scratchpads/
  paper_freeze_todo.md

archive/
  research_history/
  artifact_index.json
```

Do not require exact filenames where repository conventions make another name clearly better, but preserve this conceptual separation.

---

# 19. Repository implementation order

## Stage 0 — Freeze existing repository

Before moving code:

* hash source files;
* hash included checkpoints;
* hash included raw tables;
* record environment;
* record dependency versions;
* record current paths;
* record current evaluator behavior;
* extend provenance across imported dependencies.

Do not edit historical reports.

Do not delete historical outputs.

---

## Stage 1 — Initialize clean version control boundary

The current snapshot is not a Git worktree.

Before large moves:

* create appropriate `.gitignore`;
* exclude generated model outputs;
* exclude caches;
* exclude large raw artifacts from ordinary Git history;
* initialize repository/version control if appropriate to the working environment;
* add `pyproject.toml`;
* pin runtime dependencies.

Do not copy the existing multi-GB output tree wholesale into Git history.

---

## Stage 2 — Freeze Novel registry

Implement:

```text
freeze_novelty_registry.py
```

Run only geometric tests.

Do **not** evaluate models yet.

Save and hash the registry.

Commit/freeze it before scientific execution.

---

## Stage 3 — Add paper-facing adapters

Before physically moving old packages:

implement adapters that read existing:

* canonical checkpoints;
* Phase 1I evidence;
* Phase 1J evidence;
* Phase 1K evidence;
* Phase 1L checkpoints/evidence;
* acquisition logs.

The paper runner must have an explicit:

```text
evaluation_only = true
```

mode.

If a checkpoint is absent, fail clearly.

Never train a replacement silently.

---

## Stage 4 — Harmonize evaluator

I–K historically use an explicit NumPy recurrence while 1L uses Torch.

Build one paper-facing evaluator and regression-test it against both historical implementations.

Do not silently declare exact equivalence if floating-point ordering produces small differences.

Declare:

* dtype;
* tolerances;
* recurrence operation order.

Historical files remain authoritative historical results.

New paper tables use the declared common evaluator.

---

## Stage 5 — Run new Novel forward evaluations

With the frozen registry, run evaluation-only analyses for:

* A1;
* A3;
* A4;
* A5;
* A6;
* A7;
* A8.

A2 runs only under its prespecified matching procedure.

No optimizer steps are permitted.

---

## Stage 6 — Generate paper outputs

Produce:

* long-format panel data;
* seed-level summaries;
* paired contrast tables;
* uncertainty;
* four computational figures;
* captions;
* claim-to-source map;
* manifest.

Inspect figures visually.

Do not choose panels because they are significant.

---

## Stage 7 — Extract shared implementation

Only after paper analyses reproduce correctly:

move shared primitives into `src/context_sequence`.

Extract before archiving dependencies such as:

* canonical SRN model;
* task token encoding;
* sequence schedule;
* Phase 1F geometry math;
* timeline/provenance helpers;
* Phase 1H replay helpers;
* Phase 1I cue/intervention mechanics;
* Phase 1J injection logic;
* Phase 1K branching geometry;
* Phase 1L context projection.

Maintain compatibility wrappers while migrating.

---

## Stage 8 — Archive historical research structure

Archive old:

* phase PRDs;
* scratch analyses;
* exploratory scripts;
* obsolete paper runners;
* earlier synthesis outputs.

Do not delete evidence.

Maintain a path crosswalk:

```text
old_path
new_path_or_archive_reference
sha256
reason
paper_dependency
```

---

## Stage 9 — Clean reproduction test

From a clean environment:

1. install package;
2. validate artifacts;
3. freeze/read Novel registry;
4. run analysis-only paper pipeline;
5. recreate paper tables;
6. recreate figures;
7. verify manifest.

Full model retraining is a separate optional reproduction mode.

It is not required for paper-table reproduction when the frozen checkpoints are available.

---

# 20. TDD / RED → GREEN requirements

Write these tests before new scientific execution.

## Novelty tests

### Test N1 — Familiar cue identity

Assert paper-facing A/B cues exactly equal the canonical vectors.

### Test N2 — Novel vector identity

Assert exact registered Novel30/60 vectors.

### Test N3 — Unit norm

Assert all four Novel cues have norm 1 within numerical tolerance.

### Test N4 — Frozen novelty distance

Assert:

```text
Novel30_A = 30° from A
Novel30_B = 30° from B
Novel60_A = 60° from A
Novel60_B = 60° from B
```

### Test N5 — Outside familiar support

Assert every new Novel cue lies outside the familiar [0°,90°] angular sector.

### Test N6 — Outside convex A/B interpolation

Fail if any new Novel cue can be represented as:

$$
\lambda A+(1-\lambda)B,\quad 0\leq\lambda\leq1.
$$

### Test N7 — Symmetry

Assert A-side and B-side cues are geometric reflections.

### Test N8 — Immutable registry

Model evaluation must verify the registry hash before running.

---

## Training/firewall tests

### Test F1 — No new Novel cue in training

Search all canonical and C0–C3 paper training paths.

Fail if Novel30/60 appear in training data or losses.

### Test F2 — Evaluation-only cannot optimize

Monkeypatch / trap optimizer creation and `.step()`.

The paper-freeze evaluation runner must fail if an optimizer step occurs.

### Test F3 — Checkpoint immutability

Hash checkpoint parameters before and after evaluation.

Require exact equality.

### Test F4 — No probe-based seed exclusion

Seed registry must be fixed independently of probe outcomes.

### Test F5 — No Novel-conditioned hyperparameter selection

Fail if novelty metrics enter:

* checkpoint selection;
* closure weight selection;
* intervention strength;
* architecture choice.

---

## Evaluator tests

### Test E1 — Seven transitions

Every natural sequence evaluation contains exactly seven scored transitions.

### Test E2 — Zero-state natural rollout

Natural probe execution begins from the registered zero initial state.

### Test E3 — Teacher-forced identity

Token inputs match the frozen sequence exactly.

### Test E4 — Historical regression

Common evaluator reproduces historical natural CE within declared tolerance.

### Test E5 — NumPy/Torch audit

Save maximum state/logit/CE discrepancy between historical numerical paths.

---

## Intervention tests

### Test I1 — Composite uses A/B only

Composite target cannot depend on Mixed, NoContext, or Novel.

### Test I2 — Correct versus wrong-position semantics

Ensure wrong-position state comes from a different registered sequence transition.

### Test I3 — Orthogonal-control semantics

Ensure orthogonal control has matched intervention norm where required.

### Test I4 — Single injection really occurs once

No subsequent hidden-state overwrite is allowed.

### Test I5 — Persistent takeover is persistent

Cue is replaced on all registered post-takeover steps.

### Test I6 — Pulse occurs once

Cue replacement occurs exactly at the specified step.

### Test I7 — Identical-prestate branches

All local cue branches from a given source row must have bit-identical prestates.

---

## Phase 1L tests

### Test L1 — A/B projection

Assert:

$$
\alpha(A)=+1,\quad\alpha(B)=-1.
$$

### Test L2 — No cue special-casing

Projection function may depend only on:

* cue vector;
* frozen A/B references.

No branch may check labels such as `Mixed`, `Novel30`, etc.

### Test L3 — Effective Novel coordinates

Save computed Novel alpha values without manually assigning them.

### Test L4 — Rank-one context pathway

Context-induced preactivation for C1/C3 must lie in the declared one-dimensional context span.

### Test L5 — Closure target A/B-only

Neutral targets depend only on familiar A/B trajectories.

### Test L6 — No hidden patch in Experiment 4

Natural C0–C3 evaluation may not overwrite hidden state.

### Test L7 — C0 identity

Verify all C0 checkpoints still match their saved sequence-only source checkpoints.

---

## Statistics tests

### Test S1 — Seed pairing

Fail if paired contrasts have mismatched seeds.

### Test S2 — Nested Novel aggregation

Fail on duplicate weighting.

### Test S3 — Transition pseudoreplication

Inference function may not treat transitions as independent model observations.

### Test S4 — Intervention-position pseudoreplication

Positions must be aggregated within seed before across-seed inference.

### Test S5 — Shared bootstrap indices

Related contrasts use the same saved resampling matrix.

---

## Repository tests

### Test R1 — Analysis-only missing checkpoint

Delete/mock one required checkpoint.

Pipeline must fail.

It must not train a replacement.

### Test R2 — Unique table keys

Validate declared primary keys.

### Test R3 — Manifest completeness

Every figure panel must link to:

* source table;
* filter;
* estimator;
* source checkpoint hashes;
* registry hash.

### Test R4 — Clean import

Package installs/imports without requiring archived exploratory packages unnecessarily.

---

# 21. Required result tables

At minimum create:

```text
novelty_registry_audit.csv

a1_canonical_phenotype_transition.csv
a1_canonical_phenotype_seed.csv

a2_acquisition_match_candidates.csv
a2_acquisition_matches.csv
a2_acquisition_robustness_seed.csv

a3_composite_sufficiency_transition.csv
a3_composite_sufficiency_seed.csv

a4_maintenance_transition.csv
a4_maintenance_seed.csv

a5_cue_generation_transition.csv
a5_cue_generation_seed.csv

a6_constructive_transition.csv
a6_constructive_seed.csv

a7_geometry_behavior.csv

a8_effective_cue_audit.csv

paper_primary_contrasts.csv
paper_bootstrap_summary.csv
claim_source_map.csv
```

Each output must identify its cohort and provenance.

---

# 22. Required reporting fields

For every analysis A1–A8, the final report must state:

* scientific question;
* exact cohort;
* exact intervention or natural condition;
* controls;
* implementation status;
* evidence status;
* sample / seed count;
* exclusions;
* primary metric;
* primary contrast;
* uncertainty;
* acquisition/familiar-performance check;
* A/B differentiation check where relevant;
* result;
* interpretation;
* limitation;
* next manuscript decision.

Do not use an old Phase outcome letter as the sole result description.

---

# 23. Interpretation matrix

### Pattern: High differentiates more and generalizes less

Supports:

> differentiation and generalization are associated.

Does not alone support:

> differentiation causes the deficit.

---

### Pattern: shared-state intervention rescues Novel30/60

Supports:

> a useful shared sequence solution remains causally available under genuinely out-of-support contextual input.

Does not prove:

> natural Novel processing reaches that state.

---

### Pattern: rescue weakens with novelty

Supports:

> state sufficiency or recurrent compatibility is graded with contextual novelty.

Do not force this prediction.

---

### Pattern: native Novel cue fails to generate midpoint but patch rescues

Supports:

> availability and access are separable.

---

### Pattern: shared state is generated but rapidly lost under Novel input

Supports:

> recurrent/contextual maintenance contributes to poor transfer.

---

### Pattern: C3 improves Novel30 and Novel60 while preserving A/B split

Supports:

> contextual differentiation and out-of-support generalization need not be mutually exclusive.

---

### Pattern: only Novel30 improves

Supports:

> constructive recovery has a graded generalization boundary.

Do not tune the model to rescue Novel60.

---

### Pattern: Novel30 and Novel60 receive identical effective Phase 1L context coordinates but differ naturally

Supports:

> the contrast bottleneck discards contextual information relevant to the novelty distinction.

---

### Pattern: C3 improves Novel but not Mixed

Supports:

> unfamiliarity and familiar-cue ambiguity are computationally distinct problems.

---

### Pattern: C3 improves Mixed but not Novel

Supports:

> shared-state accessibility under familiar ambiguity does not guarantee extrapolation beyond the training context distribution.

---

### Pattern: A/B differentiation collapses when transfer improves

Supports:

> the intervention exposes a differentiation–generalization tradeoff.

It does not satisfy the stronger constructive objective of preserving contextual differentiation.

---

### Pattern: nothing improves

Supports:

> the state-level causal findings are insufficient to construct autonomous generalization under the tested mechanisms.

Stop.

---

# 24. Manuscript claims to avoid

Do not claim that:

* humans literally use the SRN context geometry;
* the new Novel cues are a completely new semantic context dimension;
* historical Novel1–4 were genuine independent context dimensions;
* midpoint geometry is necessary;
* shared geometry alone determines behavior;
* Phase 1L proves a biological mechanism;
* a one-dimensional A/B contrast is the uniquely correct context representation;
* an attractor exists without appropriate dynamical evidence;
* normalized residual improvement alone demonstrates behavioral improvement;
* Low networks provide acquisition-matched causal evidence by themselves.

---

# 25. Stop rule

After this paper-freeze program:

**Do not run:**

* Phase 1M;
* a second `rho` context pathway;
* a learned novelty detector;
* a Novel-specific adapter;
* new architectures;
* larger hidden dimensions;
* new model families;
* new closure-weight searches;
* new seed searches;
* probe-supervised fine-tuning;
* hidden-state repair for Experiment 4;
* extra epochs because Novel60 performs poorly;
* altered Novel angles because results are inconvenient;
* new ecological/pretraining programs;
* additional human data collection for this computational program.

A clean positive, mixed, or negative outcome ends the modeling program.

Future work may propose richer context representations, but that work is not part of this paper.

---

# 26. Definition of done

This program is complete when:

1. all existing source evidence is hashed and indexed before cleanup;
2. the new Novel registry is frozen before behavioral evaluation;
3. Novel30/60 pass all geometric validity tests;
4. historical Novel1–4 remain immutable and clearly relabeled in the paper-facing layer;
5. no new Novel cue enters training or tuning;
6. no optimizer updates occur during the paper-freeze run;
7. A1–A8 have validated implementations;
8. all available required seeds are evaluated;
9. acquisition uses genuine training-time observations;
10. raw CE is retained for every scored transition;
11. absolute and normalized geometry are both retained;
12. Novel variants are correctly nested within seed;
13. A/B separation is quantified throughout the relevant experiments;
14. Experiment 2 includes appropriate causal controls;
15. Experiment 3 distinguishes cue generation from state maintenance;
16. Experiment 4 uses natural evaluation only;
17. C0 remains distinct from canonical High monotonic;
18. the common evaluator is regression-tested against historical implementations;
19. paired seed inference and bootstrap indices are frozen;
20. four computational manuscript figures are generated;
21. all figure panel data are exported;
22. captions are generated from declared estimands rather than old outcome labels;
23. a claim-to-source manifest exists;
24. the paper-facing repository installs in a clean environment;
25. archived research evidence remains traceable;
26. no historical artifacts are silently deleted;
27. the final report records positive and negative results without post-result tuning;
28. the README presents the four scientific experiments rather than the historical phase sequence;
29. exact reproduction commands are documented;
30. no further model changes are made after the frozen evaluation is inspected.

At that point:

> **stop modeling and write the paper.**

---

# 27. Proposed reproduction commands

The coding agent should implement an interface approximately equivalent to:

```bash
# Validate frozen historical inputs
python scripts/audit_paper_inputs.py \
  --config configs/paper_freeze_v2.json

# Freeze Novel definitions before behavioral execution
python scripts/freeze_novelty_registry.py \
  --config configs/novelty_registry_v1.json

# Run all evaluation-only paper analyses
python scripts/run_paper_freeze.py \
  --config configs/paper_freeze_v2.json \
  --mode evaluation_only

# Run focused paper tests
pytest tests/unit tests/regression tests/integration

# Optional explicit training reproduction only
python scripts/reproduce_models.py \
  --config configs/paper_freeze_v2.json \
  --mode reproduce
```

Exact CLI syntax may follow repository conventions, but the distinction between evaluation and training reproduction is mandatory.

---

# 28. Final deliverable

Produce:

```text
results/paper/<freeze_id>/FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md
```

The report should begin with the four questions and their empirical answers.

It should not begin with Phase 1A–1L chronology.

Include a historical crosswalk:

```text
Paper Experiment 1 -> canonical confirmatory/timeline work
Paper Experiment 2 -> primarily Phase 1I
Paper Experiment 3 -> primarily Phase 1J + 1K
Paper Experiment 4 -> primarily Phase 1L
```

Historical phases remain accessible for provenance, but the manuscript-facing scientific organization is the four-experiment structure defined in this PRD.
