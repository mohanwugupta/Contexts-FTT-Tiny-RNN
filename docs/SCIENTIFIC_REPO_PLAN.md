# Plan for the four-experiment scientific repository

Prepared 2026-09-25 after cataloguing the existing project. Evidence and exact source paths are in [FINAL_EXPERIMENT_INFRASTRUCTURE.md](FINAL_EXPERIMENT_INFRASTRUCTURE.md); schemas, hashes, and dependency edges are in [FINAL_EXPERIMENT_INVENTORY.json](FINAL_EXPERIMENT_INVENTORY.json).

**Recommended approach: freeze the existing evidence, build a small paper analysis layer over it, validate it, and only then reorganize the implementation.** Experiments 2–4 already have their required manipulations, checkpoints, and transition-level data. Experiment 1 has the canonical phenotype, but its historical matched-acquisition evidence belongs to a different model/cohort. No Phase 1M, new architecture, probe tuning, or new scientific training program is needed for the recommended scope.

Historical planning document. The implementation now lives in `src/context_sequence/`, `configs/paper_freeze_v2.json`, and `scripts/run_paper_freeze.py`; see the root README and final report for the implemented scope and the A2 support limitation.

## 1. Decisions to record before the paper freeze

| Issue | Evidence | Recommended disposition |
|---|---|---|
| Which baseline is canonical? | 1I–1K share 40 High/Low monotonic checkpoints, seeds 0–19, at update 170. | Use these for Experiments 1–3. Use existing audited acquisition logs for training curves. |
| How does 1L connect to that baseline? | All 20 C0 states exactly equal saved High sequence-only states. | Keep 1L as the prospective sequence-only C0–C3 experiment. State the objective difference explicitly and compare C3 to its own paired C0. Do not pool C0 with High monotonic. |
| Does the available matched-acquisition analysis satisfy the same-model contract? | No: ordinal/layout model, seeds 1000–1019, historical cues, only 3/12/18 supported pairs at the three anchors. | Retain as clearly labeled supplementary robustness. Qualify the conclusion to supported cases and that architecture. Do not present it as a canonical 20-seed three-probe result. |
| What if canonical acquisition matching is indispensable? | The matching utility exists; canonical registered states span updates 0–170 via replay. Extended canonical states through 340/680 were not located. | First specify familiar-only matching on the existing 0–170 range and report support, using frozen replay only if needed. If support is inadequate, report that limitation. Extending training would change the current no-new-training scope and is not included in this plan. |
| What anchors the human interpretation? | Corrected retrieval and state-space adjudication are newer than the README's low-rank narrative and have explicit limitations. | Provisionally reference corrected retrieval plus the adjudication recovery boundary; leave final author selection open. Do not select human results for agreement with model outcomes. |
| Which numerical evaluator is authoritative? | I–K use explicit NumPy recurrence; L uses Torch. Same checkpoint gives up to 3.90e−5 transition CE difference across these paths. | Keep original artifacts immutable. Develop a common step/evaluation API and compare each path to its historical reference, with declared dtype and justified numerical tolerances. Record any changed derived values. |

The brief's “same checkpoints, same seeds” is achievable within the canonical experiments and within C0–C3, but not literally across the historical matched-acquisition study and both training objectives. The clean repository must make these cohort boundaries explicit rather than harmonizing away real differences.

## 2. Freeze one analysis contract

Create a human-readable `configs/paper_freeze.json` plus a schema. Freeze it before regenerating results. Include:

- **Cohorts and provenance:** canonical High/Low seeds 0–19; separate High sequence-only C0–C3 seeds 0–19; historical matched-acquisition seeds 1000–1019 only if included. Record model family, training objective, condition, checkpoint step, file hash, parameter hash, source-code hash, and preprocessing/configuration version. Preserve original labels while adding unambiguous paper labels.
- **Task:** the exact registered sequence, seven predictions, zero initial state for natural probes, teacher-forced inputs, reset boundary, and update-170 primary checkpoint. Reuse historical acquisition logs; never generate a learning curve by repeating a terminal checkpoint.
- **Cues:** one immutable A/B/Mixed/NoContext/Novel1–4 registry using the vectors catalogued in the inventory. Retain a distinct identifier for historical Novel. Preserve the inactive third channel and audit effective cue drives; do not silently modify the model to make that channel active.
- **Behavior:** per-transition raw cross-entropy in natural-log units; sequence CE is the equal mean of exactly seven predictions. A/B familiar CE is the equal mean of A and B. Optional deficit measures name their reference explicitly. Do not substitute RT, accuracy, or normalized rescue for the primary CE outcome.
- **Geometry:** centered logits `z = logits - mean(logits)`; reference `m_z = (z_A + z_B)/2`; distance `d = ||z_probe - m_z||`; A/B separation `s = ||z_A - z_B||`; normalized residual `r = d/(s + epsilon)`, computed transitionwise before averaging. Save `d`, `s`, and `r`. Retain hidden geometry as secondary. For matched-state branches, use same-prestate A/B references, distinct from natural endpoint trajectories.
- **Aggregation:** transition → cue within seed → NovelMean over exactly four variants → paired condition effect within seed → across-seed inference. Never average an already included NovelMean together with its four constituent cues. For injection/takeover analyses retain position and lag until the declared summary step; lag-zero and post-injection effects are distinct estimands.
- **Inference:** paired seed bootstrap, 10,000 resamples, one declared bootstrap seed (proposed `20260925`), 95% percentile intervals and seed counts. Prefer a saved resampling-index matrix shared across related contrasts within each cohort. Resample historical seed clusters with all their anchor rows if pooling anchors; never treat 33 matches as 33 seeds. Human uncertainty remains participant-based.
- **Contrasts:** list primary contrasts in advance of the freeze rerun, including separate Mixed/NoContext/NovelMean effects. Preserve original prospective/posthoc status; a final analysis specification does not retroactively preregister exploratory analyses. Report effect sizes and intervals, not significance-based selection of panels. If p-values are introduced, declare the comparison family and correction before generating them.
- **Training firewall:** no probe behavioral outcomes for training, tuning, checkpoint selection, or seed exclusion. Explicitly disclose A/B-derived neutral closure computation at the physical Mixed cue in C2/C3. Distinguish derived representational supervision from protected behavioral supervision.
- **Execution:** evaluation-only by default, missing artifacts cause a clear failure, fresh versioned output directory, no writes into historical evidence roots. Training and exact acquisition reconstruction must be separately named modes.

Suggested common identifiers are `experiment`, `cohort`, `model_family`, `training_objective`, `condition`, `seed`, `checkpoint_step`, `probe`, `transition`, `intervention`, `injection_position`, `lag`, and `context_continuation`. Not every table needs every field; table schemas should declare required keys and their uniqueness.

## 3. The eight paper analyses

These are bounded analysis jobs under one freeze contract, not eight additional scientific experiments.

| Job | Source and operation | Required output / acceptance condition |
|---|---|---|
| A1. Canonical phenotype | Final natural rows from Phase 1K; canonical acquisition/timeline rows from Phase 1I; High/Low checkpoint references. | A/B and all probe CE; hidden/task A/B separation; genuine acquisition plot. Exactly 20 paired seeds per family, all seven transitions and all four Novel variants. |
| A2. Acquisition robustness | Historical E1 candidate and selected-match tables, primary `.05` caliper, with matching code audited. | Separate historical-cohort table with match rates and paired raw CE/deficit effects at each anchor. Report 3/12/18 supported pairs and unmatched cases. If canonical matching is chosen instead, freeze its rule and report insufficient support rather than widening calipers after looking at probes. |
| A3. Composite-state sufficiency | Phase 1I final `training_sequence_index == 170`; repeated `patch_position == 0`; natural/free geometry separated from matched-state branches. | Raw CE and paired rescue for no patch, composite, wrong-position, orthogonal, and A/B endpoints. Novel variants retained individually and averaged within seed. Same frozen A/B target applied to all probes. |
| A4. Maintenance after placement | Phase 1J transition rows for single midpoint injection under D0/D1/D2; repeated patch is a reference. | Lag-zero rescue, lag-specific task drift and raw rescue, plus a prespecified remaining-sequence summary. Aggregate available positions within each seed before inference; state changing lag support. Assert Mixed D0=D2 and NoContext D0=D1. |
| A5. Cue-driven generation | Phase 1K identical-prestate local branches, A/B→Mixed convergence, and pulse/persistent takeover rows with endpoints. | Task-space movement toward midpoint from A, B, NoContext, and NovelMean; paired takeover rescue. Preserve positive partial cue effects while showing failure of reliable midpoint generation. |
| A6. Constructive C0–C3 comparison | Phase 1L final transition and seed tables, all 80 checkpoints, neutral rollout and provenance. | Natural CE for each of the three probes; A/B familiar CE and separation; task residuals; C3−C0 and component comparisons. All effects paired by seed within the sequence-only cohort. Preserve the failure of full three-probe recovery. |
| A7. Geometry–behavior dissociation | Join C0 and C3 rows in `phase1l_seed_summary.csv` on family/seed. | One point per seed per probe, `x = task_residual_C3 - task_residual_C0`, `y = CE_C3 - CE_C0`, three facets. Lower-left means both improve. Include zero reference lines; do not require a fitted correlation. Add absolute-distance sensitivity because separation changes across conditions. |
| A8. Effective-cue audit | Phase 1L projection table and model projection code, checked against saved metrics. | Confirm alpha and effective drive, including Mixed=NoContext=Novel1=Novel4 at alpha zero in C1/C3. Check identical trajectories where recomputed, and identical saved CE/residuals. Explain missing contextual amount/familiarity as a hypothesis, without implementing rho. |

A1 and A3–A8 can primarily consume existing tables. Forward-only reevaluation supplies standardized raw states/logits, absolute distances, or missing audit columns without new training. A2 is the only substantive cohort-scope issue. The table-based geometry–CE scatter requires no model execution at all.

Natural conditions and interventions should share one registry, but distinct estimands should not be forced into one ambiguous column. In particular, full-sequence CE, next-token CE, mean remaining CE, summed remaining rescue, and mean post-patch rescue are different quantities.

## 4. Main figures and supplements

These are the **four computational main figures**, not a decision about the total number of human/manuscript figures.

| Figure | Panels | Main source |
|---|---|---|
| 1. Splitting and generalization cost | A: genuine acquisition and familiar endpoints; B: High/Low task-space and hidden A/B separation; C: natural A/B/Mixed/NoContext/NovelMean CE. Link to separately labeled acquisition robustness. | A1; A2 primarily supplementary because its architecture differs. |
| 2. A sufficient composite state | A: natural task-space midpoint residual, with matched-state composition secondary; B: baseline/composite/control CE or paired rescue for all three probes. Show seed effects and make endpoint controls available. | A3; selected Phase 1F support only if needed. |
| 3. Generation versus maintenance | A: injection schematic and task drift/CE rescue by lag and continuation; B: physical-cue generation from matched prestates; C: pulse versus persistent takeover, if needed to connect the two. | A4–A5. |
| 4. Constructive test and its boundary | A: C0–C3 design; B: natural probe CE effects and preserved A/B separation; C: three geometry–CE scatter facets; D: cue-projection equivalence explaining the Mixed limitation. | A6–A8. |

Recommended supplements:

1. Acquisition matching, support counts, familiar-learning checks, and explicit architecture/cohort differences.
2. Full Phase 1I directional/endpoint controls, all Novel variants, hidden versus task geometry, and raw versus normalized distances.
3. Full injection/takeover positions, lag support, Low controls, and denominator-stability diagnostics for 1J/K.
4. Phase 1L component conditions, each Novel variant, neutral closure audit, per-seed C0 sequence-only identity, and absolute-distance sensitivity.
5. Optional Phase 1G comparison if its continuous results add a useful control. Do not reproduce “improvement only with collapse” without qualifying what counts as substantial/full recovery; the report table shows smaller improvements with preserved separation.

Phase 1H is optional supplementary or separate work. Phases 1B–1E and the earlier ecological/meta-learning sequence belong in the research record, not the main paper pipeline. Preserve unfavorable controls and uncertainty even when compressing the narrative.

Each figure should export a PDF/SVG plus PNG, the exact plotted long-format table, caption, and a manifest linking every panel to source hashes and filters. Error bars derive from seeds or participants, never transition rows. A paper claim should be traceable to its estimator and artifact rather than only to an outcome letter in an old report.

## 5. Proposed repository structure

```text
README.md
pyproject.toml
requirements.lock              # or an equivalent pinned environment lock
configs/
  paper_freeze.json
  experiments/                 # canonical and C0–C3 reproduction specifications
src/context_sequence/
  task/                        # sequence, cues, encoding, schedules
  models/                      # canonical SRN, contrast SRN, policy
  training/                    # canonical / sequence-only / closure objectives
  evaluation/                  # frozen rollout and intervention primitives
  metrics/                     # raw CE, centered-logit geometry
  statistics/                  # seed aggregation and paired bootstrap
  provenance/                  # registry, hashes, validation
  experiments/
    phenotype.py
    state_sufficiency.py
    generation_maintenance.py
    constructive_test.py
  human/                       # declared human analyses and adapters
  paper/                       # analysis orchestration, tables, figures, captions
scripts/
  audit_paper_inputs.py
  run_paper_freeze.py
  reproduce_models.py          # explicit opt-in reproduction, separate from freeze
tests/
  unit/
  regression/
  integration/
data/
  README.md                    # source, schema, access and preprocessing provenance
  human/                       # original filenames can stay until paths migrate
artifacts/
  manifests/                   # hashes and portable paths tracked with code
  checkpoints/                 # retained small models or checksum-verified retrieval
results/paper/<freeze_id>/
  tables/
  figures/
  captions/
  manifest.json
docs/
  methods.md
  analysis_contract.md
  experiment_map.md
  decisions.md
  provenance.md
archive/
  research_history/            # PRDs, reports, scratchpads, old entry points
  artifact_index.json          # links/hashes for larger retained research artifacts
```

The clean entry point should explain the four scientific questions, distinguish analysis-only reproduction from training reproduction, and give resource expectations. It should not ask readers to navigate phase letters or dozens of old scripts. Keep a crosswalk from paper experiment IDs to historical phase IDs for provenance.

Git should track code, specifications, tests, compact figure tables, and artifact manifests. Choose a scientific artifact store or other checksummed storage for large checkpoints/raw outputs; no particular service is required for this plan. Existing local human data can remain in place while their access and sharing terms are documented. Do not copy the 12.77 GB tree wholesale into ordinary Git history.

## 6. Implementation order and completion gates

### Step 1 — Freeze and index the evidence

Record the current code snapshot, environment, source contracts, final checkpoint identities, and selected tables before any moves. Extend the existing inventory to hashes for all imported source dependencies, not only each phase's own files. Preserve current path/hash records as historical metadata; add portable relative-path aliases.

Initialize a versioned repository as part of the implementation work after defining generated-file exclusions. Retain all existing scientific outputs unchanged. Give smoke, exploratory, confirmatory, and paper-freeze artifacts distinct identifiers.

**Done when:** every included claim has source artifacts; canonical 40 and L 80 checkpoints validate; table/seed/cue completeness checks pass; human and matching scope are explicit. A missing artifact is recorded as missing, not regenerated silently.

### Step 2 — Add paper adapters before moving packages

Implement adapters for I/J/K/L and the selected baseline/matching tables. Add explicit cohort and training-objective fields. Build one natural-baseline view, intervention views with clear timing, and one paired-seed comparison format. Keep original reports intact.

Add an analysis-only runner that reads frozen evidence, recomputes standardized derived quantities/statistics, and writes to a fresh directory. It must not invoke the historical all-phase runners or call `train_run` when a checkpoint is missing. Separate the checkpoint source root from the output root; Phase 1L currently couples them.

**Done when:** all eight analysis outputs can be generated from available evidence with zero optimizer updates, apart from any separately declared exact canonical acquisition reconstruction chosen for A2. Canonical acquisition logs should ordinarily make reconstruction unnecessary for A1.

### Step 3 — Harmonize evaluation and inference

Centralize cue definitions, metric formulas, cohort identifiers, Novel aggregation, and seed bootstrap. Freeze sign conventions: rescue can be baseline minus intervention, while the requested scatter uses C3 minus C0 on both axes. Label them distinctly.

Resolve the NumPy/Torch path difference through regression comparisons of states, logits, CE, and intervention inheritance. Do not change precision or recurrence operation order without recording the effect. Preserve absolute geometry so changes in the normalization denominator remain visible.

**Done when:** paper estimates reproduce historical results within explained numerical tolerances; no inference resamples transitions or variants; all exclusions and denominators are explicit; C0 and canonical High remain separate.

### Step 4 — Produce and inspect the four figures

Regenerate seed tables, statistics, figures, captions, and a claim-to-panel/source map. Visually inspect exported figures for readable labels, complete error bars, appropriate scales, and explicit cohort labels. Show the Mixed limitation and uncertainty even when other probes improve.

**Done when:** the figures answer the four questions, use raw CE and primary task geometry consistently, and link to exact panel data. No “necessary” or autonomous-attractor claim exceeds the interventions actually run.

### Step 5 — Extract the shared implementation

Move validated task/model/objective/hash/geometry functions into the new package in small units. Start with low-level functions and remove eager pipeline imports from package initializers. Maintain temporary compatibility wrappers so original runners still resolve while imports migrate.

In particular, extract Phase 1F math, Phase 1H replay helpers, timeline provenance helpers, Phase 1I cues/interventions, and Phase 1K geometry **before** archiving their surrounding narratives. Break dependencies from modeling primitives into human analysis and old report pipelines.

**Done when:** the new package installs in a clean environment, late-phase analyses no longer require unrelated scientific programs through imports, and checkpoint/metric/intervention regression checks still pass. Preserve original source hashes and record new hashes instead of overwriting old provenance.

### Step 6 — Archive historical material and refresh documentation

Relocate old PRDs, scratchpads, synthesis reports, and runners with a source-to-destination crosswalk. Move or externalize bulky exploratory artifacts only after their manifests and retained copies are verified. Do not delete the evidence just because it is excluded from the narrative.

Rewrite the README and methods around the four experiments. Mark the older `run_all_final.py` and `outputs/final/` products as historical so “final” no longer ambiguously identifies several generations. Add dependency declarations, installation checks, CI, artifact-access instructions, and an owner-selected license/citation record before public release.

**Done when:** a fresh checkout can reproduce the paper from documented artifacts, old evidence remains traceable, and only the intended paper path is advertised as current.

## 7. Verification strategy

Reuse the substantive tests already in `tests/test_phase1i_context_composition.py`, `test_phase1j_context_stability.py`, `test_phase1k_cue_generation.py`, `test_phase1l_contrast_closure.py`, and baseline/timeline/matching tests. Some fixtures train; separate those into explicit reproduction tests rather than running them inside the analysis-only contract.

Meaningful additional checks for the integration work are:

- Unique table keys; exact cohort/seed coverage; seven transitions; all four Novel variants; expected model and cue hashes; correct checkpoint labels.
- Frozen forward equivalence against the original evaluators, plus before/after checkpoint checksums.
- Correct-state versus wrong-position/orthogonal intervention semantics; exactly one inherited patch for single injection; one pulse versus persistent takeover; bit-identical prestates for local branches.
- Paired-seed alignment and nested Novel aggregation, including failure on a missing seed or missing cue rather than silent averaging.
- C0 identity with High sequence-only, and the C1/C3 zero-alpha equivalences. These are invariants, not new scientific hypotheses.
- Absolute and normalized geometry consistency, declared handling of small A/B denominators, and no duplicate weighting of NovelMean.
- An analysis-only execution check that cannot call optimizer steps or silently train on missing input. Test this failure behavior with a missing-checkpoint fixture.
- Figure-source manifest completeness and clean-environment import/execution checks.

Run focused checks after each extraction, then one final integration pass. Full-model reproduction is a separate optional release verification, not a prerequisite for obtaining the already-supported paper tables and figures.

## 8. Run / do not run

**Run during implementation:** provenance and completeness audits; existing-table aggregation; one frozen evaluation pass where standardized states/geometry are needed; paired seed inference; the geometry–CE scatter; cue projection/effective-drive audit; four computational figures and chosen supplements; targeted regression and release checks.

**Run only if specifically needed:** exact deterministic acquisition reconstruction for missing canonical intermediate states; selected F/G/H supplementary analyses from frozen data; a separately scoped historical matching reproduction. A2 must report its cohort exception or lack of canonical support openly.

**Do not run for this paper:** Phase 1M; a new rho pathway; new architecture/width/ecology/meta-learning searches; new probe definitions; further closure-weight or seed selection; routine retraining of C0–C3; full historical all-final pipelines; additional human data collection inferred from old research programs.

The recommended first implementation milestone is a reproducible paper analysis layer operating over the current frozen evidence. Physical cleanup follows demonstrated equivalence, which keeps the repository reorganization from becoming another modeling program.
