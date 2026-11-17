# Final experiment infrastructure inventory

Audited 2026-09-25 against the supplied four-experiment brief. This is an inventory of existing materials, followed by implementation implications. The cleanup and paper-freeze proposal is in [SCIENTIFIC_REPO_PLAN.md](SCIENTIFIC_REPO_PLAN.md). The machine-readable catalogue, including table schemas and checksums, is [FINAL_EXPERIMENT_INVENTORY.json](FINAL_EXPERIMENT_INVENTORY.json).

**The four experiments are substantially implemented and their central artifacts are present.** The remaining work is chiefly provenance reconciliation, a common analysis contract, and manuscript figures. Two exceptions prevent treating everything as one identical model cohort: the historical matched-acquisition study uses a different architecture/cohort, and Phase 1L uses a sequence-only baseline rather than the monotonic-objective checkpoints used by 1I–1K.

Only read-only inspections, checkpoint hashing/loading, frozen forward evaluations, and table comparisons were performed. No training, training replay, experiment runners, source edits, moves, or deletions were performed. The only repository additions from this task are this inventory, its JSON companion, and the plan.

## 1. Repository state and reusable infrastructure

The initial snapshot contains 11,189 files, approximately 12.77 GB, excluding `__pycache__`; 52 top-level Python packages; 62 files under `scripts/`; and 70 files under `tests/`. No `AGENTS.md` was found. This directory is **not a Git worktree**. Recent manifests also record `unavailable:not-a-git-worktree`.

| Layer | Existing infrastructure | Assessment |
|---|---|---|
| Model and baseline training | `confirmatory_srn/{model,config,contexts,training,pipeline}.py` | Correct canonical eight-unit tanh SRN; saved High/Low and High sequence-only cohorts. |
| Task primitives | `ftt_srn_execution_v2/inputs/token_onehot.py`, `training/ftt_schedule.py` | Reused throughout the newer experiments; extract rather than rewrite. |
| Objective and policy | `execution_objective_search/objectives.py`, `confirmatory_srn/model.py` | Canonical objective helpers and monotonic execution policy remain live dependencies. Raw CE can be the paper outcome without changing the historical training objective. |
| Checkpoint provenance and acquisition | `srn_training_timeline_audit/core.py`, `phase1h_plasticity_audit/replay.py` | Registered hashes, canonical payload checks, deterministic replay, optimizer/RNG audits. These are essential even if the corresponding scientific phases are omitted. |
| Cues and compositional interventions | `phase1i_context_composition/{cue_encoding,mechanics}.py` | Shared Novel1–4 definitions, centered logits, familiar composite, deterministic wrong-position and orthogonal controls. |
| Geometry and metrics | `phase1f_shared_backbone/math.py`, Phase 1I mechanics, Phase 1K interventions | Useful primitives exist, but duplicate implementations and normalization conventions need an explicit common contract. |
| Final experiments | `phase1i_context_composition/`, `phase1j_context_stability/`, `phase1k_cue_generation/`, `phase1l_contrast_closure/` | Complete configurations, engines, seed aggregation, bootstrap summaries, figures, manifests, and tests. |
| Human analyses | `mixed_novel_retrieval_corrected/`, `human_state_space_adjudication/`, earlier configured-state packages | Several distinct result generations exist; the final manuscript anchor is not declared consistently. |
| Reproduction | `reproducibility/`, `requirements.txt`, `pytest.ini`, per-phase manifests and partial caches | Strong historical checksum machinery, but no unified late-phase paper manifest or package installation configuration. |
| Old paper entry points | `scripts/run_all_final.py`, `run_srn_final.py`, `run_human_final.py`, `make_paper_figures.py` | Refer to the previous paper organization, not the four-experiment brief. Do not use as the new paper-freeze runner. |

Python 3.11.8 is available. Installed NumPy 2.2.6, pandas 2.3.2, PyTorch 2.8.0, matplotlib 3.10.5, pytest 8.4.1, SciPy 1.16.1, scikit-learn 1.7.1, and statsmodels 0.14.6 agree with the eight pins in `requirements.txt`. The historical transfer pipeline imports PyYAML; it and `tabulate` (used by pandas Markdown reporting) are not declared there. There is no root `pyproject.toml`, environment lock, or CI configuration in this snapshot.

## 2. Experiment-to-material mapping

### Experiment 1: establish the computational phenotype

**Canonical model source:** `confirmatory_srn/` and `outputs/confirmatory_srn_final/`.

- `confirmatory_checkpoints/`: 60 saved checkpoints: High monotonic, Low monotonic, and High sequence-only, each with seeds 0–19.
- `confirmatory_config.json`, `CONFIRMATORY_SRN_REPORT.md`, `confirmatory_seed_summary.csv`, training/probe tables, and hidden-state artifacts establish the baseline specification.
- `analysis/srn_training_timeline_audit/checkpoint_provenance.csv`, `training_sequence_losses.csv`, and `replay_fidelity.csv` provide the acquisition/provenance audit.
- For the paper's genuine Novel probes, use the newer natural evaluations in `analysis/phase1k/phase1k_natural_rollouts.csv` and Phase 1I's `phase1i_timeline_metrics.csv` / `phase1i_true_acquisition.csv`. Do not use the original confirmatory `Novel` as if it were Novel1–4.
- `outputs/srn_mixed_novel_diagnostics/` and `outputs/srn_nonlinearity_compositionality/` provide supporting baseline geometry; the latter includes frozen-checkpoint and Novel-input audits.

The canonical final natural CE means, verified across the newer tables, are:

| Family | A | B | Mixed | NoContext | NovelMean |
|---|---:|---:|---:|---:|---:|
| High monotonic | 0.9173 | 1.4558 | 2.2483 | 2.9400 | 2.8422 |
| Low monotonic | 0.0484 | 0.0504 | 0.0489 | 0.0628 | 0.0685 |

High/Low mean hidden A/B separation is approximately 2.9298 / 0.0664; centered-logit separation is 5.8081 / 0.1056. The baseline supports splitting and a transfer cost, but familiar CE also differs substantially. Acquisition robustness must therefore remain a separately qualified result.

**Matched-acquisition source:** `transfer_experiment_program/{analysis,experiments,modeling}.py`, `configs/transfer_experiment_program.yaml`, and `outputs/transfer_experiment_program/e1_acquisition/`.

Useful tables are `acquisition_matches.csv`, `strict_match_outcomes.csv`, `sequence_results.csv`, `transition_results.csv`, `splitting_trajectories.csv`, and `run_audit.csv`. The matching function uses labeled A/B NLL, A/B accuracy, and incoming block context, without probe selection.

**Important mismatch:** this study uses `OrdinalLayoutElmanSRN`, with token inputs plus a seven-by-42 layout-feature array, and additional seeds **1000–1019**. It is not the seven-input `ConfirmatoryElmanSRN` cohort used by 1I–1K. Its original `Novel` is also a different probe definition. The primary .05-nat matches number 3 at anchor 170, 12 at 340, and 18 at 680: **33 seed–anchor matches from 18 unique seeds**, not 33 independent networks. Other calipers coexist in the tables and must be filtered explicitly.

**Implementation needed:** one canonical phenotype table/figure, explicit acquisition-curve provenance, and a decision about whether historical matching is a labeled supplementary robustness analysis or must be reproduced on canonical states. There is no located canonical matched-acquisition result that can simply replace the historical one. No directory/report explicitly named Phase 1A was found; `output_relevant_compositionality/` is an intervention program, not an unambiguous canonical-baseline designation.

### Experiment 2: the split network contains a useful composite solution

**Primary implementation:** `phase1i_context_composition/`.

**Entry point:** `scripts/run_phase1i_context_composition.py` (`--config`, `--output`, `--smoke`). Its historical runner performs deterministic acquisition replay if partials are absent; it is not a minimal table-only paper exporter.

**Frozen output root:** `analysis/phase1i_context_composition/`.

| Artifact | Contents / size |
|---|---|
| `phase1i_seed_summary.csv` | 40 seed/family rows; geometry, causal rescue, controls, and gradient bridge. |
| `phase1i_transition_metrics.csv` | 201,600 rows; matched-state branches, free trajectories, repeated patches, and single-position controls. |
| `phase1i_timeline_metrics.csv` | 50,760 rows; acquisition checkpoints and probe/patch losses. |
| `phase1i_true_acquisition.csv` | 6,800 true pre-update acquisition rows: 40 models × 170 updates. |
| `phase1i_checkpoint_provenance.csv` | 40 canonical checkpoint identities and replay results. |
| `phase1i_cue_encoding.csv` | Eight explicit cues and the coefficient audit. |
| `phase1i_seed_contrasts.csv`, `phase1i_statistical_summary.csv` | Existing seed-level contrasts and bootstrap intervals. |
| `PHASE1I_CONTEXT_COMPOSITION_REPORT.md`, `PHASE1I_manifest.json` | Frozen report, configuration, and source hashes. |

The same target rescues High Mixed, NoContext, and NovelMean to mean CE approximately **0.8086**, from 2.2483, 2.9400, and 2.8422. Wrong-position, orthogonal, and A/B endpoint controls are retained in the raw table. Identical CE under repeated full state replacement is expected because the output state is replaced at every transition; it does not establish cue invariance of natural dynamics.

The generic composition formula uses a zero-cue reference, but with the frozen coefficients .5/.5 the zero-reference term cancels: the target is the A/B hidden midpoint. Phrase the claim as state sufficiency and positional specificity, not necessity or natural state generation.

**Supporting source:** `phase1f_shared_backbone/` and `outputs/phase1f_shared_sequence_backbone/`. Selected sequence/backbone analyses may be retained. Its original Novel and NoContext follow identical effective inputs and must not supply independent genuine-Novel evidence.

**Implementation needed:** extract the final checkpoint, distinguish repeated patches (`patch_position == 0`) from acute single-position interventions, add common centered-logit summaries for all probes, and regenerate paired seed inference and two compact panels. There is no new manipulation to implement.

### Experiment 3: generation and maintenance are distinct limitations

**Maintenance implementation:** `phase1j_context_stability/`; entry point `scripts/run_phase1j_single_injection_context_stability.py`.

**Output root:** `analysis/phase1j_single_injection_context_stability/`.

- `phase1j_transition_metrics.csv`: 100,520 rows, preserving injection position, transition, lag, intervention, cue clamp, raw CE, rescue, and hidden/task midpoint residuals.
- `phase1j_seed_summary.csv`: 2,600 rows; these are multiple cells per seed, not 2,600 independent observations.
- `phase1j_primary_comparisons.csv`, `phase1j_statistical_summary.csv`, `phase1j_provenance.csv`, `phase1j_true_acquisition.csv`, report, and manifest are present.
- D0 is natural continuation; D1 is zero-cue continuation; D2 is physical Mixed continuation. For Mixed, D0 equals D2; for NoContext, D0 equals D1.

**Generation implementation:** `phase1k_cue_generation/`; entry point `scripts/run_phase1k_cue_only_composite_generation.py`.

**Output root:** `analysis/phase1k/`.

- `phase1k_seed_summary.csv`: 40 seed/family rows.
- `phase1k_local_cue_branches.csv`: 11,200 rows with identical-prestate cue comparisons.
- `phase1k_takeover_transition_metrics.csv` and `phase1k_pulse_transition_metrics.csv`: 17,640 rows each.
- `phase1k_takeover_summary.csv`: 10,080 rows preserving takeover position.
- `phase1k_endpoint_control_metrics.csv`, `phase1k_ab_to_mixed_convergence.csv`, `phase1k_natural_rollouts.csv`, Novel aggregation, provenance, acquisition, comparisons, bootstrap, report, and manifest are present.

The useful state persists differently under continued contextual input; cue takeover gives partial benefits, but the physical cue does not reliably reconstruct the midpoint from all relevant natural prestates. Avoid turning the frozen classification into the absolute claim that cues can never help.

**Implementation needed:** one maintenance panel with raw CE rescue and task-space drift over lag, one local-generation panel with endpoint controls, and a small pulse-versus-persistent takeover comparison. Define the weighting of injection/takeover positions before aggregation. Do not average lag rows as independent observations. Use raw effects as primary: normalized rescue fractions can be unstable or extreme, especially in Low networks near the CE floor.

All these rollouts use **teacher-forced tokens over seven transitions**. They do not demonstrate a token-autonomous sequence generator or long-run attractor.

### Experiment 4: prospective constructive test

**Implementation:** `phase1l_contrast_closure/`; entry point `scripts/run_phase1l_contrast_bottleneck_trajectory_closure.py`.

**Output root:** `analysis/phase1l/`.

| Artifact | Contents / size |
|---|---|
| `checkpoints/` | All 80 High checkpoints: C0–C3 × seeds 0–19. No Low cohort was run here. |
| `phase1l_seed_summary.csv` | 80 rows; sufficient to construct the requested paired geometry–CE scatter. |
| `phase1l_final_transition_metrics.csv` | 5,040 rows: 80 × seven transitions × eight cues plus NovelMean. |
| `phase1l_cue_projection_audit.csv` | All eight cue projections; the critical audit requested in the brief is present. |
| `phase1l_training_sequence_metrics.csv` | 13,600 true acquisition rows: 80 × 170. |
| `phase1l_geometry_metrics.csv`, `phase1l_neutral_rollout_metrics.csv` | Natural final geometry and recursively inherited neutral-state trajectory. |
| `phase1l_novel_within_seed.csv` | All four variants and their within-seed mean. |
| `phase1l_provenance.csv`, comparisons, bootstrap, report, manifest | Saved identities and matched-seed inference. |

C0 is ordinary sequence CE; C1 adds the contrast bottleneck; C2 adds trajectory closure; C3 combines both. Closure weight is fixed at one. The saved `family == High_monotonic` label is misleading here: no monotonic execution policy is trained. **C0's model tensors exactly equal `High_sequence_only_seed_N.pt` in all 20 seeds.** This is specified by the Phase 1L objective; preserve it as a distinct cohort, not as a suspected error to silently repair.

The audited cue coordinates are:

| Cue | Alpha |
|---|---:|
| A / B | +1 / −1 |
| Mixed / NoContext / Novel1 / Novel4 | 0 |
| Novel2 / Novel3 | +1.154701 / −1.154701 |

For each C1/C3 seed and transition, Mixed, NoContext, Novel1, and Novel4 have exactly identical saved CE and hidden/task midpoint residuals. Novel2 and Novel3 extrapolate beyond the familiar contrast endpoints; the four Novel variants are not four independent effective directions under the bottleneck.

The requested values are reproduced by the seed table:

| Measure | C0 | C3 |
|---|---:|---:|
| Familiar mean CE | 1.1979 | 1.1008 |
| Hidden A/B separation | 2.8566 | 3.3416 |
| Task A/B separation | 5.8016 | 6.2339 |
| Mixed CE | 2.2062 | 2.3332 |
| NoContext CE | 2.9558 | 2.3332 |
| NovelMean CE | 2.8425 | 1.9197 |
| Mixed task midpoint residual | 1.0134 | 0.8409 |
| NoContext task midpoint residual | 5.6397 | 0.8409 |
| NovelMean task midpoint residual | 4.2195 | 0.7844 |

**Training-firewall wording needs precision:** C2/C3 use an A/B-derived neutral closure rollout at `cbar = (cA+cB)/2`, which is numerically the physical Mixed cue. The protected probes' behavioral targets and outcomes are not used for training or selection. It would be inaccurate to say no computation at the Mixed cue occurs during training. This is the intended PRD design, including the explicit physical-midpoint instruction for C2.

**Implementation needed:** one natural C0–C3 comparison, common seed inference, the three paired geometry–behavior scatter panels, and a cue-coordinate explanation. No retraining is needed. The current runner can train when checkpoints/partials are missing, so the paper pass needs an explicit evaluation-only path.

## 3. Probe and metric contracts that must survive cleanup

Canonical task: eight-unit tanh Elman SRN, four token channels, three cue channels, sequence `(0,1,2,0,3,1,2,3)`, seven scored transitions, 170 online updates in 17 alternating blocks of ten, Adam .005, gradient clipping 1.0, High/Low initialization scales 2.0/.02.

Canonical newer cues are A `(1,0,0)`, B `(0,1,0)`, Mixed `(.5,.5,0)`, NoContext `(0,0,0)`, and Novel1–4 equal to `(1,1,1)`, `(1,-1,1)`, `(-1,1,1)`, `(-1,-1,1)` divided by √3. These are defined in `phase1i_context_composition/cue_encoding.py` and reused downstream.

The original third cue column is forced to zero in `ConfirmatoryElmanSRN`. The newer Novel cues are behaviorally nontrivial because they also engage the familiar channels. Describe them as held-out cue combinations; do not imply the network learned a response to a novel third channel.

Task space is centered output logits, not a fitted decoder, PCA coordinate, or average probability. Most late-phase midpoint residuals divide a per-transition distance by that transition's A/B separation before averaging. This is different from the ratio of two sequence-level averages. Low has small denominators; retain raw distances alongside normalized ones. For Phase 1L, distinguish improvement in absolute distance from improvement caused partly by greater A/B separation.

NovelMean is a within-seed average over all four registered variants. Many tables already contain both variants and NovelMean. Never aggregate all five labels together or count variants as independent seeds.

## 4. Human evidence located

The raw/cleaned inputs are present in `data/`, including `mixedBreak_clean.csv` and `threeContextsBreak_clean.csv`. No new human data collection is needed for the computational inventory.

| Analysis generation | Materials | Interpretation / status |
|---|---|---|
| Corrected retrieval, latest relevant probe analysis | `mixed_novel_retrieval_corrected/`; `configs/mixed_novel_retrieval_corrected.json`; `outputs/mixed_novel_retrieval_corrected/MIXED_NOVEL_RETRIEVAL_CORRECTED_REPORT.md` | Strong provisional anchor for Mixed/Novel/NoContext expression; includes absolute-fit checks, participant uncertainty, and full timeline tables. |
| Familiar state-space adjudication | `human_state_space_adjudication/`; corresponding config and output report | M3b wins forward prediction; key recovery gate fails. Does not justify a requirement for separate memories. |
| Earlier configured-state / low-rank story | `human_configured_state/`, `human_low_rank_interpretation/`; `outputs/human_configured_state_v3/`, `outputs/human_low_rank_interpretation_v31/` | These drive the current README and old final human runner; keep as historical results unless explicitly retained. |
| Historical R analyses and bridge | `analysis_SRN.Rmd`, `state-space_SRN.Rmd`, `state_space_bridge/`, `retrieval_modeling/`, `probe_modeling/` | Preserve equations, preprocessing ancestry, and documented equivalence limits. |

The corrected retrieval report selects Split familiar anchors, reports a first-analyzed-Mixed scalar estimate near .543, but gives endpoint mixture a higher predictive likelihood. Participant bootstrap uncertainty includes zero for the model contrast, and neither candidate passes every absolute-fit check. It explicitly does not require Novel or NoContext to be midpoints. Thus “the human result proves averaging” is not an established target from these files.

Useful current human tables include `full_experiment_fit_long.csv`, `first_mixed_model_comparison.csv`, `first_mixed_profile_predictions.csv`, `first_mixed_bootstrap_summary.csv`, `participant_bootstrap_R0_R1_summary.csv`, `novel_geometry.csv`, `novel_retrieval_metrics.csv`, and `no_context_probe_metrics.csv`. These are all catalogued in the JSON companion.

The final author-selected human Results artifact remains unconfirmed. The corrected retrieval analysis is a provisional anchor only; manuscript claims must retain its uncertainty and the adjudication recovery boundary.

## 5. Disposition of the remaining project

These are proposed narrative dispositions, **not permission to remove imported code**.

| Group | Existing directories/materials | Proposed role |
|---|---|---|
| Direct paper experiments | `confirmatory_srn`, `phase1i_context_composition`, `phase1j_context_stability`, `phase1k_cue_generation`, `phase1l_contrast_closure` | Active manuscript pipeline. |
| Supporting baseline and provenance | `srn_training_timeline_audit`, `srn_mixed_novel_diagnostics`, `srn_nonlinearity_compositionality`, `srn_validation` | Keep necessary primitives and audits; optional selected panels. |
| Phase 1F–H | `phase1f_shared_backbone`, `phase1g_shared_trajectory`, `phase1h_plasticity_audit` | F selected support, G candidate supplement, H separate/supplementary story. F/H helpers remain dependencies. G's categorical outcome must be checked against continuous effects; its report includes partial improvements with substantial splitting preserved. |
| Earlier retrofit/path work | `output_relevant_compositionality`, `phase1b_direct_path`, `phase1c_pareto_path`, `phase1d_constrained_retrofit`, `phase1e_manifold_following`, `final_task_space_synthesis`, `final_mechanistic_synthesis` | Historical scientific record, except extracted helper dependencies. `final_mechanistic_synthesis` is generated output despite its root location. |
| Older execution/causal story | `ftt_srn_execution_v2`, `ftt_srn_causal_representation`, `ftt_srn_context_sequence_interaction`, `execution_objective_search`, `execution_readout_diagnostic`, `commitment_rt` | Preserve task/objective helpers; archive old narrative and runs after dependency extraction. |
| Ecologies and transfer alternatives | `structured_context_ecology`, `ordinal_layout_ecology`, `human_svg_geometry_transfer`, `differentiation_transfer_frontier`, `emergent_retrieval_program`, `hierarchical_pretraining_ecology`, `structural_prior_curriculum`, `compare_pretraining_ecologies` | Historical/supporting; ordinal layout is required to reproduce historical acquisition matching. |
| Meta-learning and architecture programs | `episodic_meta_learning_capacity`, `srn_meta_learning_calibration`, `human_meta_prior_transfer`, `diagnose_rescue_meta_prior_transfer`, `context_controlled_recurrent_architectures`, `human_context_control_transfer`, `checkpoint_sweep_random_controls` | Archive as separate programs; do not rerun for this paper. Names containing “human” often denote model transfer tasks, not final human Results. |
| Earlier diagnostics and experiment program | `transfer_experiment_program`, `time_matched_probe_overlay`, `probe_prediction_execution`, `mechanism_probe_diagnostics`, `posthoc_transition_switch`, `probe_fit_validation` | Retain selected matching/provenance utilities; archive unrelated task/power/history analyses. |
| Human and shared data machinery | `mixed_novel_retrieval_corrected`, `human_state_space_adjudication`, `human_configured_state`, `human_low_rank_interpretation`, `state_space_bridge`, `retrieval_modeling`, `probe_modeling` | Keep declared human results and their actual dependencies; separate from model training. |
| Documentation and accumulated artifacts | root PRDs/scratchpads, `docs`, `configs`, `reproducibility`, `analysis`, `outputs`, `archive` | Index by purpose and provenance; separate immutable evidence from regenerated paper outputs. |

The package-level dependency scan reaches **21 packages** from the four late-phase packages. This is a conservative static closure over all package modules, not a claim that every function executes. Package `__init__.py` files also eagerly import pipelines. In particular:

```text
Phase 1I -> Phase 1F math + Phase 1H replay + timeline audit + canonical model
Phase 1J -> Phase 1I mechanics + Phase 1H replay + timeline audit
Phase 1K -> Phase 1J loading/references + Phase 1I mechanics + Phase 1H helpers
Phase 1L -> Phase 1I cues + Phase 1K geometry + canonical model + task/hash helpers
timeline audit -> older diagnostics and Phase 1B helpers
canonical package imports -> older execution/bridge/human packages
```

This is why moving 1B–1H or the legacy execution folders first would be unsafe. The JSON catalogue lists local imports for all 52 packages.

Storage is dominated by generated outputs: approximately 10.74 GB under `outputs/`, 0.56 GB under `analysis/`, and 1.38 GB under `archive/`. The context-controlled architecture outputs alone occupy approximately 5.88 GB, while the historical transfer program occupies 0.97 GB and Phase 1C outputs 0.85 GB. These are candidates for a checksummed research-artifact archive, not for unreviewed deletion. The active paper's small model checkpoints need not carry every exploratory run into the release.

## 6. Verification completed and remaining limits

- All **200 provenance references** matched on-disk checkpoint hashes: I 40/40, J 40/40, K 40/40, L 80/80. I/J/K reuse the same 40 files; there are 120 distinct primary checkpoint files across those phases.
- Each I–L manifest's own recorded source-file hash matches the current phase source. These hashes do not comprehensively cover imported dependencies; the new freeze manifest must do so.
- All 20 L C0 model states are bit-identical to saved High sequence-only states.
- Forward-only reevaluation of all 40 canonical models with all eight cues using the original NumPy evaluator reproduces saved transition CE to maximum error **3.56e−15**. All 80 L models with all eight cues using their original Torch evaluator reproduce saved CE to **1.78e−15**.
- I/J/K natural sequence CE tables agree to approximately 2.7e−15; recomputing L seed summaries agrees to 7.2e−15.
- Substituting L's Torch evaluation path for I–K's explicit NumPy recurrence produces a maximum transition CE difference of **3.90e−5**. This is an observed implementation-path difference, not evidence of different checkpoint weights. Choose and validate a common evaluator before calling regenerated values numerically identical; do not silently relax an exact-equality assertion.
- Existing tests cover interventions, seed nesting, cue firewalls, replay, output identity, and mandatory figures. The full test suite was not run: this task is inventory/planning, and several test fixtures train models. No end-to-end clean-environment reproduction has been established by this audit.
- The timeline audit documents earlier frozen-endpoint schedule plots mislabeled as acquisition. Keep those findings and use genuine pre-update acquisition logs for the manuscript.
- `docs/PAPER_FIGURE_DATA_GUIDE.md` already flags pooled-transition/pooled-observation uncertainty errors in older paper plotting. Reuse source data, not those error-bar calculations.

The inventory supports proceeding with the four-experiment paper-freeze work. The main missing deliverable is the unified, provenance-aware analysis layer described in the companion plan.
