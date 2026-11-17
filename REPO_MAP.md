# Agent repository map

Start here, then read `AGENTS.md`. This is a research repository with multiple sequential programs; later programs do not replace or pool earlier cohorts.

## Reading order

1. `README.md`: original four-experiment paper freeze and installation.
2. `docs/experiment_map.md`: A1–A8 experiment-to-code crosswalk.
3. `docs/FINAL_EXPERIMENT_INFRASTRUCTURE.md` and its JSON inventory: historical source catalogue.
4. `PRD_SRN_CONTEXT_FAILURE_SELECTIVE_REPAIR.md`: latest fixed scientific protocol.
5. `results/context_failure_selective_repair/v1_1/FINAL_CONTEXT_FAILURE_SELECTIVE_REPAIR_REPORT.md`: latest authoritative result, including negative/infeasible outcomes.
6. Its `RESEARCHER_HANDOFF.md`, `CLAIM_LEDGER.md`, `scratchpad.md`, and `manifests/sealed_manifest.json`: equations, claims, decisions and exact hashes.
7. `PUBLICATION.md`: GitHub versus release artifact coverage, restoration and verification.

## Scientific programs

| Program | Implementation | Tests | Evidence directory | Interpretation |
| --- | --- | --- | --- | --- |
| Paper freeze A1–A8 | `src/context_sequence/`, `scripts/`, `configs/` | `tests/unit`, `tests/regression`, `tests/integration` | `results/paper/v2_20260925_novelty30_60_r2/` | Canonical High/Low and constructive C0–C3 remain separate |
| Dynamical generalization E1–E4 | `src/context_sequence/dynamics/`; `docs/dynamical_generalization.md` lists runners | Search tests for `dynamical` and `e1`/`e2` | `results/dynamical_generalization/dg_v1_20261103/` | E3 familiar gate failed; E4 not run |
| Output-compatible trajectories P0–P5 | `experiments/output_compatible_trajectories/` | `tests/output_compatible_trajectories/` | `results/output_compatible_trajectories/v1_20261111/` | P3 states are target-informed oracle existence tests, not natural retrieval |
| Context cue access S0–S4 | `experiments/context_cue_access/` | `tests/context_cue_access/` | `results/context_cue_access/v1/` | Physical cue accessibility, preparation and local input control |
| Context mixing D0–D4 | `experiments/context_mixing_sequence_continuity/` | `tests/context_mixing_sequence_continuity/` | `results/context_mixing_sequence_continuity/v1/` | Historical D3 switch training failed to give a clean repair; do not rerun training |
| Context failure R0–R5 | `experiments/context_failure_selective_repair/` | `tests/context_failure_selective_repair/` | `results/context_failure_selective_repair/v1_1/` | Selective mechanism not identified; matched feasibility limited primary coverage to one seed |

## Latest program: file-level navigation

| File | Responsibility |
| --- | --- |
| `core.py` | Frozen constants, source adapters, seed bootstrap, stage dependencies and immutable manifests |
| `audit.py` | Checkpoint/cue/indexing fidelity; reuse registry for all 360 state × cue cells |
| `factorial.py` | R1 reuses exact P4 pulse rows; separates current and future CE |
| `geometry.py` | Full-right output SVD; familiar-only future-logit Jacobians; fixed U2/W3 geometry |
| `repair.py` | Natural-gap patches, equal-norm controls, common geometric amplitude, fixed reverse sign |
| `decision.py` | R4 design-only no-training recommendation and claim restrictions |
| `reporting.py` | Independent table arithmetic, source-to-claim ledger, report and researcher handoff |
| `figures.py` | Five deterministic PNG/SVG figures from saved CSVs only |
| `verify.py` | Stage/input/output hashes and complete scientific file-set seal |
| `regenerate.py` | Table-only reproduction of 20 report products with identical hashes |

The latest program's `core.py` retains a historical default output path ending in `v1`. Always pass `--output results/context_failure_selective_repair/v1_1` for verification; use a new directory for any separately authorized execution. Initial R2 in `v1` is superseded, not a second independent scientific result.

## Actual model and data flow

`confirmatory_srn/model.py` defines the original tanh SRN and separate execution policy. `src/context_sequence/evaluation/rollout.py:Model.step` is the frozen float32 evaluator. `src/context_sequence/provenance/` validates registry identities and hashes. The latest adapters use `experiments/context_mixing_sequence_continuity/source_adapter.py` and predecessor loaders; do not replace them with a new SRN implementation.

Original High checkpoints: `outputs/confirmatory_srn_final/High_monotonic_seed_*.pt`, selected by the registry at update170, seeds0–19. Exact paths and hashes are in latest `tables/r0_checkpoints.csv`. Primary tokens are seven teacher-forced inputs; physical Mixed=(.5,.5,0). Low, C0–C3, later architectures and D3 endpoints are separate cohorts.

Latest evidence: `tables/r0_reuse_registry.csv` → `r1_branch_rollouts.csv`/`r1_cells.csv` → `r2_bases.csv`/`r2_future_sensitivity.csv`/`r2_natural_gap_components.csv` → `r3_interventions.csv`/`r3_reverse.csv`/`r3_seed_contrasts.csv` → `r5_claim_sources.csv` and reports. Primary figures live in `figures/`; RED/GREEN/REFACTOR evidence in `audit/`. Never infer independent sample size from transitions or reference rows: the trained seed is the unit.

## Historical directories

`phase1i_context_composition/`, `phase1j_context_stability/`, `phase1k_cue_generation/`, and `phase1l_contrast_closure/` are historical mechanistic branches. `checkpoint_sweep_random_controls/`, `human_meta_prior_transfer/`, `differentiation_transfer_frontier/`, and `transfer_experiment_program/` concern earlier Mixed/prior-experience/transfer studies; inspect their own protocols before comparing architectures. `archive/artifact_index.json` maps preserved old documents/packages. `data/README.md` describes historical human data; raw human data is not part of this publication.

## Working commands

Python3.11; dependencies: `pyproject.toml`, `requirements.lock`. From repository root:

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
PYTHONPATH=.:src pytest -q tests/context_failure_selective_repair
# After restoring the release evidence bundle:
PYTHONPATH=.:src python -m experiments.context_failure_selective_repair.verify --output results/context_failure_selective_repair/v1_1
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.context_failure_selective_repair.regenerate --source results/context_failure_selective_repair/v1_1 --destination /tmp/context_failure_regenerated
```

The final destination must be empty. Full tests/reproduction require predecessor evidence from the release bundle. Historical Phase1K import checks were blocked in the recorded environment; do not report them as passing. Acquisition tests18/19 train models and are excluded from read-only validation.
