# Experiment-to-infrastructure map

| Experiment | Paper job | Historical infrastructure | Current implementation and output |
| --- | --- | --- | --- |
| 1: phenotype | A1 | `confirmatory_srn/`, Phase 1I acquisition/provenance, canonical 1I–1K checkpoints | `src/context_sequence/experiments/engine.py`; `a1_canonical_phenotype_*`, Figure 1 |
| 1: acquisition robustness | A2 | Phase 1I timeline; historical transfer program only as separate-cohort context | `experiments/matching.py`; `a2_acquisition_*`, support audit, supplement |
| 2: shared-state availability | A3 | `phase1i_context_composition/` checkpoint and intervention logic | Common evaluator and `a3_composite_sufficiency_*`, Figure 2 |
| 3: state maintenance | A4 | `phase1j_context_stability/` D0/D1/D2 design | `a4_maintenance_*`, Figure 3 |
| 3: physical-cue generation | A5 | `phase1k_cue_generation/` branch/takeover design | `a5_cue_generation_*`, Figure 3 |
| 4: constructive generalization | A6 | `phase1l_contrast_closure/` C0–C3 checkpoints | `a6_constructive_*`, Figure 4 |
| 4: geometry–behavior | A7 | Phase 1L paired geometry/behavior summaries | `a7_geometry_behavior.csv`, Figure 4 |
| 4: effective cue | A8 | Phase 1L projection and contrast model | `a8_effective_cue_audit.csv`, Figure 4 |

The earlier project-wide catalogue is [FINAL_EXPERIMENT_INFRASTRUCTURE.md](FINAL_EXPERIMENT_INFRASTRUCTURE.md), with machine-readable [inventory](FINAL_EXPERIMENT_INVENTORY.json). Historical code and reports remain in place or in the [archive crosswalk](../archive/artifact_index.json). The new package uses independent task, evaluation, geometry, statistics, provenance, and paper modules and does not import exploratory packages for evaluation.
