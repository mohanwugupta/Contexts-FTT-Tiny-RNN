# Result map

This map links the paper narrative to one generating command, frozen config,
source data, statistical result, and paper-facing figure. Versioned analysis
figures remain available for diagnostics and Supplement construction.

## Figure 1 — conceptual model and hypotheses

| Field | Provenance |
|---|---|
| Scientific claim | Context can configure the currently active state of an identical learned sequence; SRNs provide a mechanism, humans provide the behavioral test. |
| Generating script | `scripts/make_paper_figures.py` |
| Configuration | `configs/srn_final.json`, `configs/human_final.json` |
| Inputs | No numerical input; the diagram encodes the prespecified analysis order and inferential boundary. |
| Output figure | `outputs/final/figures/figure_01_conceptual_model.png` |
| Statistical test | None. |
| Status | Conceptual / confirmatory framing. |

## Figure 2 — SRN splitting and execution behavior

| Field | Provenance |
|---|---|
| Scientific claim | High-scale networks split A/B hidden states; the execution objective grounds learned confidence in expression; frozen Mixed/Novel probes are graded. Low scale is the negative control. |
| Generating scripts | `scripts/run_srn_final.py`; `scripts/make_paper_figures.py` |
| Configuration | `configs/srn_final.json` |
| Inputs | Fixed sequence and synthetic alternating A10/B10 curriculum. |
| Output CSVs | `outputs/ftt_srn_execution_v2_final_checkpointed/splitting_metrics.csv`; `probe_interpolation.csv`; `execution_geometry.csv` |
| Output figure | `outputs/final/figures/figure_02_srn_splitting_execution.png` |
| Statistical test | Registered G1–G5 splitting gate and E1–E6 execution-grounding gate; within-network endpoint interpolation. |
| Status | Confirmatory mechanism/existence proof; Low and High-sequence-only are controls. |

## Figure 3 — causal configured-state interventions

| Field | Provenance |
|---|---|
| Scientific claim | Naturally occurring sequence-position-matched A/B components causally affect execution more than mismatched or orthogonal controls, while the full strict necessity chain remains limited. |
| Generating scripts | `scripts/run_srn_final.py`; `scripts/make_paper_figures.py` |
| Configuration | `configs/srn_final.json` |
| Inputs | Checksum-validated frozen High-scale checkpoints; pure natural A/B estimation passes only. |
| Output CSVs | `outputs/ftt_srn_context_sequence_v23_final/token_structured_causal_effects.csv`; `interaction_mechanism_summary.csv`; `outputs/ftt_srn_causal_v22_final/ablation_seed_summary.csv` |
| Output figure | `outputs/final/figures/figure_03_causal_configured_state.png` |
| Statistical test | 20-seed registered 80% criterion; exhaustive wrong-token comparison; random orthogonal specificity; target and sequence-preservation diagnostics. |
| Status | Confirmatory causal specificity plus negative-control/failed-necessity boundary. |

## Figure 4 — human structured retrieval

| Field | Provenance |
|---|---|
| Scientific claim | Matched-position A/B effects vary across the seven transitions; Mixed and Novel deviations are graded and structured; M3 low rank predicts held-out humans best. |
| Generating scripts | `scripts/run_human_final.py`; `scripts/make_paper_figures.py` |
| Configuration | `configs/human_final.json` |
| Inputs | `data/mixedBreak_clean.csv`; `data/threeContextsBreak_clean.csv`; frozen v2.1 checkpoints for M4 only. |
| Output CSVs | `outputs/human_configured_state_v3/context_position_model.csv`; `pure_ab_token_profiles.csv`; `mixed_midpoint_deviation.csv`; `novel_midpoint_deviation.csv`; `mixed_model_comparison.csv`; `novel_model_comparison.csv` |
| Output figure | `outputs/final/figures/figure_04_human_structured_retrieval.png` |
| Statistical test | Participant-effects Context × Position test with clustered SE; participant-held-out likelihood for M0–M4; model recovery. |
| Status | Confirmatory human result. |

## Figure 5 — low-rank human modeling and human/SRN comparison

| Field | Provenance |
|---|---|
| Scientific claim | The reproducible human solution is rank 2, but semantic interpretation is limited and the exact frozen SRN transition geometry does not match humans. |
| Generating scripts | `scripts/run_human_final.py`; `scripts/make_paper_figures.py` |
| Configuration | `configs/human_final.json` |
| Inputs | Frozen v3 pure A/B and probe tables; frozen all-seed SRN profile; raw data only for preregistered transition features. |
| Output CSVs | `outputs/human_low_rank_interpretation_v31/dimension_selection.csv`; `component_loadings.csv`; `component_stability.csv`; `cross_dataset_subspace_similarity.csv`; `human_srn_subspace_similarity.csv`; `final_dimension_interpretation_summary.csv` |
| Output figure | `outputs/final/figures/figure_05_low_rank_human_srn.png` |
| Statistical test | Participant-held-out reconstruction/probe likelihood with one-SE rank selection; 1,000 participant bootstraps; exact transition-label permutations; principal angles/canonical correlations. |
| Status | Confirmatory rank selection and negative human–SRN comparison; semantic labels are limited/exploratory. |

## Supplement and negative results

| Analysis | Recommended placement | Reason |
|---|---|---|
| Full v2.1 learning curves, calibration, transfer, context weights, High-vs-Low and sequence-only comparisons | Supplement | Establishes splitting and execution controls without crowding Figure 2. |
| Original task-trained SRN averaging failure (`archive/exploratory/legacy_task_srn/`, `archive/old_outputs/probe_analysis_combined_final/`) | Supplement / negative control | Shows why the final SRN is an existence proof rather than a post-hoc exact human model. |
| v2.2 steering sweeps, off-manifold diagnostics, random and sequence controls, checkpoint dependence | Supplement | Full causal validation and strict necessity failure. |
| v2.3 shared/specific decomposition, natural-patch identity, model comparison, Mixed/Novel generalization | Supplement | Robustness and M5/M6 inferential boundary. |
| Human preprocessing, counterbalance sensitivity, continuous-vs-categorical retrieval, temporal probes, recovery matrix | Supplement | Behavioral validity and model identifiability. |
| Component bootstraps, exact feature permutations, difficulty residualization, shared/separate probes, semantic recovery | Supplement / exploratory | Makes the limited semantic interpretation explicit. |
| Archived GRU, partial-observability, Bayesian/gate, and environmental-pressure searches | Historical only unless specifically cited | Hypothesis-development record; not part of the final paper pipeline. |

