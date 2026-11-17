# R0 — actual source and reuse audit

All20 canonical High_monotonic seeds0–19, original170, are SHA-verified. Eight tanh coordinates, four one-hot tokens, four-class sequence head, three raw context channels; third input column exactly zero. Mixed=(.5,.5,0), not the sum (1,1,0). Source evaluator `src/context_sequence/evaluation/rollout.py:Model.step` executes float32 affine/tanh/head. It folds the original two recurrent biases in float32; prior D0 established Torch/NumPy source tolerances. The execution policy is separate and not scored.

| cue | raw_ce |
| --- | --- |
| A | 0.91728 |
| B | 1.4558 |
| Mixed | 2.2483 |

All360 cells/1440 transition rows exactly reused from P4 named A/B/Mixed pulse branches, never native_source/target_native aliases or persistent schedules. Zero new factorial scientific evaluations. Fidelity only:60 natural full rollouts plus16 representative nonnative tails; every reused (M,M) baseline and familiar same-cue current output verified at1e-12 CSV roundoff. All prestate hashes verified. Every future consumes the same remaining teacher-forced tokens with physical Mixed.

Actual source leads: `results/output_compatible_trajectories/v1_20261111/reports/{FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT,P4_RESULTS,P2_RESULTS}.md` and its tables/manifests; `analysis/phase1k/PHASE1K_CUE_GENERATION_REPORT.md` plus `phase1k_cue_generation/interventions.py`; `results/context_cue_access/v1/{FINAL_CONTEXT_CUE_ACCESS_REPORT,S2_CUE_HISTORY_RESULTS,S3_LOCAL_CONTROL_RESULTS}.md`; `results/context_mixing_sequence_continuity/v1/{FINAL_CONTEXT_MIXING_SEQUENCE_REPORT,D0_SOURCE_AUDIT,D1_MIXING_BREAKDOWN}.md`. All available scientific predecessor seals verified. Phase1K CSVs are cloud placeholders, no observed hash invented and no substitute geometric CE. P4 contains exact locally available required outcomes. S3 cue Jacobians/CE gradients do not supply R2 state-to-logit label-free Jacobians. D1 e=j+r localized a nonlinear discrepancy but its correction was inconclusive; it is not repeated here. P2 full-right SVD and all-four-probability/roundoff audit are imported unchanged.

Criterion imported unchanged: future positions k+1…7, familiar bound +.10; equality passes. Current k excluded. k1 has no history effect. Two optional legacy files are unavailable, not blockers because exact required P4 evidence exists. Scratchpad template unavailable; PRD sections used transparently.
