# Dynamical generalization program — live scratchpad

## 0. Snapshot
- Start: 2026-11-04T03:27:28.232650+00:00; source commit 12408831ceac6ae0f9b3181c5a5a4094c6d46889; Python 3.11.8 / Torch 2.8.0 / NumPy 2.2.6 / pandas 2.3.2; repository-local frozen artifact roots.
- Reference paths and byte hashes: `scratchpads/dynamical_generalization_reference_index.json`.
- Required `(1)` PRD filenames are absent; authoritative identical-topic unsuffixed archived PRDs were located and indexed. No checkpoint substitution is authorized.
- Novel registry digest required: 2b28095562ea69a33d96c521817dcd0743f3d66658385ac425212f6951e00727. Existing paper contract remains immutable; new program has separate configuration and output identity.

## 1. Current scientific question
Distinguish static output compatibility, recurrent closure, propagation, independent cue information, and causal separation/closure interventions. Null: dynamics add no predictive value and interventions fail to dissociate processes. No human/RT fitting or architecture search.

## 2. Assumptions to verify in source
- [verified] confirmatory_srn/config.py: High=2.0, Low=.02, Adam=.005, clipping=1, 170 updates.
- [verified] Exact source initialization, forward/gradient recreation, and A-first A10/B10 ordering: tests/regression/test_historical_parity.py.
- [verified] 225 frozen source/input/checkpoint identities; every before/after audit matches. Exact nonlinear identity residual zero across 6,720 natural transition rows; 840 static rows obey affine equality/convexity.
- [verified] Original and all three proposed sequences occupy distinct relabeling/rotation classes; no replacements.

## 3. Decisions (append-only)
Pending prospective implementation conventions; machine-readable append-only trail is `scratchpads/decisions.jsonl`.

## 4. RED tests
Written and initially failed on missing modules; observed logs are in scratchpads/dynamical_red_tests.log. Scientific execution began only after the prospective implementation passed all relevant gates.

## 5. GREEN implementation
Core and staged pipeline implemented; final suite passes 42 tests. Logs: scratchpads/dynamical_green_tests.log and dynamical_final_tests.log.

## 6. REFACTOR / regression
Historical files are immutable. Changes will add independently identified modules and artifacts.

## 7. Run register
E1, E2, E3 pilot, E3 gated report, and E4 design-only report completed. Exact run identities/config/source hashes are recorded in results/dynamical_generalization/dg_v1_20261103/manifests/.

## 8. Results, including negative results
E1/E2 complete. E3 insufficient_familiar_support; E4 blocked_by_e3. See append-only numerical results and decision records below.

## 9. Open blockers and next decision
No unresolved implementation blockers. Scientific causal factorial/replication remain unestimated because every pilot candidate failed the prespecified familiar competence bound. All required stop-rule reporting is complete; no next model run is authorized within this registered program.

## 10. Final claim-to-evidence map
See results/dynamical_generalization/dg_v1_20261103/tables/claim_source_map.csv, per-stage manifests, and docs/dynamical_figure_captions.md.

### DG-D001 — prospective conventions
- New probes inspected? no. Float32 source-matched natural recurrence; float64 midpoint algebra/geometry and continuous Jacobians. Exact recurrence tolerance 2e-6; affine/convexity 1e-10; finite-difference Jacobian 2e-6. Numerical roundoff is measured separately.
- E3 pilot: full 48 candidate grid × five reserved seeds × four arms, cache algebraically identical objectives. Every arm requires endpoint mean CE ≤1.5, familiar mean ≤1.2, ≥4/5 improvement over initialization, mean accuracy ≥.35 and nonconstant predictions. Strong minus weak hidden separation ≥.25 and ratio ≥1.25 in each closure arm; on/off closure relative reduction ≥1%; finite nonzero objective gradients and numerical stability required. Feasible common candidate selected by ascending sum of λs, then λ_sep, λ_close, margin. No probes can enter this selection. E4 uses all 20 registered seeds per sequence if E3 feasible.
- CV: outer seed%5; inner seed%4 training seeds only; identical ridge grid [.01,.1,1,10], nested feature standardization, cohort-specific predictions.

### Core RED → GREEN group
- RED: four new unit modules failed collection with missing dynamics, cue_grid, separation_regularizers and replication imports.
- GREEN: 14 unit tests passed in 10.40 s after the smallest core implementations, including exact-vs-linear recurrence, affine convexity, finite-difference Jacobian, nested grouped CV, cue hash/alias geometry, gradient directions, stop-gradient closure, 170 changing-weight updates and sequence-cluster pairing.
- Next RED: integration collection fails on missing dynamics.pipeline; historical recreation/parity tests written before scientific execution. No scientific outcomes inspected.

### Full GREEN / pre-evaluation freeze (DG-D002)
- 36 unit/regression/integration tests passed in 6.82 s; initial broader cold-artifact run passed 35 tests in 344.48 s. Saved RED/GREEN logs in scratchpads/.
- All 120 source checkpoint hashes match. Historical source file scope exactly reproduces the original frozen paper source digest.
- New config hash: 7a8f75bcf2a818c31c6c3b405d45df9a8fe4143bf738e9b5a87336b85df3b62d; grid registry 4ba68e9ab576eb0f1c728c0eb3b364bab7efab8b3547304ddfb84f27f37bc6d4; sequence registry 1dce0ea07d91c0cf28a8b0b5af8a4a939cdfde70aa1ab680df3847f79ef387cb.
- All three proposed additional sequences are distinct equivalence classes; no replacement was necessary. No new scientific outcomes inspected.

### Run e1
- Config 7a8f75bcf2a818c31c6c3b405d45df9a8fe4143bf738e9b5a87336b85df3b62d; seeds registered in stage manifest; checkpoint/sequence identities retained; status {'implementation_status': 'validated', 'scientific_status': 'completed_mixed', 'evidence_status': 'completed', 'integrity_passed': True, 'optimizer_updates': 0, 'n_checkpoint_models': 120, 'n_transition_rows': 6720, 'max_exact_identity_error': 0.0}; output `/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/dynamical_generalization/dg_v1_20261103`. Source and checkpoint hashes identical before/after.

### Run e2
- Config 7a8f75bcf2a818c31c6c3b405d45df9a8fe4143bf738e9b5a87336b85df3b62d; seeds registered in stage manifest; checkpoint/sequence identities retained; status {'implementation_status': 'validated', 'scientific_status': 'completed_mixed', 'evidence_status': 'completed', 'integrity_passed': True, 'optimizer_updates': 0, 'n_grid_points_per_model': 35, 'n_anchors_per_model': 8, 'n_rays_per_model': 6, 'n_transition_rows': 41160}; output `/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/dynamical_generalization/dg_v1_20261103`. Source and checkpoint hashes identical before/after.

### Run e3_pilot
- Config 7a8f75bcf2a818c31c6c3b405d45df9a8fe4143bf738e9b5a87336b85df3b62d; seeds registered in stage manifest; checkpoint/sequence identities retained; status {'implementation_status': 'validated', 'scientific_status': 'insufficient_familiar_support', 'evidence_status': 'insufficient_familiar_support', 'integrity_passed': True, 'protected_probe_evaluations': 0, 'candidate_settings': 48, 'attempted_candidate_arm_seed_rows': 960, 'distinct_trained_objectives': 400, 'selected_settings': None}; output `/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/dynamical_generalization/dg_v1_20261103`. Source and checkpoint hashes identical before/after.

### DG-D003 — E3 pilot stop
- All 48 candidates × five reserved seeds × four arms recorded (960 rows), with 400 distinct objective-equivalent training runs. No candidate feasible; no selected settings. Criterion counts: {"finite": 48, "competence_passed": 0, "learning_passed": 48, "separation_passed": 12, "closure_passed": 30, "gradient_passed": 23, "feasible": 0}.
- Protected factorial probes inspected? no. No confirmatory training or coefficient expansion. E3 insufficient_familiar_support; E4 blocked_by_e3, retaining design-only sequence registry.

### Run e3
- Config 7a8f75bcf2a818c31c6c3b405d45df9a8fe4143bf738e9b5a87336b85df3b62d; seeds registered in stage manifest; checkpoint/sequence identities retained; status {'implementation_status': 'validated', 'scientific_status': 'insufficient_familiar_support', 'evidence_status': 'insufficient_familiar_support', 'integrity_passed': True, 'optimizer_updates': 0, 'confirmatory_runs': 0, 'protected_probe_evaluations': 0}; output `/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/dynamical_generalization/dg_v1_20261103`. Source and checkpoint hashes identical before/after.

### Run e4
- Config 7a8f75bcf2a818c31c6c3b405d45df9a8fe4143bf738e9b5a87336b85df3b62d; seeds registered in stage manifest; checkpoint/sequence identities retained; status {'implementation_status': 'validated', 'scientific_status': 'inconclusive', 'evidence_status': 'blocked_by_e3', 'integrity_passed': True, 'optimizer_updates': 0, 'n_sequences_run': 0, 'reason': 'No feasible frozen E3 familiar-only configuration; no replacement model or sequence-dependent calibration permitted.'}; output `/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/dynamical_generalization/dg_v1_20261103`. Source and checkpoint hashes identical before/after.

### DG-D004 — final results and stop
- E1: 120 models, 6,720 natural transitions, 840 static audits; exact recurrence residual zero. Dynamics-minus-static held-out MSE: [{"cohort":"canonical","mean":-0.1424586124,"ci_low":-0.2533531538,"ci_high":-0.0359523872},{"cohort":"constructive","mean":-0.1521305873,"ci_low":-0.2481300877,"ci_high":-0.0610370595}]. Incremental prediction is not causal mediation.
- E2: 41,160 natural grid/anchor/ray transitions; complete 35-point grid + eight anchors + six rays for all 120 models; structural projection invariants passed.
- E3 pilot: 48 finite candidates, 48 learning checks passed, 12 separation checks passed, 30 closure checks passed, 23 gradient checks passed, zero competence checks passed. All 960 candidate/arm/seed records retained, with 400 distinct 170-update training histories. No protected factorial probe evaluations or confirmatory model runs.
- E4: all four sequence definitions validated; three proposed additional sequences retained unchanged. No replication training; design-only report and explicit empty outcome tables.
- Final verification: 470 output hashes, 225 source/input/checkpoint identities, all historical bytes preserved, 42 tests passed. Four figure bundles (including infeasibility and blocked-design graphics) visually checked. Source implementation remains the prospectively committed version.
- Lightweight immutable report/manifest copies: artifacts/manifests/dynamical_generalization_v1/. Large checkpoints and transition data stay in the local versioned result root. Stop; no further model changes.
