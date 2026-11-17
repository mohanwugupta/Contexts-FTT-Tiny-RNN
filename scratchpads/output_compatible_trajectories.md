# Output-compatible trajectories — live scratchpad

## A. Snapshot and scientific claim
- Current git commit: ef43aedaad2009cd168801faa0cfa7150c271d88
- Date/time / Python / torch / device / dtype: 2026-11-11T21:23:56.014751+00:00 / 3.11.8 / verify torch in P0 / CPU / source float32, geometry float64
- Source checkpoint root: outputs/confirmatory_srn_final/confirmatory_checkpoints and analysis/phase1l/checkpoints (verify bytes).
- Frozen model IDs / seeds: primary High monotonic 0–19; supplementary C0–C3 sequence-only 0–19; no cohort substitution.
- Frozen cue registry hash: require 2b28095562ea69a33d96c521817dcd0743f3d66658385ac425212f6951e00727.
- Current question: can an output-equivalent state far from the midpoint support viable unpatched future predictions, and can tested natural cue routes access such a state?
- Hypothesis that could be falsified: exact midpoint necessity for this frozen model/cue/horizon. Finite oracle failure cannot establish necessity.

## B. Source and implementation audit
| claim or assumption | source file:function:line | verified? | discrepancy |
|---|---|---|---|
| Historical recurrence, readout and cues | tables/p0_source_function_index.csv; p0_state_dict_keys.csv; p0_output_spaces.csv | verified against 120 checkpoints | Phase 1J/1K evaluation.py aliases resolved to core.py/interventions.py; two optional CSVs unavailable |

## C. Active stage and gate
- Stage P0/P1/P2/P3/P4/P5: P5 complete; six stages passed integrity.
- Prerequisites passed: all source, replay, mathematical, oracle and cue-branch gates; final 59-test suite; final seal verified 176 files.
- Known blockers: two optional historical cloud-only CSVs; exact expected hashes and complete local E1 replacements documented. No required source/checkpoint gate blocked.
- Is any future-target oracle optimization allowed here? completed only in P3, using the fixed hidden-coordinate budget. No additional search authorized or executed.

## D. RED → GREEN → REFACTOR log (append-only)
| ID | proposed contract | RED failure / negative fixture | GREEN command+result | refactor regression | evidence file |
|---|---|---|---|---|---|

## E. Decisions (append-only)
- No new intervention outcomes inspected. Freeze budgets, tolerances, geometric edge handling and oracle starts before P2/P3.

## F. Run ledger
| stage | cohort | seed | cue | k | direction/restart | status | hash/output |
|---|---|---|---|---|---|---|---|

## G. Findings, including negative and blocked results
- Final: output-compatible far viable states found in some frozen High seeds; exact midpoint necessity falsified for those cases. Some physical cue branches were viable, but no k1 pulse route met far+viable+exact current-output compatibility. Infeasible, null, search-limited and optional blocked outcomes retained.

## H. Researcher teach-back
- Algorithm and teach-back: reports/RESEARCHER_HANDOFF.md, including a complete numerical example and full probability traces. P W_o N=0 preserves every present softmax probability; the frozen recurrence can turn currently invisible coordinates into later visible differences. A future-informed state oracle is not natural retrieval.
- Why same current softmax? Explain using P W_o N = 0: a kernel displacement changes logits only by a common additive offset.
- Why can future differ? Explain using recurrent state update: hidden recurrence may transform currently invisible coordinates into output-relevant ones later.
- What did an oracle know that the natural network did not?: future target labels and its optimization objective.
- One experimental falsifier: a far, output-equivalent state that passes full remaining-horizon viability while the midpoint fails under identical future inputs.

## I. Open questions and next permitted action
- Current gate status: all P0–P5 gates complete; final manifest sealed and verified.
- Next permitted task: researcher inspection of the final report, handoff, primary figures and exact source-to-claim map. No new training or adaptive search is part of this program.
- Conditions that require researcher review: missing/corrupt source, irreconcilable replay or an intervention invariant that cannot be repaired without changing the scientific contract.

### OCT-D001 — prospective design and core tests
- Initial RED: four missing core modules. GREEN: five initial core tests passed. Additional baseline checks demonstrated rejection of permuted target timelines, checkpoint identities, visible directions, naive clipping, corrupt manifest bytes and duplicate rows.
- A dedicated stream-schema RED test failed because column shifts were silently accepted; the minimal fix locks column names and order.
- Batch-export RED exposed only a 5.55e-17 derived-norm reduction difference; state coordinates are now asserted bit-exact, norm within 1e-12. Numerical equivalence tolerances and all scientific thresholds remain fixed.
- Initial broad legacy suite included one disposable old training fixture; no saved checkpoint or historical artifact changed. Read-only subset subsequently passed 37 tests. No old scientific E3/E4 execution resumed. All further commands exclude model-training fixtures.
- Primary P3/P4: 20 High monotonic seeds; supplementary C0–C3 P1/P2 only. Oracle is explicitly target-informed hidden-coordinate optimization, with no trained weights requiring gradients or optimizer construction. Fixed budgets are in config.py and the decision JSONL.

### OCT-D002 — unavailable legacy CSV, complete E1 source retained
- Phase 1J transition CSV is compressed/dataless; materialization reads did not complete even outside the sandbox and were interrupted. Expected SHA256 b0205f6bc82d2b5ce3db29f078538fb921984526671988810d6a8677e40fe511 is recorded from the immutable pre-freeze snapshot; no observed digest is invented.
- P0 had not created a started manifest or scientific output before the blocked read. The optional legacy table audit is explicitly blocked_missing_source_optional. Required E1 source, checkpoint bytes and exact recurrence remain available, so source/replay gates continue with E1. A negative fixture proves a cloud placeholder is not read or assigned a fabricated digest.

### OCT-D003 — second optional legacy reference
- The older A3 seed-summary CSV is also dataless, expected SHA d1648d53d60b9270df64bf9d2ae2480eaf0985ac586b9410cc0c9543ea21aab6. It is recorded unavailable, not assigned observed bytes. The local E1 static affine/Jensen audit contains all seed-level per-transition midpoint CE and is verified against its frozen manifest before direct source replay. No scientific comparison or budget changed.

### P0 full completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'models_replayed': 120, 'model_optimizer_updates': 0, 'source_alias_resolution': 'Phase 1J core.py and Phase 1K interventions.py; evaluation.py aliases were absent'}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111; manifest: manifests/p0.json; all input hashes unchanged.

### P0/P1 test proof and geometry discovery
- P0 full replay gate passed for all 120 checkpoints. All centered heads have rank 3 and nullity 5. 612/840 familiar midpoints satisfy the fixed oracle interior; the remaining starts stay explicitly flagged under the previously frozen infeasibility policy. No clipping or threshold relaxation.
- P1 RED: missing lag_audit implementation. GREEN: 13 new-program tests pass, covering identical-current distribution, six/four/two future steps, exact rescue subtraction, native/zero/Mixed aliases, position support 7−lag, and nested Novel direction averaging. Legacy read-only suite previously passed 39 checks before P1 additions.

### P1 full completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'n_models': 100, 'n_lag_rows': 50400, 'model_optimizer_updates': 0, 'native_rows_reused_from_E1': True, 'legacy_optional_tables_status': 'unavailable; complete E1 source used'}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111; manifest: manifests/p1.json; all input hashes unchanged.

### P2 RED → GREEN proof
- RED: nullspace intervention module missing. GREEN: source-current precision, rank/kernel/box, fixed counts, matched norms, memoryless zero effect, hidden-to-visible future divergence and malformed-state-before-future tests pass.
- Additional RED: partial raw Novel direction was aggregated prematurely; fixed by waiting for the complete seed matrix and using strict two-direction sums with missing-value propagation. No scientific thresholds changed.
- Original float32 head rounding is audited separately with a priori arithmetic bounds; exact centered-map and all four probabilities remain checked at float64 tolerance 1e-10. No P2/P3 scientific outcomes inspected before this freeze.

### P2 smoke completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'n_models': 20, 'planned_states': 1280, 'future_rows': 7131, 'max_current_softmax_difference_float64': 9.43689570931383e-16, 'max_current_softmax_difference_float32': 2.315965670907083e-07, 'model_optimizer_updates': 0}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111/smoke; manifest: manifests/p2.json; all input hashes unchanged.

### P2 full completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'n_models': 100, 'planned_states': 432000, 'future_rows': 1329061, 'max_current_softmax_difference_float64': 1.0269562977782698e-14, 'max_current_softmax_difference_float32': 5.691553492948032e-07, 'model_optimizer_updates': 0}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111; manifest: manifests/p2.json; all input hashes unchanged.

### P3 RED → GREEN and oracle-only authorization
- RED: staged oracle tests failed missing viability module before implementation; identical staged tests copied into the repository. GREEN: 23 new-program tests now pass, including finite differences, all iterate constraints, fixed budgets, known-solution recovery, no-influence null, no-feasible-state retention, deterministic repeats and A/B-only success bounds excluding the present output.
- Gradients are taken only with respect to kernel coordinates. Source matrices are copied constant tensors with requires_grad=False; no nn.Parameter or model optimizer is created. Adam coordinate updates are implemented directly.
- P2 full counts reconcile: 432,000 planned states and 1,329,061 transition rows, max float64 current-softmax residual 1.027e-14 and original float32 rounding difference 5.692e-7. All old bytes unchanged. P3 settings remain exactly fixed.

### P3 smoke completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'existence_outcome': 'search_limited_inconclusive', 'n_models': 4, 'planned_restarts': 64, 'feasible_restarts': 44, 'restart_history_rows': 704, 'far_viable_midpoint_failed_restart_endpoints': 0, 'hidden_coordinate_updates': 440, 'model_optimizer_updates': 0, 'oracle_future_target_information': True}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111/smoke; manifest: manifests/p3.json; all input hashes unchanged.

### P3 full completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'existence_outcome': 'far_viable_alternative_found', 'n_models': 20, 'planned_restarts': 6480, 'feasible_restarts': 4908, 'restart_history_rows': 978480, 'far_viable_midpoint_failed_restart_endpoints': 154, 'hidden_coordinate_updates': 736200, 'model_optimizer_updates': 0, 'oracle_future_target_information': True}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111; manifest: manifests/p3.json; all input hashes unchanged.

### P4 RED before implementation
- P3 full completed unchanged: 6,480 restart slots, 4,908 feasible, 978,480 history rows, 154 far+viable endpoints with failed midpoint across all registered positions/cues. Model updates zero. Smoke negative retained.
- P4 RED: missing cue_access module, tests in scratchpads/output_compatible_p4_red.log (also staged outside repo during P3). Tests specify bit-identical prestate/token, current/future cue schedule, bit-exact native replay, frozen cue vectors, oracle exclusion, missing reference NA.
- Implementing exactly the frozen eight-source, six-branch pulse/persistent menu. Oracle-reference traces exported separately to prevent accidental cue pooling. No scientific threshold or budget changed.

### P4 smoke completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'n_models': 4, 'cue_only_branch_cases': 1536, 'oracle_reference_cases': 32, 'transition_rows': 9216, 'model_optimizer_updates': 0, 'cue_generator_future_target_optimization': False}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111/smoke; manifest: manifests/p4.json; all input hashes unchanged.

### P4 full completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'n_models': 20, 'cue_only_branch_cases': 34560, 'oracle_reference_cases': 720, 'transition_rows': 172800, 'model_optimizer_updates': 0, 'cue_generator_future_target_optimization': False}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111; manifest: manifests/p4.json; all input hashes unchanged.

### P4 GREEN / P5 RED
- P4 smoke and full passed all source parity and identity checks. Full 34,560 cue-only branch cases and 720 separately exported state-oracle cases. The main long table excludes oracle rows by construction. Full model updates zero.
- P5 RED specifies preserved inconclusive/negative decision labels, duplicate-branch rejection for seed counts (T45), and hash-seal tamper/missing-file rejection. Failure: missing analysis module; scratchpads/output_compatible_p5_red.log.
- Synthesis will additionally calculate complete current-softmax distance of saved cue-generated states to midpoint. This is a diagnostic using the already fixed 1e-10 criterion, not another cue or state search. It prevents equating viable cue behavior with exact output-compatible access. No threshold, menu, cohort or budget changes.

### P5 GREEN / final refactor verification
- 59 tests passed (29 program, 30 historical read-only). Tests exclude prior training fixtures. Code identity and full command recorded in audit/final_test_result.json.
- A read-only synthesis preview on symlinks to completed tables passed; reports and figures were visually reviewed. No extra intervention/search was run. Full synthesis now follows unchanged scientific contract.
- k1 Mixed seed5 restart1: far/viable with initial r=.315152; future CE .203992 vs midpoint1.610238 and familiar .207772. No tested pulse cue succeeded for this case (best CE1.566451).
- Saved cue-state diagnostic found no k1 far+viable exact-current-output-compatible route; some cue routes were viable with changed present outputs.

### P5 full completed
- {'integrity_passed': True, 'implementation_status': 'validated', 'scientific_status': 'completed', 'model_optimizer_updates': 0, 'n_primary_seeds': 20, 'final_reports_written': True, 'manifest_seal_required': True}
- Output: /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/results/output_compatible_trajectories/v1_20261111; manifest: manifests/p5.json; all input hashes unchanged.

### Final handoff / seal
- All six stages completed; final seal verified 176 files and unchanged scientific model weights. Proof: scratchpads/output_compatible_final_seal.log.
- Required reports, figures (PNG/SVG/PDF), run registry, claim map, test evidence, decision history and all bulk CSV outcomes are saved under results/output_compatible_trajectories/v1_20261111.
- Final metadata adds only the verification command and copies test logs into the audit bundle before sealing; scientific outputs and frozen implementation files were not edited after their stages.
- No new training experiment is proposed or launched. The original dynamical E3/E4 remains untouched.
