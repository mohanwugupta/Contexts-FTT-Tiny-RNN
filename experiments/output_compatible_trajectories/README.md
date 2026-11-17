# Output-compatible recurrent trajectories

Evaluation of frozen scientific models under PRD_OUTPUT_COMPATIBLE_RNN_TRAJECTORIES.md. No scientific network optimizer or training is implemented. P3 explicitly uses future targets to optimize initial hidden kernel coordinates; it is an existence oracle.

The complete run is in `results/output_compatible_trajectories/v1_20261111/`. Start with `reports/FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md` and `reports/RESEARCHER_HANDOFF.md`. Every stage is immutable and verifies its prerequisites and checkpoint hashes. The two optional unavailable legacy tables are documented in P0; no checkpoint identity was substituted.

Reproduce in a fresh output directory, setting `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

1. `python3 -m experiments.output_compatible_trajectories.cli --stage p0 --eval-only --output NEW_DIRECTORY`
2. Repeat for `p1`, then `p2`.
3. Run `p3 --smoke` before full `p3` (same remaining arguments).
4. Run `p4`, then `p5`.
5. `python3 -m experiments.output_compatible_trajectories.verify --output NEW_DIRECTORY --seal`

Verify the existing run by omitting `--seal`. Sealing is exclusive and cannot overwrite a previous manifest. The final manifest hashes all outputs, including negative, blocked and infeasible records; local bulk CSVs are intentionally not versioned in Git.

Use the read-only test suite recorded in `audit/final_test_result.json`. The tests cover source replay, nullspace invariance of all four current probabilities, intervention placement, same-input continuations, no-influence controls, bounded oracle gradients and fixed budgets, exact native cue replay, oracle/cue separation, seed-level aggregation and tamper detection. `scratchpads/output_compatible_trajectories.md` and RED/GREEN evidence logs preserve development order.

The source recurrence runs in float32. Nullspace geometry and differentiable oracle objectives use float64; every final selected state is evaluated through the exact source recurrence. Complete-softmax invariance is checked at 1e-10 in the mathematical head, with independently declared float32 arithmetic bounds. No hidden state is clipped to satisfy a constraint.
