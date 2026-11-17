# Time-matched sequence-probe overlay scratchpad

## Scope

- Implement `PRD_time_matched_sequence_probe_overlay.md` without modifying the
  frozen ordinal-layout training or probe implementation.
- Use no human data and make no architectural recommendation.
- Primary unit is one seed × checkpoint × context sequence, summarized over
  exactly transition positions 1–7.

## Recovered reference implementation

- Figure: `ordinal_layout_ecology.plotting.figure_raw_rt` generated Figure 2,
  “Raw context-yoked RT learning.”
- Figure source: `outputs/ordinal_layout_ecology_final/stage2_layout_training.csv`.
- Transition source: `stage2_layout_transition_training.csv`.
- Frozen run: Stage 2 `confirmatory`, overlap 0; High scale 2.0 and Low scale
  0.02; seeds 0–19; 8-unit tanh Elman SRN; 170 sequences; A10/B10 blocks;
  Adam at 0.005; gradient clip 1.0; monotonic confidence readout; execution
  weight 0.20.
- Every training sequence and every frozen probe starts with zero recurrent
  state. There is no state carry between sequences.
- The reported online training RT is computed before its corresponding optimizer
  update. Saved final checkpoints are after update 170.

## Timeline audit and decisions

- Only final Stage-2 checkpoints exist. No intermediate checkpoint can be
  inferred from the final model.
- Existing Mixed and Novel tables each contain one A/B/probe evaluation at the
  final checkpoint. Their A/B rows agree exactly across protocols.
- Existing probe audit hashes match the registered final training-state hashes;
  all are zero-update, one-sequence, zero-settling evaluations. Therefore the
  old SRN comparison already had contemporaneous frozen endpoints.
- With no recorded interleaved schedule, the PRD fallback schedule is used:
  after sequences 20, 30, …, 170. Sequence 20 is the first boundary after both
  A and B have been observed. Each event is plotted at t+0.5 and adds no update.
- `endpoint_gap_tolerance = 0.01` was frozen before inspecting new results. It
  reuses the ecology's registered minimum meaningful A/B denominator and keeps
  Low-scale lumping out of the interpolation denominator.
- Representative close-ups are schedule-selected: first (20), middle (100),
  and final (170), not outcome-selected.

## Implementation firewall

- New package: `time_matched_probe_overlay/`; new runner:
  `scripts/run_time_matched_sequence_probe_overlay.py`.
- Frozen `ordinal_layout_ecology/` code and existing outputs are read only.
- Replay duplicates the exact registered update order. Diagnostic evaluation is
  inserted only after an update, then discarded.
- A, B, Mixed, and Novel branches share one model/readout state hash and differ
  only in context features. Each branch independently receives `initial_state=None`.
- Every event hashes model/readout parameters, Adam state, and Torch RNG before
  and after evaluation. A mismatch is fatal.
- Every replay must match the full original learning curve within 5e-7 and the
  final registered/saved checkpoint state hash exactly. The replayed final probe
  must match the existing probe within 5e-7.

## Test and execution log

- Added synthetic aggregation, missing/duplicate transition, between/above/
  below/near-equal classification, shared-checkpoint, seed-cluster bootstrap,
  timestamp-contract, and exact smoke-replay tests.
- Targeted test result: 10 passed.
- Full 20-seed temporary integration run completed. Maximum learning-curve
  discrepancy was 1.78e-15; all 40 final state hashes matched exactly; maximum
  existing-versus-replayed final probe discrepancy was 7.81e-17.
- Visual QA completed for the full overlay and first/middle/final close-up.

## Result noted during integration validation

- High-scale Mixed: 68 between, 213 slower than both, 23 faster than both, and
  16 endpoints too close across 320 seed-events.
- High-scale Novel: 50 between, 240 slower than both, 14 faster than both, and
  16 endpoints too close across 320 seed-events.
- All Low-scale events had A/B gaps within the frozen 0.01 tolerance and are
  deliberately unclassified for interpolation.
- This is recorded only after the fixed analysis ran; no schedule, tolerance,
  aggregation, or visual choice was changed in response.
