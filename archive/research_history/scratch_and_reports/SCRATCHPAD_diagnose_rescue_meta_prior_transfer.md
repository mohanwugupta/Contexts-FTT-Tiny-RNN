# Scratchpad: diagnose and rescue meta-prior transfer

This is a working engineering log, not a scientific result.

- Reused the original H8 `G2_shared_x` training entry point and frozen output tables directly. Stage A never substitutes the H128 target engine for the H8 baseline.
- Defined exact replay as training, frozen prediction/RT, all hidden units, state-space scores, and winner agreement within `1e-6`.
- Reproduced native adaptation as functional first-order, full-batch plain SGD: 10 steps, alpha 0.01, five A plus five B sequences. The source module remains byte-identical.
- Kept Mixed and Novel entirely outside support. They are first-exposure frozen queries.
- Attached the locked zero-initialized monotonic confidence-to-RT policy after adaptation. No execution parameter or RT gradient participates in Stage B.
- Canonicalized A/B only by swapping each seed's complete seven-transition profile; transitions are never relabeled individually.
- Operationalized the hidden permutation null as a seed-local, label-permutation null over the 14 post-adaptation A/B hidden profiles.
- Operationalized the primary Stage B deficit as the mean transitionwise distance of Mixed prediction NLL outside the A/B interval. Positive direction is P2 below shuffled in at least 16/20 paired seeds, jointly with higher between-endpoint confidence and lower hidden corridor distance.
- Stage C is callable only with a recorded Stage B pass. Its matched shuffled control preserves visible features, task marginals, episode types, support/query counts, and optimization while permuting the context-task relationship.
- No model wider than H128 is accepted by configuration validation.

## Verification log

- 2026-09-10: Ordinary smoke run completed. Stage A reproduced the frozen seed-0 training, probe, hidden-state, state-space score, and winner outputs within `1e-6`, then passed F1--F5.
- 2026-09-10: The three-seed Stage B pilot correctly hard-stopped. Both P2 and shuffled checkpoints learned A/B above chance and produced nonidentical profiles, but A/B hidden separation decreased after the native update in all pilot seeds and did not exceed the permutation null. This is a smoke result, not the 20-seed scientific result.
- 2026-09-10: A test-only Stage B override exercised Stage C end to end. The override is hidden from normal CLI help and is recorded in `native_meta_test_gate.csv`; it cannot silently alter a scientific run.
- 2026-09-10: Focused regression suite passed: 33 tests covering this pipeline plus the inherited H8 geometry, H128 calibration, and human meta-transfer modules.
