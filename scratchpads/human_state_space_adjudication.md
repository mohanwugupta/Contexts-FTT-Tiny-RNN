# Human state-space adjudication scratchpad

## Locked decisions

- The historical reproduction gate imports `state_space_bridge.state_space` unchanged. No existing model or pipeline file is edited.
- Primary fitting uses only familiar A/B sequences before the first probe, from the validated `retrieval_v1` contract. Probe outcomes are never returned by the data adapter.
- Raw sequence 1 is excluded exactly as in the existing contract. The modeled transition positions are the seven correct keypress transitions 1→2 through 7→8.
- The historical scalar recurrence is extended independently across the seven transition positions for Lump/Cross/Split. This retains its decay recurrences, bounds, pre-update readout, Gaussian/OLS observation layer, and BIC convention while allowing a transition profile.
- M3 uses a centered A/B contrast, so delta_A + delta_B is exactly zero at every transition. With only two centered contexts, M3a's learned rank-one contrast and an unregularized M3b are algebraically prediction-equivalent; M3b therefore differs only through its prespecified ridge shrinkage. This is reported rather than hidden.
- M4 uses seven shared skill coordinates plus one scalar execution state per context. The execution state learns a signed context target and has separate retention and learning-rate parameters; a seven-transition loading profile maps it to RT. It has no switch/onset/time-since-switch predictor.
- M5 is diagnostic and is not given a mechanistic interpretation unless recovery succeeds.
- Forward validation is blockwise and strictly chronological. Held-out observations can advance latent recurrences, but all fitted parameters and decoder weights are frozen.
- Mixed/Novel/NoContext have no default midpoint mapping. The separate held-out script emits an explicit “undefined” status unless a later prespecified generalization rule is supplied.
- Recovery simulates every locked participant using that participant's fitted generator and noise scale. Identical observed schedules are averaged only after simulation; candidate BIC is weighted by stratum size and summed, retaining the actual participant count and schedule/missingness distribution without making the 7×7 recovery computationally intractable.
- After the first final run stopped at participant 90, Cross/Split and M5 were rewritten as seven independent low-dimensional optimizations. Their likelihood factorizes by transition, so this changes neither the model nor its optimum; it removes an unnecessary 14–21 dimensional numerical search.
- Optimization verification no longer destroys the full run. Repeated L-BFGS-B refinement and a bounded Powell fallback are attempted; unresolved cases are retained with `optimization_robust = False` and surfaced in the selection artifact/report.
- Atomic, hash-validated checkpoints now cover real participant fits, each forward-CV fold, and every recovery generator/replicate. Reusing the same output directory resumes matching checkpoints; changed data or fit settings invalidate them automatically.
- Process-based parallelism now covers independent participant fits, chronological CV folds, and recovery replicates. Seeds are derived from task identity rather than completion order, and checkpoint writes remain in the parent process. Each worker limits BLAS/OpenMP to one thread to avoid oversubscription. Automatic mode uses at most six workers on this 12-logical-core machine and can be overridden with `--workers`.

## Implementation progress

- [x] Read complete PRD and historical R/Python state-space sources.
- [x] Add familiar-only data firewall and participant data contract.
- [x] Add seven-model fitter, multistart optimization, bounds, and diagnostics.
- [x] Add historical reproduction gate, forward CV, recovery, parameter recovery, equivalence audit.
- [x] Add required tables, frozen/hash-recorded selection artifact, report, and figures.
- [x] Add separate post-freeze held-out mapping audit.
- [x] Run unit tests and a real-data smoke pipeline; fixed the pivot-index adapter bug found by the first pass.
- [x] Re-run smoke after population-size recovery tightening: 3 participants, 756 transition rows, all seven generators, 11 figures, and frozen selection artifact completed.
- [x] Final optimizer-verification build passed 16 targeted/new regression tests and a fresh one-participant end-to-end run through fitting, CV, all 7×7 recovery cells, report writing, and selection hashing.
- [x] Crash-resilience revision passed 17 tests, reproduced the exact failing participant `MG455855` with all seven fits robust, and verified a complete second run resumed real-fit, CV, and recovery checkpoints without recomputation.
- [x] Parallel CLI smoke completed with two spawned workers across participant fitting, CV, and all recovery generators. Its five core fit/CV/recovery tables matched a one-worker run to numerical tolerance (`rtol=1e-12`, `atol=1e-10`).
