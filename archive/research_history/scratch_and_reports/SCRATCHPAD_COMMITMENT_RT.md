# Commitment RT implementation scratchpad

## Scope

- Implement the recurrent-preparation/learned-commitment PRD additively.
- Freeze all objective-search and readout-diagnostic code.
- Run only the five-seed High-scale Phase-1 screen. Do not auto-run Phase 2.
- Use no human observations, RT targets, transition targets, Mixed training, or Novel training.

## Decisions

1. The recurrent and sequence weights retain the frozen scale-2 initialization. The new
   commitment head starts at exactly zero (hazard 0.5), because it has no established
   scale-2 prior and an arbitrary saturated initial timing policy would confound the test.
2. The expected, commitment-weighted hidden state becomes the next external transition's
   state. This preserves differentiability and temporal continuity without sampling.
3. External state persists across all 170 sequences and context blocks; the graph is
   detached only at each optimizer step (one update per external sequence).
4. The hard analytical trajectory keeps the PRD's `.27,.32,.40,.52,...` prefix and ends
   at `.999`. A `.90` endpoint makes `c_time=.10` rationally give up at cycle 1, which
   would test the truncation rather than the intended time/error ordering.
5. Learned-candidate selection is the lowest prespecified `c_time` passing A--F. It never
   consults human data, probes, threshold controls, or state-space classification.
6. The `.60/.70/.80/.90` controls read commitment time from the exact saved preparation
   trajectories. They construct no optimizer and do not retrain or alter hidden states.
7. Mixed/Novel probe infrastructure is implemented and hash-audited for T8, but probes
   remain deferred until after Phase-2 architecture freezing as the PRD specifies.
8. Required `final_*` files are not fabricated during Phase 1. The report identifies
   those analyses as deferred and the config records `phase_2_started=false`.

## Verification checklist

- [x] T1 commitment probabilities sum to one.
- [x] T2 final hazard receives all remaining mass.
- [x] T3 earlier hazard means lower expected RT.
- [x] T4 higher time cost selects an earlier toy commitment.
- [x] T5 higher error cost selects a later toy commitment.
- [x] T6 training has no RT supervision or forbidden inputs.
- [x] T7 170 external sequences and A10/B10 exposure are unchanged.
- [x] T8 frozen probes make zero updates and preserve parameter hash.
- [x] T9 context vectors match the PRD exactly.
- [x] Smoke pipeline and both Phase-1/Phase-2 figure paths pass.
- [x] Five-seed screen completed technically; scientific stop gates obeyed.
- [x] Existing full test suite passes: 115/115.
- [x] Frozen pre-existing final CSV checksums remain unchanged: 112/112.

## Phase-1 result (2026-09-02)

- All four time costs passed Gate A (CE decreased) and Gate E (no >90% boundary collapse).
- `c_time=.05` and `.10` passed the RT-direction gate; `.10` had RT decline in 5/5 seeds.
- Every candidate failed Gate D. Adjacent-cycle confidence improvement was only
  `.499`--`.506` on average, and only 3/5 seeds exceeded .50 for every candidate.
- `c_time=.01` also failed context structure; `.05` failed uncertainty grounding;
  the other sign gates did not rescue the failed preparation mechanism.
- Conclusion is interpretation D: recurrent preparation itself does not reliably
  improve action readiness under this architecture.
- Phase 2 is implemented behind a saved-gate guard but was not scientifically run.
  A synthetic one-seed integration fixture exercised that code path only.
