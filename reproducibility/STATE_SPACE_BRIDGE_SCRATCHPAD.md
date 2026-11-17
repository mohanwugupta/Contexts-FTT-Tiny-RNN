# State-space bridge implementation scratchpad

## Decisions

- The investigator explicitly selected `state-space_SRN.Rmd` after the PRD was written. Its
  no-fatigue OLS model family is canonical. The earlier PRD request for fatigue/recovery is
  superseded and documented rather than silently blended into a new topology.
- Ported exact R behavior: Lump reads its post-update state; Cross/Split read the active state
  before the gated update; initial latent states equal the first observed value; bounds are
  `.60–.999` for retention and `0–.50` for cross-talk; BIC counts are 3/5/4.
- Mixed, Novel, and No Context have no branch in the canonical Python state-space engine.
- Human scalar DV is the validated sequence mean of correct keypress RTs in milliseconds after
  the existing raw-first-sequence and complete-block exclusions. Transition log RT remains the
  separate vector/retrieval DV.
- Corrected High scale 2 is implemented additively with new output roots. Existing `.5` outputs,
  configs, checkpoints, and running fits are not edited.
- Smoke SRN runs retain all 170 sequences and checkpoints, limiting only to two seeds. A shorter
  schedule cannot test the full trajectory gate or feed the frozen downstream checkpoint tools.
- The first faithful scale-2 smoke exposed a failed RT-direction gate: representational scale 2
  also produces a rapidly initialized motor head, and execution speed declined despite improving
  CE/confidence. A diagnostic cautious motor initialization made Low pass but did not rescue High.
  That experimental hook was reverted in full, leaving the existing trainer/model code unchanged.
  Per the PRD, the audited failure blocks downstream scientific fitting rather than inviting
  parameter tuning toward a preferred result.
- The categorical retrieval adapter chooses A versus B on odd probe repetitions and evaluates
  the frozen endpoint on even repetitions. Other retrieval models reuse validated v3 held-out
  predictions and are renamed `RET_*` to avoid collision with the state-space family.
- The current rank-2 result is an explicit frozen-data regression test; the rank is still read
  from dimension selection and is never assigned by bridge code.
- No Context is retained as a third retrieval target in both raw datasets. It is fitted with the
  same held-out ladder, using the frozen neutral/Novel SRN profile because the network's unseen
  context column is exactly zero. It remains excluded from scalar learning dynamics.

## Progress log

- Read the full bridge PRD, `analysis_SRN.Rmd`, `state-space_SRN.Rmd`, and both historical Rmds.
- Located and documented inconsistent historical `w_Mixed`/`w_mix` rules.
- Wrote RED tests before the bridge package, then implemented the shared numerical engine and
  scientific contracts.
- Added corrected SRN runner, data adapters, recovery, retrieval/vector adapters, six figures,
  reports, full-stage CLI, configuration audit, and this scratchpad.

## Remaining verification at implementation time

- The two-seed faithful corrected scale-2 smoke fails the speed-up/RT-down gate for both High and
  Low. The full pipeline correctly stops at this point and retains `execution_sanity_gate.csv`.
- Downstream assembly is therefore tested with unit/integration fixtures, not by bypassing the
  scientific gate.
- Full recovery verification on the actual 170-sequence schedule used 100 datasets per generating
  model and selected Lump 100%, Cross 100%, and Split 99% correctly.
- The complete repository suite passed 81/81 tests, and the cleanup checksum verifier confirmed
  all 112 frozen final CSVs remained bitwise identical.
- Run the complete pre-existing regression suite and verify frozen checksums.
- The 20-seed final run and 100-per-model recovery are intentionally exposed as a terminal
  command; they are not required for source-level smoke verification.
