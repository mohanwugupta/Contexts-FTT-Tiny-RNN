# Cleanup scratchpad

This log records implementation progress and judgment calls made during the
paper-repository cleanup. It is an audit note, not a scientific result.

## 2026-08-31 — baseline freeze

- Located the final versioned pipelines and their existing final output roots.
- Ran the original suite before moving files: 166 passed; one pre-existing
  pandas `FutureWarning`.
- Replayed v2.1, v2.2, v2.3, v3, and v3.1 as feasible and compared outputs.
- Recorded commands, configs, raw hashes, schemas, headline numerical snapshots,
  and final hashes in `baseline_manifest.json` and
  `baseline_checksums.sha256`.
- Decision: require exact SHA-256 equality because the validated environment
  already reproduces CSVs byte-for-byte; do not introduce tolerances.

## 2026-08-31 — canonical path

- Added portable paper-level JSON configs and runners.
- Decision: runners call the frozen versioned implementations instead of moving
  them into a new `src/` tree. This exposes one clear workflow without risking a
  change in preprocessing, defaults, checkpoint loading, or row ordering.
- Added a checkpoint-validation path for reusing v2.1 without retraining.
- Centralized paper-only plotting over frozen tables and copied each plotted
  table with a recorded hash.

## 2026-08-31 — RED/GREEN cleanup contract

- RED: added a repository contract test before the canonical files existed; the
  required-entry-point assertion failed while four frozen scientific assertions
  passed.
- GREEN: added the canonical configs, scripts, documentation, and provenance
  layer; the contract passed.
- Extended the contract to validate all final CSVs, all frozen checkpoints, and
  the paper figure/table bundle.

## 2026-08-31 — archival decisions

- Archived duplicate task-SRN, GRU, partial-observability, environmental-pressure,
  Bayesian/gate, sweep, obsolete runner, PRD, historical test, and old-output
  material.
- Decision: archive rather than consolidate alternate scientific
  implementations. Their differences are part of hypothesis development.
- Decision: retain `probe_modeling` and `retrieval_modeling` at top level because
  final human preprocessing imports their validated data contract.
- Deleted only caches and OS metadata.

## 2026-08-31 — final verification

- Canonical suite passed after archival.
- Canonical SRN runner reused validated v2.1 weights and reran v2.2/v2.3.
- Canonical human runner reran v3/v3.1 and retained rank 2.
- All 112 final CSVs matched the pre-cleanup manifest byte-for-byte.
- All 180 baseline checksum entries passed.
- Regenerated five paper figures and 17 hash-traced table copies.
- Final decision: report no scientific changes and keep all registered negative
  results explicit.

