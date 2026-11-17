# Repository cleanup report

Cleanup completed on 2026-08-31. The scientific analyses, frozen weights,
preprocessing, exclusions, transition indexing, model selection, and inferential
decisions were not changed.

## Outcome

The repository now has one paper-level path:

```text
configs/srn_final.json
configs/human_final.json
scripts/run_srn_final.py
scripts/run_human_final.py
scripts/run_all_final.py
scripts/make_paper_figures.py
```

The five registered analysis packages remain the scientific implementation.
The new runners validate their portable JSON contracts and call the original,
already-tested versioned entry points. This kept the cleanup organizational and
made the paper workflow obvious without rewriting scientific code.

## Baseline frozen before cleanup

Before any move or deletion:

- the original suite passed: 166 tests, with one pre-existing pandas
  `FutureWarning`;
- the v2.1 execution, v2.2 causal, v2.3 context-sequence, v3 human, and v3.1
  low-rank pipelines were independently replayed where computationally
  feasible;
- all replayed headline CSVs matched their existing final outputs byte-for-byte;
- the frozen v3.1 analysis selected rank 2;
- `reproducibility/baseline_manifest.json` recorded commands, configs, raw-data
  hashes, schemas, all 112 final CSV hashes, and numerical figure snapshots;
- `reproducibility/baseline_checksums.sha256` froze 180 inputs, configs, outputs,
  reports, and registered figures;
- the v2.1 checkpoint manifest retained SHA-256 values for all 240 checkpoints.

Cleanup did not proceed until these checks passed.

## Files moved and archived

No scientifically informative work was discarded. The following were moved out
of the canonical path:

- nine original task-SRN files, including the old runner, parameter sweeps, and
  context-switching scripts, to `archive/exploratory/legacy_task_srn/`;
- the `srn_environmental_pressures` and `srn_partial_observability_v1` packages
  to `archive/exploratory/packages/`;
- the two GRU partial-observability implementations to
  `archive/deprecated/packages/`;
- six obsolete analysis runners to `archive/exploratory/scripts/`;
- 41 historical tests to `archive/exploratory/tests/` and
  `archive/deprecated/tests/`;
- all 12 PRDs to `archive/exploratory/prds/`;
- ten version-specific and exploratory READMEs to
  `archive/exploratory/analysis_readmes/`;
- the accumulated PRD scratchpad to
  `archive/exploratory/SCRATCHPAD_PRD_HISTORY.md`;
- eleven superseded, verification-only, or exploratory output directories to
  `archive/old_outputs/`.

The archived outputs occupy approximately 1.3 GB. They preserve original
task-trained SRN failures, Low-scale comparisons, GRU/partial-observability
experiments, Bayesian/gate searches, parameter sweeps, verification runs, and
other hypothesis-development history.

## Files deleted

Only generated, non-scientific clutter was deleted:

- Python `__pycache__` directories;
- `.pytest_cache`;
- macOS `.DS_Store` files.

No data, code, output table, model checkpoint, report, PRD, or scientifically
informative figure was deleted.

## Consolidation decisions

- Paper orchestration is consolidated into the four canonical scripts above.
- Paper plotting is consolidated in `scripts/make_paper_figures.py`; it reads
  frozen CSVs and never fits a model.
- The five paper figures and 17 exact plotted-table copies are centralized under
  `outputs/final/`, with hashes in `paper_figure_sources.json`.
- Competing SRN and GRU implementations were archived rather than merged. A
  scientific implementation was not rewritten merely to resemble the proposed
  `src/` tree.
- `retrieval_modeling.data` and `probe_modeling.data` remain at top level because
  the final human loader reuses their validated exclusion/normalization contract.
  Refactoring that dependency could change preprocessing and was deliberately
  avoided.

## Documentation and provenance added

- `README.md`: environment, commands, expected data, outputs, and repository map.
- `docs/ANALYSIS_OVERVIEW.md`: SRN-first/human-second scientific flow and frozen
  inferential boundaries.
- `docs/REPRODUCIBILITY.md`: setup, replay, checksum, and portability guidance.
- `docs/RESULT_MAP.md`: claim-to-script/config/data/table/figure/test/status map
  for Figures 1–5 and recommended Supplement analyses.
- `archive/README.md` and subdirectory READMEs: historical status and usage
  boundaries.
- `reproducibility/CLEANUP_SCRATCHPAD.md`: chronological cleanup decisions.

## Tests added and retained

`tests/test_paper_repo_contract.py` adds explicit regression coverage for:

- canonical entry points and configs;
- the fixed eight-token sequence;
- A/B, Mixed, and Novel context vectors;
- the alternating 17-block A10/B10 curriculum;
- 8-unit High-scale (`0.5`) and Low-scale (`0.02`) conditions;
- zero-update Mixed/Novel probes and a zero Novel training column;
- no probe leakage into pure A/B human profiles;
- no human-outcome tuning of SRNs;
- candidate ranks and frozen selected rank 2;
- frozen Mixed and Novel model ordering;
- headline output schemas;
- all 112 final CSV hashes;
- all 240 checkpoint hashes;
- completeness and hash tracing for the five paper figures and 17 paper tables.

Existing canonical unit, integration, transition-indexing, causal-control,
preprocessing, recovery, and smoke tests remain under `tests/`. Historical tests
remain archived, and `pytest.ini` prevents them from being mistaken for the
canonical paper suite.

## Reproducibility commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_srn_final.py --config configs/srn_final.json
python scripts/run_human_final.py --config configs/human_final.json
python scripts/make_paper_figures.py
```

Run everything with:

```bash
python scripts/run_all_final.py
```

Replay all downstream analyses with the checksum-validated frozen v2.1
checkpoints:

```bash
python scripts/run_all_final.py --reuse-execution
```

Verify frozen final tables afterward:

```bash
python reproducibility/verify_final_outputs.py
shasum -a 256 -c reproducibility/baseline_checksums.sha256
```

## Before/after scientific comparison

| Check | Before cleanup | After cleanup | Change |
|---|---:|---:|---|
| Frozen final CSVs | 112 | 112 | None; every file is byte-identical |
| Baseline artifacts | 180/180 valid | 180/180 valid | None |
| Frozen SRN checkpoints | 240 manifest hashes valid | 240 manifest hashes valid | None |
| Human selected rank | 2 | 2 | None |
| Mixed best held-out model | M3 low rank | M3 low rank | None |
| Novel best held-out model | M3 low rank | M3 low rank | None |
| Paper figures | Five-source numerical baseline | Five regenerated figures | No source-table change |

The post-cleanup canonical SRN replay reused the frozen v2.1 checkpoints, reran
v2.2 and v2.3, and regenerated their registered figures. The full v3 and v3.1
human pipelines were rerun. The verifier then confirmed all 112 CSVs were
bitwise identical, and the complete 180-line checksum file passed. No numerical
tolerance exception was needed.

The scientific conclusions therefore remain unchanged: High-scale networks
show the registered split and execution behavior; Low scale is the negative
control; matched current-state components show causal specificity but the
strict necessity boundary remains; the human solution is rank 2 and structured;
semantic interpretation is limited; and exact human/SRN transition geometry
does not match.

## Unresolved technical debt

- The final human preprocessing dependency still traverses
  `retrieval_modeling.data` and `probe_modeling.data`. Only the data contract is
  canonical; the broader model modules are compatibility code.
- Archived scripts are preserved for audit, not maintained as directly runnable
  entry points. Replaying them may require restoring their historical package
  paths.
- Historical run-config JSON files contain machine-specific absolute paths as
  provenance strings. Canonical configs use relative paths; the historical
  strings were not rewritten because that would break frozen hashes.
- The 1.3 GB historical output archive is intentionally retained. Moving it to
  external archival storage can be considered separately, with its own checksum
  manifest.
- The post-cleanup default test suite covers canonical code (65 tests after this
  cleanup), while the 41 historical tests are preserved but excluded.
- This directory is not currently a Git worktree. The reproducibility manifests
  provide a file-level audit trail, but future changes would benefit from normal
  version-control history.

## Scientific ambiguities deliberately left untouched

- The human implementation estimates the registered participant A/B profile on
  the log-RT scale even though the high-level paper shorthand writes
  `RT_A - RT_B`. This was documented rather than changed.
- The v2.2 full-subspace necessity suite and v2.3 matched-component analysis have
  different inferential roles. Both remain canonical because collapsing them
  would erase the registered failed-necessity boundary.
- The SRN remains an existence proof. No human outcome was used to select a seed,
  tune a network, or alter its weights, and the failed exact human/SRN profile
  match remains visible.
- Limited component semantics were not promoted into stronger labels during
  cleanup.
