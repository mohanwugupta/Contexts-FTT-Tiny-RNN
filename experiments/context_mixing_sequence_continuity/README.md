# Context mixing and sequence continuity

Implements `PRD_CONTEXT_MIXING_SEQUENCE_INFORMATION.md` with an append-only scratchpad and recorded RED → GREEN → REFACTOR stages. The completed result root is `results/context_mixing_sequence_continuity/v1/`.

Start with `FINAL_CONTEXT_MIXING_SEQUENCE_REPORT.md`, `CLAIM_LEDGER.md`, `RESEARCHER_HANDOFF.md` and Figures1–3. The outcome is preserved whether positive, null, confounded or nonidentifiable. No further model search is included.

D0 audits all20 original canonical High checkpoints at170, next-token evaluation, original policy/loss and optimizer-state availability. D1 separates exact nonlinear mixing from propagation, exporting source float32 roundoff and fixed one-time correction controls. D2 uses a fixed grammar contrast and familiar-only prototype/ridge readouts, with explicit cross-context sanity failures. D3 alone trains new scientific copies: the same model AND monotonic policy weights in B/I/S, original loss and joint clipping, fresh matched Adam, and160 full-sequence updates per endpoint. The six reserved-seed smoke endpoints are reused, so60 models require exactly9600 scientific updates. Every final checkpoint is sealed before protected evaluation. No Mixed, Novel, zero cue, auxiliary closure or oracle state participates in training or selection.

Reproduce only in a **new** directory:

1. Copy `CONTEXT_MIXING_SCRATCHPAD_TEMPLATE.md` to `NEW_DIRECTORY/scratchpad.md`.
2. Set `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XDG_CACHE_HOME=/tmp/cms-cache MPLCONFIGDIR=/tmp/cms-mpl`.
3. Run `python3 -m experiments.context_mixing_sequence_continuity.cli --stage d0 --output NEW_DIRECTORY`, then repeat for `d1`, `d2`, `d3_smoke`, `d3`, `d4` in that order.
4. Run `python3 -m experiments.context_mixing_sequence_continuity.verify --output NEW_DIRECTORY --seal` once. Omit `--seal` to verify an existing result.

To rebuild figures and key summaries from sealed tables without further model evaluation, fitting or training:

`python3 -m experiments.context_mixing_sequence_continuity.regenerate --source results/context_mixing_sequence_continuity/v1 --destination EMPTY_DIRECTORY`

No started stage or existing output is overwritten. Original checkpoints and original source are never written. Known Finder `.DS_Store` metadata is logged/excluded from scientific-file lists; any unregistered scientific file or changed recorded content is fatal. Optional unavailable historical config/training CSVs are marked explicitly; exact original embedded checkpoint configs and actual source training code define the objective contract.

The source class is imported from `confirmatory_srn.model`, not redefined. Execution policy gradients and model gradients are jointly clipped as in the source, even though the auxiliary uncertainty paths are detached. Stored model/optimizer states include the separate two-parameter policy. Model labels remain canonical High; C0 sequence-only models are never substituted.

The exact audit, source/claim mapping, test commands, failures and numerical evidence are in the result `audit/`, `tables/claim_source_map.csv`, `scratchpad.md` and `tests/red_green_log.jsonl`. Legacy regression tests used by this implementation are read-only. New training-step tests use disposable synthetic fixtures; they never update original scientific checkpoints.
