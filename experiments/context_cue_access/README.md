# Context cue access and control

Implements all three supplied files preserved in `docs/context_cue_access/`. The experiment uses the same20 frozen canonical High checkpoints as the prior output-compatible-trajectory program. It never trains a scientific model or places a hidden state. Only bounded cue scalars are optimized, explicitly using future targets.

The completed run is `results/context_cue_access/v1/`. Read `FINAL_CONTEXT_CUE_ACCESS_REPORT.md`, `RESEARCHER_HANDOFF.md` and `CLAIM_LEDGER.md`. Figures A/B/C are exported as PNG, editable SVG and PDF; all plots and key inference tables can be regenerated without model evaluation.

To reproduce in a **fresh** directory:

1. Copy `docs/context_cue_access/CONTEXT_CUE_ACCESS_SCRATCHPAD_TEMPLATE.md` to `NEW_DIRECTORY/scratchpad.md`.
2. Set `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XDG_CACHE_HOME=/tmp/cca-cache`.
3. Run `python3 -m experiments.context_cue_access.cli --stage s0 --eval-only --output NEW_DIRECTORY`, then repeat for s1, s2, s3 and s4 in order.
4. Run `python3 -m experiments.context_cue_access.seal --output NEW_DIRECTORY --seal` only once. Omit `--seal` to verify an existing run.

Every stage verifies checkpoint/source/config hashes and prior stage outputs, and refuses overwrite. Final sealing also rejects unregistered extra files. Run the read-only tests recorded in `audit/final_test_result.json`; RED/GREEN/refactor evidence is preserved in `tests/red_green_log.jsonl`, `audit/test_evidence.jsonl` and the append-only scratchpad.

Rebuild summaries/figures from an existing sealed run with:

`python3 -m experiments.context_cue_access.regenerate --source results/context_cue_access/v1 --destination EMPTY_DIRECTORY`

This checks the seal, performs no model evaluation or search, and compares regenerated CSV/PNG/PDF/SVG hashes with the original. The source run remains untouched.

S1 current pulses and S2 preparation have different estimands. R/L have equal cue parameter counts and oracle budgets; the independent I arm has more parameters and a larger finite menu. Best-grid/oracle cues are not natural retrieval. S3 is local forced-sequence control sensitivity, with the tied derivative equal to early+late; it contains no autonomous-dynamics analyses. Source float32 recurrence determines every behavioral outcome. Double differentiation serves cue search and sensitivity diagnostics only; all source/surrogate discrepancies, null and failed outcomes remain visible.
