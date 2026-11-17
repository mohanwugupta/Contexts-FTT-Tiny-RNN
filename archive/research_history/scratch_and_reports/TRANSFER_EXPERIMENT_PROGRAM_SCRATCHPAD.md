# Transfer Experiment Program Scratchpad

Living implementation record for `PRD_human_SRN_transfer_experiment_program.md`. This file records prespecified choices, discoveries, and validation outcomes; it is not an empirical report.

## Frozen decisions

- Existing cohort: seeds 0–19. New computational cohort: seeds 1000–1019, frozen as `1000 + original_seed` before viewing new outcomes.
- Exact Stage-2 model retained: eight-unit tanh Elman SRN, 42 context features, High/Low scales 2.0/.02, Adam .005, gradient clip 1, sequence-state reset during ordinary acquisition, fixed eight-token sequence, monotonic two-parameter execution policy.
- Sequence measurements score seven next-token transitions before any authorized update. Initiation and wrap are excluded.
- Bootstrap unit is the whole seed or participant. Final budgets are 2,000 bootstrap draws and 1,000 prospective simulations; smoke budgets are 200.
- E3 uses an explicit 42-wide adapter. `00` and `11` must exactly equal repository familiar and probe encoders. Layout interpolation is labeled an input-space diagnostic and is not promoted to a human display.
- E5 preserves the physical token stream across P1/P2. P1 has 64 pairings × 10; P2 has 8 bijective pairings × 80. Readout and optimizer buffers reset at the target boundary; predictive weights carry.
- New browser-task observations are not present. E2/E3 human and E4 outcome evidence remain `awaiting_human_data`; prospective simulation is never labeled empirical.

## Source discoveries

- The legacy files are raw event exports despite their `_clean.csv` names. Stable source-row order recovers within-participant chronology.
- Raw context labels map as follows: Maze → A; Melody → B; Mixed → Mixed; Space → Novel; No Context → Default; warmup remains excluded from experimental sequence time but retained in full chronology.
- Incorrect legacy responses restart the current sequence attempt. `a/s/d/f` is the recovered response mapping.
- The existing retrieval preprocessing deliberately removes original sequence 1 in every block, then requires complete correct original positions 2–10. Therefore first-ever probe identity must be reconstructed before that exclusion and cannot silently be replaced by the first retained sequence.
- Legacy elapsed/rest durations are not recoverable. The audit leaves them missing rather than fabricating timing.
- No reusable original HTML/JavaScript task source was found. E2–E4 therefore share a new self-contained local engine, and this departure is explicit.

## Implementation log

- Added a validated YAML configuration, reusable exact SRN engine/snapshots, A/B-only matching, whole-cluster uncertainty, H0 chronology/local-reference modeling, E1–E5 modules, browser-task engine, power planning, plotting, schemas, staged CLI, and targeted tests.
- Stage 0 compares replayed original rows to the stored prediction/execution diagnostic at every checkpoint 20–170 using a 5e-7 absolute tolerance.
- Smoke testing exposed optimizer-state aliasing during checkpoint cloning: PyTorch optimizer tensors were shared with the cached parent. `SRNEngine.from_snapshot` now deep-copies optimizer state. A sibling-update/frozen-diagnostic test and Stage-0 continuation audit protect against regression.
- E1 matching reads only labeled A/B NLL, labeled A/B accuracy, and incoming schedule phase. All rejected candidates remain in the output.
- E2 distinguishes parallel familiar-practice controls from probe-adapted same-parameter zero-state endpoints and from post-return descendants.
- E5 extends only P0/P2 candidates lacking A/B-only support at 170, stopping at the existing 680 cap. Unmatched comparisons remain explicit.

## Current validation status

- Python compilation: passed using an external bytecode cache because this session has read-only shell access to the research repository.
- Targeted transfer-program tests: 14 passed. Full repository suite: 205 passed.
- Two-seed smoke stages Stage 0, H0, E1, E2, E3, E4 engineering/power, E5, and report generation have completed at least once; stages modified afterward are rerun before handoff.
- Browser automation connection was attempted through the required in-app Browser interface, but the available tool rejects calls because its sandbox metadata is incomplete before setup. Static schedule/DOM/event tests continue to run; actual browser screenshots remain an environment-level QA item until that connection works.
- The full 20-seed computational program completed. Artifact integrity passed for 2,560 reference-reproduction rows, 9 PNG + 9 PDF figure files with 9 exact source tables, the schema manifest, run manifest, and all branch audits.
- Human-SVG extension validation: 8 targeted tests passed; the full repository suite passed with 213 tests; Python compilation passed; and a fresh two-seed end-to-end CLI smoke run generated all required CSV, JSON, Markdown, PNG, and PDF artifacts.
- The smoke G0 replay matched all 112 checkpoint-reference rows within the frozen 5e-7 tolerance. All six smoke figures were rendered and visually inspected; the geometry audit shows the four source arrays by file order and does not expose their registry names as experimental labels.

## Human-SVG geometry transfer extension (PRD v2)

- Added a separate `human_svg_geometry_transfer` pipeline; no frozen Stage-2, transfer-program, execution-readout, or state-space implementation was edited.
- The supplied v3 JSON registry is authoritative. Raw-coordinate normalization is recomputed and checked within 1e-6; x and y share the horizontal half-span scale, y is never independently normalized, and all arrays are strictly P1-to-P8 left-to-right.
- Per the user's clarification, registry names have no scientific role. The four arrays are assigned in file order to A, B, Mixed, and Novel, with the zero-based assignment `0/1/2/3` frozen in `configs/human_svg_geometry_transfer.json`. Source keys remain in audit output only.
- The canonical motor sequence is always `S-A-F-S-D-A-F-D`; source SVG glyph labels are not passed to the model.
- G0 calls the repository's exact existing feature encoder. G1 inserts the four exact normalized arrays. G2 pools x across all four arrays and retains each array's y coordinates. The 42-feature width remains ordinal 8 + two 16-value layout banks + cue 2 so paired parameter initialization and G0 replay are possible.
- A and B retain their trained-bank placement. Mixed and Novel each duplicate their assigned full-layout coordinates across the two already-existing layout banks; this rule is explicit in `geometry_condition_config.json` and is not tuned.
- Screening is seeds 0--4. Only G1/G2 passing all A/B differentiation gates proceed to the remaining seeds 5--19 unless the explicitly diagnostic `--force-confirmatory` flag is used. No positive transfer claim is allowed for a geometry that fails its gate.
- All transfer deficits are computed within seed and transition relative to the worse A/B endpoint. E3, interpolation, and the eight-position spatial/hidden RSA are read-only descendants of the checkpoint-170 parent.
- Optional 680-sequence robustness is intentionally not run in the primary pipeline.

## Scientific guardrails

- Do not call the established result “no transfer.”
- Do not interpret hidden-state splitting as causal on its own.
- Do not call a missing strict match a null effect.
- Do not infer human same-snapshot endpoints; H0 local values are A/B-only modeled references.
- Do not feed the target to the SRN before scoring its prediction.
- Do not tune cue strengths, training dose, seeds, or prior-experience schedules to human probe RT.
