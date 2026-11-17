# SRN Meta-Learning Calibration Scratchpad

## Scope and frozen decisions

- Implemented `PRD_srn_meta_learning_calibration.md` as a new package and runner; the preceding episodic-capacity implementation is reused and left unchanged.
- Calibration is restricted to structured synthetic P2 families and prediction outcomes. The coordinate JSON is read as four anonymous numeric arrays: layout names and behavioral data never enter the calibration package.
- The inner loop is plain functional SGD with no momentum and no persistent optimizer. Every trajectory starts from an exact cloned state; Adam appears only in the later, gated outer-loop phases.
- The 0-step state is evaluated once per width × initialization × episode and reused across learning rates.
- Copy and unseen samples are held out from the support loss. Unseen-query columns are intentionally rejected by the hyperparameter-selection API.
- Pilot selection is width-specific and can use clipping only in the registered rescue sweep. A clipped selection is explicitly labeled `stable only with clipping`.
- Confirmatory families use a disjoint episode-number range so they cannot repeat pilot families.
- Structural gates are not evaluated when the revised copy gate fails.
- No downstream target-task transfer or execution readout is included in this PRD.

## Implementation map

- `configs/inner_loop_calibration_config.json`: frozen grids, thresholds, smoke reductions, and later-stage gates.
- `srn_meta_learning_calibration/config.py`: validated configuration loading.
- `srn_meta_learning_calibration/initialization.py`: I0–I3 initialization and realized-statistic logging.
- `srn_meta_learning_calibration/calibration.py`: fresh episodes, exact-reset trajectories, stepwise diagnostics, and outer-update invariant.
- `srn_meta_learning_calibration/analysis.py`: catastrophic flags, C1–C6, selection firewall, and width-specific choice.
- `srn_meta_learning_calibration/meta_training.py`: gated meta-batch jobs and revised P2/P1/ordinary comparison.
- `srn_meta_learning_calibration/plotting.py`: registered Figures 1–10.
- `srn_meta_learning_calibration/pipeline.py`: rescue, confirmation, stopping logic, outputs, and report.
- `scripts/run_srn_meta_learning_calibration.py`: command-line entry point.
- `tests/test_srn_meta_learning_calibration.py`: T1–T10 plus forced downstream integration.

## Verification log

- 2026-09-09: Final unforced registered smoke run completed the pilot, clipping rescue, and disjoint-family confirmation, then correctly stopped before meta-training because the tiny smoke confirmation did not pass. All unconditional tables and Figures 1–9 were produced in `/tmp/srn-meta-learning-calibration-smoke-final`.
- 2026-09-09: A forced engineering-only integration test exercised meta-batch calibration and the revised three-condition stage, including Figure 10 and its conditional CSVs.
- 2026-09-09: Dedicated PRD tests passed: `9 passed`.
- 2026-09-09: Full repository regression passed after final edits: `271 passed in 42.81s`.

## Interpretation note

Smoke-mode scientific values are engineering checks only. They use deliberately tiny samples and must not be interpreted as evidence about capacity, initialization, or structural meta-learning.
