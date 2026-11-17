"""Stage-2 execution-grounding gate."""

from __future__ import annotations

import numpy as np


def _safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def execution_grounding_gate(execution_dynamics, calibration) -> dict:
    frame = execution_dynamics.copy()
    if "seed" not in frame:
        frame["seed"] = 0
    per_seed_overall = [
        _safe_corr(group.correct_token_confidence, group.execution_speed)
        for _, group in frame.groupby("seed")
    ]
    finite_overall = [value for value in per_seed_overall if np.isfinite(value)]
    overall = float(np.mean(finite_overall)) if finite_overall else np.nan
    within = {}
    fractions = {}
    for context in ("A", "B"):
        correlations = [
            _safe_corr(group.correct_token_confidence, group.execution_speed)
            for _, group in frame[frame.context == context].groupby("seed")
        ]
        finite = [value for value in correlations if np.isfinite(value)]
        within[context] = float(np.mean(finite)) if finite else np.nan
        fractions[context] = float(np.mean(np.asarray(finite) > 0)) if finite else np.nan
    context_b = (frame.context == "B").astype(float).to_numpy()
    block = frame.block_index.astype(float).to_numpy()
    block = (block - block.mean()) / max(block.std(), 1e-12)
    confidence = frame.correct_token_confidence.astype(float).to_numpy()
    seed_dummies = pd_get_dummies(frame.seed.to_numpy())
    design = np.column_stack([np.ones(len(frame)), confidence, block, context_b, seed_dummies])
    coefficient = float(np.linalg.lstsq(design, frame.execution_speed, rcond=None)[0][1])
    calibration_monotone = bool(calibration.monotone_optimal_speed.all())
    calibration_interior = bool(calibration.interior_optimum.any())
    checks = {
        "E1_positive_overall_confidence_speed": bool(np.isfinite(overall) and overall > 0),
        "E2_positive_within_A": bool(np.isfinite(within.get("A", np.nan)) and within["A"] > 0),
        "E3_positive_within_B": bool(np.isfinite(within.get("B", np.nan)) and within["B"] > 0),
        "E4_confidence_survives_block_context_controls": bool(coefficient > 0),
        "E5_calibration_monotone": calibration_monotone,
        "E6_calibration_has_interior_optimum": calibration_interior,
    }
    return {
        "confidence_speed_correlation_overall": overall,
        "confidence_speed_correlation_A": within.get("A", np.nan),
        "confidence_speed_correlation_B": within.get("B", np.nan),
        "fraction_seeds_positive_within_A": fractions.get("A", np.nan),
        "fraction_seeds_positive_within_B": fractions.get("B", np.nan),
        "confidence_speed_controlled_coefficient": coefficient,
        **checks,
        "passed": bool(all(checks.values())),
    }


def pd_get_dummies(values):
    """Return n-1 seed indicators without importing pandas in this small module."""
    levels = list(dict.fromkeys(values.tolist()))
    if len(levels) <= 1:
        return np.empty((len(values), 0), dtype=float)
    return np.column_stack([(values == level).astype(float) for level in levels[1:]])
