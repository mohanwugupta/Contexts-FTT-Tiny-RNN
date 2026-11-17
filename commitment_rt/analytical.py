"""Pre-network sanity checks for time/error-optimal commitment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CommitmentConfig


TOY_TRAJECTORIES = {
    "easy": np.array([.60, .80, .93, .98, .99, .995, .997, .999]),
    "medium": np.array([.35, .55, .72, .84, .91, .95, .97, .98]),
    # The PRD supplies the first four values followed by an ellipsis. The final
    # near-certain state represents successful continued preparation; ending at
    # .90 instead makes a high time cost rationally "give up" at cycle 1.
    "hard": np.array([.27, .32, .40, .52, .64, .75, .84, .999]),
}


def optimal_commitment_cycle(
    confidence,
    *,
    c_error: float,
    c_time: float,
) -> tuple[int, float]:
    p = np.asarray(confidence, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Confidence trajectory must be a finite vector.")
    costs = c_error * (1.0 - p) + c_time * np.arange(len(p))
    cycle = int(np.argmin(costs)) + 1
    return cycle, float(costs[cycle - 1])


def run_analytical_calibration(config: CommitmentConfig):
    if config.max_preparation_cycles != 8:
        raise ValueError("The Phase-1 toy trajectories are calibrated for M_max=8.")
    rows = []
    for c_time in config.c_time_values:
        for difficulty, confidence in TOY_TRAJECTORIES.items():
            cycle, cost = optimal_commitment_cycle(
                confidence, c_error=config.c_error, c_time=c_time
            )
            rows.append({
                "c_time": c_time,
                "c_error": config.c_error,
                "trajectory": difficulty,
                "confidence_trajectory": ";".join(f"{value:.3f}" for value in confidence),
                "optimal_commitment_cycle": cycle,
                "normalized_optimal_rt": (cycle - 1) / (config.max_preparation_cycles - 1),
                "minimum_expected_cost": cost,
            })
    calibration = pd.DataFrame(rows)

    error_rows = []
    for c_error in (.5, 1.0, 2.0):
        cycle, cost = optimal_commitment_cycle(
            TOY_TRAJECTORIES["medium"], c_error=c_error, c_time=.025
        )
        error_rows.append({
            "c_time": .025,
            "c_error": c_error,
            "trajectory": "medium_error_cost_check",
            "confidence_trajectory": ";".join(
                f"{value:.3f}" for value in TOY_TRAJECTORIES["medium"]
            ),
            "optimal_commitment_cycle": cycle,
            "normalized_optimal_rt": (cycle - 1) / (config.max_preparation_cycles - 1),
            "minimum_expected_cost": cost,
        })
    calibration = pd.concat([calibration, pd.DataFrame(error_rows)], ignore_index=True)

    summaries = []
    main = calibration[calibration.trajectory.isin(TOY_TRAJECTORIES)]
    for c_time, group in main.groupby("c_time", sort=True):
        cycles = group.set_index("trajectory").optimal_commitment_cycle
        ordered = bool(cycles.easy <= cycles.medium <= cycles.hard)
        separated = bool(cycles.easy < cycles.hard)
        summaries.append({
            "c_time": c_time,
            "easy_cycle": int(cycles.easy),
            "medium_cycle": int(cycles.medium),
            "hard_cycle": int(cycles.hard),
            "easy_not_later_than_medium_not_later_than_hard": ordered,
            "easy_strictly_earlier_than_hard": separated,
            "analytical_candidate_passed": bool(ordered and separated),
        })
    summary = pd.DataFrame(summaries)

    time_effects = []
    for trajectory, group in main.groupby("trajectory"):
        ordered = group.sort_values("c_time").optimal_commitment_cycle.to_numpy()
        time_effects.append(bool(np.all(np.diff(ordered) <= 0)))
    error_cycles = calibration[
        calibration.trajectory.eq("medium_error_cost_check")
    ].sort_values("c_error").optimal_commitment_cycle.to_numpy()
    global_checks = {
        "easy_faster_than_hard_for_retained_candidates": bool(
            summary.loc[summary.analytical_candidate_passed].easy_strictly_earlier_than_hard.all()
        ),
        "higher_time_cost_never_later": bool(all(time_effects)),
        "higher_error_cost_never_earlier": bool(np.all(np.diff(error_cycles) >= 0)),
        "higher_error_cost_has_strict_effect": bool(np.any(np.diff(error_cycles) > 0)),
    }
    global_checks["analytical_sanity_passed"] = bool(all(global_checks.values()))
    return calibration, summary, global_checks
