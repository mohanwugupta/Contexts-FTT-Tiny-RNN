"""Monte Carlo planning for the three new human task configurations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def estimate_human_nuisance(sequences: pd.DataFrame) -> dict:
    data = sequences[
        sequences.context.isin(["A", "B"])
        & sequences.existing_cleaning_included
        & np.isfinite(sequences.sequence_log_execution_RT)
    ].copy()
    participant_means = data.groupby("participant_uid").sequence_log_execution_RT.mean()
    centered = data.sequence_log_execution_RT - data.participant_uid.map(participant_means)
    residual_sd = float(centered.std(ddof=1))
    participant_sd = float(participant_means.std(ddof=1))
    icc = participant_sd ** 2 / max(participant_sd ** 2 + residual_sd ** 2, 1e-12)
    return {
        "source": "existing_combined_and_novel_familiar_AB_sequences",
        "sequence_log_RT_residual_SD": residual_sd,
        "participant_intercept_SD": participant_sd,
        "approximate_ICC": float(icc),
        "participants": int(data.participant_uid.nunique()),
        "sequences": len(data),
    }


def _ols_interaction(y: np.ndarray, factor1: np.ndarray, factor2: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(y)), factor1, factor2, factor1 * factor2])
    if len(y) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        return np.nan, np.nan
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    sigma2 = float(residual @ residual / (len(y) - x.shape[1]))
    covariance = sigma2 * np.linalg.inv(x.T @ x)
    # With factors coded ±1, the conventional difference-in-differences is 4β_interaction.
    return float(4 * beta[3]), float(4 * np.sqrt(covariance[3, 3]))


def _one_simulation(
    rng: np.random.Generator,
    *,
    experiment: str,
    total_n: int,
    core_cells: int,
    effect_sd: float,
    residual_sd: float,
    participant_sd: float,
    participant_missingness: float,
    event_error: float,
) -> tuple[float, float, int]:
    cell = np.arange(total_n) % core_cells
    rng.shuffle(cell)
    retained = rng.random(total_n) >= participant_missingness
    if experiment == "e2":
        # Preceding A/B × first/late within participant; nuisance intercept is retained.
        participant = np.repeat(np.arange(total_n), 2)
        preceding = np.repeat(np.where(cell % 2 == 0, -1.0, 1.0), 2)
        time = np.tile(np.asarray([-1.0, 1.0]), total_n)
        kept = np.repeat(retained, 2) & (rng.random(total_n * 2) >= event_error)
        random_intercept = np.repeat(rng.normal(0, participant_sd, total_n), 2)
        y = (
            random_intercept
            + (effect_sd * residual_sd / 4.0) * preceding * time
            + rng.normal(0, residual_sd, total_n * 2)
        )
        estimate, se = _ols_interaction(y[kept], preceding[kept], time[kept])
        estimable = int(np.unique(participant[kept]).size)
    else:
        # E3 cue×layout and E4 visibility×probe-family use randomized 2×2 cores.
        factor1 = np.where(cell % 2 == 0, -1.0, 1.0)
        factor2 = np.where((cell // 2) % 2 == 0, -1.0, 1.0)
        kept = retained & (rng.random(total_n) >= event_error)
        y = (
            rng.normal(0, participant_sd, total_n)
            + (effect_sd * residual_sd / 4.0) * factor1 * factor2
            + rng.normal(0, residual_sd, total_n)
        )
        estimate, se = _ols_interaction(y[kept], factor1[kept], factor2[kept])
        estimable = int(kept.sum())
    return estimate, se, estimable


def simulate_power_planning(
    *,
    n_options: tuple[int, ...],
    effects: tuple[float, ...],
    draws: int,
    seed: int,
    residual_sd: float,
    participant_sd: float,
    participant_missingness: float,
    event_error: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    designs = {"e2": 4, "e3": 16, "e4": 16}
    rows = []
    for experiment, core_cells in designs.items():
        for total_n in n_options:
            if total_n % core_cells:
                continue
            for effect in effects:
                estimates, ses, estimable = [], [], []
                for _ in range(draws):
                    estimate, se, count = _one_simulation(
                        rng,
                        experiment=experiment,
                        total_n=total_n,
                        core_cells=core_cells,
                        effect_sd=effect,
                        residual_sd=residual_sd,
                        participant_sd=participant_sd,
                        participant_missingness=participant_missingness,
                        event_error=event_error,
                    )
                    if np.isfinite(estimate) and np.isfinite(se) and se > 0:
                        estimates.append(estimate)
                        ses.append(se)
                        estimable.append(count)
                estimates = np.asarray(estimates)
                ses = np.asarray(ses)
                rejection = np.abs(estimates / ses) > 1.96
                probability = float(rejection.mean()) if len(rejection) else np.nan
                mc_se = math.sqrt(probability * (1 - probability) / len(rejection)) if len(rejection) else np.nan
                rows.append({
                    "experiment": experiment,
                    "total_N": total_n,
                    "core_cell_count": core_cells,
                    "nominal_participants_per_core_cell": total_n / core_cells,
                    "standardized_interaction_effect": effect,
                    "draws_requested": draws,
                    "draws_estimable": len(rejection),
                    "operating_characteristic": "type_I_error" if effect == 0 else "power",
                    "rejection_probability": probability,
                    "monte_carlo_SE": mc_se,
                    "monte_carlo_95_lower": max(0.0, probability - 1.96 * mc_se),
                    "monte_carlo_95_upper": min(1.0, probability + 1.96 * mc_se),
                    "mean_95_interval_width_log_RT": float(np.mean(2 * 1.96 * ses)),
                    "mean_estimable_participants": float(np.mean(estimable)),
                    "mean_estimable_participants_per_core_cell": float(np.mean(estimable) / core_cells),
                    "participant_missingness": participant_missingness,
                    "event_error_rate": event_error,
                    "effect_scale": "residual_log_RT_SD",
                    "synthetic_planning_only": True,
                })
    return pd.DataFrame(rows)
