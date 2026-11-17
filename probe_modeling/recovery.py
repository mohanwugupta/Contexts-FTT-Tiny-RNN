"""Scientific model recovery on the empirical participant schedules."""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .fit import PARAMETER_SPECS, fit_explicit_model
from .prediction import default_parameters, predict_explicit


PRIMARY_MECHANISMS = (
    "equal_mixture",
    "contextual_inference",
    "responsibility_update",
    "third_state",
)


def simulate_mechanism(
    schedule: pd.DataFrame,
    mechanism: str,
    *,
    parameters: Optional[Mapping[str, float]] = None,
    noise_sd: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate log RT while preserving the exact empirical block schedule."""
    params = default_parameters(mechanism)
    if parameters:
        params.update(parameters)
    generated = predict_explicit(schedule, mechanism, params)
    rng = np.random.default_rng(seed)
    generated = generated.copy()
    generated["generating_mean_log_rt"] = generated.predicted_log_rt
    generated["log_rt"] = generated.generating_mean_log_rt + rng.normal(0.0, noise_sd, len(generated))
    generated["mean_rt"] = np.exp(generated.log_rt)
    generated["generating_mechanism"] = mechanism
    return generated


def _bic(info: dict, n_parameters: int) -> float:
    n = max(int(info["n"]), 1)
    return float(n * np.log(max(float(info["sse"]) / n, 1e-16)) + n_parameters * np.log(n))


def run_model_recovery(
    schedule: pd.DataFrame,
    *,
    mechanisms: Iterable[str] = PRIMARY_MECHANISMS,
    candidates: Iterable[str] = PRIMARY_MECHANISMS,
    n_simulations: int = 20,
    noise_sd: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate from and refit every primary mechanism.

    Selection uses BIC, not in-sample error, so nested update models do not win
    merely by setting their additional parameters close to zero.
    """
    mechanisms = tuple(mechanisms)
    candidates = tuple(candidates)
    rows = []
    for mechanism_i, mechanism in enumerate(mechanisms):
        if mechanism not in PRIMARY_MECHANISMS:
            raise ValueError(f"Unsupported generating mechanism {mechanism!r}")
        true_parameters = default_parameters(mechanism)
        for simulation in range(int(n_simulations)):
            simulation_seed = int(seed + 10000 * mechanism_i + simulation)
            synthetic = simulate_mechanism(
                schedule,
                mechanism,
                parameters=true_parameters,
                noise_sd=noise_sd,
                seed=simulation_seed,
            )
            scores = {}
            fitted_parameters = {}
            for candidate in candidates:
                params, info = fit_explicit_model(synthetic, candidate)
                scores[candidate] = _bic(info, len(PARAMETER_SPECS[candidate]))
                fitted_parameters[candidate] = params
            selected = min(scores, key=scores.get)
            row = {
                "generating_mechanism": mechanism,
                "simulation": simulation,
                "seed": simulation_seed,
                "noise_sd": float(noise_sd),
                "selected_model": selected,
                "correct": selected == mechanism,
                "true_parameters": json.dumps(true_parameters, sort_keys=True),
                "fitted_parameters": json.dumps(fitted_parameters, sort_keys=True),
            }
            row.update({f"bic_{candidate}": score for candidate, score in scores.items()})
            rows.append(row)
    results = pd.DataFrame(rows)
    confusion = pd.crosstab(
        results.generating_mechanism,
        results.selected_model,
        normalize="index",
    ).reindex(index=mechanisms, columns=candidates, fill_value=0.0)
    confusion.index.name = "generating_mechanism"
    confusion.columns.name = "selected_model"
    return results, confusion


def recovery_identifiability(
    results: pd.DataFrame,
    threshold: float = 0.8,
    min_simulations: int = 20,
) -> pd.DataFrame:
    """Explicitly flag mechanisms that fail the PRD's recovery criterion."""
    summary = (
        results.groupby(["generating_mechanism", "noise_sd"], as_index=False)
        .agg(recovery_rate=("correct", "mean"), n_simulations=("correct", "size"))
    )
    summary["adequate_simulations"] = summary.n_simulations.ge(min_simulations)
    summary["identifiable"] = summary.recovery_rate.ge(threshold) & summary.adequate_simulations
    summary["criterion"] = float(threshold)
    summary["minimum_simulations"] = int(min_simulations)
    summary["interpretation"] = np.select(
        [~summary.adequate_simulations, summary.identifiable],
        ["verification only: insufficient simulations", "meets recovery criterion"],
        default="not identifiable on this schedule/noise level",
    )
    return summary
