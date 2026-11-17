"""Recovery of continuous versus categorical and competing retrieval rules."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .explicit_models import predict_retrieval_model
from .fitting import fit_retrieval_model, parameter_specs


RECOVERY_MODELS = (
    "continuous_similarity",
    "categorical_similarity",
    "similarity_history",
    "similarity_evidence",
    "free_static_mixture",
    "default_state",
)

EASY_PARAMETERS = {
    "continuous_similarity": {"beta": 3.0},
    "categorical_similarity": {"beta": 3.0},
    "similarity_history": {"beta": 3.0, "rho": 0.85, "eta": 0.8},
    "similarity_evidence": {"beta": 3.0, "kappa": 0.8},
    "free_static_mixture": {"q_Both": 0.65, "q_Novel": 0.35, "q_Default": 0.5},
    "default_state": {
        "beta": 3.0, "default_bias": 1.0,
        "default_evidence_gain": 4.0, "default_offset": 0.3,
    },
}


def _complete_parameters(
    schedule: pd.DataFrame,
    model: str,
    supplied: Mapping[str, float],
    noise_sd: float,
) -> dict[str, float]:
    params = dict(supplied)
    for dataset in schedule.dataset.unique():
        params[f"log_sigma_{dataset}"] = float(np.log(noise_sd))
    return params


def simulate_retrieval_data(
    schedule: pd.DataFrame,
    model: str,
    params: Mapping[str, float],
    *,
    seed: int,
) -> pd.DataFrame:
    predicted = predict_retrieval_model(schedule, model, params)
    rng = np.random.default_rng(seed)
    sigma = predicted.predicted_sigma.to_numpy(float)
    if model == "categorical_similarity":
        choose_a = rng.random(len(predicted)) < predicted.q_A.to_numpy(float)
        intercept = predicted.get("participant_intercept", pd.Series(0.0, index=predicted.index)).to_numpy(float)
        mean = np.where(choose_a, predicted.pred_A_log_rt, predicted.pred_B_log_rt) + intercept
    else:
        mean = predicted.predicted_log_rt.to_numpy(float)
    generated = schedule.copy()
    generated["log_rt"] = mean + rng.normal(0.0, sigma)
    generated["mean_rt"] = np.exp(generated.log_rt)
    generated["generating_mean_log_rt"] = mean
    generated["generating_model"] = model
    return generated


def _human_center(model: str, empirical: Mapping[str, Mapping[str, float]] | None) -> dict[str, float]:
    center = dict(EASY_PARAMETERS[model])
    if empirical and model in empirical:
        center.update({key: float(value) for key, value in empirical[model].items() if np.isfinite(value)})
    return center


def _jitter(
    schedule: pd.DataFrame,
    model: str,
    center: Mapping[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    values = dict(center)
    for name, _, (lower, upper) in parameter_specs(schedule, model):
        if name.startswith("log_sigma_"):
            continue
        current = values.get(name, 0.5 * (lower + upper))
        values[name] = float(np.clip(current + rng.normal(0, 0.06 * (upper - lower)), lower, upper))
    return values


def run_retrieval_recovery(
    schedule: pd.DataFrame,
    *,
    mechanisms: Iterable[str] = RECOVERY_MODELS,
    candidates: Iterable[str] = RECOVERY_MODELS,
    regimes: Iterable[str] = ("easy", "human", "neighborhood"),
    n_simulations: int = 20,
    noise_sd: float = 0.08,
    seed: int = 42,
    empirical_parameters: Mapping[str, Mapping[str, float]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mechanisms, candidates, regimes = tuple(mechanisms), tuple(candidates), tuple(regimes)
    if n_simulations <= 0:
        return pd.DataFrame(), pd.DataFrame()
    block_sizes = schedule.groupby(["dataset", "participant_id", "block_id"]).size() if "block_id" in schedule else pd.Series(dtype=int)
    if len(block_sizes) and not block_sizes.eq(9).all():
        raise ValueError("Retrieval recovery requires nine analyzed observations per block.")
    rows = []
    for regime_i, regime in enumerate(regimes):
        for mechanism_i, mechanism in enumerate(mechanisms):
            center = _human_center(mechanism, empirical_parameters)
            for simulation in range(int(n_simulations)):
                simulation_seed = seed + regime_i * 1_000_000 + mechanism_i * 10_000 + simulation
                rng = np.random.default_rng(simulation_seed)
                base = EASY_PARAMETERS[mechanism] if regime == "easy" else center
                parameters = dict(base) if regime != "neighborhood" else _jitter(schedule, mechanism, base, rng)
                parameters = _complete_parameters(schedule, mechanism, parameters, noise_sd)
                if progress:
                    progress(f"Retrieval recovery {regime}: {mechanism} {simulation + 1}/{n_simulations}")
                synthetic = simulate_retrieval_data(schedule, mechanism, parameters, seed=simulation_seed)
                scores, fitted = {}, {}
                for candidate in candidates:
                    candidate_params, info = fit_retrieval_model(synthetic, candidate)
                    scores[candidate] = info["bic"]
                    fitted[candidate] = candidate_params
                selected = min(scores, key=scores.get)
                row = {
                    "regime": regime, "generating_model": mechanism,
                    "simulation": simulation, "seed": simulation_seed,
                    "selected_model": selected, "correct": selected == mechanism,
                    "true_parameters": json.dumps(parameters, sort_keys=True),
                    "fitted_parameters": json.dumps(fitted, sort_keys=True),
                }
                row.update({f"bic_{candidate}": score for candidate, score in scores.items()})
                rows.append(row)
    results = pd.DataFrame(rows)
    confusion_rows = []
    for regime, group in results.groupby("regime", sort=False):
        matrix = pd.crosstab(group.generating_model, group.selected_model, normalize="index").reindex(
            index=mechanisms, columns=candidates, fill_value=0.0
        )
        for generating in mechanisms:
            for selected in candidates:
                confusion_rows.append(
                    {
                        "regime": regime, "generating_model": generating,
                        "selected_model": selected, "proportion": float(matrix.loc[generating, selected]),
                        "n_simulations": int(n_simulations),
                    }
                )
    return results, pd.DataFrame(confusion_rows)
