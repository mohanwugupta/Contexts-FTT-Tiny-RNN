"""Multi-regime recovery on the exact 17-by-9 PRD2 schedule."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .epoch_fit import EPOCH_PARAMETER_SPECS, fit_epoch_model
from .epoch_models import predict_epoch_model


RECOVERY_MECHANISMS = (
    "static_mixture",
    "counterbalance_static",
    "contextual_inference",
    "gate_adaptation",
    "responsibility_update",
    "third_state",
)


EASY_PARAMETERS = {
    "static_mixture": {"q": 0.72},
    "counterbalance_static": {"q_logit": 0.0, "counterbalance_effect": 1.35},
    "contextual_inference": {"memory": 0.92, "cue_gain": 2.2, "bias": 0.02, "counterbalance_effect": 0.5},
    "gate_adaptation": {"q_logit": -0.3, "counterbalance_effect": 0.5, "gate_learning": 0.08, "gate_retention": 0.75},
    "responsibility_update": {"q_logit": 0.0, "counterbalance_effect": 0.5, "learning_rate": 0.04, "offset_retention": 0.92},
    "third_state": {"q": 0.6, "learning_rate": 0.04, "retention": 0.65},
}


def validate_recovery_schedule(schedule: pd.DataFrame) -> None:
    counts = schedule.groupby(["participant_id", "block_id"]).size()
    block_counts = schedule.groupby("participant_id").block_id.nunique()
    if not counts.eq(9).all() or not block_counts.eq(17).all():
        raise ValueError("PRD2 recovery requires exactly 17 blocks x 9 observations per participant.")
    required = {"Mixed_1", "Mixed_2", "Mixed_Test", "Post_Mixed_Return"}
    if not required.issubset(set(schedule.epoch.astype("object"))):
        raise ValueError("Recovery schedule is missing a required experimental epoch.")


def _human_parameters(model: str, empirical: Mapping[str, Mapping[str, float]] | None) -> dict[str, float]:
    base = dict(EASY_PARAMETERS[model])
    if empirical and model in empirical:
        for name, value in empirical[model].items():
            if name in {spec[0] for spec in EPOCH_PARAMETER_SPECS[model]} and np.isfinite(value):
                base[name] = float(value)
    return base


def _neighborhood_parameters(
    model: str,
    center: Mapping[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    values = dict(center)
    for name, _, (lower, upper) in EPOCH_PARAMETER_SPECS[model]:
        scale = 0.08 * (upper - lower)
        values[name] = float(np.clip(values[name] + rng.normal(0.0, scale), lower, upper))
    return values


def simulate_epoch_mechanism(
    schedule: pd.DataFrame,
    mechanism: str,
    parameters: Mapping[str, float],
    *,
    noise_sd: float,
    seed: int,
) -> pd.DataFrame:
    generated = predict_epoch_model(schedule, mechanism, parameters).copy()
    rng = np.random.default_rng(seed)
    generated["generating_mean_log_rt"] = generated.predicted_log_rt
    generated["log_rt"] = generated.generating_mean_log_rt + rng.normal(0.0, noise_sd, len(generated))
    generated["mean_rt"] = np.exp(generated.log_rt)
    return generated


def run_epoch_model_recovery(
    schedule: pd.DataFrame,
    *,
    n_simulations: int = 20,
    noise_sd: float = 0.05,
    seed: int = 42,
    regimes: Iterable[str] = ("easy", "human", "neighborhood"),
    empirical_parameters: Mapping[str, Mapping[str, float]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate/refit six mechanisms at easy and empirical parameter regions."""
    validate_recovery_schedule(schedule)
    regimes = tuple(regimes)
    unknown = set(regimes).difference({"easy", "human", "neighborhood"})
    if unknown:
        raise ValueError(f"Unknown recovery regimes: {sorted(unknown)}")
    rows = []
    # All distinguishing observations are fit during recovery; this is an
    # identifiability diagnostic, not one of the prospective human tests.
    recovery_mask = schedule.epoch.astype("object").isin(
        ["Mixed_1", "Mixed_2", "Post_Mixed_Return", "Mixed_Test"]
    ).to_numpy()
    for regime_i, regime in enumerate(regimes):
        for mechanism_i, mechanism in enumerate(RECOVERY_MECHANISMS):
            center = _human_parameters(mechanism, empirical_parameters)
            for simulation in range(int(n_simulations)):
                if progress is not None:
                    progress(
                        f"Recovery {regime}: generate {mechanism}, "
                        f"simulation {simulation + 1}/{int(n_simulations)}"
                    )
                simulation_seed = int(seed + regime_i * 1_000_000 + mechanism_i * 10_000 + simulation)
                rng = np.random.default_rng(simulation_seed)
                parameters = (
                    dict(EASY_PARAMETERS[mechanism]) if regime == "easy"
                    else dict(center) if regime == "human"
                    else _neighborhood_parameters(mechanism, center, rng)
                )
                synthetic = simulate_epoch_mechanism(
                    schedule, mechanism, parameters, noise_sd=noise_sd, seed=simulation_seed
                )
                scores, fitted = {}, {}
                for candidate in RECOVERY_MECHANISMS:
                    if progress is not None:
                        progress(f"Recovery candidate fit: {candidate}")
                    candidate_params, info = fit_epoch_model(synthetic, candidate, fit_mask=recovery_mask)
                    n = max(int(info["n"]), 1)
                    k = len(EPOCH_PARAMETER_SPECS[candidate])
                    scores[candidate] = float(
                        n * np.log(max(info["sse"] / n, 1e-16)) + k * np.log(n)
                    )
                    fitted[candidate] = candidate_params
                selected = min(scores, key=scores.get)
                row = {
                    "recovery_regime": regime,
                    "generating_mechanism": mechanism,
                    "simulation": int(simulation),
                    "seed": simulation_seed,
                    "noise_sd": float(noise_sd),
                    "selected_model": selected,
                    "correct": selected == mechanism,
                    "true_parameters": json.dumps(parameters, sort_keys=True),
                    "fitted_parameters": json.dumps(fitted, sort_keys=True),
                    "schedule_blocks": 17,
                    "observations_per_block": 9,
                }
                row.update({f"bic_{candidate}": score for candidate, score in scores.items()})
                rows.append(row)
                if progress is not None:
                    progress(f"Recovery selection: {selected} (generated {mechanism})")
    results = pd.DataFrame(rows)
    confusion_rows = []
    for regime, group in results.groupby("recovery_regime", sort=False):
        matrix = pd.crosstab(
            group.generating_mechanism, group.selected_model, normalize="index"
        ).reindex(index=RECOVERY_MECHANISMS, columns=RECOVERY_MECHANISMS, fill_value=0.0)
        for generating in RECOVERY_MECHANISMS:
            for selected in RECOVERY_MECHANISMS:
                confusion_rows.append(
                    {
                        "recovery_regime": regime,
                        "generating_mechanism": generating,
                        "selected_model": selected,
                        "proportion": float(matrix.loc[generating, selected]),
                        "n_simulations": int(n_simulations),
                    }
                )
    human_region = results[results.recovery_regime.isin(["human", "neighborhood"])].copy()
    return results, pd.DataFrame(confusion_rows), human_region
