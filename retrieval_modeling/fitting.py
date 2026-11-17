"""Joint shared-parameter fitting across Combined and Novel datasets."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from probe_modeling.fit import grouped_participant_split
from probe_modeling.metrics import regression_metrics

from .explicit_models import PRIMARY_RETRIEVAL_MODELS, predict_retrieval_model


BASE_PARAMETER_SPECS = {
    "equal_mixture": [],
    "free_static_mixture": [],
    "continuous_similarity": [("beta", 2.0, (0.01, 10.0))],
    "similarity_history": [
        ("beta", 2.0, (0.01, 10.0)), ("rho", 0.8, (0.0, 0.999)), ("eta", 0.2, (-5.0, 5.0)),
    ],
    "similarity_evidence": [("beta", 2.0, (0.01, 10.0)), ("kappa", 0.1, (0.0, 5.0))],
    "categorical_similarity": [("beta", 2.0, (0.01, 10.0))],
    "default_state": [
        ("beta", 2.0, (0.01, 10.0)), ("default_bias", -3.0, (-8.0, 8.0)),
        ("default_evidence_gain", 1.0, (0.0, 10.0)), ("default_offset", 0.0, (-1.0, 1.0)),
    ],
    "third_state_falsification": [
        ("beta", 2.0, (0.01, 10.0)),
        ("third_learning_rate", 0.01, (0.0, 0.15)),
        ("third_retention", 0.9, (0.0, 1.0)),
    ],
}


def parameter_specs(data: pd.DataFrame, model_name: str) -> list[tuple[str, float, tuple[float, float]]]:
    specs = list(BASE_PARAMETER_SPECS[model_name])
    if model_name == "free_static_mixture":
        for context in ("Both", "Novel", "Default"):
            if data.retrieval_context.eq(context).any():
                specs.append((f"q_{context}", 0.5, (0.0, 1.0)))
    for dataset in sorted(data.dataset.unique()):
        residual_sd = float(data.loc[data.dataset.eq(dataset), "log_rt"].std())
        initial = float(np.log(np.clip(residual_sd if np.isfinite(residual_sd) else 0.1, 0.02, 0.5)))
        specs.append((f"log_sigma_{dataset}", initial, (-6.0, 0.0)))
    return specs


def fit_retrieval_model(
    data: pd.DataFrame,
    model_name: str,
    *,
    fit_mask: np.ndarray | pd.Series | None = None,
) -> tuple[dict[str, float], dict]:
    if model_name not in PRIMARY_RETRIEVAL_MODELS:
        raise ValueError(model_name)
    y = data.log_rt.to_numpy(float)
    mask = np.isfinite(y)
    if fit_mask is not None:
        mask &= np.asarray(fit_mask, dtype=bool)
    if not mask.any():
        raise ValueError("No finite retrieval observations are available for fitting.")
    specs = parameter_specs(data, model_name)
    names = [name for name, _, _ in specs]
    x0 = np.asarray([initial for _, initial, _ in specs], dtype=float)
    bounds = [limits for _, _, limits in specs]

    def objective(values: np.ndarray) -> float:
        predicted = predict_retrieval_model(data, model_name, dict(zip(names, values)))
        likelihood = predicted.observation_log_likelihood.to_numpy(float)[mask]
        return float(-np.sum(likelihood)) if np.isfinite(likelihood).all() else 1e100

    starts = [x0]
    if "beta" in names:
        alternative = x0.copy()
        alternative[names.index("beta")] = 6.0
        starts.append(alternative)
    results = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        for start in starts
    ]
    result = min(results, key=lambda candidate: candidate.fun)
    params = {name: float(value) for name, value in zip(names, result.x)}
    prediction = predict_retrieval_model(data, model_name, params)
    log_likelihood = float(prediction.loc[mask, "observation_log_likelihood"].sum())
    n = int(mask.sum())
    k = len(params)
    return params, {
        "success": bool(result.success), "message": str(result.message), "n": n,
        "n_parameters": k, "log_likelihood": log_likelihood,
        "nll": -log_likelihood, "aic": 2 * k - 2 * log_likelihood,
        "bic": k * np.log(max(n, 1)) - 2 * log_likelihood,
    }


def fit_joint_retrieval_ladder(
    data: pd.DataFrame,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    models: Iterable[str] = PRIMARY_RETRIEVAL_MODELS,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = grouped_participant_split(data, seed=seed, train_fraction=train_fraction)
    train_ids = {
        (row.dataset, row.participant_id)
        for row in splits.itertuples()
        if row.split == "train"
    }
    training = data[
        [(dataset, participant) in train_ids for dataset, participant in zip(data.dataset, data.participant_id)]
    ].copy()
    fit_rows, prediction_frames, comparison_rows = [], [], []
    for model in models:
        if progress:
            progress(f"Joint retrieval fit: {model}")
        params, info = fit_retrieval_model(training, model)
        fit_rows.append(
            {
                "model": model, "fit_scope": "joint_shared_mechanism",
                "parameters": json.dumps(params, sort_keys=True), **info, **params,
            }
        )
        predicted = predict_retrieval_model(data, model, params)
        predicted["split"] = [
            "train" if (dataset, participant) in train_ids else "test"
            for dataset, participant in zip(predicted.dataset, predicted.participant_id)
        ]
        prediction_frames.append(predicted)
        for (dataset, split, context), group in predicted.groupby(
            ["dataset", "split", "retrieval_context"], sort=False
        ):
            metrics = regression_metrics(group.log_rt.to_numpy(), group.predicted_log_rt.to_numpy())
            comparison_rows.append(
                {
                    "model": model, "dataset": dataset, "split": split,
                    "retrieval_context": context, "evaluation_scope": "overall",
                    "probe_exposure_cohort": "All",
                    "log_likelihood": float(group.observation_log_likelihood.sum()),
                    **metrics,
                }
            )
        if "probe_exposure_cohort" in predicted:
            for (dataset, split, context, cohort), group in predicted.groupby(
                ["dataset", "split", "retrieval_context", "probe_exposure_cohort"],
                sort=False, dropna=False,
            ):
                metrics = regression_metrics(group.log_rt.to_numpy(), group.predicted_log_rt.to_numpy())
                comparison_rows.append(
                    {
                        "model": model, "dataset": dataset, "split": split,
                        "retrieval_context": context, "evaluation_scope": "exposure_cohort",
                        "probe_exposure_cohort": cohort,
                        "log_likelihood": float(group.observation_log_likelihood.sum()),
                        **metrics,
                    }
                )
    return (
        pd.DataFrame(fit_rows), pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(comparison_rows), splits,
    )
