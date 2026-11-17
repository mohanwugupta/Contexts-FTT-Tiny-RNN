"""Grouped fitting and temporal evaluation for the PRD2 model ladder."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .epoch_data import mirror_ab_labels
from .epoch_models import EPOCH_MODELS, predict_epoch_arrays, predict_epoch_model
from .fit import grouped_participant_split
from .metrics import regression_metrics


EPOCH_PARAMETER_SPECS = {
    "equal_mixture": [],
    "static_mixture": [("q", 0.5, (0.0, 1.0))],
    "counterbalance_static": [
        ("q_logit", 0.0, (-4.0, 4.0)),
        ("counterbalance_effect", 0.5, (-4.0, 4.0)),
    ],
    "contextual_inference": [
        ("memory", 0.85, (0.05, 0.999)),
        ("cue_gain", 1.5, (0.05, 6.0)),
        ("bias", 0.0, (-0.5, 0.5)),
        ("counterbalance_effect", 0.5, (-4.0, 4.0)),
    ],
    "gate_adaptation": [
        ("q_logit", 0.0, (-4.0, 4.0)),
        ("counterbalance_effect", 0.5, (-4.0, 4.0)),
        ("gate_learning", 0.0, (-0.15, 0.15)),
        ("gate_retention", 0.9, (0.0, 1.0)),
    ],
    "responsibility_update": [
        ("q_logit", 0.0, (-4.0, 4.0)),
        ("counterbalance_effect", 0.5, (-4.0, 4.0)),
        ("learning_rate", 0.01, (0.0, 0.12)),
        ("offset_retention", 0.98, (0.5, 1.0)),
    ],
    "third_state": [
        ("q", 0.5, (0.0, 1.0)),
        ("learning_rate", 0.01, (0.0, 0.12)),
        ("retention", 0.9, (0.0, 1.0)),
    ],
}


def epoch_region_labels(data: pd.DataFrame) -> np.ndarray:
    mapping = {
        "Mixed_1": "mixed_1",
        "Mixed_2": "mixed_2",
        "Post_Mixed_Return": "post_mixed_return",
        "Mixed_Test": "mixed_test",
        "A_or_B": "ordinary_AB",
    }
    return data.epoch.astype("object").map(mapping).fillna("ordinary_AB").to_numpy(str)


def mechanism_calibration_mask(data: pd.DataFrame, through: str = "Mixed_2") -> np.ndarray:
    """Rows permitted to calibrate a mechanism in a prospective test."""
    allowed = {"Mixed_1"}
    if through == "Mixed_2":
        allowed.add("Mixed_2")
    elif through != "Mixed_1":
        raise ValueError("through must be 'Mixed_1' or 'Mixed_2'")
    return data.epoch.astype("object").isin(allowed).to_numpy()


def fit_epoch_model(
    data: pd.DataFrame,
    model_name: str,
    *,
    fit_mask: np.ndarray | pd.Series | None = None,
) -> tuple[dict[str, float], dict]:
    """Fit population mechanism parameters directly to log RT."""
    if model_name not in EPOCH_MODELS:
        raise ValueError(model_name)
    y = data.log_rt.to_numpy(float)
    mask = np.isfinite(y)
    if fit_mask is not None:
        mask &= np.asarray(fit_mask, dtype=bool)
    if not mask.any():
        raise ValueError("No finite observations are available for fitting.")
    specs = EPOCH_PARAMETER_SPECS[model_name]

    def summarize(params: dict[str, float], success: bool, message: str = "") -> tuple[dict[str, float], dict]:
        prediction = predict_epoch_arrays(data, model_name, params)["predicted_log_rt"]
        residual = y[mask] - prediction[mask]
        sse = float(np.sum(residual**2))
        return params, {
            "success": bool(success), "loss": sse / int(mask.sum()), "sse": sse,
            "n": int(mask.sum()), "message": message,
        }

    if not specs:
        return summarize({}, True)
    names = [name for name, _, _ in specs]
    x0 = np.asarray([initial for _, initial, _ in specs], dtype=float)
    bounds = [bound for _, _, bound in specs]

    def objective(values: np.ndarray) -> float:
        prediction = predict_epoch_arrays(data, model_name, dict(zip(names, values)))["predicted_log_rt"]
        residual = y[mask] - prediction[mask]
        return float(np.mean(residual**2))

    starts = [x0]
    if model_name in {"counterbalance_static", "contextual_inference", "gate_adaptation"}:
        starts.append(np.asarray([spec[2][0] * 0.1 + spec[2][1] * 0.9 for spec in specs]))
    results = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
        for start in starts
    ]
    result = min(results, key=lambda candidate: candidate.fun)
    return summarize(
        {name: float(value) for name, value in zip(names, result.x)},
        bool(result.success),
        str(result.message),
    )


def _fit_record(
    dataset: str,
    model: str,
    params: dict[str, float],
    info: dict,
    *,
    fit_scope: str,
    history: str = "combined",
) -> dict:
    n = max(int(info["n"]), 1)
    k = len(EPOCH_PARAMETER_SPECS[model])
    sigma = float(np.sqrt(max(info["sse"] / n, 1e-16)))
    return {
        "dataset": dataset,
        "model": model,
        "model_id": f"M{EPOCH_MODELS.index(model)}",
        "fit_scope": fit_scope,
        "counterbalance_history": history,
        "parameters": json.dumps(params, sort_keys=True),
        "n_parameters": k,
        "train_n": n,
        "train_loss": info["loss"],
        "train_sigma": sigma,
        "bic": float(n * np.log(max(info["sse"] / n, 1e-16)) + k * np.log(n)),
        "fit_success": info["success"],
        **params,
    }


def fit_epoch_model_ladder(
    data: pd.DataFrame,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    splits: pd.DataFrame | None = None,
    models: Iterable[str] = EPOCH_MODELS,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on training participants and score each required epoch separately."""
    models = tuple(models)
    splits = splits if splits is not None else grouped_participant_split(
        data, seed=seed, train_fraction=train_fraction
    )
    fit_rows, prediction_frames, comparison_rows = [], [], []
    for dataset, schedule in data.groupby("dataset", sort=False):
        split = splits[splits.dataset.eq(dataset)]
        train_ids = set(split.loc[split.split.eq("train"), "participant_id"])
        train = schedule[schedule.participant_id.isin(train_ids)].copy()
        if train.empty:
            train = schedule.copy()
            train_ids = set(train.participant_id)
        calibration = mechanism_calibration_mask(train, "Mixed_2")
        for model in models:
            if progress is not None:
                progress(f"Explicit model fit: {model} (combined counterbalances)")
            params, info = fit_epoch_model(train, model, fit_mask=calibration)
            record = _fit_record(dataset, model, params, info, fit_scope="combined")
            fit_rows.append(record)
            sigma = record["train_sigma"]
            predicted = predict_epoch_model(schedule, model, params)
            predicted["split"] = np.where(predicted.participant_id.isin(train_ids), "train", "test")
            predicted["evaluation_region"] = epoch_region_labels(predicted)
            prediction_frames.append(predicted)
            for split_name in ("train", "test"):
                subset = predicted[predicted.split.eq(split_name)]
                if subset.empty:
                    continue
                for region in ("all", "ordinary_AB", "mixed_1", "mixed_2", "post_mixed_return", "mixed_test"):
                    region_data = subset if region == "all" else subset[subset.evaluation_region.eq(region)]
                    comparison_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "model_id": f"M{EPOCH_MODELS.index(model)}",
                            "split": split_name,
                            "region": region,
                            "counterbalance_history": "combined",
                            "delta_q_mixed2_minus_mixed1": float(
                                predicted.loc[predicted.epoch.astype("object").eq("Mixed_2"), "q"].mean()
                                - predicted.loc[predicted.epoch.astype("object").eq("Mixed_1"), "q"].mean()
                            ),
                            **regression_metrics(
                                region_data.log_rt.to_numpy(),
                                region_data.predicted_log_rt.to_numpy(),
                                sigma=sigma,
                            ),
                        }
                    )

        # Counterbalance-specific fits are diagnostic rows, intentionally kept
        # separate from the combined out-of-participant predictions above.
        for history, history_train in train.groupby("counterbalance_history", sort=False):
            history_mask = mechanism_calibration_mask(history_train, "Mixed_2")
            for model in models:
                if progress is not None:
                    progress(f"Explicit model fit: {model} ({history})")
                params, info = fit_epoch_model(history_train, model, fit_mask=history_mask)
                fit_rows.append(
                    _fit_record(
                        dataset, model, params, info, fit_scope="separate_counterbalance",
                        history=str(history),
                    )
                )
    return (
        pd.DataFrame(fit_rows),
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        pd.DataFrame(comparison_rows),
        splits,
    )


def temporal_prediction_tests(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    models: Iterable[str] = EPOCH_MODELS,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run the three prospective tests without fitting either target epoch."""
    designs = (
        ("A_Mixed1_to_Mixed2", "Mixed_1", "Mixed_2"),
        ("B_Mixed12_to_MixedTest", "Mixed_2", "Mixed_Test"),
        ("C_Mixed12_to_PostMixedReturn", "Mixed_2", "Post_Mixed_Return"),
    )
    rows = []
    for dataset, schedule in data.groupby("dataset", sort=False):
        split = splits[splits.dataset.eq(dataset)]
        train_ids = set(split.loc[split.split.eq("train"), "participant_id"])
        train = schedule[schedule.participant_id.isin(train_ids)]
        test = schedule[~schedule.participant_id.isin(train_ids)]
        if test.empty:
            test = schedule
        for test_name, through, target in designs:
            fit_mask = mechanism_calibration_mask(train, through)
            for model in models:
                if progress is not None:
                    progress(f"Temporal prediction {test_name}: {model}")
                params, info = fit_epoch_model(train, model, fit_mask=fit_mask)
                prediction = predict_epoch_model(test, model, params)
                target_rows = prediction[prediction.epoch.astype("object").eq(target)]
                rows.append(
                    {
                        "dataset": dataset,
                        "temporal_test": test_name,
                        "calibration_through": through,
                        "target_epoch": target,
                        "target_refit": False,
                        "model": model,
                        **regression_metrics(
                            target_rows.log_rt.to_numpy(), target_rows.predicted_log_rt.to_numpy(),
                            sigma=float(np.sqrt(max(info["loss"], 1e-16))),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def counterbalance_generalization(
    data: pd.DataFrame,
    *,
    models: Iterable[str] = ("static_mixture", "contextual_inference", "gate_adaptation"),
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fit one history and predict the other after an explicit A/B reversal."""
    rows = []
    for dataset, schedule in data.groupby("dataset", sort=False):
        histories = list(schedule.counterbalance_history.dropna().unique())
        for source in histories:
            source_data = schedule[schedule.counterbalance_history.eq(source)]
            target_data = schedule[~schedule.counterbalance_history.eq(source)]
            if target_data.empty:
                continue
            mirrored_target = mirror_ab_labels(target_data)
            for model in models:
                if progress is not None:
                    progress(f"Cross-counterbalance prediction from {source}: {model}")
                params, _ = fit_epoch_model(
                    source_data, model, fit_mask=mechanism_calibration_mask(source_data, "Mixed_2")
                )
                predicted = predict_epoch_model(mirrored_target, model, params)
                for target_epoch in ("Mixed_1", "Mixed_2", "Mixed_Test"):
                    subset = predicted[predicted.epoch.astype("object").eq(target_epoch)]
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "fit_counterbalance_history": source,
                            "predicted_counterbalance_history": str(target_data.counterbalance_history.iloc[0]),
                            "target_epoch": target_epoch,
                            "ab_label_reversed": True,
                            **regression_metrics(
                                subset.log_rt.to_numpy(), subset.predicted_log_rt.to_numpy()
                            ),
                        }
                    )
    return pd.DataFrame(rows)
