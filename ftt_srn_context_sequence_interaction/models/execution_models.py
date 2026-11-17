"""Capacity-controlled sequence, context, additive, and interaction regressions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MODEL_NAMES = ("sequence_only", "context_only", "additive", "interaction")


@dataclass(frozen=True)
class ExecutionModelResult:
    predictions: pd.DataFrame
    comparison: pd.DataFrame
    descriptive_coefficients: pd.DataFrame


def leave_one_seed_out_splits(frame: pd.DataFrame):
    seeds = sorted(frame.seed.unique())
    if len(seeds) < 2:
        raise ValueError("Leave-one-seed-out evaluation requires at least two seeds.")
    for held_out in seeds:
        training = frame[frame.seed != held_out].copy()
        test = frame[frame.seed == held_out].copy()
        if held_out in set(training.seed):
            raise AssertionError("Held-out seed leaked into model training.")
        yield int(held_out), training, test


def _token_levels(frame):
    return tuple(int(value) for value in sorted(frame.token_position.unique()))


def design_matrix(frame: pd.DataFrame, model: str, *, token_levels) -> tuple[np.ndarray, list[str]]:
    """Use aligned token identities as f_seq(mu_k); hidden axes are not aligned across seeds."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown execution model {model!r}.")
    columns = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    if model in {"sequence_only", "additive", "interaction"}:
        for position in token_levels[1:]:
            columns.append((frame.token_position.to_numpy() == position).astype(float))
            names.append(f"token_{position}")
    if model in {"context_only", "additive", "interaction"}:
        z = frame.z_context.to_numpy(dtype=float)
        columns.append(z)
        names.append("z_context")
    if model == "interaction":
        z = frame.z_context.to_numpy(dtype=float)
        for position in token_levels[1:]:
            columns.append(z * (frame.token_position.to_numpy() == position))
            names.append(f"z_context_x_token_{position}")
    return np.column_stack(columns), names


def _fit_beta(x, y, ridge):
    penalty = np.eye(x.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def _r2(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return float(1 - np.sum((actual - predicted) ** 2) / denominator) if denominator > 1e-20 else np.nan


def _validate_natural(frame):
    if set(frame.context) - {"A", "B"}:
        raise ValueError("Execution models may be fit only to natural A/B outcomes.")
    required = {"seed", "checkpoint", "context", "token_position", "z_context", "execution_speed"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Execution model frame is missing {sorted(missing)}.")


def fit_execution_models(natural: pd.DataFrame, *, ridge: float = 1e-8) -> ExecutionModelResult:
    _validate_natural(natural)
    prediction_rows, comparison_rows, coefficient_rows = [], [], []
    for checkpoint, checkpoint_rows in natural.groupby("checkpoint", sort=False):
        levels = _token_levels(checkpoint_rows)
        for model in MODEL_NAMES:
            fold_metrics = []
            for held_out, training, test in leave_one_seed_out_splits(checkpoint_rows):
                x_train, names = design_matrix(training, model, token_levels=levels)
                x_test, _ = design_matrix(test, model, token_levels=levels)
                y_train = training.execution_speed.to_numpy(dtype=float)
                y_test = test.execution_speed.to_numpy(dtype=float)
                beta = _fit_beta(x_train, y_train, ridge)
                predicted = x_test @ beta
                train_residual = y_train - x_train @ beta
                variance = max(float(np.mean(train_residual ** 2)), 1e-12)
                log_likelihood = float(np.sum(
                    -0.5 * (np.log(2 * np.pi * variance) + (y_test - predicted) ** 2 / variance)
                ))
                mse = float(np.mean((y_test - predicted) ** 2))
                fold_metrics.append((mse, _r2(y_test, predicted), log_likelihood))
                for source_row, value in zip(test.itertuples(index=False), predicted):
                    prediction_rows.append({
                        "model": model, "checkpoint": checkpoint, "held_out_seed": held_out,
                        "seed": int(source_row.seed), "context": source_row.context,
                        "token_position": int(source_row.token_position),
                        "z_context": float(source_row.z_context),
                        "actual_execution_speed": float(source_row.execution_speed),
                        "predicted_execution_speed": float(value),
                        "squared_error": float((source_row.execution_speed - value) ** 2),
                        "training_contexts": "A|B",
                        "training_seed_count": int(training.seed.nunique()),
                        "held_out_seed_in_training": False,
                    })
            x_all, names = design_matrix(checkpoint_rows, model, token_levels=levels)
            y_all = checkpoint_rows.execution_speed.to_numpy(dtype=float)
            beta = _fit_beta(x_all, y_all, ridge)
            residual = y_all - x_all @ beta
            sse = max(float(residual @ residual), 1e-20)
            n, parameters = len(y_all), len(beta)
            aic = float(n * np.log(sse / n) + 2 * parameters)
            bic = float(n * np.log(sse / n) + parameters * np.log(n))
            for name, value in zip(names, beta):
                coefficient_rows.append({
                    "checkpoint": checkpoint, "model": model, "term": name,
                    "coefficient": float(value), "training_contexts": "A|B",
                })
            comparison_rows.append({
                "checkpoint": checkpoint, "model": model,
                "mean_held_out_mse": float(np.mean([value[0] for value in fold_metrics])),
                "cross_validated_r2": _r2(
                    [row["actual_execution_speed"] for row in prediction_rows
                     if row["checkpoint"] == checkpoint and row["model"] == model],
                    [row["predicted_execution_speed"] for row in prediction_rows
                     if row["checkpoint"] == checkpoint and row["model"] == model],
                ),
                "mean_held_out_log_likelihood": float(np.mean([value[2] for value in fold_metrics])),
                "descriptive_aic": aic, "descriptive_bic": bic,
                "parameter_count": parameters, "training_contexts": "A|B",
                "mixed_outcomes_used_for_fit": False, "novel_outcomes_used_for_fit": False,
            })
    return ExecutionModelResult(
        pd.DataFrame(prediction_rows), pd.DataFrame(comparison_rows),
        pd.DataFrame(coefficient_rows),
    )


def fit_fold_models_and_predict(natural: pd.DataFrame, probes: pd.DataFrame, *, ridge: float = 1e-8):
    """Predict each held-out seed's probe states from other seeds' A/B outcomes only."""
    _validate_natural(natural)
    rows = []
    for checkpoint, checkpoint_rows in natural.groupby("checkpoint", sort=False):
        checkpoint_probes = probes[probes.checkpoint == checkpoint]
        levels = _token_levels(checkpoint_rows)
        for held_out, training, _ in leave_one_seed_out_splits(checkpoint_rows):
            test = checkpoint_probes[checkpoint_probes.seed == held_out]
            for model in MODEL_NAMES:
                x_train, _ = design_matrix(training, model, token_levels=levels)
                x_test, _ = design_matrix(test, model, token_levels=levels)
                beta = _fit_beta(x_train, training.execution_speed.to_numpy(dtype=float), ridge)
                predicted = x_test @ beta
                for probe_row, value in zip(test.itertuples(index=False), predicted):
                    rows.append({
                        "model": model, "checkpoint": checkpoint, "held_out_seed": held_out,
                        "seed": held_out, "context": probe_row.context,
                        "token_position": int(probe_row.token_position),
                        "z_context": float(probe_row.z_context),
                        "actual_execution_speed": float(probe_row.execution_speed),
                        "predicted_execution_speed": float(value),
                        "squared_error": float((probe_row.execution_speed - value) ** 2),
                        "training_contexts": "A|B", "probe_outcomes_used_for_fit": False,
                        "held_out_seed_in_training": False,
                    })
    return pd.DataFrame(rows)

