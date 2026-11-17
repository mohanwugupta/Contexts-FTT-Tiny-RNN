"""Leak-free scalar, token-specific, low-rank, and frozen-SRN probe models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..evaluation.cross_validation import leave_participant_out_folds


PRIMARY_MODELS = ("M0_scalar_context", "M1_scalar_mixture", "M2_token_specific",
                  "M3_low_rank", "M4_srn_informed")


def _scalar_slope(x, y, default=0.0):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    denominator = float(x @ x)
    return float(x @ y / denominator) if denominator > 1e-14 else float(default)


def _intercept_amplitude(shape, outcome):
    x = np.column_stack([np.ones(len(shape)), np.asarray(shape, dtype=float)])
    return np.linalg.pinv(x.T @ x) @ x.T @ np.asarray(outcome, dtype=float)


def _group_token_effect(training, shrinkage):
    stable = training[training.ratio_reliable & training.responsibility_q.notna()].copy()
    stable["lambda_value"] = stable.responsibility_q - 0.5
    cells = stable.groupby("transition_position").lambda_value.agg(["mean", "count"])
    overall = float(stable.lambda_value.mean()) if len(stable) else 0.0
    weight = cells["count"] / (cells["count"] + float(shrinkage))
    effects = weight * (cells["mean"] - overall)
    return effects.to_dict(), overall


def _low_rank_shape(training, training_profiles, variance_threshold):
    participants = set(training.participant_uid)
    source = training_profiles[training_profiles.participant_uid.isin(participants)]
    matrix = source.pivot_table(
        index="participant_uid", columns="transition_position", values="delta", aggfunc="first"
    ).dropna()
    group_d = training.groupby("transition_position").midpoint_deviation.mean().sort_index()
    if len(matrix) < 2 or matrix.shape[1] < 2:
        return group_d.to_dict(), 1, 1.0
    centered = matrix.to_numpy(dtype=float) - matrix.to_numpy(dtype=float).mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular ** 2
    variance = energy / energy.sum() if energy.sum() > 1e-14 else np.ones_like(energy) / len(energy)
    dimension = int(np.searchsorted(np.cumsum(variance), variance_threshold) + 1)
    dimension = min(max(dimension, 1), len(variance))
    basis = vt[:dimension]
    vector = group_d.reindex(matrix.columns).to_numpy(dtype=float)
    reconstructed = basis.T @ (basis @ vector)
    return dict(zip(matrix.columns.astype(int), reconstructed)), dimension, float(np.cumsum(variance)[dimension - 1])


def _logit(values):
    clipped = np.clip(np.asarray(values, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def _expit(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _srn_group_slope(training, q_map):
    stable = training[training.ratio_reliable & training.responsibility_q.notna()]
    human = stable.groupby("transition_position").responsibility_q.mean()
    positions = [position for position in sorted(human.index) if position in q_map]
    if len(positions) < 2:
        return 1.0
    x = _logit([q_map[position] for position in positions])
    y = _logit([human.loc[position] for position in positions])
    design = np.column_stack([np.ones(len(x)), x])
    return float((np.linalg.pinv(design.T @ design) @ design.T @ y)[1])


def _participant_srn_intercept(training, q_map, slope):
    x = np.array([_logit([q_map[int(position)]])[0]
                  for position in training.transition_position])
    delta = training.delta.to_numpy(dtype=float)
    midpoint = training.midpoint.to_numpy(dtype=float)
    actual = training.log_rt.to_numpy(dtype=float)
    objective = lambda intercept: float(np.mean(
        (actual - (midpoint + (_expit(intercept + slope * x) - 0.5) * delta)) ** 2
    ))
    return float(minimize_scalar(objective, bounds=(-6, 6), method="bounded").x)


def _noise_sd(training):
    residual = training.log_rt - training.groupby(
        ["participant_uid", "transition_position"]
    ).log_rt.transform("mean")
    value = float(residual.std(ddof=1))
    return max(value if np.isfinite(value) else 0.0, 0.05)


def _prediction_rows(test, predicted, *, model, fold, sigma, group_count,
                     parameter_count, low_rank_dimension=np.nan, srn_variant=None):
    rows = []
    for source, value in zip(test.itertuples(index=False), np.asarray(predicted)):
        error = float(source.log_rt - value)
        rows.append({
            "model": model, "srn_variant": srn_variant,
            "probe_context": source.context, "dataset": source.dataset,
            "fold": fold, "participant_uid": source.participant_uid,
            "counterbalance": source.counterbalance,
            "probe_epoch": source.probe_epoch,
            "transition_position": int(source.transition_position),
            "sequence_repetition": int(source.sequence_repetition),
            "unit_id": source.unit_id, "actual_log_rt": float(source.log_rt),
            "predicted_log_rt": float(value), "residual": error,
            "squared_error": error ** 2, "absolute_error": abs(error),
            "heldout_log_likelihood": float(
                -0.5 * (math.log(2 * math.pi * sigma ** 2) + error ** 2 / sigma ** 2)
            ),
            "training_participant_count": int(group_count),
            "heldout_participant_in_group_training": False,
            "probe_split": "test", "participant_calibration_split": "train",
            "ab_profiles_frozen_before_probe_fit": True,
            "probe_rows_used_for_ab_profiles": 0,
            "parameter_count": int(parameter_count),
            "low_rank_dimension": low_rank_dimension,
            "target_transition_preserved": True,
        })
    return rows


def fit_probe_model_ladder(probe, profiles, srn_profile, *, config):
    """Fit group structure on training participants and score even repetitions."""
    if probe.context.nunique() != 1:
        raise ValueError("Mixed and Novel must be fit in separate model ladders.")
    probe_context = str(probe.context.iloc[0])
    srn_context = "Mixed_norm" if probe_context == "Both" else "Novel"
    selected_srn = srn_profile[srn_profile.probe_context.eq(srn_context)].set_index(
        "transition_position"
    )
    srn_columns = {
        "execution": "execution_q", "full_component": "full_component_q",
        "shared_component": "shared_component_q", "specific_component": "specific_component_q",
    }
    all_rows = []
    for fold, group_training, heldout in leave_participant_out_folds(
        probe, n_folds=config.cv_folds
    ):
        sigma = _noise_sd(group_training)
        token_effect, _ = _group_token_effect(
            group_training, config.token_effect_shrinkage_strength
        )
        low_shape, low_dimension, low_variance = _low_rank_shape(
            group_training, profiles, config.low_rank_variance_threshold
        )
        q_maps = {
            name: selected_srn[column].to_dict() for name, column in srn_columns.items()
        }
        srn_slopes = {name: _srn_group_slope(group_training, values)
                      for name, values in q_maps.items()}
        for participant, participant_rows in heldout.groupby("participant_uid", sort=False):
            calibration = participant_rows[participant_rows.probe_split.eq("train")]
            test = participant_rows[participant_rows.probe_split.eq("test")]
            if calibration.empty or test.empty:
                continue
            # M0: global pure endpoints, one participant q.
            a_global = float(calibration.r_a.mean()); b_global = float(calibration.r_b.mean())
            difference = a_global - b_global
            q0 = float((calibration.log_rt.mean() - b_global) / difference) if abs(difference) > 1e-14 else 0.5
            pred0 = np.full(len(test), q0 * a_global + (1 - q0) * b_global)
            all_rows.extend(_prediction_rows(
                test, pred0, model="M0_scalar_context", fold=fold, sigma=sigma,
                group_count=group_training.participant_uid.nunique(), parameter_count=1,
            ))
            # M1: one participant scalar on the frozen token-aware endpoints.
            lambda1 = _scalar_slope(calibration.delta, calibration.midpoint_deviation)
            pred1 = test.midpoint.to_numpy() + lambda1 * test.delta.to_numpy()
            all_rows.extend(_prediction_rows(
                test, pred1, model="M1_scalar_mixture", fold=fold, sigma=sigma,
                group_count=group_training.participant_uid.nunique(), parameter_count=1,
            ))
            # M2: training-participant token deviations plus one held-out scalar.
            calibration_effect = calibration.transition_position.map(token_effect).fillna(0).to_numpy()
            lambda2 = _scalar_slope(
                calibration.delta,
                calibration.midpoint_deviation - calibration_effect * calibration.delta,
            )
            test_effect = test.transition_position.map(token_effect).fillna(0).to_numpy()
            pred2 = test.midpoint.to_numpy() + (lambda2 + test_effect) * test.delta.to_numpy()
            all_rows.extend(_prediction_rows(
                test, pred2, model="M2_token_specific", fold=fold, sigma=sigma,
                group_count=group_training.participant_uid.nunique(),
                parameter_count=1 + len(token_effect),
            ))
            # M3: training-only PCA basis and a two-parameter held-out calibration.
            calibration_shape = calibration.transition_position.map(low_shape).fillna(0).to_numpy()
            low_beta = _intercept_amplitude(calibration_shape, calibration.midpoint_deviation)
            test_shape = test.transition_position.map(low_shape).fillna(0).to_numpy()
            pred3 = test.midpoint.to_numpy() + low_beta[0] + low_beta[1] * test_shape
            all_rows.extend(_prediction_rows(
                test, pred3, model="M3_low_rank", fold=fold, sigma=sigma,
                group_count=group_training.participant_uid.nunique(),
                parameter_count=2 + low_dimension, low_rank_dimension=low_dimension,
            ))
            # M4 primary calibrated execution q; strict and component variants are secondary.
            for variant, q_map in q_maps.items():
                if set(test.transition_position) - set(q_map):
                    raise ValueError("Frozen SRN profile is missing a human transition position.")
                strict_q = test.transition_position.map(q_map).to_numpy(dtype=float)
                strict_pred = test.midpoint.to_numpy() + (strict_q - 0.5) * test.delta.to_numpy()
                strict_name = "M4_srn_strict" if variant == "execution" else f"M4_srn_{variant}_strict"
                all_rows.extend(_prediction_rows(
                    test, strict_pred, model=strict_name, srn_variant=variant,
                    fold=fold, sigma=sigma, group_count=group_training.participant_uid.nunique(),
                    parameter_count=0,
                ))
                slope = srn_slopes[variant]
                intercept = _participant_srn_intercept(calibration, q_map, slope)
                x = _logit(test.transition_position.map(q_map).to_numpy(dtype=float))
                calibrated_q = _expit(intercept + slope * x)
                calibrated_pred = test.midpoint.to_numpy() + (calibrated_q - 0.5) * test.delta.to_numpy()
                name = "M4_srn_informed" if variant == "execution" else f"M4_srn_{variant}"
                all_rows.extend(_prediction_rows(
                    test, calibrated_pred, model=name, srn_variant=variant,
                    fold=fold, sigma=sigma, group_count=group_training.participant_uid.nunique(),
                    parameter_count=2,
                ))
    result = pd.DataFrame(all_rows)
    if result.empty:
        raise ValueError(f"No held-out {probe_context} predictions were produced.")
    return result


def summarize_model_comparison(predictions, *, primary_only=True):
    frame = predictions[
        predictions.model.isin(PRIMARY_MODELS)
    ].copy() if primary_only else predictions.copy()
    rows = []
    for (probe, model), group in frame.groupby(["probe_context", "model"], sort=False):
        n = len(group); parameters = int(group.parameter_count.max())
        sse = max(float(group.squared_error.sum()), 1e-14)
        rows.append({
            "probe_context": probe, "model": model,
            "heldout_log_likelihood": float(group.heldout_log_likelihood.sum()),
            "mean_heldout_log_likelihood": float(group.heldout_log_likelihood.mean()),
            "heldout_rmse": float(np.sqrt(group.squared_error.mean())),
            "heldout_mae": float(group.absolute_error.mean()),
            "parameter_count": parameters,
            "descriptive_aic": float(n * np.log(sse / n) + 2 * parameters),
            "descriptive_bic": float(n * np.log(sse / n) + parameters * np.log(n)),
            "heldout_observations": n,
            "heldout_participants": int(group.participant_uid.nunique()),
            "primary_metric": "heldout_log_likelihood",
        })
    result = pd.DataFrame(rows)
    result["primary_rank"] = result.groupby("probe_context").heldout_log_likelihood.rank(
        ascending=False, method="min"
    ).astype(int)
    return result.sort_values(["probe_context", "primary_rank"]).reset_index(drop=True)

