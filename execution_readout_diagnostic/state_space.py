"""Conditional R-style Lump/Cross/Split fits for passing readout conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from state_space_bridge.state_space import fit_model_family
from srn_validation.state_space import fit_three_models


MODEL_LABELS = {
    "SS_LUMP": "Lump",
    "SS_CROSS": "Cross",
    "SS_SPLIT": "Split",
}


def _empty_tables() -> dict[str, pd.DataFrame]:
    return {
        "state_space_trace": pd.DataFrame(columns=[
            "readout_condition", "global_sequence", "block_index",
            "sequence_in_block", "context", "observed", "sd", "se", "n_seeds",
        ]),
        "state_space_model_fits": pd.DataFrame(columns=[
            "readout_condition", "model", "model_label", "n_observations",
            "RSS", "RMSE", "R2", "log_likelihood", "BIC", "delta_BIC",
            "BIC_weight", "winner", "cross_competitive", "converged",
            "optimizer_message", "secondary_port_converged",
            "secondary_port_BIC", "cross_port_BIC_absolute_difference",
            "cross_port_prediction_max_abs_difference", "convergence_verified",
            "canonical_R_source", "human_rows_used",
        ]),
        "state_space_parameters": pd.DataFrame(columns=[
            "readout_condition", "model", "parameter", "value",
        ]),
        "state_space_decoders": pd.DataFrame(columns=[
            "readout_condition", "model", "decoder_term", "value",
        ]),
        "state_space_predictions": pd.DataFrame(columns=[
            "readout_condition", "model", "global_sequence", "block_index",
            "sequence_in_block", "context", "observed", "predicted", "residual",
            "x_readout", "x_A", "x_B", "BIC", "delta_BIC", "winner",
        ]),
    }


def fit_passing_readouts(
    sequence_results: pd.DataFrame,
    gate_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Fit the cohort-average RT trace only after all A--E gates pass."""
    tables = _empty_tables()
    passing = gate_summary.loc[
        gate_summary.screening_passed.eq(True), "readout_condition"
    ].tolist()
    if not passing:
        return tables

    trace_frames, fit_rows, parameter_rows, decoder_rows, prediction_rows = [], [], [], [], []
    group_columns = [
        "readout_condition", "global_sequence", "block_index",
        "sequence_in_block", "context",
    ]
    source = sequence_results[sequence_results.readout_condition.isin(passing)]
    trace = (
        source.groupby(group_columns, as_index=False, sort=True).normalized_rt
        .agg(observed="mean", sd="std", n_seeds="count")
    )
    trace["sd"] = trace.sd.fillna(0.0)
    trace["se"] = trace.sd / np.sqrt(trace.n_seeds)
    trace_frames.append(trace)

    for readout_condition, condition_trace in trace.groupby(
        "readout_condition", sort=False
    ):
        condition_trace = condition_trace.sort_values("global_sequence").reset_index(drop=True)
        fits = fit_model_family(condition_trace, observed_column="observed")
        secondary_fits = fit_three_models(
            condition_trace.observed.to_numpy(dtype=float),
            condition_trace.context.to_numpy(dtype=str),
        )
        minimum_bic = min(fit.bic for fit in fits.values())
        winner_model = min(fits, key=lambda name: fits[name].bic)
        raw_weights = {
            name: np.exp(-.5 * (fit.bic - minimum_bic)) for name, fit in fits.items()
        }
        weight_total = sum(raw_weights.values())
        for model, fit in fits.items():
            secondary = secondary_fits[MODEL_LABELS[model]]
            bic_difference = abs(float(fit.bic - secondary.bic))
            prediction_difference = float(np.max(np.abs(
                fit.predicted - secondary.predicted
            )))
            secondary_agrees = bool(
                secondary.optimizer_success
                and bic_difference <= 1e-6
                and prediction_difference <= 1e-6
            )
            delta = fit.bic - minimum_bic
            common = {
                "readout_condition": readout_condition,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_observations": len(fit.observed),
                "RSS": fit.rss,
                "RMSE": fit.rmse,
                "R2": fit.r2,
                "log_likelihood": fit.log_likelihood,
                "BIC": fit.bic,
                "delta_BIC": delta,
                "BIC_weight": raw_weights[model] / weight_total,
                "winner": model == winner_model,
                "cross_competitive": model == "SS_CROSS" and delta <= 2.0,
                "converged": fit.converged,
                "optimizer_message": fit.optimizer_message,
                "secondary_port_converged": secondary.optimizer_success,
                "secondary_port_BIC": secondary.bic,
                "cross_port_BIC_absolute_difference": bic_difference,
                "cross_port_prediction_max_abs_difference": prediction_difference,
                "convergence_verified": bool(fit.converged or secondary_agrees),
                "canonical_R_source": "state-space_SRN.Rmd no-fatigue models",
                "human_rows_used": 0,
            }
            fit_rows.append(common)
            for name, value in zip(fit.parameter_names, fit.parameters):
                parameter_rows.append({
                    "readout_condition": readout_condition,
                    "model": model,
                    "parameter": name,
                    "value": float(value),
                })
            for name, value in zip(fit.decoder_names, fit.decoder):
                decoder_rows.append({
                    "readout_condition": readout_condition,
                    "model": model,
                    "decoder_term": name,
                    "value": float(value),
                })
            for index, row in condition_trace.iterrows():
                prediction_rows.append({
                    "readout_condition": readout_condition,
                    "model": model,
                    "global_sequence": int(row.global_sequence),
                    "block_index": int(row.block_index),
                    "sequence_in_block": int(row.sequence_in_block),
                    "context": row.context,
                    "observed": float(row.observed),
                    "predicted": float(fit.predicted[index]),
                    "residual": float(fit.observed[index] - fit.predicted[index]),
                    "x_readout": float(fit.x_readout[index]),
                    "x_A": float(fit.x_a[index]),
                    "x_B": float(fit.x_b[index]),
                    "BIC": fit.bic,
                    "delta_BIC": delta,
                    "winner": model == winner_model,
                })

    tables["state_space_trace"] = pd.concat(trace_frames, ignore_index=True)
    tables["state_space_model_fits"] = pd.DataFrame(fit_rows)
    tables["state_space_parameters"] = pd.DataFrame(parameter_rows)
    tables["state_space_decoders"] = pd.DataFrame(decoder_rows)
    tables["state_space_predictions"] = pd.DataFrame(prediction_rows)
    return tables
