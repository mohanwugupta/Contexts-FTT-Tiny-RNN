"""Canonical Lump/Cross/Split fits for Phase-1 emergent timing traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from state_space_bridge.state_space import fit_model_family


MODEL_LABELS = {"SS_LUMP": "Lump", "SS_CROSS": "Cross", "SS_SPLIT": "Split"}


def _empty_tables() -> dict[str, pd.DataFrame]:
    return {
        "screen_state_space_trace": pd.DataFrame(),
        "screen_state_space_fits": pd.DataFrame(),
        "screen_state_space_parameters": pd.DataFrame(),
        "screen_state_space_predictions": pd.DataFrame(),
    }


def _fit_one(trace: pd.DataFrame, metadata: dict) -> tuple[list, list, list]:
    fits = fit_model_family(trace, observed_column="observed")
    minimum = min(fit.bic for fit in fits.values())
    raw_weights = {
        name: np.exp(-.5 * (fit.bic - minimum)) for name, fit in fits.items()
    }
    denominator = sum(raw_weights.values())
    winner = min(fits, key=lambda name: fits[name].bic)
    fit_rows, parameter_rows, prediction_rows = [], [], []
    for model_name, fit in fits.items():
        delta = float(fit.bic - minimum)
        fit_rows.append({
            **metadata,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "n_observations": len(fit.observed),
            "RSS": fit.rss,
            "RMSE": fit.rmse,
            "R2": fit.r2,
            "log_likelihood": fit.log_likelihood,
            "BIC": fit.bic,
            "delta_BIC": delta,
            "BIC_weight": raw_weights[model_name] / denominator,
            "winner": model_name == winner,
            "cross_competitive": model_name == "SS_CROSS" and delta <= 2,
            "converged": fit.converged,
            "optimizer_message": fit.optimizer_message,
            "canonical_R_source": "state-space_SRN.Rmd no-fatigue models",
            "human_rows_used": 0,
        })
        for name, value in zip(fit.parameter_names, fit.parameters):
            parameter_rows.append({
                **metadata, "model": model_name,
                "parameter": name, "value": float(value),
            })
        for name, value in zip(fit.decoder_names, fit.decoder):
            parameter_rows.append({
                **metadata, "model": model_name,
                "parameter": f"decoder_{name}", "value": float(value),
            })
        for index, row in trace.reset_index(drop=True).iterrows():
            prediction_rows.append({
                **metadata,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
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
                "winner": model_name == winner,
            })
    return fit_rows, parameter_rows, prediction_rows


def fit_screen_state_space(
    training: pd.DataFrame,
    gate_summary: pd.DataFrame,
    threshold_sequences: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit passing learned candidates and every prespecified threshold control."""
    tables = _empty_tables()
    trace_rows, fit_rows, parameter_rows, prediction_rows = [], [], [], []
    passing = set(
        gate_summary.loc[gate_summary.screening_passed.eq(True), "c_time"].astype(float)
    )
    sources: list[tuple[pd.DataFrame, dict]] = []
    for c_time in sorted(passing):
        sources.append((
            training[training.c_time.eq(c_time)],
            {
                "timing_condition": f"learned_c_time_{c_time:g}",
                "policy_type": "learned_hazard",
                "c_time": c_time,
                "threshold": np.nan,
            },
        ))
    if threshold_sequences is not None and not threshold_sequences.empty:
        for (c_time, threshold), group in threshold_sequences.groupby(
            ["c_time", "threshold"], sort=True
        ):
            sources.append((
                group,
                {
                    "timing_condition": f"threshold_c_time_{c_time:g}_theta_{threshold:g}",
                    "policy_type": "fixed_confidence_threshold",
                    "c_time": float(c_time),
                    "threshold": float(threshold),
                },
            ))
    for source, metadata in sources:
        columns = ["global_sequence", "block_index", "sequence_in_block", "context"]
        trace = (
            source.groupby(columns, as_index=False, sort=True).normalized_rt
            .agg(observed="mean", sd="std", n_seeds="count")
        )
        trace["sd"] = trace.sd.fillna(0.0)
        trace["se"] = trace.sd / np.sqrt(trace.n_seeds)
        for key, value in metadata.items():
            trace[key] = value
        trace_rows.append(trace)
        one_fits, one_parameters, one_predictions = _fit_one(trace, metadata)
        fit_rows.extend(one_fits)
        parameter_rows.extend(one_parameters)
        prediction_rows.extend(one_predictions)
    if trace_rows:
        tables["screen_state_space_trace"] = pd.concat(trace_rows, ignore_index=True)
        tables["screen_state_space_fits"] = pd.DataFrame(fit_rows)
        tables["screen_state_space_parameters"] = pd.DataFrame(parameter_rows)
        tables["screen_state_space_predictions"] = pd.DataFrame(prediction_rows)
    return tables


def fit_confirmatory_state_space(training: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit canonical models separately to High/Low learned commitment traces."""
    trace_rows, fit_rows, parameter_rows, prediction_rows = [], [], [], []
    learned = training[training.commitment_policy_active.eq(True)]
    for condition, source in learned.groupby("condition", sort=False):
        columns = ["global_sequence", "block_index", "sequence_in_block", "context"]
        trace = (
            source.groupby(columns, as_index=False, sort=True).normalized_rt
            .agg(observed="mean", sd="std", n_seeds="count")
        )
        trace["sd"] = trace.sd.fillna(0.0)
        trace["se"] = trace.sd / np.sqrt(trace.n_seeds)
        metadata = {
            "timing_condition": condition,
            "policy_type": "learned_hazard",
            "c_time": float(source.c_time.iloc[0]),
            "threshold": np.nan,
        }
        for key, value in metadata.items():
            trace[key] = value
        trace_rows.append(trace)
        one_fits, one_parameters, one_predictions = _fit_one(trace, metadata)
        fit_rows.extend(one_fits)
        parameter_rows.extend(one_parameters)
        prediction_rows.extend(one_predictions)
    return {
        "final_state_space_trace": (
            pd.concat(trace_rows, ignore_index=True) if trace_rows else pd.DataFrame()
        ),
        "final_state_space_fits": pd.DataFrame(fit_rows),
        "final_state_space_parameters": pd.DataFrame(parameter_rows),
        "final_state_space_predictions": pd.DataFrame(prediction_rows),
    }
