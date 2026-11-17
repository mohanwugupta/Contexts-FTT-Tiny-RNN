"""Read-only pipeline for SRN trial traces and R-style state-space overlays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import (
    plot_observed_vs_predicted,
    plot_state_space_overlays,
    plot_trial_trajectories,
)
from .state_space import fit_three_models


REQUIRED_COLUMNS = {
    "seed", "model_condition", "global_sequence", "block_index",
    "sequence_in_block", "context", "sequence_ce", "total_loss",
    "execution_speed", "normalized_rt", "execution_objective_active",
}
CONDITION_ORDER = ["High", "Low", "High_seq_only"]


def validate_training_dynamics(
    frame: pd.DataFrame,
    *,
    expected_sequences: int = 170,
    rt_tolerance: float = 5e-6,
) -> dict:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"training_dynamics is missing required columns: {missing}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("training_dynamics contains missing values in required columns.")
    key = ["seed", "model_condition", "global_sequence"]
    duplicate_count = int(frame.duplicated(key).sum())
    if duplicate_count:
        raise ValueError(f"training_dynamics has {duplicate_count} duplicate seed/condition/trial rows.")

    expected = np.arange(1, expected_sequences + 1)
    for (condition, seed), group in frame.groupby(["model_condition", "seed"], sort=False):
        actual = np.sort(group.global_sequence.to_numpy(dtype=int))
        if not np.array_equal(actual, expected):
            raise ValueError(f"{condition} seed {seed} does not contain exactly trials 1-{expected_sequences}.")

    block_contract = frame[["global_sequence", "block_index", "sequence_in_block", "context"]].drop_duplicates()
    if len(block_contract) != expected_sequences:
        raise ValueError("Conditions/seeds disagree about the fixed training schedule.")
    block_contract = block_contract.sort_values("global_sequence")
    expected_block = (expected - 1) // 10 + 1
    expected_within = (expected - 1) % 10 + 1
    expected_context = np.where(expected_block % 2 == 1, "A", "B")
    if not np.array_equal(block_contract.block_index.to_numpy(dtype=int), expected_block):
        raise ValueError("block_index does not implement the frozen A10/B10 curriculum.")
    if not np.array_equal(block_contract.sequence_in_block.to_numpy(dtype=int), expected_within):
        raise ValueError("sequence_in_block does not implement ten sequences per block.")
    if not np.array_equal(block_contract.context.astype(str).to_numpy(), expected_context):
        raise ValueError("context labels do not alternate A10/B10 starting with A.")

    rt_error = np.abs(frame.normalized_rt.to_numpy(dtype=float)
                      - (1.0 - frame.execution_speed.to_numpy(dtype=float)))
    max_rt_error = float(rt_error.max())
    if max_rt_error > rt_tolerance:
        raise ValueError(
            f"normalized_rt is not 1 - execution_speed (max error={max_rt_error:.6g})."
        )

    seeds = frame.groupby("model_condition").seed.nunique().astype(int).to_dict()
    objectives = frame.groupby("model_condition").execution_objective_active.nunique()
    if (objectives != 1).any():
        raise ValueError("execution_objective_active changes within a model condition.")
    return {
        "rows": int(len(frame)),
        "expected_sequences": int(expected_sequences),
        "n_conditions": int(frame.model_condition.nunique()),
        "n_seeds_per_condition": seeds,
        "duplicate_rows": duplicate_count,
        "missing_required_values": 0,
        "max_rt_identity_error": max_rt_error,
        "context_by_block": {
            str(int(block)): str(context)
            for block, context in block_contract.groupby("block_index").context.first().items()
        },
    }


def prepare_average_trace(
    frame: pd.DataFrame,
    condition: str,
    outcome: str,
    *,
    max_sequence: int | None = None,
) -> pd.DataFrame:
    if outcome not in frame.columns:
        raise ValueError(f"Unknown outcome {outcome!r}.")
    subset = frame[frame.model_condition.eq(condition)].copy()
    if max_sequence is not None:
        subset = subset[subset.global_sequence.le(max_sequence)]
    if subset.empty:
        raise ValueError(f"No rows for condition {condition!r}.")
    group_columns = ["global_sequence", "block_index", "sequence_in_block", "context"]
    summary = (
        subset.groupby(group_columns, as_index=False, sort=True)[outcome]
        .agg(observed="mean", sd="std", n_seeds="count")
        .sort_values("global_sequence")
        .reset_index(drop=True)
    )
    summary["se"] = summary.sd.fillna(0.0) / np.sqrt(summary.n_seeds)
    summary["ci_low"] = summary.observed - 1.96 * summary.se
    summary["ci_high"] = summary.observed + 1.96 * summary.se
    summary["model_condition"] = condition
    summary["outcome"] = outcome
    return summary


def _fit_all(summaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metric_rows = []
    for (outcome, condition), trace in summaries.groupby(
        ["outcome", "model_condition"], sort=False
    ):
        trace = trace.sort_values("global_sequence").reset_index(drop=True)
        fits = fit_three_models(trace.observed.to_numpy(), trace.context.to_numpy())
        for model, fit in fits.items():
            dynamic = json.dumps(fit.dynamic_parameters, sort_keys=True)
            metric_rows.append({
                "outcome": outcome,
                "model_condition": condition,
                "model": model,
                "n_trials": len(trace),
                "parameter_count": fit.parameter_count,
                "rss": fit.rss,
                "rmse": fit.rmse,
                "r_squared": fit.r_squared,
                "bic": fit.bic,
                "optimizer_success": fit.optimizer_success,
                "optimizer_message": fit.optimizer_message,
                "dynamic_parameters": dynamic,
                "decoder_intercept": fit.decoder_intercept,
                "decoder_slope": fit.decoder_slope,
            })
            for record, predicted in zip(trace.to_dict("records"), fit.predicted):
                prediction_rows.append({
                    **record,
                    "model": model,
                    "predicted": float(predicted),
                    "residual": float(record["observed"] - predicted),
                })
    metrics = pd.DataFrame(metric_rows)
    metrics["delta_bic"] = metrics.bic - metrics.groupby(
        ["outcome", "model_condition"]
    ).bic.transform("min")
    unnormalized = np.exp(-0.5 * metrics.delta_bic)
    metrics["bic_weight"] = unnormalized / unnormalized.groupby(
        [metrics.outcome, metrics.model_condition]
    ).transform("sum")
    metrics["bic_rank"] = metrics.groupby(
        ["outcome", "model_condition"]
    ).bic.rank(method="min").astype(int)
    return pd.DataFrame(prediction_rows), metrics.sort_values(
        ["outcome", "model_condition", "bic"]
    ).reset_index(drop=True)


def run_validation_diagnostics(
    training_path: str | Path,
    output_dir: str | Path,
    *,
    max_sequence: int = 170,
) -> dict:
    training_path = Path(training_path).resolve()
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(training_path)
    audit = validate_training_dynamics(training, expected_sequences=170)
    if max_sequence < 20 or max_sequence > 170:
        raise ValueError("max_sequence must be between 20 and the frozen total of 170.")
    training = training[training.global_sequence.le(max_sequence)].copy()
    audit["analyzed_sequences"] = int(max_sequence)

    active_rt_conditions = [
        condition for condition in CONDITION_ORDER
        if condition in set(training.model_condition)
        and bool(training.loc[
            training.model_condition.eq(condition), "execution_objective_active"
        ].iloc[0])
    ]
    loss_conditions = [condition for condition in CONDITION_ORDER
                       if condition in set(training.model_condition)]

    summaries = []
    for condition in loss_conditions:
        summaries.append(prepare_average_trace(training, condition, "sequence_ce"))
        summaries.append(prepare_average_trace(training, condition, "total_loss"))
    for condition in active_rt_conditions:
        summaries.append(prepare_average_trace(training, condition, "normalized_rt"))
    summary = pd.concat(summaries, ignore_index=True)

    fit_summary = summary[summary.outcome.isin(["sequence_ce", "normalized_rt"])].copy()
    predictions, metrics = _fit_all(fit_summary)

    summary.to_csv(output_dir / "trialwise_observed_summary.csv", index=False)
    predictions.to_csv(output_dir / "state_space_predictions.csv", index=False)
    metrics.to_csv(output_dir / "state_space_fit_metrics.csv", index=False)

    audit.update({
        "source": str(training_path),
        "source_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "loss_conditions": loss_conditions,
        "normalized_rt_conditions": active_rt_conditions,
        "normalized_rt_definition": "1 - execution_speed (unitless; not milliseconds)",
        "state_space_reference": "state-space_SRN.Rmd lines 210-706, no-fatigue fits",
        "state_space_fit_unit": "condition-level mean over seeds at each training sequence",
        "state_space_status": "exploratory visual diagnostic; not part of frozen model selection",
    })
    (output_dir / "validation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )

    plot_trial_trajectories(
        training,
        summary,
        outcome="sequence_ce",
        conditions=loss_conditions,
        ylabel="Sequence cross-entropy",
        title="SRN sequence loss across all 170 training sequences",
        output_path=figures_dir / "figure_01_sequence_ce_over_trials.png",
    )
    plot_trial_trajectories(
        training,
        summary,
        outcome="total_loss",
        conditions=loss_conditions,
        ylabel="Total objective loss",
        title="SRN total training objective across all 170 sequences",
        output_path=figures_dir / "figure_02_total_loss_over_trials.png",
    )
    plot_trial_trajectories(
        training,
        summary,
        outcome="normalized_rt",
        conditions=active_rt_conditions,
        ylabel="Model normalized RT (1 − speed)",
        title="Emergent model RT output across training",
        output_path=figures_dir / "figure_03_normalized_rt_over_trials.png",
    )
    plot_state_space_overlays(
        predictions,
        metrics,
        outcome="sequence_ce",
        conditions=loss_conditions,
        ylabel="Observed / fitted sequence CE",
        title="R-style Lump/Cross/Split fits over observed SRN loss",
        output_path=figures_dir / "figure_04_state_space_loss_overlay.png",
    )
    plot_state_space_overlays(
        predictions,
        metrics,
        outcome="normalized_rt",
        conditions=active_rt_conditions,
        ylabel="Observed / fitted normalized RT",
        title="Exploratory state-space fits over emergent SRN RT",
        output_path=figures_dir / "figure_05_state_space_rt_overlay.png",
    )
    plot_observed_vs_predicted(
        predictions,
        output_path=figures_dir / "figure_06_observed_vs_predicted.png",
    )

    winners = metrics[metrics.bic_rank.eq(1)][
        ["outcome", "model_condition", "model", "rmse", "r_squared", "bic", "bic_weight"]
    ]
    readme = f"""# SRN trial-validation diagnostics

This folder is a read-only visual check of the frozen v2.1 training outputs. It
does not retrain an SRN and is not part of the registered paper model-selection
pipeline.

- Figures 1-3 show every seed faintly, the cohort mean by A/B block, and a 95%
  normal-approximation CI over seeds.
- Sequence CE is the next-token loss. Total loss additionally includes the
  execution objective where active.
- Model normalized RT is exactly `1 - execution_speed`; it is unitless and is
  not a prediction in milliseconds. The sequence-only head is omitted because
  its execution output was never trained.
- Figures 4-5 translate the no-fatigue Lump/Cross/Split fits from
  `state-space_SRN.Rmd` and overlay all predictions on the actual cohort-average
  trial trace.
- Figure 6 shows observed-versus-predicted values with an identity line.

Lowest-BIC descriptive fit in each panel:

```text
{winners.to_string(index=False)}
```

Use the prediction and metric CSVs to reproduce the same plots in R. BIC here is
descriptive and in-sample, exactly as in the reference notebook; it does not
replace the frozen confirmatory analyses.
"""
    (output_dir / "README.md").write_text(readme)
    return {
        "audit": audit,
        "summary": summary,
        "predictions": predictions,
        "metrics": metrics,
        "winners": winners.reset_index(drop=True),
    }
