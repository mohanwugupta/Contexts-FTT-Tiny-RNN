"""Zero-shot bidirectional transfer of retrieval parameters."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from probe_modeling.metrics import regression_metrics

from .explicit_models import predict_retrieval_model
from .fitting import fit_retrieval_model


def _ab_only_sigma(data: pd.DataFrame) -> float:
    established = data.retrieval_context.isin(["A", "B"])
    expert = np.where(data.retrieval_context.eq("A"), data.pred_A_log_rt, data.pred_B_log_rt)
    residual = data.log_rt.to_numpy(float) - expert - data.get(
        "participant_intercept", pd.Series(0.0, index=data.index)
    ).to_numpy(float)
    sigma = float(np.std(residual[established], ddof=1)) if established.sum() > 1 else 0.1
    return float(np.clip(sigma, 0.01, 1.0))


def cross_dataset_transfer(
    data: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
    model_name: str,
) -> pd.DataFrame:
    source = data[data.dataset.eq(source_dataset)].copy()
    target = data[data.dataset.eq(target_dataset)].copy()
    if source.empty or target.empty:
        return pd.DataFrame()
    params, info = fit_retrieval_model(source, model_name)
    # Target residual variance is a permitted nuisance estimated only from A/B;
    # no target probe behavior estimates a retrieval parameter.
    params[f"log_sigma_{target_dataset}"] = float(np.log(_ab_only_sigma(target)))
    predicted = predict_retrieval_model(target, model_name, params)
    probe = predicted[~predicted.retrieval_context.isin(["A", "B"])]

    def metric_row(rows: pd.DataFrame, scope: str, context: str, cohort: str) -> dict:
        metrics = regression_metrics(rows.log_rt.to_numpy(), rows.predicted_log_rt.to_numpy())
        return {
            "source_dataset": source_dataset, "target_dataset": target_dataset,
            "model": model_name, "source_parameters": json.dumps(params, sort_keys=True),
            "target_parameters_fitted": False,
            "target_sigma_source": "target A/B observations only",
            "source_fit_nll": info["nll"], "evaluation_scope": scope,
            "target_retrieval_context": context, "target_probe_exposure_cohort": cohort,
            "target_probe_log_likelihood": float(rows.observation_log_likelihood.sum()),
            **metrics,
        }

    rows = [metric_row(probe, "overall", "All", "All")]
    if "probe_exposure_cohort" in probe:
        rows.extend(
            metric_row(group, "exposure_cohort", str(context), str(cohort))
            for (context, cohort), group in probe.groupby(
                ["retrieval_context", "probe_exposure_cohort"], sort=False, dropna=False
            )
        )
    return pd.DataFrame(rows)
