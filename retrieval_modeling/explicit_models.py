"""Individual-sequence likelihood models for the PRD3 retrieval ladder."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .categorical_readout import categorical_component_posterior, categorical_log_likelihood
from .continuous_readout import continuous_log_likelihood, continuous_readout
from .evidence_strength import evidence_scaled_sigma
from .history_prior import grouped_history_responsibility
from .retrieval_weights import similarity_responsibility


PRIMARY_RETRIEVAL_MODELS = (
    "equal_mixture",
    "free_static_mixture",
    "continuous_similarity",
    "similarity_history",
    "similarity_evidence",
    "categorical_similarity",
    "default_state",
    "third_state_falsification",
)


def _dataset_sigma(data: pd.DataFrame, params: Mapping[str, float]) -> np.ndarray:
    return np.asarray(
        [np.exp(float(params.get(f"log_sigma_{dataset}", -2.3))) for dataset in data.dataset],
        dtype=float,
    )


def _base_similarity(data: pd.DataFrame, beta: float) -> pd.DataFrame:
    weights = similarity_responsibility(data.retrieval_context, beta)
    weights.index = data.index
    return weights


def _static_q(data: pd.DataFrame, params: Mapping[str, float]) -> np.ndarray:
    values = []
    for context in data.retrieval_context:
        if context == "A":
            values.append(1.0)
        elif context == "B":
            values.append(0.0)
        else:
            values.append(float(params.get(f"q_{context}", 0.5)))
    return np.asarray(values)


def _third_state_offsets(data: pd.DataFrame, learning_rate: float, retention: float) -> np.ndarray:
    offsets = pd.Series(0.0, index=data.index)
    for _, participant in data.groupby(["dataset", "participant_id"], sort=False):
        states = {"Both": 0.0, "Novel": 0.0, "Default": 0.0}
        for index, row in participant.sort_values("global_sequence", kind="stable").iterrows():
            context = row.retrieval_context
            if context in states:
                offsets.loc[index] = states[context]
                states[context] = float(retention) * states[context] - float(learning_rate)
    return offsets.to_numpy(float)


def predict_retrieval_model(
    data: pd.DataFrame,
    model_name: str,
    params: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if model_name not in PRIMARY_RETRIEVAL_MODELS:
        raise ValueError(f"Unknown retrieval model {model_name!r}.")
    required = {
        "participant_id", "dataset", "global_sequence", "retrieval_context",
        "pred_A_log_rt", "pred_B_log_rt", "log_rt",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Retrieval prediction requires {sorted(missing)}")
    params = dict(params or {})
    beta = float(params.get("beta", 1.0))
    if model_name == "similarity_history":
        weights = grouped_history_responsibility(
            data,
            beta=beta,
            rho=float(params.get("rho", 0.8)),
            eta=float(params.get("eta", 0.0)),
        )
    else:
        weights = _base_similarity(data, beta)

    if model_name == "equal_mixture":
        q = np.where(data.retrieval_context.eq("A"), 1.0, np.where(data.retrieval_context.eq("B"), 0.0, 0.5))
    elif model_name == "free_static_mixture":
        q = _static_q(data, params)
    else:
        q = weights.q_A.to_numpy(float)
    a = data.pred_A_log_rt.to_numpy(float)
    b = data.pred_B_log_rt.to_numpy(float)
    intercept = data.get("participant_intercept", pd.Series(0.0, index=data.index)).to_numpy(float)
    two_skill_mean = continuous_readout(a, b, q)
    default_weight = np.zeros(len(data), dtype=float)
    if model_name == "default_state":
        evidence = weights.absolute_evidence_E.to_numpy(float)
        logit_default = float(params.get("default_bias", -3.0)) - float(
            params.get("default_evidence_gain", 1.0)
        ) * evidence
        default_weight = 1.0 / (1.0 + np.exp(-np.clip(logit_default, -40, 40)))
        default_state = 0.5 * (a + b) + float(params.get("default_offset", 0.0))
        predicted = (1.0 - default_weight) * two_skill_mean + default_weight * default_state + intercept
    elif model_name == "third_state_falsification":
        predicted = two_skill_mean + _third_state_offsets(
            data,
            float(params.get("third_learning_rate", 0.0)),
            float(params.get("third_retention", 0.9)),
        ) + intercept
    else:
        predicted = two_skill_mean + intercept
    base_sigma = _dataset_sigma(data, params)
    sigma = (
        evidence_scaled_sigma(
            base_sigma, weights.absolute_evidence_E.to_numpy(float), float(params.get("kappa", 0.0))
        )
        if model_name == "similarity_evidence"
        else base_sigma
    )
    observed = data.log_rt.to_numpy(float)
    if model_name == "categorical_similarity":
        log_likelihood = categorical_log_likelihood(observed, a + intercept, b + intercept, q, sigma)
        posterior_a = categorical_component_posterior(observed, a + intercept, b + intercept, q, sigma)
    else:
        log_likelihood = continuous_log_likelihood(observed, predicted, sigma)
        posterior_a = np.full(len(data), np.nan)
    output = data.copy()
    for column in [
        "d_A", "d_B", "relative_evidence_R", "evidence_A", "evidence_B",
        "absolute_evidence_E", "prior_log_odds", "retrieval_logit",
    ]:
        output[column] = weights[column].to_numpy(float)
    output["history_state"] = weights.get("history_state", pd.Series(0.0, index=weights.index)).to_numpy(float)
    output["q_A"] = q
    output["q_B"] = 1.0 - q
    output["default_weight"] = default_weight
    output["predicted_log_rt"] = predicted
    output["predicted_rt"] = np.exp(predicted)
    output["predicted_sigma"] = sigma
    output["observation_log_likelihood"] = log_likelihood
    output["categorical_posterior_A"] = posterior_a
    output["model"] = model_name
    output["readout_family"] = "categorical" if model_name == "categorical_similarity" else "continuous"
    return output
