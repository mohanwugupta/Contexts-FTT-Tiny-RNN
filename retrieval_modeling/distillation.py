"""Distill learned gate outputs into the explicit retrieval equation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def distill_retrieval_rule(data: pd.DataFrame) -> pd.DataFrame:
    required = {"q_A", "relative_evidence_R", "absolute_evidence_E", "history_state", "split"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Distillation requires {sorted(missing)}")
    values = data.copy()
    values["gate_logit"] = np.log(
        np.clip(values.q_A, 1e-6, 1 - 1e-6) / np.clip(1 - values.q_A, 1e-6, 1)
    )
    features = ["relative_evidence_R", "absolute_evidence_E", "history_state"]
    train = values[values.split.eq("train")]
    test = values[values.split.eq("test")]
    if train.empty:
        train = values
    if test.empty:
        test = values
    regression = LinearRegression().fit(train[features], train.gate_logit)
    return pd.DataFrame(
        [
            {
                "equation": "logit(q_A) = beta_R*R + beta_E*E + lambda_history*h + intercept",
                "beta_R": float(regression.coef_[0]),
                "beta_E": float(regression.coef_[1]),
                "lambda_history": float(regression.coef_[2]),
                "intercept": float(regression.intercept_),
                "coefficients": json.dumps(dict(zip(features, regression.coef_.tolist())), sort_keys=True),
                "train_r2": float(regression.score(train[features], train.gate_logit)),
                "heldout_r2": float(regression.score(test[features], test.gate_logit)),
            }
        ]
    )
