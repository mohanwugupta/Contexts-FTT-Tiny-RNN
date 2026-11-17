"""Analytic recovery check for continuous versus sample-based behavior."""

from __future__ import annotations

import pandas as pd

from ..analysis.bayes_risk import mean_squared_risk, sampling_squared_risk


def normative_objective_recovery(skill_a, skill_b, q_a) -> pd.DataFrame:
    mean_risk = mean_squared_risk(skill_a, skill_b, q_a)
    sample_risk = sampling_squared_risk(skill_a, skill_b, q_a)
    return pd.DataFrame(
        [
            {"generator": "posterior_mean", "expected_mse": mean_risk, "recovered": "continuous"},
            {"generator": "posterior_sampling", "expected_mse": sample_risk, "recovered": "sampling"},
        ]
    )
