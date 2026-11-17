"""Frozen interface to the already-tested PRD2 A/B skill-state estimator."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from probe_modeling.epoch_skill_states import fit_epoch_counterfactual_states


def fit_frozen_ab_states(
    data: pd.DataFrame,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states, fits = fit_epoch_counterfactual_states(data, cross_talk=0.0, progress=progress)
    states = add_ab_only_participant_intercepts(states)
    fits = fits.copy()
    fits["retrieval_experts_frozen"] = True
    return states, fits


def add_ab_only_participant_intercepts(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate nuisance intercepts solely from established-context residuals."""
    result = data.copy()
    established = result.context.isin(["A", "B"])
    expert = np.where(result.context.eq("A"), result.pred_A_log_rt, result.pred_B_log_rt)
    residual = result.log_rt - expert
    intercepts = residual[established].groupby(result.loc[established, "participant_id"]).mean()
    result["participant_intercept"] = result.participant_id.map(intercepts).fillna(0.0)
    result["participant_intercept_source"] = "A/B observations only"
    return result
