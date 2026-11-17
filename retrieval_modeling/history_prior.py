"""History as a dynamic prior over otherwise similarity-based retrieval."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .context_features import canonical_retrieval_context
from .retrieval_weights import similarity_responsibility


def history_responsibility(
    contexts: Iterable[object],
    *,
    beta: float,
    rho: float,
    eta: float,
    initial_prior: float = 0.0,
) -> pd.DataFrame:
    names = [canonical_retrieval_context(value) for value in contexts]
    priors, prior = [], float(initial_prior)
    for context in names:
        priors.append(prior)
        cue = 1.0 if context == "A" else -1.0 if context == "B" else 0.0
        prior = float(rho) * prior + float(eta) * cue
    result = similarity_responsibility(names, beta, np.asarray(priors))
    result["history_state"] = priors
    return result


def grouped_history_responsibility(
    data: pd.DataFrame,
    *,
    beta: float,
    rho: float,
    eta: float,
) -> pd.DataFrame:
    frames = []
    for _, participant in data.groupby("participant_id", sort=False):
        ordered = participant.sort_values("global_sequence", kind="stable")
        weights = history_responsibility(
            ordered.retrieval_context, beta=beta, rho=rho, eta=eta
        )
        weights.index = ordered.index
        frames.append(weights)
    return pd.concat(frames).sort_index()
