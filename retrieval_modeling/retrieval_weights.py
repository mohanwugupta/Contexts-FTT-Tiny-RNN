"""Kernel responsibilities over fixed A/B context prototypes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .context_features import canonical_retrieval_context
from .context_geometry import geometry_arrays


def similarity_responsibility(
    contexts: Iterable[object],
    beta: float,
    prior_log_odds: Iterable[float] | float = 0.0,
) -> pd.DataFrame:
    names = [canonical_retrieval_context(value) for value in contexts]
    geometry = geometry_arrays(names, beta)
    prior = np.broadcast_to(np.asarray(prior_log_odds, dtype=float), len(names))
    logit = float(beta) * geometry["relative_evidence_R"] + prior
    q = 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))
    return pd.DataFrame(
        {
            "retrieval_context": names,
            **{key: value for key, value in geometry.items() if key != "q_A"},
            "prior_log_odds": prior,
            "retrieval_logit": logit,
            "q_A": q,
            "q_B": 1.0 - q,
        }
    )
