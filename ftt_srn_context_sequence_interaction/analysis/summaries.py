"""Small, transparent summaries for token-structured causal effects."""

from __future__ import annotations

import numpy as np


def linear_correspondence(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {"correlation": np.nan, "slope": np.nan, "r_squared": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    denominator = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((y - predicted) ** 2)) / denominator
    return {
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(slope), "r_squared": float(r_squared),
    }

