"""Absolute contextual evidence as retrieval confidence/variance."""

from __future__ import annotations

import numpy as np


def evidence_scaled_sigma(base_sigma, absolute_evidence, kappa: float) -> np.ndarray:
    base = np.asarray(base_sigma, dtype=float)
    evidence = np.asarray(absolute_evidence, dtype=float)
    return np.maximum(base * np.exp(-float(kappa) * evidence), 1e-8)
