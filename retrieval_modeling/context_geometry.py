"""Squared-distance geometry and relative/absolute retrieval evidence."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .context_features import CONTEXT_VECTORS, context_matrix


PROTOTYPE_A = np.asarray(CONTEXT_VECTORS["A"], dtype=float)
PROTOTYPE_B = np.asarray(CONTEXT_VECTORS["B"], dtype=float)


def geometry_arrays(contexts: Iterable[object], beta: float) -> dict[str, np.ndarray]:
    features = context_matrix(contexts)
    d_a = np.sum((features - PROTOTYPE_A) ** 2, axis=1)
    d_b = np.sum((features - PROTOTYPE_B) ** 2, axis=1)
    evidence_a = np.exp(-float(beta) * d_a)
    evidence_b = np.exp(-float(beta) * d_b)
    total = evidence_a + evidence_b
    q_a = np.divide(evidence_a, total, out=np.full_like(total, 0.5), where=total > 0)
    return {
        "d_A": d_a,
        "d_B": d_b,
        "relative_evidence_R": d_b - d_a,
        "evidence_A": evidence_a,
        "evidence_B": evidence_b,
        "absolute_evidence_E": total,
        "q_A": q_a,
    }


def context_geometry(beta: float = 1.0, contexts: Iterable[object] | None = None) -> pd.DataFrame:
    names = list(CONTEXT_VECTORS) if contexts is None else [str(value) for value in contexts]
    arrays = geometry_arrays(names, beta)
    return pd.DataFrame({"context": names, "beta": float(beta), **arrays})
