"""A/B-only familiar contrast projection."""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def fit_familiar_contrast(cue_a, cue_b):
    cue_a = np.asarray(cue_a, dtype=np.float32); cue_b = np.asarray(cue_b, dtype=np.float32)
    if cue_a.shape != cue_b.shape or cue_a.ndim != 1:
        raise ValueError("Familiar cues must be equal-length vectors.")
    cbar = 0.5 * (cue_a + cue_b); direction = 0.5 * (cue_a - cue_b)
    if float(direction @ direction) == 0:
        raise ValueError("Familiar cues must differ.")
    return cbar, direction


def project_context(cue, cbar, direction, epsilon=1e-12):
    cue = np.asarray(cue, dtype=np.float32); cbar = np.asarray(cbar, dtype=np.float32); direction = np.asarray(direction, dtype=np.float32)
    return float((cue - cbar) @ direction / (direction @ direction + epsilon))


def cue_projection_audit(cues, cue_a, cue_b, epsilon=1e-12):
    cbar, direction = fit_familiar_contrast(cue_a, cue_b); denominator = float(direction @ direction)
    rows = []
    for name, cue in cues.items():
        cue = np.asarray(cue, np.float32); centered = cue - cbar; alpha = project_context(cue, cbar, direction, epsilon)
        projection = alpha * direction; orthogonal = centered - projection
        rows.append({
            "cue_name": name, "raw_cue_hash": hashlib.sha256(cue.tobytes()).hexdigest(), "alpha": alpha,
            "distance_from_cbar": float(np.linalg.norm(centered)), "projection_on_dc": float(centered @ direction),
            "orthogonal_norm": float(np.linalg.norm(orthogonal)),
        })
    result = pd.DataFrame(rows)
    lookup = result.set_index("cue_name").alpha
    if not np.isclose(lookup["A"], 1.0, atol=1e-7) or not np.isclose(lookup["B"], -1.0, atol=1e-7):
        raise AssertionError("FAIL: FAMILIAR_PROJECTION_IDENTITY")
    return result
