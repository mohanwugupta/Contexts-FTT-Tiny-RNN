"""Task-derived context vectors fixed before observing probe behavior."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FEATURE_NAMES = ("A_cue", "B_cue", "novel_cue")
CONTEXT_VECTORS = {
    "A": (1.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0),
    "Both": (1.0, 1.0, 0.0),
    "Novel": (0.0, 0.0, 1.0),
    # The experiment's No Context condition contains none of the three task
    # cues. Keeping it distinct from Novel is a design-derived prediction.
    "Default": (0.0, 0.0, 0.0),
}


def canonical_retrieval_context(value: object) -> str:
    raw = str(value).strip()
    normalized = " ".join(raw.lower().replace("_", " ").replace("-", " ").split())
    mapping = {
        "a": "A", "maze": "A", "maze training": "A", "maze test": "A",
        "b": "B", "melody": "B", "melody training": "B", "melody test": "B",
        "test melodya": "B", "test melody a": "B",
        "both": "Both", "mixed": "Both", "mixed training": "Both", "mixed test": "Both",
        "probe": "Both", "combined": "Both",
        "novel": "Novel", "novel training": "Novel", "novel test": "Novel",
        "space": "Novel", "space training": "Novel", "space test": "Novel",
        "test spacea": "Novel", "test space a": "Novel",
        "third": "Novel", "third training": "Novel", "third test": "Novel",
        "third context": "Novel", "third context training": "Novel", "third context test": "Novel",
        "context c": "Novel", "c": "Novel",
        "no context": "Default", "no context training": "Default", "no context test": "Default",
        "nocontext": "Default", "default": "Default",
    }
    if normalized not in mapping:
        raise ValueError(f"Unknown retrieval context {raw!r}.")
    return mapping[normalized]


def context_vector(context: object) -> np.ndarray:
    name = canonical_retrieval_context(context)
    return np.asarray(CONTEXT_VECTORS[name], dtype=float)


def context_matrix(contexts: Iterable[object]) -> np.ndarray:
    return np.vstack([context_vector(context) for context in contexts])


def context_feature_table() -> pd.DataFrame:
    rows = []
    for context, vector in CONTEXT_VECTORS.items():
        rows.append({"context": context, **dict(zip(FEATURE_NAMES, vector)), "feature_source": "task_design"})
    return pd.DataFrame(rows)
