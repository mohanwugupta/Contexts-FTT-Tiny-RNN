"""Exact three-dimensional confirmatory context basis."""

from __future__ import annotations

import numpy as np
import torch


CONTEXTS = {
    "A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "Mixed_norm": np.array([0.5, 0.5, 0.0], dtype=np.float32),
    "Novel": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "Mixed_additive": np.array([1.0, 1.0, 0.0], dtype=np.float32),
}


def context_vector(name: str, *, as_tensor: bool = False):
    if name not in CONTEXTS:
        raise ValueError(f"Unknown confirmatory context {name!r}.")
    value = CONTEXTS[name].copy()
    return torch.as_tensor(value) if as_tensor else value
