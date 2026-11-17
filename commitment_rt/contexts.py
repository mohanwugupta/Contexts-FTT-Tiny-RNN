"""PRD-frozen context vectors for commitment training and later probes."""

from __future__ import annotations

import numpy as np
import torch


CONTEXTS = {
    "A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "Mixed_norm": np.array([0.5, 0.5, 0.0], dtype=np.float32),
    "Mixed_additive": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    "Novel": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def context_vector(name: str, *, as_tensor: bool = False):
    if name not in CONTEXTS:
        raise ValueError(f"Unknown commitment context {name!r}.")
    result = CONTEXTS[name].copy()
    return torch.as_tensor(result) if as_tensor else result
