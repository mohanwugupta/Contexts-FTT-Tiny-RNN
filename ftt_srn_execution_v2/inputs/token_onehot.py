"""Strict four-dimensional token one-hot encoding."""

import numpy as np
import torch


def encode_sequence(sequence, *, as_tensor: bool = False):
    indices = np.asarray(sequence, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= 4):
        raise ValueError("Tokens must be a vector of indices in 0..3.")
    encoded = np.eye(4, dtype=np.float32)[indices]
    return torch.as_tensor(encoded) if as_tensor else encoded

