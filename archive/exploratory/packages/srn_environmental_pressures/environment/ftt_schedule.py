"""The fixed nominal FTT ecology: ten A trials, ten B trials, repeated."""

import numpy as np


def make_ftt_schedule(n_blocks: int = 16, block_length: int = 10) -> np.ndarray:
    if block_length != 10:
        raise ValueError("FTT block_length is fixed at ten.")
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive.")
    return np.repeat(np.arange(n_blocks) % 2, block_length).astype(int)


def block_position(n_blocks: int = 16, block_length: int = 10) -> np.ndarray:
    return np.tile(np.arange(1, block_length + 1), n_blocks)

