"""Validated fixed 17-block online A10/B10 schedule."""

import numpy as np
import pandas as pd


def make_training_schedule(n_blocks: int = 17, block_length: int = 10) -> pd.DataFrame:
    if block_length != 10:
        raise ValueError("FTT block_length is fixed at ten sequence repetitions.")
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive.")
    global_sequence = np.arange(1, n_blocks * block_length + 1)
    block_index = (global_sequence - 1) // block_length + 1
    return pd.DataFrame({
        "global_sequence": global_sequence,
        "block_index": block_index,
        "sequence_in_block": (global_sequence - 1) % block_length + 1,
        "context": np.where(block_index % 2 == 1, "A", "B"),
    })

