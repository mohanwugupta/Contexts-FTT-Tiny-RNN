"""Context-switch costs in the preregistered alternating-block schedule."""

from __future__ import annotations

import numpy as np
import pandas as pd


def transition_costs(training: pd.DataFrame, checkpoint_map: dict[str, int]) -> pd.DataFrame:
    """Summarize validated absolute block-boundary jumps by checkpoint.

    This follows the original repository's transition-cost definition:
    ``abs(first value in current block - last value in previous block)``.
    """
    rows = []
    for checkpoint, through_sequence in checkpoint_map.items():
        frame = training[training.global_sequence <= through_sequence]
        switch_rows = []
        blocks = [group.sort_values("sequence_in_block")
                  for _, group in frame.groupby("block_index", sort=True)]
        for previous, current in zip(blocks[:-1], blocks[1:]):
            if previous.empty or current.empty:
                continue
            values = {}
            for source, label in (
                ("sequence_ce", "sequence"),
                ("execution_cost", "execution"),
                ("normalized_rt", "rt"),
            ):
                signed = float(current.iloc[0][source] - previous.iloc[-1][source])
                values[f"{label}_transition_signed"] = signed
                values[f"{label}_transition_cost"] = abs(signed)
            switch_rows.append(values)
        if switch_rows:
            values = pd.DataFrame(switch_rows)
            rows.append({
                "checkpoint": checkpoint,
                "checkpoint_sequence": through_sequence,
                "n_context_switches": len(values),
                **{column: float(values[column].mean()) for column in values.columns},
            })
        else:
            rows.append({
                "checkpoint": checkpoint,
                "checkpoint_sequence": through_sequence,
                "n_context_switches": 0,
                "sequence_transition_cost": np.nan,
                "sequence_transition_signed": np.nan,
                "execution_transition_cost": np.nan,
                "execution_transition_signed": np.nan,
                "rt_transition_cost": np.nan,
                "rt_transition_signed": np.nan,
            })
    return pd.DataFrame(rows)
