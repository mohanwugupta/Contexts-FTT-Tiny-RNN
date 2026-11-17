"""Nonlearned confidence-threshold timing controls on frozen cycle traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CommitmentConfig
from .gates import compute_switch_costs


def threshold_control_trajectories(
    cycles: pd.DataFrame,
    config: CommitmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read commitment time from learned preparation states without retraining.

    This is deliberately a conditional readout of the same saved cycle dynamics:
    no optimizer is constructed and no hidden-state trajectory is recomputed.
    """
    keys = [
        "c_time", "seed", "training_sequence", "global_sequence",
        "block_index", "sequence_in_block", "context", "sequence_position",
    ]
    transition_rows = []
    for key, group in cycles.groupby(keys, sort=True, dropna=False):
        group = group.sort_values("prep_cycle")
        for threshold in config.confidence_thresholds:
            reached = group.loc[group.p_correct.ge(threshold), "prep_cycle"]
            hard_cycle = (
                int(reached.iloc[0]) if len(reached)
                else config.max_preparation_cycles
            )
            transition_rows.append({
                **dict(zip(keys, key)),
                "control": "fixed_confidence_threshold",
                "threshold": float(threshold),
                "hard_rt_cycles": hard_cycle,
                "normalized_rt": (hard_cycle - 1) / (
                    config.max_preparation_cycles - 1
                ),
                "optimizer_steps": 0,
                "retraining": False,
            })
    transitions = pd.DataFrame(transition_rows)
    sequence_keys = [
        "c_time", "seed", "threshold", "training_sequence", "global_sequence",
        "block_index", "sequence_in_block", "context",
    ]
    sequences = (
        transitions.groupby(sequence_keys, as_index=False, sort=True)
        .agg(normalized_rt=("normalized_rt", "mean"),
             hard_rt_cycles=("hard_rt_cycles", "mean"))
    )
    sequences["control"] = "fixed_confidence_threshold"
    sequences["optimizer_steps"] = 0
    sequences["retraining"] = False

    summaries = []
    for (c_time, threshold), group in sequences.groupby(
        ["c_time", "threshold"], sort=True
    ):
        switches = compute_switch_costs(group, config)
        late = switches[switches.switch_block.ge(5)] if len(switches) else switches
        summaries.append({
            "c_time": float(c_time),
            "threshold": float(threshold),
            "normalized_rt_min": float(group.normalized_rt.min()),
            "normalized_rt_max": float(group.normalized_rt.max()),
            "normalized_rt_dynamic_range": float(
                group.normalized_rt.max() - group.normalized_rt.min()
            ),
            "cycle_1_fraction": float(np.mean(group.hard_rt_cycles.eq(1))),
            "max_cycle_fraction": float(np.mean(
                group.hard_rt_cycles.eq(config.max_preparation_cycles)
            )),
            "late_mean_switch_cost": (
                float(late.switch_cost.mean()) if len(late) else np.nan
            ),
            "late_fraction_positive_switch_cost": (
                float(late.switch_cost_positive.mean()) if len(late) else np.nan
            ),
            "optimizer_steps": 0,
            "retraining": False,
            "human_data_used": 0,
        })
    return transitions, sequences, pd.DataFrame(summaries)
