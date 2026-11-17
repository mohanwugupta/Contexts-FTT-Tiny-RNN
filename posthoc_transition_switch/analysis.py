"""Exact switch windows, transition costs, A/B distances, and paired tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel

from .config import HIGH_MONOTONIC, LOW_MONOTONIC, PosthocSwitchConfig


PROFILE_KEYS = ["condition", "seed", "context", "transition_position"]


def transition_profiles(
    history: pd.DataFrame, config: PosthocSwitchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return seed-level switch/early/settled profiles and full recovery."""
    eligible = history[history.block_index.ge(config.first_switch_block)].copy()

    def summarize(mask, name):
        selected = eligible[mask]
        return selected.groupby(PROFILE_KEYS, as_index=False).agg(
            **{
                f"RT_{name}": ("normalized_RT", "mean"),
                f"{name}_observations": ("normalized_RT", "size"),
                f"{name}_blocks": ("block_index", "nunique"),
            }
        )

    switch = summarize(
        eligible.within_block_sequence.eq(config.switch_sequence), "switch"
    )
    early = summarize(
        eligible.within_block_sequence.isin(config.early_sequences), "early"
    )
    settled = summarize(
        eligible.within_block_sequence.isin(config.settled_sequences), "settled"
    )
    profiles = switch.merge(early, on=PROFILE_KEYS, how="inner").merge(
        settled, on=PROFILE_KEYS, how="inner"
    )
    profiles["switch_cost"] = profiles.RT_switch - profiles.RT_settled
    profiles["early_cost"] = profiles.RT_early - profiles.RT_settled
    recovery = eligible.groupby(
        PROFILE_KEYS + ["within_block_sequence"], as_index=False
    ).agg(
        RT=("normalized_RT", "mean"),
        contributing_blocks=("block_index", "nunique"),
    )
    return profiles, recovery


def switch_cost_by_transition(profiles: pd.DataFrame) -> pd.DataFrame:
    columns = PROFILE_KEYS + [
        "RT_switch", "RT_early", "RT_settled", "switch_cost", "early_cost",
        "switch_observations", "early_observations", "settled_observations",
        "switch_blocks", "early_blocks", "settled_blocks",
    ]
    return profiles[columns].copy()


def profile_separation(
    profiles: pd.DataFrame, frozen_probes: pd.DataFrame,
) -> pd.DataFrame:
    frozen_lookup = {}
    if not frozen_probes.empty:
        for (condition, seed), group in frozen_probes.groupby(
            ["condition", "seed"], sort=True
        ):
            pivot = group.pivot(
                index="transition_position", columns="probe_context",
                values="normalized_RT",
            ).sort_index()
            frozen_lookup[(condition, int(seed))] = float(
                np.linalg.norm(pivot.A.to_numpy(float) - pivot.B.to_numpy(float))
            )
    rows = []
    for (condition, seed), group in profiles.groupby(["condition", "seed"], sort=True):
        indexed = group.set_index(["context", "transition_position"])
        values = {}
        for state in ("switch", "early", "settled"):
            a = np.asarray([
                indexed.loc[("A", position), f"RT_{state}"]
                for position in sorted(group.transition_position.unique())
            ], dtype=float)
            b = np.asarray([
                indexed.loc[("B", position), f"RT_{state}"]
                for position in sorted(group.transition_position.unique())
            ], dtype=float)
            values[state] = float(np.linalg.norm(a - b))
        rows.append({
            "condition": condition, "seed": int(seed),
            "D_switch": values["switch"], "D_early": values["early"],
            "D_settled": values["settled"],
            "D_frozen": frozen_lookup.get((condition, int(seed)), np.nan),
            "D_switch_minus_settled": values["switch"] - values["settled"],
            "D_early_minus_settled": values["early"] - values["settled"],
            "switch_separation_larger": values["switch"] > values["settled"],
            "frozen_probe_available": (condition, int(seed)) in frozen_lookup,
        })
    return pd.DataFrame(rows)


def _one_sample_row(label: str, values: pd.Series) -> dict:
    values = values.dropna().astype(float)
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    statistic, p_value = ttest_1samp(values, 0) if n > 1 else (np.nan, np.nan)
    return {
        "comparison": label, "n_paired_seeds": n, "mean_difference": mean,
        "SD_difference": sd, "SE_difference": sem,
        "CI95_lower": mean - 1.96 * sem if np.isfinite(sem) else np.nan,
        "CI95_upper": mean + 1.96 * sem if np.isfinite(sem) else np.nan,
        "paired_t": float(statistic), "paired_p_two_sided": float(p_value),
        "cohens_dz": mean / sd if np.isfinite(sd) and sd > 0 else np.nan,
        "posthoc": True,
    }


def _paired_row(label: str, first: pd.Series, second: pd.Series) -> dict:
    paired = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    differences = paired["first"] - paired["second"]
    row = _one_sample_row(label, differences)
    if len(paired) > 1:
        statistic, p_value = ttest_rel(paired["first"], paired["second"])
        row["paired_t"], row["paired_p_two_sided"] = float(statistic), float(p_value)
    return row


def primary_statistics(
    costs: pd.DataFrame, separation: pd.DataFrame,
) -> pd.DataFrame:
    seed_cost = costs.groupby(["condition", "seed"]).switch_cost.mean()
    high_cost = seed_cost.xs(HIGH_MONOTONIC, level="condition")
    low_cost = seed_cost.xs(LOW_MONOTONIC, level="condition")
    indexed = separation.set_index(["condition", "seed"])
    high_delta = indexed.xs(HIGH_MONOTONIC).D_switch_minus_settled
    low_delta = indexed.xs(LOW_MONOTONIC).D_switch_minus_settled
    rows = [
        _one_sample_row("High mean switch cost vs 0", high_cost),
        _one_sample_row("Low mean switch cost vs 0", low_cost),
        _paired_row("High minus Low mean switch cost", high_cost, low_cost),
        _one_sample_row("High D_switch minus D_settled vs 0", high_delta),
        _one_sample_row("Low D_switch minus D_settled vs 0", low_delta),
        _paired_row(
            "High minus Low D_switch-minus-settled", high_delta, low_delta
        ),
    ]
    return pd.DataFrame(rows)


def ci_summary(frame: pd.DataFrame, group_columns, value_column: str) -> pd.DataFrame:
    summary = frame.groupby(group_columns, as_index=False).agg(
        mean=(value_column, "mean"), SD=(value_column, "std"),
        seeds=("seed", "nunique"),
    )
    summary["SE"] = summary.SD / np.sqrt(summary.seeds)
    summary["CI95_lower"] = summary["mean"] - 1.96 * summary.SE
    summary["CI95_upper"] = summary["mean"] + 1.96 * summary.SE
    return summary
