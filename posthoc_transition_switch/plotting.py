"""Posthoc transition-profile and recovery figures."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "posthoc-switch-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import ci_summary  # noqa: E402
from .config import HIGH_MONOTONIC, LOW_MONOTONIC  # noqa: E402


COLORS = {
    "A": "#0072B2", "B": "#D55E00",
    HIGH_MONOTONIC: "#0072B2", LOW_MONOTONIC: "#D55E00",
}


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.5, alpha=.45)


def _finish(fig, path):
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_transition_profiles(profiles: pd.DataFrame, path):
    high = profiles[profiles.condition.eq(HIGH_MONOTONIC)]
    fig, axis = plt.subplots(figsize=(11, 6))
    for context in ("A", "B"):
        for state, line_style in (("switch", "-"), ("settled", "--")):
            summary = ci_summary(
                high[high.context.eq(context)],
                ["context", "transition_position"], f"RT_{state}",
            )
            axis.plot(summary.transition_position, summary["mean"], marker="o",
                      color=COLORS[context], ls=line_style,
                      label=f"{context} {state}")
            axis.fill_between(
                summary.transition_position, summary.CI95_lower,
                summary.CI95_upper, color=COLORS[context], alpha=.12,
            )
    axis.set_xlabel("Ordinal transition position")
    axis.set_ylabel("Normalized RT")
    axis.legend(frameon=False, ncol=2)
    axis.set_title("High-scale seed means ± 95% CI")
    _style(axis)
    fig.suptitle("Figure 1. Transition profiles by retrieval state",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_switch_costs(costs: pd.DataFrame, path):
    seed_position = costs.groupby(
        ["condition", "seed", "transition_position"], as_index=False
    ).switch_cost.mean()
    summary = ci_summary(
        seed_position, ["condition", "transition_position"], "switch_cost"
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for condition in (HIGH_MONOTONIC, LOW_MONOTONIC):
        one = summary[summary.condition.eq(condition)]
        axis.plot(one.transition_position, one["mean"], marker="o",
                  color=COLORS[condition], label=condition.replace("_monotonic", " scale"))
        axis.fill_between(one.transition_position, one.CI95_lower,
                          one.CI95_upper, color=COLORS[condition], alpha=.15)
    axis.axhline(0, color="#111827", lw=1, ls="--")
    axis.set_xlabel("Ordinal transition position")
    axis.set_ylabel("RT switch − RT settled")
    axis.legend(frameon=False)
    _style(axis)
    fig.suptitle("Figure 2. Switch cost by transition position",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_block_recovery(recovery: pd.DataFrame, path):
    seed_mean = recovery.groupby(
        ["condition", "seed", "transition_position", "within_block_sequence"],
        as_index=False,
    ).RT.mean()
    summary = ci_summary(
        seed_mean,
        ["condition", "transition_position", "within_block_sequence"], "RT",
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for position, axis in zip(range(1, 8), axes.flat):
        for condition in (HIGH_MONOTONIC, LOW_MONOTONIC):
            one = summary[
                summary.condition.eq(condition)
                & summary.transition_position.eq(position)
            ]
            axis.plot(one.within_block_sequence, one["mean"], marker="o", ms=3,
                      color=COLORS[condition],
                      label=condition.replace("_monotonic", " scale"))
            axis.fill_between(
                one.within_block_sequence, one.CI95_lower, one.CI95_upper,
                color=COLORS[condition], alpha=.12,
            )
        axis.axvspan(.75, 2.25, color="#6B7280", alpha=.07)
        axis.axvspan(7.75, 10.25, color="#009E73", alpha=.06)
        axis.set_title(f"Transition P{position}")
        axis.set_xticks(range(1, 11))
        _style(axis)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(.96, .10),
               frameon=False)
    fig.supxlabel("Sequence within A/B block")
    fig.supylabel("Normalized RT")
    fig.suptitle("Figure 3. Transition-specific recovery after context switches",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_frozen_comparison(
    profiles: pd.DataFrame, frozen: pd.DataFrame, path,
):
    if frozen.empty:
        return
    settled = profiles.groupby(
        ["condition", "seed", "context", "transition_position"], as_index=False
    ).RT_settled.mean().rename(columns={"RT_settled": "RT"})
    settled["state"] = "settled training"
    frozen_frame = frozen.rename(columns={
        "probe_context": "context", "normalized_RT": "RT",
    })[["condition", "seed", "context", "transition_position", "RT"]]
    frozen_frame["state"] = "frozen probe"
    data = pd.concat([settled, frozen_frame], ignore_index=True)
    data = data[data.condition.eq(HIGH_MONOTONIC)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, context in zip(axes, ("A", "B")):
        for state, line_style in (("settled training", "-"), ("frozen probe", "--")):
            one = data[data.context.eq(context) & data.state.eq(state)]
            summary = ci_summary(one, ["transition_position"], "RT")
            axis.plot(summary.transition_position, summary["mean"], marker="o",
                      color=COLORS[context], ls=line_style, label=state)
            axis.fill_between(summary.transition_position, summary.CI95_lower,
                              summary.CI95_upper, color=COLORS[context], alpha=.12)
        axis.set_title(f"Context {context}")
        axis.set_xlabel("Ordinal transition position")
        axis.legend(frameon=False)
        _style(axis)
    axes[0].set_ylabel("Normalized RT")
    fig.suptitle("Optional Figure 4. Settled training versus frozen A/B probes",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def make_figures(output, profiles, costs, recovery, frozen):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figure_transition_profiles(
        profiles, output / "figure_posthoc_transition_profiles"
    )
    figure_switch_costs(costs, output / "figure_posthoc_switch_costs")
    figure_block_recovery(recovery, output / "figure_posthoc_block_recovery")
    figure_frozen_comparison(
        profiles, frozen, output / "figure_posthoc_frozen_comparison"
    )
