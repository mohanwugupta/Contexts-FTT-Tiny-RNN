"""Publication-readable overlays for actual and checkpoint-matched sequence RT."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "time-matched-probe-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


COLORS = {"A": "#0072B2", "B": "#D55E00", "Mixed": "#009E73", "Novel": "#CC0077"}
CONDITIONS = ("High_monotonic", "Low_monotonic")


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.5, alpha=.42)


def _finish(fig, stem: Path):
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _training_summary(training: pd.DataFrame) -> pd.DataFrame:
    return training.groupby(
        ["condition", "training_sequence", "block_index", "context"], as_index=False
    ).mean_normalized_RT.agg(["mean", "sem"]).reset_index()


def _probe_summary(probes: pd.DataFrame) -> pd.DataFrame:
    return probes.groupby(
        ["condition", "completed_training_sequences", "event_x", "incoming_context", "probe_context"],
        as_index=False,
    ).sequence_mean_RT.agg(["mean", "sem"]).reset_index()


def _shade_blocks(axis, n_blocks: int = 17, block_length: int = 10):
    for block in range(1, n_blocks + 1):
        context = "A" if block % 2 else "B"
        axis.axvspan((block - 1) * block_length + .5, block * block_length + .5,
                     color=COLORS[context], alpha=.055, zorder=0)
        if block > 1:
            axis.axvline((block - 1) * block_length + .5, color="#6B7280",
                         lw=.45, alpha=.5, zorder=0)


def learning_curve_overlay(
    training: pd.DataFrame,
    probes: pd.DataFrame,
    existing_final: pd.DataFrame,
    *,
    probe_context: str,
    output_stem: Path,
):
    """Learning curve plus frozen same-snapshot endpoints and one probe type."""
    train = _training_summary(training)
    summary = _probe_summary(probes[probes.probe_context.isin(["A", "B", probe_context])])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for axis, condition in zip(axes, CONDITIONS):
        _shade_blocks(axis)
        one = train[train.condition.eq(condition)].sort_values("training_sequence")
        x = one.training_sequence.to_numpy(float)
        mean = one["mean"].to_numpy(float)
        se = one["sem"].fillna(0).to_numpy(float)
        axis.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                          color="#111827", alpha=.10, label="Training mean ± 95% SE")
        axis.plot(x, mean, color="#111827", lw=1.35, zorder=2)
        for context in ("A", "B"):
            local = one[one.context.eq(context)]
            axis.scatter(local.training_sequence, local["mean"], s=9,
                         color=COLORS[context], alpha=.85, zorder=3,
                         label=f"Observed training {context}")

        paired = summary[summary.condition.eq(condition)]
        piv = paired.pivot(index="completed_training_sequences", columns="probe_context", values="mean")
        for time, values in piv.iterrows():
            axis.plot([time + .5, time + .5], [values.A, values.B],
                      color="#374151", lw=.75, alpha=.6, zorder=4)
        for context, dodge, marker in (("A", -.13, "<"), ("B", .13, ">"),
                                       (probe_context, 0, "o")):
            local = paired[paired.probe_context.eq(context)]
            axis.errorbar(
                local.event_x + dodge, local["mean"], yerr=1.96 * local["sem"].fillna(0),
                fmt=marker, ms=4.8 if context != probe_context else 6.2,
                color=COLORS[context], ecolor=COLORS[context], elinewidth=.7,
                capsize=1.5, linestyle="none", zorder=5,
                label=(f"Frozen matched {context} (±95% SE)"),
            )
        actual = existing_final[
            existing_final.condition.eq(condition)
            & existing_final.probe_context.eq(probe_context)
        ]
        if not actual.empty:
            actual_mean = float(actual.sequence_mean_RT.mean())
            actual_se = float(actual.sequence_mean_RT.sem())
            axis.errorbar(170.5, actual_mean, yerr=1.96 * actual_se, fmt="*", ms=13,
                          color=COLORS[probe_context], ecolor="black", elinewidth=.8,
                          markeredgecolor="black", markeredgewidth=.7, capsize=2,
                          zorder=7, label=f"Existing final {probe_context} probe (±95% SE)")
        axis.set_ylabel("Mean normalized RT\n(7 scored transitions)")
        axis.set_title(condition.replace("_", " "))
        axis.set_xlim(.5, 172)
        _style(axis)
    axes[-1].set_xlabel(
        "External training sequence; diagnostic events occur at t+0.5 and add no update"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", frameon=False, ncol=4,
               fontsize=8)
    fig.suptitle(
        f"Raw context-yoked RT learning with time-matched {probe_context} probes",
        fontsize=15, fontweight="bold",
    )
    _finish(fig, output_stem)


def probe_event_closeups(
    training: pd.DataFrame,
    probes: pd.DataFrame,
    *,
    event_times: tuple[int, int, int],
    output_stem: Path,
):
    """First, schedule-middle, and final events chosen without looking at outcomes."""
    train = _training_summary(training)
    probe = _probe_summary(probes)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharey="row")
    for row, condition in enumerate(CONDITIONS):
        for col, time in enumerate(event_times):
            axis = axes[row, col]
            _shade_blocks(axis)
            local = train[
                train.condition.eq(condition)
                & train.training_sequence.between(max(1, time - 5), min(170, time + 5))
            ].sort_values("training_sequence")
            axis.plot(local.training_sequence, local["mean"], color="#111827", lw=1.2)
            for context in ("A", "B"):
                values = local[local.context.eq(context)]
                axis.scatter(values.training_sequence, values["mean"], color=COLORS[context], s=24,
                             label=f"Training {context}")
            matched = probe[
                probe.condition.eq(condition)
                & probe.completed_training_sequences.eq(time)
            ]
            endpoints = matched[matched.probe_context.isin(["A", "B"])].set_index("probe_context")
            axis.plot([time + .5, time + .5],
                      [endpoints.loc["A", "mean"], endpoints.loc["B", "mean"]],
                      color="#374151", lw=1)
            for context, dodge, marker in (("A", -.12, "<"), ("B", .12, ">"),
                                           ("Mixed", -.04, "o"), ("Novel", .04, "D")):
                point = matched[matched.probe_context.eq(context)].iloc[0]
                error = 1.96 * float(np.nan_to_num(point["sem"], nan=0.0))
                axis.errorbar(time + .5 + dodge, point["mean"], yerr=error,
                              fmt=marker, color=COLORS[context], ms=7, capsize=2,
                              linestyle="none", label=f"Frozen {context}")
            axis.set_xlim(max(.5, time - 5.5), min(171.5, time + 5.5))
            axis.set_title(f"{condition.replace('_', ' ')} — after sequence {time}")
            axis.set_xlabel("Training sequence / t+0.5 evaluation")
            if col == 0:
                axis.set_ylabel("Mean normalized RT")
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", frameon=False, ncol=6,
               fontsize=8)
    fig.suptitle("Schedule-defined same-snapshot probe-event close-ups",
                 fontsize=15, fontweight="bold")
    _finish(fig, output_stem)


def overlays_by_seed(training: pd.DataFrame, probes: pd.DataFrame, output_path: Path):
    """One page per seed, preserving individual trajectories and matched branches."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for seed in sorted(probes.seed.unique()):
            fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
            for row, condition in enumerate(CONDITIONS):
                observed = training[
                    training.seed.eq(seed) & training.condition.eq(condition)
                ].sort_values("training_sequence")
                for col, probe_context in enumerate(("Mixed", "Novel")):
                    axis = axes[row, col]
                    _shade_blocks(axis)
                    axis.plot(observed.training_sequence, observed.mean_normalized_RT,
                              color="#111827", lw=1)
                    for context in ("A", "B"):
                        local = observed[observed.context.eq(context)]
                        axis.scatter(local.training_sequence, local.mean_normalized_RT,
                                     color=COLORS[context], s=6, alpha=.7)
                    matched = probes[
                        probes.seed.eq(seed) & probes.condition.eq(condition)
                        & probes.probe_context.isin(["A", "B", probe_context])
                    ]
                    piv = matched.pivot(index="completed_training_sequences",
                                        columns="probe_context", values="sequence_mean_RT")
                    for time, values in piv.iterrows():
                        axis.plot([time + .5, time + .5], [values.A, values.B],
                                  color="#6B7280", lw=.55, alpha=.65)
                    for context, dodge, marker in (("A", -.1, "<"), ("B", .1, ">"),
                                                   (probe_context, 0, "o")):
                        local = matched[matched.probe_context.eq(context)]
                        axis.scatter(local.event_x + dodge, local.sequence_mean_RT,
                                     color=COLORS[context], marker=marker,
                                     s=18 if context != probe_context else 27, zorder=4)
                    axis.set_title(f"{condition.replace('_', ' ')} — {probe_context}")
                    axis.set_ylabel("Mean normalized RT")
                    axis.set_xlabel("Training sequence / t+0.5 evaluation")
                    _style(axis)
            fig.suptitle(f"Time-matched sequence probes — seed {seed}",
                         fontsize=15, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, .96))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
