"""Required PRD figures for the geometry transfer experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import CoordinateRegistry, synthetic_display_layout


COLORS = {
    "G0_synthetic": "#6B7280",
    "G1_exact_svg": "#0072B2",
    "G2_shared_x": "#009E73",
}
CONDITION_LABELS = {
    "High_monotonic": "High (scale 2)",
    "Low_monotonic": "Low (scale .02)",
}


def _save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_geometry_audit(registry: CoordinateRegistry, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False, sharey=False)
    human_reference_x = registry.arrays[0][:, 0]
    synthetic_a = synthetic_display_layout("A")
    synthetic_b = synthetic_display_layout("B")
    entries = [
        (f"Human array {i + 1}", array,
         float(np.corrcoef(human_reference_x, array[:, 0])[0, 1]))
        for i, array in enumerate(registry.arrays)
    ] + [
        ("Old synthetic A", synthetic_a,
         float(np.corrcoef(synthetic_a[:, 0], synthetic_b[:, 0])[0, 1])),
        ("Old synthetic B", synthetic_b,
         float(np.corrcoef(synthetic_a[:, 0], synthetic_b[:, 0])[0, 1])),
    ]
    for ax, (title, coordinates, x_correlation) in zip(axes.flat, entries):
        ax.plot(coordinates[:, 0], coordinates[:, 1], color="#374151", alpha=.5)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=np.arange(8), cmap="viridis", s=50)
        for index, (x, y) in enumerate(coordinates, 1):
            ax.annotate(f"P{index}", (x, y), xytext=(3, 4), textcoords="offset points", fontsize=8)
        ratio = np.ptp(coordinates[:, 1]) / np.ptp(coordinates[:, 0])
        correlation_label = "r(x,array 1)" if title.startswith("Human") else "r(x,A/B)"
        ax.set_title(
            f"{title}\ny-span/x-span={ratio:.3f}; {correlation_label}={x_correlation:.3f}"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.axhline(0, color="#D1D5DB", lw=.6)
        ax.axvline(0, color="#D1D5DB", lw=.6)
        ax.set_xlabel("normalized x")
        ax.set_ylabel("normalized y (up positive)")
    fig.suptitle("Figure 1 — Human SVG and old synthetic geometry", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_1_geometry_audit")


def plot_learning_state_space(
    training: pd.DataFrame, classifications: pd.DataFrame, output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, condition in zip(axes[0], CONDITION_LABELS):
        one = training[training.condition.eq(condition)]
        for geometry, group in one.groupby("geometry_condition"):
            mean = group.groupby("training_sequence").mean_normalized_RT.mean()
            ax.plot(mean.index, mean.values, color=COLORS[geometry], label=geometry, lw=1.7)
        ax.set_title(f"{CONDITION_LABELS[condition]} raw RT")
        ax.set_xlabel("training sequence")
        ax.set_ylabel("normalized RT")
        ax.legend(fontsize=8)
    seed = classifications[classifications.row_type.eq("seed")]
    for ax, condition in zip(axes[1], CONDITION_LABELS):
        one = seed[seed.condition.eq(condition)]
        fractions = one.groupby(["geometry_condition", "model_label"]).size().unstack(fill_value=0)
        fractions = fractions.div(fractions.sum(axis=1), axis=0)
        fractions = fractions.reindex(columns=["Lump", "Cross", "Split"], fill_value=0)
        if len(fractions):
            fractions.plot(kind="bar", stacked=True, ax=ax,
                           color=["#999999", "#E69F00", "#CC79A7"])
        ax.set_title(f"{CONDITION_LABELS[condition]} state-space winners")
        ax.set_xlabel("geometry")
        ax.set_ylabel("winner fraction")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, title="")
    fig.suptitle("Figure 2 — A/B learning and canonical state-space bridge", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_2_learning_state_space")


def plot_transfer_deficit(metrics: pd.DataFrame, output: Path) -> None:
    seed = metrics[metrics.row_type.eq("seed_mean")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, condition in enumerate(CONDITION_LABELS):
        one = seed[seed.condition.eq(condition)]
        for col, (metric, title) in enumerate((
            ("NLL_deficit_over_worse_endpoint", "Prediction NLL deficit"),
            ("RT_deficit_over_worse_endpoint", "Normalized RT deficit"),
        )):
            ax = axes[row, col]
            summary = one.groupby(["geometry_condition", "probe_context"])[metric].agg(["mean", "sem"])
            geometries = list(COLORS)
            x = np.arange(len(geometries), dtype=float)
            for shift, probe in ((-.12, "Mixed"), (.12, "Novel")):
                means = [summary.loc[(g, probe), "mean"] if (g, probe) in summary.index else np.nan for g in geometries]
                sems = [summary.loc[(g, probe), "sem"] if (g, probe) in summary.index else np.nan for g in geometries]
                ax.errorbar(x + shift, means, yerr=sems, marker="o", capsize=3, label=probe)
            ax.axhline(0, color="black", lw=.8)
            ax.set_xticks(x, geometries, rotation=20, ha="right")
            ax.set_title(f"{CONDITION_LABELS[condition]} — {title}")
            ax.legend(fontsize=8)
    fig.suptitle("Figure 3 — Mixed/Novel transfer deficit over worse A/B endpoint", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_3_transfer_deficit")


def plot_differentiation_transfer(
    metrics: pd.DataFrame, hidden: pd.DataFrame, output: Path,
) -> None:
    seed = metrics[metrics.row_type.eq("seed_mean")]
    separation = hidden[hidden.row_type.eq("AB_seed_mean")][
        ["geometry_condition", "condition", "seed", "aggregate_AB_separation"]
    ]
    joined = seed.merge(separation, on=["geometry_condition", "condition", "seed"], how="left")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, condition in enumerate(CONDITION_LABELS):
        condition_rows = joined[joined.condition.eq(condition)]
        for col, (metric, title) in enumerate((
            ("NLL_deficit_over_worse_endpoint", "Prediction deficit"),
            ("RT_deficit_over_worse_endpoint", "RT deficit"),
        )):
            ax = axes[row, col]
            for geometry, one in condition_rows.groupby("geometry_condition"):
                markers = np.where(one.probe_context.eq("Mixed"), "o", "s")
                for marker in ("o", "s"):
                    use = one[markers == marker]
                    ax.scatter(use.aggregate_AB_separation, use[metric],
                               color=COLORS[geometry], marker=marker, alpha=.72,
                               label=f"{geometry} / {'Mixed' if marker == 'o' else 'Novel'}")
            ax.axhline(0, color="black", lw=.8)
            ax.set_xlabel("mean matched-position A/B hidden distance")
            ax.set_ylabel(title)
            ax.set_title(CONDITION_LABELS[condition])
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 4 — Differentiation versus transfer", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_4_differentiation_transfer")


def plot_e3(effects: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for row, condition in enumerate(CONDITION_LABELS):
        one = effects[effects.condition.eq(condition)]
        for col, metric in enumerate(("interaction_NLL", "interaction_RT")):
            ax = axes[row, col]
            summary = one.groupby(["geometry_condition", "probe_family"])[metric].agg(["mean", "sem"])
            x = np.arange(3, dtype=float)
            for shift, family in ((-.12, "Mixed"), (.12, "Novel")):
                means = [summary.loc[(g, family), "mean"] if (g, family) in summary.index else np.nan for g in COLORS]
                sems = [summary.loc[(g, family), "sem"] if (g, family) in summary.index else np.nan for g in COLORS]
                ax.errorbar(x + shift, means, yerr=sems, marker="o", capsize=3, label=family)
            ax.axhline(0, color="black", lw=.8)
            ax.set_xticks(x, list(COLORS), rotation=20, ha="right")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(CONDITION_LABELS[condition])
            ax.legend()
    fig.suptitle("Figure 5 — E3 cue × layout interaction", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_5_e3_interaction")


def plot_interpolation(interpolation: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, condition in enumerate(CONDITION_LABELS):
        for col, channel in enumerate(("cue", "layout")):
            ax = axes[row, col]
            one = interpolation[
                interpolation.condition.eq(condition) & interpolation.channel.eq(channel)
            ]
            for geometry, group in one.groupby("geometry_condition"):
                summary = group.groupby("lambda").mean_prediction_NLL.agg(["mean", "sem"])
                ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"],
                            color=COLORS[geometry], marker="o", capsize=2, label=geometry)
            ax.set_title(f"{CONDITION_LABELS[condition]} — {channel} interpolation")
            ax.set_xlabel("A → B lambda")
            ax.set_ylabel("prediction NLL")
            ax.legend(fontsize=8)
    fig.suptitle("Figure 6 — Frozen input-space interpolation", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_6_input_interpolation")


def make_all_figures(
    registry: CoordinateRegistry,
    training: pd.DataFrame,
    classifications: pd.DataFrame,
    transfer: pd.DataFrame,
    hidden: pd.DataFrame,
    e3: pd.DataFrame,
    interpolation: pd.DataFrame,
    output: Path,
) -> None:
    plot_geometry_audit(registry, output)
    plot_learning_state_space(training, classifications, output)
    plot_transfer_deficit(transfer, output)
    plot_differentiation_transfer(transfer, hidden, output)
    plot_e3(e3, output)
    plot_interpolation(interpolation, output)
