"""Publication-readable diagnostic figures; source tables are written by pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mechanism-probe-mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "A": "#0072B2", "B": "#D55E00", "Fast": "#0072B2", "Slow": "#D55E00",
    "Mixed": "#009E73", "Novel": "#CC79A7",
}


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.5, alpha=.45)


def _finish(fig, output: Path, stem: str):
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_raw_centered(source: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for row, system in enumerate(("Human", "SRN")):
        one_system = source[source.system.eq(system)]
        for col, (panel, ylabel) in enumerate((("M", "Raw RT"), ("Z", "Centered RT"))):
            axis = axes[row, col]
            if system == "SRN":
                for _, unit in one_system.groupby("unit_id"):
                    for condition, profile in unit.groupby("condition"):
                        axis.plot(profile.transition_position, profile[panel], color=COLORS.get(condition, "#666"),
                                  alpha=.10, linewidth=.7)
            summary = one_system.groupby(["condition", "transition_position"], as_index=False)[panel].mean()
            for condition, profile in summary.groupby("condition"):
                axis.plot(profile.transition_position, profile[panel], marker="o", linewidth=2.2,
                          color=COLORS.get(condition, "#666"), label=condition)
            title = "raw profile" if panel == "M" else "centered shape"
            if system == "SRN":
                title += " (whole-profile Fast/Slow)"
            axis.set_title(f"{system}: {title}")
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Ordinal transition")
            _style(axis)
            if col == 0:
                axis.legend(frameon=False, ncol=2)
    fig.suptitle("Figure 1. Condition level and transition shape", fontsize=16, fontweight="bold")
    _finish(fig, output, "figure_1_raw_vs_centered_profiles")


def figure_delta_reuse(source: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharey=True)
    for axis, system in zip(axes, ("Human", "SRN")):
        system_rows = source[source.system.eq(system)]
        for condition, one in system_rows.groupby("condition"):
            axis.scatter(one.delta, one.profile_correlation, s=22, alpha=.55,
                         color=COLORS.get(condition, "#666"), label=condition)
            axis.scatter(one.delta.mean(), one.profile_correlation.mean(), s=170, marker="X",
                         edgecolor="black", linewidth=.7, color=COLORS.get(condition, "#666"))
        axis.axvline(0, color="#111827", ls="--", lw=1)
        axis.axhline(0, color="#111827", ls=":", lw=1)
        units = "cleaned ms" if system == "Human" else "frozen normalized RT"
        axis.set_xlabel(f"Additive δ ({units})")
        axis.set_title(system)
        axis.legend(frameon=False, fontsize=8)
        _style(axis)
    axes[0].set_ylabel("Centered-profile correlation")
    fig.suptitle("Figure 2. Global cost versus structured reuse", fontsize=16, fontweight="bold")
    _finish(fig, output, "figure_2_delta_vs_structured_reuse")


def figure_q_delta(source: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rng = np.random.default_rng(203)
    for row, system in enumerate(("Human", "SRN")):
        groups = list(source[source.system.eq(system)].groupby("condition", sort=False))
        labels = [condition for condition, _ in groups]
        for x, (condition, one) in enumerate(groups):
            jitter = rng.normal(0, .06, size=len(one))
            color = COLORS.get(condition, "#666")
            axes[row, 0].scatter(np.full(len(one), x) + jitter, one.q_unconstrained,
                                 color=color, alpha=.45, s=18)
            axes[row, 1].scatter(np.full(len(one), x) + jitter, one.delta,
                                 color=color, alpha=.45, s=18)
        for col, ylabel in enumerate(("Unconstrained q", "Additive δ")):
            axis = axes[row, col]
            axis.set_xticks(range(len(labels)), labels)
            axis.set_ylabel(ylabel)
            axis.set_title(f"{system}: {ylabel}")
            _style(axis)
        axes[row, 0].axhline(.5, color="#111827", ls="--", lw=1)
        axes[row, 1].axhline(0, color="#111827", ls="--", lw=1)
    fig.suptitle("Figure 3. Unit-level retrieval weights and global costs", fontsize=16, fontweight="bold")
    _finish(fig, output, "figure_3_q_delta_distributions")


def figure_model_scores(source: pd.DataFrame, output: Path):
    metrics = [
        ("heldout_rmse", "Raw RMSE", "raw"),
        ("condition_level_variance_explained", "Level variance explained", "raw"),
        ("centered_shape_variance_explained", "Centered-shape variance explained", "centered_shape"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for axis, (metric, title, score_type) in zip(axes, metrics):
        one = source[source.score_type.eq(score_type)].copy()
        one["label"] = one.model_label.astype(str) + "\n" + one.condition.astype(str)
        axis.barh(np.arange(len(one)), one[metric], color="#4C78A8")
        axis.set_yticks(np.arange(len(one)), one.label, fontsize=7)
        axis.set_title(title)
        _style(axis)
    fig.suptitle("Figure 4. Existing held-out models by score family", fontsize=16, fontweight="bold")
    _finish(fig, output, "figure_4_existing_model_scores")


def figure_history(source: pd.DataFrame, output: Path):
    if source.empty or not {"parameter", "estimate"}.issubset(source):
        return
    one = source[source.parameter.isin(["history_delta", "history_q"])].copy()
    if one.empty:
        return
    fig, axis = plt.subplots(figsize=(10, 5.5))
    labels = one.predictor.astype(str) + ": " + one.parameter.astype(str)
    axis.errorbar(np.arange(len(one)), one.estimate,
                  yerr=[one.estimate - one.ci_lower, one.ci_upper - one.estimate],
                  fmt="o", color="#0072B2", capsize=3)
    axis.axhline(0, color="#111827", ls="--", lw=1)
    axis.set_xticks(np.arange(len(one)), labels, rotation=35, ha="right")
    axis.set_ylabel("Estimated history effect")
    _style(axis)
    fig.suptitle("Figure 5. Identifiable history effects", fontsize=16, fontweight="bold")
    _finish(fig, output, "figure_5_history_effects")


def figure_novel_default(source: pd.DataFrame, output: Path):
    if source.empty:
        return
    actual_column = "actual_centered_standardized" if "actual_centered_standardized" in source else "actual_centered"
    predicted_column = "predicted_centered_standardized" if "predicted_centered_standardized" in source else "predicted_centered"
    summary = source.groupby(["model", "transition_position"], as_index=False).agg(
        actual=(actual_column, "mean"), predicted=(predicted_column, "mean")
    )
    models = [m for m in summary.model.unique() if m != "N0_level_only"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        one = summary[summary.model.eq(model)]
        axis.plot(one.transition_position, one.actual, marker="o", color="#111827", label="Observed")
        axis.plot(one.transition_position, one.predicted, marker="o", color="#CC79A7", label="Predicted")
        axis.axhline(0, color="#9CA3AF", lw=.8)
        axis.set_title(model.replace("_", " "), fontsize=9)
        axis.set_xlabel("Transition")
        _style(axis)
    axes[0].set_ylabel("Standardized centered Novel RT")
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 6. Independent-default Novel comparison", fontsize=16, fontweight="bold")
    _finish(fig, output, "figure_6_novel_default_comparison")
