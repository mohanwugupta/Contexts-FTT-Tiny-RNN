"""Figures for trial-wise SRN validation and state-space fit overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CONDITION_LABELS = {
    "High": "High scale + execution",
    "Low": "Low scale + execution",
    "High_seq_only": "High scale, sequence only",
}
CONTEXT_COLORS = {"A": "#2563A6", "B": "#D55E00"}
MODEL_COLORS = {"Lump": "#E69F00", "Cross": "#009E73", "Split": "#8C564B"}
MODEL_STYLES = {"Lump": "--", "Cross": "-.", "Split": ":"}


def _style_axis(axis, max_sequence: int):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.55)
    axis.set_xlim(1, max_sequence)
    ticks = sorted(set([1, max_sequence, *range(20, max_sequence, 20)]))
    axis.set_xticks(ticks)
    for boundary in range(10, max_sequence, 10):
        axis.axvline(boundary + 0.5, color="#AAAAAA", linewidth=0.45, alpha=0.45, zorder=0)


def plot_trial_trajectories(
    training: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    outcome: str,
    conditions: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
):
    fig, axes = plt.subplots(len(conditions), 1, figsize=(14, 3.6 * len(conditions)), sharex=True)
    axes = np.atleast_1d(axes)
    max_sequence = int(training.global_sequence.max())
    for axis, condition in zip(axes, conditions):
        raw = training[training.model_condition.eq(condition)]
        summary = summaries[
            summaries.model_condition.eq(condition) & summaries.outcome.eq(outcome)
        ]
        for _, seed_frame in raw.groupby("seed", sort=True):
            axis.plot(
                seed_frame.global_sequence,
                seed_frame[outcome],
                color="#6B7280",
                alpha=0.10,
                linewidth=0.7,
                zorder=1,
            )
        for (_, context), block in summary.groupby(["block_index", "context"], sort=True):
            color = CONTEXT_COLORS[context]
            x = block.global_sequence.to_numpy(dtype=float)
            axis.fill_between(
                x,
                block.ci_low.to_numpy(dtype=float),
                block.ci_high.to_numpy(dtype=float),
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            axis.plot(x, block.observed, color=color, linewidth=2.4, zorder=3)
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"{CONDITION_LABELS.get(condition, condition)}  (n={raw.seed.nunique()} seeds)",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        _style_axis(axis, max_sequence)
    axes[-1].set_xlabel("Training sequence (10 sequences per A/B block)")
    legend = [
        Line2D([0], [0], color="#6B7280", alpha=.35, lw=1, label="Individual seed"),
        Line2D([0], [0], color=CONTEXT_COLORS["A"], lw=2.5, label="A mean ± 95% CI"),
        Line2D([0], [0], color=CONTEXT_COLORS["B"], lw=2.5, label="B mean ± 95% CI"),
    ]
    axes[0].legend(handles=legend, loc="upper right", frameon=False, ncol=3)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.998)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_state_space_overlays(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    outcome: str,
    conditions: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
):
    fig, axes = plt.subplots(len(conditions), 1, figsize=(15, 4.2 * len(conditions)), sharex=True)
    axes = np.atleast_1d(axes)
    max_sequence = int(predictions.loc[predictions.outcome.eq(outcome), "global_sequence"].max())
    for axis, condition in zip(axes, conditions):
        panel = predictions[
            predictions.outcome.eq(outcome) & predictions.model_condition.eq(condition)
        ]
        observed = panel[panel.model.eq("Lump")].sort_values("global_sequence")
        for context, context_frame in observed.groupby("context", sort=False):
            axis.scatter(
                context_frame.global_sequence,
                context_frame.observed,
                s=18,
                color=CONTEXT_COLORS[context],
                alpha=.82,
                zorder=4,
                label=f"Observed {context}",
            )
        axis.plot(observed.global_sequence, observed.observed, color="#374151", alpha=.28, lw=1.0)
        axis.fill_between(
            observed.global_sequence.to_numpy(dtype=float),
            observed.ci_low.to_numpy(dtype=float),
            observed.ci_high.to_numpy(dtype=float),
            color="#111827",
            alpha=.08,
            linewidth=0,
        )
        for model in ("Lump", "Cross", "Split"):
            model_frame = panel[panel.model.eq(model)].sort_values("global_sequence")
            axis.plot(
                model_frame.global_sequence,
                model_frame.predicted,
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                linewidth=2.4,
                label=model,
                zorder=3,
            )
        panel_metrics = metrics[
            metrics.outcome.eq(outcome) & metrics.model_condition.eq(condition)
        ].sort_values("bic")
        winner = panel_metrics.iloc[0]
        axis.set_title(
            f"{CONDITION_LABELS.get(condition, condition)} — lowest BIC: "
            f"{winner.model} (RMSE={winner.rmse:.4f}, R²={winner.r_squared:.3f})",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        axis.set_ylabel(ylabel)
        _style_axis(axis, max_sequence)
    axes[-1].set_xlabel("Training sequence")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", frameon=False, ncol=5)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.998)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_observed_vs_predicted(
    predictions: pd.DataFrame,
    *,
    output_path: Path,
):
    outcomes = ["sequence_ce", "normalized_rt"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2))
    condition_colors = {"High": "#0072B2", "Low": "#D55E00", "High_seq_only": "#6B7280"}
    for row, outcome in enumerate(outcomes):
        outcome_frame = predictions[predictions.outcome.eq(outcome)]
        for column, model in enumerate(("Lump", "Cross", "Split")):
            axis = axes[row, column]
            panel = outcome_frame[outcome_frame.model.eq(model)]
            for condition, group in panel.groupby("model_condition", sort=False):
                axis.scatter(
                    group.observed,
                    group.predicted,
                    s=17,
                    alpha=.55,
                    color=condition_colors[condition],
                    label=CONDITION_LABELS.get(condition, condition),
                )
            low = min(panel.observed.min(), panel.predicted.min())
            high = max(panel.observed.max(), panel.predicted.max())
            axis.plot([low, high], [low, high], color="black", linestyle="--", lw=1)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(model, fontweight="bold")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(color="#DDDDDD", linewidth=.5, alpha=.5)
            if column == 0:
                axis.set_ylabel(
                    "Predicted sequence CE" if outcome == "sequence_ce" else "Predicted normalized RT"
                )
            if row == 1:
                axis.set_xlabel("Observed normalized RT")
            elif row == 0:
                axis.set_xlabel("Observed sequence CE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.suptitle("Observed versus state-space-predicted trial means", fontsize=16, fontweight="bold", y=1.045)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
