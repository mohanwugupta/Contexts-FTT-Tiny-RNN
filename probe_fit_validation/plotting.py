"""Publication-quality fit overlays for retrieval and low-rank probe models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "probe-fit-validation-matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


MODEL_ORDER = [
    "M0_scalar_context", "M1_scalar_mixture", "M2_token_specific",
    "M3_low_rank", "M4_srn_informed",
]
RANK2_ORDER = ["scalar_baseline", "shared_ab_basis", "separate_dataset_ab_basis"]
MODEL_COLORS = {
    "M0_scalar_context": "#6B7280",
    "M1_scalar_mixture": "#E69F00",
    "M2_token_specific": "#009E73",
    "M3_low_rank": "#7B2CBF",
    "M4_srn_informed": "#2B6CB0",
    "scalar_baseline": "#E69F00",
    "shared_ab_basis": "#7B2CBF",
    "separate_dataset_ab_basis": "#2B6CB0",
}
CONTEXT_ORDER = ["Both", "Novel"]
CONTEXT_LABELS = {"Both": "Mixed", "Novel": "Novel"}


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=.7, alpha=.55)
    axis.set_xticks(range(1, 8))


def _band(axis, x, low, high, color, alpha):
    axis.fill_between(
        np.asarray(x, dtype=float), np.asarray(low, dtype=float), np.asarray(high, dtype=float),
        color=color, alpha=alpha, linewidth=0,
    )


def plot_retrieval_overlays(curves, metrics, output_path):
    overall = curves[curves.probe_epoch.eq("All probe epochs")]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.8), sharex=True, sharey="row")
    for row, context in enumerate(CONTEXT_ORDER):
        for column, model in enumerate(MODEL_ORDER):
            axis = axes[row, column]
            panel = overall[
                overall.probe_context.eq(context) & overall.model.eq(model)
            ].sort_values("transition_position")
            if panel.empty:
                axis.text(.5, .5, "No rows", transform=axis.transAxes, ha="center")
                continue
            color = MODEL_COLORS[model]
            _band(axis, panel.transition_position, panel.actual_ci_low, panel.actual_ci_high,
                  "#111827", .11)
            _band(axis, panel.transition_position, panel.predicted_ci_low,
                  panel.predicted_ci_high, color, .16)
            axis.plot(panel.transition_position, panel.actual_mean, color="#111827",
                      marker="o", lw=2.5, label="Observed")
            axis.plot(panel.transition_position, panel.predicted_mean, color=color,
                      marker="s", lw=2.2, linestyle="--", label="Predicted")
            score = metrics[
                metrics.probe_context.eq(context) & metrics.model.eq(model)
            ].iloc[0]
            title = str(panel.model_label.iloc[0])
            axis.set_title(
                f"{title}\nheld-out R²={score.r_squared_observation:.3f}; "
                f"profile R²={score.r_squared_profile_cells:.3f}",
                fontsize=10.5, fontweight="bold",
            )
            if column == 0:
                axis.set_ylabel(f"{CONTEXT_LABELS[context]} held-out log RT")
            if row == 1:
                axis.set_xlabel("Transition position")
            _style(axis)
    handles = [
        Line2D([0], [0], color="#111827", marker="o", lw=2.5, label="Observed mean ± 95% CI"),
        Line2D([0], [0], color="#7B2CBF", marker="s", linestyle="--", lw=2.2,
               label="Model prediction ± 95% CI"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.015))
    fig.suptitle(
        "Held-out human probe data and retrieval-model predictions",
        fontsize=16, fontweight="bold", y=1.055,
    )
    _finish(fig, output_path)


def _epoch_order(context, values):
    preferred = ["Mixed1", "Mixed2", "MixedTest"] if context == "Both" else [
        "Novel1", "Novel2", "NovelTest"
    ]
    return [value for value in preferred if value in set(values)]


def plot_low_rank_by_epoch(curves, output_path):
    data = curves[
        curves.model.eq("M3_low_rank") & ~curves.probe_epoch.eq("All probe epochs")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), sharex=True, sharey="row")
    for row, context in enumerate(CONTEXT_ORDER):
        epochs = _epoch_order(context, data.loc[data.probe_context.eq(context), "probe_epoch"])
        for column, epoch in enumerate(epochs):
            axis = axes[row, column]
            panel = data[
                data.probe_context.eq(context) & data.probe_epoch.eq(epoch)
            ].sort_values("transition_position")
            _band(axis, panel.transition_position, panel.actual_ci_low, panel.actual_ci_high,
                  "#111827", .11)
            _band(axis, panel.transition_position, panel.predicted_ci_low,
                  panel.predicted_ci_high, MODEL_COLORS["M3_low_rank"], .18)
            axis.plot(panel.transition_position, panel.actual_mean, color="#111827",
                      marker="o", lw=2.5, label="Observed")
            axis.plot(panel.transition_position, panel.predicted_mean,
                      color=MODEL_COLORS["M3_low_rank"], marker="s", linestyle="--",
                      lw=2.3, label="Low-rank prediction")
            axis.set_title(epoch, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{CONTEXT_LABELS[context]} held-out log RT")
            if row == 1:
                axis.set_xlabel("Transition position")
            _style(axis)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Observed and low-rank-predicted profiles within each probe epoch (v3 rank 5)",
        fontsize=15, fontweight="bold", y=1.015,
    )
    _finish(fig, output_path)


def plot_rank2_overlays(curves, metrics, output_path):
    overall = curves[curves.probe_epoch.eq("All probe epochs")]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.8), sharex=True, sharey="row")
    for row, context in enumerate(CONTEXT_ORDER):
        context_metrics = metrics[metrics.probe_context.eq(context)]
        winner = context_metrics.loc[context_metrics.rmse.idxmin(), "model"]
        for column, model in enumerate(RANK2_ORDER):
            axis = axes[row, column]
            panel = overall[
                overall.probe_context.eq(context) & overall.model.eq(model)
            ].sort_values("transition_position")
            color = MODEL_COLORS[model]
            _band(axis, panel.transition_position, panel.actual_ci_low, panel.actual_ci_high,
                  "#111827", .11)
            _band(axis, panel.transition_position, panel.predicted_ci_low,
                  panel.predicted_ci_high, color, .18)
            axis.axhline(0, color="#9CA3AF", lw=.8)
            axis.plot(panel.transition_position, panel.actual_mean, color="#111827",
                      marker="o", lw=2.5, label="Observed deviation")
            axis.plot(panel.transition_position, panel.predicted_mean, color=color,
                      marker="s", linestyle="--", lw=2.3, label="Predicted deviation")
            score = context_metrics[context_metrics.model.eq(model)].iloc[0]
            prefix = "Best RMSE — " if model == winner else ""
            axis.set_title(
                f"{prefix}{panel.model_label.iloc[0]}\n"
                f"target R²={score.r_squared_observation:.3f}; "
                f"profile R²={score.r_squared_profile_cells:.3f}",
                fontsize=10.5, fontweight="bold",
            )
            if column == 0:
                axis.set_ylabel(f"{CONTEXT_LABELS[context]} midpoint deviation")
            if row == 1:
                axis.set_xlabel("Transition position")
            _style(axis)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Held-out probe reconstruction in the independently selected rank-2 A/B space",
        fontsize=15, fontweight="bold", y=1.015,
    )
    _finish(fig, output_path)


def _identity_panel(axis, frame, actual, predicted, title, annotation):
    axis.hexbin(
        frame[actual].to_numpy(dtype=float), frame[predicted].to_numpy(dtype=float),
        gridsize=34, mincnt=1, cmap="Blues", linewidths=0,
    )
    low = float(min(frame[actual].min(), frame[predicted].min()))
    high = float(max(frame[actual].max(), frame[predicted].max()))
    axis.plot([low, high], [low, high], color="#111827", linestyle="--", lw=1.2)
    axis.set_xlim(low, high); axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"{title}\n{annotation}", fontsize=11, fontweight="bold")
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.5, alpha=.35)


def plot_low_rank_calibration(retrieval, retrieval_metrics, rank2, rank2_metrics, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2))
    for row, context in enumerate(CONTEXT_ORDER):
        v3 = retrieval[
            retrieval.probe_context.eq(context) & retrieval.model.eq("M3_low_rank")
        ]
        v3_score = retrieval_metrics[
            retrieval_metrics.probe_context.eq(context)
            & retrieval_metrics.model.eq("M3_low_rank")
        ].iloc[0]
        _identity_panel(
            axes[row, 0], v3, "actual_log_rt", "predicted_log_rt",
            f"{CONTEXT_LABELS[context]}: v3 low-rank retrieval",
            f"held-out R²={v3_score.r_squared_observation:.3f}; RMSE={v3_score.rmse:.3f}",
        )
        context_rank2 = rank2[rank2.probe_context.eq(context)]
        metric_context = rank2_metrics[rank2_metrics.probe_context.eq(context)]
        winner = metric_context.loc[metric_context.rmse.idxmin(), "model"]
        selected = context_rank2[context_rank2.model.eq(winner)]
        selected_score = metric_context[metric_context.model.eq(winner)].iloc[0]
        _identity_panel(
            axes[row, 1], selected, "actual_log_rt", "predicted_log_rt",
            f"{CONTEXT_LABELS[context]}: {selected.model_label.iloc[0]}",
            f"held-out log-RT R²={selected_score.r_squared_observation_log_rt:.3f}; "
            f"RMSE={selected_score.rmse:.3f}",
        )
        axes[row, 0].set_ylabel("Predicted log RT")
        axes[row, 0].set_xlabel("Observed log RT")
        axes[row, 1].set_xlabel("Observed log RT")
    fig.suptitle(
        "Observation-level calibration of the two low-rank probe analyses",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _finish(fig, output_path)


def _grouped_bars(axis, metrics, value, models, title, ylabel):
    contexts = [context for context in CONTEXT_ORDER if context in set(metrics.probe_context)]
    x = np.arange(len(models), dtype=float)
    width = .34
    for offset, context, color in zip([-.17, .17], contexts, ["#805AD5", "#2F855A"]):
        values = []
        for model in models:
            cell = metrics[metrics.probe_context.eq(context) & metrics.model.eq(model)]
            values.append(float(cell[value].iloc[0]) if len(cell) else np.nan)
        axis.bar(x + offset, values, width=width, color=color, label=CONTEXT_LABELS[context])
    labels = [
        str(metrics.loc[metrics.model.eq(model), "model_label"].iloc[0]).replace(" (rank 5)", "")
        for model in models
    ]
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title(title, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.axhline(0, color="#6B7280", lw=.8)
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=.6, alpha=.5)


def plot_fit_metric_context(retrieval_metrics, rank2_metrics, state_space, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    _grouped_bars(
        axes[0, 0], retrieval_metrics, "r_squared_observation", MODEL_ORDER,
        "Retrieval: held-out observation-level fit", "R² on human probe log RT",
    )
    _grouped_bars(
        axes[0, 1], retrieval_metrics, "r_squared_profile_cells", MODEL_ORDER,
        "Retrieval: held-out epoch × transition profiles", "Profile-cell R²",
    )
    _grouped_bars(
        axes[1, 0], rank2_metrics, "r_squared_observation_log_rt", RANK2_ORDER,
        "Rank-2 analysis: held-out observation-level fit", "R² on human probe log RT",
    )
    axis = axes[1, 1]
    if state_space.empty:
        axis.text(.5, .5, "State-space reference not supplied", transform=axis.transAxes,
                  ha="center", va="center")
    else:
        labels = [f"{row.model_condition}\n{row.model}" for row in state_space.itertuples()]
        axis.bar(labels, state_space.r_squared, color=["#0072B2", "#D55E00"][:len(state_space)])
        for index, value in enumerate(state_space.r_squared):
            axis.text(index, value + .02, f"{value:.3f}", ha="center", va="bottom")
        axis.set_ylim(min(0, float(state_space.r_squared.min()) - .1), 1.05)
        axis.set_ylabel("R² on seed-averaged SRN normalized RT")
        axis.axhline(0, color="#6B7280", lw=.8)
    axis.set_title("State-space RT reference (in-sample, different target)", fontweight="bold")
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=.6, alpha=.5)
    axes[0, 0].legend(frameon=False); axes[1, 0].legend(frameon=False)
    fig.suptitle(
        "Fit context — panels use different outcomes and are not a direct model competition",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _finish(fig, output_path)


def make_probe_fit_figures(
    output_dir,
    *,
    retrieval,
    retrieval_curves,
    retrieval_metrics,
    rank2,
    rank2_curves,
    rank2_metrics,
    state_space,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_retrieval_overlays(
        retrieval_curves, retrieval_metrics, output / "figure_01_retrieval_model_overlays"
    )
    plot_low_rank_by_epoch(
        retrieval_curves, output / "figure_02_low_rank_retrieval_by_epoch"
    )
    plot_rank2_overlays(
        rank2_curves, rank2_metrics, output / "figure_03_rank2_linear_algebra_overlays"
    )
    plot_low_rank_calibration(
        retrieval, retrieval_metrics, rank2, rank2_metrics,
        output / "figure_04_low_rank_observed_vs_predicted",
    )
    plot_fit_metric_context(
        retrieval_metrics, rank2_metrics, state_space,
        output / "figure_05_fit_metric_context",
    )
