"""Required publication-oriented figures for probe analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"A": "#3569b7", "B": "#d05a45", "Probe": "#7a4da3"}


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_human_phenomenon(data: pd.DataFrame, path: Path) -> None:
    datasets = list(data.dataset.unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 3.8 * len(datasets)), squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        group = data[data.dataset.eq(dataset)].copy()
        for context in ("A", "B", "Probe"):
            summary = group[group.context.eq(context)].groupby("global_sequence").log_rt.mean()
            if len(summary):
                ax.plot(summary.index, summary.values, label=context, color=COLORS[context], linewidth=2)
        midpoint = group.assign(midpoint=0.5 * (group.pred_A_log_rt + group.pred_B_log_rt)).groupby(
            "global_sequence"
        ).midpoint.mean()
        ax.plot(midpoint.index, midpoint.values, "--", color="black", alpha=0.65, label="A/B midpoint")
        ax.set(title=f"{dataset}: human trajectories", xlabel="Global sequence", ylabel="Mean log RT")
        ax.legend(frameon=False, ncol=4)
    _finish(fig, path)


def plot_mixture_indices(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    datasets = list(metrics.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        group = metrics[metrics.dataset.eq(dataset)]
        for (probe_type, previous), rows in group.groupby(["probe_type", "previous_context"], dropna=False):
            summary = rows.groupby("sequence_in_block").mixture_index_log.agg(["mean", "sem"])
            label = f"{probe_type}; prev={previous}"
            ax.plot(summary.index, summary["mean"], marker="o", label=label)
            ax.fill_between(
                summary.index,
                summary["mean"] - summary["sem"],
                summary["mean"] + summary["sem"],
                alpha=0.15,
            )
        ax.axhline(0.5, linestyle="--", color="black", linewidth=1)
        ax.set(title=dataset, xlabel="Sequence in probe block", ylabel="Mixture index (log RT)")
        ax.legend(frameon=False, fontsize=8)
    _finish(fig, path)


def plot_model_predictions(predictions: pd.DataFrame, path: Path) -> None:
    probe = predictions[predictions.is_probe & predictions.split.eq("test")]
    if probe.empty:
        probe = predictions[predictions.is_probe]
    if probe.empty:
        return
    datasets = list(probe.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        group = probe[probe.dataset.eq(dataset)]
        observed = group.groupby("sequence_in_block").log_rt.mean()
        ax.plot(observed.index, observed.values, color="black", marker="o", linewidth=2.5, label="Observed")
        for model, rows in group.groupby("model"):
            summary = rows.groupby("sequence_in_block").predicted_log_rt.mean()
            ax.plot(summary.index, summary.values, marker=".", label=model)
        ax.set(title=dataset, xlabel="Sequence in probe block", ylabel="Log RT")
        ax.legend(frameon=False, fontsize=7)
    _finish(fig, path)


def plot_aftereffects(aftereffects: pd.DataFrame, path: Path) -> None:
    if aftereffects.empty:
        return
    summary = aftereffects.groupby(["dataset", "return_context"]).aftereffect_log_rt.agg(["mean", "sem"])
    labels = [f"{dataset}\n{return_context}" for dataset, return_context in summary.index]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(np.arange(len(labels)), summary["mean"], yerr=summary["sem"], color="#6699a8", capsize=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels, ylabel="Observed - counterfactual log RT", title="Post-probe A/B aftereffects")
    _finish(fig, path)


def _first_json_number(value: object) -> float:
    try:
        parsed = json.loads(str(value))
        return float(parsed[0]) if parsed else np.nan
    except (ValueError, TypeError, json.JSONDecodeError):
        return np.nan


def plot_gate_dynamics(gate_states: pd.DataFrame, path: Path) -> None:
    subset = gate_states[(gate_states.hidden_dim.eq(1)) & gate_states.model.eq("gru_full")].copy()
    if subset.empty:
        return
    subset["h"] = subset.hidden_state.map(_first_json_number)
    subset["delta_h"] = subset.delta_hidden.map(_first_json_number)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for context, rows in subset.groupby("context"):
        ax.scatter(rows.h, rows.delta_h, s=9, alpha=0.25, label=context, color=COLORS.get(context))
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel=r"$h_t$", ylabel=r"$\Delta h_t$", title="One-dimensional gate dynamics")
    ax.legend(frameon=False)
    _finish(fig, path)


def plot_cue_morph(interventions: pd.DataFrame, path: Path) -> None:
    if interventions.empty:
        return
    subset = interventions[
        interventions.model.isin(["gru_full", "nonrecurrent_cue"])
        & interventions.hidden_dim.isin([0, 1])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (model, history), rows in subset.groupby(["model", "history"]):
        summary = rows.groupby("cue_A").agg(q=("q", "mean"), predicted_rt=("predicted_rt", "mean")).sort_index()
        label = f"{model}: {history}"
        axes[0].plot(summary.index, summary.q, marker="o", label=label)
        axes[1].plot(summary.index, summary.predicted_rt, marker="o", label=label)
    axes[0].set(xlabel="Maze/A cue evidence", ylabel="q(A)", title="Cue interpolation")
    axes[1].set(xlabel="Maze/A cue evidence", ylabel="Predicted RT", title="Behavioral cue morph")
    axes[0].legend(frameon=False, fontsize=7)
    _finish(fig, path)


def plot_recovery(confusion_long: pd.DataFrame, path: Path) -> None:
    if confusion_long.empty:
        return
    datasets = list(confusion_long.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        group = confusion_long[confusion_long.dataset.eq(dataset)]
        matrix = group.pivot(index="generating_mechanism", columns="selected_model", values="proportion").fillna(0)
        image = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="Blues")
        n_simulations = int(group.n_simulations.iloc[0]) if "n_simulations" in group else 0
        title = f"{dataset} (n={n_simulations}/mechanism)" if n_simulations else dataset
        ax.set(xticks=np.arange(len(matrix.columns)), xticklabels=matrix.columns, yticks=np.arange(len(matrix.index)), yticklabels=matrix.index, title=title, xlabel="Selected", ylabel="Generated")
        ax.tick_params(axis="x", rotation=45)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046)
    _finish(fig, path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    heldout = comparison[(comparison.split.eq("test")) & comparison.region.eq("all")]
    if heldout.empty:
        heldout = comparison[comparison.region.eq("all")]
    if heldout.empty:
        return
    pivot = heldout.groupby(["dataset", "model"]).rmse.mean().sort_values()
    labels = [f"{dataset}\n{model}" for dataset, model in pivot.index]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.5))
    ax.bar(np.arange(len(labels)), pivot.values, color="#7b8fb2")
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels, ylabel="Held-out log-RT RMSE", title="Model comparison")
    ax.tick_params(axis="x", rotation=45)
    _finish(fig, path)


def make_required_figures(
    output_dir: Path,
    *,
    counterfactuals: pd.DataFrame,
    behavioral_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    aftereffects: pd.DataFrame,
    gate_states: pd.DataFrame,
    cue_interventions: pd.DataFrame,
    recovery_confusion: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_human_phenomenon(counterfactuals, figures / "figure_1_human_phenomenon.png")
    plot_mixture_indices(behavioral_metrics, figures / "figure_2_mixture_index.png")
    plot_model_predictions(predictions, figures / "figure_3_model_predictions.png")
    plot_aftereffects(aftereffects, figures / "figure_4_aftereffects.png")
    plot_gate_dynamics(gate_states, figures / "figure_5_gate_dynamics.png")
    plot_cue_morph(cue_interventions, figures / "figure_6_cue_morph.png")
    plot_recovery(recovery_confusion, figures / "figure_7_model_recovery.png")
    plot_model_comparison(comparison, figures / "figure_8_model_comparison.png")
