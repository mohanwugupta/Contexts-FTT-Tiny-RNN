"""The eight epoch-structured PRD2 figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPOCH_COLORS = {
    "Mixed_1": "#d9c2ef",
    "Mixed_2": "#b991dc",
    "Post_Mixed_Return": "#b9d9c4",
    "Mixed_Test": "#efc27d",
}
CONTEXT_COLORS = {"A": "#3569b7", "B": "#d05a45", "Probe": "#7a4da3"}


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _shade_epochs(ax: plt.Axes, data: pd.DataFrame) -> None:
    for epoch, color in EPOCH_COLORS.items():
        rows = data[data.epoch.astype("object").eq(epoch)]
        if rows.empty:
            continue
        ax.axvspan(rows.global_sequence.min() - 0.5, rows.global_sequence.max() + 0.5, color=color, alpha=0.35)


def plot_epoch_timeline(data: pd.DataFrame, path: Path) -> None:
    summary = data.groupby(["global_sequence", "context", "epoch"], observed=True).log_rt.mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    _shade_epochs(ax, data)
    overall = summary.groupby("global_sequence").log_rt.mean().sort_index()
    ax.plot(overall.index, overall, color="#555555", linewidth=1.2, alpha=0.7)
    for context, rows in summary.groupby("context"):
        ax.scatter(rows.global_sequence, rows.log_rt, color=CONTEXT_COLORS.get(context), s=11, label=context, zorder=3)
    ax.set(title="Human timeline: nine analyzed sequences per block", xlabel="Analyzed sequence across experiment", ylabel="Mean log RT")
    ax.legend(frameon=False, ncol=3)
    _finish(fig, path)


def plot_initial_mixed(data: pd.DataFrame, path: Path) -> None:
    rows = data[data.epoch.astype("object").eq("Mixed_1")]
    histories = list(rows.counterbalance_history.unique())
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * max(len(histories), 1), 4), squeeze=False)
    for ax, history in zip(axes[0], histories):
        group = rows[rows.counterbalance_history.eq(history)]
        by = group.groupby("analysis_sequence_in_block")
        observed = by.log_rt.mean()
        a = by.pred_A_log_rt.mean()
        b = by.pred_B_log_rt.mean()
        ax.plot(observed.index, observed, "ko-", label="Observed")
        ax.plot(a.index, a, color=CONTEXT_COLORS["A"], label="A counterfactual")
        ax.plot(b.index, b, color=CONTEXT_COLORS["B"], label="B counterfactual")
        ax.plot(a.index, 0.5 * (a + b), "--", color="gray", label="Midpoint")
        ax.set(title=str(history), xlabel="Analyzed sequence", ylabel="Log RT")
    if histories:
        axes[0, 0].legend(frameon=False, fontsize=8)
    _finish(fig, path)


def plot_immediate_adaptation(data: pd.DataFrame, path: Path) -> None:
    rows = data[data.epoch.astype("object").isin(["Mixed_1", "Mixed_2"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for epoch, group in rows.groupby("epoch", observed=True):
        summary = group.groupby("analysis_sequence_in_block").log_rt.agg(["mean", "sem"])
        axes[0].plot(summary.index, summary["mean"], marker="o", label=str(epoch))
    contrast = rows.pivot_table(
        index=["participant_id", "analysis_sequence_in_block"], columns="epoch", values="log_rt", observed=True
    ).reset_index()
    if {"Mixed_1", "Mixed_2"}.issubset(contrast.columns):
        contrast["difference"] = contrast["Mixed_2"] - contrast["Mixed_1"]
        summary = contrast.groupby("analysis_sequence_in_block").difference.agg(["mean", "sem"])
        axes[1].errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o")
    axes[0].set(title="Mixed 1 and Mixed 2", xlabel="Analyzed sequence", ylabel="Mean log RT")
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(title="Immediate adaptation: Mixed 2 - Mixed 1", xlabel="Analyzed sequence", ylabel="Log-RT contrast")
    _finish(fig, path)


def plot_delayed_mixed_test(predictions: pd.DataFrame, path: Path) -> None:
    selected = ["static_mixture", "contextual_inference", "gate_adaptation", "third_state"]
    rows = predictions[
        predictions.epoch.astype("object").eq("Mixed_Test") & predictions.model.isin(selected)
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if not rows.empty:
        observed = rows.drop_duplicates(["participant_id", "global_sequence"]).groupby("analysis_sequence_in_block").log_rt.mean()
        ax.plot(observed.index, observed, "ko-", linewidth=2.5, label="Observed")
        for model, group in rows.groupby("model"):
            summary = group.groupby("analysis_sequence_in_block").predicted_log_rt.mean()
            ax.plot(summary.index, summary, marker=".", label=model)
    ax.set(title="Delayed Mixed Test: no final-test refitting", xlabel="Analyzed sequence", ylabel="Log RT")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, path)


def plot_post_mixed(aftereffects: pd.DataFrame, path: Path) -> None:
    group_columns = ["return_context", "counterbalance_history"]
    summary = aftereffects.groupby(group_columns).aftereffect_log_rt.agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [f"{context}\n{history}" for context, history in summary.index]
    ax.bar(np.arange(len(summary)), summary["mean"], yerr=summary["sem"], color="#6699a8", capsize=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="First A/B block after 18 Mixed sequences",
        ylabel="Observed - A/B counterfactual log RT",
        xticks=np.arange(len(summary)), xticklabels=labels,
    )
    _finish(fig, path)


def plot_model_performance(comparison: pd.DataFrame, path: Path) -> None:
    regions = ["mixed_1", "mixed_2", "post_mixed_return", "mixed_test", "all"]
    rows = comparison[comparison.split.eq("test") & comparison.region.isin(regions)]
    matrix = rows.pivot_table(index="model", columns="region", values="rmse", aggfunc="mean").reindex(columns=regions)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set(xticks=np.arange(len(regions)), xticklabels=regions, yticks=np.arange(len(matrix)), yticklabels=matrix.index, title="Held-out model performance by epoch")
    ax.tick_params(axis="x", rotation=30)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.3f}" if np.isfinite(value) else "-", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(image, ax=ax, label="Log-RT RMSE")
    _finish(fig, path)


def plot_recurrent_state(gate_states: pd.DataFrame, resets: pd.DataFrame, path: Path) -> None:
    state = gate_states[gate_states.model.eq("N1_full_h1")].copy()
    if not state.empty:
        state["h"] = state.hidden_state.map(lambda value: float(json.loads(value)[0]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    if not state.empty:
        summary = state.groupby(["global_sequence", "epoch"], observed=True).h.mean().reset_index()
        _shade_epochs(axes[0], state)
        axes[0].plot(summary.global_sequence, summary.h, color="#4d4d4d")
        axes[0].set(title="One-dimensional recurrent state over the full timeline", ylabel=r"Mean $h_t$")
    if not resets.empty:
        pivot = resets.groupby(["model", "epoch"]).mean_delta_q_from_full.mean().unstack("epoch")
        pivot.plot(kind="bar", ax=axes[1])
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set(title="Hidden-state reset interventions", ylabel=r"Change in $q$ from full recurrence", xlabel="Intervention")
        axes[1].legend(frameon=False, fontsize=8)
    _finish(fig, path)


def plot_epoch_recovery(confusion: pd.DataFrame, path: Path) -> None:
    if confusion.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Recovery not run", ha="center", va="center")
        _finish(fig, path)
        return
    preferred = confusion[confusion.recovery_regime.eq("neighborhood")]
    if preferred.empty:
        preferred = confusion[confusion.recovery_regime.eq("human")]
    if preferred.empty:
        preferred = confusion
    regime = str(preferred.recovery_regime.iloc[0]) if "recovery_regime" in preferred else "configured"
    matrix = preferred.pivot_table(index="generating_mechanism", columns="selected_model", values="proportion", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    title = "Model recovery near the human parameter region" if regime in {"human", "neighborhood"} else f"Model recovery: {regime} regime"
    ax.set(xticks=np.arange(len(matrix.columns)), xticklabels=matrix.columns, yticks=np.arange(len(matrix)), yticklabels=matrix.index, xlabel="Selected", ylabel="Generated", title=title)
    ax.tick_params(axis="x", rotation=45)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    _finish(fig, path)


def make_epoch_figures(
    output_dir: Path,
    *,
    counterfactuals: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    aftereffects: pd.DataFrame,
    gate_states: pd.DataFrame,
    gate_resets: pd.DataFrame,
    recovery_confusion: pd.DataFrame,
) -> None:
    figures = Path(output_dir) / "figures_v2"
    figures.mkdir(parents=True, exist_ok=True)
    plot_epoch_timeline(counterfactuals, figures / "figure_1_human_timeline.png")
    plot_initial_mixed(counterfactuals, figures / "figure_2_initial_mixed_retrieval.png")
    plot_immediate_adaptation(counterfactuals, figures / "figure_3_immediate_adaptation.png")
    plot_delayed_mixed_test(predictions, figures / "figure_4_delayed_mixed_test.png")
    plot_post_mixed(aftereffects, figures / "figure_5_post_mixed_AB_consequence.png")
    plot_model_performance(comparison, figures / "figure_6_model_performance_by_epoch.png")
    plot_recurrent_state(gate_states, gate_resets, figures / "figure_7_recurrent_state.png")
    plot_epoch_recovery(recovery_confusion, figures / "figure_8_model_recovery.png")
