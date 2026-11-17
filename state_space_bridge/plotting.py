"""Paper-ready bridge figures with matched human/SRN visual grammar."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "state-space-bridge-matplotlib")
)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_COLORS = {
    "SS_LUMP": "#E69F00", "SS_CROSS": "#009E73", "SS_SPLIT": "#8C564B",
    "RET_CATEGORICAL": "#777777", "RET_SCALAR": "#E69F00",
    "RET_TOKEN": "#009E73", "RET_LOWRANK": "#7B2CBF", "RET_SRN": "#2B6CB0",
}
CONTEXT_COLORS = {"A": "#D55E00", "B": "#0072B2"}


def _save(fig, root: Path, name: str):
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def learning_overlays(predictions, root):
    primary = predictions[predictions.fit_scope.isin(["pre_probe_AB", "all_AB_training"])]
    panels = [
        ("human", "Combined", "Human Combined"),
        ("SRN", "High_scale_2_execution", "High scale 2 + execution"),
        ("SRN", "Low_scale_execution", "Low scale + execution"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    for ax, (agent, condition, title) in zip(axes, panels):
        data = primary[(primary.agent_type == agent) & (primary.condition == condition)]
        if data.empty:
            ax.text(.5, .5, "No eligible trace", ha="center", va="center")
            ax.set_title(title); continue
        # Keep the largest trace for legibility; every counterbalance remains in CSV.
        cb = data.groupby("counterbalance").global_step.max().idxmax()
        data = data[data.counterbalance == cb]
        observed = data.drop_duplicates("global_step").sort_values("global_step")
        for context, group in observed.groupby("context"):
            ax.scatter(group.global_step, group.observed, s=12,
                       color=CONTEXT_COLORS[context], alpha=.65, label=f"Observed {context}")
        for model, group in data.groupby("model"):
            group = group.sort_values("global_step")
            ax.plot(group.global_step, group.predicted, color=MODEL_COLORS[model],
                    lw=2, label=model.replace("SS_", ""))
        ax.set_title(f"{title} — {cb}")
        ax.set_ylabel("Sequence RT (ms)" if agent == "human" else "Normalized RT")
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Global sequence")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("SSB1. Pure A/B learning: observed trajectories and state-space fits", y=.995)
    _save(fig, Path(root), "SSB1_state_space_learning_overlays")


def bic_bridge(bic, parameters, root):
    primary = bic[bic.fit_scope.isin(["pre_probe_AB", "all_AB_training"])].copy()
    summary = primary.groupby(["agent_type", "condition", "model"], as_index=False).BIC.sum()
    summary["delta"] = summary.BIC - summary.groupby(
        ["agent_type", "condition"]
    ).BIC.transform("min")
    labels = summary[["agent_type", "condition"]].drop_duplicates()
    labels["label"] = labels.agent_type + ": " + labels.condition
    matrix = summary.merge(labels).pivot(index="label", columns="model", values="delta")
    matrix = matrix.reindex(columns=["SS_LUMP", "SS_CROSS", "SS_SPLIT"])
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, .55 * len(matrix) + 2)))
    image = axes[0].imshow(matrix.to_numpy(), cmap="viridis_r", aspect="auto")
    axes[0].set_xticks(range(3), ["Lump", "Cross", "Split"])
    axes[0].set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(3):
            axes[0].text(j, i, f"{matrix.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    axes[0].set_title("Summed delta BIC")
    fig.colorbar(image, ax=axes[0], fraction=.05)
    cross = parameters[parameters.parameter.eq("A_cross")].copy()
    if cross.empty:
        axes[1].text(.5, .5, "No Cross parameters", ha="center")
    else:
        cross["label"] = cross.agent_type + ": " + cross.condition + " / " + cross.counterbalance
        axes[1].barh(cross.label, cross.value, color=MODEL_COLORS["SS_CROSS"])
        axes[1].axvline(0, color="black", lw=1)
        axes[1].set_xlabel("Fitted A_cross")
    axes[1].set_title("Cross-talk estimates (descriptive)")
    fig.suptitle("SSB2. Functional learning-model comparison")
    _save(fig, Path(root), "SSB2_state_space_BIC_bridge")


def endpoint_figure(endpoints, root):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (dataset, group) in zip(axes, endpoints.groupby("dataset", sort=False)):
        summary = group.groupby("probe_epoch", sort=False).agg(
            observed=("probe_observed", "mean"), a=("A_prediction", "mean"),
            b=("B_prediction", "mean"),
        ).reset_index()
        x = np.arange(len(summary))
        ax.plot(x, summary.a, marker="o", label="Frozen A", color=CONTEXT_COLORS["A"])
        ax.plot(x, summary.b, marker="o", label="Frozen B", color=CONTEXT_COLORS["B"])
        ax.scatter(x, summary.observed, marker="x", color="black", label="Observed probe")
        ax.set_xticks(x, summary.probe_epoch, rotation=35, ha="right")
        ax.set_title(dataset); ax.set_ylabel("Sequence RT (ms)")
    axes[0].legend(frameon=False)
    fig.suptitle("SSB3. Frozen state-space A/B endpoints and probe observations")
    _save(fig, Path(root), "SSB3_frozen_AB_endpoints")


def retrieval_overlays(predictions, root):
    groups = list(predictions.groupby("probe_context", sort=False))
    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (probe, group) in zip(axes, groups):
        observed = group.groupby("transition_position").actual_log_rt.mean()
        ax.plot(observed.index, observed.values, color="black", marker="o", lw=3, label="Observed")
        for model, model_rows in group.groupby("model"):
            summary = model_rows.groupby("transition_position").predicted_log_rt.mean()
            ax.plot(summary.index, summary.values, marker="o", color=MODEL_COLORS[model],
                    label=model.replace("RET_", ""))
        ax.set_title("Mixed" if probe == "Both" else probe)
        ax.set_xlabel("Ordinal transition position"); ax.set_xticks(range(1, 8))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Held-out log RT")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.03, 1))
    fig.suptitle("SSB4. Separate frozen-endpoint retrieval model family")
    _save(fig, Path(root), "SSB4_retrieval_overlays")


def scalar_vector_figure(vector, basis, root):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot([0, 1], [0, 0], "o-", color="#555555")
    axes[0].text(0, .05, "x_A", ha="center"); axes[0].text(1, .05, "x_B", ha="center")
    axes[0].set_title("Scalar skill endpoints"); axes[0].set_yticks([])
    summary = vector.groupby("transition_position").agg(a=("r_a", "mean"), b=("r_b", "mean"))
    axes[1].plot(summary.index, summary.a, marker="o", color=CONTEXT_COLORS["A"], label="A")
    axes[1].plot(summary.index, summary.b, marker="o", color=CONTEXT_COLORS["B"], label="B")
    axes[1].set_title("Seven-dimensional sequence state"); axes[1].legend(frameon=False)
    for component, group in basis.groupby("component"):
        axes[2].plot(group.transition_position, group.loading, marker="o", label=f"Component {component}")
    axes[2].axhline(0, color="black", lw=.8); axes[2].legend(frameon=False)
    axes[2].set_title("Data-selected rank-2 basis")
    for ax in axes[1:]:
        ax.set_xlabel("Ordinal transition position"); ax.set_xticks(range(1, 8))
    fig.suptitle("SSB5. Scalar state → vector state → low-rank configured state")
    _save(fig, Path(root), "SSB5_scalar_to_vector_state")


def conceptual_bridge(summary, root):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    labels = [
        ("State-space learning", "What A/B skills were learned?"),
        ("Probe retrieval", "How are frozen skills expressed?"),
        ("Low-rank vector state", "What transition structure appears?"),
        ("SRN hidden state", "How can it be implemented causally?"),
    ]
    for index, (title, subtitle) in enumerate(labels):
        x = .02 + index * .25
        ax.add_patch(plt.Rectangle((x, .30), .21, .40, facecolor="#EDF2F7", edgecolor="#2D3748"))
        ax.text(x + .105, .56, title, ha="center", va="center", weight="bold")
        ax.text(x + .105, .42, subtitle, ha="center", va="center", fontsize=9, wrap=True)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + .25, .50), xytext=(x + .215, .50),
                        arrowprops={"arrowstyle": "->", "lw": 2})
    conclusion = str(summary.cross_system_conclusion.iloc[0]) if len(summary) else "Not evaluated"
    ax.text(.5, .12, f"Observed conclusion: {conclusion}", ha="center", weight="bold")
    fig.suptitle("SSB6. Functional-to-representational bridge")
    _save(fig, Path(root), "SSB6_functional_to_representational_bridge")


def make_bridge_figures(root, tables):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    learning_overlays(tables["state_space_predictions"], root)
    bic_bridge(tables["state_space_bic_comparison"], tables["state_space_parameters"], root)
    endpoint_figure(tables["frozen_AB_counterfactuals"], root)
    retrieval_overlays(tables["retrieval_predictions"], root)
    scalar_vector_figure(
        tables["human_vector_AB_profiles"], tables["human_low_rank_basis"], root
    )
    conceptual_bridge(tables["cross_system_bridge_summary"], root)
