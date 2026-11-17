"""Eight required PRD3 retrieval figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_conceptual(path: Path) -> None:
    labels = ["Task-derived\ncontext", "Similarity to\nA/B prototypes", "Responsibility\nqA / qB", "Frozen A + B\nskill experts", "Expressed\nlog RT"]
    fig, ax = plt.subplots(figsize=(11, 2.7))
    ax.axis("off")
    for index, label in enumerate(labels):
        x = 0.08 + index * 0.21
        ax.text(x, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round,pad=.5", "facecolor": "#e8eef7", "edgecolor": "#4d6480"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + 0.14, 0.5), xytext=(x + 0.07, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.set_title("Similarity-weighted retrieval over already-separated skills")
    _finish(fig, path)


def plot_geometry(features: pd.DataFrame, geometry: pd.DataFrame, path: Path) -> None:
    merged = features.merge(geometry, on="context")
    fig = plt.figure(figsize=(11, 4.5))
    axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2)]
    for row in merged.itertuples():
        axes[0].scatter(row.A_cue, row.B_cue, row.novel_cue, s=80)
        axes[0].text(row.A_cue + .025, row.B_cue + .025, row.novel_cue + .025, row.context)
    axes[0].set(xlabel="A cue", ylabel="B cue", zlabel="Novel cue", title="Task-derived context space", xlim=(-.1, 1.2), ylim=(-.1, 1.2), zlim=(-.1, 1.2))
    axes[1].scatter(merged.relative_evidence_R, merged.absolute_evidence_E, s=90)
    offsets = {"Both": (6, 10), "Default": (6, -14), "Novel": (6, 4), "A": (5, 4), "B": (5, 4)}
    for row in merged.itertuples():
        axes[1].annotate(
            row.context, (row.relative_evidence_R, row.absolute_evidence_E),
            xytext=offsets.get(row.context, (5, 4)), textcoords="offset points",
        )
    axes[1].set(xlabel="Relative evidence R = dB - dA", ylabel="Absolute evidence E", title="Retrieval geometry")
    _finish(fig, path)


def plot_human_readout(predictions: pd.DataFrame, path: Path) -> None:
    rows = predictions[
        predictions.model.eq("continuous_similarity")
        & ~predictions.retrieval_context.isin(["A", "B"])
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not rows.empty:
        summary = rows.groupby(["dataset", "retrieval_context"]).agg(
            observed=("log_rt", "mean"), predicted=("predicted_log_rt", "mean"),
            A=("pred_A_log_rt", "mean"), B=("pred_B_log_rt", "mean"),
        )
        x = np.arange(len(summary))
        ax.scatter(x, summary.A, label="A counterfactual", marker="_")
        ax.scatter(x, summary.B, label="B counterfactual", marker="_")
        ax.scatter(x, summary.observed, label="Observed", color="black")
        ax.scatter(x, summary.predicted, label="Similarity prediction", marker="x", s=70)
        ax.set_xticks(x, [f"{a}\n{b}" for a, b in summary.index])
    ax.set(title="Human readout geometry", ylabel="Mean log RT")
    ax.legend(frameon=False)
    _finish(fig, path)


def plot_structural(structural: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    selected = structural[structural.model.eq("continuous_similarity")] if "model" in structural else structural
    for context, group in selected.groupby("retrieval_context", sort=False):
        ax.scatter(group.AB_skill_separation, group.observed_probe_displacement, alpha=.65, label=context)
        order = np.argsort(group.AB_skill_separation)
        ax.plot(group.AB_skill_separation.iloc[order], group.predicted_probe_displacement.iloc[order], linewidth=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Participant A-B skill separation", ylabel="Probe - A/B midpoint", title="Participant structural prediction")
    ax.legend(frameon=False)
    _finish(fig, path)


def plot_continuous_categorical(predictions: pd.DataFrame, path: Path) -> None:
    rows = predictions[
        predictions.model.isin(["continuous_similarity", "categorical_similarity"])
        & ~predictions.retrieval_context.isin(["A", "B"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    observed = rows.drop_duplicates(["dataset", "participant_id", "global_sequence"])
    if not observed.empty:
        axes[0].hist(observed.log_rt, bins=25, density=True, alpha=.65, color="#666666")
        axes[0].set(title="Observed individual probe log RT", xlabel="Log RT", ylabel="Density")
        comparison = rows.groupby("model").observation_log_likelihood.sum()
        axes[1].bar(comparison.index, comparison.values, color=["#78909c", "#7e57c2"])
        axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Individual-sequence likelihood", ylabel="Summed log likelihood")
    _finish(fig, path)


def plot_transfer(combined_to_novel: pd.DataFrame, novel_to_combined: pd.DataFrame, path: Path) -> None:
    rows = pd.concat([combined_to_novel, novel_to_combined], ignore_index=True)
    if "evaluation_scope" in rows:
        rows = rows[rows.evaluation_scope.eq("overall")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not rows.empty:
        labels = [f"{row.source_dataset}→{row.target_dataset}\n{row.model}" for row in rows.itertuples()]
        ax.barh(np.arange(len(rows)), rows.rmse, color="#5b8db8")
        ax.set_yticks(np.arange(len(rows)), labels)
        ax.invert_yaxis()
    ax.set(title="Zero-shot cross-dataset retrieval transfer", xlabel="Target-probe RMSE")
    _finish(fig, path)


def plot_relative_absolute(evidence: pd.DataFrame, path: Path) -> None:
    probes = evidence[~evidence.retrieval_context.isin(["A", "B"])]
    summary = probes.groupby(["dataset", "retrieval_context"]).agg(
        R=("relative_evidence_R", "mean"), E=("absolute_evidence_E", "mean"),
        mean_log_rt=("log_rt", "mean"), residual_variance=("log_rt", "var"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    if not summary.empty:
        scatter = ax.scatter(summary.R, summary.E, c=summary.residual_variance, s=120, cmap="magma")
        offsets = {"Both": (8, 12), "Default": (8, -16), "Novel": (8, 2)}
        for row in summary.itertuples():
            ax.annotate(
                row.retrieval_context, (row.R, row.E),
                xytext=offsets.get(row.retrieval_context, (8, 2)), textcoords="offset points",
            )
        fig.colorbar(scatter, ax=ax, label="Observed log-RT variance")
    ax.set(xlabel="Relative evidence R", ylabel="Absolute evidence E", title="Relative choice versus absolute familiarity")
    _finish(fig, path)


def plot_distillation(gate_states: pd.DataFrame, distilled: pd.DataFrame, path: Path) -> None:
    rows = (
        gate_states[gate_states.model.eq("recurrent_gate_1d")].copy()
        if "model" in gate_states else pd.DataFrame()
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    if not rows.empty and not distilled.empty:
        rule = distilled.iloc[0]
        logit = rule.beta_R * rows.relative_evidence_R + rule.beta_E * rows.absolute_evidence_E + rule.lambda_history * rows.history_state + rule.intercept
        explicit_q = 1 / (1 + np.exp(-logit))
        ax.scatter(explicit_q, rows.q_A, s=10, alpha=.35)
        ax.plot([0, 1], [0, 1], "--", color="black")
    ax.set(xlabel="Distilled explicit q(A)", ylabel="Neural gate q(A)", title="Tiny-gate distillation")
    _finish(fig, path)


def make_retrieval_figures(
    output_dir: Path,
    *,
    features: pd.DataFrame,
    geometry: pd.DataFrame,
    predictions: pd.DataFrame,
    structural: pd.DataFrame,
    combined_to_novel: pd.DataFrame,
    novel_to_combined: pd.DataFrame,
    evidence: pd.DataFrame,
    gate_states: pd.DataFrame,
    distilled: pd.DataFrame,
) -> None:
    directory = Path(output_dir) / "figures_retrieval"
    directory.mkdir(parents=True, exist_ok=True)
    plot_conceptual(directory / "figure_1_conceptual_model.png")
    plot_geometry(features, geometry, directory / "figure_2_context_geometry.png")
    plot_human_readout(predictions, directory / "figure_3_human_readout_geometry.png")
    plot_structural(structural, directory / "figure_4_participant_structural_prediction.png")
    plot_continuous_categorical(predictions, directory / "figure_5_continuous_vs_categorical.png")
    plot_transfer(combined_to_novel, novel_to_combined, directory / "figure_6_cross_dataset_transfer.png")
    plot_relative_absolute(evidence, directory / "figure_7_relative_absolute_evidence.png")
    plot_distillation(gate_states, distilled, directory / "figure_8_tiny_gate_distillation.png")
