"""Nine figures for the SRN v1 proof-of-concept analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"Perfect": "#4c78a8", "Uncertain": "#e45756"}


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _not_run(ax, frame, title):
    reason = "staged analysis not run"
    if not frame.empty and "reason" in frame and pd.notna(frame.reason.iloc[0]):
        reason = str(frame.reason.iloc[0])
    ax.text(0.5, 0.5, f"Not run\n{reason}", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()


def _schematic(path):
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.axis("off")
    labels = [
        "Hidden context\nA / B", "Noisy cue\nxₜ", "Elman SRN\n+ yₜ₋₁",
        "Predict ŷₜ", "Observe yₜ\nfor t+1",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=.45", fc="#edf2f7", ec="#516b82"))
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", (right - 0.065, 0.5), (left + 0.065, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Partial-observability prediction task")
    _finish(fig, path)


def _episode(rows, path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not rows.empty:
        episode = rows[rows.episode_id.eq(rows.episode_id.iloc[0])]
        ax.plot(episode.time, episode.skill_a, label="S_A", lw=2)
        ax.plot(episode.time, episode.skill_b, label="S_B", lw=2)
        ax.scatter(episode.time, episode.observed_y, c=episode.true_context,
                   cmap="coolwarm", s=22, label="observed y")
        ax.plot(episode.time, episode.predicted_y, color="black", lw=1.8, label="SRN prediction")
    ax.set(xlabel="Timestep", ylabel="Performance state", title="Example held-out episode")
    ax.legend(frameon=False, ncol=4)
    _finish(fig, path)


def _averaging(ambiguity, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = ambiguity
    if not ambiguity.empty and "matched_state_history_id" in ambiguity:
        separations = (
            ambiguity.assign(separation=np.abs(ambiguity.skill_a - ambiguity.skill_b))
            .groupby("matched_state_history_id").separation.mean()
        )
        plotted = ambiguity[
            ambiguity.matched_state_history_id.eq(separations.idxmax())
        ]
    for condition, group in plotted.groupby("condition", sort=False):
        summary = group.groupby("q_star").agg(
            prediction=("predicted_y", "mean"), normative=("posterior_mean", "mean")
        ).reset_index()
        ax.plot(summary.q_star, summary.prediction, marker="o",
                color=COLORS.get(condition), label=condition)
    if not plotted.empty:
        normative = plotted.groupby("q_star").posterior_mean.mean().reset_index()
        ax.plot(normative.q_star, normative.posterior_mean, "k--", lw=2, label="Bayesian mean")
    ax.set(xlabel="Exact q* = P(A)", ylabel="Predicted performance",
           title="Emergent averaging on identical ambiguous states")
    ax.legend(frameon=False)
    _finish(fig, path)


def _condition(comparison, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    if not comparison.empty:
        means = [comparison.posterior_mean_alignment_perfect.mean(),
                 comparison.posterior_mean_alignment_uncertain.mean()]
        ax.bar(["Perfect", "Uncertain"], means,
               color=[COLORS["Perfect"], COLORS["Uncertain"]])
        for row in comparison.itertuples():
            ax.plot([0, 1], [row.posterior_mean_alignment_perfect,
                             row.posterior_mean_alignment_uncertain],
                    color="#777777", alpha=0.45)
    ax.axhline(0, color="black", lw=1)
    ax.set(ylabel="Posterior-mean R²", title="Perfect versus Uncertain training")
    _finish(fig, path)


def _responsibility(ambiguity, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, group in ambiguity.groupby("condition", sort=False):
        ax.scatter(group.q_star, group.q_implied_unclipped, s=18, alpha=0.28,
                   color=COLORS.get(condition), label=condition)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="identity")
    ax.set(xlabel="Exact q*", ylabel="Unclipped implied q",
           title="Output-implied responsibility")
    ax.legend(frameon=False)
    _finish(fig, path)


def _risk(risk, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(risk.q_star, risk.analytic_mean_risk, marker="o", label="posterior mean")
    ax.plot(risk.q_star, risk.analytic_sampling_risk, marker="o", label="posterior sample")
    ax.scatter(risk.q_star, risk.numerical_mean_risk, marker="x", color="#4c78a8")
    ax.scatter(risk.q_star, risk.numerical_sampling_risk, marker="x", color="#f28e2b")
    ax.set(xlabel="q* = P(A)", ylabel="Expected MSE", title="Mean versus posterior sampling")
    ax.legend(frameon=False)
    _finish(fig, path)


def _probes(probes, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    completed = probes[probes.status.eq("completed")] if "status" in probes else probes
    if completed.empty:
        _not_run(ax, probes, "Hidden-state representation")
    else:
        completed.groupby(["condition", "target"]).heldout_r2.mean().unstack().plot(
            kind="bar", ax=ax, color=["#59a14f", "#f28e2b", "#af7aa1"]
        )
        ax.set(ylabel="Held-out episode R²", title="Hidden-state representation")
        ax.tick_params(axis="x", rotation=0)
    _finish(fig, path)


def _dimensionality(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    completed = rows[rows.status.eq("completed")] if "status" in rows else rows
    if completed.empty:
        _not_run(axes[0], rows, "Dimensionality: prediction")
        _not_run(axes[1], rows, "Dimensionality: averaging")
    else:
        for condition, group in completed.groupby("condition", sort=False):
            summary = group.groupby("hidden_dim").agg(
                mse=("model_mse", "mean"), alignment=("posterior_mean_r2", "mean")
            ).reset_index()
            axes[0].plot(summary.hidden_dim, summary.mse, marker="o",
                         color=COLORS.get(condition), label=condition)
            axes[1].plot(summary.hidden_dim, summary.alignment, marker="o",
                         color=COLORS.get(condition), label=condition)
        axes[0].set(xlabel="Hidden units", ylabel="MSE", title="Prediction")
        axes[1].set(xlabel="Hidden units", ylabel="Posterior-mean R²", title="Averaging")
        axes[1].legend(frameon=False)
    _finish(fig, path)


def _control(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    completed = rows[rows.status.eq("completed")] if "status" in rows else rows
    if completed.empty:
        _not_run(ax, rows, "Optional GRU robustness control")
    else:
        summary = completed.groupby(["architecture", "condition"]).posterior_mean_r2.mean().unstack()
        summary.plot(kind="bar", ax=ax, color=[COLORS["Perfect"], COLORS["Uncertain"]])
        ax.set(ylabel="Posterior-mean R²", title="Matched SRN versus GRU")
        ax.tick_params(axis="x", rotation=0)
    _finish(fig, path)


def make_figures(
    output_dir: str | Path,
    *,
    example_episode,
    ambiguity,
    comparison,
    risk,
    probes,
    dimensionality,
    gru_control,
):
    directory = Path(output_dir) / "figures_v1"
    directory.mkdir(parents=True, exist_ok=True)
    _schematic(directory / "figure_1_computational_task.png")
    _episode(example_episode, directory / "figure_2_example_episode.png")
    _averaging(ambiguity, directory / "figure_3_emergent_averaging.png")
    _condition(comparison, directory / "figure_4_perfect_vs_uncertain.png")
    _responsibility(ambiguity, directory / "figure_5_implied_responsibility.png")
    _risk(risk, directory / "figure_6_mean_vs_sampling.png")
    _probes(probes, directory / "figure_7_hidden_representation.png")
    _dimensionality(dimensionality, directory / "figure_8_dimensionality.png")
    _control(gru_control, directory / "figure_9_gru_control.png")
