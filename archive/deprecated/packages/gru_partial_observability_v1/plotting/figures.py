"""Seven figures required by the deliberately minimal v1 PRD."""

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


def _schematic(path):
    fig, ax = plt.subplots(figsize=(10, 2.7))
    ax.axis("off")
    labels = ["Latent context\nA / B", "Noisy cue\nxₜ", "GRU + yₜ₋₁", "Predict ŷₜ", "Observe yₜ\nfor t+1"]
    xs = np.linspace(.08, .92, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, .5, label, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=.45", fc="#edf2f7", ec="#516b82"))
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", (right - .065, .5), (left + .065, .5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("V1 prediction timing and partial observability")
    _finish(fig, path)


def _episode(rows, path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not rows.empty:
        episode = rows[rows.episode_id.eq(rows.episode_id.iloc[0])]
        time = episode.time
        ax.plot(time, episode.skill_a, label="S_A", lw=2)
        ax.plot(time, episode.skill_b, label="S_B", lw=2)
        ax.scatter(time, episode.observed_y, c=episode.true_context, cmap="coolwarm", s=20, label="observed y")
        ax.plot(time, episode.predicted_y, color="black", lw=1.8, label="GRU prediction")
    ax.set(xlabel="Timestep", ylabel="Performance state", title="Example held-out episode")
    ax.legend(frameon=False, ncol=4)
    _finish(fig, path)


def _posterior(ambiguity, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if not ambiguity.empty:
        for condition, group in ambiguity.groupby("condition", sort=False):
            summary = group.groupby("q_star").agg(
                prediction=("predicted_y", "mean"), normative=("posterior_mean", "mean")
            ).reset_index()
            ax.plot(summary.q_star, summary.prediction, marker="o", color=COLORS.get(condition), label=condition)
        normative = ambiguity.groupby("q_star").posterior_mean.mean().reset_index()
        ax.plot(normative.q_star, normative.posterior_mean, "k--", lw=2, label="Bayesian posterior mean")
    ax.set(xlabel="Exact q* = P(A)", ylabel="Predicted performance", title="Behavior under held-out ambiguity")
    ax.legend(frameon=False)
    _finish(fig, path)


def _condition(comparison, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    if not comparison.empty:
        means = [
            comparison.posterior_mean_alignment_perfect.mean(),
            comparison.posterior_mean_alignment_uncertain.mean(),
        ]
        ax.bar(["Perfect", "Uncertain"], means, color=[COLORS["Perfect"], COLORS["Uncertain"]])
        for row in comparison.itertuples():
            ax.plot([0, 1], [row.posterior_mean_alignment_perfect, row.posterior_mean_alignment_uncertain],
                    color="#777777", alpha=.45)
    ax.axhline(0, color="black", lw=1)
    ax.set(ylabel="Posterior-mean R²", title="Identical ambiguous tests")
    _finish(fig, path)


def _risk(risk, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if not risk.empty:
        ax.plot(risk.q_star, risk.analytic_mean_risk, marker="o", label="posterior mean")
        ax.plot(risk.q_star, risk.analytic_sampling_risk, marker="o", label="posterior sample")
    ax.set(xlabel="q* = P(A)", ylabel="Expected MSE", title="Averaging versus coin-flipping")
    ax.legend(frameon=False)
    _finish(fig, path)


def _decoding(probes, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if not probes.empty:
        summary = probes.groupby(["condition", "target"]).heldout_r2.mean().unstack()
        summary.plot(kind="bar", ax=ax, color=["#59a14f", "#f28e2b", "#af7aa1"])
    ax.set(ylabel="Held-out episode R²", title="Hidden-state decoding")
    ax.tick_params(axis="x", rotation=0)
    _finish(fig, path)


def _intervention(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    performed = rows[rows.performed.eq(True)] if not rows.empty and "performed" in rows else pd.DataFrame()
    if not performed.empty:
        for condition, group in performed.groupby("condition", sort=False):
            summary = group.groupby("intervention_target_q").mean_implied_q.mean().reset_index()
            ax.plot(summary.intervention_target_q, summary.mean_implied_q, marker="o",
                    color=COLORS.get(condition), label=condition)
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "Not run: q* decoding below threshold", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Intervened decoded q", ylabel="Output-implied q", title="Exploratory belief intervention")
    _finish(fig, path)


def make_v1_figures(output_dir: str | Path, *, example_episode, ambiguity, comparison, risk, probes, interventions):
    directory = Path(output_dir) / "figures_v1"
    directory.mkdir(parents=True, exist_ok=True)
    _schematic(directory / "figure_1_task_schematic.png")
    _episode(example_episode, directory / "figure_2_example_episode.png")
    _posterior(ambiguity, directory / "figure_3_posterior_mean_behavior.png")
    _condition(comparison, directory / "figure_4_perfect_vs_uncertain.png")
    _risk(risk, directory / "figure_5_coin_flip_vs_averaging.png")
    _decoding(probes, directory / "figure_6_hidden_state_decoding.png")
    _intervention(interventions, directory / "figure_7_belief_intervention.png")
