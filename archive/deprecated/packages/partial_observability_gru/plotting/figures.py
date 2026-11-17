"""Eight PRD figures, robust to reduced smoke-test grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UNCERTAINTY_ORDER = ["perfect", "low", "moderate", "high"]
SEPARATION_ORDER = ["small", "medium", "large"]


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _conceptual(path):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    labels = ["Hidden context\nA / B", "Noisy observation\nxₜ", "Generic GRU", "One prediction\nŷₜ"]
    xs = np.linspace(0.1, 0.9, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, .5, label, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=.5", facecolor="#e8eef7", edgecolor="#496a8b"))
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", (right - .08, .5), (left + .08, .5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Partial observability creates a latent-state inference problem")
    _finish(fig, path)


def _normative(path):
    q = np.linspace(0, 1, 101)
    a, b = 0.0, 1.0
    mean = q * a + (1 - q) * b
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(q, mean, lw=3, label="posterior mean")
    ax.scatter(q[::5], np.where(q[::5] > .5, a, b), s=20, alpha=.6, label="one sampled endpoint")
    ax.set(xlabel="P(A)", ylabel="Prediction", title="Normative continuous readout")
    ax.legend(frameon=False)
    _finish(fig, path)


def _emergence(performance, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not performance.empty:
        summary = performance.groupby(["uncertainty", "objective"], sort=False).posterior_mean_r2.mean().reset_index()
        for objective, group in summary.groupby("objective", sort=False):
            order = {name: index for index, name in enumerate(UNCERTAINTY_ORDER)}
            group = group.assign(x=group.uncertainty.map(order)).sort_values("x")
            ax.plot(group.x, group.posterior_mean_r2, marker="o", label=objective)
        ticks = sorted(summary.uncertainty.map({name: i for i, name in enumerate(UNCERTAINTY_ORDER)}).dropna().unique())
        ax.set_xticks(ticks, [UNCERTAINTY_ORDER[int(x)] for x in ticks])
    ax.set(xlabel="Training uncertainty", ylabel="Posterior-mean R²", title="Emergence across uncertainty")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, path)


def _factorial(performance, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    rows = performance[performance.objective.eq("deterministic")] if not performance.empty else performance
    if not rows.empty:
        table = rows.pivot_table(index="uncertainty", columns="skill_separation", values="posterior_mean_r2")
        table = table.reindex(index=[x for x in UNCERTAINTY_ORDER if x in table.index],
                              columns=[x for x in SEPARATION_ORDER if x in table.columns])
        image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis", vmin=-.2, vmax=1)
        ax.set_xticks(range(len(table.columns)), table.columns)
        ax.set_yticks(range(len(table.index)), table.index)
        fig.colorbar(image, ax=ax, label="Posterior-mean R²")
    ax.set(xlabel="Skill separation", ylabel="Context uncertainty", title="Uncertainty × skill separation")
    _finish(fig, path)


def _hidden(probes, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not probes.empty:
        group_columns = ["hidden_dim", "target"]
        if "network_state" in probes:
            group_columns.append("network_state")
        summary = probes.groupby(group_columns, sort=False).r2.mean().reset_index()
        plot_groups = ["target"] + (["network_state"] if "network_state" in probes else [])
        for keys, group in summary.groupby(plot_groups, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            ax.plot(group.hidden_dim, group.r2, marker="o", label=" / ".join(map(str, keys)))
    ax.set(xlabel="GRU hidden dimension", ylabel="Held-out decoding R²", title="Latent information in hidden state")
    ax.legend(frameon=False)
    _finish(fig, path)


def _belief(interventions, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if not interventions.empty:
        summary = interventions.groupby("intervention_target_q").mean_implied_q.mean().reset_index()
        ax.plot(summary.intervention_target_q, summary.mean_implied_q, marker="o", lw=2)
    ax.set(xlabel="Intervened decoded P(A)", ylabel="Output-implied P(A)", title="Belief-state causal intervention")
    _finish(fig, path)


def _sampling(comparison, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if not comparison.empty:
        means = comparison[["normative_mean_risk", "normative_sampling_risk"]].mean()
        ax.bar(["posterior mean", "posterior sample"], means.values, color=["#4c78a8", "#e45756"])
    ax.set(ylabel="Expected squared error", title="Continuous versus sampling risk")
    _finish(fig, path)


def _human(probes, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    selected = probes[probes.probe.isin(["Both", "Novel", "Default"])] if not probes.empty else probes
    if not selected.empty:
        summary = selected.groupby(["probe", "objective"], sort=False).predicted_y.mean().unstack()
        summary.plot(kind="bar", ax=ax)
    ax.set(ylabel="Zero-shot predicted state", title="Human-style OOD contexts")
    ax.tick_params(axis="x", rotation=0)
    _finish(fig, path)


def make_figures(output_dir: str | Path, *, performance, hidden_probes, belief_interventions,
                 continuous_sampling, human_probes):
    directory = Path(output_dir) / "figures_partial_observability"
    directory.mkdir(parents=True, exist_ok=True)
    _conceptual(directory / "figure_1_computational_problem.png")
    _normative(directory / "figure_2_normative_solution.png")
    _emergence(performance, directory / "figure_3_emergence_uncertainty.png")
    _factorial(performance, directory / "figure_4_uncertainty_skill_separation.png")
    _hidden(hidden_probes, directory / "figure_5_hidden_representation.png")
    _belief(belief_interventions, directory / "figure_6_belief_causality.png")
    _sampling(continuous_sampling, directory / "figure_7_continuous_sampling.png")
    _human(human_probes, directory / "figure_8_human_style_probes.png")
