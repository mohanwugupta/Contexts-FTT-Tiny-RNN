"""Required ecology-comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"P0": "#777777", "P2": "#2b8cbe", "P3": "#7b3294", "P4": "#e66101", "P5": "#1b7837"}


def _save(fig, output: Path, name: str):
    fig.tight_layout(); fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def ecology_schematic(output: Path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    z = np.linspace(-1, 1, 41)
    specs = [
        ("P0 — none", np.full_like(z, np.nan)),
        ("P1 — shuffled", np.random.default_rng(1).permutation(z)),
        ("P2 — smooth", z),
        ("P3 — smooth + boundary", np.where(z < 0, .55 * z, .55 * z + .8)),
        ("P4 — compositional", np.where(z < 0, -.5, .5)),
        ("P5 — combined", .55 * z + np.where(z > .35, .45, 0)),
    ]
    for ax, (title, task) in zip(axes.flat, specs):
        if np.isfinite(task).any(): ax.plot(z, task, color="#333333", lw=2)
        else: ax.text(0, 0, "random initialization", ha="center", va="center")
        ax.set_title(title); ax.set_xlabel("latent context"); ax.set_ylabel("task structure")
    _save(fig, output, "figure_01_pretraining_ecology_schematic.png")


def internal_figures(output: Path, v1, v2, v3, v4, scores):
    ecology_schematic(output)
    figures = []
    fig, ax = plt.subplots(figsize=(9, 5))
    v1.groupby("pretraining_condition").prediction_deficit_over_worse_endpoint.mean().sort_index().plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=1); ax.set_ylabel("V1 prediction deficit"); ax.set_title("Unseen-family intermediate transfer")
    figures.append((fig, "figure_02_unseen_family_intermediate_transfer.png"))
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition, one in v2.groupby("pretraining_condition"):
        line = one.groupby("requested_z").transfer_quality.mean()
        ax.plot(line.index, line, marker="o", label=condition)
    ax.set_xlabel("latent context z"); ax.set_ylabel("transfer quality"); ax.set_title("Similarity-gradient test"); ax.legend(fontsize=7, ncol=3)
    figures.append((fig, "figure_03_similarity_gradient.png"))
    fig, ax = plt.subplots(figsize=(9, 5))
    v3.groupby("pretraining_condition").boundary_selectivity.mean().sort_index().plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=1); ax.set_ylabel("cross NLL − within NLL"); ax.set_title("Boundary selectivity")
    figures.append((fig, "figure_04_boundary_selectivity.png"))
    fig, ax = plt.subplots(figsize=(9, 5))
    v4.groupby("pretraining_condition").modified_position_accuracy.mean().sort_index().plot.bar(ax=ax)
    ax.set_ylim(0, 1); ax.set_ylabel("held-out modified-position accuracy"); ax.set_title("Compositional generalization")
    figures.append((fig, "figure_05_compositional_generalization.png"))
    fig, ax = plt.subplots(figsize=(7, 6))
    means = scores.groupby("pretraining_condition", as_index=False)[["S_smooth", "S_boundary", "S_comp"]].mean()
    for _, row in means.iterrows():
        size = 80 + 500 * max(0., float(row.S_comp)); name = row.pretraining_condition
        ax.scatter(row.S_smooth, row.S_boundary, s=size, alpha=.7, color=COLORS.get(name, "#999999"))
        ax.text(row.S_smooth, row.S_boundary, name, fontsize=8)
    ax.axhline(0, color="#bbbbbb", lw=1); ax.set_xlabel("smooth-transfer score"); ax.set_ylabel("boundary-selectivity score")
    ax.set_title("Pretraining selection frontier (size = composition)")
    figures.append((fig, "figure_06_pretraining_selection_frontier.png"))
    for fig, name in figures: _save(fig, output, name)


def human_figures(output: Path, training: pd.DataFrame, probes: pd.DataFrame, frontier: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition, one in training.groupby("pretraining_condition"):
        line = one.groupby("training_sequence").mean_normalized_RT.mean()
        ax.plot(line.index, line, alpha=.8, label=condition)
    ax.set_xlabel("human-task training sequence"); ax.set_ylabel("normalized RT"); ax.set_title("Human A/B familiar phenotype"); ax.legend(fontsize=8)
    _save(fig, output, "figure_07_human_familiar_phenotype.png")
    first = probes[probes.probe_epoch.eq("first") & probes.context.isin(["A", "B", "Mixed"])]
    fig, axes = plt.subplots(1, max(1, first.pretraining_condition.nunique()), figsize=(5 * max(1, first.pretraining_condition.nunique()), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (condition, one) in zip(axes, first.groupby("pretraining_condition")):
        for context, values in one.groupby("context"):
            line = values.groupby("transition_position").normalized_RT.mean(); ax.plot(line.index, line, marker="o", label=context)
        ax.set_title(condition); ax.set_xlabel("transition"); ax.legend(fontsize=8)
    axes[0].set_ylabel("normalized RT"); _save(fig, output, "figure_08_first_mixed_probe.png")
    late = probes[probes.probe_epoch.eq("late") & probes.context.isin(["A", "B", "Novel", "Neutral"])]
    fig, axes = plt.subplots(1, max(1, late.pretraining_condition.nunique()), figsize=(5 * max(1, late.pretraining_condition.nunique()), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (condition, one) in zip(axes, late.groupby("pretraining_condition")):
        for context, values in one.groupby("context"):
            line = values.groupby("transition_position").normalized_RT.mean(); ax.plot(line.index, line, marker="o", label=context)
        ax.set_title(condition); ax.set_xlabel("transition"); ax.legend(fontsize=8)
    axes[0].set_ylabel("normalized RT"); _save(fig, output, "figure_09_novel_probe.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    for condition, one in frontier.groupby("pretraining_condition"):
        quality = -one.NLL_deficit_over_worse_endpoint
        ax.scatter(one.aggregate_AB_separation, quality, alpha=.65, label=condition)
    ax.set_xlabel("A/B hidden separation"); ax.set_ylabel("Mixed transfer quality (−NLL deficit)")
    ax.set_title("Differentiation–transfer frontier"); ax.legend(fontsize=8)
    _save(fig, output, "figure_10_differentiation_transfer_frontier.png")
