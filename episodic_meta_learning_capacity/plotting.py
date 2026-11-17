"""Figures registered by the episodic meta-learning PRD."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, output: Path, name):
    fig.tight_layout(); fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def synthetic_figures(output, scores, v2, v3, v4, gates):
    fig, ax = plt.subplots(figsize=(11, 3)); ax.axis("off")
    ax.text(.5, .5, "new family  →  3 support contexts × 2 sequences  →  3 first-order inner updates\n"
            "↘ familiar copy query + 4 unseen queries  →  prediction query loss  →  outer Adam update",
            ha="center", va="center", fontsize=12, bbox={"boxstyle": "round", "fc": "#eef5ff"})
    _save(fig, output, "figure_01_episode_schematic.png")
    means = scores.groupby(["training_condition", "hidden_size"], as_index=False).unseen_query_NLL.mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, one in means.groupby("training_condition"): ax.plot(one.hidden_size, one.unseen_query_NLL, marker="o", label=condition)
    ax.set_xscale("log", base=2); ax.set_xticks([8, 32, 64, 128], [8, 32, 64, 128]); ax.set_xlabel("hidden size"); ax.set_ylabel("unseen-family query NLL"); ax.legend()
    _save(fig, output, "figure_02_condition_by_capacity_query_nll.png")
    pivot = means.pivot(index="hidden_size", columns="training_condition", values="unseen_query_NLL")
    fig, ax = plt.subplots(figsize=(7, 5)); advantage = pivot.P2_ordinary - pivot.P2_meta
    ax.plot(advantage.index, advantage, marker="o"); ax.axhline(0, color="black", lw=1); ax.set_xscale("log", base=2); ax.set_xticks([8, 32, 64, 128], [8, 32, 64, 128]); ax.set_ylabel("ordinary NLL − meta NLL"); ax.set_xlabel("hidden size")
    _save(fig, output, "figure_03_capacity_threshold.png")
    fig, axes = plt.subplots(1, len(sorted(v2.hidden_size.unique())), figsize=(5 * len(v2.hidden_size.unique()), 4), sharey=True)
    for ax, width in zip(np.atleast_1d(axes), sorted(v2.hidden_size.unique())):
        for condition, one in v2[v2.hidden_size.eq(width)].groupby("training_condition"):
            line = one.groupby("context_distance").prediction_deficit.mean(); ax.plot(line.index, line, marker="o", label=condition)
        ax.set_title(f"H={width}"); ax.set_xlabel("context distance"); ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("prediction deficit"); _save(fig, output, "figure_04_similarity_gradients.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, one in v3.groupby("training_condition"):
        line = one.groupby("query_ratio").query_prediction_NLL.mean(); ax.plot(line.index, line, marker="o", label=condition)
    ax.set_xlabel("query location within support interval"); ax.set_ylabel("query NLL"); ax.legend()
    _save(fig, output, "figure_05_asymmetric_queries.png")
    joined = scores.groupby(["training_condition", "hidden_size"], as_index=False).agg(copy_accuracy=("copy_accuracy", "mean"), unseen_query_NLL=("unseen_query_NLL", "mean"))
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, one in joined.groupby("training_condition"): ax.scatter(one.copy_accuracy, one.unseen_query_NLL, label=condition); [ax.text(r.copy_accuracy, r.unseen_query_NLL, f"H{int(r.hidden_size)}", fontsize=7) for _, r in one.iterrows()]
    ax.set_xlabel("copy accuracy"); ax.set_ylabel("unseen query NLL"); ax.legend()
    _save(fig, output, "figure_06_copy_vs_generalization.png")


def human_figures(output, training, metrics, interpolation, hidden):
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, one in training.groupby("model_variant"):
        line = one.groupby("training_sequence").mean_normalized_RT.mean(); ax.plot(line.index, line, label=variant)
    ax.set_xlabel("training sequence"); ax.set_ylabel("normalized RT"); ax.legend(fontsize=7)
    _save(fig, output, "figure_07_human_AB_phenotype.png")
    seed_mean = interpolation[interpolation.row_type.eq("seed_mean")]
    for context, number, name in (("Mixed", 8, "mixed"), ("Novel", 9, "novel")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, one in seed_mean[seed_mean.probe_context.eq(context)].groupby("model_variant"):
            line = one.groupby("checkpoint").between_endpoint_fraction.mean(); ax.plot(line.index, line, marker="o", label=variant)
        ax.set_xticks([20, 50, 90, 170]); ax.set_xlabel("human training checkpoint"); ax.set_ylabel("between-endpoint fraction"); ax.legend(fontsize=7)
        _save(fig, output, f"figure_{number:02d}_{name}_across_checkpoints.png")
    final = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_context.eq("Mixed")]
    sep = hidden[hidden.row_type.eq("seed_mean")][["model_variant", "seed", "checkpoint", "aggregate_AB_separation"]]
    merged = final.merge(sep, on=["model_variant", "seed", "checkpoint"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, one in merged.groupby("model_variant"): ax.scatter(one.aggregate_AB_separation, -one.NLL_deficit_over_worse_endpoint, label=variant, alpha=.7)
    ax.set_xlabel("A/B hidden separation"); ax.set_ylabel("Mixed transfer quality"); ax.legend(fontsize=7)
    _save(fig, output, "figure_10_differentiation_transfer_frontier.png")
