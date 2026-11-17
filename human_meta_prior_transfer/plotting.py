"""Registered figures for the H128 causal human-transfer experiment."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"G1": "#0072B2", "G2": "#D55E00", "G3": "#009E73", "G4": "#777777", "G5": "#CC79A7"}


def _save(fig, output, name):
    fig.tight_layout(); fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def make_figures(output, training, canonical, mixed, novel, geometry, separation, state_space, retention):
    fig, ax = plt.subplots(figsize=(12, 3)); ax.axis("off")
    ax.text(.5, .72, "Locked synthetic sources: structured meta | shuffled meta | structured ordinary | random", ha="center", fontsize=11)
    ax.annotate("", xy=(.5, .43), xytext=(.5, .65), arrowprops={"arrowstyle": "->"})
    ax.text(.5, .36, "Identical A10/B10 target training → frozen first-exposure probes", ha="center", fontsize=12, weight="bold")
    ax.text(.5, .12, "Familiar gate → Mixed causal contrasts → mechanistic diagnostics", ha="center", fontsize=11)
    _save(fig, output, "figure_01_experimental_logic.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for group, one in training.groupby("group"):
        line = one.groupby("training_sequence")
        axes[0].plot(line.mean_prediction_NLL.mean(), color=COLORS[group], label=group)
        axes[1].plot(line.mean_normalized_RT.mean(), color=COLORS[group], label=group)
    final_sep = separation[separation.row_type.eq("seed_mean")].groupby(["group", "checkpoint"], as_index=False).AB_hidden_distance.mean()
    for group, one in final_sep.groupby("group"): axes[2].plot(one.checkpoint, one.AB_hidden_distance, marker="o", color=COLORS[group], label=group)
    axes[0].set_ylabel("training NLL"); axes[1].set_ylabel("normalized RT"); axes[2].set_ylabel("A/B hidden distance")
    for ax in axes: ax.set_xlabel("training sequence / checkpoint"); ax.legend(fontsize=7)
    _save(fig, output, "figure_02_familiar_AB_learning.png")

    classes = state_space[(state_space.record_type.eq("classification")) & state_space.row_type.eq("classification_fraction")]
    fig, ax = plt.subplots(figsize=(8, 4)); pivot = classes.pivot(index="condition", columns="model_label", values="classification_fraction").fillna(0)
    pivot.reindex(columns=["Lump", "Cross", "Split"], fill_value=0).plot.bar(stacked=True, ax=ax); ax.set_ylabel("seed fraction"); ax.set_xlabel("group")
    _save(fig, output, "figure_03_state_space_phenotype.png")

    settled = canonical.checkpoint.max(); profiles = canonical[(canonical.checkpoint.eq(settled)) & canonical.canonical_context.isin(["Fast", "Mixed", "Slow"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, group in zip(axes, ("G1", "G2", "G3", "G4")):
        for context, style in (("Fast", "-"), ("Mixed", "--"), ("Slow", ":")):
            line = profiles[(profiles.group.eq(group)) & profiles.canonical_context.eq(context)].groupby("transition_position").normalized_RT.mean()
            ax.plot(line.index, line, ls=style, marker="o", label=context)
        ax.set_title(group); ax.set_xlabel("transition"); ax.legend(fontsize=7)
    axes[0].set_ylabel("normalized RT"); _save(fig, output, "figure_04_first_mixed_profile.png")

    means = mixed[mixed.row_type.eq("seed_mean")].groupby(["group", "checkpoint"], as_index=False).between_endpoint_fraction.mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    for group, one in means.groupby("group"): ax.plot(one.checkpoint, one.between_endpoint_fraction, marker="o", color=COLORS[group], label=group)
    ax.set_xlabel("checkpoint"); ax.set_ylabel("Mixed between-endpoint fraction"); ax.legend(ncol=3, fontsize=8)
    _save(fig, output, "figure_05_mixed_between_endpoint_fraction.png")

    values = mixed[mixed.row_type.eq("seed_mean")].groupby("group")[["prediction_NLL_deficit", "RT_deficit"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); values.prediction_NLL_deficit.plot.bar(ax=axes[0], color=[COLORS[x] for x in values.index]); values.RT_deficit.plot.bar(ax=axes[1], color=[COLORS[x] for x in values.index])
    axes[0].set_ylabel("Mixed NLL deficit"); axes[1].set_ylabel("Mixed RT deficit")
    _save(fig, output, "figure_06_mixed_deficits.png")

    novel_profile = canonical[(canonical.checkpoint.eq(settled)) & canonical.canonical_context.isin(["Fast", "Novel", "Slow", "Neutral"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, group in zip(axes, ("G1", "G2", "G3", "G4")):
        for context in ("Fast", "Novel", "Slow", "Neutral"):
            line = novel_profile[(novel_profile.group.eq(group)) & novel_profile.canonical_context.eq(context)].groupby("transition_position").normalized_RT.mean()
            ax.plot(line.index, line, marker="o", label=context)
        ax.set_title(group); ax.set_xlabel("transition"); ax.legend(fontsize=7)
    axes[0].set_ylabel("normalized RT"); _save(fig, output, "figure_07_novel_profile.png")

    geo = geometry[geometry.row_type.eq("seed_mean")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (group, context), one in geo.groupby(["group", "probe_context"]):
        ax.scatter(one.projection_coordinate, one.off_corridor_distance, color=COLORS[group], marker="o" if context == "Mixed" else "x", alpha=.65, label=f"{group} {context}")
    ax.set_xlabel("A/B projection coordinate"); ax.set_ylabel("off-corridor distance"); ax.legend(fontsize=6, ncol=2)
    _save(fig, output, "figure_08_hidden_corridor_geometry.png")

    frontier = separation[separation.row_type.eq("seed_mean")].merge(mixed[mixed.row_type.eq("seed_mean")][["group", "seed", "checkpoint", "between_endpoint_fraction"]], on=["group", "seed", "checkpoint"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for group, one in frontier.groupby("group"): ax.scatter(one.AB_hidden_distance, one.between_endpoint_fraction, color=COLORS[group], label=group, alpha=.7)
    ax.set_xlabel("A/B hidden separation"); ax.set_ylabel("Mixed transfer quality"); ax.legend()
    _save(fig, output, "figure_09_differentiation_transfer_frontier.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(retention):
        for group, one in retention.groupby("group"):
            line = one.groupby("target_checkpoint").unseen_query_NLL.mean(); ax.plot(line.index, line, marker="o", color=COLORS[group], label=group)
        ax.legend()
    else: ax.text(.5, .5, "Retention diagnostic unavailable", transform=ax.transAxes, ha="center")
    ax.set_xlabel("target-training checkpoint"); ax.set_ylabel("synthetic unseen-query NLL")
    _save(fig, output, "figure_10_synthetic_prior_retention.png")
