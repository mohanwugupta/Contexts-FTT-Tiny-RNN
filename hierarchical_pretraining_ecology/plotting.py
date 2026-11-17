"""Prespecified figures for the compositional pretraining experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ["P0_none", "P1_shuffled", "P2_structured"]
LABELS = {"P0_none": "No pretraining", "P1_shuffled": "Shuffled", "P2_structured": "Structured"}
COLORS = {"P0_none": "#6b6b6b", "P1_shuffled": "#d95f02", "P2_structured": "#1b9e77"}


def _save(fig, output: Path, name: str) -> None:
    fig.tight_layout(); fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def ecology_schematic(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, title, shuffled in zip(axes, ["P2 structured", "P1 structure-shuffled"], [False, True]):
        ax.axis("off"); ax.set_title(title)
        boxes = [(0, .75, "latent factors\n(c1, c2, c3)"), (.5, .75, "visible cue +\nlayout components"),
                 (.5, .2, "sparse reusable\ntask modifiers"), (0, .2, "shared family\nbackbone")]
        for x, y, text in boxes:
            ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1b9e77" if not shuffled else "#d95f02"))
        arrows = [((.12, .72), (.38, .72)), ((.12, .28), (.38, .28)),
                  ((.5, .62), (.5, .38))]
        for start, end in arrows:
            ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.7, linestyle="--" if shuffled and start[1] > .5 else "-"))
        ax.text(.5, .5, "pairing permuted" if shuffled else "shared cause aligns\nperception and task",
                transform=ax.transAxes, ha="center", color="#d95f02" if shuffled else "#1b9e77")
    _save(fig, output, "figure_01_ecology_schematic.png")


def internal_generalization(heldout, output):
    seed = heldout[(heldout.row_type.eq("seed_mean")) & heldout.phase.eq("confirmatory")]
    if seed.empty:
        seed = heldout[(heldout.row_type.eq("seed_mean")) & heldout.phase.eq("pilot")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, probe in zip(axes, ["heldout_factor_combination", "new_family_member"]):
        values = seed[seed.probe_type.eq(probe)].groupby("pretraining_condition").mean_prediction_NLL.mean()
        ax.bar([0, 1], [values.get("P1_shuffled", np.nan), values.get("P2_structured", np.nan)],
               color=[COLORS["P1_shuffled"], COLORS["P2_structured"]])
        ax.set_xticks([0, 1], ["Shuffled", "Structured"]); ax.set_title(probe.replace("_", " "))
    axes[0].set_ylabel("Held-out prediction NLL")
    _save(fig, output, "figure_02_internal_pretraining_generalization.png")


def human_phenotype(summary, hidden, classes, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    winners = classes[classes.row_type.eq("seed")]
    for index, condition in enumerate(ORDER):
        one = winners[winners.pretraining_condition.eq(condition)]
        axes[0].bar(index, one.model_label.isin(["Cross", "Split"]).mean(), color=COLORS[condition])
    axes[0].set_xticks(range(3), ["P0", "P1", "P2"]); axes[0].set_ylim(0, 1); axes[0].set_title("Cross/Split winner fraction")
    distance = hidden[hidden.row_type.eq("seed_mean")].groupby("pretraining_condition").aggregate_AB_separation.mean()
    axes[1].bar(range(3), [distance.get(c, np.nan) for c in ORDER], color=[COLORS[c] for c in ORDER])
    axes[1].set_xticks(range(3), ["P0", "P1", "P2"]); axes[1].set_title("A/B hidden differentiation")
    saw = summary.groupby("pretraining_condition").mean_context_yoked_RT_sawtooth.mean()
    axes[2].bar(range(3), [saw.get(c, np.nan) for c in ORDER], color=[COLORS[c] for c in ORDER])
    axes[2].set_xticks(range(3), ["P0", "P1", "P2"]); axes[2].set_title("Context-yoked RT sawtooth")
    _save(fig, output, "figure_03_human_AB_phenotype.png")


def first_mixed_profiles(probes, output):
    first = probes[probes.probe_epoch.eq("first")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, condition in zip(axes, ORDER):
        rows = []
        for seed, one in first[first.pretraining_condition.eq(condition)].groupby("seed"):
            pivot = one.pivot_table(index="transition_position", columns="context", values="normalized_RT")
            fast, slow = ("A", "B") if pivot.A.mean() <= pivot.B.mean() else ("B", "A")
            for position in pivot.index:
                rows.append((position, pivot.loc[position, fast], pivot.loc[position, slow], pivot.loc[position, "Mixed"]))
        array = np.asarray(rows, float)
        for index, label, color in ((1, "Fast", "#377eb8"), (2, "Slow", "#e41a1c"), (3, "Mixed", "#4daf4a")):
            means = [array[array[:, 0] == p, index].mean() for p in range(1, 8)]
            ax.plot(range(1, 8), means, marker="o", label=label, color=color)
        ax.set_title(LABELS[condition]); ax.set_xlabel("Transition")
    axes[0].set_ylabel("Normalized RT"); axes[0].legend()
    _save(fig, output, "figure_04_first_mixed_probe.png")


def mixed_transfer(metrics, interpolation, output):
    final = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late") & metrics.probe_context.eq("Mixed")]
    between = interpolation[(interpolation.row_type.eq("seed_mean")) & interpolation.probe_epoch.eq("late") & interpolation.probe_context.eq("Mixed")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    values = [between.groupby("pretraining_condition").between_endpoint_fraction.mean(),
              final.groupby("pretraining_condition").NLL_deficit_over_worse_endpoint.mean(),
              final.groupby("pretraining_condition").RT_deficit_over_worse_endpoint.mean()]
    for ax, data, title in zip(axes, values, ["Between-endpoint fraction", "NLL deficit", "RT deficit"]):
        ax.bar(range(3), [data.get(c, np.nan) for c in ORDER], color=[COLORS[c] for c in ORDER])
        ax.set_xticks(range(3), ["P0", "P1", "P2"]); ax.set_title(title)
    _save(fig, output, "figure_05_mixed_interpolation.png")


def novel_transfer(metrics, corridor, output):
    final = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late") & metrics.probe_context.eq("Novel")]
    cor = corridor[(corridor.probe_epoch.eq("late")) & corridor.probe_context.eq("Novel")].groupby("pretraining_condition").off_corridor_distance.mean()
    values = [final.groupby("pretraining_condition").NLL_deficit_over_worse_endpoint.mean(),
              final.groupby("pretraining_condition").RT_deficit_over_worse_endpoint.mean(), cor]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes, values, ["Novel NLL deficit", "Novel RT deficit", "Hidden distance to A/B corridor"]):
        ax.bar(range(3), [data.get(c, np.nan) for c in ORDER], color=[COLORS[c] for c in ORDER])
        ax.set_xticks(range(3), ["P0", "P1", "P2"]); ax.set_title(title)
    _save(fig, output, "figure_06_novel_transfer.png")


def frontier(frontier_data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mixed = frontier_data[frontier_data.probe_context.eq("Mixed")]
    for condition in ORDER:
        one = mixed[mixed.pretraining_condition.eq(condition)]
        axes[0].scatter(one.aggregate_AB_separation, one.between_endpoint_fraction,
                        color=COLORS[condition], label=LABELS[condition])
        axes[1].scatter(one.aggregate_AB_separation, one.NLL_deficit_over_worse_endpoint,
                        color=COLORS[condition], label=LABELS[condition])
    axes[0].set_ylabel("Mixed between fraction"); axes[1].set_ylabel("Mixed NLL deficit")
    for ax in axes: ax.set_xlabel("A/B hidden differentiation")
    axes[0].legend(fontsize=8)
    _save(fig, output, "figure_07_differentiation_transfer_frontier.png")


def corridor_plot(corridor, output):
    final = corridor[corridor.probe_epoch.eq("late")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for context, marker in (("Mixed", "o"), ("Novel", "s")):
        one = final[final.probe_context.eq(context)]
        lam = one.groupby("pretraining_condition").closest_lambda_AB_segment.mean()
        off = one.groupby("pretraining_condition").off_corridor_distance.mean()
        inside = one.groupby("pretraining_condition").inside_segment.mean()
        for ax, values in zip(axes, [lam, off, inside]):
            ax.plot(range(3), [values.get(c, np.nan) for c in ORDER], marker=marker, label=context)
    for ax, title in zip(axes, ["Closest lambda", "Off-corridor distance", "Inside-segment fraction"]):
        ax.set_xticks(range(3), ["P0", "P1", "P2"]); ax.set_title(title)
    axes[0].legend()
    _save(fig, output, "figure_08_hidden_representational_corridor.png")


def effect_summary(first, metrics, hidden, output):
    first_mean = first.groupby("pretraining_condition").first_probe_between_endpoint_fraction.mean()
    novel = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late") & metrics.probe_context.eq("Novel")].groupby("pretraining_condition").NLL_deficit_over_worse_endpoint.mean()
    differentiation = hidden[hidden.row_type.eq("seed_mean")].groupby("pretraining_condition").aggregate_AB_separation.mean()
    matrix = np.asarray([[differentiation.get(c, np.nan), first_mean.get(c, np.nan), novel.get(c, np.nan)] for c in ORDER])
    # Column-standardize for a compact, unit-agnostic overview while annotating raw values.
    scale = np.nanstd(matrix, axis=0); scale[scale == 0] = 1
    z = (matrix - np.nanmean(matrix, axis=0)) / scale
    z[:, 2] *= -1  # lower Novel deficit is the favorable direction
    fig, ax = plt.subplots(figsize=(7, 4)); image = ax.imshow(z, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_yticks(range(3), [LABELS[c] for c in ORDER]); ax.set_xticks(range(3), ["A/B differentiation", "First Mixed between", "Novel NLL deficit"])
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{matrix[i,j]:.3g}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="column-standardized favorability")
    _save(fig, output, "figure_09_pretraining_effect_summary.png")


def make_all_figures(output: Path, heldout, summary, hidden, classes, probes,
                     metrics, interpolation, corridor, frontier_data, first) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ecology_schematic(output); internal_generalization(heldout, output)
    human_phenotype(summary, hidden, classes, output); first_mixed_profiles(probes, output)
    mixed_transfer(metrics, interpolation, output); novel_transfer(metrics, corridor, output)
    frontier(frontier_data, output); corridor_plot(corridor, output)
    effect_summary(first, metrics, hidden, output)


def make_stopped_figures(output: Path, heldout, reason: str) -> None:
    """Preserve the figure contract while visibly marking gated analyses absent."""
    output.mkdir(parents=True, exist_ok=True)
    ecology_schematic(output); internal_generalization(heldout, output)
    names = [
        "figure_03_human_AB_phenotype.png", "figure_04_first_mixed_probe.png",
        "figure_05_mixed_interpolation.png", "figure_06_novel_transfer.png",
        "figure_07_differentiation_transfer_frontier.png",
        "figure_08_hidden_representational_corridor.png",
        "figure_09_pretraining_effect_summary.png",
    ]
    for name in names:
        fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
        ax.text(.5, .5, f"Not run: preregistered pilot gate stopped the pipeline.\n{reason}",
                ha="center", va="center", wrap=True)
        _save(fig, output, name)
