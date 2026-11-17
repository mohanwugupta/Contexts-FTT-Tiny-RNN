"""Sequential curriculum and conditional human-transfer figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = ["E0", "E1", "E2", "E3"]
STAGE_COLORS = ["#66c2a5", "#3288bd", "#fdae61", "#d53e4f"]
HUMAN = ["H0_random", "H1_shuffled", "H2_structured"]
HUMAN_COLORS = ["#666666", "#d95f02", "#1b9e77"]


def _save(fig, output: Path, name: str):
    fig.tight_layout(); fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def curriculum_schematic(output):
    fig, ax = plt.subplots(figsize=(12, 3)); ax.axis("off")
    subtitles = ["stable direct\nsemantics", "variable surface\nstable subspaces",
                 "permuted/sign-flipped\nsubspaces", "dense surface +\ntask-role rebinding"]
    for index, (stage, subtitle) in enumerate(zip(STAGES, subtitles)):
        ax.text(index, .5, f"{stage}\n{subtitle}", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=.5", facecolor="white", edgecolor=STAGE_COLORS[index], linewidth=2))
        if index < 3:
            ax.annotate("PASS", xy=(index + .72, .5), xytext=(index + .28, .5),
                        arrowprops=dict(arrowstyle="->", lw=1.7), ha="center", va="bottom")
    ax.set_xlim(-.5, 3.5); ax.set_ylim(0, 1); ax.set_title("Sequentially gated structural-prior curriculum")
    _save(fig, output, "figure_01_curriculum_schematic.png")


def internal_generalization(heldout, output):
    seed = heldout[heldout.row_type.eq("seed_mean")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("heldout_prediction_NLL", "Held-out NLL"),
               ("heldout_overall_accuracy", "Overall accuracy"),
               ("heldout_modified_position_accuracy", "Modified-position accuracy")]
    for ax, (metric, title) in zip(axes, metrics):
        for condition, marker in (("Shuffled", "s--"), ("Structured", "o-")):
            means = seed[seed.pretraining_condition.eq(condition)].groupby("stage")[metric].mean()
            ax.plot(range(4), [means.get(stage, np.nan) for stage in STAGES], marker,
                    label=condition)
        ax.set_xticks(range(4), STAGES); ax.set_title(title)
    axes[0].legend(); _save(fig, output, "figure_02_internal_compositional_generalization.png")


def gate_summary(gates, output):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    values, labels = [], []
    for stage in STAGES:
        one = gates[gates.stage.eq(stage)]
        if one.empty:
            values.append(np.nan); labels.append("NOT ATTEMPTED")
        else:
            row = one[one.phase.eq("confirmatory")].iloc[0] if one.phase.eq("confirmatory").any() else one.iloc[-1]
            values.append(float(row.gate_passed))
            labels.append(f"{str(row.phase).upper()}\n" + ("PASS" if row.gate_passed else "FAIL"))
    masked = np.ma.masked_invalid(values)
    ax.imshow(masked.reshape(1, -1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), STAGES); ax.set_yticks([0], ["last reached gate"])
    for index, label in enumerate(labels):
        ax.text(index, 0, label, ha="center", va="center", fontsize=9)
    _save(fig, output, "figure_03_stage_gate_summary.png")


def decomposition_plot(decomposition, output):
    seed = decomposition[decomposition.row_type.eq("seed_mean")]
    labels = ["factor_1_P2_P3", "factor_2_P5_P6", "shared_P1_P4_P7"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, group in zip(axes, labels):
        for condition, marker in (("Shuffled", "s--"), ("Structured", "o-")):
            values = seed[(seed.position_group.eq(group)) & seed.pretraining_condition.eq(condition)].groupby("stage").accuracy.mean()
            ax.plot(range(4), [values.get(stage, np.nan) for stage in STAGES], marker, label=condition)
        ax.set_xticks(range(4), STAGES); ax.set_title(group.replace("_", " "))
    axes[0].set_ylabel("Held-out accuracy"); axes[0].legend()
    _save(fig, output, "figure_04_modified_position_decomposition.png")


def abstraction_frontier(gates, output):
    confirm = gates[gates.phase.eq("confirmatory")].set_index("stage")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in ((axes[0], "mean_structured_minus_shuffled_NLL", "NLL advantage (lower is better)"),
                              (axes[1], "mean_structured_minus_shuffled_modified_accuracy", "Modified-accuracy advantage")):
        values = [confirm.loc[stage, metric] if stage in confirm.index else np.nan for stage in STAGES]
        ax.plot(range(4), values, marker="o"); ax.axhline(0, color=".4", lw=1)
        ax.set_xticks(range(4), STAGES); ax.set_title(title)
    _save(fig, output, "figure_05_abstraction_frontier.png")


def human_figures(output, summary, classes, hidden, probes, metrics, interpolation, corridor, frontier):
    # Figure 6: familiar phenotype.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    winners = classes[classes.row_type.eq("seed")].copy()
    winners["is_cross_split"] = winners.model_label.isin(["Cross", "Split"])
    cross = winners.groupby("pretraining_condition").is_cross_split.mean()
    distance = hidden[hidden.row_type.eq("seed_mean")].groupby("pretraining_condition").aggregate_AB_separation.mean()
    saw = summary.groupby("pretraining_condition").mean_context_yoked_RT_sawtooth.mean()
    for ax, values, title in zip(axes, [cross, distance, saw], ["Cross/Split fraction", "A/B hidden differentiation", "RT sawtooth"]):
        ax.bar(range(3), [values.get(name, np.nan) for name in HUMAN], color=HUMAN_COLORS)
        ax.set_xticks(range(3), ["H0", "H1", "H2"]); ax.set_title(title)
    _save(fig, output, "figure_06_human_AB_phenotype.png")
    # Figure 7: first Mixed profiles with whole-profile endpoint canonicalization.
    first = probes[probes.probe_epoch.eq("first")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, group in zip(axes, HUMAN):
        rows = []
        for _, one in first[first.pretraining_condition.eq(group)].groupby("seed"):
            pivot = one.pivot_table(index="transition_position", columns="context", values="normalized_RT")
            fast, slow = ("A", "B") if pivot.A.mean() <= pivot.B.mean() else ("B", "A")
            rows.extend((p, pivot.loc[p, fast], pivot.loc[p, slow], pivot.loc[p, "Mixed"]) for p in pivot.index)
        array = np.asarray(rows)
        for col, label, color in ((1, "Fast", "#377eb8"), (2, "Slow", "#e41a1c"), (3, "Mixed", "#4daf4a")):
            ax.plot(range(1, 8), [array[array[:, 0] == p, col].mean() for p in range(1, 8)], marker="o", label=label, color=color)
        ax.set_title(group); ax.set_xlabel("Transition")
    axes[0].set_ylabel("Normalized RT"); axes[0].legend()
    _save(fig, output, "figure_07_first_mixed_probe.png")
    # Figures 8/9: Mixed and Novel outcome triptychs.
    for context, name in (("Mixed", "figure_08_mixed_interpolation.png"), ("Novel", "figure_09_novel_transfer.png")):
        final = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late") & metrics.probe_context.eq(context)]
        between = interpolation[(interpolation.row_type.eq("seed_mean")) & interpolation.probe_epoch.eq("late") & interpolation.probe_context.eq(context)].groupby("pretraining_condition").between_endpoint_fraction.mean()
        data = [between, final.groupby("pretraining_condition").NLL_deficit_over_worse_endpoint.mean(),
                final.groupby("pretraining_condition").RT_deficit_over_worse_endpoint.mean()]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, values, title in zip(axes, data, ["Between fraction", "NLL deficit", "RT deficit"]):
            ax.bar(range(3), [values.get(group, np.nan) for group in HUMAN], color=HUMAN_COLORS)
            ax.set_xticks(range(3), ["H0", "H1", "H2"]); ax.set_title(title)
        fig.suptitle(context); _save(fig, output, name)
    # Figure 10: differentiation-transfer frontier.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); mixed = frontier[frontier.probe_context.eq("Mixed")]
    for group, color in zip(HUMAN, HUMAN_COLORS):
        one = mixed[mixed.pretraining_condition.eq(group)]
        axes[0].scatter(one.aggregate_AB_separation, one.between_endpoint_fraction, color=color, label=group)
        axes[1].scatter(one.aggregate_AB_separation, one.NLL_deficit_over_worse_endpoint, color=color)
    axes[0].set_ylabel("Mixed between fraction"); axes[1].set_ylabel("Mixed NLL deficit")
    for ax in axes: ax.set_xlabel("A/B differentiation")
    axes[0].legend(fontsize=8); _save(fig, output, "figure_10_human_differentiation_transfer.png")


def make_internal_figures(output: Path, heldout, gates, decomposition):
    output.mkdir(parents=True, exist_ok=True)
    curriculum_schematic(output); internal_generalization(heldout, output)
    gate_summary(gates, output); decomposition_plot(decomposition, output)
    abstraction_frontier(gates, output)
