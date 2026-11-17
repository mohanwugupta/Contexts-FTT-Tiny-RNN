from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import PROBES
from .config import C0, C1, C2, C3, CONDITIONS, HIGH


FIGURES = tuple(f"PHASE1L_FIG{i:02d}_{name}.png" for i, name in enumerate((
    "canonical_acquisition", "familiar_transition_profiles", "probe_transition_profiles",
    "probe_loss_by_seed", "AB_separation", "shared_trajectory_residual", "endpoint_geometry",
    "neutral_trajectory_closure", "cue_projection_audit", "ablation_summary",
), 1))


LABELS = {C0: "C0 baseline", C1: "C1 contrast", C2: "C2 closure", C3: "C3 full"}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def make_figures(output, training, transition, neutral, summary, cue_audit, matrix):
    figures = Path(output) / "figures"; figures.mkdir(parents=True, exist_ok=True)
    high_training = training[training.family.eq(HIGH)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITIONS):
        data = high_training[high_training.condition.eq(condition)].groupby(["training_sequence_index", "training_context"]).raw_sequence_ce_pre_update.agg(["mean", "sem"]).reset_index()
        for context, group in data.groupby("training_context"):
            error = group["sem"].fillna(0); ax.plot(group.training_sequence_index, group["mean"], label=context)
            ax.fill_between(group.training_sequence_index, group["mean"] - error, group["mean"] + error, alpha=.15)
        ax.set(title=LABELS[condition], xlabel="External training sequence", ylabel="Raw pre-update CE"); ax.legend()
    _save(fig, figures / FIGURES[0])

    high = transition[transition.family.eq(HIGH)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, probe in zip(axes, ("A", "B")):
        data = high[high.probe.eq(probe)].groupby(["condition", "transition"]).raw_ce.mean().reset_index()
        for condition, group in data.groupby("condition"):
            ax.plot(group.transition, group.raw_ce, marker="o", label=LABELS[condition])
        ax.set(title=f"Familiar {probe}", xlabel="Transition", ylabel="Raw CE"); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[1])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, probe in zip(axes, PROBES):
        data = high[high.probe.eq(probe)].groupby(["condition", "transition"]).raw_ce.mean().reset_index()
        for condition, group in data.groupby("condition"):
            ax.plot(group.transition, group.raw_ce, marker="o", label=LABELS[condition])
        ax.set(title=probe, xlabel="Transition", ylabel="Raw CE"); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[2])

    high_summary = summary[summary.family.eq(HIGH)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x = np.arange(4)
    for ax, probe in zip(axes, PROBES):
        pivot = high_summary.pivot(index="seed", columns="condition", values=f"{probe}_seq_ce").reindex(columns=CONDITIONS)
        for seed, row in pivot.iterrows(): ax.plot(x, row.values, marker="o", alpha=.55)
        ax.set_xticks(x, ["C0", "C1", "C2", "C3"]); ax.set(title=probe, xlabel="Condition", ylabel="Raw sequence CE")
    _save(fig, figures / FIGURES[3])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, column, title in ((axes[0], "AB_hidden_separation", "Hidden A/B separation"), (axes[1], "AB_task_separation", "Task A/B separation")):
        data = high[high.probe.eq("A")].groupby(["condition", "transition"])[column].mean().reset_index()
        for condition, group in data.groupby("condition"): ax.plot(group.transition, group[column], marker="o", label=LABELS[condition])
        ax.set(title=title, xlabel="Transition", ylabel="Euclidean separation"); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[4])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for col, probe in enumerate(PROBES):
        for row, (column, label) in enumerate((("hidden_midpoint_residual", "Hidden residual"), ("task_midpoint_residual", "Task residual"))):
            ax = axes[row, col]; data = high[high.probe.eq(probe)].groupby(["condition", "transition"])[column].mean().reset_index()
            for condition, group in data.groupby("condition"): ax.plot(group.transition, group[column], marker="o", label=LABELS[condition])
            ax.set(title=f"{probe}: {label}", xlabel="Transition", ylabel="Normalized residual"); ax.legend(fontsize=6)
    _save(fig, figures / FIGURES[5])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for col, probe in enumerate(PROBES):
        for row, (column, label) in enumerate((("alpha_task", "Task endpoint alpha"), ("off_axis_task", "Off-axis residual"))):
            ax = axes[row, col]; data = high[high.probe.eq(probe)].groupby(["condition", "transition"])[column].mean().reset_index()
            for condition, group in data.groupby("condition"): ax.plot(group.transition, group[column], marker="o", label=LABELS[condition])
            if column == "alpha_task": ax.axhline(.5, color="black", ls="--", lw=.7)
            ax.set(title=f"{probe}: {label}", xlabel="Transition", ylabel=label); ax.legend(fontsize=6)
    _save(fig, figures / FIGURES[6])

    high_neutral = neutral[neutral.family.eq(HIGH)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, column, title in zip(axes, ("neutral_hidden_closure_residual", "neutral_task_closure_residual", "neutral_raw_ce"),
                                 ("Neutral hidden closure", "Neutral task closure", "Neutral raw CE")):
        data = high_neutral.groupby(["condition", "transition"])[column].mean().reset_index()
        for condition, group in data.groupby("condition"): ax.plot(group.transition, group[column], marker="o", label=LABELS[condition])
        ax.set(title=title, xlabel="Transition", ylabel=title); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[7])

    fig, ax = plt.subplots(figsize=(10, 5)); colors = ["#4C78A8" if name in ("A", "B") else "#F58518" for name in cue_audit.cue_name]
    ax.bar(cue_audit.cue_name, cue_audit.alpha, color=colors); ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="Original raw cue", ylabel="A/B-derived alpha(c)", title="Frozen familiar-contrast projection"); ax.tick_params(axis="x", rotation=25)
    _save(fig, figures / FIGURES[8])

    columns = [column for column in matrix.columns if column not in ("family", "seed")]
    high_matrix = matrix[matrix.family.eq(HIGH)].set_index("seed")[columns]; scale = high_matrix.abs().max().replace(0, 1); display = high_matrix / scale
    fig, ax = plt.subplots(figsize=(17, max(5, .38 * len(display)))); image = ax.imshow(display, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(columns)), [column.replace("_", " ") for column in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(display)), display.index); ax.set(title="Continuous Phase 1L ablation summary", xlabel="Signed metric (column-scaled)", ylabel="High seed")
    fig.colorbar(image, ax=ax, label="Signed value / column max absolute value"); _save(fig, figures / FIGURES[9])


def enforce_figures(output):
    missing = [str(Path(output) / "figures" / name) for name in FIGURES if not (Path(output) / "figures" / name).exists()]
    if missing: raise RuntimeError("Missing Phase 1L mandatory figures:\n- " + "\n- ".join(missing))
