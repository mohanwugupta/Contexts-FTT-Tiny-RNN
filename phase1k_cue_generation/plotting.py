from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import HIGH, LOW


FIGURES = tuple(f"PHASE1K_FIG{i:02d}_{name}.png" for i, name in enumerate((
    "canonical_acquisition", "natural_transition_profiles", "local_cue_generation",
    "ab_to_mixed_convergence", "ab_to_mixed_raw_ce", "nocontext_takeover",
    "novel_takeover", "pulse_vs_persistent", "geometry_specificity", "seed_outcome_matrix",
), 1))


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _line_profile(ax, data, label, **kwargs):
    mean = data.groupby("transition").raw_ce.mean()
    if len(mean):
        ax.plot(mean.index, mean.values, marker="o", label=label, **kwargs)


def make_figures(output, acquisition, natural, local, takeover, pulse, endpoint, convergence, seed_level):
    figures = Path(output) / "figures"; figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, family in zip(axes, (HIGH, LOW)):
        data = acquisition[acquisition.family.eq(family)].groupby(["training_sequence_index", "training_context"]).raw_sequence_loss_pre_update.agg(["mean", "sem"]).reset_index()
        for context, group in data.groupby("training_context"):
            error = group["sem"].fillna(0); ax.plot(group.training_sequence_index, group["mean"], label=context)
            ax.fill_between(group.training_sequence_index, group["mean"] - error, group["mean"] + error, alpha=.15)
        ax.set(title=family, xlabel="Actual external training sequence", ylabel="Raw mean CE"); ax.legend()
    _save(fig, figures / FIGURES[0])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, family in zip(axes, (HIGH, LOW)):
        subset = natural[natural.family.eq(family)]
        for probe, group in subset.groupby("probe"):
            _line_profile(ax, group, probe)
        ax.set(title=family, xlabel="Transition", ylabel="Raw CE"); ax.legend(fontsize=6, ncol=2)
    _save(fig, figures / FIGURES[1])

    focus = local[local.family.eq(HIGH) & local.source_probe.isin(["A", "B", "NoContext", "NovelMean"])]
    branches = ["native", "Mixed", "A", "B", "zero"]
    metrics = (("hidden_midpoint_residual", "Hidden midpoint residual"), ("task_midpoint_residual", "Task midpoint residual"),
               ("task_direction_cosine", "Cue-to-midpoint task cosine"), ("raw_ce", "Raw CE"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (column, title) in zip(axes.flat, metrics):
        data = focus.groupby(["source_probe", "branch_cue"])[column].mean().reset_index()
        for source, group in data.groupby("source_probe"):
            values = group.set_index("branch_cue")[column].reindex(branches)
            ax.plot(branches, values, marker="o", label=source)
        ax.set(title=title, xlabel="Matched-prestate branch cue", ylabel=title); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[2])

    high_conv = convergence[convergence.family.eq(HIGH)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    data = high_conv.groupby("lag").mean(numeric_only=True)
    axes[0, 0].plot(data.index, data.normalized_AB_branch_distance, marker="o", label="A→M vs B→M")
    axes[0, 0].axhline(1, color="black", ls="--", label="Natural A/B distance"); axes[0, 0].set(title="A/B branch convergence", ylabel="Normalized distance")
    axes[0, 1].plot(data.index, data.A_branch_task_midpoint_residual, marker="o", label="A→M")
    axes[0, 1].plot(data.index, data.B_branch_task_midpoint_residual, marker="o", label="B→M")
    axes[0, 1].plot(data.index, data.natural_Mixed_task_midpoint_residual, ls="--", label="Natural Mixed")
    axes[0, 1].set(title="Task midpoint residual", ylabel="Normalized residual")
    axes[1, 0].plot(data.index, data.A_branch_alpha_task, marker="o", label="A→M")
    axes[1, 0].plot(data.index, data.B_branch_alpha_task, marker="o", label="B→M"); axes[1, 0].axhline(.5, color="black", ls="--", label="Midpoint")
    axes[1, 0].set(title="Endpoint coordinate", ylabel="Task alpha")
    axes[1, 1].plot(data.index, data.A_branch_off_axis_task, marker="o", label="A→M")
    axes[1, 1].plot(data.index, data.B_branch_off_axis_task, marker="o", label="B→M"); axes[1, 1].set(title="Off-axis residual", ylabel="Normalized residual")
    for ax in axes.flat: ax.set_xlabel("Lag from takeover"); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[3])

    positions = (1, 4, 5); fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for row, source in enumerate(("A", "B")):
        for col, position in enumerate(positions):
            ax = axes[row, col]; _line_profile(ax, natural[natural.family.eq(HIGH) & natural.probe.eq(source)], f"Natural {source}", lw=2)
            _line_profile(ax, natural[natural.family.eq(HIGH) & natural.probe.eq("Mixed")], "Natural Mixed", ls="--")
            data = takeover[takeover.family.eq(HIGH) & takeover.source_probe.eq(source) & takeover.takeover_transition.eq(position)]
            _line_profile(ax, data, f"{source}→Mixed", lw=2); ax.axvline(position, color="black", ls=":")
            ax.set(title=f"{source}→Mixed at {position}", xlabel="Transition", ylabel="Raw CE"); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[4])

    def takeover_grid(source, path):
        fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
        for position, ax in zip(range(1, 8), axes.flat):
            _line_profile(ax, natural[natural.family.eq(HIGH) & natural.probe.eq(source)], f"Natural {source}", lw=2)
            _line_profile(ax, natural[natural.family.eq(HIGH) & natural.probe.eq("Mixed")], "Natural Mixed", ls="--")
            for frame, label in ((pulse, "Mixed pulse"), (takeover, "Mixed takeover")):
                _line_profile(ax, frame[frame.family.eq(HIGH) & frame.source_probe.eq(source) & frame.takeover_transition.eq(position)], label)
            for intervention, label in (("persistent_A_takeover", "A takeover"), ("persistent_B_takeover", "B takeover")):
                data = endpoint[endpoint.family.eq(HIGH) & endpoint.source_probe.eq(source) & endpoint.takeover_transition.eq(position) & endpoint.intervention.eq(intervention)]
                _line_profile(ax, data, label)
            ax.axvline(position, color="black", ls=":", lw=.7); ax.set(title=f"Takeover at {position}", xlabel="Transition", ylabel="Raw CE")
            if position == 1: ax.legend(fontsize=6)
        axes.flat[-1].axis("off"); _save(fig, path)
    takeover_grid("NoContext", figures / FIGURES[5]); takeover_grid("NovelMean", figures / FIGURES[6])

    sources = ["A", "B", "Mixed", "NoContext", "NovelMean"]
    combined = pd.concat([pulse.assign(mode="pulse"), takeover.assign(mode="persistent")], ignore_index=True)
    combined = combined[combined.family.eq(HIGH) & combined.lag.ge(0) & combined.source_probe.isin(sources)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, column, title in zip(axes, ("raw_ce_rescue", "task_midpoint_residual", "hidden_midpoint_residual"),
                                 ("Raw CE rescue", "Task midpoint residual", "Hidden midpoint residual")):
        data = combined.groupby(["lag", "source_probe", "mode"])[column].mean().reset_index()
        for (source, mode), group in data.groupby(["source_probe", "mode"]):
            ax.plot(group.lag, group[column], marker="o", ls="-" if mode == "persistent" else "--", label=f"{source}:{mode}")
        ax.set(title=title, xlabel="Lag", ylabel=title); ax.legend(fontsize=5, ncol=2)
    _save(fig, figures / FIGURES[7])

    specificity = local[local.family.eq(HIGH) & local.source_probe.isin(["A", "B", "NoContext", "NovelMean"]) & local.branch_cue.isin(["A", "Mixed", "B"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, title in ((axes[0], "alpha_task", "Endpoint coordinate"), (axes[1], "off_axis_task", "Off-axis residual")):
        data = specificity.groupby(["source_probe", "branch_cue"])[column].mean().reset_index()
        x = np.arange(3); width = .18
        for index, (source, group) in enumerate(data.groupby("source_probe")):
            values = group.set_index("branch_cue")[column].reindex(["A", "Mixed", "B"])
            ax.bar(x + (index - 1.5) * width, values, width, label=source)
        ax.set_xticks(x, ["A cue", "Mixed cue", "B cue"]); ax.set(title=title, ylabel=title); ax.legend(fontsize=7)
    _save(fig, figures / FIGURES[8])

    columns = ["A_to_M_final_convergence", "B_to_M_final_convergence", "NoContext_local_task_generation",
               "NoContext_takeover_remaining_rescue", "NovelMean_local_task_generation", "NovelMean_takeover_remaining_rescue",
               "pulse_vs_takeover_difference", "mixed_midpoint_specificity"]
    matrix = seed_level[seed_level.family.eq(HIGH)].set_index("seed")[columns]
    scale = matrix.abs().max(axis=0).replace(0, 1); display = matrix / scale
    fig, ax = plt.subplots(figsize=(14, max(5, .38 * len(matrix))))
    image = ax.imshow(display, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(columns)), [column.replace("_", " ") for column in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix)), matrix.index); ax.set(xlabel="Continuous signed metric (column-scaled)", ylabel="High seed", title="Per-seed Phase 1K outcome matrix")
    fig.colorbar(image, ax=ax, label="Signed value / column max absolute value"); _save(fig, figures / FIGURES[9])


def enforce_figures(output):
    missing = [str(Path(output) / "figures" / name) for name in FIGURES if not (Path(output) / "figures" / name).exists()]
    if missing:
        raise RuntimeError("Missing Phase 1K mandatory figures:\n- " + "\n- ".join(missing))
