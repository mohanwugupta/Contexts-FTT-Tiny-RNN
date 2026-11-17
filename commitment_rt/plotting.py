"""Phase-1 mechanism figures for emergent commitment RT."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "commitment-rt-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {"A": "#0072B2", "B": "#D55E00"}
MODEL_COLORS = {"SS_LUMP": "#E69F00", "SS_CROSS": "#009E73", "SS_SPLIT": "#8C564B"}


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.5, alpha=.45)


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_architecture(output_path):
    fig, axis = plt.subplots(figsize=(13, 4.3))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = [
        (0.3, 1.4, 2.2, 1.2, "Current token\n+ 3D context", "#DCEEFF"),
        (3.2, 1.25, 4.2, 1.5, "Shared 8-unit tanh SRN\nh₁ → h₂ → … → h₈\n(same input each cycle)", "#E6F4EA"),
        (8.1, 2.2, 2.2, .9, "4-way next-token\nconfidence", "#FFF2CC"),
        (8.1, .8, 2.2, .9, "WAIT / COMMIT\nhazard", "#FCE8E6"),
        (11.0, 1.35, 1.7, 1.25, "RT = E[prep\ncycles]\n(no RT target)", "#EDE7F6"),
    ]
    for x, y, width, height, label, color in boxes:
        rectangle = plt.Rectangle((x, y), width, height, facecolor=color,
                                  edgecolor="#374151", linewidth=1.2)
        axis.add_patch(rectangle)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                  fontsize=10)
    arrows = [((2.5, 2.0), (3.2, 2.0)), ((7.4, 2.2), (8.1, 2.65)),
              ((7.4, 1.8), (8.1, 1.25)), ((10.3, 1.25), (11.0, 1.85))]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start,
                      arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#374151"})
    axis.set_title("Emergent RT from recurrent preparation and learned commitment",
                   fontsize=15, fontweight="bold")
    _finish(fig, output_path)


def plot_example_transitions(cycles, selected_c_time, output_path):
    data = cycles[cycles.c_time.eq(selected_c_time)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    if data.empty:
        axes[0, 0].text(.5, .5, "No passing candidate", ha="center", va="center")
        for axis in axes.ravel():
            axis.set_axis_off()
        _finish(fig, output_path)
        return
    late = data[data.training_sequence.gt(data.training_sequence.max() - 20)]
    grouped = late.groupby(["seed", "training_sequence", "sequence_position"], as_index=False)
    summary = grouped.expected_rt_cycles.first()
    easy_key = summary.sort_values("expected_rt_cycles").iloc[0]
    switch_sequences = set(
        data.loc[data.sequence_in_block.eq(1) & data.block_index.gt(1), "training_sequence"]
    )
    switch_pool = data[data.training_sequence.isin(switch_sequences)]
    switch_summary = (
        switch_pool.groupby(["seed", "training_sequence", "sequence_position"], as_index=False)
        .expected_rt_cycles.first().sort_values("expected_rt_cycles", ascending=False)
    )
    keys = [
        ("Easy late-training transition", easy_key),
        ("Context-switch transition", switch_summary.iloc[0]),
    ]
    for row_index, (label, key) in enumerate(keys):
        one = data[
            data.seed.eq(key.seed)
            & data.training_sequence.eq(key.training_sequence)
            & data.sequence_position.eq(key.sequence_position)
        ].sort_values("prep_cycle")
        for column, (metric, title) in enumerate([
            ("p_correct", "Correct-token confidence"),
            ("commit_hazard", "Commit hazard"),
            ("commit_probability", "Commit probability q"),
        ]):
            axis = axes[row_index, column]
            axis.plot(one.prep_cycle, one[metric], marker="o", color="#0072B2")
            axis.set_ylim(-.03, 1.03)
            axis.set_title(f"{label}\n{title}", fontsize=10)
            axis.set_xlabel("Preparation cycle")
            axis.set_ylabel(metric)
            _style(axis)
    _finish(fig, output_path)


def plot_learning_curves(training, selected_c_time, output_path):
    data = training[training.c_time.eq(selected_c_time)]
    metrics = [
        ("sequence_ce", "Sequence CE"),
        ("expected_rt_cycles", "Expected RT (cycles)"),
        ("hard_rt_cycles", "Hard RT (cycles)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    if data.empty:
        axes[0].text(.5, .5, "No passing candidate", ha="center", va="center")
    for axis, (metric, ylabel) in zip(axes, metrics):
        summary = data.groupby("global_sequence")[metric].agg(["mean", "std", "count"])
        x = summary.index.to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        sem = summary["std"].fillna(0).to_numpy(dtype=float) / np.sqrt(summary["count"])
        axis.plot(x, mean, color="#0072B2", lw=1.7)
        axis.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                          color="#0072B2", alpha=.14, linewidth=0)
        for boundary in range(10, int(data.global_sequence.max()), 10) if len(data) else []:
            axis.axvline(boundary + .5, color="#9CA3AF", lw=.35, alpha=.4)
        axis.set_xlabel("External training sequence")
        axis.set_ylabel(ylabel)
        _style(axis)
    fig.suptitle(f"High-scale learned commitment (c_time={selected_c_time:g})",
                 fontsize=14, fontweight="bold")
    _finish(fig, output_path)


def plot_switch_alignment(training, selected_c_time, output_path):
    data = training[training.c_time.eq(selected_c_time)]
    rows = []
    for (_, _), group in data.groupby(["seed", "block_index"], sort=True):
        block = int(group.block_index.iloc[0])
        if block <= 1:
            continue
        switch_sequence = int(group.global_sequence.min())
        seed_data = data[data.seed.eq(group.seed.iloc[0])]
        for relative in range(-3, 4):
            one = seed_data[seed_data.global_sequence.eq(switch_sequence + relative)]
            if len(one):
                rows.append({"relative_sequence": relative,
                             "normalized_rt": float(one.normalized_rt.iloc[0])})
    fig, axis = plt.subplots(figsize=(8, 4.8))
    if rows:
        import pandas as pd
        aligned = pd.DataFrame(rows).groupby("relative_sequence").normalized_rt.agg(
            ["mean", "std", "count"]
        )
        sem = aligned["std"].fillna(0) / np.sqrt(aligned["count"])
        axis.errorbar(aligned.index, aligned["mean"], yerr=1.96 * sem,
                      marker="o", color="#D55E00", capsize=3)
        axis.axvline(-.5, color="#111827", ls="--", lw=1)
    else:
        axis.text(.5, .5, "No switch observations", transform=axis.transAxes,
                  ha="center", va="center")
    axis.set_xlabel("Sequences relative to context switch (0 = first new context)")
    axis.set_ylabel("Normalized expected RT")
    axis.set_title("Emergent RT aligned to A/B switches", fontweight="bold")
    _style(axis)
    _finish(fig, output_path)


def plot_state_space_overlay(predictions, selected_c_time, output_path):
    condition = f"learned_c_time_{selected_c_time:g}"
    data = predictions[predictions.timing_condition.eq(condition)] if len(predictions) else predictions
    fig, axis = plt.subplots(figsize=(13, 5))
    if data.empty:
        axis.text(.5, .5, "State-space fit skipped: no candidate passed A–F",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
    else:
        observed = data[data.model.eq("SS_LUMP")].sort_values("global_sequence")
        axis.plot(observed.global_sequence, observed.observed, color="#111827", lw=2,
                  label="High-scale emergent RT")
        for model in ("SS_LUMP", "SS_CROSS", "SS_SPLIT"):
            one = data[data.model.eq(model)].sort_values("global_sequence")
            winner = bool(one.winner.iloc[0])
            axis.plot(one.global_sequence, one.predicted,
                      color=MODEL_COLORS[model], lw=2 if winner else 1.2,
                      ls="-" if winner else "--",
                      label=f"{model.removeprefix('SS_').title()}"
                            f"{' (BIC winner)' if winner else ''}")
        axis.set_xlabel("External training sequence")
        axis.set_ylabel("Normalized expected RT")
        axis.legend(frameon=False, ncol=4)
        _style(axis)
    axis.set_title("Canonical state-space overlay (Low-scale deferred to Phase 2)",
                   fontweight="bold")
    _finish(fig, output_path)


def make_screen_figures(output_dir, training, cycles, selected_c_time, state_tables):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_architecture(output / "figure_01_architecture")
    plot_example_transitions(cycles, selected_c_time, output / "figure_02_example_transition")
    plot_learning_curves(training, selected_c_time, output / "figure_03_learning_curves")
    plot_switch_alignment(training, selected_c_time, output / "figure_04_switch_effects")
    plot_state_space_overlay(
        state_tables["screen_state_space_predictions"], selected_c_time,
        output / "figure_05_state_space_overlay",
    )


def plot_confirmatory_state_space(predictions, output_path):
    conditions = [value for value in ("High_commitment", "Low_commitment")
                  if value in set(predictions.timing_condition)] if len(predictions) else []
    fig, axes = plt.subplots(max(1, len(conditions)), 1,
                             figsize=(13, max(4.8, 4.2 * len(conditions))), squeeze=False)
    if not conditions:
        axes[0, 0].text(.5, .5, "No confirmatory state-space fits",
                        transform=axes[0, 0].transAxes, ha="center", va="center")
        axes[0, 0].set_axis_off()
    for axis, condition in zip(axes.ravel(), conditions):
        data = predictions[predictions.timing_condition.eq(condition)]
        observed = data[data.model.eq("SS_LUMP")].sort_values("global_sequence")
        axis.plot(observed.global_sequence, observed.observed, color="#111827", lw=2,
                  label="Emergent RT")
        for model in MODEL_COLORS:
            one = data[data.model.eq(model)].sort_values("global_sequence")
            winner = bool(one.winner.iloc[0])
            axis.plot(one.global_sequence, one.predicted, color=MODEL_COLORS[model],
                      lw=2 if winner else 1.1, ls="-" if winner else "--",
                      label=f"{model.removeprefix('SS_').title()}"
                            f"{' (winner)' if winner else ''}")
        axis.set_title(condition.replace("_", " "), fontweight="bold")
        axis.set_xlabel("External training sequence")
        axis.set_ylabel("Normalized expected RT")
        axis.legend(frameon=False, ncol=4)
        _style(axis)
    fig.suptitle("Confirmatory High/Low state-space overlays", fontsize=14,
                 fontweight="bold")
    _finish(fig, output_path)


def plot_probe_rt(probes, output_path):
    fig, axis = plt.subplots(figsize=(10, 5.2))
    learned = probes[probes.probe_context.isin(["A", "B", "Mixed_norm", "Novel"])]
    summary = learned.groupby(["condition", "probe_context"]).normalized_rt.agg(
        ["mean", "std", "count"]
    ).reset_index()
    contexts = ["A", "B", "Mixed_norm", "Novel"]
    conditions = list(summary.condition.unique())
    width = .34 if len(conditions) > 1 else .6
    x = np.arange(len(contexts))
    for index, condition in enumerate(conditions):
        one = summary[summary.condition.eq(condition)].set_index("probe_context").reindex(contexts)
        mean = one["mean"].to_numpy(dtype=float)
        sem = one["std"].fillna(0).to_numpy(dtype=float) / np.sqrt(one["count"])
        offset = (index - (len(conditions) - 1) / 2) * width
        axis.bar(x + offset, mean, width=width, yerr=1.96 * sem, capsize=3,
                 label=condition.replace("_", " "), alpha=.85)
    axis.set_xticks(x, contexts)
    axis.set_ylabel("Normalized expected RT")
    axis.set_title("Frozen pure and ambiguous-context commitment probes", fontweight="bold")
    axis.legend(frameon=False)
    _style(axis)
    _finish(fig, output_path)


def plot_causal_effects(patching, output_path):
    fig, axis = plt.subplots(figsize=(10, 5.2))
    if patching.empty:
        axis.text(.5, .5, "Causal analysis skipped by confirmatory gates",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
    else:
        summary = patching.groupby(
            ["probe_context", "donor_context"]
        ).delta_normalized_rt.agg(["mean", "std", "count"]).reset_index()
        labels = [f"{row.probe_context} ← {row.donor_context}"
                  for row in summary.itertuples(index=False)]
        sem = summary["std"].fillna(0) / np.sqrt(summary["count"])
        axis.bar(np.arange(len(summary)), summary["mean"], yerr=1.96 * sem,
                 color=["#0072B2" if donor == "A" else "#D55E00"
                        for donor in summary.donor_context], capsize=3)
        axis.axhline(0, color="#111827", lw=1)
        axis.set_xticks(np.arange(len(summary)), labels, rotation=20, ha="right")
        axis.set_ylabel("Change in normalized expected RT")
        axis.set_title("Token-matched A/B component patching", fontweight="bold")
        _style(axis)
    _finish(fig, output_path)


def make_confirmatory_figures(output_dir, state_tables, probes, patching):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_confirmatory_state_space(
        state_tables["final_state_space_predictions"],
        output / "figure_05_state_space_overlay_high_low",
    )
    plot_probe_rt(probes, output / "figure_06_probe_rt")
    plot_causal_effects(patching, output / "figure_07_causal_effects")
