"""Figures for the execution-readout comparison and conditional state-space fits."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "execution-readout-diagnostic-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import READOUT_CONDITIONS, READOUT_LABELS  # noqa: E402
from .state_space import MODEL_LABELS  # noqa: E402


READOUT_COLORS = {
    "deterministic_normative": "#0072B2",
    "monotonic_two_parameter": "#009E73",
    "unrestricted_zero_initialized": "#D55E00",
}
MODEL_COLORS = {"SS_LUMP": "#E69F00", "SS_CROSS": "#009E73", "SS_SPLIT": "#8C564B"}


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.55, alpha=.5)


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_trajectories(sequence, output_path):
    metrics = (
        ("sequence_ce", "Sequence cross-entropy"),
        ("correct_token_confidence", "Correct-token confidence"),
        ("execution_speed", "Execution speed"),
        ("normalized_rt", "Normalized RT (1 - speed)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for condition in READOUT_CONDITIONS:
            data = sequence[sequence.readout_condition.eq(condition)]
            summary = data.groupby("global_sequence")[metric].agg(["mean", "std", "count"])
            sem = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
            x = summary.index.to_numpy(dtype=float)
            mean = summary["mean"].to_numpy(dtype=float)
            color = READOUT_COLORS[condition]
            axis.plot(x, mean, color=color, lw=1.6, label=READOUT_LABELS[condition])
            axis.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                              color=color, alpha=.12, linewidth=0)
        for boundary in range(10, int(sequence.global_sequence.max()), 10):
            axis.axvline(boundary + .5, color="#9CA3AF", lw=.4, alpha=.4)
        axis.set_xlabel("Training sequence")
        axis.set_ylabel(label)
        _style(axis)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Matched High-scale=2 execution-readout diagnostic",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _finish(fig, output_path)


def plot_optimum_tracking(tokens, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for axis, condition in zip(axes, READOUT_CONDITIONS):
        data = tokens[tokens.readout_condition.eq(condition)]
        axis.scatter(
            data.objective_optimal_speed, data.execution_speed,
            s=8, alpha=.16, color=READOUT_COLORS[condition], edgecolors="none",
        )
        axis.plot([0, 1], [0, 1], color="#111827", ls="--", lw=1)
        axis.set_title(READOUT_LABELS[condition], fontsize=10, fontweight="bold")
        axis.set_xlabel("Objective-implied optimal speed")
        axis.set_ylabel("Actual execution speed")
        _style(axis)
    fig.suptitle("Policy tracking across token states", fontsize=15, fontweight="bold")
    _finish(fig, output_path)


def plot_gate_matrix(summary, output_path):
    gates = [
        "gate_A_learning_direction", "gate_B_execution_grounding",
        "gate_C_objective_tracking", "gate_D_no_pathological_saturation",
        "gate_E_contextual_dynamics", "screening_passed",
    ]
    table = summary.set_index("readout_condition").loc[list(READOUT_CONDITIONS)].reset_index()
    values = table[gates].astype(bool).astype(int).to_numpy()
    fig, axis = plt.subplots(figsize=(10.5, 4.2))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(range(len(gates)), [
        "A: learning", "B: grounding", "C: tracking",
        "D: dynamic", "E: context", "All gates",
    ], rotation=25, ha="right")
    axis.set_yticks(range(len(table)), [
        READOUT_LABELS[value] for value in table.readout_condition
    ])
    for row in range(len(table)):
        for column in range(len(gates)):
            axis.text(column, row, "PASS" if values[row, column] else "FAIL",
                      ha="center", va="center", fontsize=8)
    axis.set_title("Unchanged objective-search acceptance gates", fontweight="bold")
    fig.colorbar(image, ax=axis, ticks=[0, 1], label="Gate status")
    _finish(fig, output_path)


def plot_state_space_overlays(predictions, gate_summary, output_path):
    passing = gate_summary.loc[
        gate_summary.screening_passed.eq(True), "readout_condition"
    ].tolist()
    rows = max(1, len(passing))
    fig, axes = plt.subplots(rows, 1, figsize=(13, max(4.8, 4.1 * rows)), squeeze=False)
    if not passing:
        axis = axes[0, 0]
        axis.text(.5, .5, "No readout passed A–E; state-space fitting correctly skipped",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
    for axis, condition in zip(axes.ravel(), passing):
        data = predictions[predictions.readout_condition.eq(condition)]
        observed = data[data.model.eq("SS_LUMP")].sort_values("global_sequence")
        axis.plot(observed.global_sequence, observed.observed, color="#111827", lw=2,
                  label="Observed mean RT")
        for model in MODEL_LABELS:
            model_data = data[data.model.eq(model)].sort_values("global_sequence")
            winner = bool(model_data.winner.iloc[0])
            axis.plot(
                model_data.global_sequence, model_data.predicted,
                color=MODEL_COLORS[model], lw=2 if winner else 1.2,
                ls="-" if winner else "--",
                label=f"{MODEL_LABELS[model]}{' (BIC winner)' if winner else ''}",
            )
        for boundary in range(10, int(observed.global_sequence.max()), 10):
            axis.axvline(boundary + .5, color="#9CA3AF", lw=.35, alpha=.4)
        axis.set_title(READOUT_LABELS[condition], fontweight="bold")
        axis.set_xlabel("Training sequence")
        axis.set_ylabel("Normalized RT")
        axis.legend(frameon=False, ncol=4, fontsize=8)
        _style(axis)
    fig.suptitle(
        "Canonical state-space overlays for A–E-passing readouts",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _finish(fig, output_path)


def plot_state_space_bic(fits, gate_summary, output_path):
    passing = gate_summary.loc[
        gate_summary.screening_passed.eq(True), "readout_condition"
    ].tolist()
    fig, axis = plt.subplots(figsize=(11, 5.2))
    if not passing:
        axis.text(.5, .5, "No state-space fits: no readout passed A–E",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
    else:
        width = .24
        x = np.arange(len(passing))
        for offset, model in enumerate(MODEL_LABELS):
            values = []
            for condition in passing:
                row = fits[
                    fits.readout_condition.eq(condition) & fits.model.eq(model)
                ].iloc[0]
                values.append(float(row.delta_BIC))
            axis.bar(x + (offset - 1) * width, values, width=width,
                     color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        axis.set_xticks(x, [READOUT_LABELS[value] for value in passing], rotation=12)
        axis.set_ylabel("Delta BIC (lower is better)")
        axis.legend(frameon=False)
        _style(axis)
    axis.set_title("Lump/Cross/Split comparison", fontweight="bold")
    _finish(fig, output_path)


def make_figures(output_dir, sequence, tokens, gates, state_tables):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_training_trajectories(sequence, output / "figure_01_readout_trajectories")
    plot_optimum_tracking(tokens, output / "figure_02_policy_optimum_tracking")
    plot_gate_matrix(gates, output / "figure_03_gate_matrix")
    plot_state_space_overlays(
        state_tables["state_space_predictions"], gates,
        output / "figure_04_state_space_overlays",
    )
    plot_state_space_bic(
        state_tables["state_space_model_fits"], gates,
        output / "figure_05_state_space_bic",
    )
