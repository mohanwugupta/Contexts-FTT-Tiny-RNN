"""Objective-calibration and five-seed screening figures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "execution-objective-search-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FAMILIES = ("A", "B", "C", "D", "E")
FAMILY_TITLES = {
    "A": "A — quadratic confidence risk",
    "B": "B — steeper confidence risk",
    "C": "C — predictive entropy risk",
    "D": "D — probability-margin risk",
    "E": "E — risk-optimal policy BCE",
}
METRIC_LABELS = {
    "sequence_ce": "Sequence cross-entropy",
    "correct_token_confidence": "Correct-token confidence",
    "execution_speed": "Execution speed",
    "normalized_rt": "Normalized RT (1 - speed)",
}


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.6, alpha=.5)


def plot_calibration(calibration, candidate_summary, output_path):
    families = tuple(family for family in FAMILIES if calibration.family.eq(family).any())
    fig, axes = plt.subplots(
        len(families), 2, figsize=(14, max(4.5, 3.7 * len(families))),
        sharex=True, squeeze=False,
    )
    pass_map = candidate_summary.set_index("candidate_id").calibration_passed.to_dict()
    for row, family in enumerate(families):
        data = calibration[calibration.family.eq(family)]
        for candidate_id, group in data.groupby("candidate_id", sort=False):
            group = group.sort_values("certainty")
            passed = bool(pass_map[candidate_id])
            label = candidate_id.replace("_", " ")
            style = "-" if passed else ":"
            alpha = .90 if passed else .35
            axes[row, 0].plot(
                group.certainty, group.optimal_speed, linestyle=style, alpha=alpha,
                lw=1.8, label=label,
            )
            axes[row, 1].plot(
                group.certainty, group.optimal_normalized_rt, linestyle=style,
                alpha=alpha, lw=1.8, label=label,
            )
        axes[row, 0].set_ylabel(FAMILY_TITLES[family])
        if not data.empty:
            axes[row, 0].legend(frameon=False, fontsize=6.8, ncol=2, loc="best")
        _style(axes[row, 0]); _style(axes[row, 1])
    axes[0, 0].set_title("Normative optimal speed", fontweight="bold")
    axes[0, 1].set_title("Normative optimal normalized RT", fontweight="bold")
    axes[-1, 0].set_xlabel("Certainty statistic")
    axes[-1, 1].set_xlabel("Certainty statistic")
    fig.suptitle(
        "Analytical execution optima before SRN training\n"
        "solid = passed calibration; dotted = rejected analytically",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _finish(fig, output_path)


def plot_training_metric(screening, metric, output_path):
    families = tuple(family for family in FAMILIES if screening.family.eq(family).any())
    columns = 2 if len(families) > 1 else 1
    rows = int(np.ceil(len(families) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(14 if columns == 2 else 8, max(4.5, 4.2 * rows)),
        sharex=True, squeeze=False,
    )
    for axis, family in zip(axes.ravel(), families):
        data = screening[screening.family.eq(family)]
        for candidate_id, group in data.groupby("candidate_id", sort=False):
            summary = group.groupby("global_sequence", as_index=False)[metric].mean()
            axis.plot(
                summary.global_sequence, summary[metric], lw=1.5, alpha=.82,
                label=candidate_id.replace("_", " "),
            )
        axis.set_title(FAMILY_TITLES[family], fontweight="bold")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_xlabel("Training sequence")
        if not data.empty:
            axis.legend(frameon=False, fontsize=6.8, ncol=2)
        else:
            axis.text(.5, .5, "No candidate passed analytical calibration",
                      transform=axis.transAxes, ha="center", va="center", fontsize=9)
        for boundary in range(10, int(screening.global_sequence.max()), 10):
            axis.axvline(boundary + .5, color="#9CA3AF", lw=.35, alpha=.35)
        _style(axis)
    for axis in axes.ravel()[len(families):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Five-seed High-scale screening: {METRIC_LABELS[metric]}",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _finish(fig, output_path)


def plot_speed_tracking(screening, output_path):
    families = tuple(family for family in FAMILIES if screening.family.eq(family).any())
    columns = 2 if len(families) > 1 else 1
    rows = int(np.ceil(len(families) / columns))
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(11.5 if columns == 2 else 7, max(5, 4.7 * rows)),
        sharex=True, sharey=True, squeeze=False,
    )
    for axis, family in zip(axes.ravel(), families):
        data = screening[screening.family.eq(family)]
        for candidate_id, group in data.groupby("candidate_id", sort=False):
            axis.scatter(
                group.objective_optimal_speed, group.execution_speed,
                s=9, alpha=.18, label=candidate_id.replace("_", " "),
            )
        axis.plot([0, 1], [0, 1], color="#111827", linestyle="--", lw=1)
        axis.set_title(FAMILY_TITLES[family], fontweight="bold")
        axis.set_xlabel("Objective-implied optimal speed")
        axis.set_ylabel("Actual network speed")
        if not data.empty:
            axis.legend(frameon=False, fontsize=6.5, ncol=2)
        else:
            axis.text(.5, .5, "No calibrated candidate", transform=axis.transAxes,
                      ha="center", va="center", fontsize=9)
        _style(axis)
    for axis in axes.ravel()[len(families):]:
        axis.set_visible(False)
    fig.suptitle(
        "Actual execution policy versus the objective-implied optimum",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _finish(fig, output_path)


def plot_gate_matrix(summary, output_path):
    gates = [
        "calibration_passed",
        "gate_A_learning_direction",
        "gate_B_execution_grounding",
        "gate_C_objective_tracking",
        "gate_D_no_pathological_saturation",
        "gate_E_contextual_dynamics",
        "screening_passed",
    ]
    table = summary.sort_values(
        ["family", "gamma", "c_time", "c_risk"]
    ).reset_index(drop=True)
    values = table[gates].fillna(False).astype(bool).astype(int).to_numpy()
    fig, axis = plt.subplots(figsize=(11.5, max(6, .37 * len(table) + 2)))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(range(len(gates)), [
        "Calibration", "A: learning", "B: grounding", "C: tracking",
        "D: dynamic", "E: context", "All gates",
    ], rotation=28, ha="right")
    axis.set_yticks(range(len(table)), table.candidate_id.str.replace("_", " "))
    for row in range(len(table)):
        for column in range(len(gates)):
            axis.text(column, row, "PASS" if values[row, column] else "FAIL",
                      ha="center", va="center", fontsize=7)
    axis.set_title("Objective calibration and behavioral acceptance gates", fontweight="bold")
    fig.colorbar(image, ax=axis, ticks=[0, 1], label="Gate status")
    _finish(fig, output_path)


def make_search_figures(output_dir, calibration, screening, summary):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_calibration(calibration, summary, output / "figure_01_normative_calibration")
    if screening.empty:
        return
    for number, metric in enumerate((
        "sequence_ce", "correct_token_confidence", "execution_speed", "normalized_rt"
    ), start=2):
        plot_training_metric(
            screening, metric, output / f"figure_{number:02d}_{metric}_over_training"
        )
    plot_speed_tracking(screening, output / "figure_06_actual_vs_optimal_speed")
    plot_gate_matrix(summary, output / "figure_07_candidate_gate_matrix")
