"""Six-panel prediction/execution trajectory and fixed-readout sensitivity plots."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "probe-prediction-execution-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {"A": "#0072B2", "B": "#D55E00", "Mixed": "#009E73", "Novel": "#CC0077"}
CONDITIONS = ("High_monotonic", "Low_monotonic")
METRICS = (
    ("prediction_loss", "Prediction loss\n(mean correct-token NLL)"),
    ("mean_correct_token_probability", "Correct-token probability"),
    ("mean_normalized_RT", "Normalized RT"),
)


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.5, alpha=.45)


def _finish(fig, stem: Path):
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def sequence_figure_source(sequences: pd.DataFrame) -> pd.DataFrame:
    long = sequences.melt(
        id_vars=["condition", "seed", "completed_training_sequences", "probe_context"],
        value_vars=[metric for metric, _ in METRICS],
        var_name="metric", value_name="value",
    )
    summary = long.groupby(
        ["condition", "completed_training_sequences", "probe_context", "metric"],
        as_index=False,
    ).value.agg(["mean", "sem"]).reset_index()
    summary["ci_lower"] = summary["mean"] - 1.96 * summary["sem"]
    summary["ci_upper"] = summary["mean"] + 1.96 * summary["sem"]
    summary["uncertainty"] = "across_seed_mean_plus_or_minus_1.96_SE"
    return summary


def figure_prediction_vs_execution(source: pd.DataFrame, output_stem: Path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey="col")
    for row, condition in enumerate(CONDITIONS):
        for col, (metric, ylabel) in enumerate(METRICS):
            axis = axes[row, col]
            one = source[source.condition.eq(condition) & source.metric.eq(metric)]
            for context in ("A", "B", "Mixed", "Novel"):
                local = one[one.probe_context.eq(context)].sort_values(
                    "completed_training_sequences"
                )
                x = local.completed_training_sequences.to_numpy(float)
                axis.fill_between(
                    x, local.ci_lower.to_numpy(float), local.ci_upper.to_numpy(float),
                    color=COLORS[context], alpha=.10, linewidth=0,
                )
                axis.plot(x, local["mean"], color=COLORS[context], marker="o", ms=3.5,
                          lw=1.5, label=context)
            axis.set_title(
                f"{condition.replace('_', ' ')} — {ylabel.replace(chr(10), ' ')}"
            )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Completed training sequences (post-update checkpoint)")
            axis.set_xticks([20, 50, 100, 150, 170])
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=4)
    fig.suptitle("Frozen prediction and execution across matched checkpoints",
                 fontsize=16, fontweight="bold")
    _finish(fig, output_stem)


def fixed_readout_figure_source(bootstrap_summary: pd.DataFrame) -> pd.DataFrame:
    wanted = {
        "RT_gap_over_worse_endpoint": "Actual contemporaneous readout",
        "fixed_checkpoint20_RT_gap_over_worse_endpoint": "Checkpoint-20 readout",
    }
    result = bootstrap_summary[
        bootstrap_summary.summary_type.eq("checkpoint_mean")
        & bootstrap_summary.metric.isin(wanted)
    ].copy()
    result["readout_diagnostic"] = result.metric.map(wanted)
    return result


def figure_fixed_readout_comparison(source: pd.DataFrame, output_stem: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey="row")
    style = {
        "Actual contemporaneous readout": ("-", "o"),
        "Checkpoint-20 readout": ("--", "s"),
    }
    for row, condition in enumerate(CONDITIONS):
        for col, probe in enumerate(("Mixed", "Novel")):
            axis = axes[row, col]
            one = source[source.condition.eq(condition) & source.probe_context.eq(probe)]
            for readout, (line_style, marker) in style.items():
                local = one[one.readout_diagnostic.eq(readout)].sort_values(
                    "completed_training_sequences"
                )
                x = local.completed_training_sequences.to_numpy(float)
                axis.fill_between(
                    x, local.ci_lower.to_numpy(float), local.ci_upper.to_numpy(float),
                    color=COLORS[probe], alpha=.12 if line_style == "-" else .06,
                    linewidth=0,
                )
                axis.plot(x, local["mean"], color=COLORS[probe], ls=line_style,
                          marker=marker, ms=3.5, lw=1.7, label=readout)
            axis.axhline(0, color="#111827", lw=.8, alpha=.65)
            axis.axvline(170, color="#6B7280", lw=.8, ls=":",
                         label="Prespecified sensitivity checkpoint 170")
            axis.set_title(f"{condition.replace('_', ' ')} — {probe}")
            axis.set_ylabel("Probe RT gap over slower A/B endpoint")
            axis.set_xlabel("Completed training sequences")
            axis.set_xticks([20, 50, 100, 150, 170])
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", frameon=False, ncol=3)
    fig.suptitle("Counterfactual readout diagnostic on unchanged predictions",
                 fontsize=16, fontweight="bold")
    _finish(fig, output_stem)
