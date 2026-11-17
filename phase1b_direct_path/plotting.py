from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = (
    "D0_frozen_high", "D1_centered_logit_path", "D2_output_hidden_path",
    "D3_whole_hidden_path", "D4_curvature_proxy", "D5_endpoint_anchor_only",
    "D6_frozen_low",
)
COLORS = dict(zip(CONDITION_ORDER, plt.cm.tab10(np.linspace(0, .9, len(CONDITION_ORDER)))))
MANDATORY_TS = tuple(f"P1B_TS{i:02d}_{name}.png" for i, name in enumerate((
    "finetuning_sequence_loss", "finetuning_path_and_anchor_metrics",
    "full_familiar_sequence_loss", "full_familiar_sequence_accuracy",
    "full_familiar_execution_proxy", "full_experiment_sequence_loss",
    "full_experiment_correct_token_probability", "full_experiment_execution_proxy",
    "midpoint_over_finetuning", "cue_sweep_sequence_loss",
    "cue_sweep_correct_token_probability", "cue_sweep_path_residuals",
), start=1))


def _save(fig, output: Path, name: str):
    fig.tight_layout()
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary(frame, x, y, condition):
    return frame.groupby([condition, x], as_index=False)[y].agg(
        mean="mean", low=lambda s: s.quantile(.025), high=lambda s: s.quantile(.975))


def _lines(ax, frame, x, y, *, conditions=None, condition="condition"):
    use = frame if conditions is None else frame[frame[condition].isin(conditions)]
    summary = _summary(use, x, y, condition)
    for name, group in summary.groupby(condition, sort=False):
        group = group.sort_values(x)
        color = COLORS.get(name)
        ax.plot(group[x], group["mean"], label=name, color=color, lw=1.6)
        ax.fill_between(group[x], group.low, group.high, color=color, alpha=.16, linewidth=0)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))


def _block_marks(ax, updates, include_probes=False):
    for block, start in enumerate(range(1, updates + 1, 10), start=1):
        stop = min(start + 9, updates)
        color = "#4C78A8" if block % 2 == 1 else "#F58518"
        label = "A" if block % 2 == 1 else "B"
        ax.axvspan(start - .5, stop + .5, color=color, alpha=.025, zorder=0)
        ax.text((start + stop) / 2, .985, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5, color=color)
    for x in range(10, updates, 10):
        ax.axvline(x + .5, color="0.85", lw=.45, zorder=0)
    if include_probes:
        ax.axvline(updates + .5, color="black", ls="--", lw=.8)


def _bar(ax, frame, value, title=None):
    means = frame.groupby("condition")[value].mean().reindex(CONDITION_ORDER).dropna()
    ax.bar(np.arange(len(means)), means, color=[COLORS[x] for x in means.index])
    ax.set_xticks(np.arange(len(means)), means.index, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(value.replace("_", " "))
    if title:
        ax.set_title(title)


def make_figures(output: Path, tables: dict, updates: int):
    output.mkdir(parents=True, exist_ok=True)
    training = tables["phase1b_timeseries_finetuning"]
    familiar = tables["phase1b_timeseries_familiar_full_schedule"]
    full = tables["phase1b_timeseries_full_experiment"]
    midpoint = tables["phase1b_timeseries_midpoint_checkpoints"]
    sweep = tables["phase1b_cue_sweep"]
    metrics = tables["phase1b_seed_metrics"]
    endpoint = tables["phase1b_endpoint_preservation"]
    rescue = tables["phase1b_rescue_fraction"]

    # TS1
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _lines(ax, training, "update", "sequence_loss")
    _block_marks(ax, updates); ax.legend(fontsize=6, ncol=3)
    _save(fig, output, MANDATORY_TS[0])

    # TS2: incompatible units receive separate panels.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, value in zip(axes.ravel(), (
            "AB_separation", "endpoint_hidden_drift", "endpoint_logit_drift", "training_path_loss")):
        _lines(ax, training, "update", value)
        _block_marks(ax, updates)
    axes[0, 0].legend(fontsize=6, ncol=2)
    _save(fig, output, MANDATORY_TS[1])

    for index, value in enumerate(("sequence_loss", "sequence_accuracy", "mean_execution_proxy"), start=2):
        if value == "mean_execution_proxy" and familiar[value].isna().all():
            (output / MANDATORY_TS[index].replace(".png", ".txt")).write_text(
                "Execution/RT proxy unavailable because no frozen execution policy was present.\n")
            continue
        fig, ax = plt.subplots(figsize=(11, 4.5)); _lines(ax, familiar, "sequence_index", value)
        _block_marks(ax, updates); ax.legend(fontsize=6, ncol=4)
        _save(fig, output, MANDATORY_TS[index])

    for index, value in enumerate((
            "sequence_loss", "mean_correct_token_probability", "mean_execution_proxy"), start=5):
        if value == "mean_execution_proxy" and full[value].isna().all():
            (output / MANDATORY_TS[index].replace(".png", ".txt")).write_text(
                "Execution/RT proxy unavailable because no frozen execution policy was present.\n")
            continue
        fig, ax = plt.subplots(figsize=(11, 4.5)); _lines(ax, full, "sequence_index", value)
        _block_marks(ax, updates, include_probes=True); ax.legend(fontsize=6, ncol=4)
        ax.text(updates + 1, ax.get_ylim()[1], "Mixed / NoContext / Novel probes",
                fontsize=7, va="top")
        _save(fig, output, MANDATORY_TS[index])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, value in zip(axes.ravel(), (
            "heldout_midpoint_sequence_loss", "heldout_midpoint_output_residual",
            "heldout_midpoint_logit_residual", "remaining_patch_gap")):
        _lines(ax, midpoint, "update", value)
    axes[0, 0].legend(fontsize=6, ncol=2)
    _save(fig, output, MANDATORY_TS[8])

    for index, value in ((9, "sequence_loss"), (10, "mean_correct_token_probability")):
        fig, ax = plt.subplots(figsize=(10, 4.5)); _lines(ax, sweep, "cue_lambda", value)
        ax.axvline(.5, color="black", ls="--", lw=.8, label="held-out midpoint")
        ax.legend(fontsize=6, ncol=4); _save(fig, output, MANDATORY_TS[index])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _lines(axes[0], sweep, "cue_lambda", "output_path_residual")
    _lines(axes[1], sweep, "cue_lambda", "logit_path_residual")
    for ax in axes: ax.axvline(.5, color="black", ls="--", lw=.8)
    axes[0].legend(fontsize=6, ncol=2)
    _save(fig, output, MANDATORY_TS[11])

    # Summary figures.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, value in zip(axes.ravel(), (
            "familiar_loss_ratio", "AB_separation_ratio",
            "endpoint_hidden_drift_ratio", "endpoint_logit_drift_ratio")):
        _bar(ax, endpoint, value)
        if "ratio" in value: ax.axhline(1 if value in {"familiar_loss_ratio", "AB_separation_ratio"} else .1,
                                        color="black", ls="--", lw=.8)
    _save(fig, output, "P1B_01_endpoint_preservation.png")
    for name, value in (("P1B_02_Mixed_loss_by_condition.png", "Mixed_loss"),
                        ("P1B_03_output_path_residual_by_condition.png", "output_path_residual"),
                        ("P1B_04_logit_path_residual_by_condition.png", "logit_path_residual"),
                        ("P1B_05_rescue_fraction_by_condition.png", "rescue_fraction"),
                        ("P1B_06_remaining_patch_gap.png", "remaining_patch_gap")):
        source = rescue if value in rescue else metrics
        fig, ax = plt.subplots(figsize=(10, 4.5)); _bar(ax, source, value)
        if value == "rescue_fraction": ax.axhline(.5, color="black", ls="--", lw=.8)
        _save(fig, output, name)
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, group in metrics.groupby("condition"):
        ax.scatter(group.AB_separation_ratio, group.Mixed_loss, label=condition,
                   color=COLORS.get(condition), alpha=.8)
    ax.set_xlabel("A/B separation ratio"); ax.set_ylabel("Mixed loss"); ax.legend(fontsize=6)
    _save(fig, output, "P1B_07_AB_separation_vs_Mixed_loss.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, group in metrics.groupby("condition"):
        value = "logit_path_residual" if condition == "D1_centered_logit_path" else "output_path_residual"
        ax.scatter(group[value], group.Mixed_loss, label=condition, color=COLORS.get(condition), alpha=.8)
    ax.set_xlabel("target path residual"); ax.set_ylabel("Mixed loss"); ax.legend(fontsize=6)
    _save(fig, output, "P1B_08_path_residual_vs_Mixed_loss.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    compare = metrics[metrics.condition.isin((
        "D1_centered_logit_path", "D2_output_hidden_path", "D4_curvature_proxy"))]
    _bar(axes[0], compare, "Mixed_loss", "Held-out Mixed")
    _bar(axes[1], compare, "output_path_residual", "Output-path residual")
    _save(fig, output, "P1B_09_direct_path_vs_curvature_control.png")


def enforce_mandatory_timeseries(output: Path):
    missing = []
    for name in MANDATORY_TS:
        path = output / name
        text = output / name.replace(".png", ".txt")
        if not path.exists() and not text.exists():
            missing.append(name)
    if missing:
        raise RuntimeError("T13 failed: missing mandatory sequence-level figures: " + ", ".join(missing))
