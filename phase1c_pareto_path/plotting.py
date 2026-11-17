from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TS_NAMES = (
    "P1C_TS01_finetuning_sequence_loss.png",
    "P1C_TS02_finetuning_path_and_endpoint_metrics.png",
    "P1C_TS03_midpoint_over_finetuning.png",
    "P1C_TS04_full_familiar_sequence_loss.png",
    "P1C_TS05_full_familiar_sequence_accuracy.png",
    "P1C_TS06_full_familiar_correct_token_probability.png",
    "P1C_TS07_full_familiar_execution_proxy.png",
    "P1C_TS08_full_probe_sequence_loss.png",
    "P1C_TS09_full_probe_correct_token_probability.png",
    "P1C_TS10_full_probe_execution_proxy.png",
    "P1C_TS11_cue_sweep_sequence_loss.png",
    "P1C_TS12_cue_sweep_correct_token_probability.png",
    "P1C_TS13_cue_sweep_logit_path_residual.png",
    "P1C_TS14_cue_sweep_output_path_residual.png",
)


def _save(fig, output, name):
    if not any(ax.get_label() == "<colorbar>" for ax in fig.axes):
        fig.tight_layout()
    fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def _summary(frame, x, y):
    return frame.groupby(["plot_role", x], as_index=False)[y].agg(
        mean="mean", low=lambda s: s.quantile(.025), high=lambda s: s.quantile(.975))


def _lines(ax, frame, x, y):
    use = frame[frame.plot_role.notna()]
    summary = _summary(use, x, y)
    colors = plt.cm.tab10(np.linspace(0, .9, max(summary.plot_role.nunique(), 1)))
    for color, (role, group) in zip(colors, summary.groupby("plot_role", sort=False)):
        group = group.sort_values(x)
        ax.plot(group[x], group["mean"], label=role, color=color, lw=1.5)
        ax.fill_between(group[x], group.low, group.high, color=color, alpha=.14, linewidth=0)
    ax.set_xlabel(x.replace("_", " ")); ax.set_ylabel(y.replace("_", " "))


def _schedule_marks(ax, updates, probes=False):
    for block, start in enumerate(range(1, updates + 1, 10), start=1):
        stop = min(start + 9, updates); label = "A" if block % 2 else "B"
        color = "#4C78A8" if label == "A" else "#F58518"
        ax.axvspan(start - .5, stop + .5, color=color, alpha=.025)
        ax.text((start + stop) / 2, .985, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5, color=color)
    for value in range(10, updates, 10): ax.axvline(value + .5, color=".85", lw=.45)
    if probes: ax.axvline(updates + .5, color="black", ls="--", lw=.8)


def make_figures(output: Path, tables: dict, updates: int):
    output.mkdir(parents=True, exist_ok=True)
    metrics = tables["phase1c_seed_metrics"]
    specificity = tables["phase1c_specificity_controls"]
    feasible = tables["phase1c_feasible_frontier"]

    # Pareto figures.
    primary = metrics[metrics.family.isin(("P1_correct_logit", "P3_wrong_path"))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, family in zip(axes, ("P1_correct_logit", "P3_wrong_path")):
        sub = primary[primary.family.eq(family)]
        scatter = ax.scatter(sub.normalized_logit_path_residual, sub.Mixed_loss,
                             c=sub.endpoint_hidden_drift_ratio,
                             s=18 + 18 * np.log1p(sub.alpha), cmap="viridis", alpha=.7)
        ax.axvline(.35, color="black", ls="--", lw=.7); ax.set_title(family)
        ax.set_xlabel("normalized held-out logit path residual")
    axes[0].set_ylabel("held-out Mixed loss"); fig.colorbar(scatter, ax=axes, label="endpoint drift")
    _save(fig, output, "P1C_01_path_residual_vs_Mixed_loss.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(primary.endpoint_hidden_drift_ratio, primary.normalized_logit_path_residual,
                         c=primary.Mixed_loss, s=18 + 18 * np.log1p(primary.alpha), cmap="magma", alpha=.7)
    ax.axvline(.1, color="black", ls="--", lw=.7); ax.axhline(.35, color="black", ls="--", lw=.7)
    ax.set_xlabel("endpoint drift"); ax.set_ylabel("normalized logit path residual")
    fig.colorbar(scatter, ax=ax, label="Mixed loss")
    _save(fig, output, "P1C_02_endpoint_drift_vs_path_residual.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in primary.groupby("family"):
        ax.scatter(group.endpoint_hidden_drift_ratio, group.Mixed_loss, label=family, alpha=.65)
    ax.axvline(.1, color="black", ls="--", lw=.7); ax.set_xlabel("endpoint drift"); ax.set_ylabel("Mixed loss"); ax.legend(fontsize=7)
    _save(fig, output, "P1C_03_endpoint_drift_vs_Mixed_loss.png")

    exact = specificity[specificity.match_type.eq("exact_hyperparameter")]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(exact.Mixed_loss_correct, exact.Mixed_loss_wrong, c=np.log1p(exact.alpha), alpha=.7)
    limits = [min(exact.Mixed_loss_correct.min(), exact.Mixed_loss_wrong.min()),
              max(exact.Mixed_loss_correct.max(), exact.Mixed_loss_wrong.max())]
    ax.plot(limits, limits, color="black", ls="--", lw=.8); ax.set_xlabel("correct-path Mixed loss"); ax.set_ylabel("wrong-path Mixed loss")
    _save(fig, output, "P1C_04_correct_vs_wrong_path.png")

    nearest = specificity[specificity.match_type.eq("nearest_displacement_endpoint_familiar")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(nearest.actual_logit_change_norm_correct, nearest.actual_logit_change_norm_wrong,
                    c=nearest.matching_distance, alpha=.7)
    axes[0].set_xlabel("correct displacement"); axes[0].set_ylabel("matched wrong displacement")
    axes[1].hist(nearest.delta_Mixed_loss_wrong_minus_correct.dropna(), bins=20, color="#4C78A8")
    axes[1].axvline(0, color="black", ls="--"); axes[1].set_xlabel("wrong − correct Mixed loss")
    _save(fig, output, "P1C_05_displacement_matched_specificity.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in primary.groupby("family"):
        ax.scatter(group.normalized_logit_path_residual, group.rescue_fraction_per_seed, label=family, alpha=.65)
    ax.axhline(.75, color="black", ls="--", lw=.7); ax.set_xlabel("normalized logit path residual"); ax.set_ylabel("per-seed rescue fraction"); ax.legend(fontsize=7)
    _save(fig, output, "P1C_06_rescue_fraction_frontier.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in primary.groupby("family"):
        ax.scatter(group.normalized_logit_path_residual, group.remaining_patch_gap_ratio, label=family, alpha=.65)
    ax.axhline(.35, color="black", ls="--", lw=.7); ax.set_xlabel("normalized logit path residual"); ax.set_ylabel("remaining patch gap ratio"); ax.legend(fontsize=7)
    _save(fig, output, "P1C_07_remaining_patch_gap_frontier.png")

    matrix = feasible[feasible.family.eq("P1_correct_logit")].pivot_table(
        index="seed", columns="cell_id", values="feasible", aggfunc="max").fillna(False)
    fig, ax = plt.subplots(figsize=(max(8, .22 * matrix.shape[1]), 5))
    ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix.index)), matrix.index); ax.set_xlabel("P1 alpha × anchor cell"); ax.set_ylabel("seed")
    ax.set_xticks([])
    _save(fig, output, "P1C_08_feasible_region_by_seed.png")

    # Time-series figures.
    training = tables["phase1c_timeseries_finetuning"]
    midpoint = tables["phase1c_timeseries_midpoint_checkpoints"]
    familiar = tables["phase1c_timeseries_familiar_schedule"]
    full = tables["phase1c_timeseries_full_probe_schedule"]
    sweep = tables["phase1c_timeseries_cue_sweep"]
    fig, ax = plt.subplots(figsize=(11, 4.5)); _lines(ax, training, "update", "sequence_loss")
    _schedule_marks(ax, updates); ax.legend(fontsize=6, ncol=3); _save(fig, output, TS_NAMES[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, value in zip(axes, ("training_correct_logit_path_loss", "endpoint_hidden_drift_ratio", "AB_separation_ratio")):
        _lines(ax, training, "update", value); _schedule_marks(ax, updates)
    axes[0].legend(fontsize=6); _save(fig, output, TS_NAMES[1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, value in zip(axes.ravel(), (
            "heldout_midpoint_sequence_loss", "heldout_midpoint_logit_residual",
            "endpoint_hidden_drift_ratio", "remaining_patch_gap")):
        _lines(ax, midpoint, "update", value)
    axes[0, 0].legend(fontsize=6); _save(fig, output, TS_NAMES[2])

    for index, value in enumerate(("sequence_loss", "sequence_accuracy", "mean_correct_token_probability", "mean_execution_proxy"), start=3):
        if value == "mean_execution_proxy" and familiar[value].isna().all():
            (output / TS_NAMES[index].replace(".png", ".txt")).write_text("Execution proxy unavailable: frozen checkpoint has no policy.\n"); continue
        fig, ax = plt.subplots(figsize=(11, 4.5)); _lines(ax, familiar, "sequence_index", value)
        _schedule_marks(ax, updates); ax.legend(fontsize=6, ncol=3); _save(fig, output, TS_NAMES[index])
    for index, value in enumerate(("sequence_loss", "mean_correct_token_probability", "mean_execution_proxy"), start=7):
        if value == "mean_execution_proxy" and full[value].isna().all():
            (output / TS_NAMES[index].replace(".png", ".txt")).write_text("Execution proxy unavailable: frozen checkpoint has no policy.\n"); continue
        fig, ax = plt.subplots(figsize=(11, 4.5)); _lines(ax, full, "sequence_index", value)
        _schedule_marks(ax, updates, probes=True); ax.legend(fontsize=6, ncol=3); _save(fig, output, TS_NAMES[index])
    for index, value in enumerate(("sequence_loss", "mean_correct_token_probability", "logit_path_residual", "output_path_residual"), start=10):
        fig, ax = plt.subplots(figsize=(10, 4.5)); _lines(ax, sweep, "cue_lambda", value)
        ax.axvline(.5, color="black", ls="--", lw=.8); ax.legend(fontsize=6, ncol=3); _save(fig, output, TS_NAMES[index])


def enforce_timeseries(output: Path):
    missing = [name for name in TS_NAMES
               if not (output / name).exists() and not (output / name.replace(".png", ".txt")).exists()]
    if missing:
        raise RuntimeError("T12 failed: missing mandatory time-series artifacts: " + ", ".join(missing))
