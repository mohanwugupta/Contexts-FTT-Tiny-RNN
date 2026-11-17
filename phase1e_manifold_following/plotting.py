from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = (
    "PHASE1E_FIG01_success_rate.png",
    "PHASE1E_FIG02_1D_vs_1E_mixed_loss.png",
    "PHASE1E_FIG03_correct_vs_wrong.png",
    "PHASE1E_FIG04_failure_modes.png",
    "PHASE1E_FIG05_curvature_vs_rescue.png",
)
TIMESERIES = (
    "PHASE1E_TS01_raw_familiar_loss_all_sequences.png",
    "PHASE1E_TS02_raw_full_timeline_with_probes.png",
    "PHASE1E_TS03_probe_loss_vs_A_B.png",
    "PHASE1E_TS04_matched_checkpoint_A_B_Mixed_Novel.png",
    "PHASE1E_TS05_optimizer_path_loss.png",
    "PHASE1E_TS06_constraint_violation.png",
    "PHASE1E_TS07_trust_radius.png",
    "PHASE1E_TS08_tangent_space_rotation.png",
)


COLORS = {
    "Baseline High": "#222222", "Baseline Low": "#999999",
    "Phase 1C soft penalty": "#4C78A8", "Phase 1D constrained": "#F58518",
    "Phase 1E correct": "#54A24B", "Phase 1E wrong": "#E45756",
}


def raw_loss_plot_data(frame, loss_column="raw_sequence_loss"):
    """Return plotting rows without transforming raw loss units."""
    result = frame.copy()
    result[loss_column] = frame[loss_column].to_numpy(copy=True)
    return result


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _mean_band(ax, frame, x, y, label, color=None):
    frame = frame[np.isfinite(pd.to_numeric(frame[y], errors="coerce"))]
    if frame.empty:
        return
    grouped = frame.groupby(x)[y].agg(["mean", "sem"]).reset_index()
    ax.plot(grouped[x], grouped["mean"], label=label, color=color, lw=1.8)
    ax.fill_between(grouped[x], grouped["mean"] - grouped["sem"].fillna(0),
                    grouped["mean"] + grouped["sem"].fillna(0),
                    color=color, alpha=.15)


def make_primary(output, seed, rates):
    directory = Path(output) / "figures"; directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rates))
    ax.bar(x, rates.success_fraction, color=[COLORS.get(x, "#777777") for x in rates.method])
    ax.errorbar(x, rates.success_fraction,
                yerr=[rates.success_fraction-rates.wilson_low,
                      rates.wilson_high-rates.success_fraction], fmt="none", color="black", capsize=3)
    ax.set_xticks(x, rates.method, rotation=20, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("Complete-gate success fraction")
    _save(fig, directory / PRIMARY[0])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(seed.phase1d_mixed_loss, seed.phase1e_mixed_loss, c="#54A24B")
    low = min(seed.phase1d_mixed_loss.min(), seed.phase1e_mixed_loss.min())
    high = max(seed.phase1d_mixed_loss.max(), seed.phase1e_mixed_loss.max())
    ax.plot([low, high], [low, high], "k--", lw=1)
    ax.set(xlabel="Phase 1D Mixed raw loss", ylabel="Phase 1E Mixed raw loss")
    _save(fig, directory / PRIMARY[1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(seed.phase1e_mixed_loss, seed.phase1e_wrong_mixed_loss, c="#E45756")
    low = min(seed.phase1e_mixed_loss.min(), seed.phase1e_wrong_mixed_loss.min())
    high = max(seed.phase1e_mixed_loss.max(), seed.phase1e_wrong_mixed_loss.max())
    ax.plot([low, high], [low, high], "k--", lw=1)
    ax.set(xlabel="Correct-path Mixed raw loss", ylabel="Wrong-path Mixed raw loss")
    _save(fig, directory / PRIMARY[2])

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = seed.failure_category.value_counts()
    ax.bar(np.arange(len(counts)), counts, color="#4C78A8")
    ax.set_xticks(np.arange(len(counts)), counts.index, rotation=25, ha="right")
    ax.set_ylabel("Seeds")
    _save(fig, directory / PRIMARY[3])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(seed.Q_local, seed.phase1d_minus_phase1e_mixed_loss, c="#4C78A8")
    axes[0].set(xlabel="$Q_{local}$", ylabel="Phase 1D − Phase 1E Mixed loss")
    axes[1].scatter(seed.mean_tangent_space_rotation,
                    seed.phase1d_minus_phase1e_mixed_loss, c="#F58518")
    axes[1].set(xlabel="Mean tangent-projector rotation",
                ylabel="Phase 1D − Phase 1E Mixed loss")
    _save(fig, directory / PRIMARY[4])


def make_timeseries(output, familiar, full, probes, trace, updates):
    directory = Path(output) / "timeseries"; directory.mkdir(parents=True, exist_ok=True)
    familiar = raw_loss_plot_data(familiar, "sequence_loss")
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, group in familiar.groupby("method"):
        _mean_band(ax, group, "sequence_index", "sequence_loss", method, COLORS.get(method))
    schedule = familiar[["sequence_index", "context"]].drop_duplicates("sequence_index")
    for _, group in schedule.groupby((schedule.context != schedule.context.shift()).cumsum()):
        ax.axvspan(group.sequence_index.min(), group.sequence_index.max(),
                   color="#d9e8fb" if group.context.iloc[0] == "A" else "#fde0d9", alpha=.12)
    ax.set(xlabel="Sequence index", ylabel="Raw sequence loss")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, directory / TIMESERIES[0])

    fig, ax = plt.subplots(figsize=(12, 5))
    familiar_full = full[~full.is_probe]
    for method, group in familiar_full.groupby("method"):
        _mean_band(ax, group, "sequence_index", "sequence_loss", method, COLORS.get(method))
    markers = {"Mixed": "o", "Novel": "^", "NoContextProbe": "s"}
    phase1e = full[full.method.eq("Phase 1E correct") & full.is_probe]
    for context, marker in markers.items():
        group = phase1e[phase1e.context.eq(context)].groupby("sequence_index").sequence_loss.mean()
        ax.scatter(group.index, group.values, marker=marker, s=55, label=f"{context} probe")
    ax.set(xlabel="Actual chronological sequence index", ylabel="Raw sequence loss")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, directory / TIMESERIES[1])

    plot_probes = raw_loss_plot_data(probes)
    fig, ax = plt.subplots(figsize=(11, 5))
    selected = plot_probes[plot_probes.condition.eq("Phase 1E correct")]
    styles = {"A": "-", "B": "-", "Mixed": "--", "Novel": ":", "NoContextProbe": "-."}
    for context, group in selected.groupby("context"):
        means = group.groupby("probe_checkpoint").raw_sequence_loss.mean()
        ax.plot(means.index, means.values, styles.get(context, "-"), marker="o", label=context)
    ax.set(xlabel="Registered probe checkpoint", ylabel="Raw sequence loss")
    ax.legend(ncol=5, fontsize=8)
    _save(fig, directory / TIMESERIES[2])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    methods = ("Baseline High", "Phase 1D constrained", "Phase 1E correct")
    for ax, method in zip(axes, methods):
        group = plot_probes[plot_probes.condition.eq(method)]
        means = group.groupby("context").raw_sequence_loss.mean().reindex(
            ["A", "B", "Mixed", "Novel", "NoContextProbe"])
        ax.plot(np.arange(len(means)), means, marker="o")
        ax.set_xticks(np.arange(len(means)), means.index, rotation=35, ha="right")
        ax.set_title(method); ax.set_ylabel("Raw sequence loss")
    _save(fig, directory / TIMESERIES[3])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for condition, group in trace.groupby("condition"):
        color = COLORS["Phase 1E correct" if condition.endswith("correct") else "Phase 1E wrong"]
        _mean_band(axes[0, 0], group, "update", "raw_training_path_loss", condition, color)
        _mean_band(axes[0, 1], group, "update", "raw_familiar_loss", condition, color)
        _mean_band(axes[1, 0], group, "update", "improvement_ratio", condition, color)
        _mean_band(axes[1, 1], group, "update", "normal_step_norm", condition + " normal", color)
        tangent = group.copy(); tangent["tangent_step_norm_plot"] = tangent.tangent_step_norm
        _mean_band(axes[1, 1], tangent, "update", "tangent_step_norm_plot",
                   condition + " tangent", color)
    axes[0, 0].set_ylabel("Raw training-path loss")
    axes[0, 1].set_ylabel("Raw familiar loss")
    axes[1, 0].set_ylabel("Actual / predicted improvement")
    axes[1, 0].axhline(.25, color="black", ls="--", lw=.8)
    axes[1, 0].axhline(.75, color="black", ls=":", lw=.8)
    axes[1, 1].set_ylabel("Normal / tangent step norm")
    for ax in axes.flat:
        ax.set_xlabel("Attempted optimizer update")
    axes[0, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=6, ncol=2)
    _save(fig, directory / TIMESERIES[4])

    diagnostics = (
        ("scaled_constraint_norm", "Scaled endpoint constraint norm", TIMESERIES[5]),
        ("trust_radius", "Trust radius", TIMESERIES[6]),
        ("tangent_projector_rotation", "Tangent-projector rotation", TIMESERIES[7]),
    )
    for column, ylabel, filename in diagnostics:
        fig, ax = plt.subplots(figsize=(9, 4))
        for condition, group in trace.groupby("condition"):
            _mean_band(ax, group, "update", column, condition,
                       COLORS["Phase 1E correct" if condition.endswith("correct") else "Phase 1E wrong"])
        ax.set(xlabel="Attempted optimizer update", ylabel=ylabel)
        ax.legend(fontsize=8)
        _save(fig, directory / filename)


def enforce_outputs(output):
    output = Path(output)
    missing = [str(output / folder / name)
               for folder, names in (("figures", PRIMARY), ("timeseries", TIMESERIES))
               for name in names if not (output / folder / name).exists()]
    if missing:
        raise RuntimeError("Mandatory Phase 1E figures missing:\n- " + "\n- ".join(missing))
