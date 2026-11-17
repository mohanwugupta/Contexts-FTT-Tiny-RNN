from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURES = (
    "fig_A_true_acquisition.png", "fig_B_acquisition_with_probes.png",
    "fig_C_probe_checkpoint_losses.png", "fig_D_frozen_checkpoint_diagnostic.png",
    "fig_E_phase1e_retrofit_losses.png", "fig_F_checkpoint_provenance.png",
)
CONTEXT_COLORS = {"A": "#4C78A8", "B": "#F58518", "Mixed": "#54A24B",
                  "Novel": "#B279A2", "NoContextProbe": "#E45756"}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _mean_sem(frame, x, y, by=None):
    keys = [x] if by is None else [by, x]
    return frame.groupby(keys)[y].agg(["mean", "sem"]).reset_index()


def _block_shading(ax, schedule):
    blocks = schedule[["block_index", "context", "sequence_index"]].drop_duplicates()
    for block, group in blocks.groupby("block_index"):
        ax.axvspan(group.sequence_index.min() - .5, group.sequence_index.max() + .5,
                   color=CONTEXT_COLORS[group.context.iloc[0]], alpha=.055)


def make_figures(output, training, probes, frozen, phase1e, provenance_distance):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summary = _mean_sem(training, "sequence_index", "raw_sequence_loss_pre_update", "context")

    fig, ax = plt.subplots(figsize=(12, 5))
    for context, group in summary.groupby("context"):
        ax.plot(group.sequence_index, group["mean"], color=CONTEXT_COLORS[context],
                label=f"{context} observed pre-update", lw=1.8)
        ax.fill_between(group.sequence_index, group["mean"]-group["sem"].fillna(0),
                        group["mean"]+group["sem"].fillna(0), color=CONTEXT_COLORS[context], alpha=.18)
    _block_shading(ax, training); ax.legend()
    ax.set(xlabel="External training sequence index", ylabel="Raw sequence loss (mean CE)",
           title="True original acquisition — time-matched parameter states")
    _save(fig, output / FIGURES[0])

    fig, ax = plt.subplots(figsize=(12, 5))
    for context, group in summary.groupby("context"):
        ax.plot(group.sequence_index, group["mean"], color=CONTEXT_COLORS[context],
                label=f"{context} observed", lw=1.5)
    probe_summary = _mean_sem(probes, "training_checkpoint", "raw_sequence_loss", "condition")
    for context in ("Mixed", "Novel", "NoContextProbe"):
        group = probe_summary[probe_summary.condition.eq(context)]
        ax.plot(group.training_checkpoint, group["mean"], marker="o", ms=3,
                color=CONTEXT_COLORS[context], ls="--", label=f"{context} frozen probe")
    _block_shading(ax, training); ax.legend(ncol=2, fontsize=8)
    ax.set(xlabel="External training sequence index / matched checkpoint",
           ylabel="Raw sequence loss (mean CE)", title="Acquisition with read-only matched probes")
    _save(fig, output / FIGURES[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    for context, group in probe_summary.groupby("condition"):
        ax.plot(group.training_checkpoint, group["mean"], marker="o", ms=3,
                color=CONTEXT_COLORS[context], label=context)
        ax.fill_between(group.training_checkpoint, group["mean"]-group["sem"].fillna(0),
                        group["mean"]+group["sem"].fillna(0), color=CONTEXT_COLORS[context], alpha=.12)
    ax.legend(ncol=5, fontsize=8)
    ax.set(xlabel="Registered training checkpoint", ylabel="Raw sequence loss (mean CE)",
           title="Frozen probes during original A/B acquisition")
    _save(fig, output / FIGURES[2])

    frozen_summary = _mean_sem(frozen, "sequence_index", "raw_sequence_loss", "context")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for context, group in frozen_summary.groupby("context"):
        ax.plot(group.sequence_index, group["mean"], color=CONTEXT_COLORS[context],
                label=context, lw=1.7)
    _block_shading(ax, frozen); ax.legend()
    ax.set(xlabel="Historical schedule position", ylabel="Raw sequence loss (mean CE)",
           title="Frozen final-checkpoint evaluation — not an acquisition trajectory")
    _save(fig, output / FIGURES[3])

    fig, ax = plt.subplots(figsize=(10, 5))
    phase = phase1e[phase1e.condition.eq("Phase 1E correct")].copy()
    phase_summary = _mean_sem(phase, "probe_checkpoint", "raw_sequence_loss", "context")
    for context, group in phase_summary.groupby("context"):
        ax.plot(group.probe_checkpoint, group["mean"], marker="o", ms=3,
                color=CONTEXT_COLORS[context], label=context)
    ax.legend(ncol=5, fontsize=8)
    ax.set(xlabel="Phase 1E optimizer update", ylabel="Raw sequence loss (mean CE)",
           title="Phase 1E retrofit optimization — not original acquisition")
    _save(fig, output / FIGURES[4])

    order = ["Original final", "Phase 1D selected", "Phase 1E start", "Phase 1E end"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, group in provenance_distance.groupby("seed"):
        values = group.set_index("stage").reindex(order).parameter_L2_distance_to_original
        ax.plot(np.arange(len(order)), values, color="#777777", alpha=.35, marker="o", ms=3)
    means = provenance_distance.groupby("stage").parameter_L2_distance_to_original.mean().reindex(order)
    ax.plot(np.arange(len(order)), means, color="#111111", lw=2.5, marker="o", label="Mean")
    ax.set_xticks(np.arange(len(order)), order, rotation=18, ha="right")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_ylabel("Parameter L2 distance from original final checkpoint")
    ax.set_title("Checkpoint provenance and state distance"); ax.legend()
    _save(fig, output / FIGURES[5])


def enforce_figures(output):
    missing = [name for name in FIGURES if not (Path(output) / name).exists()]
    if missing:
        raise RuntimeError("Missing timeline audit figures: " + ", ".join(missing))
