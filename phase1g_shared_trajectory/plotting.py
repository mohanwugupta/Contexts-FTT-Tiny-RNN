"""Required Phase 1G figures, all on raw mean cross-entropy scales."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES = tuple(f"PHASE1G_FIG{i:02d}_{name}.png" for i, name in enumerate((
    "transfer_by_condition", "AB_separation_by_condition", "shared_trajectory_error",
    "shared_rollout_loss", "mixed_nocontext_novel_raw_loss", "novel_cue_generalization",
    "probe_geometry", "residual_patch_need"), 1))
TIMESERIES = tuple(f"PHASE1G_TS{i:02d}_{name}.png" for i, name in enumerate((
    "true_AB_acquisition_raw_loss", "final_AB_probe_raw_loss",
    "shared_free_rollout_raw_loss", "probe_transition_raw_loss",
    "AB_separation_over_training"), 1))
CONDITION_ORDER = ["G0_baseline_high", "G1_context_dropout", "G2_shared_prediction",
                   "G3_shared_trajectory", "G4_prediction_plus_trajectory",
                   "G5_dropout_plus_trajectory", "Low_reference"]


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _bar(frame, x, y, path, title, hue=None):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    observed = list(frame[x].drop_duplicates())
    categories = ([value for value in CONDITION_ORDER if value in observed]
                  if x == "condition" else observed)
    if hue:
        groups = list(frame[hue].drop_duplicates()); width = .8/max(1, len(groups))
        for j, group in enumerate(groups):
            values = frame[frame[hue].eq(group)].groupby(x)[y].mean().reindex(categories)
            xx = np.arange(len(categories))+(j-(len(groups)-1)/2)*width
            ax.bar(xx, values, width, label=group)
        ax.legend(fontsize=7)
    else:
        values = frame.groupby(x)[y].mean().reindex(categories)
        ax.bar(np.arange(len(categories)), values)
    ax.set_xticks(np.arange(len(categories)), categories, rotation=30, ha="right")
    ax.set_ylabel(y.replace("_", " ")); ax.set_title(title); _save(fig, path)


def make_figures(output, training, probe_summary, probes, geometry, rollout, causal, seed_summary):
    output = Path(output); figures = output/"figures"; timeseries = output/"timeseries"
    figures.mkdir(parents=True, exist_ok=True); timeseries.mkdir(parents=True, exist_ok=True)
    interventions = seed_summary[~seed_summary.condition.eq("Low_reference")]
    transfer = interventions.melt(id_vars=["condition", "seed"],
        value_vars=["Mixed_improvement", "NoContext_improvement", "NovelMean_improvement"],
        var_name="probe", value_name="improvement")
    _bar(transfer, "condition", "improvement", figures/FIGURES[0],
         "Held-out transfer improvement relative to matched G0", "probe")
    sep = interventions.melt(id_vars=["condition", "seed"],
        value_vars=["AB_hidden_separation_ratio", "AB_task_separation_ratio"],
        var_name="space", value_name="ratio")
    _bar(sep, "condition", "ratio", figures/FIGURES[1], "A/B separation retained", "space")
    _bar(interventions, "condition", "shared_trajectory_error", figures/FIGURES[2],
         "One-step shared-trajectory error")
    _bar(interventions, "condition", "shared_rollout_loss", figures/FIGURES[3],
         "Shared free-rollout raw loss")
    focus = probe_summary[probe_summary.probe.isin(["Mixed", "NoContext", "NovelMean"])]
    _bar(focus, "condition", "raw_sequence_loss", figures/FIGURES[4],
         "Mixed, NoContext, and Novel raw loss", "probe")
    novel = probe_summary[probe_summary.probe.str.startswith("Novel") & ~probe_summary.probe.eq("NovelMean")]
    _bar(novel, "condition", "raw_sequence_loss", figures/FIGURES[5],
         "Generalization across preregistered Novel cues", "probe")
    task = geometry[geometry.space.eq("task")]
    _bar(task, "condition", "normalized_off_axis_norm", figures/FIGURES[6],
         "Probe geometry in centered task space", "probe_family")
    shared_causal = causal[(causal.intervention.eq("shared_state")) & causal.patch_mode.eq("transition_wise")]
    _bar(shared_causal, "probe", "residual_rescue_need", figures/FIGURES[7],
         "Residual shared-patch rescue need", "space")

    # TS1: losses captured before every real optimizer update.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, (condition, group) in zip(axes.flat, training[~training.condition.eq("Low_reference")].groupby("condition")):
        mean = group.groupby(["training_sequence", "context"]).raw_sequence_loss_pre_update.mean().reset_index()
        for context, data in mean.groupby("context"):
            ax.plot(data.training_sequence, data.raw_sequence_loss_pre_update, label=context, lw=1)
        ax.set_title(condition); ax.set_xlabel("Actual training sequence"); ax.set_ylabel("Raw CE"); ax.legend(fontsize=7)
    _save(fig, timeseries/TIMESERIES[0])

    _bar(probe_summary, "condition", "raw_sequence_loss", timeseries/TIMESERIES[1],
         "Final-checkpoint familiar and held-out raw loss", "probe")

    free = rollout[rollout["mode"].eq("free_rollout") & ~rollout.condition.eq("Low_reference")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition, group in free.groupby("condition"):
        mean = group.groupby("transition").shared_rollout_raw_loss.mean()
        ax.plot(mean.index, mean.values, marker="o", label=condition)
    ax.set(xlabel="Transition", ylabel="Raw CE", title="Shared-state free rollout"); ax.legend(fontsize=7)
    _save(fig, timeseries/TIMESERIES[2])

    focus = probes[probes.probe_family.isin(["Mixed", "NoContext", "Novel"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, family in zip(axes, ("Mixed", "NoContext", "Novel")):
        data = focus[focus.probe_family.eq(family)].groupby(["condition", "transition"]).raw_transition_loss.mean().reset_index()
        for condition, group in data.groupby("condition"):
            ax.plot(group.transition, group.raw_transition_loss, marker="o", label=condition)
        refs = probes[probes.probe.isin(["A", "B"])].groupby(["probe", "transition"]).raw_transition_loss.mean().reset_index()
        for reference, group in refs.groupby("probe"):
            ax.plot(group.transition, group.raw_transition_loss, linestyle="--", color="black", alpha=.5, label=f"{reference} ref")
        ax.set_title(family); ax.set_xlabel("Transition"); ax.set_ylabel("Raw CE")
    axes[-1].legend(fontsize=6); _save(fig, timeseries/TIMESERIES[3])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, value, label in zip(axes, ("AB_hidden_separation", "AB_task_separation"), ("Hidden", "Task")):
        for condition, group in training[~training.condition.eq("Low_reference")].groupby("condition"):
            mean = group.groupby("training_sequence")[value].mean()
            ax.plot(mean.index, mean.values, label=condition)
        ax.set(xlabel="Actual training sequence", ylabel="A/B distance", title=f"{label}-space A/B separation")
    axes[-1].legend(fontsize=6); _save(fig, timeseries/TIMESERIES[4])


def enforce_outputs(output):
    output = Path(output)
    paths = [output/"figures"/name for name in FIGURES] + [output/"timeseries"/name for name in TIMESERIES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing: raise RuntimeError("Missing Phase 1G figures:\n- " + "\n- ".join(missing))
