from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = tuple(f"PHASE1F_FIG{i:02d}_{name}.png" for i, name in enumerate((
    "probe_alpha_by_condition", "off_axis_residual_by_probe",
    "sequence_specificity_margin", "shared_patch_rescue", "axis_projection_rescue",
    "correct_vs_wrong_sequence_patch", "correct_vs_orthogonal_control",
    "high_vs_low_geometry", "hidden_vs_task_space_rescue"), start=1))
TS = tuple(f"PHASE1F_TS{i:02d}_{name}.png" for i, name in enumerate((
    "true_A_B_acquisition_raw_loss", "baseline_A_B_Mixed_Novel_NoContext_raw_loss",
    "shared_patch_raw_loss", "axis_projection_raw_loss",
    "correct_vs_wrong_sequence_raw_loss"), start=1))
COLORS = {"High_monotonic": "#E45756", "Low_monotonic": "#4C78A8"}
PROBE_COLORS = {"Mixed": "#54A24B", "Novel": "#B279A2", "NoContextProbe": "#F58518",
                "A": "#4C78A8", "B": "#E45756"}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _bar(frame, value, path, title, ylabel, hue="scale_condition"):
    groups = list(frame[hue].drop_duplicates())
    categories = list(frame.probe.drop_duplicates())
    fig, ax = plt.subplots(figsize=(8, 4.5)); width = .8 / max(len(groups), 1)
    for index, group_name in enumerate(groups):
        values = frame[frame[hue].eq(group_name)].groupby("probe")[value].mean().reindex(categories)
        x = np.arange(len(categories)) + (index-(len(groups)-1)/2)*width
        ax.bar(x, values, width=width, label=group_name, color=COLORS.get(group_name))
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(np.arange(len(categories)), categories)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
    _save(fig, path)


def make_figures(output, task_geometry, hidden_geometry, specificity, task_raw,
                 hidden_raw, seed_summary, acquisition, frozen_states):
    output = Path(output); figures = output / "figures"; timeseries = output / "timeseries"
    figures.mkdir(parents=True, exist_ok=True); timeseries.mkdir(parents=True, exist_ok=True)
    _bar(task_geometry, "alpha", figures/PRIMARY[0], "Probe coordinate on familiar A/B task axis", "Alpha")
    _bar(task_geometry, "normalized_off_axis_norm", figures/PRIMARY[1],
         "Probe off-axis residual in centered-logit space", "Normalized off-axis norm")
    _bar(specificity[specificity.space.eq("task")], "sequence_specificity_margin",
         figures/PRIMARY[2], "Correct-position shared-state specificity", "Wrong mean − correct distance")

    task_shared = task_raw[task_raw.intervention.eq("B1_shared")]
    _bar(task_shared, "raw_rescue", figures/PRIMARY[3], "Shared-midpoint task-state rescue", "Baseline − patched raw loss")
    task_axis = task_raw[task_raw.intervention.eq("B2_axis")]
    _bar(task_axis, "raw_rescue", figures/PRIMARY[4], "A/B-axis projection rescue", "Baseline − patched raw loss")

    compare = task_raw[task_raw.intervention.isin(["B1_shared", "C1_wrong_shared"])].copy()
    compare["probe_original"] = compare.probe; compare["probe"] = compare.probe + "\n" + compare.intervention
    _bar(compare, "raw_rescue", figures/PRIMARY[5], "Correct versus wrong-position shared patch",
         "Baseline − patched raw loss")
    orth = task_raw[task_raw.intervention.isin(["B1_shared", "C3_orthogonal_shared"])].copy()
    orth["probe"] = orth.probe + "\n" + orth.intervention
    _bar(orth, "raw_rescue", figures/PRIMARY[6], "Shared patch versus norm-matched orthogonal movement",
         "Baseline − patched raw loss")

    high_low = task_geometry.groupby(["scale_condition", "probe"], as_index=False).agg(
        normalized_off_axis_norm=("normalized_off_axis_norm", "mean"),
        AB_modulation_norm=("AB_modulation_norm", "mean"))
    fig, ax = plt.subplots(figsize=(6, 5))
    for condition, group in high_low.groupby("scale_condition"):
        ax.scatter(group.AB_modulation_norm, group.normalized_off_axis_norm,
                   color=COLORS[condition], label=condition, s=55)
        for row in group.itertuples(): ax.annotate(row.probe, (row.AB_modulation_norm, row.normalized_off_axis_norm), fontsize=7)
    ax.set(xlabel="A/B modulation norm", ylabel="Normalized off-axis residual",
           title="High versus Low shared-backbone geometry"); ax.legend()
    _save(fig, figures/PRIMARY[7])

    task_lookup = task_shared.set_index(["seed", "scale_condition", "probe"]).raw_rescue
    hidden = hidden_raw[hidden_raw.intervention.eq("D1_shared_transition")].copy()
    hidden["task_rescue"] = [task_lookup.get((r.seed, r.scale_condition, r.probe), np.nan)
                             for r in hidden.itertuples()]
    fig, ax = plt.subplots(figsize=(6, 5))
    for condition, group in hidden.groupby("scale_condition"):
        ax.scatter(group.task_rescue, group.raw_rescue, color=COLORS[condition], label=condition)
    low = np.nanmin([hidden.task_rescue.min(), hidden.raw_rescue.min()]); high = np.nanmax([hidden.task_rescue.max(), hidden.raw_rescue.max()])
    ax.plot([low, high], [low, high], "k--", lw=.8)
    ax.set(xlabel="Task shared-patch rescue", ylabel="Hidden shared-patch rescue",
           title="Hidden versus task-space state sufficiency"); ax.legend()
    _save(fig, figures/PRIMARY[8])

    # TS01: audited true acquisition, never a replay of the final checkpoint.
    summary = acquisition.groupby(["sequence_index", "context"]).raw_sequence_loss_pre_update.agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for context, group in summary.groupby("context"):
        ax.plot(group.sequence_index, group["mean"], color=PROBE_COLORS[context], label=context)
        ax.fill_between(group.sequence_index, group["mean"]-group["sem"].fillna(0),
                        group["mean"]+group["sem"].fillna(0), color=PROBE_COLORS[context], alpha=.15)
    ax.set(xlabel="Actual training sequence index", ylabel="Raw sequence loss (mean CE)",
           title="True canonical A/B acquisition"); ax.legend(); _save(fig, timeseries/TS[0])

    # TS02 transition-resolved frozen baseline.
    baseline = frozen_states.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, condition in zip(axes, ("High_monotonic", "Low_monotonic")):
        data = baseline[baseline.scale_condition.eq(condition)].groupby(
            ["probe_type", "transition"]).raw_transition_loss.mean().reset_index()
        for probe, group in data.groupby("probe_type"):
            ax.plot(group.transition, group.raw_transition_loss, marker="o", color=PROBE_COLORS[probe], label=probe)
        ax.set_title(condition); ax.set_xlabel("Transition"); ax.set_ylabel("Raw transition CE")
    axes[1].legend(fontsize=7); _save(fig, timeseries/TS[1])

    reference_losses = frozen_states[frozen_states.probe_type.isin(["A", "B"])].groupby(
        ["scale_condition", "probe_type"]).raw_sequence_loss.mean()

    def patch_ts(interventions, filename, title):
        data = pd.concat([task_raw.assign(source="task"), hidden_raw.assign(source="hidden")])
        data = data[data.intervention.isin(interventions)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
        for row_index, condition in enumerate(("High_monotonic", "Low_monotonic")):
            for column_index, probe in enumerate(("Mixed", "Novel", "NoContextProbe")):
                ax = axes[row_index, column_index]
                group = data[data.probe.eq(probe) & data.scale_condition.eq(condition)].groupby(
                    ["source", "intervention"]).raw_loss.mean()
                labels = ["A reference", "B reference"] + [" / ".join(x) for x in group.index]
                values = [reference_losses.loc[(condition, "A")],
                          reference_losses.loc[(condition, "B")], *group.values]
                colors = [PROBE_COLORS["A"], PROBE_COLORS["B"]] + [PROBE_COLORS[probe]]*len(group)
                ax.bar(np.arange(len(values)), values, color=colors)
                ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right", fontsize=6)
                ax.set_title(f"{condition} — {probe}"); ax.set_ylabel("Raw sequence loss")
        fig.suptitle(title); _save(fig, timeseries/filename)
    patch_ts(["B0_baseline", "B1_shared", "D0_baseline_transition", "D1_shared_transition"],
             TS[2], "Shared-state patch raw loss")
    patch_ts(["B0_baseline", "B2_axis", "D0_baseline_transition", "D2_axis_transition"],
             TS[3], "A/B-axis projection raw loss")
    patch_ts(["B0_baseline", "B1_shared", "C1_wrong_shared"], TS[4],
             "Correct versus wrong-position shared patch")


def enforce_outputs(output):
    output = Path(output)
    missing = [str(output/folder/name) for folder, names in (("figures", PRIMARY), ("timeseries", TS))
               for name in names if not (output/folder/name).exists()]
    if missing: raise RuntimeError("Missing Phase 1F figures:\n- " + "\n- ".join(missing))
