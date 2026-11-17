from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_TS = (
    "FINAL_TS01_full_familiar_sequence_loss.png",
    "FINAL_TS02_full_familiar_correct_token_probability.png",
    "FINAL_TS03_full_familiar_execution_proxy.png",
    "FINAL_TS04_full_probe_sequence_loss.png",
    "FINAL_TS05_full_probe_correct_token_probability.png",
    "FINAL_TS06_full_probe_execution_proxy.png",
    "FINAL_TS07_midpoint_over_finetuning.png",
    "FINAL_TS08_cue_sweep_sequence_loss.png",
    "FINAL_TS09_cue_sweep_task_path_residual.png",
)


def _save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    if not any(ax.get_label() == "<colorbar>" for ax in fig.axes): fig.tight_layout()
    fig.savefig(directory / name, dpi=220, bbox_inches="tight"); plt.close(fig)


def _line(ax, frame, x, y):
    summary = frame.groupby(["final_role", x], as_index=False)[y].agg(
        mean="mean", low=lambda s: s.quantile(.025), high=lambda s: s.quantile(.975))
    colors = plt.cm.tab10(np.linspace(0, .9, summary.final_role.nunique()))
    for color, (role, group) in zip(colors, summary.groupby("final_role", sort=False)):
        group = group.sort_values(x)
        ax.plot(group[x], group["mean"], label=role, color=color, lw=1.7)
        ax.fill_between(group[x], group.low, group.high, color=color, alpha=.15, linewidth=0)
    ax.set_xlabel(x.replace("_", " ")); ax.set_ylabel(y.replace("_", " "))


def _schedule(ax, updates=170, probes=False):
    for block, start in enumerate(range(1, updates + 1, 10), start=1):
        stop = min(start + 9, updates); name = "A" if block % 2 else "B"
        color = "#4C78A8" if name == "A" else "#F58518"
        ax.axvspan(start - .5, stop + .5, color=color, alpha=.025)
        ax.text((start + stop) / 2, .985, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5, color=color)
    for value in range(10, updates, 10): ax.axvline(value + .5, color=".85", lw=.45)
    if probes: ax.axvline(updates + .5, color="black", ls="--", lw=.8)


def make_primary_figures(output, master, specificity, seed_summary, formal_table, aggregate_cell):
    figures = Path(output) / "figures"
    primary = master[master.family.isin(("P1_correct_logit", "P3_wrong_path"))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for family, marker, label in (("P1_correct_logit", "o", "correct path"),
                                  ("P3_wrong_path", "x", "wrong path")):
        sub = primary[primary.family.eq(family)]
        axes[0].scatter(sub.normalized_logit_path_residual, sub.Mixed_loss,
                        s=12, alpha=.35, marker=marker, label=label)
    axes[0].set_xlabel("normalized task-path residual"); axes[0].set_ylabel("Mixed sequence loss"); axes[0].legend()
    correct = primary[primary.family.eq("P1_correct_logit")]
    axes[1].scatter(correct.normalized_logit_path_residual, correct.AB_separation_ratio,
                    c=correct.endpoint_hidden_drift_ratio, s=12, alpha=.45, cmap="viridis")
    for y in (.9, 1., 1.1): axes[1].axhline(y, color="black", ls="--" if y != 1 else "-", lw=.7)
    axes[1].set_xlabel("normalized task-path residual"); axes[1].set_ylabel("A/B separation ratio")
    scatter = axes[2].scatter(correct.normalized_logit_path_residual,
                              correct.endpoint_hidden_drift_ratio, c=correct.Mixed_loss,
                              s=12, alpha=.45, cmap="magma")
    axes[2].axhline(.1, color="black", ls="--", lw=.8)
    axes[2].set_xlabel("normalized task-path residual"); axes[2].set_ylabel("endpoint hidden drift")
    fig.colorbar(scatter, ax=axes[2], label="Mixed loss")
    _save(fig, figures, "FINAL_FIG01_mechanism_triangle.png")

    exact = specificity[specificity.match_type.eq("exact_hyperparameter")]
    delta = exact.delta_Mixed_loss_wrong_minus_correct
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].scatter(exact.Mixed_loss_correct, exact.Mixed_loss_wrong, s=14, alpha=.4)
    limits = [min(exact.Mixed_loss_correct.min(), exact.Mixed_loss_wrong.min()),
              max(exact.Mixed_loss_correct.max(), exact.Mixed_loss_wrong.max())]
    axes[0].plot(limits, limits, color="black", ls="--", lw=.8)
    axes[0].set_xlabel("correct-path Mixed loss"); axes[0].set_ylabel("matched wrong-path Mixed loss")
    axes[1].hist(delta, bins=30, color="#4C78A8", alpha=.85); axes[1].axvline(0, color="black", ls="--")
    axes[1].set_xlabel("wrong − correct Mixed loss"); axes[1].set_ylabel("matched seed-cells")
    _save(fig, figures, "FINAL_FIG02_correct_vs_wrong_path_specificity.png")

    ordered = seed_summary.sort_values("normalized_task_path_residual").reset_index(drop=True)
    x = np.arange(len(ordered)); labels = ordered.seed.astype(int).astype(str)
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    panels = (("normalized_task_path_residual", .35), ("per_seed_rescue_fraction", .75),
              ("endpoint_hidden_drift", .10), ("remaining_patch_gap_ratio", .35))
    for ax, (column, threshold) in zip(axes[:4], panels):
        ax.plot(x, ordered[column], marker="o", lw=1.2)
        ax.axhline(threshold, color="black", ls="--", lw=.7); ax.set_ylabel(column.replace("_", " "))
    axes[4].bar(x, ordered.passes_full_feasible_gate.astype(int),
                color=np.where(ordered.passes_full_feasible_gate, "#54A24B", "#BAB0AC"))
    axes[4].set_ylabel("feasible"); axes[4].set_yticks([0, 1]); axes[4].set_xticks(x, labels)
    axes[4].set_xlabel("seed, ordered by selected stable path residual")
    _save(fig, figures, "FINAL_FIG03_seed_level_sufficiency.png")

    high = formal_table[formal_table.condition.eq("Baseline High")].iloc[0]
    learned = formal_table[formal_table.condition.eq("Best stable correct-path aggregate")].iloc[0]
    patch_loss = seed_summary.baseline_patch_Mixed_loss.mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    titles = ("A. Baseline High", "B. Acute output-path patch", "C. Learned correct path")
    values = ((high.Mixed_sequence_loss, high.normalized_task_path_residual, high.AB_separation_ratio),
              (patch_loss, 0., high.AB_separation_ratio),
              (learned.Mixed_sequence_loss, learned.normalized_task_path_residual, learned.AB_separation_ratio))
    colors = ("#E45756", "#54A24B", "#4C78A8")
    for ax, title, (loss, path, split), color in zip(axes, titles, values, colors):
        ax.set_title(title, weight="bold")
        ax.bar([0, 1, 2], [loss, path, split], color=[color, "#B279A2", "#F2CF5B"])
        ax.set_xticks([0, 1, 2], ["Mixed\nloss", "task-path\nresidual", "A/B split\nratio"])
        ax.text(.5, -.23, ("Frozen endpoints; immediate counterfactual rescue" if "Acute" in title
                           else "Same familiar endpoints"), transform=ax.transAxes, ha="center", fontsize=8)
    _save(fig, figures, "FINAL_FIG04_causal_chain.png")


def make_timeseries(output, time_tables, roles):
    directory = Path(output) / "timeseries"
    role_frame = pd.DataFrame(
        [{"cell_id": cell, "final_role": role} for cell, role in roles.items()])
    familiar = time_tables["phase1c_timeseries_familiar_schedule"].merge(role_frame, on="cell_id")
    full = time_tables["phase1c_timeseries_full_probe_schedule"].merge(role_frame, on="cell_id")
    midpoint = time_tables["phase1c_timeseries_midpoint_checkpoints"].merge(role_frame, on="cell_id")
    sweep = time_tables["phase1c_timeseries_cue_sweep"].merge(role_frame, on="cell_id")
    for index, value in enumerate(("sequence_loss", "mean_correct_token_probability", "mean_execution_proxy")):
        fig, ax = plt.subplots(figsize=(11, 4.5)); _line(ax, familiar, "sequence_index", value)
        _schedule(ax); ax.legend(fontsize=7, ncol=3); _save(fig, directory, FINAL_TS[index])
    for index, value in enumerate(("sequence_loss", "mean_correct_token_probability", "mean_execution_proxy"), start=3):
        fig, ax = plt.subplots(figsize=(11, 4.5)); _line(ax, full, "sequence_index", value)
        _schedule(ax, probes=True); ax.legend(fontsize=7, ncol=3); _save(fig, directory, FINAL_TS[index])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, value in zip(axes.ravel(), ("heldout_midpoint_sequence_loss",
                                       "heldout_midpoint_logit_residual",
                                       "endpoint_hidden_drift_ratio", "remaining_patch_gap")):
        _line(ax, midpoint, "update", value)
    axes[0, 0].legend(fontsize=7); _save(fig, directory, FINAL_TS[6])
    fig, ax = plt.subplots(figsize=(10, 4.5)); _line(ax, sweep, "cue_lambda", "sequence_loss")
    ax.axvline(.5, color="black", ls="--", lw=.8); ax.legend(fontsize=7, ncol=3)
    _save(fig, directory, FINAL_TS[7])
    fig, ax = plt.subplots(figsize=(10, 4.5)); _line(ax, sweep, "cue_lambda", "logit_path_residual")
    ax.axvline(.5, color="black", ls="--", lw=.8); ax.legend(fontsize=7, ncol=3)
    _save(fig, directory, FINAL_TS[8])
    return {"familiar": familiar, "full": full, "midpoint": midpoint, "sweep": sweep}


def enforce_figures(output):
    output = Path(output)
    required = [output / "figures" / f"FINAL_FIG0{i}_{name}.png" for i, name in enumerate((
        "mechanism_triangle", "correct_vs_wrong_path_specificity",
        "seed_level_sufficiency", "causal_chain"), start=1)]
    required += [output / "timeseries" / name for name in FINAL_TS]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise RuntimeError("T12 failed: missing final figures:\n" + "\n".join(missing))
