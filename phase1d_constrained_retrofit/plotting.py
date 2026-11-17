from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = tuple(f"PHASE1D_FIG0{i}_{name}.png" for i, name in enumerate((
    "acute_patch_by_seed", "local_reachability", "success_rate_comparison",
    "seed_outcome_matrix", "reachability_predicts_learning",
    "correct_vs_wrong_constrained"), start=1))
CUE = tuple(f"PHASE1D_CUE0{i}_{name}.png" for i, name in enumerate((
    "task_path_residual", "sequence_loss", "correct_token_probability",
    "execution_proxy", "AB_axis_and_off_axis_geometry"), start=1))
TS = tuple(f"PHASE1D_TS{i:02d}_{name}.png" for i, name in enumerate((
    "familiar_sequence_loss", "familiar_correct_token_probability",
    "familiar_execution_proxy", "full_probe_sequence_loss",
    "full_probe_correct_token_probability", "full_probe_execution_proxy",
    "training_path_residual", "endpoint_constraint_violation",
    "AB_separation_ratio", "familiar_loss_ratio"), start=1))


def _save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    if not any(ax.get_label() == "<colorbar>" for ax in fig.axes):
        fig.tight_layout()
    fig.savefig(directory / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bands(ax, frame, x, y, group="method"):
    use = frame[np.isfinite(frame[y])]
    summary = use.groupby([group, x], as_index=False)[y].agg(
        mean="mean", low=lambda s: s.quantile(.025), high=lambda s: s.quantile(.975))
    colors = plt.cm.tab10(np.linspace(0, .9, max(summary[group].nunique(), 1)))
    for color, (label, values) in zip(colors, summary.groupby(group, sort=False)):
        values = values.sort_values(x)
        ax.plot(values[x], values["mean"], label=label, color=color, lw=1.6)
        ax.fill_between(values[x], values.low, values.high, color=color, alpha=.14, linewidth=0)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))


def _schedule(ax, updates, probes=False):
    for block, start in enumerate(range(1, updates + 1, 10), start=1):
        stop = min(start + 9, updates)
        label = "A" if block % 2 else "B"
        color = "#4C78A8" if label == "A" else "#F58518"
        ax.axvspan(start - .5, stop + .5, color=color, alpha=.025)
        ax.text((start + stop) / 2, .985, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5, color=color)
    if probes:
        ax.axvline(updates + .5, color="black", ls="--", lw=.8)


def make_primary_figures(output, acute, local, seed_master, rates, statistics):
    directory = Path(output) / "figures"
    ordered = acute.sort_values("seed")
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, row in enumerate(ordered.itertuples(index=False)):
        ax.plot([0, 1], [row.baseline_mixed_loss, row.patched_mixed_loss], color=".7", lw=.8)
        ax.scatter([0, 1], [row.baseline_mixed_loss, row.patched_mixed_loss], s=20)
    ax.set_xticks([0, 1], ["Baseline High", "Acute compositional patch"])
    ax.set_ylabel("Mixed sequence loss")
    _save(fig, directory, PRIMARY[0])

    local = local.sort_values("seed")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(local.seed.astype(str), local.Q_local); axes[0, 0].set_ylabel("Q local")
    axes[0, 1].bar(local.seed.astype(str), local.projected_gradient_fraction); axes[0, 1].set_ylabel("projected gradient fraction")
    axes[1, 0].bar(local.seed.astype(str), local.free_local_dimension); axes[1, 0].set_ylabel("free local dimension")
    merged = local.merge(seed_master[["seed", "phase1c_success"]], on="seed")
    axes[1, 1].scatter(merged.phase1c_success.astype(int), merged.Q_local)
    axes[1, 1].set_xticks([0, 1], ["Phase1C fail", "Phase1C pass"])
    axes[1, 1].set_ylabel("Q local")
    _save(fig, directory, PRIMARY[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rates)); y = rates.success_fraction.to_numpy(float)
    error = np.vstack((y - rates.wilson_low, rates.wilson_high - y))
    ax.bar(x, y, color=["#9D755D", "#4C78A8", "#F58518"])
    ax.errorbar(x, y, yerr=error, fmt="none", color="black", capsize=4)
    ax.set_xticks(x, rates.method, rotation=15, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("Phase 1C-comparable success fraction")
    for i, row in enumerate(rates.itertuples(index=False)):
        ax.text(i, row.success_fraction + .03, f"{row.success_n}/{row.n_seeds}", ha="center")
    _save(fig, directory, PRIMARY[2])

    matrix_frame = seed_master.sort_values("Q_local", ascending=False)
    matrix = np.column_stack((
        matrix_frame.acute_patch_delta.gt(0), matrix_frame.Q_local.ge(.5),
        matrix_frame.phase1c_success,
        matrix_frame.constrained_correct_phase1c_comparable_success,
        matrix_frame.constrained_wrong_phase1c_comparable_success,
    )).astype(float)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix_frame)), matrix_frame.seed.astype(int))
    ax.set_xticks(range(5), ["acute\nimproves", "Q≥.5", "soft\nsuccess",
                             "correct\nsuccess", "wrong\nsuccess"], rotation=20)
    ax.set_ylabel("seed, ordered by Q local")
    _save(fig, directory, PRIMARY[3])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = (
        ("Q_local", "constrained_correct_training_path_residual"),
        ("projected_gradient_fraction", "constrained_correct_training_path_residual"),
        ("Q_local", "constrained_mixed_improvement"),
        ("available_rescue", "constrained_mixed_improvement"),
    )
    plot = seed_master.assign(constrained_mixed_improvement=(
        seed_master.baseline_high_mixed_loss - seed_master.constrained_correct_mixed_loss))
    for ax, (xcol, ycol) in zip(axes.ravel(), panels):
        ax.scatter(plot[xcol], plot[ycol])
        for row in plot.itertuples(index=False):
            ax.annotate(str(int(row.seed)), (getattr(row, xcol), getattr(row, ycol)), fontsize=7)
        ax.set_xlabel(xcol.replace("_", " ")); ax.set_ylabel(ycol.replace("_", " "))
    _save(fig, directory, PRIMARY[4])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = seed_master.constrained_correct_mixed_loss
    y = seed_master.constrained_wrong_mixed_loss
    axes[0].scatter(x, y)
    limits = [min(x.min(), y.min()), max(x.max(), y.max())]
    axes[0].plot(limits, limits, color="black", ls="--")
    axes[0].set_xlabel("constrained correct Mixed loss")
    axes[0].set_ylabel("constrained wrong Mixed loss")
    delta = seed_master.wrong_minus_correct_mixed_loss
    axes[1].hist(delta, bins=max(5, min(15, len(delta))), color="#4C78A8")
    axes[1].axvline(0, color="black", ls="--")
    stat = statistics[statistics.analysis.str.startswith("Constrained")].iloc[0]
    axes[1].set_title(f"mean={stat['mean']:.3f}; {100*stat.fraction_positive:.1f}% > 0")
    axes[1].set_xlabel("wrong − correct Mixed loss")
    _save(fig, directory, PRIMARY[5])


def make_cue_figures(output, sweep):
    directory = Path(output) / "figures"
    fields = ("logit_path_residual", "sequence_loss", "mean_correct_token_probability",
              "mean_execution_proxy")
    for name, field in zip(CUE[:4], fields):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        _bands(ax, sweep, "cue_lambda", field)
        ax.axvline(.5, color="black", ls="--", lw=.8, label="held-out λ=.5")
        ax.legend(fontsize=7, ncol=2)
        _save(fig, directory, name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _bands(axes[0], sweep, "cue_lambda", "mean_axis_q")
    _bands(axes[1], sweep, "cue_lambda", "normalized_off_axis_residual")
    for ax in axes:
        ax.axvline(.5, color="black", ls="--", lw=.8)
    axes[0].legend(fontsize=7)
    _save(fig, directory, CUE[4])


def make_sequence_figures(output, familiar, full, training, updates):
    directory = Path(output) / "timeseries"
    for name, field in zip(TS[:3], (
            "sequence_loss", "mean_correct_token_probability", "mean_execution_proxy")):
        fig, ax = plt.subplots(figsize=(11, 4.6)); _bands(ax, familiar, "sequence_index", field)
        _schedule(ax, updates); ax.legend(fontsize=7, ncol=3); _save(fig, directory, name)
    for name, field in zip(TS[3:6], (
            "sequence_loss", "mean_correct_token_probability", "mean_execution_proxy")):
        fig, ax = plt.subplots(figsize=(11, 4.6)); _bands(ax, full, "sequence_index", field)
        _schedule(ax, updates, probes=True); ax.legend(fontsize=7, ncol=3); _save(fig, directory, name)
    fields = ("normalized_training_path_residual", "endpoint_constraint_violation",
              "AB_separation_ratio", "familiar_loss_ratio")
    for name, field in zip(TS[6:], fields):
        fig, ax = plt.subplots(figsize=(10, 4.6)); _bands(ax, training, "update", field)
        ax.legend(fontsize=7); _save(fig, directory, name)


def enforce_outputs(output):
    output = Path(output)
    expected = ([output / "figures" / name for name in (*PRIMARY, *CUE)]
                + [output / "timeseries" / name for name in TS])
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError("T15: mandatory Phase 1D figures are missing:\n" + "\n".join(missing))
