"""Prespecified aggregate figures (seed is always the unit of analysis)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"A": "#3B6FB6", "B": "#D65F4A", "Mixed": "#6A9F58",
          "Novel": "#8B63A9", "NoContextProbe": "#777777"}


def _save(fig, output: Path, name: str):
    fig.tight_layout()
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _condition_order(frame):
    return list(dict.fromkeys(frame["condition_label"].tolist()))


def make_figures(output: Path, tables: dict[str, pd.DataFrame]):
    decoder = tables["sequence_decoder_results"]
    geometry = tables["probe_axis_geometry"]
    factors = tables["factorization_variance"]
    subspace = tables["output_subspace_decomposition"]
    context = tables["context_decoder_scores"]
    sweep = tables["context_interpolation_sweep"]
    novel = tables["novel_input_audit"]
    history = tables["history_dependence"]
    causal = tables["causal_patching"]
    conditions = _condition_order(decoder)

    # 1
    probes = ["Held-out familiar", "Mixed", "Novel", "NoContextProbe"]
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, condition_name in zip(axes, conditions):
        sub = decoder[decoder.condition_label.eq(condition_name)]
        values = []
        for probe in probes:
            rows = sub[sub.held_out_familiar] if probe == "Held-out familiar" else sub[sub.test_context.eq(probe)]
            by_seed = rows.groupby("seed").decoder_accuracy.mean()
            values.append(by_seed)
        ax.boxplot(values, tick_labels=probes, showmeans=True)
        ax.axhline(1 / 7, color="black", ls=":", lw=1, label="chance")
        ax.set_title(condition_name); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("Position-decoder accuracy")
    _save(fig, output, "figure_01_sequence_decoder_generalization.png")

    # 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    probe_geometry = geometry[geometry.probe_context.isin(["Mixed", "Novel", "NoContextProbe"])]
    for condition_name in conditions:
        sub = probe_geometry[probe_geometry.condition_label.eq(condition_name)]
        mean = sub.groupby("probe_context").mean_q.mean().reindex(["Mixed", "Novel", "NoContextProbe"])
        axes[0].plot(mean.index, mean.values, marker="o", label=condition_name)
        residual = sub.groupby("probe_context").normalized_off_axis_residual.mean().reindex(mean.index)
        axes[1].plot(residual.index, residual.values, marker="o", label=condition_name)
    axes[0].axhspan(0, 1, alpha=.08, color="green"); axes[0].set_ylabel("Mean A/B-axis q")
    axes[1].set_ylabel("Normalized off-axis residual"); axes[1].legend(fontsize=8)
    _save(fig, output, "figure_02_probe_axis_projection.png")

    # 3
    means = factors.groupby("condition_label")[["P_sequence", "P_context", "P_interaction"]].mean().reindex(conditions)
    fig, ax = plt.subplots(figsize=(8, 4.5)); bottom = np.zeros(len(means))
    for column, color in zip(means.columns, ["#4C78A8", "#F2CF5B", "#E45756"]):
        ax.bar(means.index, means[column], bottom=bottom, label=column, color=color); bottom += means[column].to_numpy()
    ax.set_ylabel("Familiar hidden-state variance fraction"); ax.legend(); ax.tick_params(axis="x", rotation=15)
    _save(fig, output, "figure_03_factorization_variance.png")

    # 4
    means = subspace.groupby(["condition_label", "vector_type", "probe_context"]).relevant_fraction.mean().reset_index()
    means["label"] = means.vector_type.str.replace("_", " ") + "\n" + means.probe_context
    labels = list(dict.fromkeys(means.label))
    fig, ax = plt.subplots(figsize=(10, 4.5)); width = .8 / len(conditions)
    x = np.arange(len(labels))
    for i, condition_name in enumerate(conditions):
        values = means[means.condition_label.eq(condition_name)].set_index("label").reindex(labels).relevant_fraction
        ax.bar(x + (i - (len(conditions)-1)/2)*width, values, width, label=condition_name)
    ax.set_xticks(x, labels, rotation=20); ax.set_ylabel("Output-relevant energy fraction"); ax.legend(fontsize=8)
    _save(fig, output, "figure_04_output_relevant_null_decomposition.png")

    # 5
    probe_context = context[context.probe_context.isin(["A", "B", "Mixed", "Novel", "NoContextProbe"])]
    fig, ax = plt.subplots(figsize=(9, 4.5));
    for condition_name in conditions:
        mean = probe_context[probe_context.condition_label.eq(condition_name)].groupby("probe_context").mean_context_score.mean().reindex(["A", "Mixed", "Novel", "NoContextProbe", "B"])
        ax.plot(mean.index, mean, marker="o", label=condition_name)
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Familiar-trained context score (A=-1, B=+1)"); ax.legend(fontsize=8)
    _save(fig, output, "figure_05_context_decoder_probe_scores.png")

    # 6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition_name in conditions:
        sub = sweep[sweep.condition_label.eq(condition_name)].groupby("cue_lambda").sequence_loss.agg(["mean", "sem"])
        ax.plot(sub.index, sub["mean"], label=condition_name); ax.fill_between(sub.index, sub["mean"]-sub["sem"], sub["mean"]+sub["sem"], alpha=.15)
    ax.set_xlabel("Cue interpolation lambda (A to B)"); ax.set_ylabel("Sequence cross entropy"); ax.legend(fontsize=8)
    _save(fig, output, "figure_06_context_interpolation_sweep_loss.png")

    # 7
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition_name in conditions:
        sub = sweep[sweep.condition_label.eq(condition_name)].groupby("cue_lambda")[["mean_axis_q", "normalized_off_axis_residual"]].mean()
        axes[0].plot(sub.index, sub.mean_axis_q, label=condition_name); axes[1].plot(sub.index, sub.normalized_off_axis_residual, label=condition_name)
    axes[0].plot([0, 1], [0, 1], "k:", label="linear"); axes[0].set_ylabel("Mean A/B-axis q"); axes[1].set_ylabel("Normalized off-axis residual")
    for ax in axes: ax.set_xlabel("Cue lambda")
    axes[1].legend(fontsize=8)
    _save(fig, output, "figure_07_context_interpolation_sweep_geometry.png")

    # 8
    metrics = ["fraction_novel_input_norm_in_familiar_span", "induced_novel_norm", "induced_distance_to_familiar_span"]
    means = novel.groupby("condition_label")[metrics].mean().reindex(conditions)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(means.index, means[metric]); ax.set_title(metric.replace("_", " "), fontsize=9); ax.tick_params(axis="x", rotation=25)
    _save(fig, output, "figure_08_novel_input_audit.png")

    # 9
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, probe in zip(axes, ["Mixed", "Novel"]):
        mean = history[history.probe_context.eq(probe)].groupby(["condition_label", "history_condition"]).sequence_loss.mean().reset_index()
        labels = list(dict.fromkeys(mean.history_condition)); x = np.arange(len(labels)); width=.8/len(conditions)
        for i, condition_name in enumerate(conditions):
            values=mean[mean.condition_label.eq(condition_name)].set_index("history_condition").reindex(labels).sequence_loss
            ax.bar(x+(i-(len(conditions)-1)/2)*width, values, width, label=condition_name)
        ax.set_xticks(x, [v.replace("_", "\n") for v in labels], fontsize=7); ax.set_title(probe); ax.set_ylabel("Sequence cross entropy")
    axes[-1].legend(fontsize=8)
    _save(fig, output, "figure_09_history_dependence.png")

    # 10
    patched = causal[causal.patch_type.ne("baseline")]
    mean = patched.groupby(["condition_label", "probe_context", "patch_type"]).delta_loss_vs_baseline.mean().reset_index()
    labels = [f"{p}\n{t.replace('_', ' ')}" for p, t in mean[["probe_context", "patch_type"]].drop_duplicates().itertuples(index=False)]
    keys = list(mean[["probe_context", "patch_type"]].drop_duplicates().itertuples(index=False, name=None))
    fig, ax = plt.subplots(figsize=(12, 4.5)); x=np.arange(len(keys)); width=.8/len(conditions)
    for i, condition_name in enumerate(conditions):
        indexed=mean[mean.condition_label.eq(condition_name)].set_index(["probe_context", "patch_type"])
        values=[indexed.loc[key, "delta_loss_vs_baseline"] if key in indexed.index else np.nan for key in keys]
        ax.bar(x+(i-(len(conditions)-1)/2)*width, values, width, label=condition_name)
    ax.axhline(0,color="black",lw=.8); ax.set_xticks(x,labels,rotation=20,fontsize=8); ax.set_ylabel("Patched minus baseline loss (<0 = rescue)"); ax.legend(fontsize=8)
    _save(fig, output, "figure_10_causal_patching.png")
