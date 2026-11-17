"""Registered calibration and revised-test figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import stability_category


def _save(fig, output, name): fig.tight_layout(); fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def make_calibration_figures(output, steps, cells, selected, variance, revised=None):
    fig, axes = plt.subplots(1, len(sorted(steps.hidden_size.unique())), figsize=(5 * len(steps.hidden_size.unique()), 4), sharey=True)
    for ax, width in zip(np.atleast_1d(axes), sorted(steps.hidden_size.unique())):
        one = steps[steps.hidden_size.eq(width)]
        for alpha, values in one[one.inner_step.gt(0)].groupby("alpha"):
            line = values.groupby("inner_step")[["support_NLL", "copy_NLL"]].mean(); ax.plot(line.index, line.support_NLL, label=f"support a={alpha:g}"); ax.plot(line.index, line.copy_NLL, ls="--", label=f"copy a={alpha:g}")
        ax.set_title(f"H={width}"); ax.set_xlabel("inner step"); ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("NLL"); _save(fig, output, "figure_01_inner_loop_learning_curves.png")
    for metric, number, title in (("copy_NLL_post", 2, "Post-adaptation copy NLL"), ("copy_accuracy_post", 3, "Post-adaptation copy accuracy")):
        regimes = list(cells.initialization_regime.unique()); fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4), squeeze=False)
        for ax, regime in zip(axes[0], regimes):
            one = cells[(cells.initialization_regime.eq(regime)) & cells.inner_step.gt(0)]
            pivot = one.pivot_table(index="hidden_size", columns="alpha", values=metric, aggfunc="min" if metric.endswith("NLL_post") else "max")
            image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis"); ax.set_title(regime); ax.set_yticks(range(len(pivot.index)), pivot.index); ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45); fig.colorbar(image, ax=ax)
        _save(fig, output, f"figure_{number:02d}_{'copy_NLL' if number == 2 else 'copy_accuracy'}_heatmap.png")
    map_frame = cells[cells.inner_step.gt(0)].copy(); map_frame["category"] = map_frame.apply(stability_category, axis=1)
    codes = {name: index for index, name in enumerate(["stable without improvement", "support improves only", "copy improves", "catastrophic"])}
    pivot = map_frame.assign(code=map_frame.category.map(codes)).pivot_table(index="hidden_size", columns="alpha", values="code", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.imshow(pivot, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=3); ax.set_yticks(range(len(pivot.index)), pivot.index); ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns]); ax.set_xlabel("alpha"); ax.set_ylabel("width")
    _save(fig, output, "figure_04_stability_map.png")
    fig, ax = plt.subplots(figsize=(7, 4)); values = steps[steps.inner_step.gt(0)].groupby("hidden_size").update_over_parameter_norm.mean(); ax.plot(values.index, values, marker="o"); ax.set_xlabel("width"); ax.set_ylabel("update / parameter norm")
    _save(fig, output, "figure_05_effective_update_magnitude.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    for (regime, alpha), one in steps[steps.inner_step.gt(0)].groupby(["initialization_regime", "alpha"]):
        line = one.groupby("hidden_size").hidden_saturation_fraction.mean()
        ax.plot(line.index, line, marker="o", label=f"{regime}, a={alpha:g}")
    ax.set_xlabel("width"); ax.set_ylabel("fraction |h| > .95"); ax.legend(fontsize=7)
    _save(fig, output, "figure_06_hidden_saturation.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    for width, one in steps.groupby("hidden_size"):
        line = one.groupby("inner_step").recurrent_spectral_radius.mean()
        ax.plot(line.index, line, label=f"H{width}")
    ax.set_xlabel("inner step"); ax.set_ylabel("spectral radius"); ax.legend()
    _save(fig, output, "figure_07_spectral_radius_trajectory.png")
    fig, ax = plt.subplots(figsize=(10, max(2, .5 * max(1, len(selected))))); ax.axis("off")
    display = selected.fillna("").astype(str) if len(selected) else pd.DataFrame({"status": ["No stable width"]})
    ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    _save(fig, output, "figure_08_selected_regimes.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(variance):
        for width, one in variance.groupby("hidden_size"): ax.plot(one.groupby("meta_batch_size").outer_gradient_CV.mean(), marker="o", label=f"H{width}")
        ax.legend()
    else: ax.text(.5, .5, "Not reached", transform=ax.transAxes, ha="center")
    ax.set_xlabel("meta-batch size"); ax.set_ylabel("outer-gradient CV")
    _save(fig, output, "figure_09_meta_batch_gradient_variance.png")
    if revised is not None and len(revised):
        fig, ax = plt.subplots(figsize=(8, 4))
        values = revised.groupby(["training_condition", "hidden_size"], as_index=False).unseen_query_NLL.mean()
        for condition, one in values.groupby("training_condition"): ax.plot(one.hidden_size, one.unseen_query_NLL, marker="o", label=condition)
        ax.set_xlabel("width"); ax.set_ylabel("unseen query NLL"); ax.legend()
        _save(fig, output, "figure_10_revised_meta_learning.png")
