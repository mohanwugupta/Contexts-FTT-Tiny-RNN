"""Six preregistered figures for the ordinal/layout ecology."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ordinal-layout-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import representation_diagnostics  # noqa: E402
from .inputs import LAYOUT_A, LAYOUT_B  # noqa: E402


COLORS = {
    "A": "#0072B2", "B": "#D55E00", "Mixed": "#CC79A7",
    "Mixed_half": "#009E73", "High_monotonic": "#0072B2",
    "Low_monotonic": "#D55E00", "SS_LUMP": "#E69F00",
    "SS_CROSS": "#009E73", "SS_SPLIT": "#8C564B",
}


def _style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D1D5DB", linewidth=.5, alpha=.45)


def _finish(fig, path):
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _placeholder(path, title, message):
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.text(.5, .5, message, ha="center", va="center", fontsize=13)
    axis.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_environment(path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, coordinates, context in zip(axes, (LAYOUT_A, LAYOUT_B), ("A", "B")):
        axis.plot(coordinates[:, 0], coordinates[:, 1], color=COLORS[context],
                  lw=2, alpha=.75)
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=220,
                     color=COLORS[context], edgecolor="white", zorder=3)
        for index, (x, y) in enumerate(coordinates, start=1):
            axis.text(x, y, f"P{index}", color="white", ha="center", va="center",
                      fontsize=9, fontweight="bold")
            if index < 8:
                dx = coordinates[index, 0] - x
                dy = coordinates[index, 1] - y
                axis.arrow(x, y, .72 * dx, .72 * dy, width=.001,
                           head_width=.025, color=COLORS[context], alpha=.65,
                           length_includes_head=True)
        axis.set_title(f"Context {context}: full layout visible at every step")
        axis.set_xlabel("normalized horizontal coordinate")
        axis.set_ylabel("normalized vertical coordinate")
        axis.set_xlim(-.52, .52)
        axis.set_ylim(-.52, .52)
        axis.set_aspect("equal", adjustable="box")
        _style(axis)
    fig.suptitle("Figure 1. Shared P1→P8 order, different spatial realizations",
                 fontsize=16, fontweight="bold")
    fig.text(.5, .01, "Ordinal identity is shared; spatial realization differs",
             ha="center", fontsize=12)
    _finish(fig, path)


def figure_raw_rt(stage_batch, path):
    if not stage_batch or stage_batch["sequence"].empty:
        _placeholder(path, "Figure 2. Raw A/B RT learning",
                     "Stage 2 not run: prior gate failed")
        return
    data = stage_batch["sequence"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for axis, condition in zip(axes, ("High_monotonic", "Low_monotonic")):
        one = data[data.condition.eq(condition)]
        mean = one.groupby(
            ["training_sequence", "block_index", "context"], as_index=False
        ).mean_normalized_RT.mean()
        for block in sorted(mean.block_index.unique()):
            context = "A" if block % 2 else "B"
            axis.axvspan((block - 1) * 10 + .5, block * 10 + .5,
                         color=COLORS[context], alpha=.07)
            if block > 1:
                axis.axvline((block - 1) * 10 + .5, color="#6B7280", lw=.5,
                             alpha=.65)
        axis.plot(mean.training_sequence, mean.mean_normalized_RT,
                  color="#111827", lw=1.5)
        for context in ("A", "B"):
            local = mean[mean.context.eq(context)]
            axis.scatter(local.training_sequence, local.mean_normalized_RT, s=8,
                         color=COLORS[context], label=f"Context {context}", alpha=.8)
        axis.set_ylabel("Normalized RT")
        axis.set_title(condition.replace("_", " "))
        axis.legend(frameon=False, ncol=2)
        _style(axis)
    axes[-1].set_xlabel("External training sequence")
    fig.suptitle("Figure 2. Raw context-yoked RT learning",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_state_space(fits, predictions, path):
    if fits.empty or predictions.empty:
        _placeholder(path, "Figure 3. Lump/Cross/Split bridge",
                     "State-space fit not available: prior gate failed")
        return
    preferred = predictions[predictions.stage.eq("stage2_layout")]
    if preferred.cohort.eq("confirmatory").any():
        cohort = "confirmatory"
    elif preferred.cohort.eq("screen").any():
        cohort = "screen"
    else:
        _placeholder(path, "Figure 3. Lump/Cross/Split bridge",
                     "Stage 2 state-space fit not available")
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for axis, condition in zip(axes, ("High_monotonic", "Low_monotonic")):
        query = (
            predictions.stage.eq("stage2_layout")
            & predictions.cohort.eq(cohort)
            & predictions.condition.eq(condition)
            & predictions.fit_scope.eq("seed_average")
        )
        data = predictions[query]
        observed = data[data.model.eq("SS_LUMP")].sort_values("training_sequence")
        if observed.empty:
            axis.axis("off")
            continue
        axis.plot(observed.training_sequence, observed.observed, color="#111827",
                  lw=2, label="Observed RT")
        for model in ("SS_LUMP", "SS_CROSS", "SS_SPLIT"):
            one = data[data.model.eq(model)].sort_values("training_sequence")
            metric = fits[
                fits.stage.eq("stage2_layout") & fits.cohort.eq(cohort)
                & fits.condition.eq(condition) & fits.fit_scope.eq("seed_average")
                & fits.model.eq(model)
            ].iloc[0]
            axis.plot(one.training_sequence, one.predicted, color=COLORS[model],
                      ls="-" if metric.best_model else "--",
                      lw=2 if metric.best_model else 1,
                      label=f"{metric.model_label} ΔBIC={metric.delta_BIC:.1f}")
        seeds = fits[
            fits.stage.eq("stage2_layout") & fits.cohort.eq(cohort)
            & fits.condition.eq(condition) & fits.fit_scope.eq("seed")
            & fits.best_model.eq(True)
        ]
        fractions = seeds.model_label.value_counts(normalize=True)
        fraction_text = ", ".join(
            f"{name}={fractions.get(name, 0):.0%}"
            for name in ("Lump", "Cross", "Split")
        )
        winner = fits[
            fits.stage.eq("stage2_layout") & fits.cohort.eq(cohort)
            & fits.condition.eq(condition) & fits.fit_scope.eq("seed_average")
            & fits.best_model.eq(True)
        ].iloc[0]
        axis.set_title(
            f"{condition.replace('_', ' ')}; mean best={winner.model_label}\n"
            f"{fraction_text}"
        )
        axis.set_xlabel("External training sequence")
        axis.set_ylabel("Normalized RT")
        axis.legend(frameon=False, fontsize=8)
        _style(axis)
    fig.suptitle("Figure 3. Canonical Lump/Cross/Split bridge",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_representation(baseline, stage1, stage2, path):
    pieces = [("Categorical", representation_diagnostics(baseline["probes"]))]
    for label, frame in (("Ordinal", stage1), ("Ordinal + layout", stage2)):
        if not frame.empty:
            pieces.append((label, frame))
    rows = []
    for label, frame in pieces:
        cohort = "confirmatory" if frame.cohort.eq("confirmatory").any() else frame.cohort.iloc[0]
        frame = frame[frame.cohort.eq(cohort) & frame.row_type.eq("seed_aggregate")]
        for source in frame.itertuples(index=False):
            rows.append({
                "ecology": label, "condition": source.condition, "seed": source.seed,
                "separation": source.aggregate_AB_separation,
                "matched_cosine": source.AB_hidden_cosine,
            })
    data = pd.DataFrame(rows)
    order = list(dict.fromkeys(data.ecology))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    width = .35
    for offset, condition in ((-.18, "High_monotonic"), (.18, "Low_monotonic")):
        one = data[data.condition.eq(condition)]
        for axis, metric, ylabel in (
            (axes[0], "separation", "A/B hidden separation"),
            (axes[1], "matched_cosine", "Matched-position A/B cosine"),
        ):
            summary = one.groupby("ecology")[metric].agg(["mean", "sem"]).reindex(order)
            axis.bar(np.arange(len(order)) + offset, summary["mean"], width,
                     yerr=1.96 * summary["sem"].fillna(0), capsize=3,
                     color=COLORS[condition], alpha=.85,
                     label=condition.replace("_monotonic", " scale"))
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xticks(np.arange(len(order)), order, rotation=15, ha="right")
        axis.legend(frameon=False)
        _style(axis)
    fig.suptitle("Figure 4. Split yet ordinally related representations",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_mixed_profiles(probes, path):
    if probes.empty or "stage" not in probes:
        _placeholder(path, "Figure 5. Mixed RT profiles",
                     "Mixed not run: Stage 2 gate failed")
        return
    data = probes[
        probes.stage.eq("stage2_layout") & probes.condition.eq("High_monotonic")
        & probes.probe_context.isin(["A", "B", "Mixed", "Mixed_half"])
    ]
    if data.empty:
        _placeholder(path, "Figure 5. Mixed RT profiles",
                     "Mixed not run: Stage 2 gate failed")
        return
    if data.cohort.eq("confirmatory").any():
        data = data[data.cohort.eq("confirmatory")]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for context in ("A", "B", "Mixed", "Mixed_half"):
        summary = data[data.probe_context.eq(context)].groupby(
            "transition_position"
        ).normalized_RT.agg(["mean", "sem"])
        axis.errorbar(summary.index, summary["mean"], yerr=1.96 * summary["sem"],
                      color=COLORS[context], marker="o",
                      ls="--" if context == "Mixed_half" else "-",
                      label=context.replace("_half", " [.5,.5] control"), capsize=2)
    axis.set_xlabel("Ordinal transition position")
    axis.set_ylabel("Normalized RT")
    axis.legend(frameon=False)
    _style(axis)
    fig.suptitle("Figure 5. Familiar-layout recombination profiles",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_mixed_corridor(geometry, path):
    if geometry.empty or "stage" not in geometry:
        _placeholder(path, "Figure 6. Mixed representational corridor",
                     "Mixed geometry not run: Stage 2 gate failed")
        return
    data = geometry[
        geometry.stage.eq("stage2_layout")
        & geometry.condition.eq("High_monotonic")
        & geometry.probe_context.isin(["Mixed", "Mixed_half"])
    ]
    if data.empty:
        _placeholder(path, "Figure 6. Mixed representational corridor",
                     "Mixed geometry not run: Stage 2 gate failed")
        return
    if data.cohort.eq("confirmatory").any():
        data = data[data.cohort.eq("confirmatory")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for context in ("Mixed", "Mixed_half"):
        one = data[data.probe_context.eq(context)]
        lam = one.groupby("transition_position").closest_lambda_segment.mean()
        distance = one.groupby("transition_position").distance_to_AB_segment.mean()
        label = context.replace("_half", " [.5,.5] control")
        axes[0].plot(lam.index, lam, marker="o", color=COLORS[context], label=label)
        axes[1].plot(distance.index, distance, marker="o",
                     color=COLORS[context], label=label)
    axes[0].axhline(0, color=COLORS["B"], ls="--", label="B endpoint")
    axes[0].axhline(1, color=COLORS["A"], ls="--", label="A endpoint")
    axes[0].set_ylabel("Closest λ (B=0, A=1)")
    axes[1].set_ylabel("Distance to A/B segment")
    axes[0].set_title("Closest corridor location")
    axes[1].set_title("Off-corridor distance")
    for axis in axes:
        axis.set_xlabel("Ordinal transition position")
        axis.legend(frameon=False, fontsize=8)
        _style(axis)
    fig.suptitle("Figure 6. Mixed representational corridor",
                 fontsize=16, fontweight="bold")
    _finish(fig, path)


def make_figures(
    output_dir, *, baseline, stage1, stage2, stage1_representation,
    stage2_representation, fits, predictions, mixed_probe, mixed_geometry,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_environment(output / "figure_01_environment_schematic")
    figure_raw_rt(stage2 or stage1, output / "figure_02_raw_AB_RT_learning")
    figure_state_space(fits, predictions, output / "figure_03_state_space_bridge")
    figure_representation(
        baseline, stage1_representation, stage2_representation,
        output / "figure_04_representation",
    )
    figure_mixed_profiles(mixed_probe, output / "figure_05_mixed_profiles")
    figure_mixed_corridor(mixed_geometry, output / "figure_06_mixed_corridor")
