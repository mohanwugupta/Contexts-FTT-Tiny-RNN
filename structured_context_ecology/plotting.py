"""Six preregistered figures for the stepwise structured-context ecology."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "structured-ecology-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "High_monotonic": "#0072B2", "Low_monotonic": "#D55E00",
    "High_sequence_only": "#009E73", "A": "#0072B2", "B": "#D55E00",
    "Mixed": "#CC79A7", "Novel": "#009E73",
    "SS_LUMP": "#E69F00", "SS_CROSS": "#009E73", "SS_SPLIT": "#8C564B",
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


def figure_ecology_schematic(path):
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.axis("off")
    token_x = np.linspace(.10, .90, 7)
    for index, x in enumerate(token_x, start=1):
        axis.add_patch(plt.Rectangle(
            (x - .035, .47), .07, .08, color="#374151", alpha=.92
        ))
        axis.text(x, .51, f"s{index}", color="white", ha="center", va="center",
                  fontweight="bold")
    axis.plot([.07, .93], [.44, .44], color="#6B7280", lw=1.5)
    axis.text(.04, .51, "Same sequence", ha="right", va="center", fontweight="bold")
    groups = [1, 1, 2, 2, 3, 3, 3]
    for context, y, color in (("A", .72, COLORS["A"]), ("B", .27, COLORS["B"])):
        axis.text(.04, y, f"Context {context}", ha="right", va="center",
                  color=color, fontweight="bold")
        for x, group in zip(token_x, groups):
            axis.text(x - .018, y, "S", ha="center", va="center", color="white",
                      bbox={"boxstyle": "round", "fc": "#7A5195", "ec": "none"})
            axis.text(x + .032, y, f"{context}{group}", ha="center", va="center",
                      color="white", bbox={"boxstyle": "round", "fc": color, "ec": "none"})
            axis.plot([x, x], [y + (-.06 if y > .5 else .06), .55 if y > .5 else .47],
                      color=color, alpha=.35)
    axis.text(.5, .97, "Figure 1. Structured context ecology",
              ha="center", fontsize=16, fontweight="bold")
    axis.text(.5, .89, "Literal shared item S + ordinal context-specific item",
              ha="center", fontsize=14, fontweight="bold")
    axis.text(.5, .08, "Sequence tokens never change; only contextual item evidence varies",
              ha="center", fontsize=12)
    _finish(fig, path)


def _preferred(frame):
    if frame.empty or "cohort" not in frame:
        return frame
    if frame.cohort.eq("confirmatory").any():
        return frame[frame.cohort.eq("confirmatory")]
    return frame[frame.cohort.eq("screen")]


def figure_splitting(baseline, stage1, stage2, salience, path):
    pieces = []
    base = baseline["summary"][["condition", "seed", "aggregate_AB_separation"]].copy()
    base["ecology"] = "Categorical"
    pieces.append(base)
    for label, frame in (("Shared item", stage1), ("Position structured", stage2)):
        if frame.empty:
            continue
        one = _preferred(frame)
        one = one[one.row_type.eq("seed_aggregate")][[
            "condition", "seed", "aggregate_AB_separation"
        ]].copy()
        one["ecology"] = label
        pieces.append(one)
    if not salience.empty:
        one = salience[salience.row_type.eq("seed_aggregate")].copy()
        confirm = one[one.cohort.eq("confirmatory")]
        one = confirm if len(confirm) else one[one.cohort.eq("screen")]
        one["ecology"] = one.sigma_context.map(lambda x: f"Salience σ={x:g}")
        pieces.append(one[["condition", "seed", "aggregate_AB_separation", "ecology"]])
    data = pd.concat(pieces, ignore_index=True)
    order = list(dict.fromkeys(data.ecology))
    fig, axis = plt.subplots(figsize=(13, 6))
    width = .34
    for offset, condition in ((-.17, "High_monotonic"), (.17, "Low_monotonic")):
        summary = data[data.condition.eq(condition)].groupby("ecology").agg(
            mean=("aggregate_AB_separation", "mean"),
            sem=("aggregate_AB_separation", "sem"),
        ).reindex(order)
        x = np.arange(len(order)) + offset
        axis.bar(x, summary["mean"], width, yerr=1.96 * summary["sem"].fillna(0),
                 color=COLORS[condition], alpha=.85,
                 label=condition.replace("_monotonic", " scale"), capsize=3)
    axis.set_xticks(np.arange(len(order)), order, rotation=18, ha="right")
    axis.set_ylabel("Mean token-matched A/B hidden separation")
    axis.legend(frameon=False)
    axis.set_title("Stepwise splitting across environmental stages")
    _style(axis)
    fig.suptitle("Figure 2. Representational splitting", fontsize=16, fontweight="bold")
    _finish(fig, path)


def figure_state_space(baseline, fits, predictions, path):
    base_fits = baseline["fits"].copy()
    base_predictions = baseline["predictions"].copy()
    for frame in (base_fits, base_predictions):
        frame["stage"] = "stage0_categorical"
        frame["cohort"] = "reference"
        frame["sigma_context"] = 0.0
    all_fits = pd.concat([base_fits, fits], ignore_index=True, sort=False)
    all_predictions = pd.concat([base_predictions, predictions], ignore_index=True, sort=False)
    candidates = []
    for stage in all_predictions.stage.drop_duplicates():
        one = all_predictions[all_predictions.stage.eq(stage)]
        if stage == "stage0_categorical":
            candidates.append((stage, "reference", 0.0))
        elif one.cohort.eq("confirmatory").any():
            for sigma in one[one.cohort.eq("confirmatory")].sigma_context.unique():
                candidates.append((stage, "confirmatory", float(sigma)))
    if not candidates:
        candidates = [("stage0_categorical", "reference", 0.0)]
    fig, axes = plt.subplots(len(candidates), 2, figsize=(16, 4.2 * len(candidates)),
                             squeeze=False)
    for row, (stage, cohort, sigma) in enumerate(candidates):
        for column, condition in enumerate(("High_monotonic", "Low_monotonic")):
            query = (
                all_predictions.stage.eq(stage) & all_predictions.cohort.eq(cohort)
                & np.isclose(all_predictions.sigma_context, sigma)
                & all_predictions.condition.eq(condition)
                & all_predictions.fit_scope.eq("seed_average")
            )
            data = all_predictions[query]
            axis = axes[row, column]
            if data.empty:
                axis.axis("off")
                continue
            observed = data[data.model.eq("SS_LUMP")].sort_values("training_sequence")
            axis.plot(observed.training_sequence, observed.observed, color="#111827",
                      lw=2, label="Observed RT")
            for model in ("SS_LUMP", "SS_CROSS", "SS_SPLIT"):
                one = data[data.model.eq(model)].sort_values("training_sequence")
                metric = all_fits[
                    all_fits.stage.eq(stage) & all_fits.cohort.eq(cohort)
                    & np.isclose(all_fits.sigma_context, sigma)
                    & all_fits.condition.eq(condition)
                    & all_fits.fit_scope.eq("seed_average") & all_fits.model.eq(model)
                ].iloc[0]
                axis.plot(one.training_sequence, one.predicted, color=COLORS[model],
                          lw=2 if bool(metric.best_model) else 1, ls="-" if bool(metric.best_model) else "--",
                          label=f"{metric.model_label} ΔBIC={metric.delta_BIC:.1f}")
            winner = all_fits[
                all_fits.stage.eq(stage) & all_fits.cohort.eq(cohort)
                & np.isclose(all_fits.sigma_context, sigma)
                & all_fits.condition.eq(condition) & all_fits.fit_scope.eq("seed_average")
                & all_fits.best_model.eq(True)
            ].iloc[0]
            axis.set_title(
                f"{stage.replace('_', ' ')}, σ={sigma:g}, {condition.replace('_', ' ')}\n"
                f"best={winner.model_label}; RMSE={winner.RMSE:.3f}; R²={winner.R2:.2f}"
            )
            axis.set_xlabel("Training sequence")
            axis.set_ylabel("Normalized RT")
            axis.legend(frameon=False, fontsize=7)
            _style(axis)
    fig.suptitle("Figure 3. Canonical Lump/Cross/Split overlays", fontsize=16,
                 fontweight="bold")
    _finish(fig, path)


def figure_mixed(mixed_probe, path):
    if mixed_probe.empty or "condition" not in mixed_probe:
        fig, axis = plt.subplots(figsize=(10, 4)); axis.text(.5, .5, "Mixed not run: prior gate failed", ha="center"); axis.axis("off")
        _finish(fig, path); return
    data = mixed_probe[
        mixed_probe.condition.eq("High_monotonic")
        & mixed_probe.probe_context.isin(["A", "B", "Mixed"])
    ].copy()
    if data.empty:
        fig, axis = plt.subplots(figsize=(10, 4)); axis.text(.5, .5, "Mixed not run: prior gate failed", ha="center"); axis.axis("off")
        _finish(fig, path); return
    stage_order = list(dict.fromkeys(data.stage))
    fig, axes = plt.subplots(1, len(stage_order), figsize=(6 * len(stage_order), 5),
                             sharey=True, squeeze=False)
    for axis, stage in zip(axes[0], stage_order):
        one_stage = data[data.stage.eq(stage)]
        if one_stage.cohort.eq("confirmatory").any():
            one_stage = one_stage[one_stage.cohort.eq("confirmatory")]
        for context in ("A", "B", "Mixed"):
            summary = one_stage[one_stage.probe_context.eq(context)].groupby(
                "transition_position"
            ).normalized_RT.agg(["mean", "sem"])
            axis.errorbar(summary.index, summary["mean"], yerr=1.96 * summary["sem"],
                          color=COLORS[context], marker="o", label=context, capsize=2)
        sigma = float(one_stage.sigma_context.iloc[0])
        axis.set_title(f"{stage.replace('_', ' ')}; σ={sigma:g}")
        axis.set_xlabel("Transition position")
        axis.set_ylabel("Normalized RT")
        axis.legend(frameon=False)
        _style(axis)
    fig.suptitle("Figure 4. Mixed behavior across context ecologies", fontsize=16,
                 fontweight="bold")
    _finish(fig, path)


def figure_corridor(stage3, salience, path):
    data = pd.concat([stage3, salience], ignore_index=True, sort=False)
    if data.empty or "condition" not in data:
        fig, axis = plt.subplots(figsize=(10, 4)); axis.text(.5, .5, "Mixed geometry not run", ha="center"); axis.axis("off"); _finish(fig, path); return
    data = data[data.condition.eq("High_monotonic")]
    if data.empty:
        fig, axis = plt.subplots(figsize=(10, 4)); axis.text(.5, .5, "Mixed geometry not run", ha="center"); axis.axis("off"); _finish(fig, path); return
    confirm = data[data.cohort.isin(["confirmatory", "reference"])]
    if len(confirm):
        data = confirm
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (stage, sigma), group in data.groupby(["stage", "sigma_context"], sort=True):
        label = f"{stage.replace('_', ' ')} σ={sigma:g}"
        lam = group.groupby("transition_position").closest_lambda_segment.mean()
        distance = group.groupby("transition_position").distance_to_AB_segment.mean()
        axes[0].plot(lam.index, lam, marker="o", label=label)
        axes[1].plot(distance.index, distance, marker="o", label=label)
    axes[0].axhline(0, color=COLORS["B"], ls="--", label="B endpoint")
    axes[0].axhline(1, color=COLORS["A"], ls="--", label="A endpoint")
    axes[0].set(title="Closest location on A/B segment", ylabel="λ (B=0, A=1)")
    axes[1].set(title="Off-corridor distance", ylabel="Distance to A/B segment")
    for axis in axes:
        axis.set_xlabel("Transition position"); axis.legend(frameon=False, fontsize=7); _style(axis)
    fig.suptitle("Figure 5. Mixed representational corridor", fontsize=16,
                 fontweight="bold")
    _finish(fig, path)


def figure_tradeoff(baseline, stage2, salience_rep, stage3, salience_mixed, path):
    rep_parts = []
    base = baseline["summary"][["condition", "seed", "aggregate_AB_separation"]].copy()
    base["stage"] = "stage0_categorical"; base["cohort"] = "reference"; base["sigma_context"] = 0.0
    rep_parts.append(base)
    for frame in (stage2, salience_rep):
        if not frame.empty:
            rep_parts.append(frame[frame.row_type.eq("seed_aggregate")])
    rep = pd.concat(rep_parts, ignore_index=True, sort=False)
    behavior = pd.concat([stage3, salience_mixed], ignore_index=True, sort=False)
    if behavior.empty or "condition" not in behavior:
        fig, axis = plt.subplots(figsize=(10, 6))
        axis.text(.5, .5, "Tradeoff unavailable: Mixed stage not reached", ha="center")
        axis.axis("off")
        axis.set_title("High splitting and recombination are reported separately")
        fig.suptitle("Figure 6. Splitting–recombination tradeoff", fontsize=16,
                     fontweight="bold")
        _finish(fig, path)
        return
    behavior = behavior[
        behavior.condition.eq("High_monotonic")
        & behavior.responsibility_denominator_usable.eq(True)
    ]
    points = []
    keys = ["stage", "cohort", "sigma_context"]
    for key, group in behavior.groupby(keys, sort=True):
        stage, cohort, sigma = key
        matching = rep[
            rep.stage.eq(stage) & rep.cohort.eq(cohort)
            & np.isclose(rep.sigma_context, sigma) & rep.condition.eq("High_monotonic")
        ]
        if matching.empty:
            continue
        points.append({
            "stage": stage, "cohort": cohort, "sigma": sigma,
            "separation": matching.aggregate_AB_separation.mean(),
            "between": group.probe_between_A_B.mean(),
            "segment": group.distance_to_AB_segment.mean(),
        })
    points = pd.DataFrame(points)
    fig, axis = plt.subplots(figsize=(10, 6))
    if points.empty:
        axis.text(.5, .5, "Tradeoff unavailable: Mixed stage not reached", ha="center")
        axis.axis("off")
    else:
        scatter = axis.scatter(points.separation, points.between,
                               c=points.segment, cmap="viridis_r", s=90,
                               edgecolor="#111827")
        labels = []
        for row in points.itertuples(index=False):
            if row.stage == "stage0_categorical":
                label = "Categorical\n20-seed"
            elif row.stage == "stage2_structured":
                label = "Structured\n20-seed"
            elif row.cohort == "confirmatory":
                label = f"Salience σ={row.sigma:g}\n20-seed"
            else:
                label = f"Screen σ={row.sigma:g}"
            labels.append(label)
        offsets = [(8, 8), (8, -24), (8, 8), (8, 8), (8, -24), (8, 8)]
        for index, (row, label) in enumerate(zip(points.itertuples(index=False), labels)):
            axis.annotate(label, (row.separation, row.between),
                          xytext=offsets[index % len(offsets)],
                          textcoords="offset points", fontsize=7)
        fig.colorbar(scatter, ax=axis, label="Mean Mixed-to-segment distance")
        axis.set_xlabel("A/B hidden separation")
        axis.set_ylabel("Mixed between-endpoint fraction")
        axis.set_ylim(-.03, 1.03)
        x_range = float(points.separation.max() - points.separation.min())
        x_padding = max(.08, .12 * x_range)
        axis.set_xlim(points.separation.min() - x_padding,
                      points.separation.max() + x_padding)
        _style(axis)
    axis.set_title("High splitting and recombination are reported separately")
    fig.suptitle("Figure 6. Splitting–recombination tradeoff", fontsize=16,
                 fontweight="bold")
    _finish(fig, path)


def make_figures(output_dir, *, baseline, stage_summary, stage1_representation,
                 stage2_representation, salience_representation, fits, predictions,
                 mixed_probe, stage3_responsibility, salience_mixed):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    figure_ecology_schematic(output / "figure_01_context_ecology")
    figure_splitting(baseline, stage1_representation, stage2_representation,
                     salience_representation, output / "figure_02_stepwise_splitting")
    figure_state_space(baseline, fits, predictions,
                       output / "figure_03_state_space_overlays")
    figure_mixed(mixed_probe, output / "figure_04_mixed_evolution")
    figure_corridor(stage3_responsibility, salience_mixed,
                    output / "figure_05_representational_corridor")
    figure_tradeoff(baseline, stage2_representation, salience_representation,
                    stage3_responsibility, salience_mixed,
                    output / "figure_06_splitting_recombination_tradeoff")
