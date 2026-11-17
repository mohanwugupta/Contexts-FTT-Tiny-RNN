"""Publication-oriented confirmatory SRN figures."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "confirmatory-srn-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CONDITION_COLORS = {
    "High_monotonic": "#0072B2",
    "Low_monotonic": "#D55E00",
    "High_sequence_only": "#009E73",
}
CONTEXT_COLORS = {
    "A": "#0072B2", "B": "#D55E00", "Mixed_norm": "#CC79A7",
    "Novel": "#009E73", "Mixed_additive": "#7A5195",
}
MODEL_COLORS = {"SS_LUMP": "#E69F00", "SS_CROSS": "#009E73", "SS_SPLIT": "#8C564B"}


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


def _blocks(axis, maximum):
    for start in range(1, int(maximum) + 1, 10):
        block = (start - 1) // 10 + 1
        color = "#DCEEFF" if block % 2 else "#FCE8E6"
        axis.axvspan(start - .5, min(start + 9.5, maximum + .5), color=color,
                     alpha=.28, linewidth=0, zorder=-10)


def figure_training(sequence, path):
    metrics = [
        ("sequence_CE", "Sequence CE"),
        ("mean_correct_token_confidence", "Correct-token confidence"),
        ("mean_normalized_RT", "Normalized RT"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for axis, (metric, label) in zip(axes, metrics):
        _blocks(axis, sequence.training_sequence.max())
        for condition in sequence.condition.unique():
            if metric == "mean_normalized_RT" and condition == "High_sequence_only":
                continue
            data = sequence[sequence.condition.eq(condition)]
            line_style = "--" if condition == "High_sequence_only" else "-"
            for _, seed_data in data.groupby("seed"):
                axis.plot(seed_data.training_sequence, seed_data[metric],
                          color=CONDITION_COLORS[condition], alpha=.10, lw=.55,
                          ls=line_style)
            summary = data.groupby("training_sequence")[metric].agg(["mean", "std", "count"])
            sem = summary["std"].fillna(0) / np.sqrt(summary["count"])
            x = summary.index.to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            axis.plot(x, mean, color=CONDITION_COLORS[condition], lw=2,
                      ls=line_style, label=condition.replace("_", " "))
            axis.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                              color=CONDITION_COLORS[condition], alpha=.14, linewidth=0)
        axis.set_xlabel("External training sequence")
        axis.set_ylabel(label)
        _style(axis)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 1. Frozen confirmatory training dynamics", fontsize=15,
                 fontweight="bold")
    _finish(fig, path)


def figure_representation(separation, pca, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    aggregate = separation[separation.row_type.eq("seed_aggregate")]
    conditions = list(CONDITION_COLORS)
    values = [aggregate[aggregate.condition.eq(c)].aggregate_AB_separation for c in conditions]
    axes[0].boxplot(values, labels=[c.replace("_", "\n") for c in conditions],
                    patch_artist=True,
                    boxprops={"facecolor": "#E5E7EB"}, medianprops={"color": "#111827"})
    for index, condition in enumerate(conditions, start=1):
        y = aggregate[aggregate.condition.eq(condition)].aggregate_AB_separation.to_numpy()
        axes[0].scatter(np.full(len(y), index), y, color=CONDITION_COLORS[condition],
                        s=18, alpha=.7)
    axes[0].set_ylabel("Mean token-matched A/B centroid distance")
    axes[0].set_title("Seed-level representational separation")
    _style(axes[0])

    pure = pca[pca.probe_context.isin(["A", "B"])]
    for (condition, context), group in pure.groupby(["condition", "probe_context"]):
        means = group.groupby("transition_position")[["PC1", "PC2"]].mean()
        axes[1].plot(means.PC1, means.PC2, marker="o",
                     color=CONDITION_COLORS[condition],
                     ls="-" if context == "A" else "--",
                     label=f"{condition.replace('_', ' ')} / {context}")
        for position, row in means.iterrows():
            axes[1].text(row.PC1, row.PC2, str(position), fontsize=7)
    axes[1].set_xlabel("Shared PC1")
    axes[1].set_ylabel("Shared PC2")
    axes[1].set_title("Token-matched A/B trajectories")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    _style(axes[1])
    fig.suptitle("Figure 2. Representational splitting", fontsize=15, fontweight="bold")
    _finish(fig, path)


def figure_state_space(predictions, fits, classifications, path):
    conditions = ["High_monotonic", "Low_monotonic"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for row_index, condition in enumerate(conditions):
        data = predictions[
            predictions.condition.eq(condition)
            & predictions.fit_scope.eq("seed_average")
        ]
        observed = data[data.model.eq("SS_LUMP")].sort_values("training_sequence")
        axis = axes[row_index, 0]
        axis.plot(observed.training_sequence, observed.observed, color="#111827",
                  lw=2, label="Emergent RT")
        for model in MODEL_COLORS:
            one = data[data.model.eq(model)].sort_values("training_sequence")
            winner = bool(one.best_model.iloc[0])
            metric = fits[
                fits.condition.eq(condition) & fits.fit_scope.eq("seed_average")
                & fits.model.eq(model)
            ].iloc[0]
            axis.plot(one.training_sequence, one.predicted,
                      color=MODEL_COLORS[model], lw=2 if winner else 1.1,
                      ls="-" if winner else "--",
                      label=f"{model.removeprefix('SS_').title()} "
                            f"(RMSE={metric.RMSE:.3f}, R²={metric.R2:.2f})")
        axis.set_title(condition.replace("_", " "))
        axis.set_xlabel("Training sequence")
        axis.set_ylabel("Normalized RT")
        axis.legend(frameon=False, fontsize=7)
        _style(axis)

        bic_axis = axes[row_index, 1]
        mean_fits = fits[
            fits.condition.eq(condition) & fits.fit_scope.eq("seed_average")
        ]
        bic_axis.bar(mean_fits.model_label, mean_fits.delta_BIC,
                     color=[MODEL_COLORS[m] for m in mean_fits.model])
        seed_fraction = classifications[
            classifications.condition.eq(condition)
            & classifications.row_type.eq("classification_fraction")
        ]
        fraction_text = ", ".join(
            f"{r.model_label}={r.classification_fraction:.0%}"
            for r in seed_fraction.itertuples(index=False)
        )
        winner = mean_fits.loc[mean_fits.best_model, "model_label"].iloc[0]
        bic_axis.set_title(f"Mean winner: {winner}\nSeed winners: {fraction_text}")
        bic_axis.set_ylabel("ΔBIC")
        _style(bic_axis)
    fig.suptitle("Figure 3. Canonical Lump/Cross/Split bridge", fontsize=15,
                 fontweight="bold")
    _finish(fig, path)


def figure_probes(probes, path):
    execution = probes[
        probes.execution_active.eq(True)
        & probes.probe_context.isin(["A", "B", "Mixed_norm", "Novel"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, condition in zip(axes, ["High_monotonic", "Low_monotonic"]):
        data = execution[execution.condition.eq(condition)]
        for context in ["A", "B", "Mixed_norm", "Novel"]:
            summary = data[data.probe_context.eq(context)].groupby(
                "transition_position"
            ).normalized_RT.agg(["mean", "std", "count"])
            sem = summary["std"] / np.sqrt(summary["count"])
            axis.errorbar(summary.index, summary["mean"], yerr=1.96 * sem,
                          marker="o", color=CONTEXT_COLORS[context],
                          label=context, capsize=2)
        axis.set_title(condition.replace("_", " "))
        axis.set_xlabel("Transition position")
        axis.set_ylabel("Normalized RT")
        axis.legend(frameon=False)
        _style(axis)
    fig.suptitle("Figure 4. Frozen probe transition profiles", fontsize=15,
                 fontweight="bold")
    _finish(fig, path)


def figure_responsibility(responsibility, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, condition in zip(axes, ["High_monotonic", "Low_monotonic"]):
        data = responsibility[
            responsibility.condition.eq(condition)
            & responsibility.probe_context.isin(["Mixed_norm", "Novel"])
        ]
        for context in ["Mixed_norm", "Novel"]:
            one = data[data.probe_context.eq(context)]
            profile = one.groupby("transition_position").q_unclipped.agg(
                ["mean", "std", "count"]
            )
            sem = profile["std"] / np.sqrt(profile["count"])
            axis.errorbar(profile.index, profile["mean"], yerr=1.96 * sem,
                          marker="o", color=CONTEXT_COLORS[context],
                          label=f"{context} q_k", capsize=2)
            scalar = one.groupby("seed").q_global_unclipped.first().mean()
            axis.axhline(scalar, color=CONTEXT_COLORS[context], ls="--", alpha=.65,
                         label=f"{context} global q")
        axis.axhline(.5, color="#6B7280", ls=":", lw=1)
        axis.set_xlim(.5, 7.5)
        axis.set_xticks(range(1, 8))
        axis.set_title(condition.replace("_", " "))
        axis.set_xlabel("Transition position")
        axis.set_ylabel("Unclipped responsibility q")
        axis.legend(frameon=False, fontsize=8)
        _style(axis)
    fig.suptitle("Figure 5. Transition-specific and scalar responsibility",
                 fontsize=15, fontweight="bold")
    _finish(fig, path)


def figure_low_rank(results, predictions, path):
    selected = results[results.selected.eq(True)]
    if selected.empty or not selected.low_rank_supported.fillna(False).any():
        return False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, condition in zip(axes, ["High_monotonic", "Low_monotonic"]):
        data = predictions[
            predictions.condition.eq(condition)
            & predictions.probe_context.isin(["Mixed_norm", "Novel"])
        ]
        for context in ["Mixed_norm", "Novel"]:
            one = data[data.probe_context.eq(context)]
            profile = one.groupby("transition_position")[[
                "observed_midpoint_deviation", "predicted_midpoint_deviation"
            ]].mean()
            color = CONTEXT_COLORS[context]
            axis.plot(profile.index, profile.observed_midpoint_deviation,
                      marker="o", color=color, label=f"{context} observed")
            axis.plot(profile.index, profile.predicted_midpoint_deviation,
                      ls="--", color=color, label=f"{context} low-rank")
        rank = int(selected[selected.condition.eq(condition)].selected_rank.iloc[0])
        axis.set_title(f"{condition.replace('_', ' ')} (rank {rank})")
        axis.set_xlabel("Transition position")
        axis.set_ylabel("Probe midpoint deviation")
        axis.legend(frameon=False, fontsize=8)
        _style(axis)
    fig.suptitle("Figure 6. A/B-selected low-rank recombination", fontsize=15,
                 fontweight="bold")
    _finish(fig, path)
    return True


def figure_causal(patching, controls, path):
    natural = patching.drop_duplicates([
        "condition", "seed", "probe_context", "transition_position"
    ]).assign(
        display_condition="Natural probe",
        displayed_RT=lambda x: x.natural_normalized_RT,
        prediction_identity_preserved=True,
    )
    matched = patching.assign(
        display_condition=lambda x: "Probe <- " + x.donor_context,
        displayed_RT=lambda x: x.intervened_normalized_RT,
    )
    control_labels = {
        "random_orthogonal_direction": "Orthogonal random",
        "wrong_token_AB_component": "Wrong-token A/B",
        "magnitude_matched_random_vector": "Magnitude random",
    }
    control_levels = controls.assign(
        display_condition=lambda x: x.intervention_type.map(control_labels),
        displayed_RT=lambda x: x.intervened_normalized_RT,
    )
    levels = pd.concat([natural, matched, control_levels], ignore_index=True)
    order = [
        "Natural probe", "Probe <- A", "Probe <- B", "Orthogonal random",
        "Wrong-token A/B", "Magnitude random",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col")
    colors = [
        "#111827", CONTEXT_COLORS["A"], CONTEXT_COLORS["B"],
        "#9CA3AF", "#9CA3AF", "#9CA3AF",
    ]
    for row_index, condition in enumerate(["High_monotonic", "Low_monotonic"]):
        summary = levels[levels.condition.eq(condition)].groupby(
            "display_condition"
        ).agg(
            mean_RT=("displayed_RT", "mean"),
            se_RT=("displayed_RT", "sem"),
            identity_preserved=("prediction_identity_preserved", "mean"),
        ).reindex(order).reset_index()
        rt_axis, preservation_axis = axes[row_index]
        rt_axis.bar(summary.display_condition, summary.mean_RT,
                    yerr=1.96 * summary.se_RT.fillna(0), capsize=3, color=colors)
        rt_axis.tick_params(axis="x", rotation=24)
        rt_axis.set_ylabel("Normalized RT")
        rt_axis.set_title(
            f"{condition.replace('_', ' ')}: natural and intervened RT"
        )
        _style(rt_axis)
        preservation_axis.bar(
            summary.display_condition, summary.identity_preserved, color=colors
        )
        preservation_axis.axhline(.90, color="#B91C1C", ls="--", lw=1,
                                  label="90% preservation criterion")
        preservation_axis.set_ylim(0, 1.05)
        preservation_axis.tick_params(axis="x", rotation=24)
        preservation_axis.set_ylabel("Prediction identity preserved")
        preservation_axis.set_title(
            f"{condition.replace('_', ' ')}: sequence preservation"
        )
        preservation_axis.legend(frameon=False, fontsize=8)
        _style(preservation_axis)
    fig.suptitle("Figure 7. Causal A/B component intervention and controls",
                 fontsize=15, fontweight="bold")
    _finish(fig, path)


def supplementary_switch(aligned, path):
    fig, axis = plt.subplots(figsize=(9, 5))
    for condition in ["High_monotonic", "Low_monotonic"]:
        data = aligned[aligned.condition.eq(condition)]
        summary = data.groupby("relative_sequence").normalized_RT.agg(
            ["mean", "std", "count"]
        )
        sem = summary["std"] / np.sqrt(summary["count"])
        axis.errorbar(summary.index, summary["mean"], yerr=1.96 * sem,
                      color=CONDITION_COLORS[condition], marker="o", capsize=2,
                      label=condition.replace("_", " "))
    axis.axvline(-.5, color="#111827", ls="--")
    axis.set_xlabel("Sequences relative to switch (0 = first new context)")
    axis.set_ylabel("Normalized RT")
    axis.legend(frameon=False)
    axis.set_title("Supplement: switch-aligned emergent RT", fontweight="bold")
    _style(axis)
    _finish(fig, path)


def make_figures(output_dir, *, sequence, separation, pca, state_fits,
                 state_classification, state_predictions, probes, responsibility,
                 low_rank_results, low_rank_predictions, patching, controls,
                 switch_aligned):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_training(sequence, output / "figure_01_confirmatory_training")
    figure_representation(separation, pca, output / "figure_02_representation")
    figure_state_space(state_predictions, state_fits, state_classification,
                       output / "figure_03_state_space")
    figure_probes(probes, output / "figure_04_frozen_probes")
    figure_responsibility(responsibility, output / "figure_05_responsibility")
    figure_low_rank(low_rank_results, low_rank_predictions,
                    output / "figure_06_low_rank")
    figure_causal(patching, controls, output / "figure_07_causal")
    supplementary_switch(switch_aligned, output / "figure_s1_switch_aligned")
