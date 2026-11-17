#!/usr/bin/env python3
"""Generate the five paper-level figures from frozen headline CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"blue": "#2b6cb0", "purple": "#805ad5", "green": "#2f855a",
          "orange": "#dd6b20", "gray": "#718096", "red": "#c53030"}


def _load(path):
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    return json.loads(source.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _finish(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _mean_sem(frame, groups, values):
    return frame.groupby(groups, as_index=False).agg(**{
        f"{value}_mean": (value, "mean") for value in values
    }, **{
        f"{value}_sem": (value, "sem") for value in values
    })


def _figure_1(path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [
        (.02, .30, .17, .42, "A/B blocked FTT\nidentical sequence"),
        (.23, .30, .17, .42, "Context-specific\nsequence states"),
        (.44, .30, .17, .42, "Execution-control\nconsequences"),
        (.65, .30, .15, .42, "Mixed / Novel\ngraded probes"),
        (.84, .30, .14, .42, "Human low-rank\nconfigured state"),
    ]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
    for (x, y, w, h, label), color in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                                   facecolor=color, alpha=.14, edgecolor=color, linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(right[0], .51), xytext=(left[0] + left[2], .51),
                    xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(.50, .12, "Mechanistic existence proof", transform=ax.transAxes,
            ha="center", fontsize=11, color=COLORS["blue"])
    ax.text(.91, .12, "Not an exact SRN–human token-profile match", transform=ax.transAxes,
            ha="center", fontsize=11, color=COLORS["red"])
    ax.set_title("Context configures the active state of a learned motor sequence", fontsize=16)
    _finish(fig, path)


def _figure_2(path, execution):
    import matplotlib.pyplot as plt
    splitting = pd.read_csv(execution / "splitting_metrics.csv")
    probes = pd.read_csv(execution / "probe_interpolation.csv")
    subset = splitting[splitting.checkpoint.eq("human_matched")]
    split = _mean_sem(subset, ["model_condition"], ["ab_hidden_separation"])
    order = ["High", "Low", "High_seq_only"]
    split = split.set_index("model_condition").reindex(order)
    probe = probes[
        probes.model_condition.eq("High") & probes.checkpoint.eq("human_matched")
    ]
    probe = _mean_sem(probe, ["probe_type"], ["rt_a", "rt_b", "rt_probe"])
    probe_order = [value for value in ("Mixed_norm", "Mixed_additive", "Novel")
                   if value in set(probe.probe_type)]
    probe = probe.set_index("probe_type").reindex(probe_order)
    grounding = subset[subset.model_condition.eq("High")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].bar(order, split.ab_hidden_separation_mean,
                yerr=split.ab_hidden_separation_sem, capsize=3,
                color=[COLORS["blue"], COLORS["gray"], COLORS["orange"]])
    axes[0].set(ylabel="A/B hidden-state separation", title="A  Representational splitting")
    axes[0].tick_params(axis="x", rotation=18)

    x = np.arange(len(probe))
    for offset, value, label, color in (
        (-.24, "rt_a", "A endpoint", COLORS["blue"]),
        (0, "rt_probe", "Probe", COLORS["green"]),
        (.24, "rt_b", "B endpoint", COLORS["purple"]),
    ):
        axes[1].bar(x + offset, probe[f"{value}_mean"], width=.23,
                    yerr=probe[f"{value}_sem"], capsize=2, label=label, color=color)
    axes[1].set_xticks(x, [value.replace("_", "\n") for value in probe_order])
    axes[1].set(ylabel="Normalized RT", title="B  Frozen graded probes")
    axes[1].legend(frameon=False, fontsize=9)

    labels = ["Overall", "Within A", "Within B", "Controlled"]
    values = [grounding.stage2_confidence_speed_correlation_overall,
              grounding.stage2_confidence_speed_correlation_A,
              grounding.stage2_confidence_speed_correlation_B,
              grounding.stage2_confidence_speed_controlled_coefficient]
    axes[2].bar(labels, values, color=COLORS["orange"])
    axes[2].axhline(0, color=COLORS["gray"], lw=.8)
    axes[2].set(ylabel="Confidence–speed association",
                title="C  Execution grounding")
    axes[2].tick_params(axis="x", rotation=18)
    fig.suptitle("High-scale SRNs split A/B states and express graded retrieval", fontsize=16)
    _finish(fig, path)


def _figure_3(path, context_sequence, causal):
    import matplotlib.pyplot as plt
    effects = pd.read_csv(context_sequence / "token_structured_causal_effects.csv")
    summary = pd.read_csv(context_sequence / "interaction_mechanism_summary.csv")
    ablation = pd.read_csv(causal / "ablation_seed_summary.csv")
    effects = effects[effects.checkpoint.eq("human_matched")]
    grouped = _mean_sem(
        effects, ["probe_context"],
        ["full_patch_effect", "wrong_token_patch_effect", "random_patch_effect"],
    )
    decision = summary[
        summary.checkpoint.eq("human_matched") & summary.seed.eq(summary.seed.min())
    ].sort_values("probe_context")
    ablation = ablation[ablation.checkpoint.eq("human_matched")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(grouped))
    for offset, value, label, color in (
        (-.25, "full_patch_effect", "Matched", COLORS["blue"]),
        (0, "wrong_token_patch_effect", "Mismatched", COLORS["gray"]),
        (.25, "random_patch_effect", "Orthogonal random", COLORS["orange"]),
    ):
        axes[0].bar(x + offset, grouped[f"{value}_mean"], width=.24,
                    yerr=grouped[f"{value}_sem"], capsize=2, label=label, color=color)
    axes[0].set_xticks(x, grouped.probe_context)
    axes[0].set(ylabel="Execution patch effect", title="A  Position-matched specificity")
    axes[0].legend(frameon=False, fontsize=9)

    criteria = ["M1", "M2", "M3", "M4", "M5"]
    columns = ["group_fraction_M1_full_natural_patching",
               "group_fraction_M2_correct_token_specificity",
               "group_fraction_M3_token_structured_correspondence",
               "group_fraction_M4_full_beats_shared",
               "group_fraction_M5_necessity_with_sequence_preservation"]
    for color, row in zip((COLORS["green"], COLORS["purple"]), decision.itertuples()):
        axes[1].plot(criteria, [getattr(row, value) for value in columns], marker="o",
                     label=row.probe_context, color=color)
    axes[1].axhline(.8, color=COLORS["red"], ls="--", lw=1, label="Registered threshold")
    axes[1].set(ylim=(0, 1.05), ylabel="Fraction of seeds", title="B  Mechanism criteria")
    axes[1].legend(frameon=False, fontsize=9)

    values = [ablation.ablation_execution_collapse.mean(),
              ablation.ablation_sequence_preservation.mean()]
    errors = [ablation.ablation_execution_collapse.sem(),
              ablation.ablation_sequence_preservation.sem()]
    axes[2].bar(["Execution\ncollapse", "Sequence\npreservation"], values,
                yerr=errors, capsize=3, color=[COLORS["orange"], COLORS["blue"]])
    axes[2].axhline(.9, color=COLORS["red"], ls="--", lw=1)
    axes[2].set(ylim=(0, 1.1), title="C  Strict necessity boundary")
    fig.suptitle("Causal interventions support configured current-state components", fontsize=16)
    _finish(fig, path)


def _figure_4(path, human):
    import matplotlib.pyplot as plt
    pure = pd.read_csv(human / "pure_ab_token_profiles.csv")
    mixed = pd.read_csv(human / "mixed_midpoint_deviation.csv")
    novel = pd.read_csv(human / "novel_midpoint_deviation.csv")
    mixed_models = pd.read_csv(human / "mixed_model_comparison.csv")
    novel_models = pd.read_csv(human / "novel_model_comparison.csv")
    pure = _mean_sem(pure, ["dataset", "transition_position"], ["delta"])
    mixed_profile = _mean_sem(mixed, ["transition_position"], ["midpoint_deviation"])
    novel_profile = _mean_sem(novel, ["transition_position"], ["midpoint_deviation"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for dataset, group in pure.groupby("dataset"):
        axes[0].errorbar(group.transition_position, group.delta_mean,
                         yerr=group.delta_sem, marker="o", label=dataset)
    axes[0].axhline(0, color=COLORS["gray"], lw=.8)
    axes[0].set(xlabel="Transition position", ylabel="Mean A−B log RT",
                title="A  Matched-position context effects")
    axes[0].legend(frameon=False)

    axes[1].errorbar(mixed_profile.transition_position, mixed_profile.midpoint_deviation_mean,
                     yerr=mixed_profile.midpoint_deviation_sem, marker="o", label="Mixed")
    axes[1].errorbar(novel_profile.transition_position, novel_profile.midpoint_deviation_mean,
                     yerr=novel_profile.midpoint_deviation_sem, marker="o", label="Novel")
    axes[1].axhline(0, color=COLORS["gray"], lw=.8)
    axes[1].set(xlabel="Transition position", ylabel="Midpoint deviation",
                title="B  Structured probe retrieval")
    axes[1].legend(frameon=False)

    models = ["M0_scalar_context", "M1_scalar_mixture", "M2_token_specific",
              "M3_low_rank", "M4_srn_informed"]
    x = np.arange(len(models))
    axes[2].bar(x - .18, mixed_models.set_index("model").reindex(models).mean_heldout_log_likelihood,
                width=.36, label="Mixed", color=COLORS["purple"])
    axes[2].bar(x + .18, novel_models.set_index("model").reindex(models).mean_heldout_log_likelihood,
                width=.36, label="Novel", color=COLORS["green"])
    axes[2].set_xticks(x, [value.split("_", 1)[0] for value in models])
    axes[2].set(ylabel="Mean held-out log likelihood", title="C  Held-out model comparison")
    axes[2].legend(frameon=False)
    fig.suptitle("Human context effects are transition-structured and low-dimensional", fontsize=16)
    _finish(fig, path)


def _figure_5(path, low_rank):
    import matplotlib.pyplot as plt
    loadings = pd.read_csv(low_rank / "component_loadings.csv")
    dimensions = pd.read_csv(low_rank / "dimension_selection.csv")
    dimensions = dimensions[
        dimensions.row_type.eq("summary") & dimensions.method.eq("PCA")
    ]
    dim = pd.read_csv(low_rank / "human_srn_dimensionality_comparison.csv")
    similarity = pd.read_csv(low_rank / "human_srn_subspace_similarity.csv").iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for component, group in loadings.groupby("component"):
        axes[0].plot(group.transition_position, group.loading, marker="o",
                     label=f"Component {component}")
    axes[0].axhline(0, color=COLORS["gray"], lw=.8)
    axes[0].set(xlabel="Transition position", ylabel="Loading",
                title="A  Selected rank-2 human basis")
    axes[0].legend(frameon=False)

    for dataset, group in dimensions.groupby("dataset"):
        axes[1].plot(group["rank"], group.mean_heldout_probe_log_likelihood,
                     marker="o", label=dataset)
    axes[1].axvline(2, color=COLORS["red"], ls="--", lw=1, label="Selected rank")
    axes[1].set(xlabel="Candidate rank", ylabel="Mean held-out probe log likelihood",
                title="B  Participant-held-out selection")
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].bar(dim.system, dim.effective_dimensionality,
                color=[COLORS["purple"], COLORS["blue"]])
    axes[2].set(ylabel="Participation-ratio dimension",
                title=("C  Human–SRN abstraction mismatch\n"
                       f"mean angle={similarity.mean_principal_angle_degrees:.1f}°, "
                       f"max={similarity.maximum_principal_angle_degrees:.1f}°"))
    fig.suptitle("Rank 2 is reproducible, but the exact SRN geometry does not match humans", fontsize=16)
    _finish(fig, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate five paper-level figures.")
    parser.add_argument("--srn-config", default="configs/srn_final.json")
    parser.add_argument("--human-config", default="configs/human_final.json")
    parser.add_argument("--output", default="outputs/final")
    return parser.parse_args()


def main():
    args = parse_args()
    srn = _load(args.srn_config)
    human = _load(args.human_config)
    output = ROOT / args.output
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    execution = ROOT / srn["outputs"]["execution_v2_1"]
    causal = ROOT / srn["outputs"]["causal_v2_2"]
    context_sequence = ROOT / srn["outputs"]["context_sequence_v2_3"]
    configured = ROOT / human["outputs"]["configured_state_v3"]
    low_rank = ROOT / human["outputs"]["low_rank_interpretation_v3_1"]
    sources = [
        execution / "splitting_metrics.csv", execution / "probe_interpolation.csv",
        context_sequence / "token_structured_causal_effects.csv",
        context_sequence / "interaction_mechanism_summary.csv",
        causal / "ablation_seed_summary.csv",
        configured / "context_position_model.csv", configured / "pure_ab_token_profiles.csv",
        configured / "mixed_midpoint_deviation.csv", configured / "novel_midpoint_deviation.csv",
        configured / "mixed_model_comparison.csv", configured / "novel_model_comparison.csv",
        low_rank / "dimension_selection.csv", low_rank / "component_loadings.csv",
        low_rank / "cross_dataset_subspace_similarity.csv",
        low_rank / "human_srn_dimensionality_comparison.csv",
        low_rank / "human_srn_subspace_similarity.csv",
        low_rank / "final_dimension_interpretation_summary.csv",
    ]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing paper source tables: " + ", ".join(map(str, missing)))
    _figure_1(figures / "figure_01_conceptual_model.png")
    _figure_2(figures / "figure_02_srn_splitting_execution.png", execution)
    _figure_3(figures / "figure_03_causal_configured_state.png", context_sequence, causal)
    _figure_4(figures / "figure_04_human_structured_retrieval.png", configured)
    _figure_5(figures / "figure_05_low_rank_human_srn.png", low_rank)

    table_records = []
    for source in sources:
        target = tables / f"{source.parent.name}__{source.name}"
        shutil.copy2(source, target)
        table_records.append({
            "source": source.relative_to(ROOT).as_posix(),
            "paper_copy": _display_path(target),
            "sha256": _sha(source),
        })
    provenance = {
        "schema_version": "1.0.0",
        "srn_config": Path(args.srn_config).as_posix(),
        "human_config": Path(args.human_config).as_posix(),
        "source_tables": table_records,
        "figures": sorted(path.name for path in figures.glob("*.png")),
    }
    (output / "paper_figure_sources.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote five paper figures and {len(table_records)} source-table copies to {output}")


if __name__ == "__main__":
    main()
