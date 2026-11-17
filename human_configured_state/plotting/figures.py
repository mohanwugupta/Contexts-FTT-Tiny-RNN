"""Eleven paper-oriented figures required by PRD v3."""

from __future__ import annotations

from pathlib import Path

import numpy as np


COLORS = {"A": "#2b6cb0", "B": "#c53030", "Both": "#805ad5", "Novel": "#2f855a"}


def _finish(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _pure_profiles(tables, path):
    import matplotlib.pyplot as plt
    data = tables["pure_ab_token_profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, (dataset, group) in zip(axes, data.groupby("dataset", sort=False)):
        for context, column in (("A", "r_a"), ("B", "r_b")):
            summary = group.groupby("transition_position")[column].agg(["mean", "sem"])
            ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o",
                        label=context, color=COLORS[context])
        ax.set(title=dataset, xlabel="Transition position", ylabel="Shrunk log RT")
        ax.legend(frameon=False)
    fig.suptitle("Human pure A/B transition profiles"); _finish(fig, path)


def _difference(tables, path, interaction=False):
    import matplotlib.pyplot as plt
    data = tables["pure_ab_token_profiles"].copy()
    summary = data.groupby(["dataset", "transition_position"]).delta.agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for dataset, group in summary.groupby("dataset"):
        ax.errorbar(group.transition_position, group["mean"], yerr=group["sem"], marker="o", label=dataset)
    ax.axhline(0, color="#718096", lw=.8)
    title = "Estimated Context × Position effect" if interaction else "Human A/B difference profile"
    ax.set(xlabel="Transition position", ylabel="A − B log RT", title=title)
    ax.legend(frameon=False); _finish(fig, path)


def _midpoint(tables, path):
    import matplotlib.pyplot as plt
    data = tables["mixed_midpoint_deviation"]
    summary = data.groupby("transition_position").midpoint_deviation.agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o",
                color=COLORS["Both"])
    ax.axhline(0, color="#718096", ls="--")
    ax.set(xlabel="Transition position", ylabel="Mixed log RT − A/B midpoint",
           title="Mixed midpoint deviation by transition")
    _finish(fig, path)


def _scalar_token_predictions(tables, path):
    import matplotlib.pyplot as plt
    m1 = tables["model_M1_scalar_mixture"].groupby("transition_position").agg(
        actual=("actual_log_rt", "mean"), predicted=("predicted_log_rt", "mean")
    )
    m2 = tables["model_M2_token_specific"].groupby("transition_position").predicted_log_rt.mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(m1.index, m1.actual, marker="o", color="#2d3748", label="Observed")
    ax.plot(m1.index, m1.predicted, marker="o", color="#718096", label="M1 scalar")
    ax.plot(m2.index, m2, marker="o", color=COLORS["Both"], label="M2 token-specific")
    ax.set(xlabel="Transition position", ylabel="Held-out log RT",
           title="Scalar versus token-specific Mixed predictions")
    ax.legend(frameon=False); _finish(fig, path)


def _human_q(tables, path):
    import matplotlib.pyplot as plt
    data = tables["mixed_midpoint_deviation"]
    data = data[data.ratio_reliable]
    summary = data.groupby("transition_position").responsibility_q.agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o",
                color=COLORS["Both"])
    ax.axhline(.5, color="#718096", ls="--")
    ax.set(xlabel="Transition position", ylabel="Unclipped responsibility q",
           title="Human Mixed configuration profile")
    _finish(fig, path)


def _srn_q(tables, path):
    import matplotlib.pyplot as plt
    data = tables["srn_mixed_profile"]
    data = data[data.row_type.eq("all_seed_aggregate")]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(data.transition_position, data.execution_q, marker="o", color="#2b6cb0")
    ax.axhline(.5, color="#718096", ls="--")
    ax.set(xlabel="Transition position", ylabel="SRN execution q",
           title="Frozen all-seed SRN Mixed profile")
    _finish(fig, path)


def _alignment(tables, path):
    import matplotlib.pyplot as plt
    data = tables["human_srn_alignment"]
    data = data[(data.probe_context.eq("Both")) & data.profile_variant.eq("execution")]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(data.srn_q, data.human_q, s=65, color=COLORS["Both"])
    if data.srn_q.std() > 1e-12:
        slope, intercept = np.polyfit(data.srn_q, data.human_q, 1)
        x = np.linspace(data.srn_q.min(), data.srn_q.max(), 100)
        ax.plot(x, intercept + slope * x, color="#2d3748")
    ax.set(xlabel="Frozen SRN Mixed q", ylabel="Human Mixed q",
           title=f"SRN versus human Mixed profile (r={data.pearson_r.iloc[0]:.2f})")
    _finish(fig, path)


def _mixed_novel(tables, path):
    import matplotlib.pyplot as plt
    data = tables["mixed_vs_novel_profile_comparison"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(data.transition_position, data.mixed_q, marker="o", label="Mixed", color=COLORS["Both"])
    ax.plot(data.transition_position, data.novel_q, marker="o", label="Novel", color=COLORS["Novel"])
    ax.set(xlabel="Transition position", ylabel="Human responsibility q",
           title=f"Mixed versus Novel profile (r={data.profile_pearson_r.iloc[0]:.2f})")
    ax.legend(frameon=False); _finish(fig, path)


def _comparison(tables, path):
    import matplotlib.pyplot as plt
    data = tables["mixed_model_comparison"].sort_values("model")
    labels = [value.split("_")[0] for value in data.model]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(labels, data.heldout_log_likelihood, color=["#718096", "#a0aec0", "#805ad5", "#d69e2e", "#2b6cb0"])
    ax.set(xlabel="Model", ylabel="Held-out log likelihood (higher is better)",
           title="Mixed held-out model comparison")
    _finish(fig, path)


def _recovery(tables, path):
    import matplotlib.pyplot as plt
    data = tables["model_recovery_matrix"]
    order = ["M0", "M1", "M2", "M3", "M4"]
    matrix = data.pivot(index="generating_model", columns="candidate_model",
                        values="recovery_probability").reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(5), order); ax.set_yticks(range(5), order)
    ax.set(xlabel="Recovered model", ylabel="Generating model", title="Model recovery matrix")
    fig.colorbar(image, ax=ax, label="Selection probability"); _finish(fig, path)


def make_required_figures(output_dir, *, tables, config):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    _pure_profiles(tables, output / "figure_01_human_pure_ab_profiles.png")
    _difference(tables, output / "figure_02_human_ab_difference.png")
    _difference(tables, output / "figure_03_context_position_interaction.png", interaction=True)
    _midpoint(tables, output / "figure_04_mixed_midpoint_deviation.png")
    _scalar_token_predictions(tables, output / "figure_05_scalar_vs_token_predictions.png")
    _human_q(tables, output / "figure_06_human_mixed_q.png")
    _srn_q(tables, output / "figure_07_frozen_srn_mixed_q.png")
    _alignment(tables, output / "figure_08_srn_human_alignment.png")
    _mixed_novel(tables, output / "figure_09_mixed_vs_novel.png")
    _comparison(tables, output / "figure_10_heldout_model_comparison.png")
    _recovery(tables, output / "figure_11_recovery_matrix.png")

