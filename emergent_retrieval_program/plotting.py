"""Nine prespecified diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = ["V0", "C1", "C2", "E1", "E2", "E3", "E4"]
COLORS = dict(zip(VARIANTS, plt.cm.tab10.colors[:7]))


def _save(fig, output: Path, name: str) -> None:
    fig.tight_layout(); fig.savefig(output / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def architecture(output: Path, registry: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.3)); ax.axis("off")
    descriptions = {
        "V0": "G2 SRN\nbaseline", "C1": "RBF responsibility\noutput mixture",
        "C2": "temporal prior +\noutput mixture", "E1": "static learned\n2-D state",
        "E2": "sequence-recurrent\n2-D state", "E3": "prediction-error\nGRU state",
        "E4": "2-D FiLM\nmodulation",
    }
    for index, variant in enumerate(VARIANTS):
        imposed = registry[variant]["averaging_imposed"]
        color = "#d95f02" if imposed else "#1b9e77"
        ax.text(index, .55, f"{variant}\n{descriptions[variant]}\n" +
                ("AVERAGING IMPOSED" if imposed else "NO OUTPUT AVERAGING"),
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=.45", facecolor="white", edgecolor=color, linewidth=2))
    ax.set_xlim(-.6, 6.6); ax.set_ylim(0, 1)
    ax.set_title("Explicit averaging controls versus emergent context-inference models")
    _save(fig, output, "figure_01_architecture_schematic.png")


def familiar(summary, hidden, gates, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    late = summary.groupby(["model_variant", "condition"], as_index=False).NLL_late.mean()
    for condition, style in (("High_monotonic", "o-"), ("Low_monotonic", "s--")):
        one = late[late.condition.eq(condition)].set_index("model_variant").reindex(VARIANTS)
        axes[0].plot(range(7), one.NLL_late, style, label=condition.replace("_monotonic", ""))
    axes[0].set_xticks(range(7), VARIANTS); axes[0].set_title("Late familiar prediction NLL"); axes[0].legend()
    seed = hidden[hidden.row_type.eq("seed_mean")]
    values = seed.groupby(["model_variant", "condition"]).aggregate_AB_separation.mean()
    for condition, style in (("High_monotonic", "o-"), ("Low_monotonic", "s--")):
        axes[1].plot(range(7), [values.get((v, condition), np.nan) for v in VARIANTS], style, label=condition[:4])
    axes[1].set_xticks(range(7), VARIANTS); axes[1].set_title("A/B hidden differentiation")
    if len(gates):
        g = gates.set_index("model_variant").reindex(VARIANTS)
        matrix = g[["G1_familiar_learning", "G2_high_differentiation", "G3_high_cross_split", "G4_low_lump", "G5_RT_grounding", "G6_probe_firewall"]].to_numpy(float).T
        axes[2].imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        axes[2].set_yticks(range(6), [f"G{i}" for i in range(1, 7)]); axes[2].set_xticks(range(7), VARIANTS)
    axes[2].set_title("Screen gates")
    _save(fig, output, "figure_02_familiar_AB_phenotype.png")


def transfer(metrics, context, output, filename):
    one = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_context.eq(context)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes,
        ["prediction_NLL", "normalized_RT", "next_token_accuracy"],
        ["Prediction NLL", "Normalized RT", "Next-token accuracy"]):
        means = one.groupby(["model_variant", "condition"])[metric].mean()
        for condition, marker in (("High_monotonic", "o"), ("Low_monotonic", "s")):
            ax.plot(range(7), [means.get((v, condition), np.nan) for v in VARIANTS], marker=marker, label=condition[:4])
        ax.set_xticks(range(7), VARIANTS); ax.set_title(title)
    axes[0].legend(); fig.suptitle(f"Frozen {context} transfer")
    _save(fig, output, filename)


def interpolation_figure(interpolation, output):
    data = interpolation[interpolation.row_type.eq("seed_mean")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(7); width = .35
    for offset, context in ((-.5, "Mixed"), (.5, "Novel")):
        values = data[data.probe_context.eq(context)].groupby("model_variant").between_endpoint_fraction.mean()
        ax.bar(x + offset * width, [values.get(v, np.nan) for v in VARIANTS], width, label=context)
    ax.set_xticks(x, VARIANTS); ax.set_ylim(0, 1); ax.set_ylabel("Fraction between A/B endpoints"); ax.legend()
    _save(fig, output, "figure_05_behavioral_interpolation.png")


def corridor_figure(corridor, output):
    fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(7); width = .35
    for offset, context in ((-.5, "Mixed"), (.5, "Novel")):
        values = corridor[corridor.probe_context.eq(context)].groupby("model_variant").off_corridor_distance.mean()
        ax.bar(x + offset * width, [values.get(v, np.nan) for v in VARIANTS], width, label=context)
    ax.set_xticks(x, VARIANTS); ax.set_ylabel("Mean off-corridor hidden distance"); ax.legend()
    _save(fig, output, "figure_06_hidden_state_corridor.png")


def latent_figure(latent, output):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=False, sharey=False)
    for ax, variant in zip(axes, ["E1", "E2", "E3", "E4"]):
        one = (latent[(latent.model_variant.eq(variant)) & latent.condition.eq("High_monotonic")]
               if len(latent) else latent)
        for context, marker in zip(["A", "B", "Mixed", "Novel", "Neutral"], ["o", "s", "^", "D", "x"]):
            part = one[one.context.eq(context)] if len(one) else one
            if len(part):
                ax.scatter(part.latent_state_1, part.latent_state_2, s=10, alpha=.5, marker=marker, label=context)
        ax.set_title(variant); ax.axhline(0, color=".85", lw=.5); ax.axvline(0, color=".85", lw=.5)
    axes[0].legend(fontsize=7); fig.suptitle("Two-dimensional latent context geometry (High scale)")
    _save(fig, output, "figure_07_latent_context_geometry.png")


def history_figure(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    seed = history[history.row_type.eq("seed_mean")] if len(history) else history
    for ax, context in zip(axes, ["Mixed", "Novel"]):
        one = seed[seed.context.eq(context)] if len(seed) else seed
        for variant in ["C2", "E2", "E3"]:
            values = one[one.model_variant.eq(variant)].groupby("preceding_context").normalized_RT.mean()
            ax.plot(range(3), [values.get(h, np.nan) for h in ["reset", "A", "B"]], marker="o", label=variant)
        ax.set_xticks(range(3), ["reset", "after 10 A", "after 10 B"]); ax.set_title(context)
    axes[0].set_ylabel("Mean normalized RT"); axes[0].legend(); fig.suptitle("Frozen history sensitivity")
    _save(fig, output, "figure_08_history_dependence.png")


def sufficiency_figure(table, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in table.itertuples(index=False):
        marker = "s" if row.averaging_imposed else "o"
        ax.scatter(row.Mixed_between_endpoint_fraction, row.Novel_between_endpoint_fraction,
                   marker=marker, s=90, color=COLORS.get(row.model_variant, ".4"))
        ax.text(row.Mixed_between_endpoint_fraction, row.Novel_between_endpoint_fraction, row.model_variant, fontsize=9)
    ax.set_xlabel("Mixed between-endpoint fraction"); ax.set_ylabel("Novel between-endpoint fraction")
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
    ax.set_title("Sufficiency (squares) versus emergence (circles)")
    _save(fig, output, "figure_09_sufficiency_vs_emergence.png")


def make_all_figures(output: Path, registry: dict, summary: pd.DataFrame,
                     hidden: pd.DataFrame, gates: pd.DataFrame, metrics: pd.DataFrame,
                     interpolation: pd.DataFrame, corridor: pd.DataFrame,
                     latent: pd.DataFrame, history: pd.DataFrame,
                     sufficiency: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=True)
    architecture(output, registry)
    familiar(summary, hidden, gates, output)
    transfer(metrics, "Mixed", output, "figure_03_mixed_transfer.png")
    transfer(metrics, "Novel", output, "figure_04_novel_transfer.png")
    interpolation_figure(interpolation, output)
    corridor_figure(corridor, output)
    latent_figure(latent, output)
    history_figure(history, output)
    sufficiency_figure(sufficiency, output)
