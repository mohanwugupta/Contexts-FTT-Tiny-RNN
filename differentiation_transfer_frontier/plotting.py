"""Seven registered figures for the differentiation--transfer frontier."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FrontierConfig, SHARING_CONDITIONS
from .inputs import cue_vector


COLORS = dict(zip(SHARING_CONDITIONS, ["#6B7280", "#56B4E9", "#009E73", "#E69F00", "#CC79A7"]))


def _save(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _empty(ax, text: str = "Stopped by registered gate") -> None:
    ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_cue_schematic(config: FrontierConfig, output: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
    for ax, sharing in zip(axes, SHARING_CONDITIONS):
        matrix = np.vstack([cue_vector(config, sharing, "A"), cue_vector(config, sharing, "B")])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 / np.sqrt(5), aspect="auto")
        ax.set_yticks([0, 1], ["A", "B"])
        ax.set_title(f"{sharing}: {sharing[1:]} shared ({int(sharing[1:]) * 20}%)", loc="left", fontsize=10)
        for y in range(2):
            for x in range(10):
                if matrix[y, x] > 0:
                    ax.text(x, y, "1/√5", ha="center", va="center", fontsize=7, color="white")
    axes[-1].set_xticks(range(10), [f"C{i + 1}" for i in range(10)])
    fig.suptitle("Figure 1 — Constant-energy 10-D cue factorization\n5 active cues/context; L2 norm = 1", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_1_cue_factorization_schematic")


def plot_phase_a_phenotype(gates: pd.DataFrame, hidden: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    gate = gates.sort_values("shared_fraction")
    x = np.arange(len(SHARING_CONDITIONS))
    lookup = gate.drop_duplicates("environment_condition", keep="last").set_index("environment_condition")
    axes[0].bar(x, [lookup.loc[s, "high_cross_split_fraction"] for s in SHARING_CONDITIONS], color=[COLORS[s] for s in SHARING_CONDITIONS])
    axes[0].set_title("High Cross/Split fraction")
    axes[1].bar(x, [lookup.loc[s, "low_lump_fraction"] for s in SHARING_CONDITIONS], color=[COLORS[s] for s in SHARING_CONDITIONS])
    axes[1].set_title("Low Lump fraction")
    seed = hidden[hidden.row_type.eq("seed_mean")]
    for condition, marker in (("High_monotonic", "o"), ("Low_monotonic", "s")):
        summary = seed[seed.condition.eq(condition)].groupby("environment_condition").mean_matched_position_hidden_distance.agg(["mean", "sem"])
        axes[2].errorbar(x, [summary.loc[s, "mean"] for s in SHARING_CONDITIONS],
                         yerr=[summary.loc[s, "sem"] for s in SHARING_CONDITIONS],
                         marker=marker, capsize=3, label=condition.replace("_monotonic", ""))
    axes[2].set_title("A/B hidden separation")
    axes[2].legend()
    for ax in axes:
        ax.set_xticks(x, SHARING_CONDITIONS)
        ax.set_xlabel("shared-cue condition")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    fig.suptitle("Figure 2 — Phase-A A/B phenotype", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_2_phaseA_AB_phenotype")


def plot_phase_a_transfer(gates: pd.DataFrame, transfer: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    seed = transfer[transfer.row_type.eq("seed_mean") & transfer.condition.eq("High_monotonic")]
    gate = gates.drop_duplicates("environment_condition", keep="last").set_index("environment_condition")
    for row, context in enumerate(("Mixed", "Novel")):
        one = seed[seed.probe_context.eq(context)]
        for col, (metric, title) in enumerate((
            ("NLL_deficit_over_worse_endpoint", "NLL deficit"),
            ("RT_deficit_over_worse_endpoint", "RT deficit"),
        )):
            ax = axes[row, col]
            summary = one.groupby("shared_fraction")[metric].agg(["mean", "sem"])
            ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o", capsize=3)
            for sharing in SHARING_CONDITIONS:
                if not bool(gate.loc[sharing, "gate_passed"]):
                    fraction = int(sharing[1:]) / 5
                    if fraction in summary.index:
                        ax.scatter(fraction, summary.loc[fraction, "mean"], marker="x", s=90, color="#D55E00")
            ax.axhline(0, color="black", lw=.7)
            ax.set_title(f"{context}: {title}")
            ax.set_xlabel("shared fraction")
    fig.suptitle("Figure 3 — High-scale Phase-A transfer (× = failed A/B gate)", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_3_phaseA_transfer")


def plot_frontier(hidden: pd.DataFrame, transfer: pd.DataFrame, output: Path) -> None:
    separation = hidden[hidden.row_type.eq("seed_mean") & hidden.condition.eq("High_monotonic")][
        ["environment_condition", "seed", "mean_matched_position_hidden_distance"]
    ]
    seed = transfer[transfer.row_type.eq("seed_mean") & transfer.condition.eq("High_monotonic")]
    joined = seed.merge(separation, on=["environment_condition", "seed"], how="left")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, context in zip(axes, ("Mixed", "Novel")):
        for sharing, one in joined[joined.probe_context.eq(context)].groupby("environment_condition"):
            ax.scatter(one.mean_matched_position_hidden_distance,
                       one.NLL_deficit_over_worse_endpoint,
                       color=COLORS[sharing], label=f"{sharing} ({int(sharing[1:]) * 20}%)", alpha=.72)
        ax.axhline(0, color="black", lw=.7)
        ax.set_xlabel("A/B matched-position hidden distance")
        ax.set_ylabel(f"{context} NLL deficit")
        ax.set_title(context)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 4 — Differentiation–transfer frontier (High scale)", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_4_differentiation_transfer_frontier")


def plot_e3(e3: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    if e3.empty:
        for ax in axes:
            _empty(ax)
    else:
        one = e3[e3.condition.eq("High_monotonic")]
        for ax, family in zip(axes, ("Mixed", "Novel")):
            summary = one[one.probe_family.eq(family)].groupby("shared_fraction").absolute_cue_by_layout_interaction_NLL.agg(["mean", "sem"])
            ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o", capsize=3)
            ax.set_title(family)
            ax.set_xlabel("shared fraction")
            ax.set_ylabel("|cue × layout interaction|, NLL")
    fig.suptitle("Figure 5 — Frozen E3 interaction across cue sharing", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_5_phaseA_E3_interaction")


def plot_capacity(seed: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if seed.empty:
        for ax in axes:
            _empty(ax)
    else:
        high = seed[seed.condition.eq("High_monotonic")]
        for ax, (metric, title) in zip(axes, (
            ("mean_matched_position_hidden_distance", "A/B differentiation"),
            ("Mixed_NLL_deficit", "Mixed NLL deficit"),
            ("Novel_NLL_deficit", "Novel NLL deficit"),
        )):
            for environment, one in high.groupby("environment_condition"):
                summary = one.groupby("hidden_size")[metric].agg(["mean", "sem"])
                ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o", capsize=3,
                            color=COLORS[environment], label=environment)
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32], ["8", "16", "32"])
            ax.set_xlabel("hidden size")
            ax.set_title(title)
            ax.legend(fontsize=8)
    fig.suptitle("Figure 6 — Capacity × environment map (High scale)", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "figure_6_capacity_map")


def plot_phase_diagram(gates: pd.DataFrame, seed: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if gates.empty or seed.empty:
        _empty(ax)
    else:
        final = gates.drop_duplicates(["environment_condition", "hidden_size"], keep="last")
        high = seed[seed.condition.eq("High_monotonic")]
        environments = sorted(high.environment_condition.unique(), key=lambda x: int(x[1:]))
        sizes = sorted(high.hidden_size.unique())
        image = np.full((len(sizes), len(environments)), np.nan)
        for y, hidden_size in enumerate(sizes):
            for x, environment in enumerate(environments):
                cell = high[high.hidden_size.eq(hidden_size) & high.environment_condition.eq(environment)]
                gate = final[final.hidden_size.eq(hidden_size) & final.environment_condition.eq(environment)]
                if cell.empty or gate.empty:
                    continue
                mixed = cell.Mixed_NLL_deficit.mean()
                novel = cell.Novel_NLL_deficit.mean()
                fraction = float(gate.high_cross_split_fraction.iloc[0])
                image[y, x] = mixed
                ax.text(x, y, f"CS={fraction:.2f}\nM={mixed:.2f}\nN={novel:.2f}",
                        ha="center", va="center", fontsize=8)
        shown = ax.imshow(image, cmap="viridis_r", aspect="auto")
        fig.colorbar(shown, ax=ax, label="Mixed NLL deficit")
        ax.set_xticks(range(len(environments)), [f"{e}\n{int(e[1:]) * 20}%" for e in environments])
        ax.set_yticks(range(len(sizes)), sizes)
        ax.set_xlabel("shared cue fraction")
        ax.set_ylabel("hidden size")
    fig.suptitle("Figure 7 — Phase diagram: Cross/Split fraction, Mixed and Novel deficits", fontsize=13)
    fig.tight_layout()
    _save(fig, output, "figure_7_phase_diagram")


def make_all_figures(
    config: FrontierConfig,
    cue_audit: pd.DataFrame,
    phase_a_gates: pd.DataFrame,
    phase_a_hidden: pd.DataFrame,
    phase_a_transfer: pd.DataFrame,
    e3: pd.DataFrame,
    phase_b_gates: pd.DataFrame,
    phase_b_seed: pd.DataFrame,
    output: Path,
) -> None:
    plot_cue_schematic(config, output)
    plot_phase_a_phenotype(phase_a_gates, phase_a_hidden, output)
    plot_phase_a_transfer(phase_a_gates, phase_a_transfer, output)
    plot_frontier(phase_a_hidden, phase_a_transfer, output)
    plot_e3(e3, output)
    plot_capacity(phase_b_seed, output)
    plot_phase_diagram(phase_b_gates, phase_b_seed, output)
