"""
srn.plotting
============
All figure-generation functions for both analysis pipelines.

Functions are pure side-effect routines: they receive DataFrames / dicts,
produce a figure, save it to *output_dir*, and close it.

Public API (sequential-all pipeline)
-------------------------------------
plot_mechanistic_summary, plot_pca_geometry
plot_collapse_dynamics, plot_normalized_collapse, plot_phase_portrait
plot_perturbation_dynamics, plot_weight_gradient_dynamics
plot_shared_pca_panels, plot_dual_decoding_dynamics
plot_hidden_size_sweep, plot_learning_rate_sweep, plot_collapse_driver_dynamics
plot_normalized_model_vs_human, plot_rich_poor_comparison,
plot_accuracy_entropy_curves, _plot_matched_geometry_generic,
plot_matched_geometry, plot_asymptote_geometry
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Colour palette helpers
# ---------------------------------------------------------------------------
_PAL = {
    "Rich":           "darkgreen",
    "Poor":           "firebrick",
    "Lazy+RichCtx":   "steelblue",
    "Rich+LazyCtx":   "darkorange",
    "Control (Rich)": "royalblue",
}

# Human-readable labels used in PCA plot titles and panel headers
_DISPLAY_NAMES = {
    "Rich":           "High Scale Network",
    "Poor":           "Low Scale Network",
    "Lazy+RichCtx":   "High Scale Context",
    "Rich+LazyCtx":   "Low Scale Context",
    "Control (Rich)": "Control (Rich)",
}
DISPLAY_NAMES = _DISPLAY_NAMES   # public alias for import by other modules


# =============================================================================
# Mechanistic / PCA geometry
# =============================================================================

def plot_mechanistic_summary(results_list: List[Dict], output_dir: Path) -> None:
    """Aggregate mechanistic metrics across participants and save bar plots."""
    if not results_list:
        return

    df = pd.DataFrame(results_list)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Context decodability
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x="model_type", y="context_decodability",
                hue="model_type", palette=_PAL, legend=False, ax=ax, errorbar="se")
    ax.set_title("Context Decodability (Hidden States)", fontsize=14)
    ax.set_ylabel("Classification Accuracy (Chance = 0.5)")
    ax.set_ylim(0.0, 1.1)
    ax.axhline(0.5, linestyle="--", color="gray")
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / "mech_decodability.png")
    plt.close(fig)

    # 2. Weight ratio
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x="model_type", y="weight_ratio_cx_in",
                hue="model_type", palette=_PAL, legend=False, ax=ax, errorbar="se")
    ax.set_title("Weight Norm Ratio (Context / Input)", fontsize=14)
    ax.set_ylabel("||W_context|| / ||W_input||")
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / "mech_weight_ratio.png")
    plt.close(fig)

    # 3. Effective dimensionality
    if "effective_dimensionality" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(data=df, x="model_type", y="effective_dimensionality",
                    hue="model_type", palette=_PAL, legend=False, ax=ax, errorbar="se")
        ax.set_title("Effective Dimensionality (Participation Ratio)", fontsize=14)
        ax.set_ylabel("Dimensionality")
        sns.despine()
        plt.tight_layout()
        fig.savefig(output_dir / "mech_eff_dim.png")
        plt.close(fig)

    # 4. Jacobian recurrent dynamics
    jac_cols = [c for c in df.columns if "jacobian_spectral_radius" in c]
    if jac_cols:
        df_jac = df.melt(
            id_vars=["model_type"], value_vars=jac_cols,
            var_name="Context", value_name="Spectral_Radius",
        )
        df_jac["Context"] = df_jac["Context"].str.replace("jacobian_spectral_radius_", "")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=df_jac, x="model_type", y="Spectral_Radius",
                    hue="Context", palette="viridis", ax=ax, errorbar="se")
        ax.set_title("Recurrent Dynamics Stability (Jacobian)", fontsize=14)
        ax.set_ylabel("Spectral Radius (Max Eigenvalue)")
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5, label="Chaotic Edge")
        sns.despine()
        plt.tight_layout()
        fig.savefig(output_dir / "mech_jacobian_dynamics.png")
        plt.close(fig)


def plot_pca_geometry(mech_stats: Dict, model_name: str, output_dir: Path) -> None:
    """Plot the 2-D PCA projection of hidden states for a single model instance."""
    X_pca = mech_stats["pca_coords"]
    y     = mech_stats["pca_labels"]

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Context"] = y

    unique_ctx = np.unique(y)
    num_ctx    = len(unique_ctx)
    seq_len    = len(y) // num_ctx if num_ctx > 0 else 0
    df_pca["Step"] = np.tile(np.arange(seq_len), num_ctx)

    fig, ax = plt.subplots(figsize=(8, 6))
    if num_ctx > 1:
        palette = sns.color_palette("husl", num_ctx)
        sns.scatterplot(data=df_pca, x="PC1", y="PC2",
                        hue="Context", style="Context", palette=palette, s=100, ax=ax)
    else:
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", color="purple", s=100, ax=ax)

    for i, ctx in enumerate(unique_ctx):
        color  = "purple" if num_ctx == 1 else sns.color_palette("husl", num_ctx)[i]
        subset = df_pca[df_pca["Context"] == ctx].sort_values("Step")
        ax.plot(subset["PC1"], subset["PC2"], color=color, alpha=0.5)

    ax.set_title(f"Hidden State Geometry ({_DISPLAY_NAMES.get(model_name, model_name)})", fontsize=16)
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / f"mech_pca_{model_name}.png")
    plt.close(fig)


# =============================================================================
# Collapse dynamics (PRD3)
# =============================================================================

def plot_collapse_dynamics(collapse_df: pd.DataFrame, output_dir: Path) -> None:
    """Four-panel figure: trajectory_separation, decodability, dimensionality, loss."""
    if collapse_df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL

    metrics = [
        ("trajectory_separation",     "Trajectory Separation"),
        ("context_decodability",       "Context Decodability"),
        ("sequence_step_decodability", "Seq-Step Decodability"),
        ("effective_dimensionality",   "Effective Dimensionality"),
        ("collapse_index",             "Collapse Index"),
        ("loss",                       "CE Loss"),
    ]
    avail = [(c, l) for c, l in metrics if c in collapse_df.columns]
    if not avail:
        return

    ncols = 3
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    x_col = "step_or_epoch" if "step_or_epoch" in collapse_df.columns else "checkpoint_value"
    for ax_i, (col, label) in enumerate(avail):
        ax      = axes[ax_i]
        plot_df = collapse_df[collapse_df[col].notna()]
        for mt, grp in plot_df.groupby("model_type"):
            agg = grp.groupby(x_col)[col].mean().reset_index()
            ax.plot(agg[x_col], agg[col], label=mt,
                    color=pal.get(mt, "steelblue"), linewidth=1.5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Step / Epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    for ax_i in range(len(avail), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Collapse Dynamics Over Training", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "collapse_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_collapse(collapse_df: pd.DataFrame, output_dir: Path) -> None:
    """Trajectory separation normalised per participant and aligned to its peak."""
    if collapse_df.empty or "trajectory_separation" not in collapse_df.columns:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL

    x_col = "step_or_epoch" if "step_or_epoch" in collapse_df.columns else "checkpoint_value"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mt in zip(axes, ["Rich", "Poor"]):
        sub = collapse_df[collapse_df["model_type"] == mt].copy()
        if sub.empty:
            ax.set_title(f"{mt} (no data)")
            continue
        groups = (
            sub.groupby("participant_id")
            if "participant_id" in sub.columns
            else [(mt, sub)]
        )
        for _, grp in groups:
            s = grp.sort_values(x_col)["trajectory_separation"].values
            if s.max() < 1e-9:
                continue
            s_norm   = s / s.max()
            peak_idx = np.argmax(s)
            aligned  = s_norm - s_norm[peak_idx]
            xvals    = np.arange(len(aligned)) - peak_idx
            ax.plot(xvals, aligned, alpha=0.4, color=pal.get(mt, "steelblue"), linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{mt} — Normalised Separation (aligned to peak)", fontsize=11)
        ax.set_xlabel("Steps relative to peak")
        ax.set_ylabel("Normalised trajectory separation")

    plt.tight_layout()
    fig.savefig(output_dir / "collapse_normalised.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_phase_portrait(collapse_df: pd.DataFrame, output_dir: Path) -> None:
    """Phase portrait: CE loss (x) vs trajectory separation (y)."""
    if collapse_df.empty:
        return
    needed = {"loss", "trajectory_separation"}
    if not needed.issubset(collapse_df.columns):
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL
    fig, ax = plt.subplots(figsize=(7, 6))

    for mt, grp in collapse_df.groupby("model_type"):
        ax.scatter(grp["loss"], grp["trajectory_separation"],
                   label=mt, color=pal.get(mt, "steelblue"), alpha=0.4, s=15)
        agg = grp.dropna(subset=["loss", "trajectory_separation"])
        if len(agg) > 3:
            bins  = pd.cut(agg["loss"], bins=10)
            trend = agg.groupby(bins)["trajectory_separation"].mean()
            mids  = [b.mid for b in trend.index]
            ax.plot(mids, trend.values, color=pal.get(mt, "steelblue"), linewidth=2)

    ax.set_xlabel("CE Loss")
    ax.set_ylabel("Trajectory Separation")
    ax.set_title("Phase Portrait: Loss vs Context Separation", fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "collapse_phase_portrait.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_perturbation_dynamics(perturbation_df: pd.DataFrame, output_dir: Path) -> None:
    """Δ-loss and hidden-state shift per perturbation mode over checkpoints."""
    if perturbation_df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")

    for metric, ylabel in [
        ("delta_loss",         "Δ CE Loss (perturbed − baseline)"),
        ("hidden_state_shift", "Mean Hidden-State Shift (L2)"),
    ]:
        if metric not in perturbation_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        x_col = "checkpoint_value" if "checkpoint_value" in perturbation_df.columns else "global_step"
        if x_col not in perturbation_df.columns:
            plt.close(fig)
            continue
        for (mt, mode), grp in perturbation_df.groupby(["model_type", "perturbation_mode"]):
            agg   = grp.groupby(x_col)[metric].mean().reset_index()
            label = f"{mt} / {mode}"
            ax.plot(agg[x_col], agg[metric], label=label, linewidth=1.5)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Perturbation Analysis — {ylabel}", fontsize=12)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = ("context_perturbation_delta_loss.png" if metric == "delta_loss"
                 else "context_perturbation_hidden_shift.png")
        fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_weight_gradient_dynamics(wg_df: pd.DataFrame, output_dir: Path) -> None:
    """Weight norms and gradient norms over asymptote training epochs."""
    if wg_df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL

    weight_metrics = [
        ("norm_context_weights",   "||W_context||"),
        ("norm_input_weights",     "||W_input||"),
        ("norm_recurrent_weights", "||W_recurrent||"),
        ("context_input_ratio",    "Context / Input Ratio"),
        ("norm_output_weights",    "||W_output||"),
    ]
    grad_metrics = [
        ("grad_context",   "Grad: context weights"),
        ("grad_recurrent", "Grad: recurrent weights"),
        ("grad_output",    "Grad: output weights"),
    ]
    avail = [(c, l) for c, l in (weight_metrics + grad_metrics) if c in wg_df.columns]
    if not avail:
        return

    ncols = 4
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    x_col = "epoch"

    for ax_i, (col, label) in enumerate(avail):
        ax = axes[ax_i]
        for mt, grp in wg_df.groupby("model_type"):
            agg = grp.groupby(x_col)[col].mean().reset_index()
            ax.plot(agg[x_col], agg[col], label=mt,
                    color=pal.get(mt, "steelblue"), linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    for ax_i in range(len(avail), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Weight & Gradient Dynamics (Asymptote Training)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "weight_gradient_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shared_pca_panels(shared_pca_df: pd.DataFrame, output_dir: Path) -> None:
    """Early / mid / late / asymptote hidden-state trajectories on shared PCA axes."""
    if shared_pca_df.empty or "PC1" not in shared_pca_df.columns:
        return
    plt.style.use("seaborn-v0_8-whitegrid")

    for mt in shared_pca_df["model_type"].unique():
        sub         = shared_pca_df[shared_pca_df["model_type"] == mt]
        checkpoints = sorted(sub["checkpoint_value"].unique())
        n_panels    = min(len(checkpoints), 4)
        idxs        = np.linspace(0, len(checkpoints) - 1, n_panels, dtype=int)
        chosen      = [checkpoints[i] for i in idxs]

        fig, axes   = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
        axes        = axes[0]
        ctx_list    = sorted(sub["context"].unique())
        colors      = sns.color_palette("husl", len(ctx_list))

        for ax, ckpt in zip(axes, chosen):
            cp_sub = sub[sub["checkpoint_value"] == ckpt]
            for ci, ctx in enumerate(ctx_list):
                c_sub = cp_sub[cp_sub["context"] == ctx].sort_values("step_in_sequence")
                ax.plot(c_sub["PC1"], c_sub["PC2"], color=colors[ci],
                        label=ctx, linewidth=1.5, marker="o", markersize=4)
            ax.set_title(f"Checkpoint {ckpt}", fontsize=10)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(fontsize=7)

        fig.suptitle(f"Shared PCA — {_DISPLAY_NAMES.get(mt, mt)}", fontsize=13)
        plt.tight_layout()
        safe = mt.replace(" ", "_").replace("/", "_")
        fig.savefig(output_dir / f"shared_pca_{safe}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_dual_decoding_dynamics(collapse_df: pd.DataFrame, output_dir: Path) -> None:
    """Context decodability vs sequence-step decodability over training."""
    needed = {"context_decodability", "sequence_step_decodability"}
    if collapse_df.empty or not needed.issubset(collapse_df.columns):
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL

    x_col       = "step_or_epoch" if "step_or_epoch" in collapse_df.columns else "checkpoint_value"
    model_types = collapse_df["model_type"].unique()
    n           = len(model_types)
    fig, axes   = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for ax, mt in zip(axes[0], model_types):
        sub = collapse_df[collapse_df["model_type"] == mt]
        agg = sub.groupby(x_col)[
            ["context_decodability", "sequence_step_decodability"]
        ].mean().reset_index()
        color = pal.get(mt, "steelblue")
        ax.plot(agg[x_col], agg["context_decodability"],
                label="Context decodability", color=color, linewidth=1.8)
        ax.plot(agg[x_col], agg["sequence_step_decodability"],
                label="Seq-step decodability", color=color, linewidth=1.8, linestyle="--")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(f"{mt}", fontsize=12)
        ax.set_xlabel("Step / Epoch")
        ax.set_ylabel("Decoding Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    fig.suptitle("Context vs Sequence-Step Decodability Over Training", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "dual_decoding_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# PRD4 robustness sweep plots
# =============================================================================

def plot_hidden_size_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    """PRD4 Part 1: peak / final separation and compression vs hidden_dim."""
    if df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL
    panels = [
        ("peak_separation",    "Peak Separation"),
        ("final_separation",   "Final Separation"),
        ("compression_amount", "Compression Amount (peak − final)"),
        ("final_loss",         "Final CE Loss"),
    ]
    avail = [(c, l) for c, l in panels if c in df.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, avail):
        for mt, grp in df.groupby("model_type"):
            grp_s = grp.sort_values("hidden_dim")
            ax.plot(grp_s["hidden_dim"], grp_s[col], marker="o",
                    label=mt, color=pal.get(mt, "steelblue"), linewidth=1.8)
        ax.set_xlabel("Hidden Dimension")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle("Hidden-Size Robustness Sweep", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "hidden_size_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_learning_rate_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    """PRD4 Part 2: peak / final separation and compression vs learning_rate."""
    if df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL
    panels = [
        ("peak_separation",    "Peak Separation"),
        ("final_separation",   "Final Separation"),
        ("compression_amount", "Compression Amount (peak − final)"),
        ("final_loss",         "Final CE Loss"),
    ]
    avail = [(c, l) for c, l in panels if c in df.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, avail):
        for mt, grp in df.groupby("model_type"):
            grp_s = grp.sort_values("learning_rate")
            ax.plot(grp_s["learning_rate"], grp_s[col], marker="o",
                    label=mt, color=pal.get(mt, "steelblue"), linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle("Learning-Rate Robustness Sweep", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "learning_rate_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_collapse_driver_dynamics(driver_df: pd.DataFrame, output_dir: Path) -> None:
    """PRD4 Part 3: gradient alignment, JS divergence, separation, prob-gap over training."""
    if driver_df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal   = {"Rich": "darkgreen", "Poor": "firebrick"}
    x_col = "checkpoint_value" if "checkpoint_value" in driver_df.columns else "step_or_epoch"
    panels = [
        ("gradient_alignment",     "Gradient Alignment\n(cosine sim, ctx A vs B)"),
        ("output_js_divergence",   "Output JS Divergence\n(ctx A vs B)"),
        ("trajectory_separation",  "Trajectory Separation"),
        ("correct_token_prob_gap", "Correct-Token Prob Gap"),
    ]
    avail = [(c, l) for c, l in panels if c in driver_df.columns]
    if not avail:
        return
    ncols = min(len(avail), 4)
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_i, (col, label) in enumerate(avail):
        ax  = axes[ax_i]
        sub = driver_df[driver_df[col].notna()]
        for mt, grp in sub.groupby("model_type"):
            agg = grp.groupby(x_col)[col].mean().reset_index()
            ax.plot(agg[x_col], agg[col], label=mt,
                    color=pal.get(mt, "steelblue"), linewidth=1.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    for ax_i in range(len(avail), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Collapse Driver Dynamics", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "collapse_driver_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Learning-curve plots (sequential-all pipeline)
# =============================================================================

def plot_normalized_model_vs_human(data_input, output_dir: Path) -> None:
    """Normalised RT (human) and CE loss (model) trajectories over training."""
    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise ValueError("Input data must be a pandas DataFrame")

    group_cols = ["model_type", "context", "block_id", "sequence_in_block"]
    agg_df = (
        df.groupby(group_cols)[["loss", "transfer_loss", "rt", "global_step"]]
        .mean()
        .reset_index()
        .sort_values(["block_id", "sequence_in_block"])
    )

    for m_type in df["model_type"].unique():
        model_data = agg_df[agg_df["model_type"] == m_type].copy()

        def normalize(series):
            rng = series.max() - series.min()
            return (series - series.min()) / rng if rng > 0 else series * 0

        model_data["norm_rt"]   = normalize(model_data["rt"])
        model_data["norm_loss"] = normalize(model_data["loss"])

        fig, ax = plt.subplots(figsize=(14, 8))

        def plot_series(sub_df, y_col, color, label, marker):
            first = True
            for b_id in sorted(sub_df["block_id"].unique()):
                bd = sub_df[sub_df["block_id"] == b_id].sort_values("sequence_in_block")
                ax.plot(bd["global_step"].values, bd[y_col].values,
                        color=color, label=label if first else None,
                        marker=marker, linewidth=2, alpha=0.9)
                first = False

        maze_df   = model_data[model_data["context"] == "Maze"]
        melody_df = model_data[model_data["context"] == "Melody"]

        plot_series(maze_df,   "norm_rt",   "lightblue", "Maze Human RT",    "o")
        plot_series(maze_df,   "norm_loss", "darkblue",  "Maze Model Loss",  None)
        plot_series(melody_df, "norm_rt",   "salmon",    "Melody Human RT",  "s")
        plot_series(melody_df, "norm_loss", "darkred",   "Melody Model Loss", None)

        ax.set_title(
            f"Normalized Trajectories: {m_type} Model (Solid) vs Human (Markers)",
            fontsize=16,
        )
        ax.set_xlabel("Cumulative Sequence Index", fontsize=12)
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(output_dir / f"{m_type}_normalized_trajectories.png", dpi=300)
        plt.close(fig)


def plot_rich_poor_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Generalization gap and learning curves for Rich vs Poor models."""
    agg_df = (
        df.groupby(["model_type", "global_step"])[["loss", "transfer_loss", "rt"]]
        .mean()
        .reset_index()
    )
    agg_df["generalization_gap"] = agg_df["transfer_loss"] - agg_df["loss"]
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Generalization gap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=agg_df, x="global_step", y="generalization_gap",
                 hue="model_type", palette=_PAL, linewidth=1.5, ax=ax)
    ax.set_title("Context Specificity (Generalization Gap)", fontsize=16)
    ax.set_ylabel("Transfer Loss − Training Loss")
    ax.set_xlabel("Sequence Index")
    ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
    fig.savefig(output_dir / "generalization_gap_analysis.png", dpi=300)
    plt.close(fig)

    # 2. Learning curves (loss)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=agg_df, x="global_step", y="loss",
                 hue="model_type", palette=_PAL, linewidth=1.5, ax=ax)
    ax.set_title("Learning Dynamics (Cross Entropy Loss)", fontsize=16)
    fig.savefig(output_dir / "learning_curves.png", dpi=300)
    plt.close(fig)


def plot_accuracy_entropy_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """Five-panel figure: loss, accuracy, entropy, confidence, correct-trial surprisal."""
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL

    metrics = [
        ("loss",              "Mean CE Loss",
                               "CE loss (↓ = better)"),
        ("accuracy",          "Accuracy",
                               "Fraction correct (↑ = better)"),
        ("entropy",           "Prediction Entropy",
                               "Output entropy / nats (↓ = more certain)"),
        ("confidence",        "Prediction Confidence",
                               "Max-softmax probability (↑ = more confident)"),
        ("correct_surprisal", "Correct-Trial Surprisal\n(RT analogue)",
                               "−log p(correct) on correct trials (↓ = less surprised)"),
    ]

    agg = (
        df.groupby(["model_type", "global_step"])[
            [m for m, _, _ in metrics]
        ].mean().reset_index()
    )

    nrows, ncols = 2, 3
    fig, axes    = plt.subplots(nrows, ncols, figsize=(18, 10))
    axes         = axes.flatten()

    for ax_i, (col, title, ylabel) in enumerate(metrics):
        ax = axes[ax_i]
        if col not in agg.columns:
            ax.set_visible(False)
            continue
        sns.lineplot(data=agg, x="global_step", y=col,
                     hue="model_type", palette=pal, linewidth=1.5, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Sequence Index")
        ax.set_ylabel(ylabel)
        if ax.get_legend():
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Model", fontsize=10, loc="center")
    axes[-1].set_visible(True)
    axes[-1].axis("off")

    fig.suptitle("Learning Trajectories — Loss, Accuracy, Entropy & Confidence",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "learning_curves_full.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Performance-matched geometry plots
# =============================================================================

def _plot_matched_geometry_generic(
    matched_df: pd.DataFrame,
    threshold_col: str,
    threshold_label: str,
    save_path: Path,
) -> None:
    """Generic matched-geometry line plot for any threshold column."""
    if matched_df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL
    geo_metrics = [
        ("context_decodability",     "Context Decodability"),
        ("weight_ratio_cx_in",       "Context / Input Weight Ratio"),
        ("effective_dimensionality", "Effective Dimensionality"),
        ("trajectory_separation",    "PCA Trajectory Separation"),
    ]
    geo_metrics = [(c, l) for c, l in geo_metrics if c in matched_df.columns]
    if not geo_metrics:
        return

    n = len(geo_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, geo_metrics):
        for mt, grp in matched_df.groupby("model_type"):
            agg = grp.groupby(threshold_col)[col].agg(["mean", "sem"]).reset_index()
            ax.plot(agg[threshold_col], agg["mean"], marker="o", label=mt,
                    color=pal.get(mt, "steelblue"), linewidth=1.8)
            ax.fill_between(agg[threshold_col],
                            agg["mean"] - agg["sem"],
                            agg["mean"] + agg["sem"],
                            color=pal.get(mt, "steelblue"), alpha=0.15)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(threshold_label)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle("Representational Geometry at Matched Performance", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_matched_geometry(matched_df: pd.DataFrame, output_dir: Path) -> None:
    """Compare geometry metrics at matched CE performance levels."""
    if matched_df.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = _PAL

    geo_metrics = [
        ("context_decodability",     "Context Decodability"),
        ("weight_ratio_cx_in",       "Context / Input Weight Ratio"),
        ("effective_dimensionality", "Effective Dimensionality"),
        ("trajectory_separation",    "PCA Trajectory Separation"),
    ]
    geo_metrics = [(c, l) for c, l in geo_metrics if c in matched_df.columns]
    if not geo_metrics:
        return

    n = len(geo_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, geo_metrics):
        for mt, grp in matched_df.groupby("model_type"):
            agg = grp.groupby("ce_threshold")[col].agg(["mean", "sem"]).reset_index()
            ax.plot(agg["ce_threshold"], agg["mean"], marker="o", label=mt,
                    color=pal.get(mt, "steelblue"), linewidth=1.8)
            ax.fill_between(agg["ce_threshold"],
                            agg["mean"] - agg["sem"],
                            agg["mean"] + agg["sem"],
                            color=pal.get(mt, "steelblue"), alpha=0.15)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("CE threshold (matched performance)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        if col == "context_decodability":
            ax.axhline(0.5, linestyle="--", color="gray", alpha=0.4)

    fig.suptitle("Representational Geometry at Matched Predictive Performance",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "matched_geometry.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_asymptote_geometry(asymptote_mechs: List[Dict], output_dir: Path) -> None:
    """Compare geometry metrics (Rich vs Poor) after training to asymptote."""
    if not asymptote_mechs:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    df  = pd.DataFrame(asymptote_mechs)
    pal = _PAL

    geo_metrics = [
        ("context_decodability",     "Context Decodability"),
        ("weight_ratio_cx_in",       "Context / Input Weight Ratio"),
        ("effective_dimensionality", "Effective Dimensionality"),
        ("trajectory_separation",    "PCA Trajectory Separation"),
    ]
    geo_metrics = [(c, l) for c, l in geo_metrics if c in df.columns]
    if not geo_metrics:
        return

    n = len(geo_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, geo_metrics):
        sns.barplot(data=df, x="model_type", y=col,
                    hue="model_type", palette=pal, legend=False, errorbar="se", ax=ax)
        sns.stripplot(data=df, x="model_type", y=col,
                      color="black", alpha=0.3, size=4, ax=ax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(label)

    fig.suptitle("Representational Geometry After Training to Asymptote", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "asymptote_geometry.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
