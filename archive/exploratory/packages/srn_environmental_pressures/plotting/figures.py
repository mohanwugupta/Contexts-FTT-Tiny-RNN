"""Nine required environmental-pressure figures, including staged placeholders."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _completed(frame):
    if frame.empty:
        return frame
    if "status" in frame:
        return frame[frame.status.eq("completed")]
    return frame


def _placeholder(ax, frame, title):
    reason = "stage not run"
    if not frame.empty and "reason" in frame and pd.notna(frame.reason.iloc[0]):
        reason = str(frame.reason.iloc[0])
    ax.text(0.5, 0.5, f"Not run\n{reason}", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()


def _ecology(path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    labels = ["Nominal A\n10 trials", "Nominal B\n10 trials", "Nominal A\n10 trials", "Nominal B\n10 trials"]
    xs = np.linspace(0.12, 0.88, 4)
    colors = ["#4c78a8", "#e45756"] * 2
    for x, label, color in zip(xs, labels, colors):
        ax.text(x, 0.68, label, ha="center", va="center", color="white",
                bbox=dict(boxstyle="round,pad=.6", fc=color, ec=color))
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", (right - .09, .68), (left + .09, .68), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(.5, .25, "nominal c → probabilistic latent z → graded cue x → SRN prediction → outcome feedback",
            ha="center", va="center", fontsize=12)
    ax.set_title("Fixed FTT training ecology")
    _finish(fig, path)


def _why_bayes(ambiguity, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if ambiguity.empty:
        _placeholder(ax, ambiguity, "Why posterior inference is useful")
    else:
        row = ambiguity.iloc[(ambiguity.q_star - 0.5).abs().argmin()]
        values = [row.prior_q, row.cue_only_q, row.q_star]
        ax.bar(["Block prior", "Cue only", "Full posterior"], values,
               color=["#4c78a8", "#f28e2b", "#59a14f"])
        ax.axhline(.5, color="black", lw=1, ls="--")
        ax.set_ylim(0, 1)
        ax.set(ylabel="P(latent state A)", title=f"Prior + likelihood → posterior (x={row.x_t:.2f})")
        ax.text(1.0, .05, f"Bayes mean = {row.bayes_mean:.3f}", ha="center")
    _finish(fig, path)


def _emergence(rows, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    completed = _completed(rows)
    if completed.empty:
        _placeholder(ax, rows, "Emergence by completed A/B pair")
    else:
        ranked = completed.groupby("condition_id").posterior_mean_r2.mean().nlargest(6).index
        for condition_id, group in completed[completed.condition_id.isin(ranked)].groupby("condition_id"):
            summary = group.groupby("switch_number").posterior_mean_r2.mean()
            ax.plot(summary.index, summary.values, marker="o", label=condition_id)
        ax.axhline(.8, color="black", ls="--", lw=1)
        ax.set(xlabel="Completed A/B block pairs", ylabel="Posterior-mean R²",
               title="Emergence under fixed training exposure")
        ax.legend(frameon=False, fontsize=7)
    _finish(fig, path)


def _heatmap(frame, x, y, value, title, path):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    completed = _completed(frame)
    if completed.empty:
        _placeholder(ax, frame, title)
    else:
        pivot = completed.pivot(index=y, columns=x, values=value).sort_index(ascending=False)
        image = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis", vmin=-.2, vmax=1)
        ax.set_xticks(range(len(pivot.columns)), [f"{item:.2f}" for item in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{item:.2f}" for item in pivot.index])
        ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), title=title)
        fig.colorbar(image, ax=ax, label="Mean posterior-mean R²")
    _finish(fig, path)


def _calibration(ambiguity, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    if ambiguity.empty:
        _placeholder(ax, ambiguity, "q* calibration")
    else:
        summary = ambiguity.groupby("q_star").implied_q.agg(["mean", "std"]).reset_index()
        ax.errorbar(summary.q_star, summary["mean"], yerr=summary["std"].fillna(0), marker="o", capsize=3)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlabel="Full Bayes q*", ylabel="SRN-implied q", title="Posterior calibration")
    _finish(fig, path)


def _shortcuts(similarity, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    completed = _completed(similarity)
    if completed.empty:
        _placeholder(ax, similarity, "Bayes versus shortcut models")
    else:
        summary = completed.groupby("candidate_model").srn_similarity_mse.mean().sort_values()
        ax.bar(summary.index, summary.values, color="#4c78a8")
        ax.set(ylabel="MSE from SRN prediction (lower is closer)", title="Bayes versus shortcuts")
        ax.tick_params(axis="x", rotation=25)
    _finish(fig, path)


def _ablations(rows, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    completed = _completed(rows)
    if completed.empty:
        _placeholder(ax, rows, "Environmental ablations")
    else:
        summary = completed.groupby("manipulation").posterior_mean_r2.mean().sort_values()
        ax.bar(summary.index, summary.values, color="#e45756")
        ax.axhline(.8, color="black", ls="--", lw=1)
        ax.set(ylabel="Posterior-mean R²", title="Necessity/follow-up environments")
        ax.tick_params(axis="x", rotation=35)
    _finish(fig, path)


def _confirmation(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    completed = _completed(rows)
    if completed.empty:
        _placeholder(ax, rows, "Confirmatory replication")
    else:
        conditions = list(completed.condition_id.unique())
        values = [completed.loc[completed.condition_id.eq(condition), "posterior_mean_r2"].to_numpy()
                  for condition in conditions]
        ax.boxplot(values, tick_labels=conditions, showmeans=True)
        ax.axhline(.8, color="black", ls="--", lw=1)
        ax.set(ylabel="Posterior-mean R²", title="Fresh-seed confirmation")
        ax.tick_params(axis="x", rotation=25, labelsize=7)
    _finish(fig, path)


def make_figures(output_dir: str | Path, frames: dict):
    directory = Path(output_dir) / "figures_environmental_pressures"
    directory.mkdir(parents=True, exist_ok=True)
    _ecology(directory / "figure_1_fixed_ftt_ecology.png")
    _why_bayes(frames["ambiguity_diagnostics"], directory / "figure_2_why_bayes.png")
    _emergence(frames["emergence_by_switch"], directory / "figure_3_emergence_by_switch.png")
    _heatmap(frames["phase_diagram_validity_x_uncertainty"], "cue_accuracy", "context_validity",
             "mean_posterior_mean_r2", "Context validity × cue accuracy",
             directory / "figure_4_validity_x_uncertainty.png")
    _heatmap(frames["phase_diagram_uncertainty_x_separation"], "state_separation", "cue_accuracy",
             "mean_posterior_mean_r2", "Cue accuracy × state separation",
             directory / "figure_5_uncertainty_x_separation.png")
    _calibration(frames["ambiguity_diagnostics"], directory / "figure_6_q_calibration.png")
    _shortcuts(frames["model_similarity_comparison"], directory / "figure_7_bayes_vs_shortcuts.png")
    _ablations(frames["environmental_ablation_results"], directory / "figure_8_ablations.png")
    _confirmation(frames["confirmation_results"], directory / "figure_9_confirmation.png")

