"""Compact, deterministic rendering of the nine v2.1 figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "A": "#2b6cb0", "B": "#c53030", "Mixed_norm": "#805ad5",
    "Mixed_additive": "#b7791f", "Novel": "#2f855a",
    "High": "#6b46c1", "Low": "#718096", "High_seq_only": "#d69e2e",
}


def _finish(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _architecture(path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    boxes = [
        (0.3, 3.9, 2.2, 1.1, "4D token\none-hot"),
        (0.3, 1.5, 2.2, 1.1, "3D context\nbasis"),
        (3.7, 2.7, 2.6, 1.4, "8-unit tanh\nElman SRN"),
        (7.3, 4.1, 2.2, 1.1, "4-logit\nnext-token head"),
        (7.3, 1.3, 2.2, 1.1, "sigmoid\nexecution-speed head"),
    ]
    for x, y, width, height, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.08",
                                    facecolor="#edf2f7", edgecolor="#4a5568"))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    for start, end in [((2.5, 4.45), (3.7, 3.6)), ((2.5, 2.05), (3.7, 3.2)),
                       ((6.3, 3.6), (7.3, 4.65)), ((6.3, 3.2), (7.3, 1.85))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.set_title("FTT SRN with a shared hidden state and two readout heads")
    _finish(fig, path)


def _splitting(metrics: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    final = metrics[metrics.checkpoint == "final"]
    columns = ["ab_hidden_separation", "sequence_transition_cost",
               "absolute_cross_context_disruption"]
    labels = ["A/B hidden\nseparation", "switch CE\ncost", "cross-context\ndisruption"]
    means = final.groupby("model_condition")[columns].mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, column, label in zip(axes, columns, labels):
        conditions = [name for name in ("High", "Low") if name in means.index]
        ax.bar(conditions, [means.loc[name, column] for name in conditions],
               color=[COLORS[name] for name in conditions])
        ax.set_title(label); ax.axhline(0, color="#4a5568", lw=.7)
    fig.suptitle("Stage-1 representational splitting prerequisite")
    _finish(fig, path)


def _training(execution: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    data = execution[execution.model_condition == "High"].groupby(
        ["global_sequence", "context"], as_index=False
    )[["correct_token_confidence", "execution_speed", "normalized_rt"]].mean()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["correct_token_confidence", "execution_speed", "normalized_rt"],
        ["Correct-token confidence", "Execution speed", "Normalized RT (1 − speed)"],
    ):
        for context, group in data.groupby("context"):
            ax.plot(group.global_sequence, group[metric], label=context,
                    color=COLORS[context], lw=1.5)
        ax.set_ylabel(title); ax.set_ylim(-.02, 1.02)
    axes[0].legend(frameon=False, ncol=2); axes[-1].set_xlabel("Training sequence")
    fig.suptitle("High-Scale learning and emergent execution control")
    _finish(fig, path)


def _calibration(calibration: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(calibration.sequence_confidence, calibration.optimal_speed, color="#2b6cb0", lw=2)
    interior = calibration[calibration.interior_optimum]
    ax.scatter(interior.sequence_confidence, interior.optimal_speed, s=18,
               color="#dd6b20", label="interior optimum")
    ax.set(xlabel="Correct-token confidence", ylabel="Cost-minimizing speed",
           xlim=(0, 1), ylim=(0, 1), title="Pre-probe speed–accuracy calibration")
    ax.legend(frameon=False)
    _finish(fig, path)


def _probe_bar(probes: pd.DataFrame, contexts: list[str], title: str, path: Path):
    import matplotlib.pyplot as plt
    data = probes[(probes.model_condition == "High") & (probes.checkpoint == "final")]
    means = data.groupby("context").normalized_rt.mean()
    sems = data.groupby("context").normalized_rt.sem().fillna(0)
    available = [context for context in contexts if context in means.index]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(available, [means[c] for c in available], yerr=[sems[c] for c in available],
           color=[COLORS[c] for c in available], capsize=3)
    ax.set(ylabel="Normalized RT", title=title)
    _finish(fig, path)


def _structural(interpolation: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    data = interpolation[(interpolation.model_condition == "High") &
                         (interpolation.checkpoint == "final") &
                         (interpolation.probe_type.isin(["Mixed_norm", "Novel"]))]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for probe, group in data.groupby("probe_type"):
        ax.scatter(group.rt_endpoint_signed_separation, group.midpoint_deviation,
                   label=probe, color=COLORS[probe], alpha=.8)
    ax.axhline(0, color="#718096", lw=.8); ax.axvline(0, color="#718096", lw=.8)
    ax.set(xlabel="Within-network RT(A) − RT(B)",
           ylabel="Probe RT − within-network midpoint",
           title="Seed-level structural prediction")
    ax.legend(frameon=False)
    _finish(fig, path)


def _hidden_pca(hidden: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    subset = hidden[(hidden.model_condition == "High") & (hidden.checkpoint == "final")]
    if not subset.empty:
        subset = subset[subset.seed == subset.seed.min()]
    hidden_columns = sorted([c for c in subset if c.startswith("h") and c[1:].isdigit()],
                            key=lambda value: int(value[1:]))
    matrix = subset[hidden_columns].to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0) if len(matrix) else np.zeros((1, 8))
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vh[:2].T
    subset = subset.copy(); subset["pc1"] = coordinates[:, 0]; subset["pc2"] = coordinates[:, 1]
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    for context in ("A", "B", "Mixed_norm", "Novel"):
        group = subset[subset.context == context].sort_values("token_position")
        if not group.empty:
            ax.plot(group.pc1, group.pc2, marker="o", ms=4, label=context,
                    color=COLORS[context], alpha=.85)
    ax.set(xlabel="Shared PC1", ylabel="Shared PC2", title="Frozen hidden trajectories (one final High seed)")
    ax.legend(frameon=False)
    _finish(fig, path)


def _factorial(metrics: pd.DataFrame, probes: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    final = metrics[metrics.checkpoint == "final"]
    probe = probes[(probes.checkpoint == "final") & (probes.context == "Mixed_norm")]
    order = [name for name in ("Low", "High_seq_only", "High")
             if name in set(final.model_condition)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    split = final.groupby("model_condition").ab_hidden_separation.mean()
    speed = probe.groupby("model_condition").execution_speed.mean()
    axes[0].bar(order, [split.get(name, np.nan) for name in order],
                color=[COLORS[name] for name in order]); axes[0].set_title("A/B hidden separation")
    axes[1].bar(order, [speed.get(name, np.nan) for name in order],
                color=[COLORS[name] for name in order]); axes[1].set_title("Mixed probe speed")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Initialization scale × execution-objective control")
    _finish(fig, path)


def make_required_figures(output_dir, *, calibration, training_dynamics,
                          execution_dynamics, splitting_metrics, probes,
                          interpolation, hidden_states):
    del training_dynamics  # token-level execution dynamics contain the plotted quantities.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _architecture(output / "figure_01_model_architecture.png")
    _splitting(splitting_metrics, output / "figure_02_splitting_prerequisite.png")
    _training(execution_dynamics, output / "figure_03_confidence_execution_training.png")
    _calibration(calibration, output / "figure_04_speed_accuracy_calibration.png")
    _probe_bar(probes, ["A", "B", "Mixed_norm", "Mixed_additive"],
               "Pure-context endpoints and Mixed probes", output / "figure_05_mixed_probe.png")
    _probe_bar(probes, ["A", "B", "Novel"],
               "Pure-context endpoints and Novel probe", output / "figure_06_novel_probe.png")
    _structural(interpolation, output / "figure_07_structural_prediction.png")
    _hidden_pca(hidden_states, output / "figure_08_hidden_geometry.png")
    _factorial(splitting_metrics, probes, output / "figure_09_scale_objective_comparison.png")
