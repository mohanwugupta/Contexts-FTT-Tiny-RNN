"""The ten PRD v2.3 context-by-sequence figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


COLORS = {
    "A": "#2b6cb0", "B": "#c53030", "Mixed_norm": "#805ad5",
    "Novel": "#2f855a", "full": "#4c51bf", "shared": "#718096",
    "specific": "#d69e2e",
}


def _finish(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _hypothesis(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    boxes = (
        (0.2, 3.1, 2.2, .85, "Sequence position"),
        (0.2, 1.1, 2.2, .85, "Context"),
        (4.0, 2.1, 3.0, 1.0, "Configured\nmotor state"),
        (8.0, 2.1, 1.8, 1.0, "Execution\nhead"),
        (10.6, 2.1, 1.2, 1.0, "Speed"),
    )
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=.08",
            facecolor="#edf2f7", edgecolor="#4a5568",
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    for start in ((2.4, 3.5), (2.4, 1.5)):
        ax.annotate("", xy=(4.0, 2.6), xytext=start,
                    arrowprops={"arrowstyle": "->", "lw": 2})
    for x1, x2 in ((7.0, 8.0), (9.8, 10.6)):
        ax.annotate("", xy=(x2, 2.6), xytext=(x1, 2.6),
                    arrowprops={"arrowstyle": "->", "lw": 2})
    ax.set_title("Refined hypothesis: context configures the active sequence state")
    _finish(fig, path)


def _components(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["token_specific_context_components"]
    data = data[data.checkpoint == config.primary_checkpoint]
    grouped = data.groupby("token_position")[[
        "shared_context_norm", "specific_context_norm"
    ]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(grouped.index, grouped.shared_context_norm, marker="o",
            color=COLORS["shared"], label="Shared PC1 projection")
    ax.plot(grouped.index, grouped.specific_context_norm, marker="o",
            color=COLORS["specific"], label="Token-specific residual")
    ax.set(xlabel="Token position", ylabel="Component norm",
           title="Shared versus token-specific A/B structure")
    ax.legend(frameon=False); _finish(fig, path)


def _natural_patch(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["full_context_patching"]
    data = data[(data.checkpoint == config.primary_checkpoint) &
                (data.probe_context == "Mixed_norm")]
    values = [
        data.natural_b_speed.mean(),
        data[data.donor_context == "B"].intervened_execution_speed.mean(),
        data.natural_probe_speed.mean(),
        data[data.donor_context == "A"].intervened_execution_speed.mean(),
        data.natural_a_speed.mean(),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(["B", "Mixed←B", "Mixed", "Mixed←A", "A"], values,
           color=[COLORS["B"], "#fc8181", COLORS["Mixed_norm"], "#63b3ed", COLORS["A"]])
    ax.set(ylabel="Execution speed", title="Natural token-matched Mixed patching")
    _finish(fig, path)


def _correct_wrong(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["token_structured_causal_effects"]
    data = data[(data.checkpoint == config.primary_checkpoint) &
                (data.probe_context == "Mixed_norm")]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.bar(["Correct token", "Wrong token\n(all j≠k)"],
           [data.full_patch_effect.mean(), data.wrong_token_patch_effect.mean()],
           color=[COLORS["full"], "#a0aec0"])
    ax.axhline(0, color="#4a5568", lw=.8)
    ax.set(ylabel="Naturally signed A/B patch contrast",
           title="Correct-token versus wrong-token patch")
    _finish(fig, path)


def _natural_causal(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["token_structured_causal_effects"]
    data = data[(data.checkpoint == config.primary_checkpoint) &
                (data.probe_context == "Mixed_norm")]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(data.natural_ab_execution_difference, data.full_patch_contrast,
               alpha=.7, color=COLORS["Mixed_norm"])
    if len(data) > 1 and data.natural_ab_execution_difference.std() > 1e-12:
        slope, intercept = np.polyfit(
            data.natural_ab_execution_difference, data.full_patch_contrast, 1
        )
        x = np.linspace(data.natural_ab_execution_difference.min(),
                        data.natural_ab_execution_difference.max(), 100)
        ax.plot(x, slope * x + intercept, color="#2d3748")
    ax.axhline(0, color="#a0aec0", lw=.7); ax.axvline(0, color="#a0aec0", lw=.7)
    ax.set(xlabel="Natural A−B execution difference",
           ylabel="Mixed←A minus Mixed←B speed",
           title="Natural versus causal token-specific effect")
    _finish(fig, path)


def _shared_full(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["token_structured_causal_effects"]
    data = data[(data.checkpoint == config.primary_checkpoint) &
                (data.probe_context == "Mixed_norm")]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.bar(["Shared only", "Specific only", "Full token-matched"],
           [data.shared_patch_effect.mean(), data.specific_patch_effect.mean(),
            data.full_patch_effect.mean()],
           color=[COLORS["shared"], COLORS["specific"], COLORS["full"]])
    ax.axhline(0, color="#4a5568", lw=.8)
    ax.set(ylabel="Naturally signed patch contrast",
           title="Shared-only versus full patch effect")
    _finish(fig, path)


def _necessity(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["seed_summary"]
    data = data[(data.checkpoint == config.primary_checkpoint) &
                (data.probe_context == "Mixed_norm")]
    labels = ["Shared", "Specific", "Full"]
    collapse = [data.shared_ablation_collapse.mean(),
                data.specific_ablation_collapse.mean(), data.full_ablation_collapse.mean()]
    preservation = [data.shared_sequence_identity_preservation.mean(),
                    data.specific_sequence_identity_preservation.mean(),
                    data.full_sequence_identity_preservation.mean()]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    axes[0].bar(labels, collapse, color=[COLORS["shared"], COLORS["specific"], COLORS["full"]])
    axes[0].set(title="A/B execution collapse", ylabel="Collapse proportion")
    axes[1].bar(labels, preservation,
                color=[COLORS["shared"], COLORS["specific"], COLORS["full"]])
    axes[1].set(title="Same-sequence preservation", ylabel="Prediction identity", ylim=(0, 1.05))
    fig.suptitle("Component necessity with sequence preservation")
    _finish(fig, path)


def _model_comparison(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["execution_model_comparison"]
    data = data[data.checkpoint == config.primary_checkpoint]
    order = ["sequence_only", "context_only", "additive", "interaction"]
    values = data.set_index("model").reindex(order).mean_held_out_mse
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar([value.replace("_", "\n") for value in order], values,
           color=["#718096", "#d69e2e", "#805ad5", "#2b6cb0"])
    ax.set(ylabel="Leave-one-seed-out MSE", title="Cross-validated execution-model comparison")
    _finish(fig, path)


def _generalization(tables, config, probe, path):
    import matplotlib.pyplot as plt
    key = "mixed_generalization" if probe == "Mixed" else "novel_generalization"
    data = tables[key]
    data = data[data.checkpoint == config.primary_checkpoint]
    order = ["sequence_only", "context_only", "additive", "interaction"]
    values = data.groupby("model").squared_error.mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar([value.replace("_", "\n") for value in order], values,
           color=["#718096", "#d69e2e", "#805ad5", "#2b6cb0"])
    ax.set(ylabel="Held-out MSE", title=f"{probe} out-of-sample prediction (A/B fit only)")
    _finish(fig, path)


def make_required_figures(output_dir, *, tables, config):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _hypothesis(output / "figure_01_refined_hypothesis.png")
    _components(tables, config, output / "figure_02_shared_vs_specific.png")
    _natural_patch(tables, config, output / "figure_03_natural_mixed_patching.png")
    _correct_wrong(tables, config, output / "figure_04_correct_vs_wrong_token.png")
    _natural_causal(tables, config, output / "figure_05_natural_vs_causal_token_effect.png")
    _shared_full(tables, config, output / "figure_06_shared_vs_full_patch.png")
    _necessity(tables, config, output / "figure_07_component_necessity.png")
    _model_comparison(tables, config, output / "figure_08_model_comparison.png")
    _generalization(tables, config, "Mixed", output / "figure_09_mixed_generalization.png")
    _generalization(tables, config, "Novel", output / "figure_10_novel_generalization.png")

