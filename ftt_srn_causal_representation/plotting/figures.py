"""The ten preregistered causal-mechanism figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


COLORS = {"A": "#2b6cb0", "B": "#c53030", "Mixed_norm": "#805ad5", "Novel": "#2f855a"}


def _finish(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _hypothesis(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 4.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [(0.3, 2, 2, 1, "Context"), (3.1, 2, 2.7, 1, "A/B hidden\nsubspace"),
             (6.6, 2, 2.1, 1, "Execution\nhead"), (9.6, 2, 2, 1, "Speed")]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08",
                                    facecolor="#edf2f7", edgecolor="#4a5568"))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center")
    for x1, x2 in ((2.3, 3.1), (5.8, 6.6), (8.7, 9.6)):
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.scatter([3.7, 4.45, 5.2], [1.2, 1.2, 1.2], s=90,
               color=[COLORS["B"], COLORS["Mixed_norm"], COLORS["A"]])
    ax.text(4.45, .65, "B        Mixed        A", ha="center")
    ax.set_title("Causal hypothesis: subspace position controls execution")
    _finish(fig, path)


def _directions(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["ab_directions"]
    data = data[(data.checkpoint == config.primary_checkpoint) & (data.seed == data.seed.min())]
    columns = sorted([c for c in data if c.startswith("unit_d")], key=lambda x: int(x[6:]))
    matrix = data[columns].to_numpy(dtype=float)
    similarity = tables["token_direction_similarity"]
    similarity = similarity[(similarity.checkpoint == config.primary_checkpoint) &
                            (similarity.seed == similarity.seed.min())]
    size = len(data); cosine = np.eye(size)
    for row in similarity.itertuples():
        cosine[row.token_position_1-1, row.token_position_2-1] = row.pairwise_cosine_similarity
        cosine[row.token_position_2-1, row.token_position_1-1] = row.pairwise_cosine_similarity
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    first = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_yticks(np.arange(size), np.arange(1, size + 1))
    axes[0].set(xlabel="Hidden unit", ylabel="Token position", title="Token-specific unit A/B vectors")
    fig.colorbar(first, ax=axes[0], fraction=.046)
    second = axes[1].imshow(cosine, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_xticks(np.arange(size), np.arange(1, size + 1))
    axes[1].set_yticks(np.arange(size), np.arange(1, size + 1))
    axes[1].set(xlabel="Token position", ylabel="Token position", title="Pairwise direction cosine")
    fig.colorbar(second, ax=axes[1], fraction=.046)
    _finish(fig, path)


def _scree(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["ab_subspace_summary"]
    data = data[data.checkpoint == config.primary_checkpoint].groupby("component", as_index=False)[
        ["variance_explained", "cumulative_variance_explained"]
    ].mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.bar(data.component, data.variance_explained, color="#718096", label="component")
    ax.plot(data.component, data.cumulative_variance_explained, marker="o", color="#805ad5",
            label="cumulative")
    ax.axhline(config.variance_threshold, color="#c53030", ls="--", label="90% threshold")
    ax.set(xlabel="Component", ylabel="Variance proportion", ylim=(0, 1.03),
           title="A/B difference-subspace dimensionality")
    ax.legend(frameon=False); _finish(fig, path)


def _steering(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["steering_results"]
    data = data[(data.checkpoint == config.primary_checkpoint) &
                (data.base_context == "Mixed_norm") &
                (data.intervention_level == "instantaneous")]
    curve = data.groupby("alpha", as_index=False).execution_speed.mean()
    a, b = data.natural_a_speed.mean(), data.natural_b_speed.mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.plot(curve.alpha, curve.execution_speed, marker="o", color=COLORS["Mixed_norm"])
    ax.axhline(a, color=COLORS["A"], ls="--", label="natural A")
    ax.axhline(b, color=COLORS["B"], ls="--", label="natural B")
    ax.set(xlabel="Alpha along B→A token direction", ylabel="Execution speed",
           title="Instantaneous directional steering")
    ax.legend(frameon=False); _finish(fig, path)


def _random_control(tables, config, path):
    import matplotlib.pyplot as plt
    ab = tables["steering_seed_summary"]
    ab = ab[(ab.checkpoint == config.primary_checkpoint) & (ab.base_context == "Mixed_norm")]
    random = tables["random_direction_controls"]
    random = random[(random.checkpoint == config.primary_checkpoint) &
                    (random.control_test == "steering") &
                    (random.base_context == "Mixed_norm")]
    random_slopes = []
    for seed, group in random.groupby("seed"):
        zero = group[group.alpha == 0].groupby("token_position").execution_speed_signed.first()
        delta = np.array([row.execution_speed_signed - zero.loc[row.token_position]
                          for row in group.itertuples()])
        random_slopes.append(np.polyfit(group.alpha, delta, 1)[0])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    means = [ab.steering_slope.mean(), np.mean(random_slopes)]
    ax.bar(["A/B direction", "Random orthogonal"], means, color=["#805ad5", "#718096"])
    ax.axhline(0, color="#4a5568", lw=.8); ax.set(ylabel="Signed speed slope", title="Specificity control")
    _finish(fig, path)


def _ablation(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["ablation_seed_summary"]
    data = data[data.checkpoint == config.primary_checkpoint]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].bar(["Natural", "A/B component\nremoved"],
                [data.natural_ab_execution_separation.mean(), data.ablated_ab_execution_separation.mean()],
                color=["#805ad5", "#a0aec0"])
    axes[0].set_title("A/B execution separation")
    axes[1].bar(["Natural", "A/B component\nremoved"],
                [data.natural_next_token_accuracy.mean(), data.ablated_next_token_accuracy.mean()],
                color=["#2b6cb0", "#90cdf4"])
    axes[1].set(title="Same-sequence accuracy", ylim=(0, 1.05))
    fig.suptitle("Necessity with sequence preservation"); _finish(fig, path)


def _patch(tables, config, probe, path):
    import matplotlib.pyplot as plt
    key = "mixed_patching_results" if probe == "Mixed" else "novel_patching_results"
    context = "Mixed_norm" if probe == "Mixed" else "Novel"
    data = tables[key]; data = data[data.checkpoint == config.primary_checkpoint]
    natural = tables["steering_results"]
    natural = natural[(natural.checkpoint == config.primary_checkpoint) &
                      (natural.base_context.isin(["A", "B", context])) &
                      (natural.intervention_level == "instantaneous") & (natural.alpha == 0)]
    values = [natural[natural.base_context == "B"].execution_speed.mean(),
              data[data.donor_context == "B"].patched_execution_speed.mean(),
              natural[natural.base_context == context].execution_speed.mean(),
              data[data.donor_context == "A"].patched_execution_speed.mean(),
              natural[natural.base_context == "A"].execution_speed.mean()]
    labels = ["B", f"{probe}←B", probe, f"{probe}←A", "A"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color=[COLORS["B"], "#fc8181", COLORS[context], "#63b3ed", COLORS["A"]])
    ax.set(ylabel="Execution speed", title=f"{probe} A/B-component patching")
    _finish(fig, path)


def _sequence_preservation(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["ablation_results"]
    data = data[data.checkpoint == config.primary_checkpoint].copy()
    data["ce_change"] = data.ablated_next_token_ce - data.natural_next_token_ce
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    for context, group in data.groupby("context"):
        ax.scatter(group.execution_speed_change, group.ce_change, alpha=.65,
                   label=context, color=COLORS[context])
    ax.axhline(0, color="#718096", lw=.7); ax.axvline(0, color="#718096", lw=.7)
    ax.set(xlabel="Execution-speed change", ylabel="Next-token CE change",
           title="Functional dissociation under A/B ablation")
    ax.legend(frameon=False); _finish(fig, path)


def _checkpoints(tables, config, path):
    import matplotlib.pyplot as plt
    data = tables["causal_mechanism_summary"]
    grouped = data.groupby("checkpoint", sort=False).agg(
        steering=("steering_slope", "mean"),
        collapse=("ablation_execution_collapse", "mean"),
        patch=("Mixed_patch_A_proximity_improvement", "mean"),
    )
    grouped = grouped.reindex([
        checkpoint for checkpoint in config.checkpoints if checkpoint in grouped.index
    ])
    fig, ax = plt.subplots(figsize=(9, 4.7))
    x = np.arange(len(grouped)); width = .25
    ax.bar(x-width, grouped.steering, width, label="steering slope")
    ax.bar(x, grouped.collapse, width, label="execution collapse")
    ax.bar(x+width, grouped.patch, width, label="Mixed←A proximity")
    ax.set_xticks(x, grouped.index, rotation=18); ax.axhline(0, color="#4a5568", lw=.7)
    ax.set(title="Checkpoint dependence of causal effects", ylabel="Within-network effect")
    ax.legend(frameon=False); _finish(fig, path)


def make_required_figures(output_dir, *, tables, config):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    _hypothesis(output / "figure_01_causal_hypothesis.png")
    _directions(tables, config, output / "figure_02_token_ab_directions.png")
    _scree(tables, config, output / "figure_03_ab_subspace_dimensionality.png")
    _steering(tables, config, output / "figure_04_steering_curve.png")
    _random_control(tables, config, output / "figure_05_random_direction_control.png")
    _ablation(tables, config, output / "figure_06_ab_component_ablation.png")
    _patch(tables, config, "Mixed", output / "figure_07_mixed_patching.png")
    _patch(tables, config, "Novel", output / "figure_08_novel_patching.png")
    _sequence_preservation(tables, config, output / "figure_09_sequence_preservation.png")
    _checkpoints(tables, config, output / "figure_10_checkpoint_dependence.png")
