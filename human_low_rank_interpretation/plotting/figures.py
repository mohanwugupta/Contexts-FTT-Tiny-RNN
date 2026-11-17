"""Eleven registered PRD v3.1 figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


COLORS = ["#2b6cb0", "#805ad5", "#2f855a", "#d69e2e", "#c53030", "#319795", "#718096"]


def _finish(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _loadings(tables, path):
    import matplotlib.pyplot as plt
    data = tables["component_loadings"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for component, group in data.groupby("component"):
        ax.plot(group.transition_position, group.loading, marker="o",
                color=COLORS[int(component) - 1], label=f"Component {component}")
    ax.axhline(0, color="#718096", lw=.8)
    ax.set(xlabel="Transition position", ylabel="Loading",
           title="Human low-rank component profiles")
    ax.legend(frameon=False, ncol=2); _finish(fig, path)


def _bootstrap(tables, path):
    import matplotlib.pyplot as plt
    data = tables["component_loadings"]
    fig, axes = plt.subplots(data.component.nunique(), 1, figsize=(7.2, 2.7 * data.component.nunique()),
                             sharex=True, squeeze=False)
    for ax, (component, group) in zip(axes[:, 0], data.groupby("component")):
        ax.errorbar(group.transition_position, group.bootstrap_mean_loading,
                    yerr=[group.bootstrap_mean_loading - group.bootstrap_ci_lower,
                          group.bootstrap_ci_upper - group.bootstrap_mean_loading],
                    marker="o", color=COLORS[int(component) - 1])
        ax.axhline(0, color="#718096", lw=.8)
        ax.set(ylabel="Loading", title=f"Component {component}")
    axes[-1, 0].set_xlabel("Transition position")
    fig.suptitle("Participant-bootstrap component stability"); _finish(fig, path)


def _dimensions(tables, path):
    import matplotlib.pyplot as plt
    data = tables["dimension_selection"]
    data = data[(data.row_type.eq("summary")) & data.method.eq("PCA")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for dataset, group in data.groupby("dataset"):
        axes[0].plot(group["rank"], group.heldout_reconstruction_mse, marker="o", label=dataset)
        axes[1].plot(group["rank"], group.mean_heldout_probe_log_likelihood,
                     marker="o", label=dataset)
    axes[0].set(xlabel="Dimensions", ylabel="Held-out reconstruction MSE")
    axes[1].set(xlabel="Dimensions", ylabel="Mean held-out probe log likelihood")
    axes[0].legend(frameon=False); axes[1].legend(frameon=False)
    fig.suptitle("Held-out performance versus dimensionality"); _finish(fig, path)


def _feature_scatter(tables, path, features, title):
    import matplotlib.pyplot as plt
    loadings = tables["component_loadings"]
    table = tables["transition_feature_table"].set_index("transition_position")
    components = sorted(loadings.component.unique())
    # A single feature is easier to compare across components in one horizontal row.
    horizontal = len(features) == 1
    nrows, ncols = (1, len(components)) if horizontal else (len(components), len(features))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.4 * ncols, 3.5 * nrows), squeeze=False)
    for row, component in enumerate(components):
        values = loadings[loadings.component.eq(component)].sort_values("transition_position")
        for column, feature in enumerate(features):
            ax = axes[0, row] if horizontal else axes[row, column]
            x = table.loc[values.transition_position, feature].to_numpy(dtype=float)
            y = values.loading.to_numpy(dtype=float)
            ax.scatter(x, y, color=COLORS[int(component) - 1], s=50)
            keep = np.isfinite(x) & np.isfinite(y)
            if keep.sum() >= 2 and np.std(x[keep]) > 1e-12:
                slope, intercept = np.polyfit(x[keep], y[keep], 1)
                grid = np.linspace(x[keep].min(), x[keep].max(), 50)
                ax.plot(grid, intercept + slope * grid, color="#2d3748")
            ax.set(xlabel=feature.replace("_", " "), ylabel=f"Component {component}")
    fig.suptitle(title); _finish(fig, path)


def _residual(tables, path):
    import matplotlib.pyplot as plt
    data = tables["difficulty_residualized_results"]
    data = data[data.row_type.eq("residual_loading")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for component, group in data.groupby("component"):
        ax.plot(group.transition_position, group.difficulty_residualized_loading,
                marker="o", label=f"Component {component}", color=COLORS[int(component)-1])
    ax.axhline(0, color="#718096", lw=.8)
    ax.set(xlabel="Transition position", ylabel="Difficulty-residualized loading",
           title="Component structure beyond generic difficulty")
    ax.legend(frameon=False); _finish(fig, path)


def _probe_coefficients(tables, path):
    import matplotlib.pyplot as plt
    # Output tables intentionally include both participant-level rows and summary rows;
    # the figure must use the latter exactly once per component.
    mixed = tables["mixed_component_coefficients"]
    novel = tables["novel_component_coefficients"]
    if "row_type" in mixed:
        mixed = mixed[mixed.row_type.eq("group_summary")]
    if "row_type" in novel:
        novel = novel[novel.row_type.eq("group_summary")]
    mixed = mixed.sort_values("component").reset_index(drop=True)
    novel = novel.sort_values("component").reset_index(drop=True)
    if not mixed.component.equals(novel.component):
        raise ValueError("Mixed and Novel coefficient summaries have different components.")
    data = np.arange(len(mixed))
    mixed_error = np.maximum(0, np.vstack([
        mixed.mean_coefficient - mixed.bootstrap_ci_lower,
        mixed.bootstrap_ci_upper - mixed.mean_coefficient,
    ]).astype(float))
    novel_error = np.maximum(0, np.vstack([
        novel.mean_coefficient - novel.bootstrap_ci_lower,
        novel.bootstrap_ci_upper - novel.mean_coefficient,
    ]).astype(float))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(data - .18, mixed.mean_coefficient, width=.36, color="#805ad5", label="Mixed",
           yerr=mixed_error, capsize=3)
    ax.bar(data + .18, novel.mean_coefficient, width=.36, color="#2f855a", label="Novel",
           yerr=novel_error, capsize=3)
    ax.set_xticks(data, [f"C{value}" for value in mixed.component])
    ax.set(xlabel="Fixed A/B component", ylabel="Mean coefficient",
           title="Mixed and Novel locations in the same basis")
    ax.legend(frameon=False); _finish(fig, path)


def _temporal(tables, path):
    import matplotlib.pyplot as plt
    data = tables["temporal_mixed_coefficients"]
    data = data[data.probe_context.eq("Both")]
    order = [value for value in ("Mixed1", "Mixed2", "MixedTest") if value in set(data.probe_epoch)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for component, group in data.groupby("component"):
        group = group.set_index("probe_epoch").reindex(order)
        ax.plot(order, group.coefficient, marker="o", label=f"Component {component}",
                color=COLORS[int(component)-1])
    ax.axhline(0, color="#718096", lw=.8)
    ax.set(xlabel="Mixed epoch", ylabel="Coefficient",
           title="Temporal projection into the fixed A/B basis")
    ax.legend(frameon=False); _finish(fig, path)


def _cross_dataset(tables, path):
    import matplotlib.pyplot as plt
    data = tables["cross_dataset_component_alignment"]
    fig, axes = plt.subplots(1, data.component.nunique(),
                             figsize=(4.2 * data.component.nunique(), 3.8), squeeze=False)
    for ax, (component, group) in zip(axes[0], data.groupby("component")):
        ax.plot(group.transition_position, group.combined_loading, marker="o", label="Combined")
        ax.plot(group.transition_position, group.novel_loading_aligned, marker="o", label="Novel")
        ax.set(title=f"Component {component}\ncos={group.matched_cosine.iloc[0]:.2f}",
               xlabel="Transition", ylabel="Loading")
        ax.legend(frameon=False)
    fig.suptitle("Independent cross-dataset basis alignment"); _finish(fig, path)


def _human_srn(tables, path):
    import matplotlib.pyplot as plt
    dims = tables["human_srn_dimensionality_comparison"]
    variance = tables["human_srn_shared_specific_variance"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    axes[0].bar(dims.system, dims.effective_dimensionality, color=["#805ad5", "#2b6cb0"])
    axes[0].set(ylabel="Participation-ratio dimension", title="Effective dimensionality")
    axes[1].bar(variance.system, variance.shared_pc1_variance_fraction,
                color=["#805ad5", "#2b6cb0"])
    axes[1].set(ylim=(0, 1), ylabel="PC1 variance fraction", title="Shared vs state-specific variance")
    fig.suptitle("Human–SRN comparison at the abstraction level"); _finish(fig, path)


def _recovery(tables, path):
    import matplotlib.pyplot as plt
    data = tables["semantic_recovery_matrix"]
    generators = list(data.generating_semantic.drop_duplicates())
    candidates = list(data.recovered_semantic.drop_duplicates())
    matrix = data.pivot(index="generating_semantic", columns="recovered_semantic",
                        values="recovery_probability").reindex(index=generators, columns=candidates)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for row in range(len(generators)):
        for column in range(len(candidates)):
            ax.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center")
    ax.set_xticks(range(len(candidates)), candidates, rotation=35, ha="right")
    ax.set_yticks(range(len(generators)), generators)
    ax.set(xlabel="Recovered semantic", ylabel="Generating semantic",
           title="Semantic recovery matrix")
    fig.colorbar(image, ax=ax, label="Probability"); _finish(fig, path)


def make_required_figures(output_dir, *, tables):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    _loadings(tables, output / "figure_01_component_loadings.png")
    _bootstrap(tables, output / "figure_02_component_bootstrap_stability.png")
    _dimensions(tables, output / "figure_03_heldout_dimension_selection.png")
    _feature_scatter(tables, output / "figure_04_component_vs_difficulty.png",
                     ["mean_pure_logRT"], "Components versus generic transition difficulty")
    _feature_scatter(tables, output / "figure_05_component_vs_context_switch.png",
                     ["A_minus_B", "switch_cost"], "Components versus context sensitivity and switch cost")
    _residual(tables, output / "figure_06_difficulty_residualized_components.png")
    _probe_coefficients(tables, output / "figure_07_mixed_novel_coefficients.png")
    _temporal(tables, output / "figure_08_temporal_mixed_coefficients.png")
    _cross_dataset(tables, output / "figure_09_cross_dataset_alignment.png")
    _human_srn(tables, output / "figure_10_human_srn_abstraction.png")
    _recovery(tables, output / "figure_11_semantic_recovery.png")
