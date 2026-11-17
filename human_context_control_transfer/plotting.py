"""Compact paper-check figures required by the human transfer PRD."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"A-S": "#0072B2", "A-Sh": "#56B4E9", "B-S": "#D55E00", "B-Sh": "#E69F00", "B-S-RP": "#CC79A7", "H8": "#555555"}


def _save(fig, output, name):
    fig.tight_layout(); fig.savefig(Path(output) / name, dpi=180, bbox_inches="tight"); plt.close(fig)


def _blank(output, name, title, message="No gate-passing data"):
    fig, ax = plt.subplots(figsize=(7, 4)); ax.axis("off"); ax.text(.5, .58, title, ha="center", fontsize=15); ax.text(.5, .42, message, ha="center"); _save(fig, output, name)


def make_figures(output, timeline, training, gates, canonical, mixed, geometry, controller, retention, causal, contrasts, human_profiles):
    output = Path(output)
    # 1: logic schematic
    fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off")
    boxes = [("Frozen synthetic\nstructured/shuffled prior", .12), ("Human familiar\nA/B only", .40), ("Familiar\ngates F1–F6", .66), ("Frozen Mixed / Novel /\nNoContext", .90)]
    for label, x in boxes: ax.text(x, .5, label, ha="center", va="center", bbox=dict(boxstyle="round", fc="#eef3f8", ec="#444"))
    for x in (.25, .53, .78): ax.annotate("", (x + .08, .5), (x - .04, .5), arrowprops=dict(arrowstyle="->"))
    _save(fig, output, "figure_01_training_logic.png")
    # 2 timeline
    fig, ax = plt.subplots(figsize=(10, 3)); probes = timeline.dropna(subset=["probe_name"])
    ax.hlines(0, 0, 120, color="black"); ax.scatter(probes.model_training_sequence_index, np.zeros(len(probes)), color="#D55E00")
    for i, row in probes.iterrows(): ax.annotate(row.probe_name, (row.model_training_sequence_index, 0), xytext=(0, 8 + (i % 3) * 13), textcoords="offset points", rotation=45, fontsize=8)
    ax.set(xlabel="Completed familiar A/B updates", yticks=[], title="Exact human/model timeline alignment", xlim=(-3, 123)); _save(fig, output, "figure_02_timeline_alignment.png")
    # 3 acquisition phenotype
    fig, ax = plt.subplots(figsize=(9, 5))
    for group, one in training.groupby("group"):
        curve = one.groupby("training_sequence").mean_normalized_RT.mean(); ax.plot(curve.index, curve, label=group, color=COLORS.get(group))
    ax.set(xlabel="Familiar training sequence", ylabel="Normalized RT", title="Familiar A/B acquisition phenotype"); ax.legend(ncol=3); _save(fig, output, "figure_03_familiar_phenotype.png")
    # 4 first mixed profiles
    first = canonical[(canonical.probe_name == "Mixed1") & canonical.canonical_context.isin(["Fast", "Slow", "Mixed"])] if len(canonical) else canonical
    if len(first):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        for ax, group in zip(axes.ravel(), COLORS):
            one = first[first.group == group]
            for context, style in (("Fast", "--"), ("Mixed", "-"), ("Slow", ":")):
                values = one[one.canonical_context == context].groupby("transition_position").normalized_RT.mean(); ax.plot(values.index, values, style, label=context)
            ax.set_title(group); ax.set_xlabel("Transition"); ax.set_ylabel("Normalized RT")
        axes[0, 0].legend(); _save(fig, output, "figure_04_first_mixed_profile.png")
    else: _blank(output, "figure_04_first_mixed_profile.png", "First Mixed profile")
    # 5 causal contrasts
    if len(contrasts):
        fig, ax = plt.subplots(figsize=(10, 5)); table = contrasts.pivot(index="metric", columns="contrast", values="paired_mean_difference"); table.plot.bar(ax=ax); ax.axhline(0, color="black", lw=1); ax.set(title="Structured minus shuffled: first Mixed", ylabel="Paired mean difference"); _save(fig, output, "figure_05_structured_shuffled_contrasts.png")
    else: _blank(output, "figure_05_structured_shuffled_contrasts.png", "Structured–shuffled contrasts")
    # 6 model-human profiles
    if len(first) and "Mixed" in human_profiles:
        fig, ax = plt.subplots(figsize=(9, 5)); human = np.asarray(human_profiles["Mixed"]); human = (human-human.mean())/max(human.std(),1e-12); ax.plot(range(1,8), human, "ko-", lw=3, label="Human Mixed1")
        for group, one in first[first.canonical_context == "Mixed"].groupby("group"):
            v = one.groupby("transition_position").normalized_RT.mean().to_numpy(); v=(v-v.mean())/max(v.std(),1e-12); ax.plot(range(1,8),v,"o-",label=group,color=COLORS.get(group))
        ax.set(xlabel="Transition", ylabel="Within-profile z", title="Unfitted human/model Mixed profiles"); ax.legend(ncol=3); _save(fig, output, "figure_06_human_model_profiles.png")
    else: _blank(output, "figure_06_human_model_profiles.png", "Human/model profiles")
    # 7/8 geometry summaries
    for frame, column, name, title in ((controller, "controller_projection", "figure_07_controller_geometry.png", "Model B controller geometry"), (geometry, "familiar_axis_projection", "figure_08_hidden_geometry.png", "Hidden-state geometry")):
        if len(frame) and column in frame:
            fig, ax = plt.subplots(figsize=(9, 5)); values=frame[frame.probe_name=="Mixed1"].groupby(["group","transition_position"])[column].mean().unstack(0); values.plot(ax=ax, marker="o"); ax.axhspan(0,1,color="#ddd",alpha=.4); ax.set(xlabel="Transition",ylabel="A/B-axis coordinate",title=title); _save(fig,output,name)
        else: _blank(output,name,title)
    # 9 restricted plasticity
    data = mixed.query("row_type == 'seed_summary' and probe_name == 'Mixed1' and group in ['B-S','B-S-RP']") if len(mixed) else mixed
    if len(data):
        fig, ax=plt.subplots(figsize=(6,4)); data.groupby("group").between_endpoint_fraction.mean().plot.bar(ax=ax,color=[COLORS.get(x) for x in sorted(data.group.unique())]); ax.set(ylabel="Between-endpoint fraction",title="Full vs restricted Model B plasticity"); _save(fig,output,"figure_09_restricted_plasticity.png")
    else:_blank(output,"figure_09_restricted_plasticity.png","Restricted plasticity")
    # 10 retention
    if len(retention) and {"group", "checkpoint", "synthetic_ambiguity_NLL"}.issubset(retention.columns):
        fig,ax=plt.subplots(figsize=(9,5)); retention.groupby(["group","checkpoint"]).synthetic_ambiguity_NLL.mean().unstack(0).plot(ax=ax,marker="o");ax.set(xlabel="Human updates",ylabel="Synthetic ambiguity NLL",title="Prior retention");_save(fig,output,"figure_10_prior_retention.png")
    else:_blank(output,"figure_10_prior_retention.png","Prior retention")
    # 11 interventions
    intervention_parts = [v.assign(family=k) for k,v in causal.items() if len(v) and "normalized_RT" in v]
    intervention = pd.concat(intervention_parts, ignore_index=True) if intervention_parts else pd.DataFrame()
    if len(intervention):
        fig,ax=plt.subplots(figsize=(10,5)); intervention.groupby(["family","intervention"]).normalized_RT.mean().plot.bar(ax=ax);ax.set(ylabel="Mean normalized RT",title="Frozen causal interventions");_save(fig,output,"figure_11_causal_interventions.png")
    else:_blank(output,"figure_11_causal_interventions.png","Causal interventions")
    # 12 frontier
    summaries=mixed.query("row_type == 'seed_summary' and probe_name == 'Mixed1'") if len(mixed) else mixed
    if len(summaries):
        gate_seed=gates.query("row_type == 'seed'")[["group","seed","AB_RT_gap"]]; points=summaries.merge(gate_seed,on=["group","seed"])
        fig,ax=plt.subplots(figsize=(7,5));
        for group,one in points.groupby("group"):ax.scatter(one.AB_RT_gap,one.between_endpoint_fraction,label=group,color=COLORS.get(group),alpha=.7)
        ax.set(xlabel="Familiar A/B RT differentiation",ylabel="First-Mixed between fraction",title="Differentiation–transfer frontier");ax.legend(ncol=2);_save(fig,output,"figure_12_transfer_frontier.png")
    else:_blank(output,"figure_12_transfer_frontier.png","Differentiation–transfer frontier")
