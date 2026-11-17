"""Compact preregistered diagnostics for Stages A--C."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, output, name):
    fig.tight_layout(); fig.savefig(output / name, dpi=170); plt.close(fig)


def stage_a_figures(result, output):
    comp=result["training_comparison"]
    fig,ax=plt.subplots(figsize=(8,4))
    mean=comp.groupby("training_sequence")[["mean_prediction_NLL_reference","mean_prediction_NLL_replay"]].mean(); ax.plot(mean.index,mean.iloc[:,0],label="old frozen output"); ax.plot(mean.index,mean.iloc[:,1],"--",label="exact replay"); ax.legend(); ax.set(xlabel="Training sequence",ylabel="Prediction NLL",title="A1. Old versus replay H8 learning")
    _save(fig,output,"figure_A1_h8_learning_trajectory.png")
    fig,ax=plt.subplots(figsize=(5,5)); hidden=result["hidden_comparison"]; replay=[c for c in hidden if c.startswith("hidden_state_") and c.endswith("_replay")]; reference=[c.replace("_replay","_reference") for c in replay]; old=[]; new=[]
    familiar=hidden[hidden.context.isin(["A","B"])]
    for _,one in familiar.groupby(["seed","transition_position"]):
        lookup=one.set_index("context"); old.append(np.linalg.norm(lookup.loc["A",reference].to_numpy(float)-lookup.loc["B",reference].to_numpy(float))); new.append(np.linalg.norm(lookup.loc["A",replay].to_numpy(float)-lookup.loc["B",replay].to_numpy(float)))
    old=np.asarray(old); new=np.asarray(new); ax.scatter(old,new,s=12,alpha=.6); bounds=[min(old.min(),new.min()),max(old.max(),new.max())]; ax.plot(bounds,bounds,"k--"); ax.set(xlabel="Old A/B hidden separation",ylabel="Replay A/B hidden separation",title="A2. Old versus replay A/B separation"); _save(fig,output,"figure_A2_h8_hidden_separation.png")
    fig,ax=plt.subplots(figsize=(6,4)); state=result["state_comparison"]; labels=sorted(state.model.unique()); x=np.arange(len(labels)); old=state.groupby("model").BIC_reference.mean().reindex(labels); new=state.groupby("model").BIC_replay.mean().reindex(labels); ax.plot(x,old,"o-",label="old"); ax.plot(x,new,"x--",label="replay"); ax.set_xticks(x,labels,rotation=20); ax.set(ylabel="Mean BIC",title="A3. State-space winner replay"); ax.legend(); _save(fig,output,"figure_A3_h8_state_space_replay.png")


def stage_b_figures(result,output):
    raw=result["canonical"]; post=raw[raw.phase.eq("post")]
    fig,ax=plt.subplots(figsize=(8,2.7)); ax.axis("off"); ax.text(.02,.7,"Locked H128 checkpoint  →  10-step full-batch SGD on A/B only",fontsize=13); ax.text(.02,.35,"Frozen A/B/Mixed/Novel probes  →  descriptive confidence-to-RT mapping",fontsize=12); _save(fig,output,"figure_B1_native_meta_test_schematic.png")
    fam=result["familiar"]; fig,ax=plt.subplots(figsize=(7,4))
    for group,one in fam.groupby("group"): ax.scatter(one.AB_hidden_distance_pre,one.AB_hidden_distance_post,label=group,alpha=.75)
    bounds=[min(fam.AB_hidden_distance_pre.min(),fam.AB_hidden_distance_post.min()),max(fam.AB_hidden_distance_pre.max(),fam.AB_hidden_distance_post.max())]; ax.plot(bounds,bounds,"k--"); ax.legend(); ax.set(xlabel="Pre-adaptation A/B distance",ylabel="Post-adaptation A/B distance",title="B2. Familiar differentiation")
    _save(fig,output,"figure_B2_AB_differentiation.png")
    fig,ax=plt.subplots(figsize=(8,4)); profile=post[post.context.eq("Mixed")].groupby(["group","transition_position"],as_index=False).correct_token_probability.mean()
    for group,one in profile.groupby("group"): ax.plot(one.transition_position,one.correct_token_probability,marker="o",label=group)
    ax.legend(); ax.set(xlabel="Transition",ylabel="Mixed correct-token probability",title="B3. Mixed prediction profile"); _save(fig,output,"figure_B3_mixed_prediction_profile.png")
    mixed=result["mixed"]
    geometry=result["geometry"]; fig,ax=plt.subplots(figsize=(7,4)); summary=geometry.groupby(["group","transition_position"],as_index=False).off_corridor_distance.mean()
    for group,one in summary.groupby("group"): ax.plot(one.transition_position,one.off_corridor_distance,marker="o",label=group)
    ax.legend(); ax.set(xlabel="Transition",ylabel="Off-corridor distance",title="B4. Mixed hidden corridor geometry"); _save(fig,output,"figure_B4_mixed_hidden_corridor.png")
    pivot=mixed.pivot(index="seed",columns="group",values="prediction_NLL_deficit"); delta=pivot.P2_meta-pivot.P1_shuffled_meta
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(np.arange(len(delta)),delta); ax.axhline(0,color="k",lw=1); ax.set(xlabel="Seed",ylabel="P2 − shuffled Mixed NLL deficit",title="B5. Structured versus shuffled paired differences"); _save(fig,output,"figure_B5_structured_shuffled_paired.png")
    novel=result["novel"]; fig,ax=plt.subplots(figsize=(7,4)); groups=list(novel.group.unique()); ax.boxplot([novel[novel.group.eq(g)].prediction_NLL_deficit for g in groups],tick_labels=groups); ax.set(ylabel="Novel prediction NLL deficit",title="B6. Novel diagnostics"); _save(fig,output,"figure_B6_novel_diagnostics.png")


def stage_c_figures(result,output):
    validation=result["validation"]
    fig,ax=plt.subplots(figsize=(8,2.8)); ax.axis("off"); ax.text(.03,.72,"smooth visible context:  −1  ─────────  boundary  ─────────  +1",fontsize=12); ax.text(.03,.38,"task law: within-side similarity + across-boundary discontinuity",fontsize=12); ax.text(.03,.08,"40% smooth / 40% discrimination / 20% ambiguous episodes",fontsize=11); _save(fig,output,"figure_C1_boundary_ecology_schematic.png")
    specs=(("C2_within_cross_transfer","boundary_selectivity","Within versus cross-boundary transfer"),
           ("C3_ambiguous_query","within_query_NLL","Ambiguous-query behavior"),
           ("C4_structured_shuffled","within_query_NLL","Synthetic structured versus shuffled"))
    for stem,metric,title in specs:
        one=validation if stem!="C3_ambiguous_query" else validation[validation.episode_type.eq("ambiguous_boundary")]
        fig,ax=plt.subplots(figsize=(7,4)); groups=list(one.training_condition.unique())
        if groups: ax.boxplot([one[one.training_condition.eq(g)][metric] for g in groups],tick_labels=groups)
        else: ax.text(.5,.5,"No observations in this engineering profile",ha="center",va="center")
        ax.tick_params(axis="x",rotation=15); ax.set(ylabel=metric,title=title); _save(fig,output,f"figure_{stem}.png")
    if "human_probe_metrics" in result:
        metrics=result["human_probe_metrics"]; means=metrics[metrics.row_type.eq("seed_mean")]
        familiar=result["human_familiar"]; run=familiar[familiar.row_type.eq("run")]; fig,ax=plt.subplots(figsize=(8,4)); groups=list(run.group.unique()); ax.bar(np.arange(len(groups)),[run[run.group.eq(g)].F2_context_differentiation.mean() for g in groups]); ax.set_xticks(np.arange(len(groups)),groups,rotation=20); ax.set(ylabel="Fraction passing A/B differentiation",title="C5. Human A/B phenotype"); _save(fig,output,"figure_C5_human_AB_phenotype.png")
        for number,metric,title in ((6,"between_endpoint_fraction","First Mixed transfer"),(7,"prediction_NLL_deficit","Differentiation-transfer frontier")):
            fig,ax=plt.subplots(figsize=(8,4)); groups=list(means.group.unique()); ax.boxplot([means[means.group.eq(g)][metric] for g in groups],tick_labels=groups); ax.tick_params(axis="x",rotation=20); ax.set(ylabel=metric,title=title); _save(fig,output,f"figure_C{number}_{'human_mixed_transfer' if number==6 else 'human_frontier'}.png")


def prior_overwriting_figure(frame,output):
    if frame.empty: return
    fig,ax=plt.subplots(figsize=(8,4)); long=frame[frame.adaptation_regime.eq("long_target_Adam")]; ax.plot(long.target_checkpoint,long.AB_hidden_distance,marker="o",label="long Adam A/B separation"); native=frame[frame.adaptation_regime.eq("native_10_step_SGD")]
    if len(native): ax.scatter([10],[native.AB_hidden_distance.iloc[0]],marker="*",s=140,label="native 10-step adaptation")
    ax.set(xlabel="Target adaptation step",ylabel="A/B hidden distance",title="C8. Prior retention versus specialization"); ax.legend(); _save(fig,output,"figure_C8_prior_retention_specialization.png")
