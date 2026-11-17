"""The twelve preregistered checkpoint-sweep figures."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS={"A-S":"#0072B2","A-Sh":"#56B4E9","A-Rand":"#999999","B-S":"#D55E00","B-Sh":"#E69F00","B-Rand":"#777777","H8":"#009E73"}


def _save(fig,output,name):fig.tight_layout();fig.savefig(Path(output)/name,dpi=180,bbox_inches="tight");plt.close(fig)
def _blank(output,name,title):
    fig,ax=plt.subplots(figsize=(7,4));ax.axis("off");ax.text(.5,.58,title,ha="center",fontsize=15);ax.text(.5,.42,"No eligible data",ha="center");_save(fig,output,name)


def make_figures(output,training,familiar,transfer,state_space,components,selected,selected_raw,selected_metrics,frontier,controller,h8_reference):
    output=Path(output);summary=familiar.query("row_type == 'seed_summary'")
    # 1 acquisition
    fig,ax=plt.subplots(figsize=(10,5))
    for group,one in training.groupby("group"):ax.plot(one.groupby("training_sequence").mean_normalized_RT.mean(),label=group,color=COLORS.get(group))
    ax.set(xlabel="Familiar update",ylabel="Normalized RT",title="Familiar acquisition trajectories");ax.legend(ncol=4);_save(fig,output,"figure_01_familiar_acquisition.png")
    # 2 transfer
    rt=transfer[transfer.metric=="RT"]
    fig,ax=plt.subplots(figsize=(10,5))
    for group,one in rt.groupby("group"):ax.plot(one.groupby("checkpoint").cross_context_transfer_index.mean(),marker="o",label=group,color=COLORS.get(group))
    ax.axhline(0,color="black",lw=.7);ax.axhline(1,color="black",lw=.7,ls="--");ax.set(xlabel="Checkpoint",ylabel="RT transfer index",title="Cross-context transfer over training");ax.legend(ncol=4);_save(fig,output,"figure_02_cross_context_transfer.png")
    # 3 differentiation
    fig,ax=plt.subplots(figsize=(10,5))
    for group,one in summary.groupby("group"):ax.plot(one.groupby("checkpoint").AB_RT_mean_absolute_difference.mean(),marker="o",label=group,color=COLORS.get(group))
    ax.set(xlabel="Checkpoint",ylabel="Mean |A−B| RT",title="A/B behavioral differentiation");ax.legend(ncol=4);_save(fig,output,"figure_03_AB_differentiation.png")
    # 4 state-space phenotype
    fits=state_space[(state_space.fit_status=="fit")&(state_space.fit_scope=="seed")&(state_space.best_model==True)]
    if len(fits):
        value=fits.assign(split_like=fits.model_label.isin(["Cross","Split"])).groupby(["group","checkpoint"]).split_like.mean().unstack(0)
        fig,ax=plt.subplots(figsize=(10,5));value.plot(ax=ax,marker="o");ax.set(xlabel="Checkpoint",ylabel="Fraction Cross/Split",title="State-space phenotype over training");_save(fig,output,"figure_04_state_space_phenotype.png")
    else:_blank(output,"figure_04_state_space_phenotype.png","State-space phenotype")
    # 5 familiar score
    fig,ax=plt.subplots(figsize=(10,5))
    for group,one in components.groupby("group"):
        ax.plot(one.checkpoint,one.S_familiar,marker="o",label=group,color=COLORS.get(group));choice=selected[selected.group==group]
        if len(choice) and choice.selected_checkpoint.notna().any():
            x=float(choice.selected_checkpoint.iloc[0]);y=float(one.loc[one.checkpoint==x,"S_familiar"].iloc[0]);ax.scatter([x],[y],s=100,facecolors="none",edgecolors="black")
    ax.set(xlabel="Checkpoint",ylabel="Equal-weight familiar score",title="Familiar-only selection score");ax.legend(ncol=3);_save(fig,output,"figure_05_familiar_match_score.png")
    # 6 selected familiar profiles
    eligible=selected[selected.selected_checkpoint.notna()]
    if len(eligible):
        fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
        for ax,(_,choice) in zip(axes.ravel(),eligible.iterrows()):
            one=familiar[(familiar.group==choice.group)&(familiar.checkpoint==choice.selected_checkpoint)&(familiar.row_type=="transition")]
            for context,style in (("A","-"),("B","--")):ax.plot(one[one.context==context].groupby("transition_position").normalized_RT.mean(),style,label=context)
            ax.set_title(f"{choice.group} @ {int(choice.selected_checkpoint)}");ax.set_xlabel("Transition");ax.set_ylabel("RT")
        axes[0,0].legend();_save(fig,output,"figure_06_selected_familiar_profiles.png")
    else:_blank(output,"figure_06_selected_familiar_profiles.png","Selected familiar profiles")
    # 7 selected Mixed profiles
    if len(selected_raw) and "canonical_context" in selected_raw:
        fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
        for ax,group in zip(axes.ravel(),selected_raw.group.unique()):
            one=selected_raw[selected_raw.group==group]
            for context,style in (("Fast","--"),("Mixed","-"),("Slow",":")):ax.plot(one[one.canonical_context==context].groupby("transition_position").normalized_RT.mean(),style,label=context)
            ax.set_title(group);ax.set_xlabel("Transition");ax.set_ylabel("RT")
        axes[0,0].legend();_save(fig,output,"figure_07_selected_mixed_profiles.png")
    else:_blank(output,"figure_07_selected_mixed_profiles.png","Selected first-Mixed profiles")
    # 8 selected contrasts
    mixed_summary=selected_metrics.query("row_type == 'seed_summary'") if len(selected_metrics) and "row_type" in selected_metrics else pd.DataFrame()
    if len(mixed_summary):
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,architecture in zip(axes,("A","B")):
            subset=mixed_summary[mixed_summary.architecture==architecture]
            if len(subset):
                groups=sorted(subset.group.unique());subset.groupby("group").between_endpoint_fraction.mean().reindex(groups).plot.bar(ax=ax,color=[COLORS.get(g) for g in groups])
            else:ax.text(.5,.5,"No gate-passing checkpoint",ha="center",va="center",transform=ax.transAxes)
            ax.set_title(architecture);ax.set_ylabel("Between fraction")
        _save(fig,output,"figure_08_selected_mixed_contrasts.png")
    else:_blank(output,"figure_08_selected_mixed_contrasts.png","Selected-checkpoint Mixed contrasts")
    # 9/10 frontiers
    for x,y,name,title in (("AB_RT_mean_absolute_difference","between_endpoint_fraction","figure_09_differentiation_mixed_frontier.png","A/B differentiation vs Mixed quality"),("cross_context_transfer_index","between_endpoint_fraction","figure_10_transfer_mixed_frontier.png","Cross-context transfer vs Mixed quality")):
        valid=frontier.dropna(subset=[x,y]) if y in frontier else pd.DataFrame()
        if len(valid):
            fig,ax=plt.subplots(figsize=(8,5))
            for group,one in valid.groupby("group"):scatter=ax.scatter(one[x],one[y],c=one.checkpoint,cmap="viridis",label=group,alpha=.7)
            fig.colorbar(scatter,ax=ax,label="Checkpoint");ax.set(xlabel=x,ylabel=y,title=title);ax.legend(ncol=3);_save(fig,output,name)
        else:_blank(output,name,title)
    # 11 B mechanism
    b=frontier[frontier.architecture=="B"]
    if len(b) and "coherent_AB_controller_distance" in b:
        fig,ax=plt.subplots(figsize=(10,5))
        for group,one in b.groupby("group"):
            mean=one.groupby("checkpoint")[["coherent_AB_controller_distance","coherent_AB_hidden_distance"]].mean();ax.plot(mean.index,mean.coherent_AB_controller_distance,label=f"{group} controller",color=COLORS.get(group));ax.plot(mean.index,mean.coherent_AB_hidden_distance,ls="--",label=f"{group} hidden",color=COLORS.get(group))
        ax.set(xlabel="Checkpoint",ylabel="Coherent A/B separation",title="Model-B controller and task-core separation");ax.legend(ncol=3,fontsize=8);_save(fig,output,"figure_11_model_B_mechanism.png")
    else:_blank(output,"figure_11_model_B_mechanism.png","Model-B mechanism")
    # 12 H8 region overlay
    fig,ax=plt.subplots(figsize=(9,5))
    for group,one in summary.groupby("group"):ax.plot(one.groupby("checkpoint").AB_RT_mean_absolute_difference.mean(),label=group,color=COLORS.get(group),lw=3 if group=="H8" else 1)
    ax.set(xlabel="Checkpoint",ylabel="A/B RT separation",title="H8 familiar region over checkpoint trajectories");ax.legend(ncol=4);_save(fig,output,"figure_12_H8_familiar_region.png")
