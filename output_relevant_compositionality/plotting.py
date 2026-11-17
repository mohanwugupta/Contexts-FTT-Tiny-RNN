from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS={"C0_high_baseline":"#D55E00","C1_output_curvature":"#009E73","C2_whole_curvature":"#56B4E9",
        "C3_null_curvature":"#CC79A7","C4_low_baseline":"#0072B2"}


def _save(fig,out,name):
    fig.tight_layout();fig.savefig(Path(out)/name,dpi=180,bbox_inches="tight");plt.close(fig)


def _bar(metrics,column,out,name,ylabel):
    order=list(COLORS);values=metrics.groupby("condition")[column].mean().reindex(order)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(order,values,color=[COLORS[x] for x in order]);ax.tick_params(axis="x",rotation=20);ax.set_ylabel(ylabel);_save(fig,out,name)


def make_phase1_figures(out,t):
    metrics=t["phase1_seed_metrics"];patch=t["phase1_patching"];sweep=t["phase1_cue_sweep"]
    _bar(metrics,"AB_hidden_separation",out,"P1_01_AB_separation_by_condition.png","Mean A/B hidden separation")
    _bar(metrics,"Mixed_loss",out,"P1_02_Mixed_loss_by_condition.png","Mixed sequence loss")
    _bar(metrics,"K_out_mid",out,"P1_03_output_curvature_by_condition.png","Midpoint output curvature")
    _bar(metrics,"K_logit_mid",out,"P1_04_logit_curvature_by_condition.png","Midpoint logit curvature")
    for x,y,name,xlabel,ylabel in (
        ("AB_hidden_separation","Mixed_loss","P1_05_AB_separation_vs_Mixed_loss.png","A/B separation","Mixed loss"),
        ("AB_hidden_separation","K_out_mid","P1_06_AB_separation_vs_output_curvature.png","A/B separation","Output curvature"),
        ("K_out_mid","Mixed_loss","P1_07_output_curvature_vs_Mixed_loss.png","Output curvature","Mixed loss")):
        fig,ax=plt.subplots(figsize=(7,5))
        for condition,sub in metrics.groupby("condition"):ax.scatter(sub[x],sub[y],label=condition,color=COLORS.get(condition),alpha=.75)
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.legend(fontsize=7);_save(fig,out,name)
    fig,ax=plt.subplots(figsize=(10,4.5));sub=patch[patch.patch_type.ne("baseline_actual")]
    order=list(COLORS);patches=["full_linear_path","output_only_linear_path","null_only_linear_path"];x=np.arange(len(patches));width=.8/len(order)
    for i,condition in enumerate(order):
        values=sub[sub.condition.eq(condition)].groupby("patch_type").delta_loss_vs_actual.mean().reindex(patches)
        ax.bar(x+(i-2)*width,values,width,label=condition,color=COLORS[condition])
    ax.axhline(0,color="black",lw=.7);ax.set_xticks(x,[p.replace("_","\n") for p in patches]);ax.set_ylabel("Patch − actual loss");ax.legend(fontsize=6,ncol=2);_save(fig,out,"P1_08_linear_path_patching.png")
    for column,name,ylabel in (("sequence_loss","P1_09_cue_sweep_loss.png","Sequence loss"),("output_residual_norm","P1_10_cue_sweep_output_residual.png","Output residual norm")):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for condition,sub in sweep.groupby("condition"):
            mean=sub.groupby("cue_lambda")[column].mean();ax.plot(mean.index,mean,label=condition,color=COLORS.get(condition))
        ax.set_xlabel("Cue lambda");ax.set_ylabel(ylabel);ax.legend(fontsize=7);_save(fig,out,name)


def make_phase2_figures(out,t):
    general=t["phase2_pretraining_generalization"];formal=t["phase2_formal_model_fit"];human=t["phase2_human_transfer"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,column in zip(axes,["heldout_accuracy","heldout_loss"]):
        mean=general.groupby(["family","ecology"])[column].mean().unstack();mean.plot.bar(ax=ax);ax.set_ylabel(column.replace("_"," "));ax.tick_params(axis="x",rotation=0)
    _save(fig,out,"P2_01_pretraining_heldout_generalization.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,column in zip(axes,["K_out","K_logit"]):
        general.groupby(["family","ecology"])[column].mean().unstack().plot.bar(ax=ax);ax.set_ylabel(column);ax.tick_params(axis="x",rotation=0)
    _save(fig,out,"P2_02_pretraining_output_curvature.png")
    fig,ax=plt.subplots(figsize=(9,4.5));f2=formal[formal.model.eq("F2")]
    f2.groupby(["family","ecology"]).heldout_output_R2.mean().unstack().plot.bar(ax=ax);ax.set_ylabel("F2 held-out output R²");ax.tick_params(axis="x",rotation=0);_save(fig,out,"P2_03_pretraining_formal_model_fit.png")
    for column,name,ylabel in (("AB_hidden_separation","P2_04_human_AB_separation.png","A/B separation"),
                              ("Mixed_loss","P2_05_human_Mixed_loss.png","Mixed loss"),
                              ("K_out_mid","P2_06_human_output_curvature.png","Output curvature")):
        fig,ax=plt.subplots(figsize=(10,4.5));mean=human.groupby("source")[column].mean().sort_index();ax.bar(mean.index,mean);ax.tick_params(axis="x",rotation=25);ax.set_ylabel(ylabel);_save(fig,out,name)
    fig,ax=plt.subplots(figsize=(8,5))
    for source,sub in human.groupby("source"):ax.scatter(sub.AB_hidden_separation,sub.Mixed_loss,s=30+sub.K_out_mid,alpha=.65,label=source)
    ax.set_xlabel("A/B separation");ax.set_ylabel("Mixed loss");ax.legend(fontsize=7);_save(fig,out,"P2_07_human_tradeoff_map.png")
    fig,ax=plt.subplots(figsize=(7,6));
    for family in sorted(general.family.unique()):
        structured=general[(general.family==family)&(general.ecology=="structured")].set_index("seed")
        shuffled=general[(general.family==family)&(general.ecology=="shuffled")].set_index("seed")
        ax.scatter(shuffled.heldout_loss,structured.heldout_loss,label=family,alpha=.75)
    limits=ax.get_xlim();lo=min(limits[0],ax.get_ylim()[0]);hi=max(limits[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],"k:");ax.set_xlabel("Shuffled held-out loss");ax.set_ylabel("Structured held-out loss");ax.legend();_save(fig,out,"P2_08_structured_vs_shuffled_seed_scatter.png")
