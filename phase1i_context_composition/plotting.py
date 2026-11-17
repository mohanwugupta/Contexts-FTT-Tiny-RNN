from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import HIGH,LOW


FIGURES=tuple(f"PHASE1I_FIG{i:02d}_{name}.png" for i,name in enumerate((
    "true_canonical_acquisition","probe_loss_registered_checkpoints","final_transition_profiles",
    "full_sequence_patch_timeseries","novel_nocontext_composite_rescue",
    "matched_vs_free_composition","forward_composition_vs_backward_transfer"),1))


def _save(fig,path):fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def make_figures(output,acquisition,timeline,transition,summary,config):
    output=Path(output);figures=output/"figures";figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),sharey=True)
    for ax,family in zip(axes,(HIGH,LOW)):
        data=acquisition[acquisition.family.eq(family)].groupby(["training_sequence_index","training_context"]).raw_sequence_loss_pre_update.agg(["mean","sem"]).reset_index()
        for context,g in data.groupby("training_context"):
            ax.plot(g.training_sequence_index,g["mean"],label=context);ax.fill_between(g.training_sequence_index,g["mean"]-g["sem"].fillna(0),g["mean"]+g["sem"].fillna(0),alpha=.15)
        ax.set(xlabel="Actual external training sequence",ylabel="Raw mean CE",title=family);ax.legend()
    _save(fig,figures/FIGURES[0])
    registered=timeline[timeline.series_type.eq("probe_checkpoint") & timeline.intervention.eq("no_patch")]
    fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True)
    for ax,family in zip(axes,(HIGH,LOW)):
        data=registered[registered.family.eq(family)].copy()
        novel=data[data.probe.str.startswith("Novel")].groupby("training_sequence_index",as_index=False).raw_sequence_loss.mean();novel["probe"]="NovelMean"
        data=pd.concat([data[["training_sequence_index","probe","raw_sequence_loss"]],novel],ignore_index=True)
        data=data.groupby(["training_sequence_index","probe"]).raw_sequence_loss.mean().reset_index().rename(columns={"probe":"probe_family"})
        for probe,g in data.groupby("probe_family"):ax.plot(g.training_sequence_index,g.raw_sequence_loss,marker="o",label=probe)
        ax.set(xlabel="Registered replay checkpoint",ylabel="Raw mean CE",title=family);ax.legend(fontsize=7)
    _save(fig,figures/FIGURES[1])
    # High is the preregistered primary family; Low remains visible in the
    # acquisition/checkpoint and forward/backward comparison figures.
    final=transition[transition.training_sequence_index.eq(config.training_sequences)&transition.family.eq(HIGH)]
    selected=final[(final.intervention.eq("free_rollout"))|((final.patch_position.fillna(-1)==0)&final.intervention.isin(["composite_patch","wrong_position_patch","orthogonal_patch"]))]
    selected=selected.copy();selected["panel"]=np.where(selected.probe.str.startswith("Novel"),"Novel",selected.probe)
    fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
    for ax,panel in zip(axes,("Mixed","NoContext","Novel")):
        data=selected[selected.panel.eq(panel)].groupby(["transition","intervention"]).raw_transition_loss.mean().reset_index()
        for intervention,g in data.groupby("intervention"):ax.plot(g.transition,g.raw_transition_loss,marker="o",label=intervention)
        refs=final[(final.intervention.eq("free_rollout"))&final.probe.isin(["A","B"])].groupby(["transition","probe"]).raw_transition_loss.mean().reset_index()
        for ref,g in refs.groupby("probe"):ax.plot(g.transition,g.raw_transition_loss,"--",label=f"{ref} reference")
        ax.set(title=panel,xlabel="Transition",ylabel="Raw transition CE");ax.legend(fontsize=6)
    fig.suptitle("High-scale final-checkpoint transition profiles",y=1.02);_save(fig,figures/FIGURES[2])
    fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True)
    for ax,family in zip(axes,(HIGH,LOW)):
        data=timeline[timeline.family.eq(family)&(((timeline.probe.isin(["A","B"]))&timeline.intervention.eq("no_patch"))|((timeline.probe.eq("Mixed"))&timeline.intervention.isin(["no_patch","composite_patch","wrong_position_patch","orthogonal_patch"])))].groupby(["training_sequence_index","probe","intervention"]).raw_sequence_loss.mean().reset_index()
        for (probe,intervention),g in data.groupby(["probe","intervention"]):ax.plot(g.training_sequence_index,g.raw_sequence_loss,label=f"{probe}:{intervention}",lw=1)
        ax.set(xlabel="Actual external sequence index",ylabel="Raw mean CE",title=family);ax.legend(fontsize=6)
    _save(fig,figures/FIGURES[3])
    patches=final[(final.patch_position.fillna(-1)==0)&final.probe.isin(["NoContext","Novel1","Novel2","Novel3","Novel4"])&final.intervention.isin(["no_patch","composite_patch","wrong_position_patch","orthogonal_patch"])]
    fig,ax=plt.subplots(figsize=(11,5));probes=list(patches.probe.drop_duplicates());interventions=list(patches.intervention.drop_duplicates());width=.8/len(interventions)
    for j,intervention in enumerate(interventions):
        values=patches[patches.intervention.eq(intervention)].groupby("probe").raw_sequence_loss.mean().reindex(probes)
        ax.bar(np.arange(len(probes))+(j-(len(interventions)-1)/2)*width,values,width,label=intervention)
    ax.set_xticks(np.arange(len(probes)),probes);ax.set(ylabel="Raw mean CE",title="Synthetic familiar-composite control for Novel and NoContext");ax.legend(fontsize=7)
    references=final[(final.intervention.eq("free_rollout"))&final.probe.isin(["A","B"])].groupby("probe").raw_transition_loss.mean()
    for reference,style in (("A","--"),("B",":")):ax.axhline(references[reference],color="black",linestyle=style,linewidth=1,label=f"{reference} reference")
    ax.legend(fontsize=7)
    _save(fig,figures/FIGURES[4])
    comp=final[final.intervention.isin(["matched_state_branch","free_rollout"])&final.probe.isin(["Mixed","NoContext","Novel1","Novel2","Novel3","Novel4"])]
    comp=comp.copy();comp["probe_family"]=comp.probe.where(~comp.probe.str.startswith("Novel"),"Novel")
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=False)
    for ax,column,title in zip(axes,("hidden_composition_residual_normalized","task_composition_residual_normalized"),("Hidden","Task")):
        data=comp.groupby(["transition","probe_family","intervention"])[column].mean().reset_index()
        for (probe,mode),g in data.groupby(["probe_family","intervention"]):ax.plot(g.transition,g[column],marker="o",label=f"{probe}:{mode}")
        ax.set(xlabel="Transition",ylabel="Normalized residual",title=f"{title}-space composition");ax.legend(fontsize=6)
    _save(fig,figures/FIGURES[5])
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for family,g in summary.groupby("family"):
        axes[0,0].scatter(g.gradient_cosine_AB,g.mixed_matched_hidden_residual,label=family)
        axes[0,1].scatter(g.gradient_cosine_AB,g.mixed_free_hidden_residual,label=family)
        axes[1,0].scatter(g.cross_transfer,g.mixed_rescue,label=family)
        axes[1,1].scatter(g.AB_hidden_separation,g.mixed_matched_hidden_residual,label=family)
        for row in g.itertuples():
            axes[0,0].annotate(str(row.seed),(row.gradient_cosine_AB,row.mixed_matched_hidden_residual),fontsize=6)
            axes[0,1].annotate(str(row.seed),(row.gradient_cosine_AB,row.mixed_free_hidden_residual),fontsize=6)
            axes[1,0].annotate(str(row.seed),(row.cross_transfer,row.mixed_rescue),fontsize=6)
            axes[1,1].annotate(str(row.seed),(row.AB_hidden_separation,row.mixed_matched_hidden_residual),fontsize=6)
    axes[0,0].set(xlabel="A/B gradient cosine",ylabel="Mixed matched-state residual",title="Matched composition vs gradient alignment")
    axes[0,1].set(xlabel="A/B gradient cosine",ylabel="Mixed free-rollout residual",title="Recurrent composition vs gradient alignment")
    axes[1,0].set(xlabel="Actual cross-context CE improvement",ylabel="Mixed composite-patch rescue",title="Causal rescue vs backward transfer")
    axes[1,1].set(xlabel="A/B hidden separation",ylabel="Mixed matched-state residual",title="Composition vs representational separation")
    for ax in axes.flat:ax.legend();_save(fig,figures/FIGURES[6])


def enforce_figures(output):
    missing=[str(Path(output)/"figures"/name) for name in FIGURES if not (Path(output)/"figures"/name).exists()]
    if missing:raise RuntimeError("Missing Phase 1I mandatory figures:\n- "+"\n- ".join(missing))
