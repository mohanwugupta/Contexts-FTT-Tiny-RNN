from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import HIGH,LOW


FIGURES=tuple(f"PHASE1J_FIG{i:02d}_{name}.png" for i,name in enumerate((
    "canonical_acquisition_audit","final_natural_probe_profiles","single_injection_raw_loss",
    "post_injection_hidden_drift","post_injection_task_drift","behavioral_rescue_persistence",
    "context_clamp_decomposition","immediate_vs_remaining_rescue","repeated_vs_single_persistence"),1))
EXTRA=("PHASE1J_SUPP_FIG03_all_injection_positions.png","PHASE1J_FULLSEQ_HIGH.png","PHASE1J_FULLSEQ_LOW.png")


def _save(fig,path):fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def _novel_family(frame):
    result=frame.copy();result["probe_family"]=result.probe.where(~result.probe.str.startswith("Novel"),"NovelMean");return result


def make_figures(output,acquisition,metrics,summary,config):
    output=Path(output);figures=output/"figures";figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),sharey=True)
    for ax,family in zip(axes,(HIGH,LOW)):
        data=acquisition[acquisition.family.eq(family)].groupby(["training_sequence_index","training_context"]).raw_sequence_loss_pre_update.agg(["mean","sem"]).reset_index()
        for context,g in data.groupby("training_context"):
            ax.plot(g.training_sequence_index,g["mean"],label=context);ax.fill_between(g.training_sequence_index,g["mean"]-g["sem"].fillna(0),g["mean"]+g["sem"].fillna(0),alpha=.15)
        ax.set(xlabel="Actual external training sequence",ylabel="Raw mean CE",title=family);ax.legend()
    _save(fig,figures/FIGURES[0])
    natural=metrics[metrics.intervention.eq("natural_baseline")]
    fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True)
    for ax,family in zip(axes,(HIGH,LOW)):
        data=natural[natural.family.eq(family)].groupby(["probe","transition"]).raw_ce.mean().reset_index()
        for probe,g in data.groupby("probe"):ax.plot(g.transition,g.raw_ce,marker="o",label=probe)
        ax.set(xlabel="Transition",ylabel="Raw CE",title=family);ax.legend(fontsize=6)
    _save(fig,figures/FIGURES[1])

    def injection_grid(positions,path):
        fig,axes=plt.subplots(3,len(positions),figsize=(5*len(positions),12),sharey=True)
        for r,probe in enumerate(("Mixed","NoContext","NovelMean")):
            baseline=natural[natural.family.eq(HIGH)&natural.probe.eq(probe)].groupby("transition").raw_ce.mean()
            repeated=metrics[metrics.family.eq(HIGH)&metrics.probe.eq(probe)&metrics.intervention.eq("repeated_midpoint")].groupby("transition").raw_ce.mean()
            for c,position in enumerate(positions):
                ax=axes[r,c] if len(positions)>1 else axes[r]
                ax.plot(baseline.index,baseline.values,label="no patch",lw=2);ax.plot(repeated.index,repeated.values,label="repeated midpoint",lw=2)
                data=metrics[metrics.family.eq(HIGH)&metrics.probe.eq(probe)&metrics.injection_position.eq(position)&metrics.context_clamp.eq("D0_natural")&metrics.intervention.isin(["single_midpoint","wrong_position","orthogonal","A_endpoint","B_endpoint"])]
                for intervention,g in data.groupby("intervention"):
                    mean=g.groupby("transition").raw_ce.mean();ax.plot(mean.index,mean.values,label=intervention,alpha=.85)
                ax.axvline(position,color="black",ls=":",lw=.7);ax.set(title=f"{probe}, injection {position}",xlabel="Transition",ylabel="Raw CE")
                if r==0 and c==len(positions)-1:ax.legend(fontsize=6)
        _save(fig,path)
    injection_grid((1,3,5),figures/FIGURES[2]);injection_grid(tuple(range(1,8)),figures/EXTRA[0])

    post=metrics[metrics.intervention.eq("single_midpoint")&metrics.lag_from_patch.ge(0)].copy();post=_novel_family(post)
    def drift_figure(column,path,title):
        fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True)
        for ax,family in zip(axes,(HIGH,LOW)):
            data=post[post.family.eq(family)&post.probe_family.isin(["Mixed","NoContext","NovelMean"])].groupby(["lag_from_patch","probe_family","context_clamp"])[column].mean().reset_index()
            for (probe,clamp),g in data.groupby(["probe_family","context_clamp"]):ax.plot(g.lag_from_patch,g[column],marker="o",label=f"{probe}:{clamp}")
            ax.set(xlabel="Lag from injection",ylabel="Normalized midpoint residual",title=f"{family}: {title}");ax.legend(fontsize=6)
        _save(fig,path)
    drift_figure("hidden_midpoint_residual",figures/FIGURES[3],"hidden drift")
    drift_figure("task_midpoint_residual",figures/FIGURES[4],"task drift")
    fig,ax=plt.subplots(figsize=(9,5))
    data=post[post.family.eq(HIGH)&post.context_clamp.eq("D0_natural")&post.probe_family.isin(["Mixed","NoContext","NovelMean"])].groupby(["lag_from_patch","probe_family"]).raw_rescue.mean().reset_index()
    for probe,g in data.groupby("probe_family"):ax.plot(g.lag_from_patch,g.raw_rescue,marker="o",label=probe)
    ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Lag from injection",ylabel="Raw CE rescue",title="High-scale behavioral rescue persistence");ax.legend();_save(fig,figures/FIGURES[5])

    focus=summary[summary.family.eq(HIGH)&summary.intervention.eq("single_midpoint")&summary.probe.isin(["Mixed","NoContext","NovelMean"])]
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for ax,probe in zip(axes,("Mixed","NoContext","NovelMean")):
        pivot=focus[focus.probe.eq(probe)].pivot(index="seed",columns="context_clamp",values="behavioral_persistence_score")
        effects=pd.DataFrame({"D1-D0":pivot["D1_zero"]-pivot["D0_natural"],"D2-D0":pivot["D2_composite"]-pivot["D0_natural"]})
        for seed,row in effects.iterrows():ax.plot([0,1],row.values,marker="o",alpha=.55)
        ax.axhline(0,color="black",lw=.7);ax.set_xticks([0,1],effects.columns);ax.set(title=probe,ylabel="Persistence improvement")
    _save(fig,figures/FIGURES[6])
    d0=focus[focus.context_clamp.eq("D0_natural")]
    fig,ax=plt.subplots(figsize=(7,5))
    for probe,g in d0.groupby("probe"):ax.scatter(g.immediate_rescue,g.remaining_sequence_rescue,label=probe)
    ax.set(xlabel="Immediate raw CE rescue",ylabel="Remaining-sequence rescue (sum CE)",title="Immediate versus persistent benefit");ax.legend();_save(fig,figures/FIGURES[7])
    probes=("Mixed","NoContext","NovelMean");fig,ax=plt.subplots(figsize=(9,5));labels=[];values=[];colors=[]
    for probe in probes:
        base=natural[natural.family.eq(HIGH)&natural.probe.eq(probe)].raw_ce.mean()
        single=post[post.family.eq(HIGH)&post.probe_family.eq(probe)&post.context_clamp.eq("D0_natural")].raw_ce.mean()
        repeated=metrics[metrics.family.eq(HIGH)&metrics.probe.eq(probe)&metrics.intervention.eq("repeated_midpoint")].raw_ce.mean()
        labels.extend([f"{probe}\nbaseline",f"{probe}\nsingle",f"{probe}\nrepeated"]);values.extend([base,single,repeated]);colors.extend(["#999999","#F58518","#54A24B"])
    ax.bar(np.arange(len(values)),values,color=colors);ax.set_xticks(np.arange(len(labels)),labels,rotation=25,ha="right");ax.set(ylabel="Raw CE",title="State sufficiency versus single-injection maintenance");_save(fig,figures/FIGURES[8])

    for family,name in ((HIGH,EXTRA[1]),(LOW,EXTRA[2])):
        fig,ax=plt.subplots(figsize=(10,5))
        refs=natural[natural.family.eq(family)&natural.probe.isin(["A","B","Mixed","NoContext","NovelMean"])].groupby(["probe","transition"]).raw_ce.mean().reset_index()
        for probe in ("A","B","Mixed","NoContext","NovelMean"):
            g=refs[refs.probe.eq(probe)];ax.plot(g.transition,g.raw_ce,label=f"{probe} natural",linestyle="--" if probe in ("A","B") else "-")
        single=metrics[metrics.family.eq(family)&metrics.injection_position.eq(1)&metrics.intervention.eq("single_midpoint")&metrics.context_clamp.eq("D0_natural")&metrics.probe.isin(["Mixed","NoContext","NovelMean"])].groupby(["probe","transition"]).raw_ce.mean().reset_index()
        for probe,g in single.groupby("probe"):ax.plot(g.transition,g.raw_ce,marker="o",label=f"{probe} single@1")
        ax.set(xlabel="Transition",ylabel="Raw CE",title=f"{family}: familiar and probe performance");ax.legend(fontsize=7);_save(fig,figures/name)


def enforce_figures(output):
    missing=[str(Path(output)/"figures"/name) for name in FIGURES+EXTRA if not (Path(output)/"figures"/name).exists()]
    if missing:raise RuntimeError("Missing Phase 1J mandatory figures:\n- "+"\n- ".join(missing))
