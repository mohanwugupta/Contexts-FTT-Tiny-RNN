"""Required Phase 1H figures and raw-loss time series."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import HIGH,LOW


FIGURES=tuple(f"PHASE1H_FIG{i:02d}_{name}.png" for i,name in enumerate((
    "shared_vs_private_change","private_shared_ratio_by_seed","sequence_decoding_shared_private",
    "AB_gradient_alignment","parameter_block_gradient_alignment","within_vs_cross_context_transfer",
    "transfer_vs_shared_update_energy","transfer_vs_gradient_alignment","high_vs_low_plasticity",
    "shared_private_subspace_angles"),1))
TIMESERIES=tuple(f"PHASE1H_TS{i:02d}_{name}.png" for i,name in enumerate((
    "true_AB_acquisition_raw_loss","within_cross_transfer_raw_loss",
    "shared_private_change_over_training","update_energy_over_training",
    "gradient_alignment_over_training"),1))
COLORS={HIGH:"#E45756",LOW:"#4C78A8","shared":"#54A24B","private":"#B279A2",
        "within":"#4C78A8","cross":"#F58518"}


def _save(fig,path): fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def _paired_bar(frame,columns,labels,path,title,ylabel):
    conditions=[HIGH,LOW];x=np.arange(2);width=.8/len(columns);fig,ax=plt.subplots(figsize=(7.5,4.8))
    for j,(column,label) in enumerate(zip(columns,labels)):
        values=[frame[frame.scale_condition.eq(c)][column].mean() for c in conditions]
        ax.bar(x+(j-(len(columns)-1)/2)*width,values,width,label=label)
    ax.set_xticks(x,conditions);ax.set_ylabel(ylabel);ax.set_title(title);ax.legend();_save(fig,path)


def make_figures(output,states,shared_private,updates,subspaces,gradients,blocks,functional,decoders,summary):
    output=Path(output);figures=output/"figures";timeseries=output/"timeseries"
    figures.mkdir(parents=True,exist_ok=True);timeseries.mkdir(parents=True,exist_ok=True)
    _paired_bar(summary,["mean_shared_change","mean_private_change"],["shared","private"],figures/FIGURES[0],
                "Representational change over familiar learning","Mean state-change norm")
    fig,ax=plt.subplots(figsize=(8,4.8))
    for condition,group in summary.groupby("scale_condition"):
        ax.scatter(group.seed,group.median_private_shared_ratio,label=condition,color=COLORS[condition])
    ax.axhline(1,color="black",ls="--",lw=.8);ax.set(xlabel="Seed",ylabel="Median private/shared squared-change ratio",
        title="Private/shared plasticity ratio by seed");ax.legend();_save(fig,figures/FIGURES[1])
    final=decoders[decoders.training_sequence_index.eq(decoders.training_sequence_index.max())]
    _paired_bar(final,["sequence_position_accuracy_shared","sequence_position_accuracy_private","context_accuracy_private"],
                ["position: shared","position: private","context: private"],figures/FIGURES[2],
                "Leave-seed-out decoding at final checkpoint","Accuracy")
    _paired_bar(summary,["mean_gradient_alignment"],["A/B gradient cosine"],figures/FIGURES[3],
                "A/B gradient alignment","Cosine")
    blockmean=blocks.groupby(["scale_condition","parameter_block"],as_index=False).cosine_gradient_alignment.mean()
    blocks_order=list(blockmean.parameter_block.drop_duplicates());fig,ax=plt.subplots(figsize=(10,4.8));width=.35
    for j,condition in enumerate((HIGH,LOW)):
        values=blockmean[blockmean.scale_condition.eq(condition)].set_index("parameter_block").cosine_gradient_alignment.reindex(blocks_order)
        ax.bar(np.arange(len(blocks_order))+(j-.5)*width,values,width,label=condition,color=COLORS[condition])
    ax.set_xticks(np.arange(len(blocks_order)),blocks_order,rotation=25,ha="right");ax.set_ylabel("Gradient cosine")
    ax.set_title("Alignment by parameter block");ax.legend();_save(fig,figures/FIGURES[4])
    _paired_bar(summary,["mean_within_context_transfer","mean_cross_context_transfer"],["within","cross"],figures/FIGURES[5],
                "Actual one-step loss improvement","Raw CE improvement")
    actual=functional[functional.analysis_type.eq("actual_training_update")]
    energy=updates.groupby(["seed","scale_condition","training_sequence_index"],as_index=False).shared_update_energy_fraction.mean()
    merged=actual.merge(energy,on=["seed","scale_condition","training_sequence_index"])
    fig,ax=plt.subplots(figsize=(6.5,5))
    for condition,group in merged.groupby("scale_condition"):
        ax.scatter(group.shared_update_energy_fraction,group.cross_context_loss_change,s=8,alpha=.25,color=COLORS[condition],label=condition)
    ax.set(xlabel="Shared update-energy fraction",ylabel="Cross-context raw CE improvement",
           title="Transfer versus shared representational update energy");ax.legend();_save(fig,figures/FIGURES[6])
    gradmerge=actual.merge(gradients[["seed","scale_condition","training_sequence_index","cosine_gradient_alignment"]],
                           on=["seed","scale_condition","training_sequence_index"])
    fig,ax=plt.subplots(figsize=(6.5,5))
    for condition,group in gradmerge.groupby("scale_condition"):
        ax.scatter(group.cosine_gradient_alignment,group.cross_context_loss_change,s=18,alpha=.45,color=COLORS[condition],label=condition)
    ax.set(xlabel="A/B gradient cosine",ylabel="Cross-context raw CE improvement",
           title="Transfer versus local gradient alignment");ax.legend();_save(fig,figures/FIGURES[7])
    _paired_bar(summary,["median_private_shared_ratio","private_update_energy_fraction","shared_update_energy_fraction"],
                ["private/shared ratio","private energy","shared energy"],figures/FIGURES[8],
                "High-versus-Low plasticity structure","Metric value")
    angle=subspaces.groupby(["scale_condition","rank"],as_index=False).mean_principal_angle_degrees.mean()
    fig,ax=plt.subplots(figsize=(7,4.8))
    for condition,group in angle.groupby("scale_condition"):
        ax.plot(group["rank"],group.mean_principal_angle_degrees,marker="o",color=COLORS[condition],label=condition)
    ax.set(xlabel="Subspace rank",ylabel="Mean principal angle (degrees)",title="Shared/private subspace angles");ax.legend();_save(fig,figures/FIGURES[9])

    # Raw-loss and time-resolved figures.
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),sharey=True)
    for ax,condition in zip(axes,(HIGH,LOW)):
        data=actual[actual.scale_condition.eq(condition)].groupby(["training_sequence_index","training_context"]).raw_sequence_loss_pre_update.mean().reset_index()
        for context,group in data.groupby("training_context"):
            ax.plot(group.training_sequence_index,group.raw_sequence_loss_pre_update,label=context)
        ax.set(xlabel="Actual external training sequence",ylabel="Raw mean CE",title=condition);ax.legend()
    _save(fig,timeseries/TIMESERIES[0])
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),sharey=True)
    for ax,condition in zip(axes,(HIGH,LOW)):
        data=actual[actual.scale_condition.eq(condition)].groupby("training_sequence_index")[["within_context_loss_change","cross_context_loss_change"]].mean()
        ax.plot(data.index,data.within_context_loss_change,label="within",color=COLORS["within"])
        ax.plot(data.index,data.cross_context_loss_change,label="cross",color=COLORS["cross"])
        ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Actual update",ylabel="Raw CE improvement",title=condition);ax.legend()
    _save(fig,timeseries/TIMESERIES[1])
    change=shared_private[shared_private.training_sequence_index.gt(0)].groupby(["scale_condition","training_sequence_index"])[["shared_state_change","private_state_change"]].mean().reset_index()
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),sharey=True)
    for ax,condition in zip(axes,(HIGH,LOW)):
        data=change[change.scale_condition.eq(condition)];ax.plot(data.training_sequence_index,data.shared_state_change,label="shared",color=COLORS["shared"])
        ax.plot(data.training_sequence_index,data.private_state_change,label="private",color=COLORS["private"])
        ax.set(xlabel="Actual update",ylabel="Mean state-change norm",title=condition);ax.legend()
    _save(fig,timeseries/TIMESERIES[2])
    energy=updates.groupby(["scale_condition","training_sequence_index"])[["shared_update_energy_fraction","private_update_energy_fraction"]].mean().reset_index()
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),sharey=True)
    for ax,condition in zip(axes,(HIGH,LOW)):
        data=energy[energy.scale_condition.eq(condition)];ax.plot(data.training_sequence_index,data.shared_update_energy_fraction,label="shared",color=COLORS["shared"])
        ax.plot(data.training_sequence_index,data.private_update_energy_fraction,label="private",color=COLORS["private"])
        ax.set(xlabel="Actual update",ylabel="Update-energy fraction",title=condition);ax.legend()
    _save(fig,timeseries/TIMESERIES[3])
    grad=gradients.groupby(["scale_condition","training_sequence_index"]).cosine_gradient_alignment.mean().reset_index()
    fig,ax=plt.subplots(figsize=(9,4.8))
    for condition,group in grad.groupby("scale_condition"):
        ax.plot(group.training_sequence_index,group.cosine_gradient_alignment,marker="o",color=COLORS[condition],label=condition)
    ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Registered replay checkpoint",ylabel="A/B gradient cosine",
        title="Gradient alignment over training");ax.legend();_save(fig,timeseries/TIMESERIES[4])


def enforce_outputs(output):
    output=Path(output);paths=[output/"figures"/x for x in FIGURES]+[output/"timeseries"/x for x in TIMESERIES]
    missing=[str(x) for x in paths if not x.exists()]
    if missing: raise RuntimeError("Missing Phase 1H figures:\n- "+"\n- ".join(missing))
