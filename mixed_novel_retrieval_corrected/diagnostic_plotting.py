"""Registered absolute-fit figures A--J."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS={"Observed":"black","R0":"#d95f02","R1":"#1b9e77","A":"#377eb8","B":"#984ea3"}


def _save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def _hist(ax,values,label,color,bins=35,alpha=.35,density=True):
    ax.hist(np.asarray(values,float),bins=bins,density=density,histtype="stepfilled",alpha=alpha,label=label,color=color)


def make_absolute_fit_figures(output,long_table,primary,r0_sim,r1_sim,geometry,decomposition,
                              transition_diagnostics,anchor_uncertainty):
    root=Path(output); root.mkdir(parents=True,exist_ok=True)

    familiar=long_table[~long_table.is_probe].copy()
    counterbalances=[value for value in familiar.counterbalance.dropna().unique()]
    fig,axes=plt.subplots(max(1,len(counterbalances)),1,figsize=(13,4.5*max(1,len(counterbalances))),sharex=False)
    axes=np.atleast_1d(axes)
    for ax,cb in zip(axes,counterbalances):
        current=familiar[familiar.counterbalance.eq(cb)]
        summary=current.groupby(["trial","context"],as_index=False)[["observed_RT","familiar_predicted_RT"]].mean()
        for context,color in [("A",COLORS["A"]),("B",COLORS["B"])]:
            part=summary[summary.context.eq(context)]
            ax.scatter(part.trial,part.observed_RT,s=14,alpha=.7,color=color,label=f"{context} observed")
        prediction=current.groupby("trial",as_index=False).familiar_predicted_RT.mean()
        ax.plot(prediction.trial,prediction.familiar_predicted_RT,color="#e41a1c",lw=2,label="Frozen familiar prediction")
        probes=long_table[(long_table.is_probe)&long_table.counterbalance.eq(cb)].groupby(["trial","probe_type"],as_index=False).size()
        for _,row in probes.iterrows(): ax.axvline(row.trial,color="grey",alpha=.12,lw=.8)
        ax.set(title=f"{cb}: complete familiar fit (vertical marks = probe insertions)",ylabel="RT (ms)")
        ax.legend(ncol=3,fontsize=8)
    axes[-1].set_xlabel("Chronological sequence")
    _save(fig,root/"figure_A_full_familiar_fit.png")

    probe=long_table[long_table.is_probe].copy(); types=list(probe.probe_type.dropna().unique())
    fig,axes=plt.subplots(len(types),1,figsize=(13,4*len(types)),sharex=False); axes=np.atleast_1d(axes)
    for ax,kind in zip(axes,types):
        current=probe[probe.probe_type.eq(kind)]
        summary=current.groupby("trial",as_index=False)[["observed_RT","A_anchor","B_anchor","R0_expected_RT","R1_predicted_RT"]].mean()
        for column,label,color,style in [("observed_RT","Observed","black","o-"),("A_anchor","A anchor",COLORS["A"],"--"),
            ("B_anchor","B anchor",COLORS["B"],"--"),("R0_expected_RT","R0 mixture expectation",COLORS["R0"],"-"),
            ("R1_predicted_RT","R1 prediction",COLORS["R1"],"-")]:
            ax.plot(summary.trial,summary[column],style,color=color,label=label,alpha=.9)
        ax.set(title=f"{kind}: all probe exposures",ylabel="RT (ms)"); ax.legend(ncol=5,fontsize=8)
    axes[-1].set_xlabel("Chronological sequence")
    _save(fig,root/"figure_B_full_probe_timeline.png")

    first=primary.groupby("transition_position").agg(observed=("observed_rt","mean"),observed_se=("observed_rt","sem"),
        A=("A_anchor","mean"),B=("B_anchor","mean")).reset_index()
    r0=long_table[(long_table.is_probe)&long_table.probe_type.eq("Mixed")&long_table.probe_exposure_index.eq(1)]
    pred=r0.groupby("transition")[["R0_expected_RT","R1_predicted_RT"]].mean()
    fig,ax=plt.subplots(figsize=(9,5.5)); x=first.transition_position
    ax.plot(x,first.A,"o--",color=COLORS["A"],label="A anchor"); ax.plot(x,first.B,"o--",color=COLORS["B"],label="B anchor")
    ax.errorbar(x,first.observed,yerr=first.observed_se,fmt="ko-",capsize=3,label="Observed first Mixed")
    ax.plot(x,pred.R0_expected_RT,"o-",color=COLORS["R0"],label="R0 mixture expectation")
    ax.plot(x,pred.R1_predicted_RT,"o-",color=COLORS["R1"],label="R1 prediction")
    ax.set(xlabel="Transition",ylabel="RT (ms)",title="First-Mixed absolute profile fit"); ax.legend(ncol=2)
    _save(fig,root/"figure_C_first_mixed_profile_fit.png")

    observed_profiles=primary.groupby("participant_uid").observed_rt.mean()
    r0_means=r0_sim.groupby("simulation_id").simulated_RT.mean(); r1_means=r1_sim.groupby("simulation_id").simulated_RT.mean()
    fig,axes=plt.subplots(2,4,figsize=(16,8)); ax=axes.flat[0]
    _hist(ax,observed_profiles,"Observed",COLORS["Observed"]); _hist(ax,r0_means,"R0",COLORS["R0"],alpha=.2); _hist(ax,r1_means,"R1",COLORS["R1"],alpha=.2)
    ax.set(title="Participant mean RT",xlabel="Mean RT",ylabel="Density"); ax.legend(fontsize=8)
    for transition,ax in zip(range(1,8),axes.flat[1:]):
        _hist(ax,primary.loc[primary.transition_position.eq(transition),"observed_rt"],"Observed",COLORS["Observed"],alpha=.25)
        _hist(ax,r0_sim.loc[r0_sim.transition.eq(transition),"simulated_RT"],"R0",COLORS["R0"],alpha=.15)
        _hist(ax,r1_sim.loc[r1_sim.transition.eq(transition),"simulated_RT"],"R1",COLORS["R1"],alpha=.15)
        ax.set(title=f"Transition {transition}",xlabel="RT")
    _save(fig,root/"figure_D_first_mixed_raw_distributions.png")

    fig,ax=plt.subplots(figsize=(9,5.5))
    for source in ["Observed","R0","R1"]:
        _hist(ax,geometry.loc[geometry.source.eq(source),"q_proj"],source,COLORS[source],alpha=.25)
    ax.axvline(0,color=COLORS["A"],ls="--",label="A endpoint"); ax.axvline(1,color=COLORS["B"],ls="--",label="B endpoint")
    ax.set(xlabel="Unclamped q projection (A=0, B=1)",ylabel="Density",title="Participant familiar-axis projections"); ax.legend()
    _save(fig,root/"figure_E_participant_projection_distribution.png")

    metrics=[("q_proj","Familiar-axis projection"),("off_axis_distance","Off-axis distance"),
             ("mean_RT","Profile mean RT"),("profile_variance","Within-profile variance")]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(column,title) in zip(axes.flat,metrics):
        for source in ["Observed","R0","R1"]: _hist(ax,geometry.loc[geometry.source.eq(source),column],source,COLORS[source],alpha=.22)
        ax.set(title=title,xlabel=column,ylabel="Density"); ax.legend(fontsize=8)
    _save(fig,root/"figure_F_posterior_predictive_geometry.png")

    fig,axes=plt.subplots(1,3,figsize=(17,5))
    fam=familiar.groupby("chronological_index",as_index=False).familiar_residual.mean()
    axes[0].scatter(fam.chronological_index,fam.familiar_residual,s=8,alpha=.6); axes[0].axhline(0,color="black",lw=1)
    axes[0].set(title="Familiar residual over time",xlabel="Chronological index",ylabel="Observed − predicted RT")
    probe_summary=probe.groupby("chronological_index",as_index=False)[["R0_residual_expected","R1_residual"]].mean()
    axes[1].plot(probe_summary.chronological_index,probe_summary.R0_residual_expected,".",color=COLORS["R0"],label="R0 expectation")
    axes[1].plot(probe_summary.chronological_index,probe_summary.R1_residual,".",color=COLORS["R1"],label="R1")
    axes[1].axhline(0,color="black",lw=1); axes[1].set(title="Probe residual over time",xlabel="Chronological index"); axes[1].legend()
    by=probe.groupby("transition")[["R0_residual_expected","R1_residual"]].mean()
    axes[2].plot(by.index,by.R0_residual_expected,"o-",color=COLORS["R0"],label="R0 expectation")
    axes[2].plot(by.index,by.R1_residual,"o-",color=COLORS["R1"],label="R1"); axes[2].axhline(0,color="black",lw=1)
    axes[2].set(title="Probe residual by transition",xlabel="Transition"); axes[2].legend()
    _save(fig,root/"figure_G_residuals_full_timeline.png")

    first_long=probe[probe.probe_type.eq("Mixed")&probe.probe_exposure_index.eq(1)]
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,column,title in [(axes[0],"R0_expected_RT","R0 mixture expectation (descriptive)"),(axes[1],"R1_predicted_RT","R1 prediction")]:
        x=first_long[column]; y=first_long.observed_RT; lo=float(min(x.min(),y.min())); hi=float(max(x.max(),y.max()))
        ax.scatter(x,y,s=12,alpha=.25); ax.plot([lo,hi],[lo,hi],"k--")
        rmse=float(np.sqrt(np.mean((y-x)**2))); corr=float(np.corrcoef(x,y)[0,1]); denominator=float(np.sum((y-y.mean())**2)); r2=float(1-np.sum((y-x)**2)/denominator)
        ax.set(xlabel="Predicted RT",ylabel="Observed RT",title=f"{title}\nRMSE={rmse:.1f}, R²={r2:.2f}, r={corr:.2f}")
    _save(fig,root/"figure_H_observed_vs_predicted.png")

    ordered=decomposition.sort_values("delta_LL_R1_minus_R0").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(11,5)); colors=np.where(ordered.delta_LL_R1_minus_R0>0,COLORS["R1"],COLORS["R0"])
    ax.bar(np.arange(len(ordered)),ordered.delta_LL_R1_minus_R0,color=colors,width=1); ax.axhline(0,color="black")
    ax.set(xlabel="Participants ordered by ΔLL",ylabel="LL(R1) − LL(R0)",title="Participant-level likelihood decomposition")
    _save(fig,root/"figure_I_participant_delta_LL.png")

    fig,ax=plt.subplots(figsize=(9,5.5)); _hist(ax,anchor_uncertainty.delta_LL,"Anchor bootstrap",COLORS["R0"],alpha=.5)
    ax.axvline(0,color="black",ls="--"); ax.set(xlabel="Predictive LL(R1) − LL(R0)",ylabel="Density",title="Familiar-anchor uncertainty in model comparison")
    _save(fig,root/"figure_J_anchor_uncertainty_delta_LL.png")
