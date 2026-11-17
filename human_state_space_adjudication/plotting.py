"""Registered diagnostic figures for the adjudication."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import MODEL_NAMES


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def make_figures(output: Path, tables: dict[str,pd.DataFrame]):
    figures=output/"figures"; figures.mkdir(parents=True,exist_ok=True)
    colors=dict(zip(MODEL_NAMES,plt.cm.tab10(np.linspace(0,.85,len(MODEL_NAMES)))))

    fig,ax=plt.subplots(figsize=(12,5)); ax.axis("off")
    labels=[("Lump","shared skill"),("Cross","7 A/B skill states\npartial cross-update"),("Split","7 A/B skill states"),
            ("M3","shared skill +\nstatic expression"),("M4","shared skill +\n1D learned expression")]
    for i,(name,text) in enumerate(labels):
        x=.1+i*.2; ax.text(x,.75,name,ha="center",weight="bold",fontsize=13)
        ax.add_patch(plt.Rectangle((x-.075,.28),.15,.34,fill=False,lw=2))
        ax.text(x,.45,text,ha="center",va="center",fontsize=9)
        ax.annotate("RT",(x,.13),(x,.28),arrowprops={"arrowstyle":"->"},ha="center")
    ax.set_title("Memory pathways and persistent expression pathways")
    _save(fig,figures/"figure_01_model_cartoons.png")

    hist=tables["historical_state_space_reproduction"]
    fig,ax=plt.subplots(figsize=(8,5)); h=hist.groupby("model").bic.mean().sort_values()
    ax.bar(h.index,h.values); ax.set(ylabel="Mean BIC",title="Historical Lump/Cross/Split reproduction")
    _save(fig,figures/"figure_02_historical_reproduction.png")

    cv=tables["human_state_space_forward_cv"]
    fig,ax=plt.subplots(figsize=(10,5)); groups=[cv.loc[cv.model.eq(m),"heldout_log_likelihood"] for m in MODEL_NAMES]
    ax.boxplot(groups,tick_labels=MODEL_NAMES,showfliers=False); ax.set(ylabel="Held-out log likelihood",title="Forward blockwise prediction")
    _save(fig,figures/"figure_03_forward_cv.png")

    confusion=tables["human_state_space_recovery_confusion"].set_index("generating_model").reindex(MODEL_NAMES)[list(MODEL_NAMES)]
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(confusion,cmap="Blues",vmin=0,vmax=1)
    ax.set(xticks=range(7),yticks=range(7),xticklabels=MODEL_NAMES,yticklabels=MODEL_NAMES,xlabel="Selected",ylabel="Generated",title="Model recovery")
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); fig.colorbar(im,ax=ax)
    _save(fig,figures/"figure_04_model_recovery.png")

    pr=tables["human_state_space_parameter_recovery"]
    fig,ax=plt.subplots(figsize=(7,6))
    if len(pr):
        for name,g in pr.groupby("generating_model"): ax.scatter(g.true_value,g.recovered_value,s=14,alpha=.5,label=name)
        lo=min(pr.true_value.min(),pr.recovered_value.min()); hi=max(pr.true_value.max(),pr.recovered_value.max()); ax.plot([lo,hi],[lo,hi],"k--")
    ax.set(xlabel="True",ylabel="Recovered",title="Parameter recovery"); ax.legend(fontsize=7,ncol=2)
    _save(fig,figures/"figure_05_parameter_recovery.png")

    latent=tables["human_state_space_latent_states"]
    mean=latent.groupby(["model","global_sequence","context"],as_index=False)[["observed","predicted"]].mean()
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True); axes=axes.ravel()
    for ax,model in zip(axes,MODEL_NAMES):
        g=mean[mean.model.eq(model)]
        for cx,style in [("A","-"),("B","--")]:
            z=g[g.context.eq(cx)]; ax.plot(z.global_sequence,z.observed,color="black",alpha=.35,ls=style); ax.plot(z.global_sequence,z.predicted,color=colors[model],ls=style)
        ax.set_title(model)
    axes[-1].axis("off"); fig.suptitle("Observed (black) and predicted acquisition by context")
    _save(fig,figures/"figure_06_observed_predicted_acquisition.png")

    prof=latent.groupby(["model","context","transition_position"],as_index=False)[["observed","predicted"]].mean()
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True); axes=axes.ravel()
    for ax,model in zip(axes,MODEL_NAMES):
        for cx,marker in [("A","o"),("B","s")]:
            g=prof[(prof.model==model)&(prof.context==cx)]; ax.plot(g.transition_position,g.observed,color="black",marker=marker,alpha=.4); ax.plot(g.transition_position,g.predicted,color=colors[model],marker=marker)
        ax.set_title(model)
    axes[-1].axis("off"); fig.suptitle("Sustained seven-transition A/B profiles")
    _save(fig,figures/"figure_07_transition_profiles.png")

    chosen=latent[latent.model.isin(["Cross","Split","M4"])].groupby(["model","global_sequence"],as_index=False)[["shared_skill","skill_A","skill_B","persistent_expression_A","persistent_expression_B"]].mean()
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,model in zip(axes,["Cross","Split","M4"]):
        g=chosen[chosen.model.eq(model)]; ax.plot(g.global_sequence,g.skill_A,label="skill A"); ax.plot(g.global_sequence,g.skill_B,label="skill B")
        if model=="M4": ax.plot(g.global_sequence,g.persistent_expression_A,label="expression A"); ax.plot(g.global_sequence,g.persistent_expression_B,label="expression B")
        ax.set_title(model); ax.legend(fontsize=7)
    _save(fig,figures/"figure_08_latent_trajectories.png")

    dec=tables["human_state_space_rt_decomposition"].groupby("model",as_index=False)[["memory_component","expression_component","residual"]].apply(lambda x:x.abs().mean()).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(dec)); ax.bar(x,dec.memory_component,label="memory"); ax.bar(x,dec.expression_component,bottom=dec.memory_component,label="expression")
    ax.set(xticks=x,xticklabels=dec.model,ylabel="Mean absolute component",title="RT difference decomposition"); ax.legend()
    _save(fig,figures/"figure_09_rt_decomposition.png")

    scores=tables["human_state_space_subject_scores"]; pivot=scores.pivot(index="participant_id",columns="model",values="bic"); delta=pivot.sub(pivot.min(axis=1),axis=0)
    fig,ax=plt.subplots(figsize=(11,6)); ax.imshow(delta.T,aspect="auto",cmap="magma_r",vmin=0,vmax=np.nanpercentile(delta,90)); ax.set(yticks=range(7),yticklabels=MODEL_NAMES,xlabel="Participant",title="Participant-level delta BIC")
    _save(fig,figures/"figure_10_participant_model_preference.png")

    late=latent.copy(); cutoff=late.groupby("participant_id").global_sequence.transform("max")*.75; late=late[late.global_sequence>=cutoff]
    late=late.groupby(["model","context","transition_position"],as_index=False)[["observed","predicted"]].mean()
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True); axes=axes.ravel()
    for ax,model in zip(axes,MODEL_NAMES):
        for cx in ["A","B"]:
            g=late[(late.model==model)&(late.context==cx)]; ax.plot(g.transition_position,g.observed,"k--",alpha=.35); ax.plot(g.transition_position,g.predicted,color=colors[model])
        ax.set_title(model)
    axes[-1].axis("off"); fig.suptitle("Late-training observed and predicted profiles")
    _save(fig,figures/"figure_11_late_training.png")
