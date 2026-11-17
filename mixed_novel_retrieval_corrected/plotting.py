from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig,path): fig.tight_layout(); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def make_figures(output,tables):
    root=Path(output)/"figures"; root.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,4)); ax.axis("off")
    for x,label in zip([.15,.5,.85],["Corrected familiar\nA/B learning","Frozen A/B anchors\nat probe time","Mixed / Novel /\nNoContext retrieval"]):
        ax.text(x,.5,label,ha="center",va="center",bbox=dict(boxstyle="round",fc="white",ec="black"),fontsize=12)
    ax.annotate("",(.39,.5),(.26,.5),arrowprops={"arrowstyle":"->"}); ax.annotate("",(.74,.5),(.61,.5),arrowprops={"arrowstyle":"->"})
    _save(fig,root/"figure_01_modeling_schematic.png")

    profile=tables["first_mixed_profile_predictions"]
    observed=profile.groupby("transition_position")[["Fast_anchor","Slow_anchor","observed_rt"]].mean()
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(observed.index,observed.Fast_anchor,"o-",label="Fast"); ax.plot(observed.index,observed.observed_rt,"o-",label="First Mixed"); ax.plot(observed.index,observed.Slow_anchor,"o-",label="Slow"); ax.legend(); ax.set(xlabel="Transition",ylabel="RT (ms)",title="First Mixed and frozen familiar endpoints")
    _save(fig,root/"figure_02_first_mixed_endpoints.png")

    comparison=tables["first_mixed_model_comparison"]
    fig,ax=plt.subplots(figsize=(8,5)); primary=comparison[comparison.model_code.isin(["R0","R1"])]
    ax.bar(primary.model_code,primary.predictive_log_likelihood); ax.set(ylabel="Participant-CV log likelihood",title="Categorical versus continuous retrieval")
    _save(fig,root/"figure_03_categorical_continuous_likelihood.png")

    fig,ax=plt.subplots(figsize=(8,5)); r1=profile[profile.retrieval_model.eq("R1_scalar_continuous")].groupby("transition_position")[["observed_rt","predicted_rt"]].mean()
    ax.plot(r1.index,r1.observed_rt,"ko-",label="Observed"); ax.plot(r1.index,r1.predicted_rt,"o-",label="R1"); ax.legend(); ax.set(xlabel="Transition",ylabel="RT (ms)",title="Scalar continuous fit")
    _save(fig,root/"figure_04_scalar_continuous_fit.png")

    fig,ax=plt.subplots(figsize=(9,5)); structured=comparison[comparison.model_code.isin(["R1","R2","R3"])]
    ax.bar(structured.model_code,structured.predictive_log_likelihood.fillna(structured.log_likelihood)); ax.set(ylabel="Predictive / fitted log likelihood",title="Structured retrieval comparison")
    _save(fig,root/"figure_05_structured_comparison.png")

    boot=tables["first_mixed_bootstrap"]
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(boot.q_R1,bins=30,alpha=.8); ax.axvline(boot.q_R1.median(),color="black"); ax.set(xlabel="R1 q(A)",ylabel="Bootstrap draws",title="First Mixed responsibility uncertainty")
    _save(fig,root/"figure_06_q_bootstrap.png")

    geometry=pd.concat([tables["first_mixed_geometry"],tables["novel_geometry"],tables["no_context_probe_geometry"]],ignore_index=True)
    fig,ax=plt.subplots(figsize=(9,5)); summary=geometry.groupby("probe_type")[["q_projection_A_to_B","off_axis_distance"]].mean(); ax.scatter(summary.q_projection_A_to_B,summary.off_axis_distance,s=90)
    for label,row in summary.iterrows(): ax.text(row.q_projection_A_to_B,row.off_axis_distance,label)
    ax.axvspan(0,1,color="grey",alpha=.1); ax.set(xlabel="Projection A→B",ylabel="Off-axis distance",title="Familiar-axis geometry")
    _save(fig,root/"figure_07_familiar_axis_geometry.png")

    exposure=tables["mixed_all_exposures_parameters"]; q=exposure[(exposure.model_code=="R1")&(exposure.parameter=="q_A")]
    fig,ax=plt.subplots(figsize=(9,5));
    for phase,g in q.groupby("probe_phase"): ax.plot(g.probe_exposure_index,g.estimate,"o-",label=phase)
    ax.legend(); ax.set(xlabel="Mixed exposure",ylabel="q(A)",title="First versus later Mixed")
    _save(fig,root/"figure_08_mixed_exposure_evolution.png")

    sens=tables["familiar_model_sensitivity"]
    fig,ax=plt.subplots(figsize=(9,5)); q=sens[(sens.model_code=="R1")]
    ax.bar(q.anchor_model,q.q_A); ax.set(ylabel="R1 q(A)",title="Familiar-model sensitivity")
    _save(fig,root/"figure_09_familiar_model_sensitivity.png")

    uncertainty=tables["anchor_uncertainty_sensitivity"]
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(uncertainty.q_R1,bins=30); ax.axvline(tables["first_mixed_parameter_estimates"].query("model_code=='R1' and parameter=='q_A'").estimate.iloc[0],color="red",label="Fixed anchors"); ax.legend(); ax.set(xlabel="R1 q(A)",title="Propagated familiar-anchor uncertainty")
    _save(fig,root/"figure_10_anchor_uncertainty.png")
