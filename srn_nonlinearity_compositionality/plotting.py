from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS={"High-Scale + High Context":"#D55E00","Low-Scale + Low Context":"#0072B2"}


def _save(fig,output,name):
    fig.tight_layout(); fig.savefig(output/name,dpi=180,bbox_inches="tight"); plt.close(fig)


def _line(frame,y,ax,ylabel=None):
    for condition,sub in frame.groupby("condition_label",sort=False):
        values=sub.groupby("cue_lambda")[y].agg(["mean","sem"])
        color=COLORS.get(condition)
        ax.plot(values.index,values["mean"],label=condition,color=color)
        ax.fill_between(values.index,values["mean"]-values["sem"],values["mean"]+values["sem"],alpha=.15,color=color)
    ax.set_xlabel("Cue lambda (A → B)"); ax.set_ylabel(ylabel or y.replace("_"," "))


def make_figures(output:Path,t):
    output=Path(output); decoder=t["decoder_loso_results"]; hidden=t["cue_sweep_hidden_residuals"]
    out=t["cue_sweep_output_residuals"]; logit=t["cue_sweep_logit_residuals"]
    gain=t["cue_sweep_tanh_gain"]; saturation=t["cue_sweep_saturation"]
    curvature=t["cue_sweep_curvature"]; recurrent=t["recurrent_jacobian_metrics"]
    master=t["srn_mechanism_master_table"]; formal=t["formal_model_familiar_fit"]
    prediction=t["formal_model_mixed_prediction"]; patch=t["causal_linear_path_patching"]
    linearized=t["linearized_activation_counterfactual"]
    conditions=list(dict.fromkeys(decoder.condition_label))

    fig,ax=plt.subplots(figsize=(9,4.5)); contexts=["A","B","Mixed","Novel","NoContextProbe"]
    x=np.arange(len(contexts)); width=.8/len(conditions)
    for i,condition in enumerate(conditions):
        values=decoder[decoder.condition_label.eq(condition)].groupby("test_context").decoder_accuracy.mean().reindex(contexts)
        ax.bar(x+(i-(len(conditions)-1)/2)*width,values,width,label=condition,color=COLORS.get(condition))
    ax.axhline(1/7,color="black",ls=":"); ax.set_xticks(x,contexts); ax.set_ylabel("LOSO position-decoder accuracy"); ax.legend(fontsize=8)
    _save(fig,output,"figure_01_decoder_apples_to_apples.png")

    fig,ax=plt.subplots(figsize=(8,4.5)); _line(hidden,"E_whole",ax,"Whole-state normalized residual energy"); ax.legend(fontsize=8)
    _save(fig,output,"figure_02_hidden_linear_residual_vs_lambda.png")

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,["E_out","E_null"],["Output-relevant","Output-null"]): _line(out,metric,ax,title+" normalized residual energy")
    axes[-1].legend(fontsize=8); _save(fig,output,"figure_03_output_relevant_vs_null_residual.png")

    fig,ax=plt.subplots(figsize=(8,4.5)); _line(logit,"E_logit",ax,"Normalized logit residual energy"); ax.legend(fontsize=8)
    _save(fig,output,"figure_04_logit_residual_vs_lambda.png")

    fig,ax=plt.subplots(figsize=(8,4.5)); _line(gain,"mean_tanh_gain",ax,"Mean tanh local gain"); ax.legend(fontsize=8)
    _save(fig,output,"figure_05_tanh_gain_vs_lambda.png")

    fig,ax=plt.subplots(figsize=(8,4.5))
    styles={"saturation_08":"-","saturation_09":"--","saturation_095":":"}
    for condition,sub in saturation.groupby("condition_label",sort=False):
        for metric,style in styles.items():
            values=sub.groupby("cue_lambda")[metric].mean(); ax.plot(values.index,values,label=f"{condition}: {metric[-3:]}",ls=style,color=COLORS.get(condition))
    ax.set_xlabel("Cue lambda"); ax.set_ylabel("Saturated-unit fraction"); ax.legend(fontsize=7,ncol=2)
    _save(fig,output,"figure_06_saturation_vs_lambda.png")

    fig,ax=plt.subplots(figsize=(8,4.5)); _line(curvature,"K_whole",ax,"Whole hidden curvature"); ax.legend(fontsize=8)
    _save(fig,output,"figure_07_hidden_curvature_vs_lambda.png")

    fig,ax=plt.subplots(figsize=(8,4.5)); _line(curvature,"K_out",ax,"Output-relevant curvature"); ax.legend(fontsize=8)
    _save(fig,output,"figure_08_output_curvature_vs_lambda.png")

    fig,ax=plt.subplots(figsize=(8,4.5)); _line(curvature,"K_logit",ax,"Logit curvature"); ax.legend(fontsize=8)
    _save(fig,output,"figure_09_logit_curvature_vs_lambda.png")

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    mixed=recurrent[np.isclose(recurrent.cue_lambda,.5)]
    for condition,sub in mixed.groupby("condition_label",sort=False):
        mean=sub.groupby("position")[["spectral_norm","cumulative_sensitivity_norm"]].mean()
        axes[0].plot(mean.index,mean.spectral_norm,marker="o",label=condition,color=COLORS.get(condition))
        axes[1].plot(mean.index,mean.cumulative_sensitivity_norm,marker="o",label=condition,color=COLORS.get(condition))
    axes[0].set_ylabel("Local recurrent Jacobian norm"); axes[1].set_ylabel("Cumulative sensitivity norm")
    for ax in axes: ax.set_xlabel("Sequence position")
    axes[1].legend(fontsize=8); _save(fig,output,"figure_10_recurrent_jacobian_spectrum.png")

    fig,ax=plt.subplots(figsize=(7,5))
    for condition,sub in master.groupby("condition_label",sort=False): ax.scatter(sub.K_out_mid,sub.Mixed_loss,label=condition,alpha=.8,color=COLORS.get(condition))
    ax.set_xlabel("Output-relevant curvature at midpoint"); ax.set_ylabel("Mixed sequence loss"); ax.legend(fontsize=8)
    _save(fig,output,"figure_11_curvature_vs_loss_seed_scatter.png")

    fig,ax=plt.subplots(figsize=(7,5))
    for condition,sub in master.groupby("condition_label",sort=False): ax.scatter(sub.P_interaction,sub.K_out_mid,label=condition,alpha=.8,color=COLORS.get(condition))
    ax.set_xlabel("Familiar context × sequence interaction fraction"); ax.set_ylabel("Mixed output curvature"); ax.legend(fontsize=8)
    _save(fig,output,"figure_12_interaction_vs_curvature_seed_scatter.png")

    fig,ax=plt.subplots(figsize=(9,4.5)); models=["F0_shared_sequence","F1_additive","F2_bilinear","F3_unconstrained"]
    x=np.arange(len(models)); width=.8/len(conditions)
    for i,condition in enumerate(conditions):
        values=formal[formal.condition_label.eq(condition)].groupby("model").familiar_R2.mean().reindex(models)
        ax.bar(x+(i-(len(conditions)-1)/2)*width,values,width,label=condition,color=COLORS.get(condition))
    ax.set_xticks(x,[v.split("_",1)[0] for v in models]); ax.set_ylabel("Familiar reconstruction R²"); ax.legend(fontsize=8)
    _save(fig,output,"figure_13_formal_model_reconstruction.png")

    fig,axes=plt.subplots(1,3,figsize=(13,4)); metrics=["Mixed_whole_R2","Mixed_output_R2","Mixed_logit_R2"]
    pred_models=["F0_shared_sequence","F1_additive","F2_bilinear"]
    for ax,metric in zip(axes,metrics):
        for condition,sub in prediction.groupby("condition_label",sort=False):
            values=sub.groupby("model")[metric].mean().reindex(pred_models)
            ax.plot([m[:2] for m in pred_models],values,marker="o",label=condition,color=COLORS.get(condition))
        ax.axhline(0,color="black",lw=.7); ax.set_title(metric.replace("_"," "),fontsize=9)
    axes[-1].legend(fontsize=8); _save(fig,output,"figure_14_formal_model_mixed_prediction.png")

    fig,ax=plt.subplots(figsize=(10,4.5)); patches=["full_linear_path","output_only_linear_path","null_only_linear_path"]
    x=np.arange(len(patches)); width=.8/len(conditions)
    for i,condition in enumerate(conditions):
        values=patch[patch.condition_label.eq(condition)].groupby("patch_type").delta_loss_vs_actual.mean().reindex(patches)
        ax.bar(x+(i-(len(conditions)-1)/2)*width,values,width,label=condition,color=COLORS.get(condition))
    ax.axhline(0,color="black",lw=.8); ax.set_xticks(x,[p.replace("_","\n") for p in patches]); ax.set_ylabel("Patched − actual Mixed loss"); ax.legend(fontsize=8)
    _save(fig,output,"figure_15_linear_path_patching.png")

    fig,ax=plt.subplots(figsize=(8,4.5)); nearest=linearized[linearized.linearization_anchor.eq("nearest_endpoint")]
    for condition,sub in nearest.groupby("condition_label",sort=False):
        actual=sub.groupby("cue_lambda").actual_sequence_loss.mean(); counter=sub.groupby("cue_lambda").linearized_sequence_loss.mean()
        ax.plot(actual.index,actual,label=f"{condition}: tanh",color=COLORS.get(condition)); ax.plot(counter.index,counter,ls="--",label=f"{condition}: endpoint-linearized",color=COLORS.get(condition))
    ax.set_xlabel("Cue lambda"); ax.set_ylabel("Sequence loss"); ax.legend(fontsize=7,ncol=2)
    _save(fig,output,"figure_16_linearized_activation_counterfactual.png")

    fig,ax=plt.subplots(figsize=(8,5.5)); sizes=35+165*master.K_out_mid/master.K_out_mid.max()
    for condition,sub in master.groupby("condition_label",sort=False):
        index=sub.index; ax.scatter(sub.AB_hidden_separation,sub.Mixed_loss,s=sizes.loc[index],alpha=.7,label=condition,color=COLORS.get(condition),edgecolor="white")
    ax.set_xlabel("Familiar A/B hidden separation"); ax.set_ylabel("Mixed sequence loss"); ax.legend(fontsize=8,title="Size = output curvature")
    ax.text(.02,.98,"Target: high separation, low loss, small points",transform=ax.transAxes,va="top",fontsize=9)
    _save(fig,output,"figure_17_summary_tradeoff_map.png")
