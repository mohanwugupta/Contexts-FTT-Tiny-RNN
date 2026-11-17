"""Required synthetic, human, and causal figures with explicit gate placeholders."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig,output,name): fig.tight_layout();fig.savefig(output/name,dpi=170);plt.close(fig)
def _placeholder(output,name,title,message):
    fig,ax=plt.subplots(figsize=(8,3));ax.axis("off");ax.text(.5,.62,title,ha="center",fontsize=14);ax.text(.5,.34,message,ha="center",wrap=True);_save(fig,output,name)


def synthetic_figures(output,registry,validation,gates,continuum):
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,row in zip(axes,registry.itertuples()): ax.axis("off");ax.text(.5,.7,row.label,ha="center",fontsize=12);ax.text(.5,.38,f"parameters: {row.parameter_count:,}\ncontext state: {row.context_size}D",ha="center");ax.set_title(f"Model {row.architecture}")
    _save(fig,output,"figure_01_architecture_schematic.png")
    fig,ax=plt.subplots(figsize=(10,3));ax.axis("off");ax.text(.02,.72,"10 clear A  →  10 clear B  →  10 alternating clear A/B  →  6 ambiguous",fontsize=13);ax.text(.02,.38,"latent cause: task law    |    observable cue/layout: noisy evidence only",fontsize=12);ax.text(.02,.08,"structured and shuffled cells share all marginals; only evidence→task assignment differs",fontsize=11);_save(fig,output,"figure_02_latent_cause_ecology.png")
    split=validation["split"].groupby(["architecture","structure_condition","seed"],as_index=False).AB_hidden_distance.mean();amb=validation["ambiguity"].groupby(["architecture","structure_condition","seed"],as_index=False).ambiguous_query_NLL.mean();front=split.merge(amb)
    fig,ax=plt.subplots(figsize=(7,5))
    for key,one in front.groupby(["architecture","structure_condition"]): ax.scatter(one.AB_hidden_distance,-one.ambiguous_query_NLL,label=f"{key[0]} {key[1]}")
    ax.set(xlabel="Clear A/B hidden differentiation",ylabel="Ambiguous transfer quality (−NLL)",title="Synthetic split–ambiguity frontier");ax.legend(fontsize=8);_save(fig,output,"figure_03_split_ambiguity_frontier.png")
    fig,ax=plt.subplots(figsize=(8,4));summary=continuum.groupby(["architecture","structure_condition","reliability","evidence_lean"],as_index=False).decoded_p_A.mean()
    for key,one in summary[summary.structure_condition.eq("structured")].groupby(["architecture","evidence_lean"]): ax.plot(one.reliability,one.decoded_p_A,marker="o",label=f"{key[0]} lean {key[1]}")
    ax.set(xlabel="Cue reliability",ylabel="Decoded P(A)",title="Uncertainty continuum");ax.legend(fontsize=8,ncol=2);_save(fig,output,"figure_04_uncertainty_continuum.png")
    fig,ax=plt.subplots(figsize=(8,4));switch=validation["switch"].groupby(["architecture","structure_condition"],as_index=False).prediction_recovery_sequences.mean();structured=switch[switch.structure_condition.eq("structured")];ax.bar(structured.architecture,structured.prediction_recovery_sequences);ax.set(ylabel="Sequences to recover",title="Switch recovery");_save(fig,output,"figure_05_switch_recovery.png")
    fig,ax=plt.subplots(figsize=(7,5));shared=validation["shared"].groupby(["architecture","structure_condition","seed"],as_index=False).shared_token_hidden_cosine.mean();both=split.merge(shared);structured=both[both.structure_condition.eq("structured")]
    for architecture,one in structured.groupby("architecture"): ax.scatter(one.AB_hidden_distance,one.shared_token_hidden_cosine,label=architecture)
    ax.set(xlabel="A/B differentiation",ylabel="Shared-transition hidden cosine",title="Shared versus context-specific geometry");ax.legend();_save(fig,output,"figure_06_shared_context_geometry.png")


def human_figures(output,human=None,interventions=None):
    if human is None:
        for number,title in ((7,"Human familiar A/B phenotype"),(8,"First Mixed profile"),(9,"Controller-state geometry"),(10,"Model-C gate overlap"),(11,"Differentiation–transfer frontier"),(12,"Causal interventions")):
            stem=title.lower().replace(" ","_").replace("–","_").replace("-","_").replace("/","_")
            _placeholder(output,f"figure_{number:02d}_{stem}.png",title,"Not run: no architecture passed the prerequisite confirmatory synthetic gate.")
        return
    gates=human["gates"];run=gates[gates.row_type.eq("run")];fig,ax=plt.subplots(figsize=(9,4));summary=run.groupby("group").gate_passed.mean();ax.bar(summary.index,summary.values);ax.tick_params(axis="x",rotation=25);ax.set(ylabel="Fraction passing familiar gate",title="Human familiar A/B phenotype");_save(fig,output,"figure_07_human_familiar_AB.png")
    mixed=human["mixed"];fig,ax=plt.subplots(figsize=(9,4));summary=mixed.groupby(["group","transition_position"],as_index=False).RT_deficit.mean()
    for group,one in summary.groupby("group"): ax.plot(one.transition_position,one.RT_deficit,label=group)
    ax.legend(fontsize=7);ax.set(xlabel="Transition",ylabel="First Mixed RT deficit",title="First Mixed profile");_save(fig,output,"figure_08_first_mixed_profile.png")
    context=human["context_state"];cols=[c for c in context if c.startswith("context_state_")]
    if cols:
        fig,ax=plt.subplots(figsize=(7,5));values=context.groupby(["architecture","context"])[cols].mean();
        for key,row in values.iterrows(): ax.scatter(row.iloc[0],row.iloc[1],label=f"{key[0]} {key[1]}")
        ax.legend(fontsize=7);ax.set(xlabel="Controller dim 1",ylabel="Controller dim 2",title="Controller-state geometry");_save(fig,output,"figure_09_controller_geometry.png")
    else:_placeholder(output,"figure_09_controller_geometry.png","Controller-state geometry","Selected architecture has no explicit controller.")
    gate=human["gate_vectors"];gcols=[c for c in gate if c.startswith("gate_")]
    if gcols:
        fig,ax=plt.subplots(figsize=(8,4));means=gate.groupby("context")[gcols].mean();ax.imshow(means.to_numpy(),aspect="auto");ax.set_yticks(range(len(means)),means.index);ax.set(xlabel="Gate unit",title="Model-C gate overlap");_save(fig,output,"figure_10_gate_overlap.png")
    else:_placeholder(output,"figure_10_gate_overlap.png","Model-C gate overlap","Model C was not selected or did not reach human transfer.")
    fig,ax=plt.subplots(figsize=(7,5));gm=human["geometry"].groupby(["group","seed"],as_index=False).off_corridor_distance.mean();hm=run[["group","seed","AB_hidden_distance"]].merge(gm); 
    for group,one in hm.groupby("group"): ax.scatter(one.AB_hidden_distance,-one.off_corridor_distance,label=group)
    ax.legend(fontsize=7);ax.set(xlabel="A/B differentiation",ylabel="Mixed transfer (−off-corridor)",title="Differentiation–transfer frontier");_save(fig,output,"figure_11_human_frontier.png")
    intervention_frame=None if interventions is None else interventions.get("context_state_swap")
    if intervention_frame is not None and len(intervention_frame):
        fig,ax=plt.subplots(figsize=(8,4));summary=intervention_frame.groupby("group").prediction_NLL.mean();ax.bar(summary.index,summary.values);ax.tick_params(axis="x",rotation=20);ax.set(ylabel="NLL under context-state swap",title="Causal interventions");_save(fig,output,"figure_12_causal_interventions.png")
    else:_placeholder(output,"figure_12_causal_interventions.png","Causal interventions","No context-controlled group passed the human familiar gate.")
