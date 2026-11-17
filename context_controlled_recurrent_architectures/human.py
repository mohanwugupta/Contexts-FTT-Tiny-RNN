"""Post-selection human transfer, familiar gates, probes, and frozen interventions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from confirmatory_srn.state_space import fit_confirmatory_state_space
from episodic_meta_learning_capacity.human import human_features
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from human_svg_geometry_transfer.geometry import compose_features
from human_svg_geometry_transfer.geometry import geometry_features
from transfer_experiment_program.modeling import context_at_sequence
from transfer_experiment_program.modeling import SRNEngine

from .models import build_model, model_hash


@dataclass
class HumanRun:
    group: str
    architecture: str
    source: str
    seed: int
    model: object
    policy: object
    training: pd.DataFrame
    probes: pd.DataFrame


def _features(registry,context):
    if context in {"A","B","Mixed","Novel"}: return human_features(registry,context)
    layout=np.column_stack([registry.pooled_x,np.zeros(8,np.float32)]).reshape(-1); return compose_features(np.zeros(2,np.float32),layout,layout)


def _batch(registry,context,sequence):
    sequence=torch.as_tensor(sequence,dtype=torch.long); tokens=F.one_hot(sequence[:-1],4).float().unsqueeze(0); evidence=torch.as_tensor(_features(registry,context),dtype=torch.float32).unsqueeze(0); return tokens,evidence,sequence[1:].unsqueeze(0)


def _forward(model,policy,batch,*,overrides=None):
    tokens,evidence,targets=batch; output=model(tokens,evidence,overrides=overrides); logits=output["logits"]; confidence,entropy,margin=predictive_statistics(logits,targets.squeeze(0)); speed=policy(confidence); rt=normalized_rt(speed)
    ce=F.cross_entropy(logits.reshape(-1,4),targets.reshape(-1)); candidate=ObjectiveCandidate("A",.5,1.,2.,.2); execution,optimum,_=execution_loss(logits,targets.squeeze(0),speed,candidate)
    return output,targets,confidence,entropy,margin,rt,optimum,ce,execution.mean()


def _probe_rows(run,registry,checkpoint,sequence):
    rows=[]; before=model_hash(run.model); run.model.eval(); run.policy.eval()
    with torch.no_grad():
        for context in ("A","B","Mixed","Novel"):
            output,targets,confidence,entropy,margin,rt,optimum,ce,execution=_forward(run.model,run.policy,_batch(registry,context,sequence))
            for position in range(7):
                logits=output["logits"][0,position]; row={"group":run.group,"architecture":run.architecture,"source":run.source,"seed":run.seed,"checkpoint":checkpoint,"context":context,
                    "transition_position":position+1,"current_token":sequence[position],"target_token":sequence[position+1],"predicted_token":int(logits.argmax()),
                    "correct_token_probability":float(confidence[0,position]),"correct_token_nll":float(F.cross_entropy(logits.unsqueeze(0),targets[0,position].view(1))),
                    "token_correct":bool(logits.argmax()==targets[0,position]),"predictive_entropy":float(entropy[0,position]),"probability_margin":float(margin[0,position]),
                    "normalized_RT":float(rt[0,position]),"probe_exposure_index":1,"primary_first_exposure":True,"weight_policy":"frozen","optimizer_updates":0,
                    **{f"hidden_state_{i+1}":float(value) for i,value in enumerate(output["hidden"][0,position])}}
                if output["context_state"] is not None: row.update({f"context_state_{i+1}":float(value) for i,value in enumerate(output["context_state"][0,position])})
                if output["gate"] is not None: row.update({f"gate_{i+1}":float(value) for i,value in enumerate(output["gate"][0,position])})
                rows.append(row)
    if model_hash(run.model)!=before: raise AssertionError("T8/T12: frozen human probe mutated parameters.")
    return rows


def train_human_model(config,registry,architecture,source,seed,state,*,restricted=False):
    settings=config.payload["human_transfer"]; model=build_model(architecture,seed); model.load_state_dict(deepcopy(state)); policy=MonotonicExecutionPolicy()
    group=f"{architecture}_{source}"+("_restricted" if restricted else ""); parameters=list(model.parameters())+list(policy.parameters())
    if restricted and architecture in {"B","C"}:
        for parameter in model.shared_core_parameters(): parameter.requires_grad_(False)
        parameters=[p for p in model.parameters() if p.requires_grad]+list(policy.parameters())
    optimizer=torch.optim.Adam(parameters,lr=float(settings["learning_rate"])); sequence=tuple(settings["token_sequence"]); rows=[]; probes=[]
    run=HumanRun(group,architecture,source,seed,model,policy,pd.DataFrame(),pd.DataFrame())
    for index in range(1,int(settings["training_sequences"])+1):
        context=context_at_sequence(index,block_length=int(settings["block_length"])); model.train();policy.train();optimizer.zero_grad()
        output,targets,confidence,entropy,margin,rt,optimum,ce,execution=_forward(model,policy,_batch(registry,context,sequence)); total=ce+float(settings["lambda_execution"])*execution
        total.backward(); torch.nn.utils.clip_grad_norm_(parameters,float(settings["gradient_clip"]));optimizer.step()
        rows.append({"group":group,"condition":group,"architecture":architecture,"source":source,"restricted_plasticity":restricted,"seed":seed,"training_sequence":index,"context":context,
            "block_index":(index-1)//10+1,"within_block_sequence":(index-1)%10+1,"is_context_switch":bool(index>1 and (index-1)%10==0),
            "mean_prediction_NLL":float(ce.detach()),"mean_correct_token_probability":float(confidence.mean().detach()),"token_accuracy":float(output["logits"].argmax(-1).eq(targets).float().mean()),
            "mean_normalized_RT":float(rt.mean().detach()),"execution_active":True,"optimizer_steps":index})
        if index in set(settings["probe_checkpoints"]): probes.extend(_probe_rows(run,registry,index,sequence))
    run.training=pd.DataFrame(rows);run.probes=pd.DataFrame(probes);return run


def exact_h8_human_run(config,registry,seed):
    """Run the original validated H8 engine on the same G2 target contract."""
    settings=config.payload["human_transfer"]; engine=SRNEngine("High_monotonic",seed,max_sequences=170); training=[];probes=[];sequence=tuple(settings["token_sequence"])
    for index in range(1,171):
        context=context_at_sequence(index,block_length=10); _,row,_=engine.score_sequence(context=context,branch_id=f"context-control:H8:seed{seed}:train:{index}",update=True,
            features_np=geometry_features("G2_shared_x",context,registry),completed_training_sequences=index-1,record_transitions=False,audit_hashes=False,
            metadata={"experiment":"context_controlled_recurrent_architectures","cohort":"H8_exact","evaluation_role":"A_B_training","state_policy":"reset","primer":"none","repetition":index})
        training.append({"group":"H8_exact","condition":"H8_exact","architecture":"H8","source":"exact_validated_baseline","restricted_plasticity":False,"seed":seed,
            "training_sequence":index,"context":context,"block_index":(index-1)//10+1,"within_block_sequence":(index-1)%10+1,"is_context_switch":bool(index>1 and (index-1)%10==0),
            "mean_prediction_NLL":row["mean_prediction_NLL"],"mean_correct_token_probability":row["mean_correct_token_probability"],"token_accuracy":row["token_accuracy"],"mean_normalized_RT":row["mean_normalized_RT"],"execution_active":True,"optimizer_steps":index})
        if index in set(settings["probe_checkpoints"]):
            for probe_context in settings["probe_contexts"]:
                frame,_,_=engine.score_sequence(context=probe_context,branch_id=f"context-control:H8:seed{seed}:t{index}:{probe_context}",update=False,
                    features_np=geometry_features("G2_shared_x",probe_context,registry),completed_training_sequences=index,
                    metadata={"experiment":"context_controlled_recurrent_architectures","cohort":"H8_exact","evaluation_role":"first_frozen_probe","state_policy":"reset","primer":"none","repetition":1})
                frame=frame.rename(columns={"parent_checkpoint":"checkpoint"}); frame["group"]="H8_exact";frame["architecture"]="H8";frame["source"]="exact_validated_baseline";frame["probe_exposure_index"]=1;frame["primary_first_exposure"]=True;probes.append(frame)
    return HumanRun("H8_exact","H8","exact_validated_baseline",seed,engine.model,engine.policy,pd.DataFrame(training),pd.concat(probes,ignore_index=True))


def canonicalize(probes):
    parts=[]
    for key,one in probes.groupby(["group","architecture","source","seed","checkpoint"],sort=True):
        means=one[one.context.isin(["A","B"])].groupby("context").normalized_RT.mean(); fast,slow=("A","B") if means.A<=means.B else ("B","A")
        value=one.copy();value["fast_source_context"]=fast;value["slow_source_context"]=slow;value["canonical_context"]=value.context.replace({fast:"Fast",slow:"Slow"});value["canonicalization_scope"]="whole_7_transition_profile";parts.append(value)
    return pd.concat(parts,ignore_index=True)


def analyze_human(config,runs):
    training=pd.concat([r.training for r in runs],ignore_index=True);probes=pd.concat([r.probes for r in runs],ignore_index=True);canonical=canonicalize(probes)
    hcols=sorted([c for c in probes if c.startswith("hidden_state_")],key=lambda x:int(x.rsplit("_",1)[1])); ccols=sorted([c for c in probes if c.startswith("context_state_")],key=lambda x:int(x.rsplit("_",1)[1])); gcols=sorted([c for c in probes if c.startswith("gate_")],key=lambda x:int(x.rsplit("_",1)[1])); metrics=[];geometry=[];state_rows=[];gate_rows=[]
    fits,classes,predictions=fit_confirmatory_state_space(training); winners=classes[classes.row_type.eq("seed")].set_index(["condition","seed"]).model_label
    settled=max(config.payload["human_transfer"]["probe_checkpoints"])
    for (group,architecture,source,seed),one_training in training.groupby(["group","architecture","source","seed"]):
        ordered=one_training.sort_values("training_sequence"); early=ordered.head(20);late=ordered.tail(20); one=canonical[(canonical.group.eq(group))&canonical.seed.eq(seed)&canonical.checkpoint.eq(settled)];lookup=one.set_index(["canonical_context","transition_position"])
        distances=[];rt_gaps=[];controller_distances=[];gate_distances=[]
        for k in range(1,8):
            fast=lookup.loc[("Fast",k)];slow=lookup.loc[("Slow",k)];distances.append(np.linalg.norm(fast[hcols].dropna().to_numpy(float)-slow[hcols].dropna().to_numpy(float)));rt_gaps.append(abs(fast.normalized_RT-slow.normalized_RT))
            if ccols and fast[ccols].notna().any(): controller_distances.append(np.linalg.norm(fast[ccols].dropna().to_numpy(float)-slow[ccols].dropna().to_numpy(float)))
            if gcols and fast[gcols].notna().any(): gate_distances.append(np.linalg.norm(fast[gcols].dropna().to_numpy(float)-slow[gcols].dropna().to_numpy(float)))
        correlation=float(np.corrcoef(ordered.mean_correct_token_probability,ordered.mean_normalized_RT)[0,1]); winner=winners.get((group,seed),"missing")
        checks={"F1_sequence_learning":late.mean_prediction_NLL.mean()<early.mean_prediction_NLL.mean(),"F2_AB_hidden_differentiation":np.mean(distances)>1e-6,
                "F3_context_yoked_RT":np.mean(rt_gaps)>=.005,"F4_Cross_or_Split":winner in {"Cross","Split"},"F5_negative_confidence_RT":correlation<0}
        gate_rows.append({"row_type":"run","group":group,"architecture":architecture,"source":source,"seed":seed,"state_space_winner":winner,"AB_hidden_distance":np.mean(distances),
                          "AB_controller_state_distance":float(np.mean(controller_distances)) if controller_distances else np.nan,"AB_gate_distance":float(np.mean(gate_distances)) if gate_distances else np.nan,
                          "AB_RT_gap":np.mean(rt_gaps),"confidence_RT_correlation":correlation,**checks,"gate_passed":bool(all(checks.values()))})
    run_gates=pd.DataFrame(gate_rows); group_rows=[]
    for (group,architecture,source),one in run_gates.groupby(["group","architecture","source"]):
        required=int(math.ceil(.8*len(one))); checks={column:int(one[column].sum())>=required for column in ("F1_sequence_learning","F2_AB_hidden_differentiation","F3_context_yoked_RT","F4_Cross_or_Split","F5_negative_confidence_RT")}
        group_rows.append({"row_type":"group","group":group,"architecture":architecture,"source":source,"seed":np.nan,"n_seeds":len(one),"required_passing_seeds":required,**checks,"gate_passed":bool(all(checks.values()))})
    gates=pd.concat([run_gates,pd.DataFrame(group_rows)],ignore_index=True,sort=False)
    for key,one in canonical.groupby(["group","architecture","source","seed","checkpoint"]):
        lookup=one.set_index(["canonical_context","transition_position"])
        for context in ("Mixed","Novel"):
            for k in range(1,8):
                fast,slow,probe=(lookup.loc[(name,k)] for name in ("Fast","Slow",context)); lower=min(fast.normalized_RT,slow.normalized_RT);upper=max(fast.normalized_RT,slow.normalized_RT)
                fh=fast[hcols].dropna().to_numpy(float);sh=slow[hcols].dropna().to_numpy(float);ph=probe[hcols].dropna().to_numpy(float);direction=sh-fh;den=float(direction@direction);coordinate=float((ph-fh)@direction/den) if den>1e-14 else np.nan;clipped=np.clip(coordinate,0,1) if np.isfinite(coordinate) else 0
                metrics.append(dict(zip(["group","architecture","source","seed","checkpoint"],key))|{"probe_context":context,"transition_position":k,"probe_exposure_index":1,
                    "between_endpoints":lower<=probe.normalized_RT<=upper,"probe_RT":probe.normalized_RT,"Fast_RT":fast.normalized_RT,"Slow_RT":slow.normalized_RT,
                    "RT_deficit":max(lower-probe.normalized_RT,0,probe.normalized_RT-upper),"prediction_NLL":probe.correct_token_nll,
                    "prediction_NLL_deficit":max(min(fast.correct_token_nll,slow.correct_token_nll)-probe.correct_token_nll,0,probe.correct_token_nll-max(fast.correct_token_nll,slow.correct_token_nll)),"token_accuracy":probe.token_correct})
                off=float(np.linalg.norm(ph-(fh+clipped*direction)));geometry.append(dict(zip(["group","architecture","source","seed","checkpoint"],key))|{"probe_context":context,"transition_position":k,"projection_coordinate":coordinate,"inside_corridor":bool(np.isfinite(coordinate) and 0<=coordinate<=1),"off_corridor_distance":off,"shared_subspace_engagement":float(1/(1+off))})
    metrics=pd.DataFrame(metrics);geometry=pd.DataFrame(geometry)
    context_state=probes[[c for c in probes if not c.startswith("hidden_state_") and not c.startswith("gate_")]].copy(); context_cols=[c for c in probes if c.startswith("context_state_")]; context_state=context_state[["group","architecture","source","seed","checkpoint","context","transition_position","probe_exposure_index"]+context_cols]
    gate_cols=[c for c in probes if c.startswith("gate_")]; gate_vectors=probes[["group","architecture","source","seed","checkpoint","context","transition_position","probe_exposure_index"]+gate_cols]
    return {"training":training,"gates":gates,"probes":probes,"mixed":metrics[metrics.probe_context.eq("Mixed")],"novel":metrics[metrics.probe_context.eq("Novel")],"geometry":geometry,
            "context_state":context_state,"gate_vectors":gate_vectors,"state_space":fits,"state_predictions":predictions}


def interventions(config,registry,runs,passing_groups):
    rows={"context_state_swap":[],"gate_swap":[],"controller_ablation":[],"artificial_interpolation":[]};sequence=tuple(config.payload["human_transfer"]["token_sequence"])
    for run in runs:
        if run.group not in passing_groups or run.architecture not in {"B","C"}: continue
        before=model_hash(run.model);run.model.eval();run.policy.eval()
        with torch.no_grad():
            base={}
            for context in ("A","B","Mixed"):
                batch=_batch(registry,context,sequence); values=_forward(run.model,run.policy,batch);base[context]=(batch,values)
            for context,donor in (("A","B"),("B","A")):
                forced=base[donor][1][0]["context_state"]; values=_forward(run.model,run.policy,base[context][0],overrides={"context_state":forced});rows["context_state_swap"].append({"group":run.group,"architecture":run.architecture,"seed":run.seed,"recipient":context,"donor":donor,"prediction_NLL":float(values[7]),"mean_RT":float(values[5].mean())})
                if run.architecture=="C":
                    gate=base[donor][1][0]["gate"]; values=_forward(run.model,run.policy,base[context][0],overrides={"gate":gate});rows["gate_swap"].append({"group":run.group,"architecture":run.architecture,"seed":run.seed,"recipient":context,"donor":donor,"prediction_NLL":float(values[7]),"mean_RT":float(values[5].mean())})
            zero=torch.zeros_like(base["A"][1][0]["context_state"]); values=_forward(run.model,run.policy,base["A"][0],overrides={"context_state":zero});rows["controller_ablation"].append({"group":run.group,"architecture":run.architecture,"seed":run.seed,"context":"A","ablation":"zero","prediction_NLL":float(values[7]),"mean_RT":float(values[5].mean())})
            midpoint=.5*(base["A"][1][0]["context_state"]+base["B"][1][0]["context_state"]); values=_forward(run.model,run.policy,base["Mixed"][0],overrides={"context_state":midpoint});rows["artificial_interpolation"].append({"group":run.group,"architecture":run.architecture,"seed":run.seed,"interpolation":"0.5_context_A_plus_B","prediction_NLL":float(values[7]),"mean_RT":float(values[5].mean())})
            if run.architecture=="C":
                midpoint_gate=.5*(base["A"][1][0]["gate"]+base["B"][1][0]["gate"]);values=_forward(run.model,run.policy,base["Mixed"][0],overrides={"gate":midpoint_gate});rows["artificial_interpolation"].append({"group":run.group,"architecture":run.architecture,"seed":run.seed,"interpolation":"0.5_gate_A_plus_B","prediction_NLL":float(values[7]),"mean_RT":float(values[5].mean())})
        if model_hash(run.model)!=before: raise AssertionError("T12: intervention mutated the trained parent.")
    return {name:pd.DataFrame(value) for name,value in rows.items()}
