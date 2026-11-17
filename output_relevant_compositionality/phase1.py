from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import ConfirmatoryElmanSRN
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS, axis_geometry, factorization
from srn_nonlinearity_compositionality.math import (
    cue_curvature, exact_forward, get_projectors, interpolation_metrics, project, softmax,
)


CONDITIONS={
    "C0_high_baseline":("none",2.0),
    "C1_output_curvature":("output",2.0),
    "C2_whole_curvature":("whole",2.0),
    "C3_null_curvature":("null",2.0),
    "C4_low_baseline":("none",.02),
}


def model_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def separation_penalty(distance,target):
    return torch.relu(torch.as_tensor(target,dtype=distance.dtype,device=distance.device)-distance).square()


def _tokens(sequence):
    return encode_sequence(sequence[:-1],as_tensor=True).unsqueeze(0)


def _context(value,device=None):
    return torch.tensor([[1-value,value,0.]],dtype=torch.float32,device=device)


def differentiable_geometry(model,tokens,centers,delta,kind):
    hidden_by_value={}
    for center in centers:
        for value in (center-delta,center,center+delta):
            key=round(value,8)
            if key not in hidden_by_value:
                _,hidden,_=model(tokens,_context(value,tokens.device));hidden_by_value[key]=hidden
    penalties=[]
    if kind=="null":
        weight=model.sequence_head.weight.detach()
        projector=(torch.linalg.pinv(weight)@weight).detach()
        null=torch.eye(weight.shape[1],device=weight.device)-projector
    for center in centers:
        curvature=(hidden_by_value[round(center+delta,8)]-2*hidden_by_value[round(center,8)]
                   +hidden_by_value[round(center-delta,8)])
        if kind=="output": values=curvature@model.sequence_head.weight.T
        elif kind=="whole": values=curvature
        elif kind=="null": values=curvature@null
        else: raise ValueError(kind)
        penalties.append(values.square().sum())
    return torch.stack(penalties).mean()


def familiar_separation(model,tokens):
    _,a,_=model(tokens,_context(0.,tokens.device));_,b,_=model(tokens,_context(1.,tokens.device))
    return torch.linalg.vector_norm(a-b,dim=-1).mean()


@dataclass
class TrainResult:
    model: ConfirmatoryElmanSRN
    summary: dict
    checkpoint: dict


def train_model(config,condition,seed,regularizer_weight=0.,separation_weight=0.,separation_target=0.,progress=None):
    kind,scale=CONDITIONS[condition]
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=scale)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    tokens=_tokens(config.sequence);targets=torch.tensor(config.sequence[1:],dtype=torch.long)
    schedule=make_training_schedule(config.n_blocks,config.block_length)
    task_values=[];curvature_values=[];separation_values=[];total_values=[]; task_contexts=[]
    initial_hash=model_hash(model)
    for trial in schedule.itertuples(index=False):
        context_name=str(trial.context); task_contexts.append(context_name)
        if context_name not in {"A","B"}: raise AssertionError("T1 failed: non-familiar task-loss context.")
        cue=0. if context_name=="A" else 1.
        optimizer.zero_grad();logits,_,_=model(tokens,_context(cue))
        task=F.cross_entropy(logits.squeeze(0),targets)
        if kind=="none": curvature=torch.zeros((),dtype=task.dtype)
        else: curvature=differentiable_geometry(model,tokens,config.curvature_centers,config.curvature_delta,kind)
        distance=familiar_separation(model,tokens)
        sep=separation_penalty(distance,separation_target) if separation_weight else torch.zeros_like(task)
        total=task+regularizer_weight*curvature+separation_weight*sep
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step();model.enforce_novel_neutral()
        task_values.append(float(task.detach()));curvature_values.append(float(curvature.detach()));
        separation_values.append(float(distance.detach()));total_values.append(float(total.detach()))
    if set(task_contexts)!={"A","B"}: raise AssertionError("T1 failed: task loss did not use exactly familiar A/B.")
    summary={"condition":condition,"seed":seed,"regularizer_type":kind,"init_scale":scale,
             "regularizer_weight":regularizer_weight,"separation_weight":separation_weight,
             "separation_target":separation_target,"optimizer_steps":len(schedule),
             "task_training_contexts":"A;B","mixed_task_loss_steps":0,"novel_task_loss_steps":0,
             "mean_task_loss":np.mean(task_values),"final_task_loss":task_values[-1],
             "mean_curvature_penalty":np.mean(curvature_values),"final_curvature_penalty":curvature_values[-1],
             "final_AB_separation":separation_values[-1],"final_total_loss":total_values[-1],
             "initial_model_sha256":initial_hash,"final_model_sha256":model_hash(model)}
    checkpoint={"condition":condition,"seed":seed,"regularizer_weight":regularizer_weight,
                "separation_weight":separation_weight,"separation_target":separation_target,
                "model_state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    return TrainResult(model,summary,checkpoint)


def evaluate_model(config,condition,seed,model):
    row,null,_,w_out=get_projectors(model)
    runs={value:exact_forward(model,config.sequence,np.array([1-value,value,0.]),audit_manual=False) for value in config.cue_lambdas}
    a,b,m=runs[0.],runs[1.],runs[.5]
    geometry=axis_geometry(a["hidden"],b["hidden"],m["hidden"])
    factors=factorization(a["hidden"],b["hidden"])
    curves=cue_curvature(runs,config.cue_lambdas,row,null,w_out)
    midpoint=next(x for x in curves if np.isclose(x["cue_lambda"],.5))
    interp=interpolation_metrics(m,a,b,.5,row,null)
    common={"condition":condition,"seed":seed}
    metric={**common,
            "AB_hidden_separation":float(np.linalg.norm(b["hidden"]-a["hidden"],axis=1).mean()),
            "AB_output_separation":float(np.linalg.norm(project(b["hidden"]-a["hidden"],row),axis=1).mean()),
            "familiar_loss":float((a["loss"].mean()+b["loss"].mean())/2),"Mixed_loss":float(m["loss"].mean()),
            "Mixed_q":geometry["mean_q"],"Mixed_whole_residual":interp["whole_residual_norm"],
            "Mixed_output_residual":interp["output_residual_norm"],"Mixed_null_residual":interp["null_residual_norm"],
            "Mixed_logit_residual":interp["logit_residual_norm"],
            "K_whole_mid":midpoint["K_whole"],"K_out_mid":midpoint["K_out"],
            "K_null_mid":midpoint["K_null"],"K_logit_mid":midpoint["K_logit"],
            "P_sequence":factors["P_sequence"],"P_context":factors["P_context"],"P_interaction":factors["P_interaction"]}
    sweep=[]
    for value in config.cue_lambdas:
        inter=interpolation_metrics(runs[value],a,b,value,row,null)
        curve=next((x for x in curves if np.isclose(x["cue_lambda"],value)),{})
        sweep.append({**common,"cue_lambda":value,"sequence_loss":inter["sequence_loss"],
                      "sequence_accuracy":inter["sequence_accuracy"],"E_whole":inter["E_whole"],
                      "E_out":inter["E_out"],"E_null":inter["E_null"],"E_logit":inter["E_logit"],
                      "output_residual_norm":inter["output_residual_norm"],"logit_residual_norm":inter["logit_residual_norm"],**curve})
    patches={"baseline_actual":m["hidden"],"full_linear_path":(a["hidden"]+b["hidden"])/2,
             "output_only_linear_path":project((a["hidden"]+b["hidden"])/2,row)+project(m["hidden"],null),
             "null_only_linear_path":project(m["hidden"],row)+project((a["hidden"]+b["hidden"])/2,null)}
    patch_rows=[];baseline=float(m["loss"].mean());targets=m["targets"]
    weight=model.sequence_head.weight.detach().numpy().astype(float);bias=model.sequence_head.bias.detach().numpy().astype(float)
    for name,hidden in patches.items():
        logits=hidden@weight.T+bias;prob=softmax(logits);loss=float(-np.log(np.clip(prob[np.arange(len(targets)),targets],1e-12,1)).mean())
        patch_rows.append({**common,"patch_type":name,"sequence_loss":loss,
                           "delta_loss_vs_actual":0. if name=="baseline_actual" else loss-baseline})
    return metric,sweep,[{**common,**row} for row in curves],patch_rows


def selection_metrics(config,model):
    """Allowed signals only: A/B loss/separation and synthetic cue curvature."""
    row,null,_,w_out=get_projectors(model)
    runs={value:exact_forward(model,config.sequence,np.array([1-value,value,0.]),audit_manual=False) for value in config.cue_lambdas}
    curves=cue_curvature(runs,config.cue_lambdas,row,null,w_out)
    midpoint=next(x for x in curves if np.isclose(x["cue_lambda"],.5))
    a,b=runs[0.],runs[1.]
    return {"familiar_loss":float((a["loss"].mean()+b["loss"].mean())/2),
            "AB_hidden_separation":float(np.linalg.norm(b["hidden"]-a["hidden"],axis=1).mean()),
            "K_whole_mid":midpoint["K_whole"],"K_out_mid":midpoint["K_out"],
            "K_null_mid":midpoint["K_null"],"K_logit_mid":midpoint["K_logit"],
            "selection_used_mixed_loss":False,"selection_used_novel":False}


def baseline_separation_target(config):
    values=[]
    from srn_mixed_novel_diagnostics.core import load_frozen_network
    for seed in config.seeds:
        network=load_frozen_network(Path(config.baseline_checkpoint_dir),"High_monotonic",seed)
        a=exact_forward(network.model,config.sequence,CONTEXT_VECTORS["A"])["hidden"]
        b=exact_forward(network.model,config.sequence,CONTEXT_VECTORS["B"])["hidden"]
        values.append(np.linalg.norm(b-a,axis=1).mean())
    return float(np.median(values)),pd.DataFrame({"seed":config.seeds,"frozen_high_AB_separation":values})
