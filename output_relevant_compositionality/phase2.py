from __future__ import annotations

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from confirmatory_srn.model import ConfirmatoryElmanSRN
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_nonlinearity_compositionality.math import get_projectors, project, r2_score_array

from .phase1 import evaluate_model


FAMILIES=("additive","multiplicative")
ECOLOGIES=("structured","shuffled")


def latent_sequences(n_sequences,seed):
    rng=np.random.default_rng(seed); sequences=[]
    while len(sequences)<n_sequences:
        seq=[int(rng.integers(0,4))]
        while len(seq)<8:
            value=int(rng.integers(0,4))
            if value!=seq[-1]: seq.append(value)
        candidate=tuple(seq)
        if candidate not in sequences: sequences.append(candidate)
    return tuple(sequences)


def ecology_contexts(n_contexts):
    angles=np.linspace(0,2*np.pi,n_contexts,endpoint=False)
    return np.stack([np.cos(angles),np.sin(angles),np.zeros_like(angles)],axis=1).astype(np.float32)


def heldout_pairs(n_sequences,n_contexts):
    pairs={(sequence,sequence%n_contexts) for sequence in range(n_sequences)}
    if {s for s,_ in pairs}!=set(range(n_sequences)) or {c for _,c in pairs}!=set(range(n_contexts)):
        raise AssertionError("T10 failed: held-out split does not cover every sequence/context.")
    return pairs


def ecology_spec(config,seed,family,ecology):
    rng=np.random.default_rng(config.random_seed+1009*seed+(0 if family=="additive" else 37))
    sequences=latent_sequences(config.phase2_sequences,config.random_seed)
    contexts=ecology_contexts(config.phase2_contexts)
    additive=rng.normal(size=(4,3)); additive*=config.phase2_transform_magnitude/max(np.linalg.norm(additive,axis=0).mean(),1e-12)
    b=rng.normal(size=(config.phase2_multiplicative_rank,3))
    matrices=rng.normal(size=(config.phase2_multiplicative_rank,4,4))
    matrices*=config.phase2_transform_magnitude/max(np.linalg.norm(matrices,axis=(1,2)).mean(),1e-12)
    permutations=[]
    for sequence in range(config.phase2_sequences):
        permutation=np.random.default_rng(config.random_seed+5003*seed+sequence).permutation(config.phase2_contexts)
        permutations.append(permutation)
    return {"sequences":sequences,"contexts":contexts,"additive":additive.astype(np.float32),
            "b":b.astype(np.float32),"matrices":matrices.astype(np.float32),
            "permutations":np.asarray(permutations),"family":family,"ecology":ecology,
            "heldout":heldout_pairs(config.phase2_sequences,config.phase2_contexts)}


def surface_inputs(spec,sequence_index,context_index):
    transform_index=context_index if spec["ecology"]=="structured" else int(spec["permutations"][sequence_index,context_index])
    cue=spec["contexts"][context_index]
    transform_cue=spec["contexts"][transform_index]
    tokens=np.eye(4,dtype=np.float32)[np.asarray(spec["sequences"][sequence_index][:-1])]
    if spec["family"]=="additive":
        return tokens+(spec["additive"]@transform_cue)[None,:]
    result=tokens.copy()
    coefficients=spec["b"]@transform_cue
    for coefficient,matrix in zip(coefficients,spec["matrices"]): result+=coefficient*(tokens@matrix.T)
    return result


def exposure_table(spec):
    rows=[]
    for sequence_index in range(len(spec["sequences"])):
        for context_index in range(len(spec["contexts"])):
            rows.append({"sequence_index":sequence_index,"context_index":context_index,
                         "heldout":(sequence_index,context_index) in spec["heldout"],
                         "token_sequence":";".join(map(str,spec["sequences"][sequence_index])),
                         "transform_index":context_index if spec["ecology"]=="structured" else int(spec["permutations"][sequence_index,context_index])})
    return pd.DataFrame(rows)


def train_pretraining(config,seed,family,ecology):
    spec=ecology_spec(config,seed,family,ecology); torch.manual_seed(seed)
    model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=config.high_scale)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    train_pairs=[(s,c) for s in range(config.phase2_sequences) for c in range(config.phase2_contexts) if (s,c) not in spec["heldout"]]
    losses=[]
    for _ in range(config.phase2_pretraining_epochs):
        for sequence_index,context_index in train_pairs:
            x=torch.tensor(surface_inputs(spec,sequence_index,context_index)).unsqueeze(0)
            cue=torch.tensor(spec["contexts"][context_index]).unsqueeze(0)
            target=torch.tensor(spec["sequences"][sequence_index][1:],dtype=torch.long)
            optimizer.zero_grad();logits,_,_=model(x,cue);loss=F.cross_entropy(logits.squeeze(0),target)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step()
            losses.append(float(loss.detach()))
    return model,spec,{"seed":seed,"family":family,"ecology":ecology,"optimizer_steps":len(train_pairs)*config.phase2_pretraining_epochs,
                       "training_pairs":len(train_pairs),"heldout_pairs":len(spec["heldout"]),"final_training_loss":np.mean(losses[-len(train_pairs):]),
                       "curvature_regularizer_weight":0.,"separation_regularizer_weight":0.}


def evaluate_pair(model,spec,sequence_index,context_index):
    x=torch.tensor(surface_inputs(spec,sequence_index,context_index)).unsqueeze(0)
    cue=torch.tensor(spec["contexts"][context_index]).unsqueeze(0)
    target=torch.tensor(spec["sequences"][sequence_index][1:],dtype=torch.long)
    with torch.no_grad(): logits,hidden,_=model(x,cue);loss=F.cross_entropy(logits.squeeze(0),target)
    return hidden[0].numpy(),float(loss),float((logits.argmax(-1)[0]==target).float().mean())


def pretraining_curvature(model,spec):
    row,null,_,w_out=get_projectors(model); values=[]; delta=.1
    for sequence_index in range(len(spec["sequences"])):
        for c0,c1 in ((0,1),(2,3),(4,5)):
            endpoint_inputs=[surface_inputs(spec,sequence_index,c) for c in (c0,c1)]
            endpoint_cues=[spec["contexts"][c] for c in (c0,c1)]
            hidden=[]
            for lam in (.5-delta,.5,.5+delta):
                x=(1-lam)*endpoint_inputs[0]+lam*endpoint_inputs[1]
                cue=(1-lam)*endpoint_cues[0]+lam*endpoint_cues[1]
                with torch.no_grad(): _,h,_=model(torch.tensor(x).unsqueeze(0),torch.tensor(cue).unsqueeze(0))
                hidden.append(h[0].numpy())
            kappa=(hidden[2]-2*hidden[1]+hidden[0])/delta**2
            values.append((np.linalg.norm(project(kappa,row)),np.linalg.norm(kappa@w_out.T)))
    return float(np.mean([v[0] for v in values])),float(np.mean([v[1] for v in values]))


def _additive_predictions(observations,mask):
    # observations: S×C×D, NaN/ignored outside mask
    grand=observations[mask].mean(axis=0)
    sequence=np.stack([observations[s,mask[s]].mean(axis=0) for s in range(mask.shape[0])])
    context=np.stack([observations[mask[:,c],c].mean(axis=0) for c in range(mask.shape[1])])
    f0=np.broadcast_to(sequence[:,None,:],observations.shape).copy()
    f1=sequence[:,None,:]+context[None,:,:]-grand
    return f0,f1


def _complete_feature(matrix,mask,rank,iterations=40):
    fill=np.where(mask,matrix,np.nanmean(matrix[mask]))
    for _ in range(iterations):
        u,s,vh=np.linalg.svd(fill,full_matrices=False);r=min(rank,len(s))
        approximation=np.zeros_like(fill) if rank==0 else (u[:,:r]*s[:r])@vh[:r]
        updated=np.where(mask,matrix,approximation)
        if np.max(np.abs(updated-fill))<1e-8: break
        fill=updated
    return approximation


def formal_pretraining_models(hidden_grid,train_mask,row_projector,w_out,random_seed):
    # H grid is S×C×K×H; flatten trajectories for matrix completion.
    shape=hidden_grid.shape; values=hidden_grid.reshape(shape[0],shape[1],-1)
    f0,f1=_additive_predictions(values,train_mask);residual=values-f1
    rng=np.random.default_rng(random_seed);train_indices=np.argwhere(train_mask);rng.shuffle(train_indices)
    validation=train_indices[:max(1,len(train_indices)//5)];cv_mask=train_mask.copy();cv_mask[validation[:,0],validation[:,1]]=False
    scores={}
    for rank in (0,1,2,3):
        prediction=np.zeros_like(residual)
        for feature in range(residual.shape[2]): prediction[:,:,feature]=_complete_feature(residual[:,:,feature],cv_mask,rank)
        scores[rank]=float(np.mean((prediction[validation[:,0],validation[:,1]]-residual[validation[:,0],validation[:,1]])**2))
    selected=min(scores,key=lambda r:(scores[r],r));interaction=np.zeros_like(residual)
    for feature in range(residual.shape[2]): interaction[:,:,feature]=_complete_feature(residual[:,:,feature],train_mask,selected)
    f2=f1+interaction;heldout=~train_mask
    rows=[]
    for name,prediction in (("F0",f0),("F1",f1),("F2",f2)):
        actual_train=values[train_mask];pred_train=prediction[train_mask]
        actual_test=values[heldout];pred_test=prediction[heldout]
        # Project each hidden step before flattening for task-space metrics.
        actual_h=actual_test.reshape(-1,shape[2],shape[3]);pred_h=pred_test.reshape(-1,shape[2],shape[3])
        rows.append({"model":name,"familiar_R2":r2_score_array(actual_train,pred_train),
                     "heldout_pair_R2":r2_score_array(actual_test,pred_test),
                     "heldout_output_R2":r2_score_array(project(actual_h,row_projector),project(pred_h,row_projector)),
                     "heldout_logit_R2":r2_score_array(actual_h@w_out.T,pred_h@w_out.T),
                     "selected_rank":selected if name=="F2" else np.nan,"rank_cv_MSE":scores[selected] if name=="F2" else np.nan,
                     "fit_heldout_rows":0})
    rows.append({"model":"F3","familiar_R2":1.,"heldout_pair_R2":np.nan,"heldout_output_R2":np.nan,
                 "heldout_logit_R2":np.nan,"selected_rank":np.nan,"rank_cv_MSE":np.nan,"fit_heldout_rows":0})
    return rows


def evaluate_pretraining(config,seed,family,ecology,model,spec):
    grid=np.empty((config.phase2_sequences,config.phase2_contexts,7,8));rows=[]
    table=exposure_table(spec)
    for item in table.itertuples(index=False):
        hidden,loss,accuracy=evaluate_pair(model,spec,item.sequence_index,item.context_index)
        grid[item.sequence_index,item.context_index]=hidden
        rows.append({"seed":seed,"family":family,"ecology":ecology,"sequence_index":item.sequence_index,
                     "context_index":item.context_index,"heldout":item.heldout,"loss":loss,"accuracy":accuracy})
    row,_,_,w_out=get_projectors(model);k_out,k_logit=pretraining_curvature(model,spec)
    formal=formal_pretraining_models(grid,~table.pivot(index="sequence_index",columns="context_index",values="heldout").to_numpy(bool),row,w_out,seed)
    summary=pd.DataFrame(rows).groupby("heldout",as_index=False).agg(loss=("loss","mean"),accuracy=("accuracy","mean"))
    held=summary[summary.heldout].iloc[0]
    general={"seed":seed,"family":family,"ecology":ecology,"heldout_loss":held.loss,"heldout_accuracy":held.accuracy,
             "K_out":k_out,"K_logit":k_logit,"pretraining_task_contexts":"synthetic_only"}
    return general,[{"seed":seed,"family":family,"ecology":ecology,**row} for row in formal],rows


def human_finetune(config,seed,source,pretrained=None):
    if pretrained is None:
        torch.manual_seed(seed);model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=config.high_scale)
    else: model=copy.deepcopy(pretrained)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    tokens=encode_sequence(config.sequence[:-1],as_tensor=True).unsqueeze(0);targets=torch.tensor(config.sequence[1:],dtype=torch.long)
    schedule=make_training_schedule(config.n_blocks,config.block_length)
    for trial in schedule.itertuples(index=False):
        cue=torch.tensor([[1.,0.,0.] if trial.context=="A" else [0.,1.,0.]])
        optimizer.zero_grad();logits,_,_=model(tokens,cue);loss=F.cross_entropy(logits.squeeze(0),targets);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step();model.enforce_novel_neutral()
    metrics,sweep,curvature,patch=evaluate_model(config,source,seed,model)
    metrics.update({"source":source,"human_optimizer_steps":len(schedule),"human_task_contexts":"A;B",
                    "mixed_optimizer_steps":0,"novel_optimizer_steps":0,"optimizer_state_reset":True})
    return model,metrics,sweep,curvature,patch
