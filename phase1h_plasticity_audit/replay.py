"""Exact canonical replay, frozen gradients, and counterfactual Adam updates."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.contexts import context_vector
from confirmatory_srn.model import ConfirmatoryElmanSRN, MonotonicExecutionPolicy, normalized_rt
from confirmatory_srn.training import state_hash
from execution_objective_search.config import ObjectiveCandidate
from execution_objective_search.objectives import execution_loss, predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_training_timeline_audit.core import optimizer_state_hash, schedule_hash, sha256_file

from .config import HIGH, LOW, AuditConfig


FAMILIAR_CONTEXTS = frozenset(("A","B"))
BLOCKS = {
    "input_weights": ("recurrent.weight_ih_l0",),
    "recurrent_weights": ("recurrent.weight_hh_l0",),
    "hidden_bias": ("recurrent.bias_ih_l0", "recurrent.bias_hh_l0"),
    "output_weights": ("sequence_head.weight",),
    "output_bias": ("sequence_head.bias",),
}


def _rng_hash():
    digest=hashlib.sha256(); digest.update(torch.random.get_rng_state().numpy().tobytes())
    digest.update(repr(np.random.get_state()).encode()); return digest.hexdigest()


def assert_training_context_labels(labels):
    values=pd.Series(labels)
    if values.isna().any() or not set(values.astype(str)).issubset(FAMILIAR_CONTEXTS):
        raise AssertionError("FAIL: TRAINING_CONTEXT_LABEL")
    return True


def _model_config_hash(config):
    payload=(config.hidden_dim,config.token_dim,config.context_dim,config.high_scale,config.low_scale,
             config.sequence,config.learning_rate,config.gradient_clip,config.lambda_execution)
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def assert_canonical_payload(payload, condition, config):
    try:
        state=payload["model_state_dict"]
        hidden=int(state["recurrent.weight_hh_l0"].shape[0]); inputs=int(state["recurrent.weight_ih_l0"].shape[1])
        outputs=int(state["sequence_head.weight"].shape[0]); scale=float(payload["init_scale"])
        expected=config.high_scale if condition==HIGH else config.low_scale
        valid=(hidden==8 and inputs==7 and outputs==4 and scale==expected and payload.get("condition")==condition)
    except Exception as error:
        raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT") from error
    if not valid: raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
    return True


def _setup(config, condition, seed):
    torch.use_deterministic_algorithms(True); torch.manual_seed(int(seed))
    scale=config.high_scale if condition==HIGH else config.low_scale
    model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=scale,context_alpha=1.0)
    policy=MonotonicExecutionPolicy()
    params=list(model.parameters())+list(policy.parameters())
    optimizer=torch.optim.Adam(params,lr=config.learning_rate)
    tokens=encode_sequence(config.sequence[:-1],as_tensor=True).unsqueeze(0)
    targets=torch.as_tensor(config.sequence[1:],dtype=torch.long)
    return model,policy,optimizer,params,tokens,targets


def _candidate(config):
    return ObjectiveCandidate("A",config.normative_c_time,config.normative_c_risk,
                              config.normative_gamma,config.lambda_execution)


def canonical_objective(model,policy,tokens,targets,context,config):
    vector=context_vector(context,as_tensor=True).unsqueeze(0)
    logits,hidden,_=model(tokens,vector); ce=F.cross_entropy(logits.squeeze(0),targets)
    confidence,_,_=predictive_statistics(logits,targets); speed=policy(confidence)
    execution,_,_=execution_loss(logits,targets,speed,_candidate(config))
    return ce+config.lambda_execution*execution.mean(),ce,logits,hidden


def frozen_state_rows(model,policy,optimizer,tokens,targets,config,condition,seed,t,
                      training_context,checkpoint_hash,metadata):
    before=(state_hash(model,policy),optimizer_state_hash(optimizer),_rng_hash())
    rows=[]
    with torch.no_grad():
        for context in ("A","B"):
            vector=context_vector(context,as_tensor=True).unsqueeze(0)
            logits,hidden,_=model(tokens,vector)
            loss=F.cross_entropy(logits.squeeze(0),targets)
            confidence,_,_=predictive_statistics(logits,targets)
            rt=normalized_rt(policy(confidence))
            centered=logits-logits.mean(-1,keepdim=True)
            for k in range(config.transitions):
                rows.append({
                    "seed":seed,"scale_condition":condition,"training_sequence_index":t,
                    "training_context":training_context,"transition":k+1,"evaluation_context":context,
                    "raw_sequence_loss":float(loss),"raw_transition_loss":float(F.cross_entropy(logits[0,k:k+1],targets[k:k+1])),
                    "correct_token_probability":float(confidence[0,k]),"execution_proxy":float(rt[0,k]),
                    "checkpoint_hash":checkpoint_hash,**metadata,
                    **{f"hidden_state_{j+1}":float(hidden[0,k,j]) for j in range(8)},
                    **{f"raw_logit_{j+1}":float(logits[0,k,j]) for j in range(4)},
                    **{f"centered_logit_{j+1}":float(centered[0,k,j]) for j in range(4)},
                })
    after=(state_hash(model,policy),optimizer_state_hash(optimizer),_rng_hash())
    if before!=after: raise AssertionError("FAIL: FROZEN_EVALUATION_MUTATION")
    return rows


def _flat(values): return torch.cat([value.reshape(-1) for value in values])


def frozen_gradients(model,policy,tokens,targets,config,context):
    parameters=dict(model.named_parameters())
    if any(p.grad is not None for p in parameters.values()): raise AssertionError("FAIL: GRADIENT_ISOLATION")
    objective,ce,_,_=canonical_objective(model,policy,tokens,targets,context,config)
    gradients=torch.autograd.grad(objective,tuple(parameters.values()),retain_graph=False,create_graph=False)
    result={name:grad.detach().clone() for name,grad in zip(parameters,gradients)}
    if any(p.grad is not None for p in parameters.values()): raise AssertionError("FAIL: GRADIENT_ISOLATION")
    return result,float(ce.detach()),float(objective.detach())


def _alignment(a,b,epsilon=1e-12):
    left=_flat(a); right=_flat(b); dot=float(left@right)
    na=float(torch.linalg.vector_norm(left)); nb=float(torch.linalg.vector_norm(right))
    return dot/(na*nb+epsilon),dot,na,nb


def gradient_rows(model,policy,tokens,targets,config,condition,seed,t,training_context,checkpoint_hash,metadata):
    ga,loss_a,total_a=frozen_gradients(model,policy,tokens,targets,config,"A")
    gb,loss_b,total_b=frozen_gradients(model,policy,tokens,targets,config,"B")
    cosine,dot,norm_a,norm_b=_alignment(list(ga.values()),list(gb.values()),config.epsilon)
    overall={"seed":seed,"scale_condition":condition,"training_sequence_index":t,
             "training_context":training_context,"transition":0,"checkpoint_hash":checkpoint_hash,
             "cosine_gradient_alignment":cosine,"dot_product_gradient_alignment":dot,
             "gradient_norm_A":norm_a,"gradient_norm_B":norm_b,"raw_loss_A":loss_a,"raw_loss_B":loss_b,
             "canonical_objective_A":total_a,"canonical_objective_B":total_b,**metadata}
    block=[]
    for label,names in BLOCKS.items():
        cosine,dot,na,nb=_alignment([ga[name] for name in names],[gb[name] for name in names],config.epsilon)
        block.append({"seed":seed,"scale_condition":condition,"training_sequence_index":t,
                      "training_context":training_context,"transition":0,"checkpoint_hash":checkpoint_hash,
                      "parameter_block":label,"cosine_gradient_alignment":cosine,
                      "dot_product_gradient_alignment":dot,"gradient_norm_A":na,"gradient_norm_B":nb,**metadata})
    return overall,block


def _raw_losses(model,tokens,targets):
    values={}
    with torch.no_grad():
        for context in ("A","B"):
            logits,_,_=model(tokens,context_vector(context,as_tensor=True).unsqueeze(0))
            values[context]=float(F.cross_entropy(logits.squeeze(0),targets))
    return values


def _counterfactual_clone(model,policy,optimizer,config):
    clone_model=copy.deepcopy(model); clone_policy=copy.deepcopy(policy)
    params=list(clone_model.parameters())+list(clone_policy.parameters())
    clone_optimizer=torch.optim.Adam(params,lr=config.learning_rate)
    clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return clone_model,clone_policy,clone_optimizer,params


def counterfactual_row(model,policy,optimizer,tokens,targets,config,condition,seed,t,
                       training_context,checkpoint_hash,metadata):
    original=(state_hash(model,policy),optimizer_state_hash(optimizer),_rng_hash())
    before=_raw_losses(model,tokens,targets); post={}
    for update_context in ("A","B"):
        m,p,o,parameters=_counterfactual_clone(model,policy,optimizer,config)
        o.zero_grad(); total,_,_,_=canonical_objective(m,p,tokens,targets,update_context,config)
        total.backward(); torch.nn.utils.clip_grad_norm_(parameters,config.gradient_clip); o.step(); m.enforce_novel_neutral()
        post[update_context]=_raw_losses(m,tokens,targets)
    restored=(state_hash(model,policy),optimizer_state_hash(optimizer),_rng_hash())
    if original!=restored: raise AssertionError("FAIL: COUNTERFACTUAL_RESTORATION")
    return {"seed":seed,"scale_condition":condition,"training_sequence_index":t,
            "training_context":training_context,"transition":0,"checkpoint_hash":checkpoint_hash,
            "analysis_type":"counterfactual_optimizer_step",
            "delta_A_after_A_update":before["A"]-post["A"]["A"],
            "delta_B_after_A_update":before["B"]-post["A"]["B"],
            "delta_A_after_B_update":before["A"]-post["B"]["A"],
            "delta_B_after_B_update":before["B"]-post["B"]["B"],
            "original_parameter_restored":True,"original_optimizer_restored":True,**metadata}


def replay_condition(config:AuditConfig,condition,seed,progress=None):
    config.validate(); model,policy,optimizer,parameters,tokens,targets=_setup(config,condition,seed)
    schedule=make_training_schedule(config.n_blocks,config.block_length)
    assert_training_context_labels(schedule.context)
    sched_hash=schedule_hash(type("C",(),{"n_blocks":config.n_blocks,"block_length":config.block_length})())
    metadata={"training_schedule_hash":sched_hash,"model_config_hash":_model_config_hash(config)}
    states=[]; gradients=[]; blocks=[]; functional=[]
    current_context="initial"; current_hash=state_hash(model,policy)
    states.extend(frozen_state_rows(model,policy,optimizer,tokens,targets,config,condition,seed,0,current_context,current_hash,metadata))
    if 0 in config.diagnostic_checkpoints:
        g,b=gradient_rows(model,policy,tokens,targets,config,condition,seed,0,current_context,current_hash,metadata)
        gradients.append(g);blocks.extend(b);functional.append(counterfactual_row(model,policy,optimizer,tokens,targets,config,condition,seed,0,current_context,current_hash,metadata))
    for trial in schedule.itertuples(index=False):
        t=int(trial.global_sequence); context=str(trial.context)
        pre_hash=state_hash(model,policy); pre_losses=_raw_losses(model,tokens,targets)
        optimizer.zero_grad();total,ce,_,_=canonical_objective(model,policy,tokens,targets,context,config)
        if abs(float(ce.detach())-pre_losses[context])>1e-7: raise AssertionError("FAIL: RAW_LOSS_IDENTITY")
        total.backward();torch.nn.utils.clip_grad_norm_(parameters,config.gradient_clip);optimizer.step();model.enforce_novel_neutral()
        # Gradients are not optimizer state. Clear them before frozen diagnostics so
        # A and B gradients are isolated and cannot inherit the actual-update buffer.
        optimizer.zero_grad(set_to_none=True)
        post_hash=state_hash(model,policy);post_losses=_raw_losses(model,tokens,targets)
        other="B" if context=="A" else "A"
        within=pre_losses[context]-post_losses[context];cross=pre_losses[other]-post_losses[other]
        functional.append({"seed":seed,"scale_condition":condition,"training_sequence_index":t,
                           "training_context":context,"transition":0,"checkpoint_hash":post_hash,
                           "analysis_type":"actual_training_update","within_context_state_change":np.nan,
                           "cross_context_state_change":np.nan,"within_context_loss_change":within,
                           "cross_context_loss_change":cross,"transfer_ratio":cross/(within+config.epsilon),
                           "raw_sequence_loss_pre_update":pre_losses[context],
                           "raw_sequence_loss_post_update":post_losses[context],
                           "parameter_hash_pre":pre_hash,"parameter_hash_post":post_hash,**metadata})
        states.extend(frozen_state_rows(model,policy,optimizer,tokens,targets,config,condition,seed,t,context,post_hash,metadata))
        if t in config.diagnostic_checkpoints:
            g,b=gradient_rows(model,policy,tokens,targets,config,condition,seed,t,context,post_hash,metadata)
            gradients.append(g);blocks.extend(b);functional.append(counterfactual_row(model,policy,optimizer,tokens,targets,config,condition,seed,t,context,post_hash,metadata))
        if progress and t%10==0: progress(f"{condition} seed {seed}: replay {t}/{config.training_sequences}")
    checkpoint_path=config.resolve(config.checkpoint_dir)/f"{condition}_seed_{seed}.pt"
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False);assert_canonical_payload(payload,condition,config)
    expected_model=payload["model_state_dict"];expected_policy=payload["policy_state_dict"]
    exact=all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in expected_model.items()) and all(
        torch.equal(policy.state_dict()[k].cpu(),v.cpu()) for k,v in expected_policy.items())
    if not exact: raise AssertionError("FAIL: CANONICAL_REPLAY_MISMATCH")
    provenance={"seed":seed,"scale_condition":condition,"checkpoint_path":str(checkpoint_path),
                "checkpoint_file_hash":sha256_file(checkpoint_path),"replay_final_hash":state_hash(model,policy),
                "replay_matches_final_checkpoint":True,"hidden_size":8,
                "init_scale":config.high_scale if condition==HIGH else config.low_scale,
                **metadata}
    return pd.DataFrame(states),pd.DataFrame(gradients),pd.DataFrame(blocks),pd.DataFrame(functional),provenance
