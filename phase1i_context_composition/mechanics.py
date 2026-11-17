"""Matched-state branching, natural rollouts, and deterministic causal patches."""

from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch.nn import functional as F

from phase1f_shared_backbone.math import decompose,prediction_metrics

from .cue_encoding import CUES,PROBES,composite


def centered(values):
    values=np.asarray(values,float);return values-values.mean(axis=-1,keepdims=True)


def token_vectors(sequence):return np.eye(4,dtype=np.float32)[np.asarray(sequence[:-1],int)]


def one_step(model,token,cue,prestate):
    wih=model.recurrent.weight_ih_l0.detach().numpy();whh=model.recurrent.weight_hh_l0.detach().numpy()
    bias=(model.recurrent.bias_ih_l0+model.recurrent.bias_hh_l0).detach().numpy()
    preactivation=wih@np.concatenate([np.asarray(token,np.float32),np.asarray(cue,np.float32)])+whh@np.asarray(prestate,np.float32)+bias
    hidden=np.tanh(preactivation);logits=hidden@model.sequence_head.weight.detach().numpy().T+model.sequence_head.bias.detach().numpy()
    return hidden,preactivation,logits


def natural_rollout(model,sequence,cue,initial=None):
    previous=np.zeros(model.hidden_dim,dtype=np.float32) if initial is None else np.asarray(initial,np.float32).copy()
    hidden=[];preactivation=[];logits=[]
    for token in token_vectors(sequence):
        previous,pre,z=one_step(model,token,cue,previous);hidden.append(previous);preactivation.append(pre);logits.append(z)
    return {"hidden":np.stack(hidden),"preactivation":np.stack(preactivation),"logits":np.stack(logits),
            "centered_logits":centered(np.stack(logits)),"targets":np.asarray(sequence[1:],int)}


def assert_same_prestates(prestates):
    first=prestates[0]
    if not all(np.array_equal(first,value) for value in prestates[1:]):
        raise AssertionError("FAIL: BRANCH_PRESTATES_DIFFER")
    return True


def matched_state_rows(model,sequence,alpha_a,alpha_b,seed,family,checkpoint,checkpoint_hash):
    rolls={name:natural_rollout(model,sequence,cue) for name,cue in CUES.items()};rows=[]
    for source in ("A","B","NoContext"):
        for k,token in enumerate(token_vectors(sequence)):
            prestate=np.zeros(model.hidden_dim,dtype=np.float32) if k==0 else rolls[source]["hidden"][k-1]
            branched={name:one_step(model,token,cue,prestate) for name,cue in CUES.items()}
            assert_same_prestates([prestate.copy() for _ in branched])
            hcomp=composite(branched["NoContext"][0],branched["A"][0],branched["B"][0],alpha_a,alpha_b)
            zcomp=composite(centered(branched["NoContext"][2]),centered(branched["A"][2]),centered(branched["B"][2]),alpha_a,alpha_b)
            hden=np.linalg.norm(branched["A"][0]-branched["B"][0]);zden=np.linalg.norm(centered(branched["A"][2])-centered(branched["B"][2]))
            # Audit the cue-drive term itself in float64.  The shared token,
            # recurrent, and bias terms cancel algebraically but subtracting
            # them after a float32 full preactivation needlessly loses precision.
            wctx=model.recurrent.weight_ih_l0.detach().numpy()[:,4:].astype(np.float64)
            drives={name:wctx@cue.astype(np.float64) for name,cue in CUES.items()}
            pcomp=composite(drives["NoContext"],drives["A"],drives["B"],alpha_a,alpha_b)
            pres_res=float(np.linalg.norm(drives["Mixed"]-pcomp))
            for probe in PROBES:
                hidden_res=float(np.linalg.norm(branched[probe][0]-hcomp));task_res=float(np.linalg.norm(centered(branched[probe][2])-zcomp))
                rows.append({"family":family,"seed":seed,"training_sequence_index":checkpoint,"checkpoint_hash":checkpoint_hash,
                             "probe":probe,"transition":k+1,"prestate_source":source,"intervention":"matched_state_branch",
                             "hidden_composition_residual":hidden_res,"task_composition_residual":task_res,
                             "hidden_composition_residual_normalized":hidden_res/(hden+1e-12),
                             "task_composition_residual_normalized":task_res/(zden+1e-12),
                             "preactivation_composition_residual":pres_res,"raw_transition_loss":prediction_metrics(branched[probe][2][None],[sequence[k+1]])["raw_loss"]})
    return rows,rolls


def free_composition_rows(rolls,alpha_a,alpha_b,seed,family,checkpoint,checkpoint_hash):
    hcomp=composite(rolls["NoContext"]["hidden"],rolls["A"]["hidden"],rolls["B"]["hidden"],alpha_a,alpha_b)
    zcomp=composite(rolls["NoContext"]["centered_logits"],rolls["A"]["centered_logits"],rolls["B"]["centered_logits"],alpha_a,alpha_b)
    rows=[]
    for probe in CUES:
        for k in range(7):
            hden=np.linalg.norm(rolls["A"]["hidden"][k]-rolls["B"]["hidden"][k]);zden=np.linalg.norm(rolls["A"]["centered_logits"][k]-rolls["B"]["centered_logits"][k])
            hd=float(np.linalg.norm(rolls[probe]["hidden"][k]-hcomp[k]));zd=float(np.linalg.norm(rolls[probe]["centered_logits"][k]-zcomp[k]))
            axis_h=decompose(rolls[probe]["hidden"][k],.5*(rolls["A"]["hidden"][k]+rolls["B"]["hidden"][k]),.5*(rolls["A"]["hidden"][k]-rolls["B"]["hidden"][k]))
            axis_z=decompose(rolls[probe]["centered_logits"][k],.5*(rolls["A"]["centered_logits"][k]+rolls["B"]["centered_logits"][k]),.5*(rolls["A"]["centered_logits"][k]-rolls["B"]["centered_logits"][k]))
            metric=prediction_metrics(rolls[probe]["logits"][k:k+1],[rolls[probe]["targets"][k]])
            rows.append({"family":family,"seed":seed,"training_sequence_index":checkpoint,"checkpoint_hash":checkpoint_hash,
                         "probe":probe,"transition":k+1,"prestate_source":"natural","intervention":"free_rollout",
                         "hidden_composition_residual":hd,"task_composition_residual":zd,
                         "hidden_composition_residual_normalized":hd/(hden+1e-12),"task_composition_residual_normalized":zd/(zden+1e-12),
                         "AB_hidden_separation":hden,"AB_task_separation":zden,
                         "raw_transition_loss":metric["raw_loss"],"correct_token_probability":metric["correct_token_probability"],
                         "AB_axis_coordinate_hidden":axis_h["alpha"],"off_axis_hidden":axis_h["off_axis_norm"],
                         "AB_axis_coordinate_task":axis_z["alpha"],"off_axis_task":axis_z["off_axis_norm"]})
    return rows,hcomp


def derangement(count=7):
    values=np.roll(np.arange(count),-1)
    if np.any(values==np.arange(count)):raise AssertionError("FAIL: WRONG_POSITION_NOT_DERANGED")
    return values


def orthogonal_matched(delta,seed,position,epsilon=1e-10):
    delta=np.asarray(delta,float);norm=np.linalg.norm(delta)
    if norm<=epsilon:return np.zeros_like(delta)
    rng=np.random.default_rng(900_001+int(seed)*101+int(position));candidate=rng.normal(size=delta.size)
    candidate-=delta*(candidate@delta)/(delta@delta);candidate_norm=np.linalg.norm(candidate)
    if candidate_norm<=epsilon:raise AssertionError("FAIL: ORTHOGONAL_CONTROL")
    result=candidate*(norm/candidate_norm)
    if abs(result@delta)>epsilon*max(1.,norm**2) or abs(np.linalg.norm(result)-norm)>epsilon*max(1.,norm):
        raise AssertionError("FAIL: ORTHOGONAL_CONTROL")
    # Retain float64 so the exported control satisfies the preregistered
    # orthogonality/norm identities; the recurrent step performs its own
    # explicit float32 conversion when this state is propagated.
    return result


def _target(intervention,k,natural,targets,a,b,midpoint,seed):
    if intervention=="no_patch":return natural
    if intervention in ("composite_patch","single_composite"):return targets[k]
    if intervention in ("wrong_position_patch","single_wrong"):return targets[derangement()[k]]
    if intervention in ("orthogonal_patch","single_orthogonal"):return natural+orthogonal_matched(targets[k]-natural,seed,k)
    if intervention in ("A_endpoint_patch","single_A"):return a[k]
    if intervention in ("B_endpoint_patch","single_B"):return b[k]
    if intervention=="midpoint_patch":return midpoint[k]
    raise ValueError(intervention)


def patched_rollout(model,sequence,cue,targets,a,b,intervention,seed,patch_position=None):
    previous=np.zeros(model.hidden_dim,dtype=np.float32);hidden=[];logits=[];midpoint=.5*(a+b)
    for k,token in enumerate(token_vectors(sequence)):
        natural,_,_=one_step(model,token,cue,previous)
        active=patch_position is None or k==patch_position
        current=_target(intervention,k,natural,targets,a,b,midpoint,seed) if active else natural
        z=current@model.sequence_head.weight.detach().numpy().T+model.sequence_head.bias.detach().numpy()
        hidden.append(current);logits.append(z);previous=current
    return {"hidden":np.stack(hidden),"logits":np.stack(logits),"targets":np.asarray(sequence[1:],int)}


def patch_rows(model,sequence,rolls,composite_targets,seed,family,checkpoint_hash):
    rows=[];interventions=("no_patch","composite_patch","wrong_position_patch","orthogonal_patch","A_endpoint_patch","B_endpoint_patch","midpoint_patch")
    for probe in PROBES:
        for intervention in interventions:
            run=patched_rollout(model,sequence,CUES[probe],composite_targets,rolls["A"]["hidden"],rolls["B"]["hidden"],intervention,seed)
            metrics=prediction_metrics(run["logits"],run["targets"])
            for k in range(7):
                rows.append({"family":family,"seed":seed,"training_sequence_index":170,"checkpoint_hash":checkpoint_hash,
                             "probe":probe,"transition":k+1,"prestate_source":"natural","intervention":intervention,
                             "patch_position":0,"raw_transition_loss":float(metrics["transition_losses"][k]),
                             "raw_sequence_loss":metrics["raw_loss"],"correct_token_probability":float(metrics["correct_probabilities"][k])})
        for patch_position in range(7):
            for intervention in ("single_composite","single_wrong","single_orthogonal","single_A","single_B"):
                run=patched_rollout(model,sequence,CUES[probe],composite_targets,rolls["A"]["hidden"],rolls["B"]["hidden"],intervention,seed,patch_position)
                metrics=prediction_metrics(run["logits"],run["targets"])
                for k in range(7):
                    remaining=metrics["transition_losses"][patch_position:]
                    rows.append({"family":family,"seed":seed,"training_sequence_index":170,"checkpoint_hash":checkpoint_hash,
                                 "probe":probe,"transition":k+1,"prestate_source":"natural","intervention":intervention,
                                 "patch_position":patch_position+1,"raw_transition_loss":float(metrics["transition_losses"][k]),
                                 "raw_sequence_loss":metrics["raw_loss"],"next_token_raw_loss":float(metrics["transition_losses"][patch_position]),
                                 "remaining_sequence_raw_loss":float(np.mean(remaining)),"correct_token_probability":float(metrics["correct_probabilities"][k])})
    return rows
