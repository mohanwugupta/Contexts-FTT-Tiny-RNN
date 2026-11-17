"""Frozen natural and single-injection rollouts for Phase 1J."""

from __future__ import annotations

import hashlib,json
import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import ConfirmatoryElmanSRN,MonotonicExecutionPolicy
from confirmatory_srn.training import state_hash
from phase1f_shared_backbone.math import prediction_metrics
from phase1h_plasticity_audit.replay import _rng_hash,_setup,assert_canonical_payload,canonical_objective
from phase1i_context_composition.cue_encoding import CUES,PROBES
from phase1i_context_composition.mechanics import centered,derangement,natural_rollout,one_step,orthogonal_matched,token_vectors
from srn_training_timeline_audit.core import sha256_file

from .config import HIGH,ExperimentConfig


CLAMPS=("D0_natural","D1_zero","D2_composite")
SINGLE_INTERVENTIONS=("single_midpoint","wrong_position","orthogonal","A_endpoint","B_endpoint")


def midpoint_targets(a_hidden,b_hidden):
    midpoint=.5*(np.asarray(a_hidden)+np.asarray(b_hidden))
    if not np.allclose(midpoint,.5*(np.asarray(a_hidden)+np.asarray(b_hidden)),atol=0,rtol=0):
        raise AssertionError("FAIL: MIDPOINT_IDENTITY")
    return midpoint


def assert_mixed_cue():
    if not np.allclose(CUES["Mixed"],.5*(CUES["A"]+CUES["B"]),atol=1e-7,rtol=0):
        raise AssertionError("FAIL: MIXED_CUE_NOT_MIDPOINT")
    return True


def cue_for_step(probe,clamp,transition,injection_position):
    if transition<=injection_position:return CUES[probe]
    if clamp=="D0_natural":return CUES[probe]
    if clamp=="D1_zero":return CUES["NoContext"]
    if clamp=="D2_composite":return CUES["Mixed"]
    raise ValueError(clamp)


def _replacement(intervention,k,natural,midpoint,a,b,seed):
    if intervention=="single_midpoint":return midpoint[k]
    if intervention=="wrong_position":return midpoint[derangement()[k]]
    if intervention=="orthogonal":return natural+orthogonal_matched(midpoint[k]-natural,seed,k)
    if intervention=="A_endpoint":return a[k]
    if intervention=="B_endpoint":return b[k]
    raise ValueError(intervention)


def intervention_rollout(model,sequence,probe,midpoint,a,b,intervention,clamp,seed,injection_position=None):
    """Teacher-forced rollout; positions are zero-based internally."""
    previous=np.zeros(model.hidden_dim,dtype=np.float32);hidden=[];logits=[];prestates=[];used_tokens=[];patches=[]
    for k,token in enumerate(token_vectors(sequence)):
        prestates.append(previous.copy());used_tokens.append(token.copy())
        cue=cue_for_step(probe,clamp,k,injection_position if injection_position is not None else -1)
        natural,_,_=one_step(model,token,cue,previous);current=natural
        if intervention=="repeated_midpoint":current=midpoint[k];patches.append(k)
        elif intervention in SINGLE_INTERVENTIONS and k==injection_position:
            current=_replacement(intervention,k,natural,midpoint,a,b,seed);patches.append(k)
        z=current@model.sequence_head.weight.detach().numpy().T+model.sequence_head.bias.detach().numpy()
        hidden.append(current);logits.append(z);previous=current
    result={"hidden":np.stack(hidden),"logits":np.stack(logits),"centered_logits":centered(np.stack(logits)),
            "targets":np.asarray(sequence[1:],int),"prestates":np.stack(prestates),"tokens":np.stack(used_tokens),"patches":tuple(patches)}
    if intervention in SINGLE_INTERVENTIONS:
        if result["patches"]!=(injection_position,):raise AssertionError("FAIL: PATCH_COUNT")
        if injection_position<6 and not np.array_equal(result["prestates"][injection_position+1],result["hidden"][injection_position]):
            raise AssertionError("FAIL: RECURRENT_INHERITANCE")
    return result


def natural_reference_rows(model,policy,config,family,seed,checkpoint_hash,parameter_hash):
    before=(state_hash(model,policy),_rng_hash());rolls={name:natural_rollout(model,config.sequence,cue) for name,cue in CUES.items()}
    midpoint=midpoint_targets(rolls["A"]["hidden"],rolls["B"]["hidden"])
    midpoint_task=centered(midpoint@model.sequence_head.weight.detach().numpy().T+model.sequence_head.bias.detach().numpy())
    rows=[]
    for probe,run in rolls.items():
        metrics=prediction_metrics(run["logits"],run["targets"])
        for k in range(config.transitions):
            hden=np.linalg.norm(rolls["A"]["hidden"][k]-rolls["B"]["hidden"][k]);zden=np.linalg.norm(rolls["A"]["centered_logits"][k]-rolls["B"]["centered_logits"][k])
            rows.append({"family":family,"seed":seed,"probe":probe,"injection_position":0,"transition":k+1,
                         "intervention":"natural_baseline","context_clamp":"D0_natural","raw_ce":float(metrics["transition_losses"][k]),
                         "baseline_raw_ce":float(metrics["transition_losses"][k]),"raw_rescue":0.,
                         "hidden_midpoint_residual":float(np.linalg.norm(run["hidden"][k]-midpoint[k])/(hden+config.epsilon)),
                         "task_midpoint_residual":float(np.linalg.norm(run["centered_logits"][k]-midpoint_task[k])/(zden+config.epsilon)),
                         "distance_to_A_hidden":float(np.linalg.norm(run["hidden"][k]-rolls["A"]["hidden"][k])),
                         "distance_to_B_hidden":float(np.linalg.norm(run["hidden"][k]-rolls["B"]["hidden"][k])),
                         "distance_to_A_task":float(np.linalg.norm(run["centered_logits"][k]-rolls["A"]["centered_logits"][k])),
                         "distance_to_B_task":float(np.linalg.norm(run["centered_logits"][k]-rolls["B"]["centered_logits"][k])),
                         "lag_from_patch":np.nan,"normalized_rescue_persistence":np.nan,"persistence_denominator_stable":False,
                         "parameter_hash":parameter_hash,"checkpoint_hash":checkpoint_hash})
    if before!=(state_hash(model,policy),_rng_hash()):raise AssertionError("FAIL: PARAMETER_MUTATION")
    return pd.DataFrame(rows),rolls,midpoint,midpoint_task


def _rollout_rows(run,baseline,rolls,midpoint,midpoint_task,config,family,seed,probe,position,intervention,clamp,checkpoint_hash,parameter_hash):
    metrics=prediction_metrics(run["logits"],run["targets"]);base=prediction_metrics(baseline["logits"],baseline["targets"])
    immediate=float(base["transition_losses"][position]-metrics["transition_losses"][position]) if position is not None else np.nan
    rows=[]
    for k in range(config.transitions):
        hden=np.linalg.norm(rolls["A"]["hidden"][k]-rolls["B"]["hidden"][k]);zden=np.linalg.norm(rolls["A"]["centered_logits"][k]-rolls["B"]["centered_logits"][k])
        raw_rescue=float(base["transition_losses"][k]-metrics["transition_losses"][k]);lag=(k-position) if position is not None else np.nan
        stable=position is not None and abs(immediate)>=config.rescue_epsilon
        rows.append({"family":family,"seed":seed,"probe":probe,"injection_position":position+1 if position is not None else 0,
                     "transition":k+1,"intervention":intervention,"context_clamp":clamp,"raw_ce":float(metrics["transition_losses"][k]),
                     "baseline_raw_ce":float(base["transition_losses"][k]),"raw_rescue":raw_rescue,
                     "hidden_midpoint_residual":float(np.linalg.norm(run["hidden"][k]-midpoint[k])/(hden+config.epsilon)),
                     "task_midpoint_residual":float(np.linalg.norm(run["centered_logits"][k]-midpoint_task[k])/(zden+config.epsilon)),
                     "distance_to_A_hidden":float(np.linalg.norm(run["hidden"][k]-rolls["A"]["hidden"][k])),
                     "distance_to_B_hidden":float(np.linalg.norm(run["hidden"][k]-rolls["B"]["hidden"][k])),
                     "distance_to_A_task":float(np.linalg.norm(run["centered_logits"][k]-rolls["A"]["centered_logits"][k])),
                     "distance_to_B_task":float(np.linalg.norm(run["centered_logits"][k]-rolls["B"]["centered_logits"][k])),
                     "lag_from_patch":lag,"normalized_rescue_persistence":raw_rescue/immediate if stable else np.nan,
                     "persistence_denominator_stable":stable,"parameter_hash":parameter_hash,"checkpoint_hash":checkpoint_hash})
    return rows


def all_intervention_rows(model,policy,config,family,seed,checkpoint_hash,parameter_hash,rolls,midpoint,midpoint_task):
    before=(state_hash(model,policy),_rng_hash());rows=[]
    for probe in PROBES:
        baseline=rolls[probe]
        repeated=intervention_rollout(model,config.sequence,probe,midpoint,rolls["A"]["hidden"],rolls["B"]["hidden"],"repeated_midpoint","D0_natural",seed)
        rows.extend(_rollout_rows(repeated,baseline,rolls,midpoint,midpoint_task,config,family,seed,probe,None,"repeated_midpoint","D0_natural",checkpoint_hash,parameter_hash))
        for position in range(config.transitions):
            for clamp in CLAMPS:
                run=intervention_rollout(model,config.sequence,probe,midpoint,rolls["A"]["hidden"],rolls["B"]["hidden"],"single_midpoint",clamp,seed,position)
                rows.extend(_rollout_rows(run,baseline,rolls,midpoint,midpoint_task,config,family,seed,probe,position,"single_midpoint",clamp,checkpoint_hash,parameter_hash))
            for intervention in ("wrong_position","orthogonal","A_endpoint","B_endpoint"):
                run=intervention_rollout(model,config.sequence,probe,midpoint,rolls["A"]["hidden"],rolls["B"]["hidden"],intervention,"D0_natural",seed,position)
                rows.extend(_rollout_rows(run,baseline,rolls,midpoint,midpoint_task,config,family,seed,probe,position,intervention,"D0_natural",checkpoint_hash,parameter_hash))
    if before!=(state_hash(model,policy),_rng_hash()):raise AssertionError("FAIL: PARAMETER_MUTATION")
    return pd.DataFrame(rows)


def add_novel_mean(frame):
    novel=frame[frame.probe.isin(("Novel1","Novel2","Novel3","Novel4"))].copy()
    keys=["family","seed","injection_position","transition","intervention","context_clamp","lag_from_patch",
          "persistence_denominator_stable","parameter_hash","checkpoint_hash"]
    numeric=[c for c in frame.columns if c not in keys+["probe"] and pd.api.types.is_numeric_dtype(frame[c])]
    mean=novel.groupby(keys,dropna=False,as_index=False)[numeric].mean();mean["probe"]="NovelMean"
    # Verify every aggregate contains each preregistered cue exactly once.
    counts=novel.groupby(keys,dropna=False).probe.nunique()
    if not counts.eq(4).all():raise AssertionError("FAIL: NOVELMEAN_MEMBERSHIP")
    return pd.concat([frame,mean[frame.columns]],ignore_index=True)


def load_final_model(config,family,seed):
    path=config.resolve(config.checkpoint_dir)/f"{family}_seed_{seed}.pt";payload=torch.load(path,map_location="cpu",weights_only=False)
    assert_canonical_payload(payload,family,config);scale=config.high_scale if family==HIGH else config.low_scale
    model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=scale,context_alpha=1.0);policy=MonotonicExecutionPolicy()
    model.load_state_dict(payload["model_state_dict"]);policy.load_state_dict(payload["policy_state_dict"]);model.eval();policy.eval()
    registry=pd.read_csv(config.resolve(config.provenance_table));row=registry[(registry.condition.eq(family))&(registry.seed.eq(seed))]
    if len(row)!=1 or sha256_file(path)!=str(row.iloc[0].checkpoint_hash):raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
    cue_hash=hashlib.sha256(json.dumps({k:v.tolist() for k,v in CUES.items()},sort_keys=True).encode()).hexdigest()
    sequence_hash=hashlib.sha256(repr(config.sequence).encode()).hexdigest()
    out_hash=hashlib.sha256(b"".join(x.detach().numpy().tobytes() for x in (model.sequence_head.weight,model.sequence_head.bias))).hexdigest()
    provenance={"family":family,"seed":seed,"source_checkpoint_path":str(path),"source_checkpoint_hash":sha256_file(path),
                "architecture":"tanh Elman SRN","hidden_size":8,"activation":"tanh","initialization_scale":scale,
                "cue_vectors":json.dumps({k:v.tolist() for k,v in CUES.items()},sort_keys=True),
                "cue_vectors_hash":cue_hash,"sequence":json.dumps(list(config.sequence)),
                "sequence_hash":sequence_hash,"output_layer_hash":out_hash,
                "parameter_hash":state_hash(model,policy)}
    return model,policy,provenance


def replay_acquisition(config,family,seed):
    model,policy,optimizer,parameters,tokens,targets=_setup(config,family,seed);rows=[]
    from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
    schedule=make_training_schedule(config.n_blocks,config.block_length)
    for trial in schedule.itertuples(index=False):
        before=state_hash(model,policy);context=str(trial.context);optimizer.zero_grad()
        total,ce,_,_=canonical_objective(model,policy,tokens,targets,context,config);total.backward()
        torch.nn.utils.clip_grad_norm_(parameters,config.gradient_clip);optimizer.step();model.enforce_novel_neutral();after=state_hash(model,policy)
        rows.append({"family":family,"seed":seed,"training_sequence_index":int(trial.global_sequence),"training_context":context,
                     "raw_sequence_loss_pre_update":float(ce.detach()),"parameter_hash_pre":before,"parameter_hash":after,
                     "parameters_changed":before!=after})
    path=config.resolve(config.checkpoint_dir)/f"{family}_seed_{seed}.pt";payload=torch.load(path,map_location="cpu",weights_only=False)
    exact=all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in payload["model_state_dict"].items()) and all(torch.equal(policy.state_dict()[k].cpu(),v.cpu()) for k,v in payload["policy_state_dict"].items())
    if not exact or not all(row["parameters_changed"] for row in rows):raise AssertionError("FAIL: ACQUISITION_PROVENANCE")
    return pd.DataFrame(rows)
