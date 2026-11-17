"""Exact acquisition replay plus time-matched compositional evaluations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.training import state_hash
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from phase1f_shared_backbone.math import prediction_metrics
from phase1h_plasticity_audit.replay import (
    _raw_losses,_rng_hash,_setup,assert_canonical_payload,canonical_objective,frozen_gradients,
)
from srn_training_timeline_audit.core import optimizer_state_hash,schedule_hash,sha256_file

from .config import HIGH,ExperimentConfig
from .cue_encoding import CUES,PROBES,composite,cue_coefficients
from .mechanics import (free_composition_rows,matched_state_rows,natural_rollout,patch_rows,
                        patched_rollout)


def _gradient_cosine(model,policy,tokens,targets,config):
    ga,_,_=frozen_gradients(model,policy,tokens,targets,config,"A")
    gb,_,_=frozen_gradients(model,policy,tokens,targets,config,"B")
    left=torch.cat([x.reshape(-1) for x in ga.values()]);right=torch.cat([x.reshape(-1) for x in gb.values()])
    return float(left@right/(torch.linalg.vector_norm(left)*torch.linalg.vector_norm(right)+config.epsilon))


def _sequence_metric(run):return prediction_metrics(run["logits"],run["targets"])["raw_loss"]


def evaluate_timeline_state(model,sequence,seed,family,t,training_context,parameter_hash,registered):
    alpha_a,alpha_b,_=cue_coefficients();needed=list(CUES) if registered else ["A","B","Mixed","NoContext"]
    rolls={name:natural_rollout(model,sequence,CUES[name]) for name in needed}
    targets=composite(rolls["NoContext"]["hidden"],rolls["A"]["hidden"],rolls["B"]["hidden"],alpha_a,alpha_b)
    rows=[]
    for name in needed:
        rows.append({"family":family,"seed":seed,"training_sequence_index":t,"update_count":t,
                     "training_context":training_context,"checkpoint_hash":parameter_hash,"series_type":"probe_checkpoint" if registered else "timeline",
                     "probe":name,"intervention":"no_patch","raw_sequence_loss":_sequence_metric(rolls[name]),
                     "probe_used_for_construction":name in ("A","B","NoContext")})
    for intervention in ("composite_patch","wrong_position_patch","orthogonal_patch"):
        run=patched_rollout(model,sequence,CUES["Mixed"],targets,rolls["A"]["hidden"],rolls["B"]["hidden"],intervention,seed)
        rows.append({"family":family,"seed":seed,"training_sequence_index":t,"update_count":t,
                     "training_context":training_context,"checkpoint_hash":parameter_hash,"series_type":"timeline",
                     "probe":"Mixed","intervention":intervention,"raw_sequence_loss":_sequence_metric(run),
                     "probe_used_for_construction":False})
    return rows,rolls,targets


def _registered_rows(model,config,seed,family,t,training_context,parameter_hash,rolls):
    alpha_a,alpha_b,_=cue_coefficients()
    matched,full_rolls=matched_state_rows(model,config.sequence,alpha_a,alpha_b,seed,family,t,parameter_hash)
    free,targets=free_composition_rows(full_rolls,alpha_a,alpha_b,seed,family,t,parameter_hash)
    for row in matched+free:row["training_context"]=training_context
    return matched+free,targets


def _registered_provenance(config,family,seed,checkpoint_path):
    registry=pd.read_csv(config.resolve(config.provenance_table))
    row=registry[(registry.condition.eq(family))&(registry.seed.eq(seed))]
    if len(row)!=1:raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
    observed=sha256_file(checkpoint_path);expected=str(row.iloc[0].checkpoint_hash)
    if observed!=expected:raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
    return expected,str(row.iloc[0].training_schedule_hash),str(row.iloc[0].config_hash)


def replay_seed(config:ExperimentConfig,family,seed,progress=None):
    config.validate();model,policy,optimizer,parameters,tokens,targets=_setup(config,family,seed)
    schedule=make_training_schedule(config.n_blocks,config.block_length)
    checkpoint_path=config.resolve(config.checkpoint_dir)/f"{family}_seed_{seed}.pt"
    registered_file_hash,registered_schedule_hash,registered_config_hash=_registered_provenance(config,family,seed,checkpoint_path)
    if schedule_hash(type("C",(),{"n_blocks":17,"block_length":10})())!=registered_schedule_hash:
        raise AssertionError("FAIL: NONCANONICAL_CHECKPOINT")
    timeline=[];transition=[];acquisition=[];functional=[];preactivation=[]
    previous_hash=None
    def audit_state(t,training_context):
        nonlocal previous_hash
        parameter_hash=state_hash(model,policy)
        frozen_before=(parameter_hash,optimizer_state_hash(optimizer),_rng_hash())
        registered=t in config.registered_checkpoints
        rows,rolls,comp=evaluate_timeline_state(model,config.sequence,seed,family,t,training_context,parameter_hash,registered)
        timeline.extend(rows)
        if registered:
            transition.extend(_registered_rows(model,config,seed,family,t,training_context,parameter_hash,rolls)[0])
            for row in transition[-(3*7*6+7*8):]:
                if "preactivation_composition_residual" in row:preactivation.append(row["preactivation_composition_residual"])
        if previous_hash is not None and parameter_hash==previous_hash:
            raise AssertionError("FAIL: REGISTERED_CHECKPOINT_IDENTITY")
        frozen_after=(state_hash(model,policy),optimizer_state_hash(optimizer),_rng_hash())
        if frozen_before!=frozen_after:raise AssertionError("FAIL: FROZEN_EVALUATION_MUTATION")
        previous_hash=parameter_hash
        return rolls,comp,parameter_hash
    audit_state(0,"initial")
    for trial in schedule.itertuples(index=False):
        t=int(trial.global_sequence);context=str(trial.context);other="B" if context=="A" else "A"
        pre_hash=state_hash(model,policy);pre_losses=_raw_losses(model,tokens,targets)
        optimizer.zero_grad();total,ce,_,_=canonical_objective(model,policy,tokens,targets,context,config)
        if abs(float(ce.detach())-pre_losses[context])>1e-7:raise AssertionError("FAIL: RAW_CE_MISMATCH")
        total.backward();torch.nn.utils.clip_grad_norm_(parameters,config.gradient_clip);optimizer.step();model.enforce_novel_neutral();optimizer.zero_grad(set_to_none=True)
        post_hash=state_hash(model,policy);post_losses=_raw_losses(model,tokens,targets)
        if pre_hash==post_hash:raise AssertionError("FAIL: TRUE_ACQUISITION_FROZEN")
        acquisition.append({"family":family,"seed":seed,"training_sequence_index":t,"training_context":context,
                            "checkpoint_hash_pre":pre_hash,"checkpoint_hash":post_hash,"parameters_changed":True,
                            "raw_sequence_loss_pre_update":pre_losses[context],"raw_sequence_loss_post_update":post_losses[context]})
        functional.append({"family":family,"seed":seed,"training_sequence_index":t,"training_context":context,
                           "checkpoint_hash":post_hash,"within_transfer":pre_losses[context]-post_losses[context],
                           "cross_transfer":pre_losses[other]-post_losses[other]})
        _,_,_=audit_state(t,context)
        if progress and t%10==0:progress(f"{family} seed {seed}: replay {t}/{config.training_sequences}")
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False);assert_canonical_payload(payload,family,config)
    exact=all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in payload["model_state_dict"].items()) and all(
        torch.equal(policy.state_dict()[k].cpu(),v.cpu()) for k,v in payload["policy_state_dict"].items())
    if not exact:raise AssertionError("FAIL: CANONICAL_REPLAY_MISMATCH")
    final_hash=state_hash(model,policy);alpha_a,alpha_b,_=cue_coefficients()
    final_rolls={name:natural_rollout(model,config.sequence,cue) for name,cue in CUES.items()}
    final_target=composite(final_rolls["NoContext"]["hidden"],final_rolls["A"]["hidden"],final_rolls["B"]["hidden"],alpha_a,alpha_b)
    transition.extend(patch_rows(model,config.sequence,final_rolls,final_target,seed,family,final_hash))
    gradient={"family":family,"seed":seed,"checkpoint_hash":final_hash,"gradient_cosine_AB":_gradient_cosine(model,policy,tokens,targets,config)}
    provenance={"family":family,"seed":seed,"checkpoint_path":str(checkpoint_path),"checkpoint_hash":registered_file_hash,
                "final_parameter_hash":final_hash,"training_schedule_hash":registered_schedule_hash,"model_config_hash":registered_config_hash,
                "replay_matches_final_checkpoint":True,"preactivation_composition_residual_max":max(preactivation or [0.]),
                "optimizer_state_hash_final":optimizer_state_hash(optimizer)}
    return (pd.DataFrame(timeline),pd.DataFrame(transition),pd.DataFrame(acquisition),pd.DataFrame(functional),gradient,provenance)
