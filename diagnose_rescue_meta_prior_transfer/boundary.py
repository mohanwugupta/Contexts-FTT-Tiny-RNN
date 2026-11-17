"""Stage C: conditional boundary-aware ecology and matched shuffled control."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from episodic_meta_learning_capacity.model import model_hash
from human_meta_prior_transfer.analysis import canonicalize_familiar_profiles, familiar_gates, hidden_separation, state_space_tables, transfer_metrics
from human_meta_prior_transfer.checkpoints import load_locked_checkpoint
from human_meta_prior_transfer.config import load_human_meta_transfer_config
from human_meta_prior_transfer.training import train_transfer_group
from human_svg_geometry_transfer.geometry import load_coordinate_registry
from srn_meta_learning_calibration.initialization import initialize_model


CONDITIONS = ("P_boundary_structured_meta", "P_boundary_shuffled_meta")
TASKS = (
    (0,1,2,0,3,1,2,3), (0,1,2,3,0,1,2,3), (0,2,1,0,3,2,1,3), (0,1,3,0,2,1,3,2),
    (3,2,1,3,0,2,1,0), (3,1,2,3,0,1,2,0), (3,2,0,3,1,2,0,1), (3,1,0,3,2,1,0,2),
)


@dataclass(frozen=True)
class BoundaryEpisode:
    seed: int
    index: int
    episode_type: str
    boundary: float
    latent: tuple[float, ...]
    features: tuple[np.ndarray, ...]
    tasks: tuple[tuple[int, ...], ...]
    assigned_tasks: tuple[tuple[int, ...], ...]
    support: tuple[int, int]
    query: int
    copy: int
    task_permutation: tuple[int, ...]


@dataclass
class BoundaryJob:
    condition: str
    seed: int
    model: object
    training: pd.DataFrame
    manifest: pd.DataFrame


def _episode_type(index):
    slot = index % 10
    return "smooth_interpolation" if slot < 4 else ("boundary_discrimination" if slot < 8 else "ambiguous_boundary")


def make_episode(seed, index, registry, condition):
    rng = np.random.default_rng(seed * 1000003 + index * 97 + 51001); boundary = float(rng.uniform(-.2, .2))
    latent = tuple(float(x) for x in np.sort(rng.uniform(-1, 1, 5)))
    # Smooth visible inputs deliberately do not reveal the task discontinuity.
    a = torch.as_tensor(__import__("episodic_meta_learning_capacity.human", fromlist=["human_features"]).human_features(registry, "A")).numpy()
    b = torch.as_tensor(__import__("episodic_meta_learning_capacity.human", fromlist=["human_features"]).human_features(registry, "B")).numpy()
    features = tuple(((1 - (z + 1) / 2) * a + ((z + 1) / 2) * b).astype(np.float32) for z in latent)
    tasks = []
    for z in latent:
        side_offset = 0 if z < boundary else 4
        relative = (z + 1) / max(boundary + 1, 1e-6) if z < boundary else (z - boundary) / max(1 - boundary, 1e-6)
        tasks.append(TASKS[side_offset + min(3, int(np.clip(relative, 0, .999999) * 4))])
    kind = _episode_type(index)
    left = [i for i,z in enumerate(latent) if z < boundary]; right = [i for i,z in enumerate(latent) if z >= boundary]
    if not left or not right:  # extremely rare grid imbalance: enforce a usable latent split
        latent = (-.9, -.45, 0., .45, .9); left = [i for i,z in enumerate(latent) if z < boundary]; right = [i for i,z in enumerate(latent) if z >= boundary]
        features = tuple(((1 - (z + 1) / 2) * a + ((z + 1) / 2) * b).astype(np.float32) for z in latent)
        tasks = [TASKS[(0 if z < boundary else 4) + min(3, int((abs(z-boundary) / max(1+(-boundary if z < boundary else boundary), 1e-6)) * 4))] for z in latent]
    if kind == "smooth_interpolation":
        side = left if len(left) >= 2 else right; support = (side[0], side[-1]); query = side[len(side)//2]
    elif kind == "boundary_discrimination":
        support = (left[-1], right[0]); query = left[0] if index % 2 else right[-1]
    else:
        support = (left[-1], right[0]); query = min(range(5), key=lambda i: abs(latent[i] - boundary))
    copy = support[0]
    permutation = tuple(rng.permutation(5).tolist())
    assigned = tuple(tasks[permutation[i]] for i in range(5)) if condition.endswith("shuffled_meta") else tuple(tasks)
    return BoundaryEpisode(seed, index, kind, boundary, tuple(latent), features, tuple(tasks), assigned, support, query, copy, permutation)


def _batch(episode, indices, repeats=1, *, counterfactual=False):
    tokens, features, targets = [], [], []
    for index in indices:
        task = episode.assigned_tasks[index]
        if counterfactual:
            opposite = [j for j,z in enumerate(episode.latent) if (z < episode.boundary) != (episode.latent[index] < episode.boundary)]
            task = episode.tasks[min(opposite, key=lambda j: abs(episode.latent[j]-episode.latent[index]))]
        sequence = torch.as_tensor(task, dtype=torch.long)
        for _ in range(repeats):
            tokens.append(F.one_hot(sequence[:-1], 4).float()); features.append(torch.as_tensor(episode.features[index])); targets.append(sequence[1:])
    return torch.stack(tokens), torch.stack(features), torch.stack(targets)


def _loss(model, batch, params=None):
    tokens, features, targets = batch; logits, hidden, _ = model(tokens, features, params=params)
    return F.cross_entropy(logits.reshape(-1,4), targets.reshape(-1)), float(logits.argmax(-1).eq(targets).float().mean().detach()), hidden


def _adapt(model, episode, steps, alpha):
    params = OrderedDict(model.named_parameters()); support = _batch(episode, episode.support, repeats=2); values=[]
    for _ in range(steps):
        loss, _, _ = _loss(model, support, params); gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=False)
        params = OrderedDict((name, value-alpha*gradient.detach()) for (name,value),gradient in zip(params.items(), gradients)); values.append(float(loss.detach()))
    return params, values


def train_boundary_job(config, registry, condition, seed, families, phase, progress=None):
    settings=config.payload["stage_c"]; regimes=load_human_meta_transfer_config(config.human_meta_config).payload["locked_initialization_regimes"]
    model,_=initialize_model(128,"I3_orthogonal",regimes,seed); optimizer=torch.optim.Adam(model.parameters(),lr=float(settings["outer_learning_rate"]))
    batch_size=int(settings["meta_batch"]); curves=[]; manifest=[]
    for update,start in enumerate(range(0,families,batch_size),1):
        objectives=[]
        for episode_index in range(start,min(start+batch_size,families)):
            episode=make_episode(seed,episode_index,registry,condition); adapted,inner=_adapt(model,episode,10,.01)
            query,query_acc,_=_loss(model,_batch(episode,(episode.query,)),adapted); copy,copy_acc,_=_loss(model,_batch(episode,(episode.copy,)),adapted)
            objectives.append(query+float(settings["copy_weight"])*copy)
            curves.append({"phase":phase,"training_condition":condition,"seed":seed,"outer_update":update,"episode":episode_index,
                           "episode_type":episode.episode_type,"boundary":episode.boundary,"support_NLL_pre":inner[0],"support_NLL_post":inner[-1],
                           "query_NLL":float(query.detach()),"query_accuracy":query_acc,"copy_NLL":float(copy.detach()),"copy_accuracy":copy_acc})
            for role,indices in (("support",episode.support),("unseen_query",(episode.query,)),("copy_query",(episode.copy,))):
                for visible in indices: manifest.append({"phase":phase,"training_condition":condition,"seed":seed,"episode":episode_index,
                    "episode_type":episode.episode_type,"boundary":episode.boundary,"role":role,"visible_context_index":visible,
                    "latent":episode.latent[visible],"task_sequence":"-".join(map(str,episode.assigned_tasks[visible])),
                    "used_for_inner_adaptation":role=="support","used_as_query":role!="support","support_count":4,"query_count":2,
                    "task_permutation":"-".join(map(str,episode.task_permutation))})
        outer=torch.stack(objectives).mean(); optimizer.zero_grad(); outer.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(settings["outer_gradient_clip"])); optimizer.step()
        if progress and (update==1 or update%50==0 or start+batch_size>=families): progress(f"Stage C {phase} {condition} seed {seed}: {min(start+batch_size,families)}/{families} families")
    return BoundaryJob(condition,seed,model,pd.DataFrame(curves),pd.DataFrame(manifest))


def validate_boundary_jobs(config,registry,jobs,families,phase):
    rows=[]
    for job in jobs:
        for index in range(families):
            episode=make_episode(job.seed,index+900000,registry,job.condition); pre_copy,_,_=_loss(job.model,_batch(episode,(episode.copy,)))
            adapted,_=_adapt(job.model,episode,10,.01); query,qacc,_=_loss(job.model,_batch(episode,(episode.query,)),adapted)
            copy,cacc,_=_loss(job.model,_batch(episode,(episode.copy,)),adapted); cross,_,_=_loss(job.model,_batch(episode,(episode.query,),counterfactual=True),adapted)
            rows.append({"phase":phase,"training_condition":job.condition,"seed":job.seed,"validation_episode":index,
                         "episode_type":episode.episode_type,"within_query_NLL":float(query.detach()),"within_query_accuracy":qacc,
                         "cross_boundary_NLL":float(cross.detach()),"boundary_selectivity":float(cross.detach()-query.detach()),
                         "copy_NLL_pre":float(pre_copy.detach()),"copy_NLL_post":float(copy.detach()),"copy_NLL_delta":float(copy.detach()-pre_copy.detach()),"copy_accuracy":cacc})
    return pd.DataFrame(rows)


def boundary_gate(validation,required_fraction=.8):
    rows=[]
    for phase,one_phase in validation.groupby("phase"):
        seed=one_phase.groupby(["seed","training_condition"],as_index=False).agg(within=("within_query_NLL","mean"),selectivity=("boundary_selectivity","mean"),copy_acc=("copy_accuracy","mean"),copy_delta=("copy_NLL_delta","mean"))
        pivot=seed.pivot(index="seed",columns="training_condition",values="within").dropna(); structured=seed[seed.training_condition.eq(CONDITIONS[0])]
        per_seed=(pivot[CONDITIONS[0]]<pivot[CONDITIONS[1]]) if len(pivot) else pd.Series(dtype=bool); required=int(math.ceil(required_fraction*len(structured)))
        checks={"V1_within_transfer":float(structured.within.mean())<math.log(4),"V2_boundary_selectivity":float(structured.selectivity.mean())>0,
                "V3_ambiguous_finite":bool(np.isfinite(one_phase[one_phase.episode_type.eq("ambiguous_boundary")].within_query_NLL).all()),
                "V4_copy_accuracy":float(structured.copy_acc.mean())>=.5,"V4_copy_NLL_improves":float(structured.copy_delta.mean())<0,
                "V5_structured_beats_shuffled":bool(len(pivot) and pivot[CONDITIONS[0]].mean()<pivot[CONDITIONS[1]].mean()),
                "V5_seed_consistency":int(per_seed.sum())>=required}
        rows.append({"phase":phase,"n_seeds":len(structured),"required_passing_seeds":required,**checks,"expected_direction_seed_count":int(per_seed.sum()),"gate_passed":bool(all(checks.values()))})
    return pd.DataFrame(rows)


def run_boundary_program(config,progress,*,stage_b_pass=False):
    if not stage_b_pass: raise RuntimeError("T9: Stage C may only be entered through a recorded Stage B PASS.")
    transfer_config=load_human_meta_transfer_config(config.human_meta_config,smoke=False)
    registry=load_coordinate_registry(transfer_config.coordinate_registry,{"A":0,"B":1,"Mixed":2,"Novel":3})
    settings=config.payload["stage_c"]; pilot_families=int(config.payload["smoke"]["stage_c_families"] if config.smoke else settings["pilot_families"])
    validation_families=int(config.payload["smoke"]["stage_c_validation_families"] if config.smoke else settings["validation_families"])
    jobs=[]
    for condition in CONDITIONS:
        for seed in config.stage_c_pilot_seeds: jobs.append(train_boundary_job(config,registry,condition,seed,pilot_families,"pilot",progress))
    validation=validate_boundary_jobs(config,registry,jobs,validation_families,"pilot"); gates=boundary_gate(validation,float(settings["gate_seed_fraction"]))
    if not bool(gates.iloc[-1].gate_passed):
        return {"jobs":jobs,"manifest":pd.concat([j.manifest for j in jobs],ignore_index=True),"training":pd.concat([j.training for j in jobs],ignore_index=True),"validation":validation,"gate":gates}
    confirm_jobs=jobs if config.smoke else []
    if not config.smoke:
        for condition in CONDITIONS:
            for seed in config.stage_c_confirmatory_seeds: confirm_jobs.append(train_boundary_job(config,registry,condition,seed,int(settings["confirmatory_families"]),"confirmatory",progress))
        confirm_validation=validate_boundary_jobs(config,registry,confirm_jobs,validation_families,"confirmatory"); validation=pd.concat([validation,confirm_validation],ignore_index=True); gates=boundary_gate(validation,float(settings["gate_seed_fraction"]))
        if not bool(gates[gates.phase.eq("confirmatory")].iloc[0].gate_passed):
            return {"jobs":confirm_jobs,"manifest":pd.concat([j.manifest for j in confirm_jobs],ignore_index=True),"training":pd.concat([j.training for j in confirm_jobs],ignore_index=True),"validation":validation,"gate":gates}
    # Human transfer is reached only after the synthetic gate. All groups use the same locked target engine.
    active={j.condition.replace("P_boundary_","").replace("_meta",""):j for j in confirm_jobs}
    runs=[]
    for seed in config.stage_c_confirmatory_seeds:
        by_condition={j.condition:j for j in confirm_jobs if j.seed==seed}
        specifications=[
            ({"group":"boundary_structured","label":"Boundary structured H128","source":"boundary_structured","hidden_size":128},by_condition[CONDITIONS[0]]),
            ({"group":"boundary_shuffled","label":"Boundary shuffled H128","source":"boundary_shuffled","hidden_size":128},by_condition[CONDITIONS[1]]),
            ({"group":"P2_meta","label":"Original P2 H128","source":"P2_meta","hidden_size":128},None),
            ({"group":"random_I3","label":"Random I3 H128","source":"random_I3","hidden_size":128},None),
            ({"group":"H8_exact","label":"Exact H8 baseline","source":"high_random","hidden_size":8},None)]
        for specification,job in specifications:
            override=None
            if job is not None: override=(deepcopy(job.model.state_dict()),model_hash(job.model),"boundary_meta_checkpoint","not_materialized",job.condition)
            runs.append(train_transfer_group(transfer_config,registry,specification,seed,state_override=override))
    training=pd.concat([r.training for r in runs],ignore_index=True); probes=pd.concat([r.probes for r in runs],ignore_index=True); audits=pd.DataFrame([r.audit for r in runs])
    canonical=canonicalize_familiar_profiles(probes); geometry_metrics,hidden_geometry=transfer_metrics(canonical); separation=hidden_separation(canonical,audits)
    state,predictions,classes=state_space_tables(training); familiar=familiar_gates(transfer_config,training,pd.concat([r.rt_training for r in runs],ignore_index=True),separation,classes,audits,canonical)
    return {"jobs":confirm_jobs,"manifest":pd.concat([j.manifest for j in confirm_jobs],ignore_index=True),"training":pd.concat([j.training for j in confirm_jobs],ignore_index=True),
            "validation":validation,"gate":gates,"human_training":training,"human_familiar":familiar,
            "human_probe_metrics":geometry_metrics,"human_hidden_geometry":hidden_geometry,"human_state_space":state,"human_state_predictions":predictions}
