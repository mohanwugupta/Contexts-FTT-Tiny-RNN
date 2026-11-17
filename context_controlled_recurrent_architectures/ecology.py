"""Paired latent-cause worlds; labels remain generator metadata only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentWorld:
    seed: int
    index: int
    shared_sequence: tuple[int,...]
    sequence_A: tuple[int,...]
    sequence_B: tuple[int,...]
    latent_causes: tuple[str,...]
    evidence_regimes: tuple[str,...]
    evidence: np.ndarray
    oracle_p_A: np.ndarray
    shuffled_task_causes: tuple[str,...]
    shuffle_permutation: tuple[int,...]
    world_sha256: str


def _schedule():
    causes=["A"]*10+["B"]*10+list("AB"*5)+["A","B","A","B","A","B"]
    regimes=["clear_A"]*10+["clear_B"]*10+[f"clear_{x}" for x in "AB"*5]+["ambiguous"]*6
    return causes,regimes


def generate_world(seed,index,config):
    rng=np.random.default_rng(int(config.payload["synthetic"]["world_seed_offset"])+seed*1000003+index*101)
    shared=rng.integers(0,4,8).tolist(); shared[0]=0
    a=shared.copy(); b=shared.copy()
    for position in (2,3): a[position]=(a[position]+int(rng.integers(1,4)))%4
    for position in (5,6): b[position]=(b[position]+int(rng.integers(1,4)))%4
    causes,regimes=_schedule(); direction=rng.normal(size=34); direction/=max(np.linalg.norm(direction),1e-12); midpoint=rng.normal(0,.25,34)
    evidence=[]; posterior=[]
    for cause,regime in zip(causes,regimes):
        reliability=.92 if regime.startswith("clear") else .08; sign=1 if cause=="A" else -1
        surface=midpoint+sign*reliability*direction+rng.normal(0,.18,34)
        logit=float(2*surface.dot(direction)/(.18**2)); posterior.append(float(1/(1+np.exp(-np.clip(logit,-30,30)))))
        evidence.append(np.column_stack([np.eye(8,dtype=np.float32)[:7],np.tile(surface.astype(np.float32),(7,1))]))
    permutation=tuple(rng.permutation(36).tolist()); shuffled=tuple(causes[i] for i in permutation)
    payload={"seed":seed,"index":index,"shared":shared,"A":a,"B":b,"causes":causes,"regimes":regimes,"permutation":permutation}
    sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return LatentWorld(seed,index,tuple(shared),tuple(a),tuple(b),tuple(causes),tuple(regimes),np.stack(evidence),np.asarray(posterior,np.float32),shuffled,permutation,sha)


def world_batch(world,condition):
    task_causes=world.latent_causes if condition=="structured" else world.shuffled_task_causes
    sequence_a=np.asarray(world.sequence_A,dtype=np.int64);sequence_b=np.asarray(world.sequence_B,dtype=np.int64);choose_a=np.asarray(task_causes)=="A"
    sequences=torch.as_tensor(np.where(choose_a[:,None],sequence_a[None,:],sequence_b[None,:]),dtype=torch.long)
    tokens=F.one_hot(sequences[:,:-1],4).float().reshape(1,36*7,4);targets=sequences[:,1:].reshape(1,36*7)
    return tokens,torch.as_tensor(world.evidence.reshape(36*7,42)).unsqueeze(0),targets


def manifest_record(world,condition,phase):
    task=world.latent_causes if condition=="structured" else world.shuffled_task_causes
    regimes=np.asarray(world.evidence_regimes); causes=np.asarray(world.latent_causes); tasks=np.asarray(task)
    return {"phase":phase,"world_seed":world.seed,"world_index":world.index,"world_sha256":world.world_sha256,
        "structure_condition":condition,"sequences_in_world":36,"clear_A_count":int((regimes=="clear_A").sum()),"clear_B_count":int((regimes=="clear_B").sum()),
        "alternating_clear_count":10,"ambiguous_count":int((regimes=="ambiguous").sum()),"true_latent_A_count":int((causes=="A").sum()),
        "true_latent_B_count":int((causes=="B").sum()),"task_A_count":int((tasks=="A").sum()),"task_B_count":int((tasks=="B").sum()),
        "ambiguous_true_A_count":int(((regimes=="ambiguous")&(causes=="A")).sum()),"ambiguous_true_B_count":int(((regimes=="ambiguous")&(causes=="B")).sum()),
        "oracle_p_A_mean_clear_A":float(world.oracle_p_A[regimes=="clear_A"].mean()),"oracle_p_A_mean_clear_B":float(world.oracle_p_A[regimes=="clear_B"].mean()),
        "oracle_p_A_mean_ambiguous":float(world.oracle_p_A[regimes=="ambiguous"].mean()),"explicit_latent_label_input":False,"explicit_mixed_class":False,
        "RT_supervision":False,"shuffle_permutation":"-".join(map(str,world.shuffle_permutation))}


def manifest_rows(world,condition,phase):
    return pd.DataFrame([manifest_record(world,condition,phase)])
