"""Strict source imports; no duplicate scientific SRN class or training fallback."""
from dataclasses import asdict
import numpy as np
import torch
from context_sequence.provenance.registry import load_model
from context_sequence.evaluation.rollout import Model
from experiments.context_cue_access.audit import sources
from confirmatory_srn.model import ConfirmatoryElmanSRN,MonotonicExecutionPolicy
from confirmatory_srn.config import ConfirmatoryConfig
from .config import ROOT,SEQUENCE


def records():return sources()['records']


def cues():return sources()['cues']


def validate_record(r):
    if r['condition']!='High_monotonic' or r['cohort']!='canonical' or r['checkpoint_step']!=170:raise ValueError('Canonical High-at170 required; C0 is not an alias')
    if r['seed'] not in range(20):raise ValueError('Unregistered seed')


def load_original(root,r):
    validate_record(r)
    try:return load_model(root,r)
    except FileNotFoundError as e:raise FileNotFoundError('blocked_missing_source: '+str(e)) from e


def assert_unchanged(model,expected):
    if model.digest()!=expected:raise ValueError('Original model tensor changed')
    return True


def load_torch(root,r):
    load_original(root,r) # verifies identity and bytes before strict state loading
    p=torch.load(ROOT/r['path'] if root is None else root/r['path'],map_location='cpu',weights_only=False)
    cfg=ConfirmatoryConfig(**p['config']);cfg.validate()
    if p['condition']!='High_monotonic' or p['seed']!=r['seed'] or p['init_scale']!=2.:raise ValueError('Original payload identity mismatch')
    if tuple(cfg.fixed_sequence)!=tuple(SEQUENCE) or cfg.lambda_execution!=.2 or cfg.normative_c_time!=.5 or cfg.normative_c_risk!=1 or cfg.normative_gamma!=2:raise ValueError('blocked_training_objective_fidelity: embedded objective differs')
    with torch.random.fork_rng(devices=[]):
        model=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=2.,context_alpha=1.);policy=MonotonicExecutionPolicy()
    model.load_state_dict(p['model_state_dict'],strict=True)
    if p.get('policy_state_dict') is None:raise ValueError('blocked_training_objective_fidelity: missing original policy')
    policy.load_state_dict(p['policy_state_dict'],strict=True);model.eval();policy.eval()
    return model,policy,p


def frozen_from_state(state):
    def array(key):return state[key].detach().cpu().numpy().copy()
    m=Model(array('recurrent.weight_ih_l0'),array('recurrent.weight_hh_l0'),array('recurrent.bias_ih_l0')+array('recurrent.bias_hh_l0'),array('sequence_head.weight'),array('sequence_head.bias'))
    for value in vars(m).values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    return m
