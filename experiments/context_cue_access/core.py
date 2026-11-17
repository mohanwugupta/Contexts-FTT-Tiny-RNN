"""Only original-source recurrent steps; no externally placed hidden states."""
import numpy as np
import pandas as pd
import torch
from context_sequence.provenance.registry import load_model
from experiments.output_compatible_trajectories.continuation import row,future_summary,frozen_weight_guard,digest_state
from .config import SEQUENCE


def load_checkpoint(root,record):
    try:return load_model(root,record)
    except FileNotFoundError as e:raise FileNotFoundError('blocked_missing_source: '+str(e)) from e


def source_trace(model,schedule,allow_third_for_test=False):
    c=np.asarray(schedule,float)
    if c.shape!=(7,3) or not np.isfinite(c).all() or (not allow_third_for_test and np.any(c[:,2]!=0)):raise ValueError('Cue schedule must be seven finite 3-vectors with third coordinate zero')
    previous=np.zeros(8,np.float32);rows=[]
    for i in range(7):
        pre=digest_state(previous);h,z=model.step(SEQUENCE[i],c[i],previous)
        q=row(h,z,i+1,1,SEQUENCE,c[i]);q['prestate_hash']=pre;rows.append(q);previous=h
    return pd.DataFrame(rows)


def metrics(trace,k,reference):
    if k not in [1,3,5]:raise ValueError('Unknown registered position')
    if trace.transition.tolist()!=list(range(1,8)) or trace.token.tolist()!=list(SEQUENCE[:-1]) or trace.target.tolist()!=list(SEQUENCE[1:]):raise ValueError('Source timeline mismatch')
    f=trace[trace.transition.ge(k)].copy();f['lag']=f.transition-k
    q=future_summary(f,reference)
    q.update(current_ce=float(trace.raw_ce.iloc[k-1]),full_sequence_ce=float(trace.raw_ce.mean()))
    return q


def frozen_constants(model):
    if model.context_u is not None:raise ValueError('Primary cohort requires canonical unrestricted source')
    return {k:torch.tensor(np.array(v,copy=True),dtype=torch.float64) for k,v in [('wx',model.wih[:,:4]),('wc',model.wih[:,4:6]),('wh',model.whh),('bias',model.bias),('wo',model.output),('bo',model.output_bias)]}


def compact(trace,k,refs):
    q=metrics(trace,k,refs)
    q.update({f'ce_{t}':float(trace.raw_ce.iloc[t-1]) for t in range(1,8)})
    q.update({f'current_probability_{i}':float(trace[f'probability_{i}'].iloc[k-1]) for i in range(4)})
    q['prestate_hash']=trace.prestate_hash.iloc[k-1]
    return q
