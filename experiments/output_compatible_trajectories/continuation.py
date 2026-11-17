"""Post-update placement followed by actual frozen-source recurrent steps."""
from contextlib import contextmanager
from unittest.mock import patch
import hashlib
import numpy as np
import pandas as pd
import torch
from context_sequence.dynamics.closure import natural, references
from .config import SEQUENCE,CONTRACT
from .state_geometry import output_values


def digest_state(state):return hashlib.sha256(np.asarray(state,dtype=np.float32).tobytes()).hexdigest()


def row(state,logits,t,k,sequence,cue):
    h=np.asarray(state,float);z=np.asarray(logits,float);p=np.exp(z-z.max());p/=p.sum()
    target=sequence[t];ce=float(z.max()+np.log(np.exp(z-z.max()).sum())-z[target])
    return {'transition':t,'lag':t-k,'token':sequence[t-1],'target':target,'raw_ce':ce,'hidden_norm':float(np.linalg.norm(h)),
        **{f'hidden_{i}':float(x) for i,x in enumerate(h)},**{f'logit_{i}':float(x) for i,x in enumerate(z)},
        **{f'probability_{i}':float(x) for i,x in enumerate(p)},**{f'cue_{i}':float(x) for i,x in enumerate(cue)}}


def natural_trace(model,cue,sequence=SEQUENCE):
    q=natural(model,cue,sequence)
    return pd.DataFrame([row(q['hidden'][i],q['logits'][i],i+1,1,sequence,cue) for i in range(7)])


def validate_timeline(trace,sequence=SEQUENCE):
    if trace.transition.tolist()!=list(range(1,8)) or trace.token.tolist()!=list(sequence[:-1]) or trace.target.tolist()!=list(sequence[1:]):raise ValueError('Scored token/target timeline mismatch')
    return True


def continue_from(model,state,k,cue,sequence=SEQUENCE,*,future_cue=None,repeated_target=None):
    if k not in range(1,8):raise ValueError('Patch index must be a 1-based scored post-update position')
    current=np.asarray(state,float).copy();z=output_values(model,current)['logits'];rows=[row(current,z,k,k,sequence,cue)]
    for t in range(k+1,8):
        c=cue if future_cue is None else future_cue
        current,z=model.step(sequence[t-1],c,current)
        if repeated_target is not None:current=repeated_target[t-1];z=output_values(model,current)['logits']
        rows.append(row(current,z,t,k,sequence,c))
    return pd.DataFrame(rows)


def continue_many(model,states,k,cue,sequence=SEQUENCE):
    """Batch the exports; each future update still calls the exact source step."""
    if k not in range(1,8):raise ValueError('Invalid scored post-state index')
    states=np.asarray(states,float);n=len(states);length=8-k
    hidden=np.empty((n,length,states.shape[1]));logits=np.empty((n,length,4))
    hidden[:,0]=states;logits[:,0]=states@model.output.astype(float).T+model.output_bias.astype(float)
    for j,t in enumerate(range(k+1,8),1):
        for i in range(n):hidden[i,j],logits[i,j]=model.step(sequence[t-1],cue,hidden[i,j-1])
    z=logits.reshape(-1,4);h=hidden.reshape(-1,hidden.shape[-1]);t=np.tile(np.arange(k,8),n)
    maximum=z.max(1);p=np.exp(z-maximum[:,None]);p/=p.sum(1,keepdims=True)
    y=np.asarray(sequence)[t]
    data={'batch_index':np.repeat(np.arange(n),length),'transition':t,'lag':t-k,'token':np.asarray(sequence)[t-1],'target':y,
          'raw_ce':maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(len(z)),y],
          'hidden_norm':np.linalg.norm(h,axis=1)}
    data.update({f'hidden_{i}':h[:,i] for i in range(h.shape[1])})
    data.update({f'logit_{i}':z[:,i] for i in range(4)})
    data.update({f'probability_{i}':p[:,i] for i in range(4)})
    data.update({f'cue_{i}':np.full(len(z),v) for i,v in enumerate(cue)})
    return pd.DataFrame(data)


def future_summary(trace,reference=None):
    f=trace[trace.lag.gt(0)]
    if not len(f):return {'n_future':0,'future_ce':np.nan,'worst_future_ce':np.nan,'zero_future':True}
    q={'n_future':len(f),'future_ce':float(f.raw_ce.mean()),'worst_future_ce':float(f.raw_ce.max()),'zero_future':False}
    if reference is not None:
        a,b=reference;k=int(trace.transition.iloc[0]);end=(a['raw_ce'][k:]+b['raw_ce'][k:])/2
        bound=float(end.mean());worse=float(np.maximum(a['raw_ce'][k:],b['raw_ce'][k:]).mean())
        q.update(familiar_bound=bound,worse_endpoint_bound=worse,viable_continuation=q['future_ce']<=bound+.10,viable_strict=q['future_ce']<=bound,viable_worse_endpoint=q['future_ce']<=worse+.10)
        lost=f.raw_ce.to_numpy()>end+.10;q['first_criterion_loss_lag']=int(np.flatnonzero(lost)[0]+1) if lost.any() else np.nan
    return q


def add_geometry(trace,model,midpoints,separation):
    f=trace.copy();h=f[[f'hidden_{i}' for i in range(8)]].to_numpy();m=midpoints[f.transition.to_numpy()-1]
    delta=h-m;w=model.output.astype(float);w-=w.mean(axis=0,keepdims=True)
    f['midpoint_hidden_distance']=np.linalg.norm(delta,axis=1);f['midpoint_task_distance']=np.linalg.norm(delta@w.T,axis=1)
    f['AB_hidden_separation']=separation[f.transition.to_numpy()-1]
    return f


@contextmanager
def frozen_weight_guard(model=None):
    before=model.digest() if model is not None else None
    def forbidden(*a,**kw):raise RuntimeError('Model optimizer/backward forbidden; P3 uses explicit hidden-coordinate autograd only')
    try:
        with patch.object(torch.optim.Optimizer,'__init__',forbidden),patch.object(torch.Tensor,'backward',forbidden):yield
    finally:
        if model is not None and before!=model.digest():raise RuntimeError('Frozen model tensors changed')
