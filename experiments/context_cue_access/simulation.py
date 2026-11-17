"""Many schedules, each through the untouched source one-step function."""
import numpy as np
import pandas as pd
from experiments.output_compatible_trajectories.continuation import future_summary,digest_state
from .config import SEQUENCE


def schedules(template,bindings,values):
    values=np.atleast_2d(np.asarray(values,float));result=np.broadcast_to(np.asarray(template,float),(len(values),7,3)).copy()
    for t,offset in bindings:result[:,t,:2]=values[:,offset:offset+2]
    if not np.isfinite(result).all() or np.any(result[:,:,2]!=0) or np.abs(values).max()>1.5+1e-12:raise ValueError('Invalid bounded cue schedule')
    return result


def evaluate(model,template,bindings,values,k,refs):
    cues=schedules(template,bindings,values);n=len(cues);h=np.zeros((n,8),np.float32)
    hidden=np.empty((n,7,8),np.float32);z=np.empty((n,7,4),np.float32);pre=np.empty((n,7),object)
    for t in range(7):
        for i in range(n):
            pre[i,t]=digest_state(h[i]);h[i],z[i,t]=model.step(SEQUENCE[t],cues[i,t],h[i]);hidden[i,t]=h[i]
    logits=z.astype(float);maximum=logits.max(2);p=np.exp(logits-maximum[:,:,None]);p/=p.sum(2,keepdims=True)
    ce=maximum+np.log(np.exp(logits-maximum[:,:,None]).sum(2))-logits[:,np.arange(7),list(SEQUENCE[1:])]
    rows=[]
    for i in range(n):
        f=pd.DataFrame({'transition':np.arange(k,8),'lag':np.arange(0,8-k),'raw_ce':ce[i,k-1:]})
        q=future_summary(f,refs);q.update(candidate_index=i,current_ce=ce[i,k-1],full_sequence_ce=ce[i].mean(),prestate_hash=pre[i,k-1],prep_initial_hash=pre[i,max(0,k-3)])
        q.update({f'ce_{t+1}':ce[i,t] for t in range(7)})
        q.update({f'current_probability_{j}':p[i,k-1,j] for j in range(4)})
        q.update({f'current_centered_logit_{j}':logits[i,k-1,j]-logits[i,k-1].mean() for j in range(4)})
        rows.append(q)
    t=np.tile(np.arange(1,8),n);flat={'candidate_index':np.repeat(np.arange(n),7),'transition':t,'lag':t-k,'token':np.asarray(SEQUENCE)[t-1],'target':np.asarray(SEQUENCE)[t],
      'raw_ce':ce.ravel(),'prestate_hash':pre.ravel()}
    flat.update({f'hidden_{j}':hidden[:,:,j].ravel() for j in range(8)})
    flat.update({f'logit_{j}':logits[:,:,j].ravel() for j in range(4)})
    flat.update({f'probability_{j}':p[:,:,j].ravel() for j in range(4)})
    flat.update({f'cue_{j}':cues[:,:,j].ravel() for j in range(3)})
    return pd.DataFrame(rows),pd.DataFrame(flat)


def source_select(frame):
    f=frame[np.isfinite(frame.future_ce)]
    return f.future_ce.idxmin() if len(f) else None
