"""Static compatibility and exact nonlinear displacement, with explicit roundoff."""
import numpy as np
import pandas as pd
from context_sequence.task.sequence import SEQUENCE
from context_sequence.evaluation.rollout import centered
from .jacobian import jacobian


def ce(logits,targets):
    z=np.asarray(logits,dtype=float);m=z.max(axis=-1,keepdims=True)
    return (m+np.log(np.exp(z-m).sum(axis=-1,keepdims=True))).ravel()-z[np.arange(len(z)),targets]


def natural(model,cue,sequence=SEQUENCE,patch_position=None,target=None):
    seq=tuple(sequence);previous=np.zeros(model.whh.shape[0],dtype=np.float32);hs=[];zs=[]
    for k,token in enumerate(seq[:-1]):
        h,z=model.step(token,cue,previous)
        if k==patch_position:
            h=np.asarray(target[k],dtype=float).copy()
            z=model.output.astype(float)@h+model.output_bias.astype(float)
        hs.append(h);zs.append(z);previous=h
    h=np.stack(hs);z=np.stack(zs)
    p=np.exp(z.astype(float)-z.astype(float).max(axis=1,keepdims=True));p/=p.sum(1,keepdims=True)
    return {'hidden':h,'logits':z,'raw_ce':ce(z,np.asarray(seq[1:])),'probabilities':p}


def references(model,sequence=SEQUENCE):
    a=natural(model,[1,0,0],sequence);b=natural(model,[0,1,0],sequence)
    m=(a['hidden'].astype(float)+b['hidden'].astype(float))/2
    return a,b,m


def static_audit(model,sequence=SEQUENCE):
    a,b,m=references(model,sequence);w=model.output.astype(float);bias=model.output_bias.astype(float)
    za=a['hidden'].astype(float)@w.T+bias;zb=b['hidden'].astype(float)@w.T+bias
    midpoint=m@w.T+bias;averaged=(za+zb)/2;targets=np.asarray(sequence[1:])
    endpoint=(ce(za,targets)+ce(zb,targets))/2
    return pd.DataFrame({'transition':np.arange(1,len(sequence)),'target':targets,
        'midpoint_ce':ce(midpoint,targets),'averaged_logit_ce':ce(averaged,targets),
        'familiar_endpoint_ce':endpoint,'affine_error':np.abs(midpoint-averaged).max(1),
        'recorded_float32_logit_roundoff':np.abs(averaged-(a['logits'].astype(float)+b['logits'].astype(float))/2).max(1),
        'convexity_gap':endpoint-ce(midpoint,targets)})


def decompose(model,cue,sequence=SEQUENCE,*,run=None,refs=None,start=0):
    a,b,m=references(model,sequence) if refs is None else refs
    q=natural(model,cue,sequence) if run is None else run
    zero=np.zeros(model.whh.shape[0]);previous_mid=zero;previous_error=zero
    w=centered(model.output.T).T  # P W_o: center output-class rows for each hidden coordinate.
    midpoint_logits=m@model.output.astype(float).T+model.output_bias.astype(float)
    sep_h=np.linalg.norm(a['hidden'].astype(float)-b['hidden'].astype(float),axis=1)
    sep_z=np.linalg.norm(centered(a['logits'])-centered(b['logits']),axis=1)
    rows=[];product=np.eye(len(zero))
    for k,token in enumerate(sequence[:-1]):
        fm=model.step(token,cue,previous_mid)[0].astype(float)
        reconstructed=model.step(token,cue,previous_mid+previous_error)[0].astype(float)
        error=q['hidden'][k].astype(float)-m[k];closure=fm-m[k]
        propagation=reconstructed-fm;j=jacobian(model,token,cue,previous_mid)
        linear=j@previous_error;remainder=propagation-linear
        # An explicit patch is exogenous at start, and is not labeled a recurrence identity.
        patched=(run is not None and k==start)
        if k>=start:
            product=j@product if k>start else j.copy()
            z=q['logits'][k];p=q['probabilities'][k]
            row={'transition':k+1,'token':token,'target':sequence[k+1],
                 'raw_ce':q['raw_ce'][k],'familiar_A_ce':a['raw_ce'].mean(),'familiar_B_ce':b['raw_ce'].mean(),
                 'familiar_mean_ce':(a['raw_ce'].mean()+b['raw_ce'].mean())/2,
                 'hidden_separation':sep_h[k],'task_separation':sep_z[k],
                 'error_hidden_norm':np.linalg.norm(error),'error_task_norm':np.linalg.norm(w@error),
                 'task_residual':np.linalg.norm(w@error)/(sep_z[k]+1e-12),
                 'closure_hidden_norm':np.linalg.norm(closure),'closure_task_norm':np.linalg.norm(w@closure),
                 'propagated_hidden_norm':np.linalg.norm(propagation),'propagated_task_norm':np.linalg.norm(w@propagation),
                 'linear_propagation_hidden_norm':np.linalg.norm(linear),'linear_propagation_task_norm':np.linalg.norm(w@linear),
                 'taylor_remainder_hidden_norm':np.linalg.norm(remainder),'taylor_remainder_task_norm':np.linalg.norm(w@remainder),
                 'identity_error':0. if patched else np.max(np.abs(error-closure-propagation)),
                 'exogenous_patch':patched,'jacobian_spectral_norm':np.linalg.norm(j,2),
                 'finite_horizon_amplification':np.linalg.norm(product,2),'midpoint_ce':ce(midpoint_logits[k:k+1],np.array([sequence[k+1]]))[0],
                 'correct_token_probability':p[sequence[k+1]]}
            for name,value in [('hidden',q['hidden'][k]),('midpoint',m[k]),('error',error),('closure',closure),('propagated',propagation),('linear_propagation',linear),('taylor_remainder',remainder),('logit',z),('probability',p)]:
                row.update({f'{name}_{i}':float(v) for i,v in enumerate(value)})
            rows.append(row)
        previous_mid=m[k];previous_error=error
    return pd.DataFrame(rows)


def single_patch_replay(model,cue,sequence=SEQUENCE):
    refs=references(model,sequence);out=[]
    for k in range(len(sequence)-1):
        run=natural(model,cue,sequence,patch_position=k,target=refs[2])
        f=decompose(model,cue,sequence,run=run,refs=refs,start=k)
        f['position']=k+1;f['lag']=f.transition-k-1;out.append(f)
    return pd.concat(out,ignore_index=True)
