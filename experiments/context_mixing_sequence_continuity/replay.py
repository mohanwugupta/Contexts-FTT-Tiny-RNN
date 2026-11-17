"""Original float32 sequence predictions with explicit preactivation audit."""
import numpy as np
import pandas as pd
from experiments.context_cue_access.core import source_trace
from .config import SEQUENCE


def natural_trace(model,cue):
    f=source_trace(model,np.tile(cue,(7,1)))
    previous=np.r_[np.zeros((1,8)),f[[f'hidden_{i}' for i in range(8)]].to_numpy()[:-1]]
    for t,token in enumerate(SEQUENCE[:-1]):
        inp=np.r_[np.eye(4)[token],np.asarray(cue,float)]
        pre=model.wih.astype(float)@inp+model.whh.astype(float)@previous[t]+model.bias.astype(float)
        for j in range(8):f.loc[t,f'preactivation64_{j}']=pre[j]
    logits=f[[f'logit_{i}' for i in range(4)]].to_numpy();targets=f.target.to_numpy();wrong=logits.copy();wrong[np.arange(7),targets]=-np.inf
    f['correct_target_margin']=logits[np.arange(7),targets]-wrong.max(1)
    return f


def separation(a,b,model):
    delta=a[[f'hidden_{i}' for i in range(8)]].to_numpy()-b[[f'hidden_{i}' for i in range(8)]].to_numpy()
    w=model.output.astype(float);w-=w.mean(0,keepdims=True)
    return np.linalg.norm(delta,axis=1),np.linalg.norm(delta@w.T,axis=1)
