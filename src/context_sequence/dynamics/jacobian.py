"""Smooth real-valued counterpart of the historical float32 tanh recurrence."""
import numpy as np


def smooth_step(model, token, cue, previous):
    c=np.asarray(cue,dtype=float);p=np.asarray(previous,dtype=float)
    if model.context_u is None:
        drive=model.wih.astype(float)@np.r_[np.eye(4)[token],c]
    else:
        # Same label-blind historical projection, including its fixed epsilon.
        alpha=(c[0]-c[1])/(1+2e-12)
        drive=model.wih.astype(float)@np.eye(4)[token]+model.context_u.astype(float)*alpha
    return np.tanh(drive+model.whh.astype(float)@p+model.bias.astype(float))


def jacobian(model, token, cue, previous):
    h=smooth_step(model,token,cue,previous)
    return (1-h*h)[:,None]*model.whh.astype(float)
