"""One declared recurrence for all cohorts; no optimizer or training dependency."""
from dataclasses import dataclass
import hashlib
import numpy as np
from context_sequence.task.sequence import SEQUENCE


def centered(z):
    z=np.asarray(z,dtype=np.float64);return z-z.mean(axis=-1,keepdims=True)


def raw_ce(logits):
    z=np.asarray(logits,dtype=np.float64);m=z.max(axis=1)
    return m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(7),np.array(SEQUENCE[1:])]


def midpoint(a,b):return .5*(np.asarray(a)+np.asarray(b))


def cue_projection(cue):
    c=np.asarray(cue,dtype=np.float64);bar=np.array([.5,.5,0.]);d=np.array([.5,-.5,0.])
    alpha=float((c-bar)@d/(d@d+1e-12));residual=c-bar-alpha*d
    return alpha,residual


def orthogonal(delta,seed,position):
    delta=np.asarray(delta,dtype=np.float64);norm=np.linalg.norm(delta)
    if norm<=1e-10:return np.zeros_like(delta)
    v=np.random.default_rng(900001+int(seed)*101+int(position)).normal(size=len(delta))
    v-=delta*(v@delta)/(delta@delta)
    return v*norm/np.linalg.norm(v)


@dataclass
class Model:
    wih: np.ndarray
    whh: np.ndarray
    bias: np.ndarray
    output: np.ndarray
    output_bias: np.ndarray
    context_u: np.ndarray | None = None

    def drive(self,cue):
        c=np.asarray(cue,dtype=np.float32)
        if self.context_u is None:return self.wih[:,4:]@c
        # Float32 execution coordinate matches the stored contrast model.
        d=np.array([.5,-.5,0.],dtype=np.float32);bar=np.array([.5,.5,0.],dtype=np.float32)
        alpha=((c-bar)*d).sum()/((d*d).sum()+np.float32(1e-12))
        return self.context_u*np.float32(alpha)

    def step(self,token,cue,previous):
        tokenvec=np.eye(4,dtype=np.float32)[token];prev=np.asarray(previous,dtype=np.float32)
        if self.context_u is None:
            pre=self.wih@np.concatenate([tokenvec,np.asarray(cue,dtype=np.float32)])+self.whh@prev+self.bias
        else:pre=self.wih@tokenvec+self.drive(cue)+self.whh@prev+self.bias
        h=np.tanh(pre);return h,h@self.output.T+self.output_bias

    def digest(self):
        return hashlib.sha256(b''.join(v.tobytes() for v in [self.wih,self.whh,self.bias,self.output,self.output_bias] + ([] if self.context_u is None else [self.context_u]))).hexdigest()


def rollout(model,cue,*,target=None,patch=None,position=None,seed=0,takeover=None,pulse=False,continuation=None):
    previous=np.zeros(8,dtype=np.float32);hidden=[];logits=[];prestates=[];used=[];patches=[]
    for k,token in enumerate(SEQUENCE[:-1]):
        prestates.append(np.asarray(previous,dtype=np.float32).copy());c=cue
        if takeover is not None and k>=position and (not pulse or k==position):c=takeover
        if continuation is not None and k>position:c=continuation
        h,z=model.step(token,c,previous)
        if patch is not None and (position is None or k==position):
            if target is None:raise ValueError('Patch requires familiar target')
            if patch=='correct':h=target[k]
            elif patch=='wrong':h=target[(k+1)%7]
            elif patch=='orthogonal':h=h+orthogonal(target[k]-h,seed,k)
            else:raise ValueError(patch)
            z=h@model.output.T+model.output_bias;patches.append(k)
        hidden.append(h);logits.append(z);used.append(c);previous=h
    logits=np.stack(logits)
    return {'hidden':np.stack(hidden),'logits':logits,'centered_logits':centered(logits),'raw_ce':raw_ce(logits),
            'prestates':np.stack(prestates),'tokens':np.array(SEQUENCE[:-1]),'cues':np.array(used),'patch_positions':patches}
