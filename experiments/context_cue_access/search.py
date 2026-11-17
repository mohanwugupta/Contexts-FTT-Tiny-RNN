"""Fixed-budget future-target-informed cue-input oracle, never hidden placement."""
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiments.output_compatible_trajectories.continuation import frozen_weight_guard
from .core import frozen_constants
from .config import CONTRACT,SEQUENCE


class CueObjective:
    def __init__(self,model,template,bindings,k):
        self.constants=frozen_constants(model);self.template=torch.tensor(np.array(template,copy=True),dtype=torch.float64)
        self.bindings=dict(bindings);self.k=k
        self.tokens=torch.eye(4,dtype=torch.float64)[list(SEQUENCE[:-1])]
        self.targets=torch.tensor(SEQUENCE[k+1:],dtype=torch.long)
        if self.template.requires_grad:raise ValueError('Template must be constant')

    def loss_cues(self,values):
        b=len(values);h=torch.zeros((b,8),dtype=torch.float64);zs=[];c=self.constants
        for t in range(7):
            cue=values[:,self.bindings[t]:self.bindings[t]+2] if t in self.bindings else self.template[t,:2].expand(b,2)
            h=torch.tanh(self.tokens[t]@c['wx'].T+cue@c['wc'].T+h@c['wh'].T+c['bias'])
            if t>=self.k:zs.append(h@c['wo'].T+c['bo'])
        z=torch.stack(zs,1)
        return F.cross_entropy(z.reshape(-1,4),self.targets.repeat(b),reduction='none').reshape(b,7-self.k).mean(1)


def optimize(model,template,bindings,starts,k):
    starts=np.array(starts,float);n,d=starts.shape
    if n!=6 or d not in [2,4]:raise ValueError('Frozen six-start cue budget / dimension mismatch')
    initial_valid=np.isfinite(starts).all(1)&(np.max(np.abs(np.nan_to_num(starts,nan=2)),axis=1)<=1.5)
    ratio=np.nan_to_num(starts/1.5,nan=0.);ratio=np.clip(ratio,-np.nextafter(1.,0.),np.nextafter(1.,0.))
    theta=np.arctanh(ratio);actual=1.5*np.tanh(theta)
    obj=CueObjective(model,template,bindings,k);best=np.full(n,np.inf);selected=np.full((n,d),np.nan);selected_iter=np.full(n,-1)
    active=initial_valid.copy();statuses=np.where(active,'completed','nonfinite_initialization').astype(object)
    momentum=np.zeros_like(theta);variance=np.zeros_like(theta);updates=np.zeros(n,int);evals=np.zeros(n,int);hist=[]
    with frozen_weight_guard(model):
        for iteration in range(81):
            a=torch.tensor(theta,dtype=torch.float64,requires_grad=True);v=1.5*torch.tanh(a);loss=obj.loss_cues(v);losses=loss.detach().numpy();cues=v.detach().numpy()
            evals[active]+=1;bad=active&~np.isfinite(losses);statuses[bad]='nonfinite_objective';active[bad]=False
            improved=active&(losses<best);best[improved]=losses[improved];selected[improved]=cues[improved];selected_iter[improved]=iteration
            gradients=np.full((n,d),np.nan)
            if iteration<80 and active.any():
                good=torch.tensor(np.flatnonzero(active));gradients=torch.autograd.grad(loss[good].sum(),a)[0].detach().numpy()
                bad=active&~np.isfinite(gradients).all(1);statuses[bad]='nonfinite_gradient';active[bad]=False
            for i in range(n):
                hist.append({'restart_index':i,'iteration':iteration,'status':statuses[i],'active':bool(active[i]),'smooth_future_ce':float(losses[i]) if active[i] else np.nan,
                 'gradient_norm':float(np.linalg.norm(gradients[i])) if active[i] and iteration<80 else np.nan,'saturated':bool(np.max(np.abs(cues[i]))>1.49999),
                 **{f'cue_{j}':float(cues[i,j]) if initial_valid[i] else np.nan for j in range(d)}})
            if iteration<80:
                g=np.where(active[:,None],gradients,0.);momentum=.9*momentum+.1*g;variance=.999*variance+.001*g*g
                theta[active]-=.04*(momentum[active]/(1-.9**(iteration+1)))/(np.sqrt(variance[active]/(1-.999**(iteration+1)))+1e-8);updates[active]+=1
                theta[~active]=0
    result=pd.DataFrame({'restart_index':np.arange(n),'status':statuses,'updates_executed':updates,'objective_evaluations':evals,'best_smooth_loss':np.where(np.isfinite(best),best,np.nan),'selected_iteration':selected_iter,'cue_parameters':d,'model_updates':0,'hidden_state_parameters':0,'oracle_future_targets':True})
    for j in range(d):result[f'requested_initial_{j}']=starts[:,j];result[f'actual_initial_{j}']=actual[:,j];result[f'selected_{j}']=selected[:,j]
    return pd.DataFrame(hist),result
