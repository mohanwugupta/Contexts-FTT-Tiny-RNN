"""Centered C-minus-shared prediction and context geometry, no norm causality."""
from .common import *
from experiments.nonzero_context_generalization.core import operators

def extra_measures(m,obs,controls):
 d=m if isinstance(m,dict) else m.state_dict();wi=d['recurrent.weight_ih_l0'].detach().double().numpy()[:,4:];s,k=operators(d);c=wi[:,2];a=wi[:,0];b=wi[:,1];g=(a+b)/2;center=lambda x:x-x.mean(-1,keepdims=True);delta=obs['C']['logits']-obs['Mixed-average']['logits'];expected=np.einsum('tcj,j->tc',k,c-g);rows=[]
 cosine=lambda x,y:float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))) if np.linalg.norm(x)*np.linalg.norm(y)>0 else np.nan
 for t in range(7):
  tol=2*(controls[t]['float32_logit_allowance']+controls[t]['double_tolerance']);assert np.all(abs(delta[t]-expected[t])<=tol)
  r=dict(C_minus_avg_centered_norm=float(np.linalg.norm(center(delta[t]))),C_minus_avg_CE=float(obs['C']['ce'][t]-obs['Mixed-average']['ce'][t]),C_minus_avg_margin=float(obs['C']['margin'][t]-obs['Mixed-average']['margin'][t]),C_removed_CE=float(obs['NoContext']['ce'][t]),C_minus_baseline_CE=float(obs['C']['ce'][t]-obs['NoContext']['ce'][t]),C_minus_avg_operator_error=float(abs(delta[t]-expected[t]).max()),C_minus_avg_allowance=tol)
  for j in range(4):r[f'C_minus_avg_raw_{j}']=float(delta[t,j]);r[f'C_minus_avg_centered_{j}']=float(center(delta[t])[j]);r[f'C_minus_avg_operator_{j}']=float(expected[t,j])
  rows.append(r)
 extra={name:float(np.mean([r[name] for r in rows])) for name in ['C_minus_avg_centered_norm','C_minus_avg_CE','C_minus_avg_margin','C_minus_baseline_CE']};extra.update(C_A_cosine=cosine(c,a),C_B_cosine=cosine(c,b),A_B_cosine=cosine(a,b),C_common_cosine=cosine(c,g),C_common_weight_distance=float(np.linalg.norm(c-g)))
 for j in range(8):extra[f'C_unit_{j}']=float(c[j]/np.linalg.norm(c)) if np.linalg.norm(c)>0 else np.nan
 return extra,rows
