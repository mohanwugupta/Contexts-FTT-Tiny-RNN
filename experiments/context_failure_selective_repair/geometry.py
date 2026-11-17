"""R2: familiar-only state-to-centered-logit derivatives. No labels/objective/search."""
import numpy as np
import pandas as pd
import torch
from experiments.output_compatible_trajectories.state_geometry import output_spaces,assert_equivalent
from experiments.output_compatible_trajectories.nullspace import source_head_audit
from .core import *


def require_references(names):
 if list(names)!=['A','B']:raise ValueError('Both fixed reference sides required; no best-reference selection')
 return True

def _validate(k,g):
 if k not in K or g not in ['A','B']:raise ValueError('Familiar-only known-input Jacobian required')

def future_jacobian(model,state,k,familiar_cue):
 _validate(k,familiar_cue)
 wh=torch.tensor(model.whh.copy(),dtype=torch.float64);wi=torch.tensor(model.wih.copy(),dtype=torch.float64);b=torch.tensor(model.bias.copy(),dtype=torch.float64);o=torch.tensor(model.output.copy(),dtype=torch.float64);ob=torch.tensor(model.output_bias.copy(),dtype=torch.float64)
 inputs=[torch.tensor(np.r_[np.eye(4)[SEQUENCE[t-1]],CONTRACT['cue'][familiar_cue]],dtype=torch.float64) for t in range(k+1,8)]
 def f(h):
  z=[]
  for x in inputs:
   h=torch.tanh(wi@x+wh@h+b);q=o@h+ob;z.append(q-q.mean())
  return torch.cat(z)
 return torch.autograd.functional.jacobian(f,torch.tensor(np.asarray(state),dtype=torch.float64),create_graph=False).detach().numpy()

def chain_jacobian(model,state,k,familiar_cue):
 _validate(k,familiar_cue);h=np.asarray(state,float).copy();J=np.eye(8);out=[];o=model.output.astype(float);o-=o.mean(0)
 for t in range(k+1,8):
  x=np.r_[np.eye(4)[SEQUENCE[t-1]],CONTRACT['cue'][familiar_cue]];h=np.tanh(model.wih.astype(float)@x+model.whh.astype(float)@h+model.bias.astype(float));J=(1-h*h)[:,None]*model.whh.astype(float)@J;out.append(o@J)
 return np.concatenate(out)

def source_logits(model,state,k,g):
 h=np.asarray(state,float).copy();out=[]
 for t in range(k+1,8):
  h,z=model.step(SEQUENCE[t-1],CONTRACT['cue'][g],h);z=z.astype(float);out.extend(z-z.mean())
 return np.asarray(out)

def source_fd(model,state,k,g,eps):
 _validate(k,g);h=np.asarray(state,float);cols=[];one=0
 for j in range(8):
  e=np.eye(8)[j]*eps
  if h[j]+eps>1:col=(source_logits(model,h,k,g)-source_logits(model,h-e,k,g))/eps;one+=1
  elif h[j]-eps< -1:col=(source_logits(model,h+e,k,g)-source_logits(model,h,k,g))/eps;one+=1
  else:col=(source_logits(model,h+e,k,g)-source_logits(model,h-e,k,g))/(2*eps)
  cols.append(col)
 return np.asarray(cols).T,one

def sensitivity_basis(G,N):
 _,s,vt=np.linalg.svd(G@N,full_matrices=True);threshold=max(CONTRACT['absolute_rank_tolerance'],CONTRACT['relative_rank_tolerance']*(s[0] if len(s) else 0));rank=int((s>threshold).sum())
 identifiable=len(s)>=3 and s[1]>threshold and s[1]-s[2]>threshold
 return dict(U=N@vt[:2].T,W=N@vt[2:].T,singular_values=s,rank=rank,threshold=threshold,status='identifiable' if identifiable else 'not_identifiable')

def run(out):
 nat=read(out,'r0_native_replay');fac=read(out,'r1_branch_rollouts');cells=read(out,'r1_cells');require_references(['A','B']);audit=[];sens=[];gaps=[];basisrows=[];jacrows=[]
 for rec in records():
  m=load_original(ROOT,rec);sp=output_spaces(m);N,V=sp['N'],sp['R'];O=sp['M'];before=m.digest()
  if sp['rank']!=3 or sp['nullity']!=5:raise ValueError('Canonical output rank mismatch')
  identity=max(np.max(abs(V.T@N)),np.max(abs(V@V.T+N@N.T-np.eye(8))),np.max(abs(O@N)))
  if identity>1e-10:raise ValueError('SVD identity failed')
  audit.append(dict(seed=rec['seed'],rank=sp['rank'],nullity=sp['nullity'],projection_max_error=identity,checkpoint_sha256=rec['checkpoint_sha256'],**{f'singular_{j}':v for j,v in enumerate(sp['singular_values'])}))
  with frozen_weight_guard(m):
   for k in K:
    G=[];local=[]
    for g in ['A','B']:
     h=native(nat,rec['seed'],g)[H].iloc[k-1].to_numpy();J=future_jacobian(m,h,k,g);chain=chain_jacobian(m,h,k,g);err=np.max(abs(J-chain))
     if err>1e-10:raise ValueError('Autodiff/chain Jacobian mismatch')
     row=dict(seed=rec['seed'],k=k,familiar_source=g,rows=J.shape[0],columns=8,chain_error=err,labels_used=False,future_cue=g,future_input_tokens='|'.join(map(str,SEQUENCE[k:-1])),local_rank=np.linalg.matrix_rank(J))
     for eps in CONTRACT['fd_scales']:
      fd,one=source_fd(m,h,k,g,eps);delta=np.max(abs(fd-J));row[f'fd_{eps}_max_error']=delta;row[f'fd_{eps}_relative_error']=delta/(np.max(abs(J))+1e-12);row[f'fd_{eps}_one_sided']=one
     sens.append(row);local.append(row);G.append(J)
     for i,j in np.ndindex(J.shape):jacrows.append(dict(seed=rec['seed'],k=k,familiar_source=g,future_lag=i//4+1,output_class=i%4,input_coordinate=j,derivative=J[i,j]))
    G=np.concatenate(G);b=sensitivity_basis(G,N);U,W=b['U'],b['W'];status=b['status']
    for row in local:
     row.update(status=status,future_kernel_rank=b['rank'],rank_threshold=b['threshold'],**{f'singular_{j}':v for j,v in enumerate(b['singular_values'])})
    for name,mat in [('V',V),('N',N),('U',U),('W',W)]:
     for i,j in np.ndindex(mat.shape):basisrows.append(dict(seed=rec['seed'],k=k,space=name,coordinate=i,axis=j,value=mat[i,j],status=status))
    h=select(fac,rec['seed'],k,'Mixed','Mixed')[H].iloc[0].to_numpy()
    for side in ['A','B']:
     for kind in ['factorial','natural_diagnostic']:
      r=select(fac,rec['seed'],k,side,'Mixed')[H].iloc[0].to_numpy() if kind=='factorial' else native(nat,rec['seed'],side)[H].iloc[k-1].to_numpy()
      d=r-h;dv=V@(V.T@d);dn=N@(N.T@d);du=U@(U.T@d);dw=W@(W.T@d)
      if np.max(abs(dv+dn-d))>1e-10:raise ValueError('Gap decomposition failed')
      proof=assert_equivalent(m,h,h+dn,target=SEQUENCE[k],check_box=False);rounding=source_head_audit(m,h,h+dn,SEQUENCE[k])
      q=dict(seed=rec['seed'],k=k,reference=side,reference_kind=kind,geometry_status=status,gap_norm=np.linalg.norm(d),visible_norm=np.linalg.norm(dv),silent_norm=np.linalg.norm(dn),U_norm=np.linalg.norm(du),W_norm=np.linalg.norm(dw),familiar_derivative_gap_norm=np.linalg.norm(G@d),algebra_only_not_feasible_placement=True,checkpoint_sha256=rec['checkpoint_sha256'],**proof,**rounding)
      for name,v in [('gap',d),('visible',dv),('silent',dn),('U',du),('W',dw),('anchor',h),('reference_state',r)]:q.update({f'{name}_{j}':float(x) for j,x in enumerate(v)})
      for name in ['visible','silent','U','W']:q[f'{name}_norm_fraction']=q[f'{name}_norm']/q['gap_norm'] if q['gap_norm']>1e-12 else np.nan
      # Outcome labels join only after all weights-only geometry has been frozen above.
      rr=cells[(cells.seed==rec['seed'])&(cells.k==k)&(cells.source_context==side)&(cells.branch=='Mixed')].iloc[0]
      q.update(reference_future_ce=rr.future_ce if kind=='factorial' else np.nan,reference_viable=rr.viable_continuation if kind=='factorial' else None);gaps.append(q)
  if m.digest()!=before:raise ValueError('Weights changed')
 for name,rows in [('r2_output_kernel_audit',audit),('r2_future_sensitivity',sens),('r2_natural_gap_components',gaps),('r2_bases',basisrows),('r2_jacobians',jacrows)]:save(out,name,pd.DataFrame(rows))
 ids=pd.DataFrame(sens).drop_duplicates(['seed','k']);report(out,'R2_TASK_RELEVANT_STATE_REPORT.md','# R2 — current-output geometry and familiar future sensitivity\n\nAll20 original heads have centered output rank3 and nullity5. V and N are the full-right-SVD row/kernel bases. d=reference−Mixed decomposes into VVᵀd+NNᵀd. The complete current softmax is invariant under NNᵀd because centered output changes vanish; raw logits may shift by a common constant. Algebraic full-gap proof does not assert a feasible tanh-interior placement. Double all-four-probability identity and source float32 roundoff are exported separately.\n\nG contains derivatives of future centered logits under familiar A and B continuations only. Known input tokens and frozen weights enter; no correct labels, Mixed losses or optimization enters basis construction. Exact double autodiff agrees with an independently coded chain rule; source float32 finite differences at both fixed scales and one-sided boundary flags are exported, including nonlinear/saturation discrepancies. Jacobians describe local sensitivity, not sequence information or CE mediation.\n\nTop two right singular vectors of G_fam N define U; W is the other three kernel dimensions. Fixed rank2, absolute1e-8/relative1e-4 singular threshold plus top-two gap test. Identifiable seed/k geometries: '+str(int(ids.status.eq('identifiable').sum()))+'/40. All singular spectra, bases and label-free Jacobian entries are saved; no dimension tuning. Both factorial A/B references and separate natural-full-A/B diagnostics retained. Viability joined after decomposition; no chosen-best reference. Norm fractions are not fractions of CE caused.\n')
 return dict(output_rank=3,nullity=5,identifiable_geometries=int(ids.status.eq('identifiable').sum()),geometries=40,gaps=len(gaps),model_updates=0,labels_used_for_geometry=False)
