"""Canonical full-loss updates and isolated model+policy+Adam diagnostic clones."""
import copy,hashlib
import numpy as np
import torch
from confirmatory_srn.model import MonotonicExecutionPolicy
from confirmatory_srn.config import ConfirmatoryConfig
from execution_objective_search.objectives import execution_loss,predictive_statistics
from .design import *
from .network import trace


def policy():return MonotonicExecutionPolicy()

def optimizer(m,p):return torch.optim.Adam(list(m.parameters())+list(p.parameters()),lr=.005)

def loss(z,p):
 y=torch.tensor(TARGET,dtype=torch.long);ce=torch.nn.functional.cross_entropy(z,y);confidence=predictive_statistics(z[None],y)[0];speed=p(confidence);ex=execution_loss(z[None],y,speed,ConfirmatoryConfig().execution_candidate)[0].mean();return ce+.2*ex,ce,ex

def state_hash(m,p,o=None):
 h=hashlib.sha256()
 def walk(x):
  if isinstance(x,torch.Tensor):h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().cpu().numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=lambda k:str(k)):h.update(repr(k).encode());walk(x[k])
  elif isinstance(x,(tuple,list)):
   for v in x:walk(v)
  else:h.update(repr(x).encode())
 walk((m.state_dict(),p.state_dict(),None if o is None else o.state_dict()));return h.hexdigest()

def familiar(m):
 return {c:trace(m,c) for c in ['A','B']}

def step(m,p,o,cue):
 if cue not in ['A','B']:raise ValueError('Protected cue forbidden in learning update')
 params=list(m.parameters())+list(p.parameters())
 if [id(x) for g in o.param_groups for x in g['params']]!=[id(x) for x in params]:raise ValueError('Optimizer parameter references do not match model+policy')
 o.zero_grad();z,_=m(tokens().to(m.wi),torch.tensor(CUES[cue],dtype=m.wi.dtype));total,ce,ex=loss(z,p);r=dict(ce_before=float(ce.detach()),execution_loss=float(ex.detach()),total_loss=float(total.detach()),gradient_norm=np.nan,optimizer_step_executed=False,status='nonfinite_loss')
 if not torch.isfinite(total) or not torch.isfinite(z).all():return r
 total.backward()
 if any(v.grad is None or not torch.isfinite(v.grad).all() for v in params):r['status']='nonfinite_gradient';return r
 norm=torch.nn.utils.clip_grad_norm_(params,1.);r['gradient_norm']=float(norm)
 if not torch.isfinite(norm):r['status']='nonfinite_norm';return r
 o.step();m.enforce_novel_neutral();r['optimizer_step_executed']=True;r['status']='completed' if all(torch.isfinite(v).all() for v in params) else 'nonfinite_postupdate';return r

def validate_clone(m,p,o,cm,cp,co):
 expected=[id(v) for v in list(cm.parameters())+list(cp.parameters())];actual=[id(v) for group in co.param_groups for v in group['params']]
 if expected!=actual or set(expected).intersection(id(v) for v in list(m.parameters())+list(p.parameters())) or state_hash(m,p,o)!=state_hash(cm,cp,co):raise ValueError('Diagnostic clone did not preserve independent model/policy/Adam state')
 return True

def diagnostic(m,p,o):
 before=state_hash(m,p,o);rng=torch.get_rng_state().clone();base=familiar(m);rows=[]
 for train_cue in ['A','B']:
  cm,cp,co=copy.deepcopy((m,p,o));validate_clone(m,p,o,cm,cp,co);starting=state_hash(cm,cp,co);result=step(cm,cp,co,train_cue)
  after=familiar(cm) if result['status']=='completed' else None
  for eval_cue in ['A','B']:
   initial=float(base[eval_cue]['ce'].mean());final=float(after[eval_cue]['ce'].mean()) if after else np.nan
   rows.append(dict(train_context=train_cue,evaluation_context=eval_cue,within_context=train_cue==eval_cue,loss_before=initial,loss_after=final,improvement=initial-final,clone_start_hash=starting,scientific_start_hash=before,optimizer_restored=True,diagnostic_status=result['status'],diagnostic_update_executed=result['optimizer_step_executed']))
 if state_hash(m,p,o)!=before or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Diagnostic changed scientific model/policy/Adam/RNG')
 for r in rows:r['scientific_state_unchanged']=True
 return rows
