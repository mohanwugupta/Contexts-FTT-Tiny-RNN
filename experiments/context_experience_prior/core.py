"""Source-exact identity RNN, nonzero learnable C and episodic Reptile."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from confirmatory_srn.model import MonotonicExecutionPolicy
from confirmatory_srn.config import ConfirmatoryConfig
from execution_objective_search.objectives import predictive_statistics,execution_loss
from experiments.high_scale_initialization.core import LinearElman
from experiments.recurrent_learning_transfer.learning import state_hash as _source_hash
from types import SimpleNamespace
from experiments.recurrent_learning_transfer.design import training_cue
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/context_experience_prior/v1';SOURCE=ROOT/'results/recurrent_context_factorial/v1';TARGET_PATH=[0,1,2,0,3,1,2,3];TARGET=np.array(TARGET_PATH[1:]);ORG=['I','C'];PRIORS=['P0','P1','P2'];CONDS=[p+'-'+o for o in ORG for p in PRIORS];DIAG=[20,40,80,120,170,330];PROBES=[0,80,100,120,170,330];CUES={'A':[1,0,0],'B':[0,1,0],'Mixed-average':[.5,.5,0],'Mixed-both':[1,1,0],'C':[0,0,1],'NoContext':[0,0,0]}

def hash_state(m,p=None,o=None):
 return _source_hash(m,SimpleNamespace(state_dict=lambda:{}) if p is None else p,o)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def append(s):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+s+'\n')
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/context_experience_prior').glob('*.py')}
def hashes(root,d):
 for n,h in d.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('Changed dependency '+n)
def ordered_frame(frame,columns):
 if set(frame.columns)!=set(columns):raise ValueError('Append schema mismatch')
 return frame.loc[:,columns]

def save_csv(name,rows):
 p=OUT/'tables'/f'{name}.csv';f=pd.DataFrame(rows)
 if p.exists():f=ordered_frame(f,pd.read_csv(p,nrows=0).columns.tolist())
 f.to_csv(p,index=False,mode='a',header=not p.exists())

class ExperienceLinear(nn.Module):
 def __init__(self):
  super().__init__();self.recurrent=nn.RNN(7,8,nonlinearity='tanh',batch_first=True);self.sequence_head=nn.Linear(8,4)
  # Same containers/default RNG consumption/normal redraw as canonical factory.
  # nn.RNN.forward is NEVER called; the source manual identity forward below is used.
  for module in (self.recurrent,self.sequence_head):
   for name,p in module.named_parameters():
    if 'weight' in name:nn.init.normal_(p,mean=0.,std=2.)
    else:nn.init.zeros_(p)
 forward=LinearElman.forward

def base_pair(seed):
 with torch.random.fork_rng():
  torch.manual_seed(int(seed));m=ExperienceLinear()
 with torch.no_grad():m.recurrent.weight_hh_l0.mul_(.9/torch.linalg.matrix_norm(m.recurrent.weight_hh_l0,2))
 c=copy.deepcopy(m)
 with torch.no_grad():d=(c.recurrent.weight_ih_l0[:,4]-c.recurrent.weight_ih_l0[:,5])/2;c.recurrent.weight_ih_l0[:,4].copy_(d);c.recurrent.weight_ih_l0[:,5].copy_(-d)
 return {'I':m,'C':c}

def policy():return MonotonicExecutionPolicy()
def optimizer(m,p):return torch.optim.Adam(list(m.parameters())+list(p.parameters()),lr=.005)
def tensors(path=TARGET_PATH,dtype=torch.float32):return torch.eye(4,dtype=dtype)[torch.tensor(path[:-1])],torch.tensor(path[1:],dtype=torch.long)

def objective(z,p,y):
 ce=nn.functional.cross_entropy(z[0],y);confidence=predictive_statistics(z,y)[0];speed=p(confidence);ex=execution_loss(z,y,speed,ConfirmatoryConfig().execution_candidate)[0].mean();return ce+.2*ex,ce,ex

def step(m,p,o,cue,path=TARGET_PATH,meta=False):
 if cue not in (['A','B','C'] if meta else ['A','B']):raise ValueError('Protected orinvalid training cue')
 if not meta and list(path)!=TARGET_PATH:raise ValueError('Target sequence fixed')
 params=list(m.parameters())+list(p.parameters());assert [id(v) for g in o.param_groups for v in g['params']]==[id(v) for v in params];before=m.recurrent.weight_ih_l0[:,6].detach().clone();o.zero_grad();x,y=tensors(path);z,h,_=m(x,torch.tensor(CUES[cue]));total,ce,ex=objective(z,p,y);r=dict(ce_before=float(ce.detach()),accuracy_before=float((z[0].argmax(1)==y).float().mean()),execution_loss=float(ex.detach()),total_loss=float(total.detach()),gradient_norm=np.nan,hidden_absmax=float(h.detach().abs().max()),optimizer_step_executed=False,status='nonfinite_loss')
 if not torch.isfinite(total) or not torch.isfinite(h).all() or not torch.isfinite(z).all():return r
 total.backward()
 if any(v.grad is None or not torch.isfinite(v.grad).all() for v in params):r['status']='nonfinite_gradient';return r
 if not meta:assert torch.count_nonzero(m.recurrent.weight_ih_l0.grad[:,6])==0
 norm=torch.nn.utils.clip_grad_norm_(params,1.);r['gradient_norm']=float(norm)
 if not torch.isfinite(norm):r['status']='nonfinite_norm';return r
 o.step();r['optimizer_step_executed']=True;r['status']='completed' if all(torch.isfinite(v).all() for v in params) else 'nonfinite_postupdate'
 if not meta:
  assert torch.equal(before,m.recurrent.weight_ih_l0[:,6]);state=o.state[m.recurrent.weight_ih_l0];assert torch.count_nonzero(state['exp_avg'][:,6])==torch.count_nonzero(state['exp_avg_sq'][:,6])==0
 return r

@torch.no_grad()
def reptile(meta,task,epsilon):
 for a,b in zip(meta.parameters(),task.parameters()):a.copy_(a+epsilon*(b-a))

def score(z,y=TARGET):
 z=np.asarray(z,float);mx=z.max(1);q=np.exp(z-mx[:,None]);probs=q/q.sum(1,keepdims=True);ce=mx+np.log(q.sum(1))-z[np.arange(7),y];wrong=z.copy();wrong[np.arange(7),y]=-np.inf;return dict(logits=z,probabilities=probs,ce=ce,accuracy=z.argmax(1)==y,margin=z[np.arange(7),y]-wrong.max(1))

def trace(m,cue,path=TARGET_PATH,dtype=None):
 mm=m if dtype is None else copy.deepcopy(m).to(dtype=dtype);par=next(mm.parameters());x,y=tensors(path,par.dtype)
 with torch.no_grad():z,h,_=mm(x,torch.tensor(CUES[cue],dtype=par.dtype))
 d=score(z[0].numpy().astype(float),np.array(path[1:]));d['hidden']=h[0].numpy().astype(float);return d

def diagnostic(m,p,o):
 before=hash_state(m,p,o);rng=torch.get_rng_state().clone();base={c:trace(m,c) for c in ['A','B']};rows=[]
 for cue in ['A','B']:
  cm,cp,co=copy.deepcopy((m,p,o));assert hash_state(cm,cp,co)==before;assert [id(v) for g in co.param_groups for v in g['params']]==[id(v) for v in list(cm.parameters())+list(cp.parameters())];assert not {id(v) for v in cm.parameters()}&{id(v) for v in m.parameters()};r=step(cm,cp,co,cue)
  after={c:trace(cm,c) for c in ['A','B']} if r['status']=='completed' else None
  for c in ['A','B']:
   a=base[c]['ce'].mean();b=after[c]['ce'].mean() if after else np.nan;rows.append(dict(train_context=cue,evaluation_context=c,loss_before=a,loss_after=b,improvement=a-b,optimizer_restored=True,main_start_hash=before,clone_start_hash=before,diagnostic_update_executed=r['optimizer_step_executed'],diagnostic_status=r['status']))
 assert hash_state(m,p,o)==before and torch.equal(rng,torch.get_rng_state())
 for r in rows:r['main_unchanged']=True
 return rows
