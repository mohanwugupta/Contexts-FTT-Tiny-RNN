"""SD2 reference, stable identity recurrence and nonrecurrent minimum-RMS groups."""
from .common import *
TOL_ABS=1e-7;TOL_REL=2e-7

def groups(m):
 d=dict(m.named_parameters());w=d['recurrent.weight_ih_l0'];return dict(token=w[:,:4],context_A=w[:,4],context_B=w[:,5],context_C=w[:,6],output=d['sequence_head.weight'])
def rms(x):return float(torch.sqrt(torch.mean(x.detach().double()**2)))
def project(m,floors,optimizer=None,enabled=True):
 assert set(floors)==set(groups(m));before=copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None;unmodified={n:p.detach().clone() for n,p in m.named_parameters() if n not in ['recurrent.weight_ih_l0','sequence_head.weight']};rows=[]
 for name,x in groups(m).items():
  s=rms(x);floor=floors[name]
  if not np.isfinite(s) or not np.isfinite(floor) or floor<2:raise ValueError('Nonfinite/invalid scale group '+name)
  if enabled and s==0:raise ValueError('Zero-norm violation has no registered radial direction')
  mult=max(1.,floor/s) if enabled else 1.;old=x.detach().clone()
  with torch.no_grad():x.mul_(mult)
  after=rms(x)
  if enabled and after<floor-TOL_ABS-TOL_REL*floor:raise ValueError('Constraint outside fixed float32 tolerance')
  rows.append(dict(group=name,before_rms=s,after_rms=after,floor=floor,multiplier=mult,projected=mult>1,parameter_change_norm=float(torch.linalg.vector_norm(x.detach().double()-old.double())),enabled=enabled))
 assert all(torch.equal(p,unmodified[n]) for n,p in m.named_parameters() if n in unmodified)
 if optimizer is not None:
  after=optimizer.state_dict();assert before['param_groups']==after['param_groups'] and before['state'].keys()==after['state'].keys()
  for i,st in before['state'].items():
   for k,v in st.items():assert torch.equal(v,after['state'][i][k]) if torch.is_tensor(v) else v==after['state'][i][k]
 return rows

def initialize(seed):
 raw=core.base_pair(seed)['I'];high=copy.deepcopy(raw);low=copy.deepcopy(raw);floors={k:max(2.,rms(v)) for k,v in groups(raw).items()}
 with torch.no_grad():
  for x in groups(low).values():x.mul_(.01)
 project(high,floors);return high,low,floors,raw
