"""Native task content/order plus independent, sealed augmentation banks."""
import copy,json
from pathlib import Path
import numpy as np
from experiments.context_experience_prior.registry import draw,allowed
ROOT=Path(__file__).resolve().parents[2]
def deliver(c,length=7,onset=False):
 c=np.asarray(c);out=np.tile(c,(length,1))
 if onset:out[:]=0;out[0]=np.sqrt(length)*c
 return out
def haar(rng):
 q,r=np.linalg.qr(rng.normal(size=(3,3)));return q*np.where(np.diag(r)>=0,1.,-1.)
def validate_episode(e):
 if not all(allowed(p) for p in e['paths']):raise ValueError('Target-equivalent or invalid task')
 if len(e['support_P2'])!=20:raise ValueError('Native20 support required')
 for k,r in enumerate(e['support_P2']):
  if r['context']!=[e['first_context'],e['second_context']][(k//5)%2] or r['inner_update']!=k+1:raise ValueError('Wrong native five-update schedule')
 return True
def make(seed):
 meta=copy.deepcopy(json.loads((ROOT/f'results/high_scale_meta_progression/v1/registry/R_s{seed:02}.json').read_text())['records'][:240]);assert len(meta)==240
 for e in meta:validate_episode(e)
 rng=np.random.default_rng(np.random.SeedSequence([20261123,6101,seed]));seen={tuple(p) for e in meta for p in e['paths']};extra=set()
 def new(kind,pair,index):
  def get(prefix=None,branch_not=None):
   while True:
    p=draw(rng,prefix,branch_not)
    if tuple(p) not in seen|extra:extra.add(tuple(p));return p
  p=get();paths=[p] if kind=='shared' else [p,get(p[:3],p[3])];first,second=pair;rows=[dict(inner_update=k+1,block=k//5+1,context=[first,second][(k//5)%2],sequence=0 if kind=='shared' else (k//5)%2) for k in range(20)]
  return dict(episode=index,kind=kind,pair=pair,paths=paths,first_context=first,second_context=second,support_P2=rows)
 specs=[(k,p) for p in ['AB','AC','BC'] for k in ['shared','conflict']]
 result=dict(seed=seed,meta=meta)
 for key,n in [('background',6),('calibration',4),('calibration_validation',4),('width',6)]:result[key]=[new(k,p,i) for i,(k,p) in enumerate(specs[:n],1)]
 aug=np.random.default_rng(np.random.SeedSequence([20261123,6102,seed]));result['meta_Q']=[haar(aug).tolist() for _ in meta];result['background_Q']=[haar(aug).tolist() for _ in range(6)]
 masks=[]
 for e in meta:
  mask=[False]*20
  if e['kind']=='shared':
   for block in range(4):mask[block*5+int(aug.integers(5))]=True
  masks.append(mask)
 result['missing_masks']=masks
 return result
