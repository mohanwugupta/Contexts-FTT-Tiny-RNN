"""Fixed dose grid and counterbalanced continuation of the sealed generator."""
import copy
import numpy as np
from experiments.context_experience_prior.registry import draw,validate
DOSES=[0,40,80,120,180,240,480,960,1920];NEW_TARGET_DOSES=[40,80,120,180,480,960,1920];MATCH_BINS=[(2.,4.),(1.,2.),(.5,1.),(.1,.5)]
def epsilon(episode):
 if not 241<=episode<=1920:raise ValueError('Continuation must start241 andstop1920')
 return .01

def task_signature(ep):return (ep['pair'],ep['kind'],tuple(sorted(tuple(p) for p in ep['paths'])))

def continuation_block(seed,block,seen):
 if not 1<=block<=7:raise ValueError('Seven continuationblocks only')
 seen=set(seen);rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261120,73191,int(seed),99173,int(block)])));spec=[]
 for pair in ['AB','AC','BC']:
  for kind in ['shared','conflict']:
   for q in range(40):spec.append(dict(pair=pair,kind=kind,order=q%2,mapping=(q//2)%2,pattern=(q//4)%2))
 rng.shuffle(spec);episodes=[];updates=[];rejected=[]
 for i,item in enumerate(spec,1):
  e=block*240+i
  while True:
   p=draw(rng);paths=[p] if item['kind']=='shared' else [p,draw(rng,p[:3],p[3])];sig=task_signature(dict(**item,paths=paths))
   if sig not in seen:seen.add(sig);break
   rejected.append(dict(episode=e,**item,paths=paths,reason='exact_task_library_duplicate_within_seed'))
  first,second=item['pair'][::1 if item['order']==0 else -1];contexts=[first,second,first,second];m=item['mapping'];a=[m,m,1-m,1-m] if item['pattern']==0 else [m,1-m,1-m,m];b=[m,1-m,m,1-m]
  if item['kind']=='shared':a=b=[0]*4
  rows={c:[dict(inner_update=k+1,block=k//5+1,context=contexts[k//5],sequence=seq[k//5]) for k in range(20)] for c,seq in [('P1',a),('P2',b)]};episodes.append(dict(episode=e,**item,paths=paths,first_context=first,second_context=second));updates.append(rows)
 result=dict(seed=int(seed),continuation_block=block,episodes=episodes,updates=updates,rejections=rejected);local=copy.deepcopy(result)
 for i,e in enumerate(local['episodes'],1):e['episode']=i
 validate(local);return result
