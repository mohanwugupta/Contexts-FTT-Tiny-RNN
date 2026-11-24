"""Fixed matched pretraining library and episodic context/sequence schedules."""
import json
from pathlib import Path
import numpy as np
FULL=[0,1,2,0,3,1,2,3]

def orbit(x):
 x=tuple(x);return {z[i:]+z[:i] for z in [x,x[::-1]] for i in range(len(x))}
EX8=orbit(FULL);EX7=orbit(FULL[:-1])|orbit(FULL[1:])
def transition_counts(path):
 a=np.zeros((4,4),int)
 for x,y in zip(path[:-1],path[1:]):a[x,y]+=1
 return a

def allowed(path):
 if len(path)!=8 or set(path)!=set(range(4)):return False
 if tuple(path) in EX8 or tuple(path[:-1]) in EX7 or tuple(path[1:]) in EX7:return False
 return bool(((transition_counts(path)>0).sum(1)>1).any())

def draw(rng,prefix=None,branch_not=None):
 while True:
  p=rng.integers(0,4,8).tolist()
  if prefix is not None:p[:3]=prefix
  if branch_not is not None and p[3]==branch_not:continue
  if allowed(p):return p

def make_registry(seed):
 rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261120,73191,int(seed)])));spec=[]
 for pair in ['AB','AC','BC']:
  for kind in ['shared','conflict']:
   for q in range(40):spec.append(dict(pair=pair,kind=kind,order=q%2,mapping=(q//2)%2,pattern=(q//4)%2))
 rng.shuffle(spec);episodes=[];updates=[]
 for e,item in enumerate(spec,1):
  p=draw(rng);paths=[p] if item['kind']=='shared' else [p,draw(rng,p[:3],p[3])];first,second=item['pair'][::1 if item['order']==0 else -1];contexts=[first,second,first,second];m=item['mapping'];p1=[m,m,1-m,1-m] if item['pattern']==0 else [m,1-m,1-m,m];p2=[m,1-m,m,1-m]
  if item['kind']=='shared':p1=p2=[0]*4
  rows={}
  for condition,sequences in [('P1',p1),('P2',p2)]:rows[condition]=[dict(inner_update=k+1,block=k//5+1,context=contexts[k//5],sequence=sequences[k//5]) for k in range(20)]
  episodes.append(dict(episode=e,**item,paths=paths,first_context=first,second_context=second));updates.append(rows)
 return dict(seed=int(seed),episodes=episodes,updates=updates)

def validate(r):
 if len(r['episodes'])!=240 or len(r['updates'])!=240:raise ValueError('Wrong episode budget')
 counts={};cells={}
 for ep,rows in zip(r['episodes'],r['updates']):
  counts[(ep['pair'],ep['kind'])]=counts.get((ep['pair'],ep['kind']),0)+1;key=(ep['pair'],ep['kind'],ep['order'],ep['mapping']);cells[key]=cells.get(key,0)+1
  if not all(allowed(p) for p in ep['paths']):raise ValueError('Target-equivalent,invalid or lookup-only sequence leaked')
  if ep['kind']=='conflict' and not(ep['paths'][0][:3]==ep['paths'][1][:3] and ep['paths'][0][3]!=ep['paths'][1][3]):raise ValueError('Conflict must follow common prefix')
  if ep['kind']=='shared' and rows['P1']!=rows['P2']:raise ValueError('Shared episodes must match exactly')
  for c in ['P1','P2']:
   if len(rows[c])!=20:raise ValueError('Wrong inner budget')
   for j,x in enumerate(rows[c]):
    if x['inner_update']!=j+1 or x['block']!=j//5+1 or x['context']!=[ep['first_context'],ep['second_context']][(j//5)%2]:raise ValueError('Wrong block schedule')
  for field in ['context','sequence']:
   a={x:sum(q[field]==x for q in rows['P1']) for x in {q[field] for q in rows['P1']}};b={x:sum(q[field]==x for q in rows['P2']) for x in {q[field] for q in rows['P2']}}
   if a!=b:raise ValueError('Marginal exposure mismatch')
  if ep['kind']=='conflict':
   for c,expected in [('P1',[5]*4),('P2',[10]*2)]:
    joint={}
    for q in rows[c]:joint[(q['context'],q['sequence'])]=joint.get((q['context'],q['sequence']),0)+1
    if sorted(joint.values())!=expected:raise ValueError('Incorrect context information')
 if any(v!=40 for v in counts.values()) or len(counts)!=6 or any(v!=10 for v in cells.values()) or len(cells)!=24:raise ValueError('Counterbalancing failed')
 return True
