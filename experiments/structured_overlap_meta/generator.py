"""Matched two-example families; critical/transfer cells use complete prefixes."""
import copy,itertools
import numpy as np
from experiments.context_experience_prior.registry import draw,allowed
CONTRASTS=[(a,b) for a in range(4) for b in range(4) if a!=b];ROLES=[dict(zip(['X','Y','Q'],p)) for p in itertools.permutations('ABC')]

def masks(paths):
 critical=np.zeros((4,7),bool);novel=np.zeros((2,7),bool)
 for j in range(2):
  x,y=paths['X'][j],paths['Y'][j]
  for t in range(7):
   same=x[:t+1]==y[:t+1]
   if same and x[t+1]!=y[t+1]:critical[j,t]=critical[j+2,t]=True
   if same and x[t+1]==y[t+1]:novel[j,t]=True
 return critical,novel

def family(rng,roles,contrast,orientation,one_position,reserved):
 a,b=map(int,contrast);assert a!=b and orientation in [0,1] and one_position in [5,6];base=[];alts=[];rejected={};positions=[5 if j==orientation else 6 for j in range(2)]
 for j,p in enumerate(positions):
  for attempt in range(100000):
   x=draw(rng);y=x.copy();y[p]=b;reason=None
   if x[p]!=a:reason='registered_contrast_target'
   elif j and x[0]==base[0][0]:reason='ambiguous_within_context_start'
   elif not allowed(y):reason='invalid_alternative'
   elif tuple(x) in reserved or tuple(y) in reserved:reason='reserved_path'
   if reason:rejected[reason]=rejected.get(reason,0)+1;continue
   base.append(x);alts.append(y);break
  else:raise RuntimeError('Fixed generator constraints exhausted; do not relax')
 return dict(base=base,alternatives=alts,roles=copy.deepcopy(roles),contrast=[a,b],positions=positions,one_position=int(one_position),one_subtrial=positions.index(one_position),orientation=int(orientation),rejection_counts=rejected)

def episode(f,k):
 assert k in [0,1,2];active=[] if k==0 else ([int(f['one_subtrial'])] if k==1 else [0,1]);paths={'X':copy.deepcopy(f['base']),'Y':[copy.deepcopy(f['alternatives'][j] if j in active else f['base'][j]) for j in range(2)]};crit,nov=masks(paths)
 return dict(family_id=f.get('family_id'),roles=copy.deepcopy(f['roles']),n_exceptions=k,active_subtrials=active,registered_positions=f['positions'],paths=paths,inputs={c:[p[:-1] for p in paths[c]] for c in ['X','Y']},targets={c:[p[1:] for p in paths[c]] for c in ['X','Y']},critical_mask=crit.tolist(),novel_mask=nov.tolist(),eligible_count=int(nov.sum()),critical_cells=int(crit.sum()))

def validate(e,reserved):
 try:
  assert set(e['roles'].values())==set('ABC') and set(e['roles'])=={'X','Y','Q'}
  x,y=e['paths']['X'],e['paths']['Y'];assert len(x)==len(y)==2 and x[0][0]!=x[1][0]
  assert all(allowed(p) and tuple(p) not in reserved for p in x+y)
  active=[];preferences={}
  for j in range(2):
   diff=[i for i in range(8) if x[j][i]!=y[j][i]]
   if diff:
    assert diff==[e['registered_positions'][j]] and diff[0] in [5,6];active.append(j)
    p=diff[0];preferences.setdefault(p,[]).append((x[j][p],y[j][p]))
   assert e['inputs']['X'][j]==x[j][:-1] and e['inputs']['Y'][j]==y[j][:-1] and e['targets']['X'][j]==x[j][1:] and e['targets']['Y'][j]==y[j][1:]
  assert active==e['active_subtrials'] and len(active)==e['n_exceptions']
  if len(active)==2:assert set(e['registered_positions'])=={5,6}
  # Full prefixes under each practiced context cannot carry contradictory labels.
  for c in ['X','Y']:
   seen={}
   for path in e['paths'][c]:
    for t in range(7):
     prefix=tuple(path[:t+1]);assert prefix not in seen or seen[prefix]==path[t+1];seen[prefix]=path[t+1]
  for p,edges in preferences.items():
   reach=np.zeros((4,4),bool)
   for a,b in edges:reach[a,b]=True
   for k in range(4):reach|=reach[:,k,None]&reach[None,k,:]
   assert not np.diag(reach).any()
  crit,nov=masks(e['paths']);np.testing.assert_array_equal(e['critical_mask'],crit);np.testing.assert_array_equal(e['novel_mask'],nov);assert crit.sum()==2*e['n_exceptions'] and nov.sum()>0 and e['eligible_count']==nov.sum();assert not (crit[:2]&nov).any()
  return True
 except (AssertionError,KeyError,TypeError,IndexError) as exc:raise ValueError('Invalid structured family, prefix supervision or query mask') from exc

def block_specs(rng):
 rows=[]
 # Identical joint-arm profiles across all models; every profile count/contrast is balanced.
 profiles=[(20,{'A':0,'B':0,'C':0}),(20,{'A':0,'B':1,'C':0}),(20,{'A':2,'B':1,'C':1}),(12,{'A':2,'B':2,'C':1}),(8,{'A':2,'B':2,'C':2})]
 for group,(per_role,k) in enumerate(profiles):
  roles=[copy.deepcopy(r) for r in ROLES for _ in range(per_role)];rng.shuffle(roles);n=len(roles)//12;attrs=[]
  for i,ab in enumerate(CONTRASTS):
   combos=[(0,5),(0,6),(1,5),(1,6)]*(n//4)
   if n%4:combos+=([(0,5),(1,6)] if i%2==0 else [(0,6),(1,5)])
   assert len(combos)==n
   for orientation,p in combos:attrs.append(dict(contrast=list(ab),orientation=orientation,one_position=p))
  rng.shuffle(attrs)
  for role,attr in zip(roles,attrs):rows.append(dict(profile=group,roles=role,k=copy.deepcopy(k),**attr))
 rng.shuffle(rows);assert len(rows)==480;return rows
