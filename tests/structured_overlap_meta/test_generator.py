import copy
import numpy as np
import pytest
from experiments.structured_overlap_meta.generator import family,episode,masks,validate,block_specs
from experiments.context_experience_prior.registry import allowed

def test_full_prefix_masks_no_false_post_divergence_conflicts():
 rng=np.random.default_rng(902101);f=family(rng,{'X':'A','Y':'B','Q':'C'},(0,1),0,5,set())
 for k,n in [(0,14),(1,11),(2,9)]:
  e=episode(f,k);assert validate(e,set()) and sum(map(sum,e['novel_mask']))==n and sum(map(sum,e['critical_mask']))==2*k
  assert all(allowed(p) for c in ['X','Y'] for p in e['paths'][c])
 e=episode(f,2);assert np.array(e['critical_mask'])[:,6].sum()==0 and np.array(e['novel_mask'])[:,6].sum()==0
 bad=copy.deepcopy(e);bad['novel_mask'][0][5]=True
 with pytest.raises(ValueError):validate(bad,set())
 bad=copy.deepcopy(e);bad['paths']['Y'][0][0]=(bad['paths']['Y'][0][0]+1)%4
 with pytest.raises(ValueError):validate(bad,set())
 with pytest.raises(ValueError):validate(e,{tuple(e['paths']['X'][0])})

def test_fixed_ecology_balance_and_one_exception_position_targets():
 specs=block_specs(np.random.default_rng(388));assert len(specs)==480
 for arm,counts in [('A',{0:240,1:0,2:240}),('B',{0:120,1:240,2:120}),('C',{0:240,1:192,2:48})]:
  assert {k:sum(s['k'][arm]==k for s in specs) for k in range(3)}==counts
  for role in ['X','Y','Q']:assert {c:sum(s['roles'][role]==c for s in specs) for c in 'ABC'}==dict.fromkeys('ABC',160)
  critical=[]
  for s in specs:
   k=s['k'][arm]
   if k==2:critical.extend([(5,tuple(s['contrast'])),(6,tuple(s['contrast']))])
   elif k==1:critical.append((s['one_position'],tuple(s['contrast'])))
  for p in [5,6]:
   counts={ab:critical.count((p,ab)) for ab in [(a,b) for a in range(4) for b in range(4) if a!=b]};assert len(set(counts.values()))==1
 assert sum(s['k']['A'] for s in specs)==sum(s['k']['B'] for s in specs)==480 and sum(s['k']['C'] for s in specs)==288
