from experiments.structured_overlap_meta.streams import make_stream,summary

def test_all_arms_share_families_but_differ_only_in_exception_distribution():
 families,arms=make_stream(99,480,set());assert len(families)==480
 for i,f in enumerate(families):
  for a in 'ABC':assert arms[a][i]['paths']['X']==f['base'] and arms[a][i]['roles']==f['roles']
 sums={a:summary(v) for a,v in arms.items()};assert sums['A']['exceptions']==sums['B']['exceptions']==480 and sums['C']['exceptions']==288
 assert sums['A']['eligible_cells']==sums['B']['eligible_cells']==5520 and sums['C']['eligible_cells']==6000
 assert sums['A']['critical_cells']==sums['B']['critical_cells']==960 and sums['C']['critical_cells']==576
 assert [sums[a]['type_counts'] for a in 'ABC']==[{0:240,1:0,2:240},{0:120,1:240,2:120},{0:240,1:192,2:48}]
 assert all(s['role_counts']=={r:{c:160 for c in 'ABC'} for r in ['X','Y','Q']} for s in sums.values())

def test_conditional_critical_objective_mass_is_disclosed_not_falsely_matched():
 from experiments.structured_overlap_meta.generator import block_specs
 import numpy as np
 specs=block_specs(np.random.default_rng(78))
 # Mean critical loss is active per nonshared EPISODE, not per exception.
 mass={a:.5*.25*sum(s['k'][a]>0 for s in specs)/480 for a in 'ABC'}
 assert mass=={'A':.0625,'B':.09375,'C':.0625}
