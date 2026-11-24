import numpy as np
import pandas as pd
import pytest
from experiments.recurrent_learning_development.analysis import blocks,interval,align_diagnostics,geometry_join

def fixture():
 rows=[]
 # Block1: A 2->1, B 3->3.5 (negative transfer); block2 B3.5->1.5,A1->.5.
 for k,a,b in [(0,2,3),(10,1,3.5),(20,.5,1.5)]:
  for c,v in [('A',a),('B',b)]:rows.append(dict(seed=0,architecture='RNN-Tanh',update=k,cue=c,ce=v))
 return pd.DataFrame(rows)

def test_signed_effects_and_endpoints():
 b=blocks(fixture(),ends=[10,20]);assert b.start.tolist()==[0,10]
 assert b.practiced.tolist()==['A','B'];np.testing.assert_allclose(b.own,[1,2]);np.testing.assert_allclose(b.cross,[-.5,.5]);np.testing.assert_allclose(b.specificity,[1.5,1.5])
 assert not b.both_start_floor.any()

def test_floor_is_not_shared_learning_and_crossing_is_explicit():
 f=fixture();f.loc[f['update']==0,'ce']=.04
 b=blocks(f,ends=[10,20]);assert b.iloc[0].both_start_floor and b.iloc[0].own_floor_limited
 assert b.iloc[0].specificity!=0 and 'shared_learning' not in b.columns
 f.loc[f['update']==10,'ce']=.01;b=blocks(f,ends=[10,20]);assert b.iloc[0].whole_block_below_marker
 f.loc[(f['update']==0)&(f.cue=='A'),'ce']=.05;b=blocks(f,ends=[10,20]);assert not b.iloc[0].own_floor_limited

def test_reject_missing_duplicate_or_wrong_cue_evaluations():
 f=fixture()
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]]),f.assign(cue='Mixed')]:
  with pytest.raises(ValueError):blocks(bad,ends=[10,20])

def test_seed_unit_intervals_do_not_accept_repeated_blocks_as_networks():
 f=pd.DataFrame({'seed':range(20),'value':range(20)})
 r=interval(f,'value');assert r['n_valid']==20 and r['mean']==9.5
 with pytest.raises(ValueError):interval(pd.concat([f,f]),'value')
 f['value']=np.nan;r=interval(f,'value');assert r['n_valid']==0 and np.isnan(r['mean'])

def test_diagnostics_align_start_and_end_not_future_chosen_best():
 b=blocks(fixture(),ends=[10,20]);d=[]
 for k in [0,10,20,330]:
  for train in ['A','B']:
   for cue in ['A','B']:d.append(dict(seed=0,architecture='RNN-Tanh',checkpoint=k,train_context=train,evaluation_context=cue,improvement=(2 if cue==train else -1)))
 a=align_diagnostics(b,pd.DataFrame(d));assert len(a)==4
 assert set(a.loc[a.alignment=='start','checkpoint'])=={0,10}
 assert set(a.loc[a.alignment=='end','checkpoint'])=={10,20}
 assert a.diagnostic_specificity.eq(3).all() and not a.checkpoint.eq(330).any()

def test_geometry_is_sparse_no_interpolation_or_classifier():
 b=blocks(fixture(),ends=[10,20]);g=pd.DataFrame([dict(seed=0,architecture='RNN-Tanh',endpoint=20,hidden_centroid_distance=9)])
 d=pd.DataFrame([dict(seed=0,architecture='RNN-Tanh',checkpoint=20,mean_specificity=.1)])
 out=geometry_join(b,g,d);assert len(out)==1 and out.endpoint.tolist()==[20]
 assert out.specificity.iloc[0]==1.5 and 'functional_split' not in out
