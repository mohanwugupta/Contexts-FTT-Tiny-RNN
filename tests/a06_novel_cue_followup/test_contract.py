import copy
import numpy as np
import pytest
import torch
from experiments.a06_novel_cue_followup import protocol as p

def test_registered_sphere_and_C_arc_exact_and_deterministic():
 a=p.cue_bank();b=p.cue_bank();assert a==b and len(a)==12
 vectors=np.array([r['vector'] for r in a]);np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-12,rtol=0)
 for v in vectors:assert min(np.linalg.norm(v-c) for c in np.eye(3))>1e-10
 for row,angle in zip(a[8:],[15,30,45,60]):assert row['similarity_C']==pytest.approx(np.cos(np.deg2rad(angle)),abs=1e-12)
 assert len({tuple(r['perpendicular_direction']) for r in a[8:]})==1

def test_only_readout_initialization_changes_and_bias_stays():
 base={'sequence_head.weight':torch.arange(32,dtype=torch.float32).reshape(4,8),'sequence_head.bias':torch.tensor([1.,-2.,3.,.5]),'other':torch.ones(8)}
 before=copy.deepcopy(base);changed=p.a06_state(base)
 for name in base:assert torch.equal(base[name],before[name]) and torch.equal(changed[name],base[name]*.25 if name=='sequence_head.weight' else base[name])

def test_full_state_coverage_not_inferred_or_interpolated():
 available=[0,10,20,30,40,50,60,100,170,330];rows=p.coverage(0,'A06',available,'cached.pt')
 assert len(rows)==331 and sum(r['status']=='missing_archived_state' for r in rows)==321
 assert not any(r['status']=='missing_archived_state' for r in rows if r['update'] in p.NOVEL_TIMES)

def test_budget_no_wrong_seed_no_repeat_no_extra_work(tmp_path):
 e=p.Ledger(tmp_path,max_steps=2);e.charge(2,1);e.charge(2,2)
 with pytest.raises(ValueError):e.charge(2,2)
 with pytest.raises(ValueError):e.charge(0,1)
 with pytest.raises(ValueError):e.charge(3,1)
 assert e.count==2

def test_bootstrap_paired_seed_units_not_cues():
 x=np.array([-1.,0.,1.,2.,3.,4.]);r=p.paired_interval(x);assert r['n']==6 and r['mean']==1.5 and r['n_unchanged']==1
 with pytest.raises(ValueError):p.paired_interval(np.repeat(x,12))
