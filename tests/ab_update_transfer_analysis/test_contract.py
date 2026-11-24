import copy
import numpy as np
import pytest
import torch
from experiments.ab_update_transfer_analysis import adapter, design

def fixture():
 from experiments.context_experience_prior import core
 from experiments.meta_learning_scaling.common import source_model
 m=source_model(core.base_pair(0)['I'].state_dict());p=core.policy();o=core.optimizer(m,p)
 return m,p,o

def test_fixed_plan_and_semantic_masks():
 assert design.PRIMARY==[0,10,20,40,60] and design.SECONDARY==[100,170,330,990]
 assert design.maximum_steps()==18108
 m,p,o=fixture();masks=adapter.masks(m.state_dict());keys=list(m.state_dict())
 for k in keys:
  total=sum(masks[g][k].int() for g in ['shared','cue_A','cue_B','cue_C'])
  assert torch.equal(total,torch.ones_like(total))
 assert masks['shared']['recurrent.weight_ih_l0'][:,:4].all()
 assert masks['cue_C']['recurrent.weight_ih_l0'][:,6].all()

def test_held_and_probe_isolation_and_exact_restore():
 m,p,o=fixture();s=adapter.capture(m,p,o,0);before=adapter.digest(s)
 obs=adapter.probes(m,p,o);assert set(obs)==set(design.PROBES)
 assert adapter.digest(adapter.capture(m,p,o,0))==before
 mm,pp,oo=adapter.restore(s);assert adapter.digest(adapter.capture(mm,pp,oo,0))==before
 bad=copy.deepcopy(s);bad['update']=10
 with pytest.raises(ValueError):adapter.restore(bad)
 assert not o.state

def test_gain_signs_and_mask_replacements():
 assert design.gain(3.,5.)==-2 and design.specificity(2.,-1.)==3
 m,p,o=fixture();pre=copy.deepcopy(m.state_dict());post=copy.deepcopy(pre)
 post['recurrent.weight_ih_l0'][:,4]+=1
 post['sequence_head.bias']+=.1
 variants=adapter.replacements(pre,post,'A')
 assert all(torch.equal(variants['full'][k],post[k]) for k in post)
 assert torch.equal(variants['cue_only']['sequence_head.bias'],pre['sequence_head.bias'])
 assert torch.equal(variants['shared_only']['recurrent.weight_ih_l0'][:,4],pre['recurrent.weight_ih_l0'][:,4])
 with pytest.raises(AssertionError):
  bad=copy.deepcopy(post);bad['recurrent.weight_ih_l0'][:,6]+=1;adapter.replacements(pre,bad,'A')

def test_statistics_shared_seed_draws_and_missingness():
 from experiments.ab_update_transfer_analysis.statistics import interval,indices
 a=np.array([1.,-2.,3.,np.nan,5.,6.]);r=interval(a)
 assert r['n']==5 and r['n_planned']==6 and r['n_negative']==1
 np.testing.assert_array_equal(indices(),np.random.default_rng(20261122).integers(6,size=(10000,6)))
 assert interval(a)['mean']==np.nanmean(a)
