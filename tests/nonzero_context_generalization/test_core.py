import copy
import numpy as np
import torch
import pytest
from experiments.nonzero_context_generalization.core import registry,inject,column_rollout,operators,baseline64,assess,PARAMS,TARGET

def fixture():
 g=torch.Generator().manual_seed(912)
 shapes=[(8,7),(8,8),(8,),(8,),(4,8),(4,)]
 d={n:torch.randn(shape,generator=g) for n,shape in zip(PARAMS,shapes)};d[PARAMS[0]][:,6]=0;return d

def test_private_rng_nonzero_fixed_and_no_rng_change():
 t=torch.get_rng_state().clone();n=np.random.get_state();a,b=registry();aa,bb=registry()
 assert a.shape==(20,20,8) and b.dtype==np.float32 and np.all(np.linalg.norm(b,axis=2)>0)
 np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb);assert not np.array_equal(b[0],b[1]);assert torch.equal(t,torch.get_rng_state())
 nn=np.random.get_state();assert n[0]==nn[0] and np.array_equal(n[1],nn[1]) and n[2:]==nn[2:]

def test_readonly_equivalence_and_affine_indexing():
 d=fixture();before=copy.deepcopy(d);v=registry()[1][0]
 for alpha in [0,.25,.5,1,2]:
  a=inject(d,alpha*v,dtype=torch.float64);b=column_rollout(d,alpha*v,dtype=torch.float64);s,k=operators(d);z0,h0=baseline64(d);expectedh=h0[None]+np.einsum('tij,nj->nti',s,alpha*v.astype(float));expectedz=z0[None]+np.einsum('tcj,nj->ntc',k,alpha*v.astype(float))
  np.testing.assert_allclose(a[0],b[0],atol=1e-10,rtol=1e-12);np.testing.assert_allclose(a[1],expectedh,atol=1e-10,rtol=1e-12);np.testing.assert_allclose(a[0],expectedz,atol=1e-10,rtol=1e-12)
  if alpha:np.testing.assert_allclose(a[1][:,0]-h0[0],alpha*v.astype(float),atol=1e-10)
 for n in d:assert torch.equal(d[n],before[n]) and d[n].grad is None
 # t=1 includes I, not Wh. No future target enters K.
 np.testing.assert_allclose(k[0],d[PARAMS[4]].numpy(),atol=0)

def test_score_keeps_signed_changes_and_complete_distribution():
 z=np.zeros((20,7,4));z[:,:,0]=1;s=assess(z)
 assert s['ce'].shape==(20,7) and s['margin'].shape==(20,7)
 np.testing.assert_allclose(s['probabilities'].sum(-1),1)
 with pytest.raises(ValueError):inject(fixture(),np.zeros((20,7,8)))
