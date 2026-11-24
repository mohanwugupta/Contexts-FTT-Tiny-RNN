import numpy as np
import torch
from experiments.high_scale_constrained_scaling_10k import mechanism as m
from experiments.high_scale_constrained_scaling_10k.common import OUT,source_model

def test_seven_position_centered_linear_identity():
 fresh=torch.load(OUT/'initial/s00_scale4.pt',weights_only=False)['model']
 q,pos,dirs=m.geometry(fresh,fresh)
 assert len(pos)==7 and len(dirs)==172
 assert q['centered_contrast_mean']>0
 assert q['AB_identity_error']<1e-10
 assert all(p['position']==i+1 for i,p in enumerate(pos))
 assert all(abs(sum(p[f'centered_class_{k}'] for k in range(4)))<1e-10 for p in pos)
 assert all(abs(q['cosine_fresh_'+g]-1)<1e-10 for g in m.GROUPS)
 # A class-common offset must not alter the centered contrast norm.
 z=np.array([[3.,1.,4.,2.],[0.,-2.,1.,5.]])
 np.testing.assert_allclose(m.center(z),m.center(z+np.array([[100.],[-90.]])))

def test_projection_records_retain_no_projection_and_signed_changes():
 f={'token':1.,'context_A':2.,'context_B':3.,'context_C':4.,'output':5.}
 raw=np.zeros((3,172),np.float32);raw[0]=np.linspace(-3,3,172,dtype=np.float32)
 prev=raw[0]+.25
 a=m.legacy_projection(prev,raw[0],f)
 assert a.shape==(5,8)
 assert np.all(a[:,4]==1) and np.all(a[:,6:]==0)
 assert np.any(a[:,1]<a[:,0]) and np.all(a[:,5]>=0)

def test_dense_saved_weight_geometry_matches_independent_operator():
 from experiments.high_scale_constrained_scaling_10k.common import SOURCE
 for p in [OUT/'initial/s00_scale4.pt',SOURCE/'meta/s00_scale4/T_e1920.pt']:
  state=torch.load(p,weights_only=False)['model'];q,pos,_=m.geometry(state,state);v=m.flat(state).astype(float);d=m.dense_geometry(v)
  np.testing.assert_allclose(d['position_norms'],[x['centered_contrast'] for x in pos],atol=1e-10,rtol=1e-12)
  assert abs(d['centered_contrast_mean']-q['centered_contrast_mean'])<1e-10
  shifted=v.copy();shifted[136:168]+=np.tile(np.arange(8),4)
  np.testing.assert_allclose(m.dense_geometry(shifted)['position_norms'],d['position_norms'],atol=1e-10,rtol=1e-12)
