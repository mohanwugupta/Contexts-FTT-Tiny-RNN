import numpy as np
import pandas as pd
import pytest
import torch
from context_sequence.evaluation.rollout import Model
from experiments.output_compatible_trajectories.state_geometry import output_spaces,assert_equivalent
from experiments.output_compatible_trajectories.continuation import frozen_weight_guard
from experiments.context_failure_selective_repair.repair import joint_scale, directions, assert_matched, fixed_reverse, patch_trace, selective_gate, seed_contrasts, fixed_policy


def toy(delayed=True):
 w=np.zeros((4,8),'f');w[3,0]=2
 wh=np.zeros((8,8),'f')
 if delayed:wh[0,1]=1
 return Model(np.zeros((8,7),'f'),wh,np.zeros(8,'f'),w,np.zeros(4,'f'))


def test_geometric_norm_match_and_no_clipping():
 N=np.eye(8)[:,3:];U=N[:,:2];W=N[:,2:];d=np.arange(8)*.01
 vec,identity=directions(d,U,W,N,0,3,'A')
 a,status=joint_scale(np.zeros(8),[vec[q] for q in ['U','W','random']]);assert status=='valid' and a==.95
 assert_matched([a*vec[q] for q in ['U','W','random']])
 with pytest.raises(ValueError):assert_matched([vec['U'],2*vec['W'],vec['random']])
 a,status=joint_scale(np.ones(8),[np.ones(8)]*3);assert status=='infeasible' and np.isnan(a)
 a,status=joint_scale(np.ones(8),[-np.ones(8)]*3);assert status=='valid' and a>0
 _,status=joint_scale(np.zeros(8),[np.zeros(8)]*3);assert status=='not_identifiable'
 with pytest.raises(ValueError):patch_trace(toy(),np.ones(8)*1.1,3,'Mixed')


def test_current_equality_delayed_effect_and_visible_only_negative():
 for delayed in [False,True]:
  m=toy(delayed);h=np.zeros(8);v=h.copy();v[1]=.2;before=m.digest()
  assert_equivalent(m,h,v)
  a=patch_trace(m,h,3,'Mixed');b=patch_trace(m,v,3,'Mixed')
  assert a.token.tolist()==b.token.tolist() and np.array_equal(a[['cue_0','cue_1','cue_2']],b[['cue_0','cue_1','cue_2']])
  assert np.max(abs(a.filter(like='probability_').iloc[0]-b.filter(like='probability_').iloc[0]))==0
  delta=abs(a[a.lag>0].raw_ce.mean()-b[b.lag>0].raw_ce.mean())
  assert (delta>1e-4) if delayed else (delta==0)
  assert m.digest()==before
 with pytest.raises(ValueError):assert_equivalent(toy(),np.zeros(8),np.eye(8)[0]*.2)


def test_reverse_is_fixed_and_familiar_future_kept():
 r=np.arange(8)*.01;d=np.arange(8)*.02
 assert np.allclose(fixed_reverse(r,d,.4),r-.4*d)
 for g,c in [('A',[1,0,0]),('B',[0,1,0]),('Mixed',[.5,.5,0])]:
  f=patch_trace(toy(),np.zeros(8),3,g);assert np.all(f[['cue_0','cue_1','cue_2']].to_numpy()==c)


def test_nonspecific_improvement_missing_and_fixed_protocol():
 stats=pd.DataFrame([dict(contrast=c,ci_high=.1,mean=-.2,n_valid=20) for c in ['U_minus_W','U_minus_random']]);assert not selective_gate(stats,0)
 stats.ci_high=-.1;assert selective_gate(stats,0)
 assert not selective_gate(stats,1)
 stats.loc[0,'n_valid']=19;assert not selective_gate(stats,0)
 rows=pd.DataFrame([dict(seed=0,k=3,reference='A',arm=a,status='valid',future_ce=v) for a,v in [('natural',2),('U',1),('W',2),('random',2)]])
 q=seed_contrasts(rows);assert len(q[q.contrast=='U_minus_W'])==20
 missing=q[(q.seed==1)&(q.contrast=='U_minus_W')].iloc[0];assert missing.n_references==0 and np.isnan(missing.effect)
 with pytest.raises(ValueError):fixed_policy({'rank':3})
 with frozen_weight_guard():
  with pytest.raises(RuntimeError):torch.optim.SGD([torch.zeros(1,requires_grad=True)],lr=.1)
  with pytest.raises(RuntimeError):torch.ones(1,requires_grad=True).sum().backward()
