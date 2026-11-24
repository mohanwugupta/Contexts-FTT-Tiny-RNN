import numpy as np
import pandas as pd
import pytest
from experiments.centered_context_linear.analysis import block_table,switch_table,qualify,validate_grid

def fixture():
 rows=[]
 for t in range(21):
  for cue in ['A','B']:
   # First block A improves by1, B worsens by.5. Second block B improves by1, A by.2.
   a=2-min(t,10)*.1-max(t-10,0)*.02;b=2+min(t,10)*.05-max(t-10,0)*.1
   rows.append(dict(seed=0,arm='Centered',update=t,cue=cue,ce=a if cue=='A' else b,accuracy=1))
 return pd.DataFrame(rows)

def test_signed_transfer_and_frozen_switches():
 f=fixture();b=block_table(f);assert len(b)==2
 np.testing.assert_allclose(b.practiced_gain,[1,1]);np.testing.assert_allclose(b.unpracticed_gain,[-.5,.2]);np.testing.assert_allclose(b.specificity,[1.5,.8])
 s=switch_table(f);assert len(s)==1 and s.iloc[0].boundary==10
 assert s.iloc[0].switch_cost==1.5
 # Initial bias subtraction must be explicit, with unchanged raw cost.
 f.loc[f.cue=='B','ce']+=3;s=switch_table(f);assert s.iloc[0].switch_cost==4.5 and s.iloc[0].baseline_adjusted_cost==1.5

def test_floor_not_shared_learning():
 f=fixture();f['ce']=.001;b=block_table(f);assert b.floor_at_start.all() and not b.directional_sign_evidence.any()

def test_familiar_only_qualification():
 f=pd.DataFrame([dict(seed=0,arm='Standard',update=170,cue=c,ce=l,accuracy=a) for c,l,a in [('A',.4,1),('B',.3,6/7),('Mixed',999,0),('C',999,0)]])
 assert qualify(f).strong_familiar.all()
 f.loc[f.cue=='B','accuracy']=5/7;assert not qualify(f).strong_familiar.any()

def test_incomplete_or_duplicate_rejected():
 f=fixture()
 with pytest.raises(ValueError):validate_grid(f,seeds=1,arms=['Centered'],last=20,cues=['A','B','Mixed','C'])
 with pytest.raises(ValueError):validate_grid(pd.concat([f,f.iloc[:1]]),seeds=1,arms=['Centered'],last=20,cues=['A','B'])

def test_jensen_does_not_guarantee_accuracy_cutoff():
 a=np.tile([8.,-8.,-8.,-8.],(7,1));b=a.copy()
 a[0]=[-.1,0,-8,-8];b[0]=[.01,0,-8,-8]
 a[1]=[.01,0,-8,-8];b[1]=[-.1,0,-8,-8]
 ce=lambda z:np.log(np.exp(z-z.max(1)[:,None]).sum(1))+z.max(1)-z[:,0]
 assert ce(a).mean()<.5 and ce(b).mean()<.5
 assert (a.argmax(1)==0).sum()==6 and (b.argmax(1)==0).sum()==6
 mid=(a+b)/2
 assert ce(mid).mean()<=(ce(a).mean()+ce(b).mean())/2 and (mid.argmax(1)==0).sum()==5

def test_per_position_geometry_removes_common_logit_offset():
 from experiments.centered_context_linear.analysis import position_geometry
 rows=[]
 for cue in ['A','B']:
  for t in range(1,8):
   rows.append(dict(seed=0,arm='Centered',update=0,cue=cue,position=t,**{f'hidden_{j}':1. if cue=='A' else 0. for j in range(8)},**{f'logit_{j}':float(j)+(100 if cue=='A' else 0) for j in range(4)}))
 p=position_geometry(pd.DataFrame(rows))
 assert len(p)==7
 np.testing.assert_allclose(p.hidden_separation,np.sqrt(8));assert p.centered_logit_separation.eq(0).all()
