import numpy as np
import pandas as pd
import pytest
from experiments.original_learning_curves.analysis import actual_training,gaps,pooling

def learning():
 return pd.DataFrame([dict(seed=s,architecture='RNN-Linear',update=k,cue=c,ce=(a if c=='A' else b)) for s in [0] for k,a,b in [(0,4,6),(9,2.2,5.2),(10,2,5),(19,1.5,4.2),(20,1.4,4),(29,1.2,3.6),(30,1,3.5),(330,.01,.02)] for c in ['A','B']])

def test_actual_is_practiced_preupdate_loss_not_total_or_postupdate():
 l=learning();h=pd.DataFrame([dict(seed=0,architecture='RNN-Linear',update=10,training_cue='A',ce_before=2.2,total_loss=50),dict(seed=0,architecture='RNN-Linear',update=11,training_cue='B',ce_before=5.,total_loss=50)])
 q=actual_training(h,l);assert q.ce_before.tolist()==[2.2,5] and q.frozen_state.tolist()==[9,10]
 with pytest.raises(ValueError):actual_training(h.assign(training_cue='A'),l)
 with pytest.raises(ValueError):actual_training(h.assign(ce_before=50),l)

def test_absolute_gap_before_cohort_average_and_signed_B_minus_A():
 l=learning();other=l.copy();other['seed']=1;other['ce']=10-other.ce
 q=gaps(pd.concat([l,other]));f=q[q.state==10]
 assert f.signed_B_minus_A.tolist()==[3,-3] and f.gap_AB.mean()==3
 assert q.loc[q.state==10,'executed_A_to_B_switch'].all()
 assert not q.loc[q.state==330,'executed_A_to_B_switch'].any()
 assert q.loc[q.state==330,'end_censored'].all()

def test_pooling_signed_gains_and_final_missing_actual_switch():
 l=learning();q=pooling(l,ends=[10,30]);np.testing.assert_allclose(q.gain_A,[2,.4]);np.testing.assert_allclose(q.gain_B,[1,.5]);assert not any('ratio' in c for c in q.columns)
 l.loc[(l['update']==10)&(l.cue=='B'),'ce']=7;q=pooling(l,ends=[10]);assert q.gain_B.iloc[0]==-1
 with pytest.raises(ValueError):pooling(l.iloc[:-1],ends=[330])

def test_unpaired_or_duplicate_evaluation_rejected():
 l=learning()
 for bad in [l.iloc[:-1],pd.concat([l,l.iloc[:1]]),l.assign(cue='Mixed')]:
  with pytest.raises(ValueError):gaps(bad)
