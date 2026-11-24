import numpy as np
import pandas as pd
import pytest
from experiments.context_experience_prior.analysis import validate_grid,seed_interval,paired_contrasts,endpoint_gate

def test_complete_grid_and_paired_not_pooled():
 f=pd.DataFrame([dict(seed=s,condition=c,update=t,cue=q,ce=s+(2 if c=='P2-I' else 0),accuracy=1.) for s in range(20) for c in ['P1-I','P2-I'] for t in [0] for q in ['A']])
 validate_grid(f,conditions=['P1-I','P2-I'],updates=[0],cues=['A'])
 with pytest.raises(ValueError):validate_grid(pd.concat([f.iloc[:-1],f.iloc[:1]]),conditions=['P1-I','P2-I'],updates=[0],cues=['A'])
 p=paired_contrasts(f)
 x=p[(p.contrast=='P2-P1')&(p.organization=='I')&(p.metric=='ce')];assert len(x)==20 and (x.effect==2).all()
 r=seed_interval(x.effect);assert r['mean']==r['median']==r['ci_low']==r['ci_high']==2

def test_exact_original_functional_gate_not_geometry_or_floor():
 from experiments.recurrent_learning_transfer.measures import split_summary
 learning=pd.DataFrame([dict(seed=0,architecture='X',update=t,cue=c,ce=1.+(.2 if (t%20==10 and c=='B') or (t%20==0 and t>0 and c=='A') else 0)) for t in range(331) for c in ['A','B']])
 wide=pd.DataFrame([dict(seed=0,architecture='X',checkpoint=t,delta_A_A=.1,delta_B_A=.01,delta_B_B=.1,delta_A_B=.01) for t in [20,40,80,120,170,330]])
 old=split_summary(learning,wide);new=endpoint_gate(learning,wide)
 assert new.functional_split.tolist()==old.functional_split.tolist()==[True,True]
 wide[['delta_A_A','delta_B_A','delta_B_B','delta_A_B']]=0
 assert not endpoint_gate(learning,wide).functional_split.any()
