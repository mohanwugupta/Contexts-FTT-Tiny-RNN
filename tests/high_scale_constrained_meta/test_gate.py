import pandas as pd
from experiments.high_scale_constrained_meta.gate import decide

def fixture():return pd.DataFrame([dict(seed=s,arm=a,complete=True,strong=True,own_A=.6,own_B=.6,D_A=.3 if a=='High' else .02,D_B=.5 if a=='High' else .02,floor_A=False,floor_B=False,cycle=.25 if a=='High' else .01) for s in range(6) for a in ['High','Low']])
def test_majority_joint_and_paired_continuous_gate():
 f=fixture();assert decide(f)['passed']
 f.loc[f.arm.eq('High')&f.seed.le(2),'strong']=False;assert not decide(f)['passed']
 f=fixture();f.loc[f.arm.eq('High'),'D_B']=.03;assert not decide(f)['passed']
 f=fixture();f.loc[f.arm.eq('High'),'floor_A']=True;assert not decide(f)['passed']
 f=fixture();f.loc[f.arm.eq('High'),'D_A']=.07;assert not decide(f)['passed']
