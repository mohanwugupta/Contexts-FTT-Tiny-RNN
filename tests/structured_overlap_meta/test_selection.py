import pandas as pd,pytest
from experiments.structured_overlap_meta.selection import choose
from experiments.structured_overlap_meta.evaluation import jobs_for

def data():return pd.DataFrame([dict(seed=s,arm=a,gap=.2,D_A=.1,D_B=.2,strong_familiar=1,conflict_accuracy=.8,nonfloor_both=True,Mixed_both_success=0) for s in range(6) for a in 'ABC'])
def test_fixed_gate_and_ties():
 f=data();r,q=choose(f);assert r['selected']=='B'
 f.loc[f.arm=='B','D_A']=-.01;assert choose(f)[0]['selected']=='C'
 f.loc[f.arm=='C','gap']=.31;assert choose(f)[0]['selected'] is None
 with pytest.raises(ValueError):choose(f.iloc[1:])
 f=data();f.loc[(f.arm=='B')&f.seed.isin([0,1]),'strong_familiar']=0;assert choose(f)[0]['selected']=='C'

def test_fixed_jobs():
 assert len(jobs_for('screen'))==18
 assert len(jobs_for('confirm','B'))==56
 with pytest.raises(ValueError):jobs_for('confirm','A')
