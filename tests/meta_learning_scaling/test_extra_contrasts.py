import pandas as pd
from experiments.meta_learning_scaling.extra_contrasts import paired_rows

def test_functional_contrasts_remain_within_seed():
 f=pd.DataFrame([dict(seed=s,condition=p+'-I',dose=d,end=10,specificity=100*s+(2 if p=='P2' else 0)+(3 if d==1920 else 0)) for s in range(20) for p in ['P1','P2'] for d in [240,1920]])
 r=paired_rows(f,['end'],['specificity']);assert r[r.contrast=='P2-P1'].effect.eq(2).all();assert r[r.contrast=='P1:1920-240'].effect.eq(3).all()
