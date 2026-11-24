import pandas as pd
from experiments.context_experience_prior.timing import first_crossings

def test_first_pass_not_best_checkpoint_or_censored_zero():
 f=pd.DataFrame([dict(seed=0,condition='P1-I',update=t,strong_familiar=(t==1),cue=c,ce=(.1 if c!='C' else 2),accuracy=1) for t in range(3) for c in ['A','B','C','Mixed-both']]);r=first_crossings(f).iloc[0]
 assert r.first_familiar==1 and pd.isna(r.first_C) and r.C_order=='not_observed' and r.familiar_success_states==1
