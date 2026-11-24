import numpy as np
import pandas as pd
import pytest
from experiments.architecture_averaging_factorial.statistics import contrasts,bootstrap


def fixture():
 vals={'FF-Linear':-.1,'FF-Tanh':.2,'RNN-Linear':-.2,'RNN-Tanh':.6}
 return pd.DataFrame([dict(seed=s,architecture=a,step=1000,Mixed_gap=v,familiar_ce=.1,strong_familiar=True,perfect_familiar=True) for s in range(20) for a,v in vals.items()])


def test_paired_factorial_interaction_sign_and_coverage():
 c=contrasts(fixture());assert len(c)==20
 assert np.allclose(c.recurrence_tanh,.4) and np.allclose(c.interaction,.5)
 assert c.tanh_pair_qualified.all() and c.all_four_qualified.all()
 with pytest.raises(ValueError):contrasts(fixture().iloc[:-1])


def test_seed_units_missing_and_conditional_qualification():
 f=pd.DataFrame({'seed':range(20),'effect':range(20)});assert bootstrap(f)['n_valid']==20
 with pytest.raises(ValueError):bootstrap(pd.concat([f,f]))
 f.loc[1:,'effect']=np.nan;q=bootstrap(f);assert q['n_valid']==1 and q['status']=='one_seed_not_population_interval'
 f['effect']=np.nan;assert bootstrap(f)['status']=='not_estimable'
