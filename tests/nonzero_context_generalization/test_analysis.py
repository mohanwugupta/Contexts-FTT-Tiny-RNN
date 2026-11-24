import numpy as np
import pandas as pd
import pytest
from experiments.nonzero_context_generalization.analysis import reduce_network,validate_vectors,paired

def fixture():
 return pd.DataFrame([dict(seed=s,arm=a,update=330,alpha=1.,vector=j,ce=j/10,accuracy=1.,margin=2.,above_chance=j/10>np.log(4),success=j<=5,delta_zero=j/10-.1,gap=j/10-.2,margin_change=-1.,novel_contribution_norm=4.,native_delta_norm=4.,fixed_competitor_margin=0.,competitor_switch_penalty=1.,strong_familiar=True) for s in range(20) for a in ['HI','HC','NI','NC'] for j in range(20)])

def test_twenty_contexts_reduce_before_seed_inference():
 f=fixture();validate_vectors(f,last=330,only_updates=[330]);s=reduce_network(f)
 assert len(s)==80 and s.n_vectors.eq(20).all();np.testing.assert_allclose(s.R,.3);np.testing.assert_allclose(s.mean_ce,.95);np.testing.assert_allclose(s.median_ce,.95)
 # One exceptional context is an outcome, not 20 independent network replicates.
 f.loc[(f.seed==0)&(f.arm=='NC')&(f.vector==19),'ce']=100
 r=reduce_network(f);assert r[(r.seed==0)&(r.arm=='NC')].mean_ce.iloc[0]==pytest.approx((sum(np.arange(19)/10)+100)/20)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[:1]])]:
  with pytest.raises(ValueError):validate_vectors(bad,last=330,only_updates=[330])

def test_signed_paired_contrasts_keep_seed():
 f=reduce_network(fixture());f['mean_ce']=[{'HI':1,'HC':3,'NI':4,'NC':10}[a]+s for a,s in zip(f.arm,f.seed)];r=paired(f,'mean_ce');assert len(r)==20
 np.testing.assert_allclose(r.normalization,5);np.testing.assert_allclose(r.centering,4);np.testing.assert_allclose(r.interaction,4);np.testing.assert_allclose(r.NC_minus_NI,6)
