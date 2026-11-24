import numpy as np
import pandas as pd
from experiments.nonzero_context_generalization.mechanism import estimates

def test_mechanism_paired_seed_unit_and_complete_metrics():
 f=pd.DataFrame([dict(seed=s,arm=a,update=330,baseline_ce=1+s+{'HI':0,'HC':2,'NI':3,'NC':9}[a],K_centered_norm=2+s+{'HI':0,'HC':2,'NI':3,'NC':9}[a]) for s in range(20) for a in ['HI','HC','NI','NC']])
 raw,ci=estimates(f,['baseline_ce','K_centered_norm']);assert len(raw)==40 and ci.n.eq(20).all()
 np.testing.assert_allclose(raw.normalization,5);np.testing.assert_allclose(raw.centering,4);np.testing.assert_allclose(raw.interaction,4)
 assert ci[ci.effect=='NC_minus_NI']['mean'].eq(6).all()
