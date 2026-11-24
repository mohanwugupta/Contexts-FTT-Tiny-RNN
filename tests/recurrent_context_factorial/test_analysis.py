import numpy as np
import pandas as pd
import pytest
from experiments.recurrent_context_factorial.analysis import factorial_contrasts,assert_complete,endpoint_table

def test_signed_paired_factorial_not_pooled():
 f=pd.DataFrame([dict(seed=s,update=100,HI=100*s,HC=100*s+2,NI=100*s+3,NC=100*s+9) for s in range(20)])
 z=factorial_contrasts(f)
 np.testing.assert_array_equal(z.normalization,5);np.testing.assert_array_equal(z.centering,4);np.testing.assert_array_equal(z.interaction,4)
 with pytest.raises(ValueError):factorial_contrasts(pd.concat([f,f.iloc[[0]]]))

def test_all_seed_state_cue_cells_required():
 f=pd.DataFrame([dict(seed=s,arm=a,update=k,cue=c) for s in range(20) for a in ['HI','HC','NI','NC'] for k in range(331) for c in ['A','B','Mixed-average','Mixed-both','C','NoContext']]);assert_complete(f)
 with pytest.raises(ValueError):assert_complete(f.iloc[:-1])
 with pytest.raises(ValueError):assert_complete(pd.concat([f,f.iloc[[0]]]))

def test_qualification_ignores_protected_results():
 f=pd.DataFrame([dict(seed=0,arm='HI',update=100,cue=c,ce=.4 if c in ['A','B'] else 999,accuracy=1 if c in ['A','B'] else 0) for c in ['A','B','Mixed-average','Mixed-both','C','NoContext']]);q=endpoint_table(f)
 assert q.strong_familiar.all() and q.delta_mixed.iloc[0]==0 and not q[~q.cue.isin(['A','B'])].success.any()
 f['accuracy']=f.accuracy.astype(float);f.loc[f.cue=='B','accuracy']=5/7;assert not endpoint_table(f).strong_familiar.any()

def test_full_strength_mixed_is_not_jensen_bounded_or_probability_average():
 from experiments.recurrent_context_factorial.core import TARGET,score
 a=np.zeros((7,4));b=a.copy();c=a.copy();a[np.arange(7),TARGET]=4;c[np.arange(7),TARGET]=10
 avg=(a+b)/2;both=a+b-c
 assert (score(avg)['ce']<=(score(a)['ce']+score(b)['ce'])/2+1e-12).all()
 assert (score(both)['ce']>(score(a)['ce']+score(b)['ce'])/2).all()
 assert not np.allclose(score(avg)['probabilities'],(score(a)['probabilities']+score(b)['probabilities'])/2)

def test_decomposition_verifier_rejects_corrupt_margin_and_centering():
 import copy
 from experiments.recurrent_context_factorial.core import models,trace,CUES,decomposition
 from experiments.recurrent_context_factorial.verify import verify_decomposition
 m=models(0)['NC'];d={c:trace(m,c) for c in CUES};rows=pd.DataFrame(decomposition(d));z={c:d[c]['logits'] for c in CUES};verify_decomposition(z,rows)
 for column in ['u_centered_0','actual_avg_margin_change','margin_competitor_switch_penalty']:
  bad=rows.copy();bad.loc[0,column]+=100
  with pytest.raises(AssertionError):verify_decomposition(z,bad)
