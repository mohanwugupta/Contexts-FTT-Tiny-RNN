import numpy as np
import pandas as pd
from experiments.a06_novel_cue_followup import analysis as a
def test_static_uses_matching_bias_and_preserves_probability_classes():
 z=np.ones((5,7,4));bias=np.array([1.,-2.,3.,4.]);s=a.static(z,bias)
 np.testing.assert_array_equal(s,.25*(z-bias)+bias);assert not np.array_equal(s,z*.25)
def test_missing_predictions_not_zero_effect_or_wrong_class():
 m,p=a.rows(None,0,'A06',1,'C','missing_archived_state','fixture')
 assert np.isnan(m['ce']) and np.isnan(m['correct_count']) and not m['status']=='completed'
 assert len(p)==7 and all(np.isnan(r['predicted_class']) for r in p)
def test_cue_summary_is_within_model_before_seed_comparison():
 f=pd.DataFrame(dict(cue=['r1','r2'],kind=['random_sphere']*2,ce=[.1,2.],accuracy=[1.,3/7],useful=[True,False],similarity_C=[.9,-.5]))
 r=a.distribution(f);assert r['fraction_useful']==.5 and r['median_ce']==1.05 and r['best_accuracy']==1.
 assert r['worst_accuracy']==3/7 and r['n_cues']==2
