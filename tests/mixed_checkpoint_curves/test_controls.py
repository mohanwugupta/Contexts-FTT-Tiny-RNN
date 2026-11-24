import numpy as np
import pandas as pd
import pytest
from experiments.mixed_checkpoint_curves.evaluate import double_control,require_cues,summary

def ce(z):
 a=z.max(1);return a+np.log(np.exp(z-a[:,None]).sum(1))-z[:,0]

def fixture():
 a=np.tile([3.,0,0,0],(7,1));b=np.tile([0.,3,0,0],(7,1));m=(a+b)/2
 return {c:dict(hidden=z,logits=z,ce=ce(z),probabilities=np.exp(z)/np.exp(z).sum(1,keepdims=True)) for c,z in [('A',a),('B',b),('Mixed',m)]}

def test_midpoint_and_convex_ce_do_not_imply_probability_midpoint():
 d=fixture();r=double_control(d);assert r['passed'] and r['double_ce_gap']<=0
 assert not np.allclose(d['Mixed']['probabilities'],(d['A']['probabilities']+d['B']['probabilities'])/2)
 d['Mixed']['hidden']=d['Mixed']['hidden']+.001
 with pytest.raises(ValueError):double_control(d)

def test_wrong_physical_mixed_cue_rejected():
 assert require_cues({'A':[1,0,0],'B':[0,1,0],'Mixed':[.5,.5,0]})
 with pytest.raises(ValueError):require_cues({'A':[1,0,0],'B':[0,1,0],'Mixed':[.5,.5,1]})

def test_all20_seed_summaries_keep_exact_chance_threshold_and_weak_learners():
 rows=[]
 for seed in range(20):
  for cue in ['A','B','Mixed']:rows.append(dict(seed=seed,architecture='RNN-Tanh',checkpoint=80,cue=cue,ce=(np.log(4)+.00001 if cue=='Mixed' and seed<3 else .2),accuracy=1.))
 s=summary(pd.DataFrame(rows));assert s.iloc[0].Mixed_above_chance==3 and s.iloc[0].n_seeds==20
 with pytest.raises(ValueError):summary(pd.DataFrame(rows).iloc[:-1])
