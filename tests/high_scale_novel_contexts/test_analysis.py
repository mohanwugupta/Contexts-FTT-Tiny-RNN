import pandas as pd
import numpy as np
from experiments.high_scale_novel_contexts.analysis import paired330,aggregate_within_seed

def test_common_strong_pairing_keeps_actual_n_and_direction():
 rows=[]
 for seed in [0,1]:
  for a,value,good in [('RNN-Linear',5.,True),('RNN-Tanh',2.,seed==0)]:rows.append(dict(seed=seed,architecture=a,checkpoint=330,cue='Novel30_A',ce=value,gap=value-.5,accuracy=.5,strong_familiar=good))
 f=pd.DataFrame(rows);pairs,common=paired330(f);assert common==[0] and len(pairs)==1
 assert pairs.iloc[0].ce_Tanh_minus_Linear==-3 and pairs.iloc[0].gap_Tanh_minus_Linear==-3
 f.strong_familiar=False;pairs,common=paired330(f);assert pairs.empty and common==[]

def test_angle_means_computed_inside_seed_not_treating_cues_as_networks():
 f=pd.DataFrame([dict(seed=s,architecture='RNN-Linear',checkpoint=330,cue=c,ce=x+s,gap=x+s,accuracy=1.,strong_familiar=True) for s in [0,1] for c,x in [('Novel30_A',1),('Novel30_B',3),('Novel60_A',5),('Novel60_B',7)]])
 q=aggregate_within_seed(f);assert len(q)==6
 np.testing.assert_allclose(q[q.cue=='NovelMean'].sort_values('seed').ce,[4,5])
