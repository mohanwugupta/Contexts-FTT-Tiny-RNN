import pandas as pd,numpy as np
from experiments.high_scale_meta_progression.target import validate_frozen_grid
from experiments.high_scale_meta_progression.audit import archived_target
from experiments.high_scale_meta_progression.measures import extended_boundaries
from experiments.recurrent_learning_transfer.measures import boundaries
import pytest

def test_saved330_switch_reconstruction_and_bad_grid():
 d=archived_target(0,4)['directory'];f=pd.read_csv(d/'evaluations.csv',float_precision='round_trip');p=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip');validate_frozen_grid(f,p,330)
 x=f[f.cue.isin(['A','B'])].assign(seed=0,architecture='test');pd.testing.assert_frame_equal(extended_boundaries(x),boundaries(x))
 with pytest.raises(ValueError):validate_frozen_grid(pd.concat([f,f.iloc[:1]]),p,330)
 wrong=f.copy();wrong.loc[wrong.cue.eq('Mixed-both'),'cue']='Mixed-average'
 with pytest.raises(ValueError):validate_frozen_grid(wrong,p,330)

def test_original_functional_gate_reproduces_archived_screen():
 from experiments.recurrent_learning_transfer.measures import specificity,split_summary
 from experiments.high_scale_meta_progression.common import SWEEP
 d=archived_target(0,4)['directory'];f=pd.read_csv(d/'evaluations.csv',float_precision='round_trip');dg=pd.read_csv(d/'diagnostics.csv',float_precision='round_trip');wide=[]
 for k,g in dg.groupby('update'):
  v={(r.train_context,r.evaluation_context):r.improvement for r in g.itertuples()};wide.append(dict(seed=0,architecture='test',checkpoint=k,delta_A_A=v['A','A'],delta_B_A=v['A','B'],delta_B_B=v['B','B'],delta_A_B=v['B','A']))
 q=split_summary(f[f.cue.isin(['A','B'])].assign(seed=0,architecture='test'),pd.DataFrame(wide));old=pd.read_csv(SWEEP/'tables/screen_functional_splitting.csv');old=old[old.seed.eq(0)&old.arm.eq('Fresh')&old.dose.eq(4)]
 for r in q.itertuples():
  z=old[old['update'].eq(r.endpoint)].iloc[0];assert bool(r.functional_split)==bool(z.functional_split);assert abs(r.mean_cycle_cost-z.mean_cycle_cost)<1e-12
