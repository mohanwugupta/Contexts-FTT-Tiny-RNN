import numpy as np,pandas as pd
from experiments.ab_update_transfer_analysis import analysis

def test_exact_band_boundaries_and_no_floor_as_failed_transfer():
 assert analysis.loss_band(4.)=='2-4' and analysis.loss_band(2.)=='2-4'
 assert analysis.loss_band(1.)=='1-2' and analysis.loss_band(.5)=='0.5-1'
 assert analysis.loss_band(.1)=='0.1-0.5' and analysis.loss_band(.05)=='outside_registered_bands'
 assert analysis.effect(0.01,0.02,1.,2.)==dict(gain=-.01,own_gain=-.01,cross_gain=-1.,specificity=.99,own_floor=True,probe_floor=True,own_deterioration=True,cross_deterioration=True)

def test_realized_gains_exposure_and_negative_sign():
 rows=pd.DataFrame([dict(target_update=k,ce=5.-k if k<3 else 6.+k,probe_context='C') for k in range(21)])
 f=analysis.realized_intervals(rows)
 assert f[f.interval.eq('0-10')].n_A_updates.iloc[0]==10 and f[f.interval.eq('0-10')].n_B_updates.iloc[0]==0
 z=f[f.interval.eq('10-20')];assert z.n_A_updates.iloc[0]==0 and z.n_B_updates.iloc[0]==10
 assert f.realized_gain.min()<0

def test_parameter_motion_reports_other_cue_and_policy():
 import torch,copy
 pre=dict(model={'recurrent.weight_ih_l0':torch.zeros(8,7),'out':torch.zeros(1)},policy={'raw':torch.tensor(0.)})
 post=copy.deepcopy(pre);post['model']['recurrent.weight_ih_l0'][:,5]=.5;post['policy']['raw']+=.1
 r=analysis.motions(pre,post)
 assert r['motion_cue_B']>0 and r['motion_cue_A']==0 and r['motion_cue_C']==0
 assert r['policy_motion']>0

def test_early_metrics_match_original_without_changing_criterion():
 import pytest
 e=pd.DataFrame([dict(trajectory_id='x',source_context='A',target_update=0,branch_step=10,probe_context='C',own_gain=3.,cross_gain=1.,specificity=2.),dict(trajectory_id='x',source_context='B',target_update=10,branch_step=10,probe_context='C',own_gain=4.,cross_gain=1.,specificity=3.)])
 original=pd.DataFrame([dict(trajectory_id='x',D_A=2.,D_B=3.,own_A=3.,own_B=4.,cross_A=1.,cross_B=1.)]);analysis.check_early(e,original)
 with pytest.raises(AssertionError):analysis.check_early(e.assign(specificity=99.),original)

def test_required_parquet_storage_keeps_exact_values_and_rejects_changes(tmp_path):
 import pytest
 f=pd.DataFrame([dict(seed=0,gain=-1.25,probe='NoContext')]);p=tmp_path/'raw.parquet';analysis.parquet(p,f);original=p.read_bytes();analysis.parquet(p,f);assert p.read_bytes()==original
 with pytest.raises(AssertionError):analysis.parquet(p,f.assign(gain=1.25))
 pd.testing.assert_frame_equal(pd.read_parquet(p),f)

def test_wide_gain_matrix_has_signed_D_A_D_B_and_keeps_missing_probes():
 import numpy as np
 f=pd.DataFrame([dict(trajectory_id='x',source_context=s,probe_context=p,gain=v) for s,p,v in [('A','A',3.),('A','B',-1.),('B','A',1.),('B','B',4.)]])
 w=analysis.wide_gains(f,keys=['trajectory_id']).iloc[0]
 assert w.D_A==4 and w.D_B==3 and w.G_A_B==-1
 assert np.isnan(w.G_A_C) and np.isnan(w.G_B_NoContext)
