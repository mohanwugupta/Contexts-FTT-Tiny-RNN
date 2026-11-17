import numpy as np
import pandas as pd
import pytest
from experiments.output_compatible_trajectories.lag_audit import lag_seed, paired_rows
from experiments.output_compatible_trajectories.common import bootstrap_seed,add_novel_means


def test_T12_T14_paired_lags_and_nested_directions():
    rows=[]
    for seed in range(20):
        for cue in ['Novel30_A','Novel30_B']:
            for k in range(1,8):
                for t in range(k,8):rows.append(dict(cohort='x',condition='x',seed=seed,cue=cue,k=k,transition=t,lag=t-k,continuation='D0',raw_ce=float(t),natural_ce=float(t+1),raw_rescue=1.,midpoint_hidden_distance=2.,midpoint_task_distance=3.))
    f=pd.DataFrame(rows);s=lag_seed(f)
    assert s[s.cue.eq('Novel30Mean')].groupby('lag').n_positions.first().to_dict()=={0:7,1:6,2:5,3:4,4:3,5:2,6:1}
    assert s.raw_rescue.eq(1).all()
    b=bootstrap_seed(s[s.cue.eq('Novel30Mean')],['raw_rescue'],['cohort','condition','cue','continuation','lag'])
    assert b.n_seeds.eq(20).all() and b['mean'].eq(1).all()
    with pytest.raises(ValueError):lag_seed(pd.concat([f,f.iloc[:1]]))


def test_T12_pairing_rejects_misaligned_baseline():
    a=pd.DataFrame({'transition':[1,2],'raw_ce':[.2,.3],'lag':[0,1]})
    b=pd.DataFrame({'transition':[1,2],'raw_ce':[.6,.8],'lag':[0,1]})
    out=paired_rows(a,b)
    np.testing.assert_allclose(out.raw_rescue,[.4,.5])
    with pytest.raises(ValueError):paired_rows(a,b.iloc[::-1])
