import numpy as np
import pandas as pd
from context_sequence.dynamics.closure import natural
from experiments.output_compatible_trajectories.config import SEQUENCE
from experiments.output_compatible_trajectories.continuation import digest_state
from experiments.output_compatible_trajectories.cue_access import cue_branch,branch_vectors,viable_distance,cue_only
from context_sequence.evaluation.rollout import Model

def synthetic(carry):
    wx=np.zeros((8,7),'float32');wh=np.eye(8,dtype='float32')*.4
    out=np.zeros((4,8),'float32');out[2,0]=3;out[3,0]=-3
    return Model(wx,wh,np.zeros(8,'float32'),out,np.zeros(4,'float32'))


def test_T40_T42_identical_prestate_and_explicit_schedules():
    m=synthetic(True);pre=np.arange(8,dtype='float32')/20;before=pre.tobytes();weights=m.digest()
    for persistent in [False,True]:
        t=cue_branch(m,pre,3,[1,0,0],[0,1,0],persistent)
        assert pre.tobytes()==before and m.digest()==weights
        assert t.token.tolist()==list(SEQUENCE[2:7]) and t.target.tolist()==list(SEQUENCE[3:])
        np.testing.assert_array_equal(t[['cue_0','cue_1','cue_2']].iloc[0],[1,0,0])
        expected=[1,0,0] if persistent else [0,1,0]
        np.testing.assert_array_equal(t[['cue_0','cue_1','cue_2']].iloc[1:],np.tile(expected,(4,1)))
        assert t.prestate_hash.nunique()==1 and t.prestate_hash.iloc[0]==digest_state(pre)
    pulse=cue_branch(m,pre,3,[1,0,0],[0,1,0],False)
    takeover=cue_branch(m,pre,3,[1,0,0],[0,1,0],True)
    np.testing.assert_array_equal(pulse.filter(regex='^hidden_').iloc[0],takeover.filter(regex='^hidden_').iloc[0])


def test_T41_native_replay_and_T43_frozen_vectors():
    m=synthetic(True);m.wih[:,4:]=np.arange(24,dtype='float32').reshape(8,3)/20
    cues={'A':np.array([1.,0,0]),'B':np.array([0.,1,0]),'Mixed':np.array([.5,.5,0]),'NoContext':np.zeros(3),'Novel30_A':np.array([.3,.4,.5])}
    vectors=branch_vectors(cues,'A','Novel30_A')
    for key,name in [('native_source','A'),('A','A'),('B','B'),('Mixed','Mixed'),('zero','NoContext'),('target_native','Novel30_A')]:np.testing.assert_array_equal(vectors[key],cues[name])
    q=natural(m,cues['A'],SEQUENCE)
    for k in [1,3,5]:
        pre=np.zeros(8,'float32') if k==1 else q['hidden'][k-2].copy()
        t=cue_branch(m,pre,k,cues['A'],cues['A'],False)
        np.testing.assert_array_equal(t[[f'hidden_{i}' for i in range(8)]],q['hidden'][k-1:])
        np.testing.assert_array_equal(t[[f'logit_{i}' for i in range(4)]],q['logits'][k-1:])


def test_T44_T46_oracle_exclusion_and_missing_reference():
    f=pd.DataFrame({'intervention_type':['cue_only','state_oracle_reference'],'future_ce':[2.,0.]})
    assert cue_only(f).future_ce.tolist()==[2.]
    d,status=viable_distance(np.zeros(8),[])
    assert np.isnan(d) and status=='not_established'
    d,status=viable_distance(np.zeros(8),[np.ones(8)])
    assert d==np.sqrt(8) and status=='finite_found_set'
