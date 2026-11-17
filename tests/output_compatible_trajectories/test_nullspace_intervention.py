import numpy as np
import pytest
from context_sequence.evaluation.rollout import Model
from context_sequence.dynamics.closure import references
from experiments.output_compatible_trajectories.nullspace import generate_states,score_states


def synthetic(carry):
    wx=np.zeros((8,7),'float32');wh=np.zeros((8,8),'float32')
    if carry:wh[0,1]=1;wh[1,1]=1
    out=np.zeros((4,8),'float32');out[2,0]=3;out[3,0]=-3
    return Model(wx,wh,np.zeros(8,'float32'),out,np.zeros(4,'float32'))


def test_T24_no_memory_vs_hidden_to_visible_recurrence():
    for carry in [False,True]:
        m=synthetic(carry);a,b,mid=references(m)
        states=generate_states(m,np.zeros(8),2.,0,5,'midpoint',smoke=True)
        s,f,inv=score_states(m,np.zeros(8),states,5,[.5,.5,0],mid,(a,b))
        effect=s[s.kind.eq('nullspace')&s.feasible].future_ce_effect.to_numpy()
        if carry:assert np.max(np.abs(effect))>1e-4
        else:np.testing.assert_array_equal(effect,0)
        assert inv[inv.feasible].passed.all()


def test_T25_fixed_counts_signs_radii_and_norm_matching():
    m=synthetic(True)
    for anchor,expected in [('midpoint',192),('natural',48)]:
        f=generate_states(m,np.zeros(8),2.,4,1,anchor)
        assert len(f)==expected and set(f['sign'])=={-1,1}
        null=f[f.kind.eq('nullspace')];control=f[f.kind.eq('visible_control')]
        assert len(null)==len(control)==expected//2
        assert len(null[['direction_index','radius_fraction','sign']].drop_duplicates())==len(null)
        for pair,g in f.groupby('pair_id'):
            if g.feasible.all():np.testing.assert_allclose(g.realized_radius.iloc[0],g.realized_radius.iloc[1],atol=1e-12)


def test_T27_bad_equivalence_rejected_before_future_scoring(monkeypatch):
    from experiments.output_compatible_trajectories import nullspace
    m=synthetic(True);a,b,mid=references(m)
    states=generate_states(m,np.zeros(8),2.,0,1,'midpoint',smoke=True).iloc[:1].copy()
    bad=np.zeros(8);bad[0]=.4;states.at[states.index[0],'state']=bad
    def forbidden(*a,**kw):raise AssertionError('Future scorer called before current-output validation')
    monkeypatch.setattr(nullspace,'continue_many',forbidden)
    with pytest.raises(ValueError):score_states(m,np.zeros(8),states,1,[.5,.5,0],mid,(a,b))


def test_actual_float32_head_difference_is_only_roundoff():
    from experiments.output_compatible_trajectories.nullspace import source_head_audit
    m=synthetic(True);anchor=np.zeros(8);null=anchor.copy();null[1]=.5
    q=source_head_audit(m,anchor,null,2)
    assert q['source_float32_softmax_difference']<=q['source_float32_probability_bound']
    visible=anchor.copy();visible[0]=.5
    with pytest.raises(ValueError):source_head_audit(m,anchor,visible,2)


def test_raw_direction_summary_precedes_strict_variant_aggregation():
    import pandas as pd
    from experiments.output_compatible_trajectories.nullspace import summarize_states,strict_novel_means
    f=pd.DataFrame([dict(cohort='canonical',condition='High_monotonic',seed=0,cue='Novel30_A',k=1,anchor_kind='midpoint',kind='nullspace',feasible=True,zero_radius=False,future_ce=1.,anchor_future_ce=1.2,future_ce_effect=-.2,absolute_future_ce_effect=.2,relative_radius=.3,viable_continuation=True,far=True,future_softmax_max_difference=.1)])
    a=summarize_states(f)
    assert a.cue.tolist()==['Novel30_A']
    keys=['cohort','condition','seed','k','anchor_kind','kind']
    with pytest.raises(ValueError):strict_novel_means(a,keys,['mean_effect'])
    b=a.copy();b['cue']='Novel30_B';b['mean_effect']=np.nan
    combined=strict_novel_means(pd.concat([a,b]),keys,['mean_effect'])
    assert combined[combined.cue.eq('Novel30Mean')].mean_effect.isna().all()
