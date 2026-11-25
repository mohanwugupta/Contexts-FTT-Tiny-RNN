import numpy as np,pandas as pd,pytest
from experiments.cloud_confidence_selected_v1.analysis import classify,accounting,paired,tables,validate


def test_category_contributions_use_all_seven_positions():
    p=pd.DataFrame(dict(condition=['candidate']*7,seed=[21000]*7,task_id=['task00']*7,update=[10]*7,cue=['C']*7,position=range(1,8),correct_reference=[False,False,True,True,True,False,True],correct_alternative=[False,True,False,True,True,False,True],ce_reference=[8.,2.,.1,.2,.3,7.,.2],ce_alternative=[4.,.1,2.,.2,.3,3.,.2]))
    p=classify(p);assert p.category.tolist()==['both_wrong','became_correct','became_wrong','both_correct','both_correct','both_wrong','both_correct']
    a=accounting(p,['condition','seed','task_id','update','cue']);assert a.positions.sum()==7
    assert a.set_index('category').loc['both_wrong','CE_gain_contribution']==pytest.approx(8/7)
    assert a.CE_gain_contribution.sum()==pytest.approx((p.ce_reference-p.ce_alternative).sum()/7)
    assert a.set_index('category').loc['became_wrong','correct_count_change']==-1


def test_pairing_rejects_mismatched_targets_or_duplicates():
    f=pd.DataFrame([dict(condition='candidate',seed=21000,task_id='task00',update=0,cue='C',position=1,view=v,target=target,token=1,ce=2.,correct=False,margin=-1.,predicted_class=2) for v,target in [('native',0),('quarter',1)]])
    with pytest.raises(ValueError):paired(f,'view','native','quarter',['condition','seed','task_id','update','cue','position'])
    f.target=0
    with pytest.raises(ValueError):paired(pd.concat([f,f.iloc[[0]]]),'view','native','quarter',['condition','seed','task_id','update','cue','position'])


def grid():
    rows=[]
    for condition in ['candidate','control']:
      for seed in [21000,21001]:
       for task in ['task00','task01','task02','task03']:
        for update in [0,10,20,40,80,120,170,330]:
         for cue in ['A','B','C','Mixed-average','NoContext']:
          for position in range(1,8):
           for view in ['native','quarter']:
            # Synthetic saved observations only, no model construction.
            z=np.array([-2.,2.,0.,-1.])*(.5 if condition=='candidate' else 1.)*(.25 if view=='quarter' else 1.)
            if update: z=z*.8
            s=z-z.max();logp=s-np.log(np.exp(s).sum());probs=np.exp(logp)
            rows.append(dict(condition=condition,seed=seed,task_id=task,update=update,cue=cue,position=position,view=view,target=0,token=1,ce=-logp[0],correct=False,margin=z[0]-max(z[1:]),predicted_class=1,**{f'logit_{i}':z[i] for i in range(4)},**{f'probability_{i}':probs[i] for i in range(4)}))
    return pd.DataFrame(rows)


def test_complete_grid_baselines_and_raw_probabilities():
    f=grid();validate(f);r=tables(f)
    assert len(r['within_readout_positions'])==4480 and len(r['initialization_positions'])==4480
    p=r['initialization_metrics'];assert len(p)==640
    assert p[p['update'].eq(0)].CE_gain_change_since0.eq(0).all()
    assert p[p['update'].eq(10)].CE_gain_change_since0.lt(0).all()
    assert r['within_readout_positions'].filter(regex='probability_').shape[1]==8
    for bad in [f.iloc[1:],pd.concat([f,f.iloc[[0]]]),f.replace({'NoContext':'Novel'})]:
        with pytest.raises(ValueError):validate(bad)
    b=f.copy();b.loc[0,'probability_0']+=.1
    with pytest.raises(ValueError):validate(b)


def test_fixed_bias_can_change_argmax():
    bias=np.array([0.,2.,0.,0.]);signal=np.array([3.,0.,-5.,-5.])
    assert np.argmax(signal+bias)==0
    assert np.argmax(.25*signal+bias)==1
    assert np.argmax(.25*(signal+bias))==0


def test_bootstrap_unit_is_seed_after_task_averaging():
    from experiments.cloud_confidence_selected_v1.analysis import seed_intervals
    f=pd.DataFrame([dict(view='native',cue='C',update=0,seed=seed,task_id=str(task),CE_gain=value,correct_count_change=0.,margin_change=value,CE_gain_change_since0=0.,correct_change_since0=0.) for seed,value in [(21000,1.),(21001,3.)] for task in range(4)])
    means,ci=seed_intervals(f,['view','cue','update'])
    assert len(means)==2 and ci.n.eq(2).all()
    r=ci[ci.metric.eq('CE_gain')].iloc[0]
    assert (r['mean'],r.ci_low,r.ci_high)==(2.,1.,3.)
