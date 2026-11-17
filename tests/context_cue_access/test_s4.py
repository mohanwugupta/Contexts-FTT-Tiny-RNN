import hashlib
import numpy as np
import pandas as pd
import pytest
from experiments.context_cue_access.analysis import seed_intervals,novel_means,paired_history,subgroup_summary,source_claim
from experiments.context_cue_access.seal import verify_output_files


def test_T33_T34_seed_clusters_and_novel_directions():
    f=pd.DataFrame([{'seed':s,'cue':cue,'policy':'x','k':1,'loss':s+offset} for s in range(4) for cue,offset in [('Novel30_A',0),('Novel30_B',2)]])
    g=novel_means(f,['policy','k'],['loss']);n=g[g.cue.eq('Novel30Mean')]
    assert len(n)==4 and n.loss.tolist()==[1,2,3,4]
    a=seed_intervals(n,['loss'],['cue','policy','k']);b=seed_intervals(n,['loss'],['cue','policy','k'])
    pd.testing.assert_frame_equal(a,b);assert a.n_seeds.iloc[0]==4
    with pytest.raises(ValueError):seed_intervals(pd.concat([n,n]),['loss'],['cue','policy','k'])
    with pytest.raises(ValueError):novel_means(f.iloc[:-1],['policy','k'],['loss'])


def test_T35_T36_T37_subgroup_missing_and_source_claims():
    f=pd.DataFrame({'seed':[0,1,2],'cue':['Mixed']*3,'k':[1]*3,'policy':['bounded_oracle']*3,'p3_reference_status':['available','available','no_feasible_prior_reference'],
      'p3_oracle_positive_native_failing':[True,False,np.nan],'future_ce':[1.,2.,np.nan],'viable_continuation':[True,False,np.nan]})
    q=subgroup_summary(f);assert q.n_full.iloc[0]==3 and q.n_subgroup.iloc[0]==1 and q.n_prior_unavailable.iloc[0]==1
    assert source_claim(0.01,2.,.2,True)=='source_replay_failed_viability'
    assert source_claim(2.,.01,.2,True)=='oracle_selected_source_confirmed'
    assert source_claim(0.01,.01,.2,False)=='native_source_confirmed'
    assert source_claim(.01,.01,np.nan,True)=='not_established_missing_reference'
    missing=seed_intervals(f,['future_ce'],['cue','k','policy']);assert missing.status.iloc[0]!='complete_seed_support' and np.isnan(missing['mean'].iloc[0])


def test_T38_sealed_files_detect_mutation_missing_and_extra(tmp_path):
    p=tmp_path/'data.csv';p.write_text('original');files={'data.csv':hashlib.sha256(p.read_bytes()).hexdigest()}
    assert verify_output_files(tmp_path,files)
    (tmp_path/'unexpected').write_text('extra')
    with pytest.raises(ValueError):verify_output_files(tmp_path,files)
    (tmp_path/'unexpected').unlink();p.write_text('changed')
    with pytest.raises(ValueError):verify_output_files(tmp_path,files)
    p.unlink()
    with pytest.raises(FileNotFoundError):verify_output_files(tmp_path,files)


def test_history_need_not_be_unique_when_early_only_also_succeeds():
    rows=[]
    for policy in ['fixed_candidates','bounded_oracle']:
        for arm,viable in [('E',True),('L',False),('R',True),('I',True)]:
            rows.append({'seed':0,'cue':'Mixed','k':3,'policy':policy,'arm':arm,'future_ce':.1 if viable else 2.,'viable_continuation':viable,'native_viable':False,'p3_oracle_positive_native_failing':True,'p3_reference_status':'available'})
    f=pd.DataFrame(rows);p=paired_history(f)
    assert p.R_success_L_failed.eq(1).all()
    assert p.R_success_both_singles_failed.eq(0).all()
    f.loc[f.arm.eq('E'),'viable_continuation']=False
    assert paired_history(f).R_success_both_singles_failed.eq(1).all()
