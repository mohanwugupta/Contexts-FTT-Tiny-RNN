import numpy as np
import pandas as pd
import pytest
from experiments.context_mixing_sequence_continuity.analysis import interpretation,retention_checks,derived_tables
from experiments.context_mixing_sequence_continuity.verify import verify_outputs


def test_d4_guardrails_do_not_turn_confounds_into_mechanism():
    assert 'familiar_competence_confounded' in interpretation(-1.,-1.2,-.5,.3,1.,1.)
    assert 'repair_via_merging_possible' in interpretation(-1.,-1.2,-.5,.1,.2,1.)
    assert 'no_clear_switch_specific_benefit' in interpretation(-.1,-.3,.2,0.,1.,1.)
    assert 'switching_increases_mixed_error' in interpretation(.2,.1,.3,0.,1.,1.)
    assert 'incomplete_outcomes' in interpretation(np.nan,np.nan,np.nan,0.,1.,1.)


def test_d4_cohort_separation_ratio_not_mean_of_ratios():
    f=pd.DataFrame([{'seed':seed,'arm':arm,'A_ce':.1,'B_ce':.2,'familiar_mean_ce':.15,'hidden_separation':value,'task_separation':value} for seed,values in [(0,{'B':1,'I':1,'S':.9}),(1,{'B':10,'I':10,'S':7})] for arm,value in values.items()])
    q=retention_checks(f,original_hidden_mean=6.)
    assert abs(q['hidden_retention_ratio']-7.9/11)<1e-12
    assert not q['separation_guard_passed'] and q['familiar_guard_passed']
    assert q['original_hidden_mean']==6.
    bad=f.copy();bad.loc[bad.arm.eq('S'),'hidden_separation']=np.nan
    q=retention_checks(bad,6.)
    assert np.isnan(q['hidden_retention_ratio']) and q['interpretation_status']=='incomplete_outcomes'


def test_d4_seal_detects_changed_missing_extra_scientific_files(tmp_path):
    from context_sequence.provenance.hashing import sha256
    p=tmp_path/'table.csv';p.write_text('original');files={'table.csv':sha256(p)}
    assert verify_outputs(tmp_path,files)
    (tmp_path/'.DS_Store').write_text('Finder metadata');assert verify_outputs(tmp_path,files)
    (tmp_path/'extra.csv').write_text('new unsealed evidence')
    with pytest.raises(ValueError):verify_outputs(tmp_path,files)
    (tmp_path/'extra.csv').unlink();p.write_text('changed')
    with pytest.raises(ValueError):verify_outputs(tmp_path,files)
    p.unlink()
    with pytest.raises(FileNotFoundError):verify_outputs(tmp_path,files)
