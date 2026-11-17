import pandas as pd
import pytest
from experiments.output_compatible_trajectories.analysis import adjudicate,headline_counts
from experiments.output_compatible_trajectories.verify import verify_files
from experiments.output_compatible_trajectories.common import bootstrap_seed


def test_decision_table_preserves_negative_and_inconclusive():
    assert adjudicate(False,False,False)=='search_limited_inconclusive'
    assert adjudicate(True,False,False)=='alternative_exists_tested_cues_failed'
    assert adjudicate(True,True,False)=='alternative_exists_cue_viability_found'
    assert adjudicate(True,True,True)=='alternative_exists_exact_output_cue_access_found'


def test_T45_summary_never_counts_branches_as_networks():
    f=pd.DataFrame([dict(seed=i,k=1,cue='Mixed',oracle_far_viable_found=True,any_pulse_viable=False,native_viable=False,oracle_found_no_tested_pulse_viable=True) for i in range(2)])
    assert headline_counts(f).iloc[0].n_seeds==2
    with pytest.raises(ValueError):headline_counts(pd.concat([f,f]))
    with pytest.raises(ValueError):bootstrap_seed(pd.DataFrame({'seed':[0,0],'cue':['Mixed','Mixed'],'x':[1.,0.]}),['x'],['cue'])


def test_seal_detects_tamper_and_missing_file(tmp_path):
    import hashlib
    f=tmp_path/'x';f.write_text('original');digest=hashlib.sha256(f.read_bytes()).hexdigest()
    assert verify_files(tmp_path,{'x':digest})
    f.write_text('changed')
    with pytest.raises(ValueError):verify_files(tmp_path,{'x':digest})
    with pytest.raises(FileNotFoundError):verify_files(tmp_path,{'absent':digest})
