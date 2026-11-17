from pathlib import Path
import numpy as np
import pytest
from experiments.context_mixing_sequence_continuity.source_adapter import records,validate_record,load_original,assert_unchanged,load_torch
from experiments.context_mixing_sequence_continuity.replay import natural_trace
from experiments.context_mixing_sequence_continuity.config import ROOT,SEQUENCE


def test_d0_expected_checkpoints_and_shas(tmp_path):
    r=records()[0]
    with pytest.raises(FileNotFoundError,match='blocked_missing_source'):load_original(tmp_path,r)
    changed={**r,'checkpoint_sha256':'0'*64}
    with pytest.raises(ValueError):load_original(ROOT,changed)


def test_d0_token_target_alignment():
    from context_sequence.evaluation.rollout import Model
    m=Model(np.zeros((8,7),'float32'),np.zeros((8,8),'float32'),np.zeros(8,'float32'),np.zeros((4,8),'float32'),np.zeros(4,'float32'))
    t=natural_trace(m,[1,0,0])
    assert t.token.tolist()==[0,1,2,0,3,1,2] and t.target.tolist()==[1,2,0,3,1,2,3]
    assert t.transition.tolist()==list(range(1,8))
    assert len(t[t.token.eq(0)].target.unique())==2


def test_d0_physical_mixed_identity():
    from experiments.context_mixing_sequence_continuity.source_adapter import cues
    c=cues();np.testing.assert_array_equal(c['Mixed'],(np.array(c['A'])+np.array(c['B']))/2)
    assert all(v[2]==0 for v in c.values())


def test_d0_native_source_replay():
    import pandas as pd
    r=records()[0];m=load_original(ROOT,r);t=natural_trace(m,[.5,.5,0])
    p=pd.read_csv(ROOT/'results/output_compatible_trajectories/v1_20261111/tables/p0_natural_trace.csv')
    p=p[p.condition.eq('High_monotonic')&p.seed.eq(0)&p.cue.eq('Mixed')]
    np.testing.assert_allclose(t[[f'logit_{i}' for i in range(4)]],p[[f'logit_{i}' for i in range(4)]],atol=1e-12,rtol=0)
    np.testing.assert_allclose(t.raw_ce,p.raw_ce,atol=1e-12,rtol=0)


def test_d0_no_weight_change():
    m=load_original(ROOT,records()[0]);before=m.digest();natural_trace(m,[1,0,0]);assert_unchanged(m,before)
    from context_sequence.evaluation.rollout import Model
    bad=Model(m.wih.copy(),m.whh.copy(),m.bias.copy(),m.output.copy(),m.output_bias.copy());bad.output[0,0]+=.1
    with pytest.raises(ValueError):assert_unchanged(bad,before)


def test_d0_cohort_identity():
    r=records()[0]
    for bad in [{**r,'condition':'C0_canonical'},{**r,'cohort':'constructive'},{**r,'checkpoint_step':160}]:
        with pytest.raises(ValueError):validate_record(bad)
    model,policy,payload=load_torch(ROOT,r)
    assert payload['policy_state_dict'] is not None and len(list(policy.parameters()))==2
    assert not any('optimizer' in key for key in payload)
    for key,value in model.state_dict().items():np.testing.assert_array_equal(value.detach().numpy(),payload['model_state_dict'][key].numpy())


def test_d0_objective_contract_rejects_token_only_and_wrong_weight():
    from experiments.context_mixing_sequence_continuity.audit import objective_contract
    _,_,p=load_torch(ROOT,records()[0]);c=objective_contract(p)
    assert c['lambda_execution']==.2 and c['optimizer_mode']=='fine_tune_from_weights_optimizer_reset'
    for change in [{'lambda_execution':0.},{'normative_c_time':.1}]:
        bad={**p,'config':{**p['config'],**change}}
        with pytest.raises(ValueError,match='blocked_training_objective_fidelity'):objective_contract(bad)


def test_d0_prior_freeze_only_ignores_finder_metadata(tmp_path):
    import json,hashlib
    from experiments.context_mixing_sequence_continuity.audit import verify_prior_scientific
    (tmp_path/'manifests').mkdir();p=tmp_path/'table.csv';p.write_text('scientific evidence')
    manifest={'inputs':{},'code_hashes':{},'files':{'table.csv':hashlib.sha256(p.read_bytes()).hexdigest()}}
    (tmp_path/'manifests/sealed_manifest.json').write_text(json.dumps(manifest));(tmp_path/'.DS_Store').write_text('Finder metadata')
    assert verify_prior_scientific(tmp_path)[0]['path']=='.DS_Store'
    (tmp_path/'unregistered.csv').write_text('unverified data')
    with pytest.raises(ValueError):verify_prior_scientific(tmp_path)
    (tmp_path/'unregistered.csv').unlink();p.write_text('changed evidence')
    with pytest.raises(ValueError):verify_prior_scientific(tmp_path)
