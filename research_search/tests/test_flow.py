import copy,json
import pytest
from research_search.common import PACKAGE,read
from research_search.orchestration.store import Store
from research_search.orchestration import coordinator as c
from research_search.runners.candidate import DEFAULT

def test_three_roles_admission_is_durable_and_does_not_train(tmp_path,monkeypatch):
    s=Store(tmp_path,dict(tokens=100000,cpu_seconds=1000,optimizer_steps=100000,wall_seconds=100,proposals=3,active_branches=2));p=dict(id='one',parent_candidate=None,research_area='architecture',hypothesis='A fixed output initialization may alter later ordinary learning',mechanism=dict(readout=.25),proposed_change='initial target head only',predicted_behavior='C accuracy may improve',alternative_predictions=['calibration only','changed predictive path'],matched_control='baseline',required_code_changes=[],estimated_training_steps=1,estimated_runtime=1,estimated_agent_tokens=1000,evaluation_plan='all4 development tasks and2 seeds',stop_conditions='registered limits',novelty_justification='Different held-out sequence bank, not a new scale claim',config=copy.deepcopy(DEFAULT),ablation_primary=dict(metric='C_ce',direction='min'));p['config']['target']['readout_initial_multiplier']=.25
    original=c.read;monkeypatch.setattr(c,'read',lambda path:dict(runner_hash='fixture-hash') if path.name=='evaluator_seal.json' else original(path));called=[]
    def fake(store,role,prompt,kind='proposal',context=None):
        called.append(role);r=p if len(called)==1 else dict(accept=True,explanations=['one','two']) if role=='critic' else dict(valid=True);return dict(text=json.dumps(r))
    monkeypatch.setattr(c,'invoke',fake)
    for _ in range(6):c.advance(s)
    assert called==['architecture','critic','architecture'] and s.get('phase')=='evaluate'
    assert len(s.nodes())==1 and s.totals()['optimizer_steps']==0

def test_stopping_is_not_undone_by_missing_usage(tmp_path):
    s=Store(tmp_path,dict(tokens=1000,cpu_seconds=10,optimizer_steps=3,wall_seconds=100,proposals=2,active_branches=2));s.reserve_call('x','hypothesis',100);s.control('stopped');s.reconcile_call('x',None)
    assert s.get('status')=='stopped'
    with pytest.raises(Exception):s.control('running')

def test_confirmation_seed_private_registration_not_fixture_seed():
    from research_search.evaluation.development import make_banks
    a,b=make_banks(123);aa,bb=make_banks(456)
    assert a==aa and b!=bb and set(a['seeds']).isdisjoint(b['seeds'])
