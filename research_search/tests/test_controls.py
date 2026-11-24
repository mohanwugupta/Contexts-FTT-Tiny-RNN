import copy,json
import pytest
from research_search.orchestration.store import Store
from research_search.orchestration.budget import BudgetError,usage
from research_search.orchestration.queue import enqueue
from research_search.evaluation.confirmation import commit,agent_view

def limits():return dict(tokens=1000,cpu_seconds=10,optimizer_steps=3,wall_seconds=100,proposals=2,active_branches=2)

def test_queue_reserves_whole_jobs_atomically(tmp_path):
    s=Store(tmp_path,limits())
    with pytest.raises(BudgetError):enqueue(s,'large','x',dict(kind='target'))
    assert not s.db.execute('SELECT 1 FROM jobs').fetchone()
    enqueue(s,'small','x',dict(kind='target',fixture_steps=2))
    assert s.totals()['reserved_optimizer_steps']==2
    with pytest.raises(BudgetError):enqueue(s,'second','x',dict(kind='target',fixture_steps=2))

def test_no_skipped_step_indices(tmp_path):
    s=Store(tmp_path,limits())
    with pytest.raises(BudgetError):s.charge_step('job',2)

def test_confirmation_once_and_no_results_in_agent_context(tmp_path):
    s=Store(tmp_path,limits());plan=dict(frozen=True,development_qualified=True,ablation_supported=True)
    commit(s,'family','hash',plan)
    with pytest.raises(ValueError):commit(s,'family','hash',plan)
    with pytest.raises(ValueError):commit(s,'other','hash',dict(plan,ablation_supported=False))
    assert agent_view([dict(phase='confirmation',secret=1),dict(phase='development',value=2)])==[dict(phase='development',value=2)]

def test_provider_aggregate_preferred_without_adding_breakdowns():
    assert usage(dict(input_tokens=100,output_tokens=50,cached_input_tokens=30,total_tokens=170,reasoning_tokens=20))==(170,30)
    with pytest.raises(ValueError):usage(dict(input_tokens=100,output_tokens=50,cached_input_tokens=30,total_tokens=120))

def test_abandoned_or_edited_mechanism_cannot_evade_duplicate(tmp_path):
    s=Store(tmp_path,limits());p=dict(id='one',parent_candidate=None,config=dict(x=1),mechanism=dict(label='a'));s.register(p,'hash')
    with pytest.raises(ValueError):s.register(dict(p,id='two',mechanism=dict(label='another label')),'hash')

def test_repair_and_revision_limits_are_durable(tmp_path):
    s=Store(tmp_path,limits());s.register(dict(id='one',parent_candidate=None,config={},mechanism={}), 'hash')
    s.repair('one');s.repair('one')
    with pytest.raises(BudgetError):s.repair('one')
    s.repair('one',True)
    with pytest.raises(BudgetError):s.repair('one',True)

def test_manual_checkpoints_required_before_launch(monkeypatch,tmp_path):
    from research_search.orchestration import agent_backend as a
    monkeypatch.setattr(a,'authenticated',lambda:'codex')
    s=Store(tmp_path,limits())
    with pytest.raises(RuntimeError,match='manual'):a.invoke(s,'architecture','x')
    assert s.totals()['tokens']==0 and not s.totals()['reserved_tokens']
