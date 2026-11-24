import pytest
from research_search.common import write,sha
from research_search.orchestration.store import Store
from experiments.research_search_integration_v1.protocol import LIMITS
from experiments.research_search_integration_v1.agent import invoke
from experiments.research_search_integration_v1.controller import accept_reply, integrity


def test_missing_or_wrong_individual_approval_never_dispatches(tmp_path,monkeypatch):
    s=Store(tmp_path,LIMITS);s.set('phase','specialist_initialization')
    write(tmp_path/'requests_agent/fixture.json',dict(phase='specialist_initialization',prompt='fixture',role='specialist_initialization',reservation=19000))
    monkeypatch.setattr('experiments.research_search_integration_v1.agent.create_runtime',lambda *a:pytest.fail('unapproved dispatch'))
    with pytest.raises(PermissionError):invoke(s,'fixture',None)
    with pytest.raises(PermissionError):invoke(s,'fixture','wrong')
    assert s.get('started') is None and s.totals()['tokens']==0
    s.close()


def test_critic_cannot_launch_and_review_requires_stop(tmp_path):
    s=Store(tmp_path,LIMITS);s.set('phase','critic')
    critic=dict(failed_outcomes=['C'],competing_explanations=['confidence','interference'],scientific_usefulness='cached',mechanistic_limits='no intervention',cheapest_followup='design only',recommendation='inconclusive')
    accept_reply(s,'critic',critic)
    assert s.get('phase')=='coordinator_review'
    assert not s.db.execute('SELECT 1 FROM jobs').fetchone()
    review=dict(decision='inconclusive',evidence='cached only',critic_response='agreed',next_step_design_only='estimate first',readiness_limits=['no training'],stop_now=True)
    accept_reply(s,'coordinator_review',review)
    assert s.get('status')=='stopped' and s.get('phase')=='complete'
    with pytest.raises(Exception):s.control('running')
    s.close()


def test_unregistered_reply_does_not_advance(tmp_path):
    s=Store(tmp_path,LIMITS);s.set('phase','critic')
    with pytest.raises(ValueError):accept_reply(s,'critic',dict(launch_training=True))
    assert s.get('phase')=='critic'
    s.close()


def test_live_adapter_retains_isolation_and_usage_contract():
    import inspect
    from experiments.research_search_pilot_v2.agent import invoke as old
    a=inspect.getsource(invoke)
    addition="\n    CALL_RESERVATION=request['reservation']\n    if request['role'] not in RESERVATIONS or CALL_RESERVATION!=RESERVATIONS[request['role']]:raise ValueError('Unregistered call reservation')"
    a=a.replace(addition,'').replace("'srn-integration1-'","'srn-pilot2-'")
    assert a==inspect.getsource(old)
