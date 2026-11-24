import pytest
from research_search.common import write
from research_search.orchestration.store import Store
from experiments.research_search_pilot_v2.protocol import LIMITS
from experiments.research_search_pilot_v2.agent import invoke


def test_individual_approval_is_checked_before_any_dispatch(tmp_path,monkeypatch):
    s=Store(tmp_path,LIMITS)
    write(tmp_path/'requests_agent/proposal.json',dict(phase='proposal_pending',prompt='fixture',role='proposal'))
    def forbidden(*args,**kw):pytest.fail('A call was dispatched without individual approval')
    monkeypatch.setattr('experiments.research_search_pilot_v2.agent.create_runtime',forbidden)
    try:
        with pytest.raises(PermissionError):invoke(s,'proposal',None)
        with pytest.raises(PermissionError):invoke(s,'proposal','wrong')
        assert s.get('started') is None and s.totals()['tokens']==0
    finally:s.close()
