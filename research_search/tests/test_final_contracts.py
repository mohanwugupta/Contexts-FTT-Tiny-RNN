import sys
import pytest

from research_search.orchestration.quota import read_quota
from research_search.orchestration.store import Store
from research_search.orchestration.budget import BudgetError
from research_search.orchestration.agent_backend import invoke


def test_quota_framing_drains_buffered_notifications_without_another_write(tmp_path):
    """A local dummy process, never authenticated/external service traffic."""
    script = tmp_path / 'fake-codex'
    script.write_text('#!' + sys.executable + '''
import json,sys,os,time
for line in sys.stdin:
    request=json.loads(line)
    if request['method']=='initialize':
        os.write(1,b'{"id":0,"result":{}}\\n{"method":"notification"}\\n')
    if request['method']=='account/rateLimits/read':
        os.write(1,b'{"method":"notification"}\\n{"id":1,"result":{"ordinaryUsageAllowed":true}}\\n')
        time.sleep(2)
''')
    script.chmod(0o700)
    result = read_quota(str(script), {}, timeout=1)
    assert result['available'], result
    assert result['ordinary_usage_allowed'] is True
    assert result['paid_fallback_requested'] is False


def test_exhausted_campaign_cannot_be_restarted(tmp_path):
    s = Store(tmp_path, dict(tokens=1000,cpu_seconds=10,optimizer_steps=3,wall_seconds=100,proposals=2,active_branches=2))
    try:
        s.control('budget_exhausted')
        with pytest.raises(BudgetError):
            s.control('running')
        s.control('stopped')
    finally:
        s.close()


def test_confirmation_blocks_backend_before_authentication(tmp_path, monkeypatch):
    s = Store(tmp_path, dict(tokens=1000,cpu_seconds=10,optimizer_steps=3,wall_seconds=100,proposals=2,active_branches=2))
    try:
        s.set('confirmation_started', True)
        def forbidden():
            pytest.fail('Authentication attempted after confirmation')
        monkeypatch.setattr('research_search.orchestration.agent_backend.authenticated', forbidden)
        with pytest.raises(RuntimeError, match='No further research-agent'):
            invoke(s, 'critic', 'must never send')
        assert s.totals()['tokens'] == 0
    finally:
        s.close()


def test_programmatic_start_also_requires_live_backend_proof(tmp_path, monkeypatch):
    from research_search import campaign, preflight
    s = Store(tmp_path, dict(tokens=1000,cpu_seconds=10,optimizer_steps=3,wall_seconds=100,proposals=2,active_branches=2))
    try:
        monkeypatch.setattr(campaign, 'integrity', lambda store: None)
        monkeypatch.setattr(preflight, 'readiness', lambda: dict(infrastructure_passed=True,live_backend_verified=False,ready_for_scientific_launch=False))
        def forbidden():
            pytest.fail('Authentication attempted before backend validation')
        monkeypatch.setattr('research_search.orchestration.agent_backend.authenticated', forbidden)
        with pytest.raises(RuntimeError, match='backend'):
            campaign.start(s, manual=True)
        assert s.get('status') == 'ready'
    finally:
        s.close()
