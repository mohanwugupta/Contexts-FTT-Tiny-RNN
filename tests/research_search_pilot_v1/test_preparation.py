import pytest
from research_search.common import PACKAGE
from experiments.research_search_pilot_v1.control import definition, prepare, initialize, gate


def test_separate_pilot_limits_and_native_science():
    d=definition()
    assert d['limits']==dict(tokens=50000,cpu_seconds=3600,optimizer_steps=6000,wall_seconds=7200,proposals=2,active_branches=1)
    assert d['target_updates']==330 and d['development_seeds']==[21000,21001]
    assert d['confirmation_authorized'] is False
    assert d['manual_approval_each_agent_call'] is True
    assert d['prior_episodes']==0


def test_prepare_never_creates_operational_campaign(tmp_path):
    r=prepare(tmp_path)
    assert r['operational_campaign_created'] is False
    assert (tmp_path/'PILOT_DEFINITION.json').exists()
    assert not list(tmp_path.rglob('*.sqlite'))


def test_initialize_blocks_before_database_on_failed_connection(tmp_path):
    ready=dict(live_backend_verified=False)
    with pytest.raises(RuntimeError,match='connection'):
        initialize(tmp_path/'campaign',readiness=ready,unresolved=[],hard_token_cap_verified=False)
    assert not (tmp_path/'campaign').exists()


def test_unresolved_usage_cannot_be_waived_by_proof(tmp_path):
    with pytest.raises(RuntimeError,match='unresolved'):
        gate(dict(live_backend_verified=True),[dict(id='old',status='unknown')],True)


def test_strict_ceiling_not_claimed_without_provider_enforcement():
    with pytest.raises(RuntimeError,match='hard'):
        gate(dict(live_backend_verified=True),[],False)


def test_native_sealed_package_does_not_contain_pilot_override():
    assert not (PACKAGE/'config/pilot.yaml').exists()
