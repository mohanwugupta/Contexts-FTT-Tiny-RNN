import pytest
from experiments.cloud_confidence_selected_v1.driver import deadline_seconds
from experiments.cloud_confidence_selected_v1.sandbox import lockdown


def test_deadline_is_explicit_and_not_reset():
    assert deadline_seconds('2026-11-25T09:00:00Z',now=1795597100)==100
    assert deadline_seconds('2026-11-25T09:00:00Z',now=1795597200)==0
    with pytest.raises(ValueError):deadline_seconds('2026-11-25T09:00:00')


def test_unsupported_security_platform_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr('experiments.cloud_confidence_selected_v1.sandbox.platform.system',lambda:'Darwin')
    with pytest.raises(RuntimeError,match='Linux'):lockdown(tmp_path,tmp_path,tmp_path)
