from pathlib import Path
import pytest
from experiments.research_search_cloud_v1.port import permitted,verify,gate_agent_usage


def test_export_allowlist_blocks_sensitive_and_bulk():
    assert permitted('experiments/research_search_cloud_v1/port.py',set())
    assert permitted('research_search/database/controlled_pilot_v2/jobs/j/metrics.csv',{'research_search/database/controlled_pilot_v2/jobs/j/metrics.csv'})
    for p in ['.codex/auth.json','research_search/database/benchmark_banks/sealed/confirmation.json','research_search/database/controlled_pilot_v2/jobs/j/states.pt','data/raw.csv','../outside.py','research_search/database/controlled_pilot_v2/agent_evidence/raw.json','docs/key.pem']:
        assert not permitted(p,{p}),p


def test_hash_tamper_and_escape_detected(tmp_path):
    import hashlib,json
    p=tmp_path/'docs/a.md';p.parent.mkdir();p.write_text('saved')
    manifest={'files':{'docs/a.md':hashlib.sha256(p.read_bytes()).hexdigest()}}
    verify(tmp_path,manifest)
    p.write_text('changed')
    with pytest.raises(ValueError):verify(tmp_path,manifest)
    with pytest.raises(ValueError):verify(tmp_path,{'files':{'../escape':'x'}})
    p.unlink();p.symlink_to(tmp_path.parent/'outside')
    with pytest.raises(ValueError):verify(tmp_path,manifest)


def test_missing_remote_telemetry_and_unapproved_calls_stay_blocked():
    with pytest.raises(ValueError):gate_agent_usage(None,100000)
    with pytest.raises(ValueError):gate_agent_usage({'input_tokens':100,'output_tokens':10},100000)
    assert gate_agent_usage({'input_tokens':100,'output_tokens':10,'cached_input_tokens':0},100000)==110
    with pytest.raises(ValueError):gate_agent_usage({'input_tokens':100001,'output_tokens':10,'cached_input_tokens':0},100000)
