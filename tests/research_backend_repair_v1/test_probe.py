import pytest
from experiments.research_backend_repair_v1.probe import run_probe,assess_reply,validate_packet


def test_no_live_call_or_directory_without_exact_approval(tmp_path,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('Live dispatch before approval')
    monkeypatch.setattr('experiments.research_backend_repair_v1.probe.prepare',lambda:dict(approval_sha='expected'))
    monkeypatch.setattr('experiments.research_backend_repair_v1.probe._execute_approved',forbidden)
    with pytest.raises(PermissionError):run_probe(None)
    with pytest.raises(PermissionError):run_probe('wrong')


def test_missing_or_unexpected_usage_and_tools_cannot_pass():
    basic=dict(returncode=0,timed_out=False,cancelled=False)
    assert not assess_reply(basic,[])['passed']
    records=[dict(type='turn.completed',usage=dict(input_tokens=10,output_tokens=5,cached_input_tokens=0)),dict(type='item.completed',item=dict(type='agent_message',text='{"fixture":true}'))]
    assert assess_reply(basic,records)['passed']
    assert not assess_reply(basic,records+[dict(type='item.started',item=dict(type='command_execution'))])['passed']
    assert not assess_reply(basic|dict(returncode=1),records)['passed']
    records[0]['usage']['output_tokens']=10000
    assert not assess_reply(basic,records)['passed']


def test_packet_does_not_silently_clear_old_unknown_usage():
    with pytest.raises(ValueError):validate_packet(dict(old_unknown_reservation=0))


@pytest.mark.parametrize('missing_usage',[False,True,'interrupt'])
def test_complete_probe_fixture_and_no_retry(tmp_path,monkeypatch,missing_usage):
    import contextlib,json
    from pathlib import Path
    from experiments.research_backend_repair_v1 import probe as p
    from research_search.common import sha
    original=tmp_path/'original';(original/'.codex').mkdir(parents=True)
    auth=original/'.codex/auth.json';auth.write_text('synthetic-auth-no-real-credentials')
    executable=tmp_path/'fake-codex';executable.write_text('never executed')
    old=[dict(id='historical-call',status='unknown',reserved=8000,total=None,cached=None)]
    packet=dict(version='backend-v3-fixed-ping',old_unknown_reservation=8000,old_unknown_usage_status='unknown',
        maximum_live_model_invocations=1,wall_seconds=60,CPU_seconds=60,token_reservation=8000,
        payload={'fixture':True},prompt=p.PROMPT,output_schema=p.SCHEMA,automatic_retries=0,
        clears_prior_unknown_usage=False,unlocks_full_campaign=False,hard_token_limit_guaranteed=False,
        model_training_authorized=False,code={},codex_executable=str(executable),codex_sha256=sha(executable),prior_unresolved_calls=old)
    monkeypatch.setattr(p,'STATE',tmp_path/'state')
    monkeypatch.setattr(p,'verify_artifacts',lambda:dict(passed=True))
    monkeypatch.setattr(p,'prior_unknown',lambda:old)
    monkeypatch.setattr(p,'worker_lock',lambda _:contextlib.nullcontext())
    monkeypatch.setattr(Path,'home',staticmethod(lambda:original))
    monkeypatch.setattr(p.tempfile,'mkdtemp',lambda **kw:str(tmp_path/'runtime'))
    monkeypatch.setattr(p,'rpc_quota',lambda *a,**kw:dict(ordinary_usage_allowed=True,buckets={}))
    calls=[]
    def fake(command,*args,**kwargs):
        if 'login' in command:return dict(returncode=0,stdout='',stderr='Logged in using ChatGPT')
        calls.append(command)
        if missing_usage=='interrupt':raise KeyboardInterrupt('synthetic local cancellation')
        records=[dict(type='item.completed',item=dict(type='agent_message',text='{"fixture":true}'))]
        if not missing_usage:records.append(dict(type='turn.completed',usage=dict(input_tokens=10,output_tokens=5,cached_input_tokens=0)))
        return dict(returncode=0,stdout='\n'.join(json.dumps(r) for r in records),stderr='',timed_out=False,cancelled=False)
    monkeypatch.setattr(p,'run',fake)
    result=p._execute_approved(packet)
    assert len(calls)==1 and result['passed']==(not missing_usage)
    assert result['original_auth_unchanged'] and result['prior_unknown_reservation_unchanged']
    assert result['ledger']['totals']['optimizer_steps']==0
    assert result['ledger']['totals']['reserved_tokens']==(8000 if missing_usage else 0)
    assert result['reported_tokens']==(None if missing_usage else 15) # Synthetic, not real usage.
    with pytest.raises(RuntimeError,match='already exists'):p._execute_approved(packet)
