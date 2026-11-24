"""Explicit separately approved non-scientific live validation, never auto-run."""
from .common import *

def run():
    from .preflight import verify_seal
    from .orchestration.store import Store
    from .orchestration.agent_backend import invoke
    verify_seal();dest=PACKAGE/'runtime_validation/backend_v2';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'proof.json').exists():return read(dest/'proof.json')
    state=PACKAGE/'database/backend_validation_v2';limits=dict(tokens=100000,cpu_seconds=120,optimizer_steps=0,wall_seconds=60,proposals=1,active_branches=1);s=Store(state,limits)
    if s.db.execute('SELECT 1 FROM calls').fetchone():raise RuntimeError('Existing probe call requires adjudication; no automatic retry')
    s.set('phase','non-scientific-subscription-probe');s.set('manual_checkpoints',True);s.control('running')
    try:
        result=invoke(s,'coordinator','Non-scientific fixture ping. Return only {"fixture":true}. No tools, code, model training or research proposal.',kind='coordinator',context=dict(fixture=True))
        parsed=json.loads(result['text']);passed=parsed=={'fixture':True};proof=dict(passed=passed,status='verified_non_scientific_ping' if passed else 'invalid_fixed_reply',runner_hash=read(PACKAGE/'audit/evaluator_seal.json')['runner_hash'],real_agent_fixture_calls=1,scientific_experiments=0,reported_usage=result['usage'],call_id=result['call_id'],API_key_fallback=False,raw_event_directory=str(state/'workspaces'/result['call_id']));write(dest/'proof.json',proof,True);return proof
    finally:s.close()
