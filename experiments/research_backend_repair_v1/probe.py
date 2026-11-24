"""Prepare one immutable fixed ping; execute only after exact manual approval."""
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import tempfile
import time

from research_search.common import ROOT,PACKAGE,read,write,sha,fingerprint
from research_search.preflight import verify_artifacts
from research_search.orchestration.agent_backend import parse_events
from research_search.orchestration.budget import usage
from research_search.orchestration.queue import worker_lock
from research_search.orchestration.store import Store
from .backend import create_runtime,run,exec_argv,overrides,rpc_quota

OUT=ROOT/'results/research_backend_repair/v1'
STATE=PACKAGE/'database/backend_validation_v3'
PAYLOAD={'fixture':True}
PROMPT='Connection test only. Return exactly {"fixture":true}. Do not use tools, read files, write code, or perform research.'
SCHEMA={'type':'object','properties':{'fixture':{'type':'boolean','const':True}},'required':['fixture'],'additionalProperties':False}


def prior_unknown():
    path=PACKAGE/'database/backend_validation_v2/campaign.sqlite'
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:
        rows=[dict(x) for x in db.execute("SELECT id,status,reserved,total,cached FROM calls WHERE status!='reconciled'")]
    finally:db.close()
    return rows


def validate_packet(packet):
    expected=dict(version='backend-v3-fixed-ping',old_unknown_reservation=8000,
        old_unknown_usage_status='unknown',maximum_live_model_invocations=1,
        wall_seconds=60,CPU_seconds=60,token_reservation=8000,payload=PAYLOAD,
        prompt=PROMPT,output_schema=SCHEMA,automatic_retries=0,
        clears_prior_unknown_usage=False,unlocks_full_campaign=False,
        hard_token_limit_guaranteed=False,model_training_authorized=False)
    if any(packet.get(k)!=v for k,v in expected.items()):
        raise ValueError('Fixed connection packet changed or old unknown usage erased')
    return packet


def prepare():
    verify_artifacts()
    codex=Path(shutil.which('codex')).resolve()
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    old=prior_unknown()
    packet=dict(version='backend-v3-fixed-ping',old_unknown_reservation=sum(r['reserved'] for r in old),
        preparation_revision=2,supersedes_unapproved_packet_sha=sha(OUT/'PENDING_CONNECTION_TEST.json'),
        old_unknown_usage_status='unknown' if any(r['status']=='unknown' for r in old) else 'changed',
        prior_unresolved_calls=old,maximum_live_model_invocations=1,
        wall_seconds=60,CPU_seconds=60,token_reservation=8000,
        total_retained_and_proposed_reservations=16000,payload=PAYLOAD,prompt=PROMPT,
        output_schema=SCHEMA,automatic_retries=0,clears_prior_unknown_usage=False,
        unlocks_full_campaign=False,hard_token_limit_guaranteed=False,
        model_training_authorized=False,
        runtime='fresh external0700 workspace; local home/state/log/tmp; read-only auth symlink',
        auth='Existing ChatGPT login only; no API key, credential copy or credential writes',
        transmitted_context='Fixed prompt/schema, standard Codex system/client metadata and isolated workspace path only; no research context, history, evaluator, checkpoint or confirmation data',
        approval_scope='One additional diagnostic despite retained prior unknown8000 reservation; does not adjudicate it or authorize a pilot/research campaign',
        read_only_quota_before_generation=True,quota_paid_fallback_requested=False,
        tools='Shell, browser, apps, plugins, hooks, MCP, memories and nested agents disabled',
        native_transport_retries='Native defaults; built-in provider override is rejected by this CLI. One CLI invocation, no wrapper retry, total60s wall bound.',
        code=files,codex_executable=str(codex),codex_sha256=sha(codex),
        native_manifest_sha=sha(PACKAGE/'IMPLEMENTATION_MANIFEST.json'),
        native_evaluator_seal_sha=sha(PACKAGE/'audit/evaluator_seal.json'))
    validate_packet(packet)
    path=OUT/'PENDING_CONNECTION_TEST_v2.json';write(path,packet,True)
    approval=sha(path)
    return dict(packet=packet,approval_sha=approval,packet_path=str(path.relative_to(ROOT)),
        command=f'PYTHONPATH=.:src python -m experiments.research_backend_repair_v1 run --approve {approval}',
        live_test_run=False,scientific_campaign_started=False)


def assess_reply(receipt,events):
    try:
        parsed=parse_events(events)
        total,cached=usage(parsed['usage'])
    except (ValueError,KeyError,TypeError):
        return dict(passed=False,status='missing_or_invalid_usage',reported_usage=None,reported_tokens=None)
    unexpected=any(x.get('type') in ['item.started','item.completed'] and x.get('item',{}).get('type') not in ['agent_message','reasoning'] for x in events)
    try:
        value=json.loads(parsed['text'])
        exact=value==PAYLOAD and type(value.get('fixture')) is bool
    except (ValueError,AttributeError):exact=False
    passed=bool(exact and not parsed['failed'] and not parsed['nested_agent'] and not unexpected and
        receipt['returncode']==0 and not receipt.get('timed_out') and not receipt.get('cancelled') and total<=8000)
    return dict(passed=passed,status='verified_fixed_ping' if passed else 'reply_or_execution_or_reservation_failure',
        reported_usage=parsed['usage'],reported_tokens=total,cached_input_tokens=cached,
        unexpected_tool_item=unexpected,reply_exact=exact)


def run_probe(approval):
    prepared=prepare()
    if not approval or approval!=prepared['approval_sha']:
        raise PermissionError('Explicit manual approval of this exact packet hash is required; no live call dispatched')
    return _execute_approved(prepared['packet'])


def _execute_approved(packet):
    validate_packet(packet);verify_artifacts()
    # Every check precedes network. Never retry a failed/uncertain v3 attempt.
    if STATE.exists():
        raise RuntimeError('v3 state already exists; inspect retained evidence, no automatic retry')
    for name,h in packet['code'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Approved backend code changed')
    codex=Path(packet['codex_executable'])
    if sha(codex)!=packet['codex_sha256']:raise RuntimeError('Approved Codex executable changed')
    if prior_unknown()!=packet['prior_unresolved_calls']:raise RuntimeError('Prior usage state changed; prepare separately versioned evidence')
    limits=dict(tokens=8000,cpu_seconds=60,optimizer_steps=0,wall_seconds=60,proposals=1,active_branches=1)
    with worker_lock(PACKAGE/'database'):
        s=Store(STATE,limits);s.set('phase','approved-fixed-connection-v3');s.set('manual_checkpoints',True);s.control('running')
        write(STATE/'approved_packet.json',packet,True)
        s.reserve_cpu('complete-probe',60)
        before=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
        call='fixed-connection-v3';reserved=False;rt=None;auth_before=None;result=None
        try:
            auth=Path.home()/'.codex/auth.json';auth_before=sha(auth)
            rt=create_runtime(Path(tempfile.mkdtemp(prefix='srn-codex-approved-v3-',dir='/private/tmp')),auth)
            write(STATE/'runtime.json',dict(workspace=str(rt.workspace),auth_read_only=True),True)
            login=run([codex,*overrides(rt),'login','status'],rt,network=False,timeout=5,cpu_seconds=5)
            if login['returncode'] or 'Logged in using ChatGPT' not in login['stdout']+login['stderr']:
                raise RuntimeError('Read-only existing ChatGPT authentication failed')
            quota=rpc_quota(rt,network=True,timeout=min(10,60-(time.monotonic()-start)))
            write(STATE/'quota.json',quota,True)
            if quota.get('ordinary_usage_allowed') is not True or any(b.get('rateLimitReachedType') or b.get('spendControlReached') for b in quota.get('buckets',{}).values()):
                raise RuntimeError('Ordinary included quota unavailable; no fallback and no model dispatched')
            remaining=60-(time.monotonic()-start)
            if remaining<=1:raise RuntimeError('Probe wall allowance exhausted before model dispatch')
            s.reserve_call(call,'connection',8000,speculative=False);reserved=True
            s.set('agent_inflight',dict(id=call,status='dispatched_or_uncertain'))
            schema=rt.workspace/'output-schema.json';write(schema,SCHEMA,True)
            receipt=run(exec_argv(codex,rt,schema),rt,network=True,stdin=PROMPT,
                        timeout=min(45,remaining),cpu_seconds=30)
            for name,text in [('events.jsonl',receipt['stdout']),('stderr.log',receipt['stderr'])]:
                path=rt.workspace/name;path.write_text(text);path.chmod(0o600)
            write(STATE/'process_receipt.json',{k:v for k,v in receipt.items() if k not in ['stdout','stderr']},True)
            try:events=[json.loads(line) for line in receipt['stdout'].splitlines() if line.startswith('{')]
            except ValueError:events=[]
            result=assess_reply(receipt,events)
            s.reconcile_call(call,result['reported_usage'])
            result|=dict(model_invocations=1,raw_event_directory=str(rt.workspace),model_payload=PAYLOAD)
            s.set('agent_inflight',dict(id=call,status='completed' if result['passed'] else 'failed_or_unknown'))
        except BaseException as ex:
            if reserved:
                row=s.db.execute('SELECT status FROM calls WHERE id=?',(call,)).fetchone()
                if row['status']=='reserved':s.reconcile_call(call,None)
            result=dict(passed=False,status='stopped_after_error',error_type=type(ex).__name__,
                error=str(ex)[:256],reported_usage=None,reported_tokens=None,model_invocations=int(reserved))
        finally:
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu=time.process_time()-before+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime
            s.reconcile_cpu('complete-probe',cpu)
            if s.get('status') not in ['budget_exhausted','manual_checkpoint']:
                s.control('manual_checkpoint')
            unchanged=bool(auth_before is not None and sha(Path.home()/'.codex/auth.json')==auth_before)
            result|=dict(wall_seconds=time.monotonic()-start,CPU_seconds=cpu,
                original_auth_unchanged=unchanged,scientific_optimizer_updates=0,
                prior_unknown_reservation_unchanged=prior_unknown()==packet['prior_unresolved_calls'],
                full_campaign_unlocked=False,hard_token_limit_guaranteed=False,ledger=s.state())
            if not unchanged or not result['prior_unknown_reservation_unchanged']:
                result['passed']=False;result['status']='protected_source_changed'
            write(STATE/'receipt.json',result,True)
            if result['passed']:write(STATE/'proof.json',result,True)
            s.close()
    return result
