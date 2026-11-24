"""One individually approved Codex call with a shared pilot ledger."""
import json,resource,shutil,tempfile,time
from pathlib import Path
from research_search.common import ROOT,PACKAGE,read,write,sha
from research_search.orchestration.queue import worker_lock
from experiments.research_backend_repair_v1.backend import create_runtime,run,exec_argv,overrides,rpc_quota,app_server_argv
from experiments.research_search_pilot_v2.events import assess
from .protocol import RESERVATIONS
from .executable import resolve_codex


def invoke(store,ticket,approval):
    path=store.directory/'requests_agent'/f'{ticket}.json'
    request=read(path)
    CALL_RESERVATION=request['reservation']
    if request['role'] not in RESERVATIONS or CALL_RESERVATION!=RESERVATIONS[request['role']]:raise ValueError('Unregistered call reservation')
    if not approval or approval!=sha(path):raise PermissionError('Exact individual manual approval required before any network call')
    if store.get('phase')!=request['phase']:raise RuntimeError('Approval belongs to another pilot phase')
    if store.db.execute('SELECT 1 FROM calls WHERE id=?',(ticket,)).fetchone():raise RuntimeError('Attempt already exists; never retry silently')
    if len(request['prompt'])>16000:raise ValueError('Pilot prompt exceeds registered compact-context size')
    with worker_lock(PACKAGE/'database'):
        store.control('running');store.check();store.reserve_cpu(ticket,90)
        start=time.monotonic();before=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
        reserved=False;result=None;auth=Path.home()/'.codex/auth.json';original_auth=sha(auth)
        try:
            codex=resolve_codex(request['codex_sha256'],explicit=request['codex_path'])
            if sha(codex)!=request['codex_sha256']:raise RuntimeError('Approved CLI changed')
            rt=create_runtime(Path(tempfile.mkdtemp(prefix='srn-integration11-'+ticket+'-',dir='/private/tmp')),auth)
            login=run([codex,*overrides(rt),'login','status'],rt,network=False,timeout=5,cpu_seconds=5)
            if login['returncode'] or 'Logged in using ChatGPT' not in login['stdout']+login['stderr']:raise RuntimeError('Existing read-only ChatGPT login unavailable')
            quota=rpc_quota(rt,command=app_server_argv(codex,rt),network=True,timeout=10);write(store.directory/'agent_evidence'/f'{ticket}-quota.json',quota,True)
            if quota.get('ordinary_usage_allowed') is not True or any(b.get('rateLimitReachedType') or b.get('spendControlReached') for b in quota.get('buckets',{}).values()):raise RuntimeError('Ordinary included allowance unavailable; no paid fallback')
            store.reserve_call(ticket,'pilot_agent',CALL_RESERVATION,speculative=False);reserved=True
            store.set('agent_inflight',dict(id=ticket,status='dispatched_or_uncertain',role=request['role']))
            schema=rt.workspace/'reply-schema.json';write(schema,request['schema'],True)
            remaining=store.limits['wall_seconds']-(time.time()-store.get('started'))
            if remaining<=1:raise RuntimeError('Pilot wall limit reached before dispatch')
            receipt=run(exec_argv(codex,rt,schema),rt,network=True,stdin=request['prompt'],timeout=min(300,remaining),cpu_seconds=60,cancel_check=lambda:store.get('status') in ['paused','stopped'])
            for name,text in [('events.jsonl',receipt['stdout']),('stderr.log',receipt['stderr'])]:
                p=rt.workspace/name;p.write_text(text);p.chmod(0o600)
            try:events=[json.loads(x) for x in receipt['stdout'].splitlines() if x.startswith('{')]
            except ValueError:events=[]
            result=assess(receipt,events,CALL_RESERVATION)
            store.reconcile_call(ticket,result['reported_usage'])
            result|=dict(call_id=ticket,role=request['role'],raw_event_directory=str(rt.workspace),process={k:v for k,v in receipt.items() if k not in ['stdout','stderr']})
            store.set('agent_inflight',dict(id=ticket,status='completed' if result['passed'] else 'failed_or_unknown',role=request['role']))
        except BaseException as ex:
            if reserved:
                row=store.db.execute('SELECT status FROM calls WHERE id=?',(ticket,)).fetchone()
                if row['status']=='reserved':store.reconcile_call(ticket,None)
            result=dict(passed=False,status='stopped_after_error',error_type=type(ex).__name__,error=str(ex)[:400],reported_tokens=None,reported_usage=None,call_id=ticket)
        finally:
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu=time.process_time()-before+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime
            store.reconcile_cpu(ticket,cpu)
            if store.get('status') not in ['stopped','budget_exhausted','manual_checkpoint']:store.control('manual_checkpoint')
            result|=dict(wall_seconds=time.monotonic()-start,CPU_seconds=cpu,original_auth_unchanged=sha(auth)==original_auth,ledger=store.state())
            if not result['original_auth_unchanged']:result['passed']=False;result['status']='authentication_changed'
            write(store.directory/'agent_evidence'/f'{ticket}.json',result,True)
    return result
