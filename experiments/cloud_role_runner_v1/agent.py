"""Drop-in invoke(store,ticket,approval,runtime) using existing requests/ledger.

Does not advance scientific phases, start a clock, initialize budgets or dispatch
another call. The versioned rehearsal controller must accept the returned reply.
"""
import json,os,time
from pathlib import Path
from .contract import ROLES,auth_bytes,remaining,sha,save
from .docker import launch
from .accounting import assess

def approval_value(store,ticket,runtime):
 from research_search.common import identifier
 remaining(store.get('started'),store.limits['wall_seconds'])
 path=store.directory/'requests_agent'/(identifier(ticket)+'.json')
 rtpath=Path(runtime['directory'])/'RUNTIME.json'
 if json.loads(rtpath.read_text())!=runtime:raise ValueError('Runtime definition changed')
 return dict(version='linux-role-approval-v1',ticket=ticket,request_sha256=sha(path),runtime_sha256=sha(rtpath),definition_sha256=sha(store.directory/'definition.json'),started=store.get('started'),deadline_epoch=store.get('started')+store.limits['wall_seconds'],limits=store.limits)

def prepare_approval(store,ticket,runtime):
 value=approval_value(store,ticket,runtime);path=store.directory/'requests_linux'/(ticket+'.json')
 if path.exists():
  if json.loads(path.read_text())!=value:raise ValueError('Approval envelope already exists with different inputs')
 else:save(path,value)
 return dict(ticket=ticket,approval_file=str(path),approval_sha=sha(path),reservation=json.loads((store.directory/'requests_agent'/(ticket+'.json')).read_text())['reservation'],deadline_epoch=value['deadline_epoch'],new_agent_calls=0)

def validate_request(store,ticket,approval,runtime):
 from research_search.common import identifier
 path=store.directory/'requests_agent'/(identifier(ticket)+'.json')
 envelope=store.directory/'requests_linux'/(identifier(ticket)+'.json')
 if not approval or not envelope.exists() or sha(envelope)!=approval:raise PermissionError('Exact Linux transport envelope approval required; old request approval is insufficient')
 if json.loads(envelope.read_text())!=approval_value(store,ticket,runtime):raise ValueError('Approved request/runtime/deadline/limits changed')
 request=json.loads(path.read_text())
 if request['ticket']!=ticket or request['role'] not in ROLES or request['phase']!=store.get('phase'):raise ValueError('Role/phase/ticket changed')
 if request.get('definition_sha')!=sha(store.directory/'definition.json'):raise ValueError('Rehearsal definition changed')
 if request.get('codex_sha256')!=runtime.get('codex_sha256'):raise ValueError('Approved CLI identity changed; no silent upgrade')
 if runtime.get('synthetic'):raise ValueError('Fixture runtime cannot make live calls')
 if not isinstance(request.get('prompt'),str) or len(request['prompt'])>16000:raise ValueError('Approved compact prompt required')
 if type(request.get('reservation')) is not int or request['reservation']<=0:raise ValueError('Registered positive reservation required')
 def check_schema(v):
  if isinstance(v,dict):
   if '$ref' in v:raise ValueError('Remote/indirect schema references prohibited')
   for x in v.values():check_schema(x)
  if isinstance(v,list):
   for x in v:check_schema(x)
 check_schema(request['schema'])
 remaining(store.get('started'),store.limits['wall_seconds'])
 if store.get('confirmation_started') or not store.get('manual_checkpoints'):raise ValueError('Manual pre-confirmation rehearsal required')
 if store.db.execute('SELECT 1 FROM calls WHERE id=?',(ticket,)).fetchone():raise RuntimeError('Existing call requires adjudication; no retry')
 if (store.directory/'linux_role_attempts'/ticket).exists():raise RuntimeError('Prior Linux attempt exists; no automatic retry')
 return request


def invoke(store,ticket,approval,runtime,auth_file=None):
 request=validate_request(store,ticket,approval,runtime)
 from research_search.orchestration.queue import worker_lock
 directory=store.directory/'linux_role_attempts'/ticket
 directory.mkdir(parents=True,mode=0o700)
 auth_file=Path(auth_file or Path.home()/'.codex/auth.json').resolve()
 if not auth_file.is_file():raise ValueError('Existing ChatGPT file-based login unavailable')
 # Never print or persist authentication. Only its identity is retained.
 auth=auth_bytes(auth_file.read_bytes());before_auth=sha(auth_file)
 save(directory/'APPROVED.json',dict(request_sha256=approval,runtime_sha256=sha(Path(runtime['directory'])/'RUNTIME.json'),codex_sha256=runtime['codex_sha256'],existing_started=store.get('started'),deadline_epoch=store.get('started')+store.limits['wall_seconds'],reservation=request['reservation']))
 cpu_id='linux-'+ticket;start=time.process_time();reserved=False;cpu_reserved=False;measurements=[];result=None
 with worker_lock(Path(runtime['directory'])):
  try:
   store.control('running');store.check();store.reserve_cpu(cpu_id,540);cpu_reserved=True
   left=lambda: min(300,remaining(store.get('started'),store.limits['wall_seconds']))
   cancelled=lambda:store.get('status') in ['paused','stopped','budget_exhausted']
   # Read-only login/quota inside the SAME enforced role transport, not host CLI.
   login=launch(runtime,directory/'login','login',{},auth,timeout=min(20,left()),cancel_check=cancelled);measurements.append(login)
   if not login['passed'] or 'Logged in using ChatGPT' not in login.get('stdout','')+login.get('stderr',''):raise RuntimeError('Isolated existing ChatGPT login unavailable')
   quota=launch(runtime,directory/'quota','quota',{},auth,timeout=min(20,left()),cancel_check=cancelled);measurements.append(quota)
   value=quota.get('process',{}).get('quota',{})
   if not quota['passed'] or value.get('ordinary_usage_allowed') is not True or any(b.get('rateLimitReachedType') or b.get('spendControlReached') for b in value.get('buckets',{}).values()):raise RuntimeError('Ordinary included quota unavailable; no paid fallback')
   save(directory/'QUOTA.json',value)
   store.reserve_call(ticket,'pilot_agent',request['reservation'],speculative=False);reserved=True
   store.set('agent_inflight',dict(id=ticket,role=request['role'],status='dispatched_or_uncertain'))
   receipt=launch(runtime,directory/'exec','exec',request['schema'],auth,request['prompt'],left(),cancelled);measurements.append(receipt)
   try:events=[json.loads(x) for x in receipt.get('stdout','').splitlines() if x.startswith('{')]
   except ValueError:events=[]
   result=assess(events,receipt.get('returncode',1),request['reservation'],receipt.get('timed_out',False))
   store.reconcile_call(ticket,result['reported_usage'])
   if result['passed']:
    import jsonschema
    jsonschema.validate(result['reply'],request['schema'])
   result['passed']=bool(result['passed'] and receipt['passed'] and store.get('status')!='budget_exhausted')
   result.update(call_id=ticket,role=request['role'],new_agent_calls=1,live_usage_verified=result['reported_usage'] is not None,role_isolation_verified=bool(receipt.get('isolation',{}).get('passed')),parent_agent_isolation_certified=False)
   store.set('agent_inflight',dict(id=ticket,role=request['role'],status='completed' if result['passed'] else 'failed_or_unknown'))
  except BaseException as error:
   if reserved:
    row=store.db.execute('SELECT status FROM calls WHERE id=?',(ticket,)).fetchone()
    if row and row['status']=='reserved':store.reconcile_call(ticket,None)
   result=(result or dict(reported_usage=None,reported_tokens=None))|dict(passed=False,status='stopped_after_error',error_type=type(error).__name__,reason=str(error)[:300],model_dispatch_attempted=reserved,new_agent_calls=1 if result and result.get('reported_usage') else None if reserved else 0,parent_agent_isolation_certified=False)
  finally:
   actual_cpu=time.process_time()-start
   complete_cpu=True
   for m in measurements:
    if m.get('CPU_seconds') is None or m.get('relay_CPU_seconds') is None:complete_cpu=False
    else:actual_cpu+=m['CPU_seconds']+m['relay_CPU_seconds']
   if cpu_reserved and complete_cpu:store.reconcile_cpu(cpu_id,actual_cpu)
   elif cpu_reserved:result.update(passed=False,CPU_seconds=None,CPU_reservation_held=True)
   if sha(auth_file)!=before_auth:result.update(passed=False,status='host_authentication_changed')
   if store.get('status') not in ['stopped','budget_exhausted','manual_checkpoint']:store.control('manual_checkpoint')
   result.update(original_auth_unchanged=sha(auth_file)==before_auth,CPU_seconds=actual_cpu if complete_cpu else None,ledger=store.state(),requires_manual_checkpoint=True)
   save(directory/'RESULT.json',result)
 return result
