"""On-demand authenticated Codex CLI calls, fail-closed usage and isolation."""
import json,os,shutil,subprocess,uuid
import time
from research_search.common import PACKAGE,ROOT,write,identifier,encoded
from .budget import CALL_LIMITS
from .isolation import run,safe_workspace
from .queue import worker_lock

REQUIRED=['id','parent_candidate','research_area','hypothesis','mechanism','proposed_change','predicted_behavior','alternative_predictions','matched_control','required_code_changes','estimated_training_steps','estimated_runtime','estimated_agent_tokens','evaluation_plan','stop_conditions','config']
ROLES=['coordinator','architecture','learning','experience','initialization','critic']
DISABLED=['multi_agent','multi_agent_v2','apps','plugins','hooks','browser_use','browser_use_external','browser_use_full_cdp_access','computer_use','image_generation','in_app_browser','in_app_local_automation','shell_tool','unified_exec','workspace_dependencies','memories','skill_search','skill_mcp_dependency_install','remote_plugin','code_mode_host','code_mode','sleep_tool','view_image','unbounded_connection_retries']

def validate_proposal(p):
    if not isinstance(p,dict) or any(k not in p for k in REQUIRED):raise ValueError('Incomplete proposal schema')
    identifier(p['id'])
    if p['research_area'] not in ROLES[1:5] or not p['hypothesis'] or not p['matched_control'] or not p['alternative_predictions']:raise ValueError('Need falsifiable mechanism, competing predictions and matched control')
    if len(json.dumps(p).split())>1000:raise ValueError('Proposal exceeds1000 words')
    for k in ['estimated_training_steps','estimated_runtime','estimated_agent_tokens']:
        if type(p[k]) not in [int,float] or not 0<=p[k]<float('inf'):raise ValueError('Invalid declared resource estimate')
    if not isinstance(p['mechanism'],dict) or not isinstance(p['config'],dict):raise ValueError('Structured principal mechanism and configuration required')
    return p

def subscription_env(env=None):
    # Never inherit API-key routing/billing, tracing or preload hooks.
    e=dict(os.environ if env is None else env)
    for key in list(e):
        if key.endswith(('API_KEY','API_TOKEN')) or key.startswith(('OPENAI_BASE','OPENAI_ORG','ANTHROPIC_','AZURE_','LD_PRELOAD','DYLD_','PYTHONPATH','PYTHONSTARTUP')):e.pop(key,None)
    return e

def parse_events(events):
    completed=[r for r in events if r.get('type')=='turn.completed']
    if len(completed)>1:raise ValueError('Unexpected multiple agent turns; do not double-count aggregate usage')
    messages=[r['item'].get('text','') for r in events if r.get('type')=='item.completed' and r.get('item',{}).get('type')=='agent_message']
    return dict(usage=completed[0].get('usage') if completed else None,text=messages[-1] if messages else '',failed=any(r.get('type') in ['turn.failed','error'] for r in events),nested_agent=any(r.get('item',{}).get('type')=='collab_tool_call' for r in events))

def authenticated():
    codex=shutil.which('codex')
    if not codex:raise RuntimeError('Codex CLI missing; no billing fallback')
    r=subprocess.run([codex,'login','status'],text=True,capture_output=True,env=subscription_env(),timeout=15)
    if r.returncode or 'Logged in using ChatGPT' not in r.stdout+r.stderr:raise RuntimeError('Authenticated ChatGPT subscription login required; API login not accepted')
    return codex

def invoke(store,role,prompt,kind='proposal',context=None):
    if role not in ROLES:raise ValueError(role)
    if store.get('confirmation_started'):raise RuntimeError('No further research-agent dispatch after confirmation begins')
    previous=store.get('agent_inflight');phase=store.get('phase')
    if previous:
        if previous['status']!='completed':raise RuntimeError('Previous agent call is unresolved; inspect retained events, never silently rerun')
        if previous['phase']==phase:
            if previous['role']!=role:raise RuntimeError('Uncommitted prior call belongs to another role')
            if store.get('status')=='running':store.control('manual_checkpoint')
            return previous['response']
    codex=authenticated();call='call-'+uuid.uuid4().hex;activity={'critic':'criticism','coding':'implementation','review':'criticism','proposal':'hypothesis','coordinator':'reporting'}[kind]
    # Codex reports usage at turn end but has no verified provider-enforced
    # per-call token cap. Every real call therefore requires a manual checkpoint.
    if not store.get('manual_checkpoints'):raise RuntimeError('Unattended global token ceiling cannot be guaranteed by this CLI; manual checkpoints required')
    from .quota import read_quota
    quota=read_quota(codex,subscription_env());store.set('latest_quota',quota)
    if quota.get('ordinary_usage_allowed') is not True:raise RuntimeError('Ordinary included usage unavailable or exhausted; do not enable reserve credits or paid fallback')
    if any(b.get('rateLimitReachedType') or b.get('spendControlReached') for b in quota.get('buckets',{}).values()):raise RuntimeError('Platform reports insufficient ordinary allowance; no further call')
    store.reserve_call(call,activity,CALL_LIMITS[kind],speculative=kind in ['proposal','coding'])
    active=store.get('active')
    if active and kind!='proposal':store.db.execute('INSERT INTO call_links VALUES(?,?)',(call,active['id']))
    store.set('agent_inflight',dict(id=call,phase=phase,role=role,status='dispatched_or_uncertain'))
    store.reserve_cpu(call,60)
    w=safe_workspace(store.directory/'workspaces'/call,store.directory/'workspaces');write(w/'context.json',context or {},True)
    instructions=(PACKAGE/f'agents/{role}.md').read_text();(w/'AGENTS.md').write_text(instructions)
    argv=[codex,'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','--sandbox','workspace-write','--json','-C',str(w)]
    for name in DISABLED:argv+=['--disable',name]
    argv+=['-']
    try:
        remaining=store.limits['wall_seconds']-(time.time()-(store.get('started') or time.time()))
        if remaining<=0:raise RuntimeError('Campaign wall-clock limit reached before dispatch')
        with worker_lock(PACKAGE/'database'):r=run(argv,w,[ROOT],timeout=min(600,remaining),cpu_seconds=60,network=True,executable=codex,env=subscription_env(),stdin=instructions+'\n'+prompt+'\nContext:\n'+json.dumps(context or {}),cancel_check=lambda:store.get('status') in ['paused','stopped'])
        (w/'events.jsonl').write_text(r['stdout']);(w/'stderr.log').write_text(r['stderr']);events=[json.loads(line) for line in r['stdout'].splitlines() if line.startswith('{')];parsed=parse_events(events)
    except BaseException:
        store.reconcile_call(call,None);raise
    store.reconcile_cpu(call,r['cpu_seconds'])
    store.reconcile_call(call,parsed['usage']);write(w/'receipt.json',{k:v for k,v in r.items() if k not in ['stdout','stderr']})
    if parsed['usage'] is None or parsed['failed'] or parsed['nested_agent'] or r['returncode']:
        if store.get('status') in ['ready','running']:store.control('manual_checkpoint')
        raise RuntimeError('Agent failed, quota/telemetry missing, or unsupported nested call; inspect retained reservation and events')
    if store.get('status')=='budget_exhausted':raise RuntimeError('Reported usage exceeded reservation; no further dispatch')
    response=dict(call_id=call,text=parsed['text'],usage=parsed['usage'],requires_manual_checkpoint=True)
    store.set('agent_inflight',dict(id=call,phase=phase,role=role,status='completed',response=response))
    if store.get('status')=='running':store.control('manual_checkpoint')
    return response
