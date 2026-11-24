"""Opt-in launch/control; never imported as an automatic research loop."""
import copy,time
from .common import *
from .orchestration.store import Store
from .orchestration.budget import BudgetError

def initialize(directory):
    p=Path(directory).resolve()
    if not p.is_relative_to(PACKAGE/'database') or p==PACKAGE/'database':raise ValueError('Campaign must use a dedicated research_search/database child')
    p.mkdir(parents=True,exist_ok=True);limits=read(PACKAGE/'config/budgets.yaml');write(p/'budgets.json',limits,True)
    from .evaluation.development import register
    a,b=register(PACKAGE/'database/benchmark_banks');write(p/'development.json',a,True);write(p/'sealed/confirmation.json',b,True)
    s=Store(p,limits)
    if not s.get('definition'):
        s.set('definition',dict(development_hash=sha(p/'development.json'),confirmation_hash=sha(p/'sealed/confirmation.json'),evaluator_seal=sha(PACKAGE/'audit/evaluator_seal.json'),source_manifest=sha(PACKAGE/'history/SOURCE_MANIFEST.json')));s.set('phase','propose')
        s.set('unmetered_setup_reservation',100000);s.set('setup_actual_tokens',None)
    return s

def integrity(store):
    from .preflight import verify_seal
    verify_seal();d=store.get('definition')
    for path,key in [(store.directory/'development.json','development_hash'),(store.directory/'sealed/confirmation.json','confirmation_hash'),(PACKAGE/'audit/evaluator_seal.json','evaluator_seal'),(PACKAGE/'history/SOURCE_MANIFEST.json','source_manifest')]:
        if sha(path)!=d[key]:raise ValueError('Campaign scientific identity changed: '+key)

def start(store,manual=False):
    from .orchestration.agent_backend import authenticated
    from .preflight import readiness
    integrity(store);ready=readiness()
    if not ready['infrastructure_passed']:raise RuntimeError('Preflight acceptance did not pass')
    if not ready['live_backend_verified']:raise RuntimeError('Live backend proof is missing; complete the separately approved fixed JSON-only probe before campaign launch')
    if not manual:raise RuntimeError('Unattended token ceiling cannot be guaranteed. Use --manual-checkpoints, one bounded call then explicit resume.')
    authenticated();store.set('manual_checkpoints',True);store.control('running')

def recover_paused(store):
    import torch
    from .orchestration.recovery import resume_job
    for row in store.db.execute("SELECT * FROM jobs WHERE status IN ('paused','running')").fetchall():
        cp=store.directory/'jobs'/row['id']/'progress.pt'
        if cp.exists():saved=torch.load(cp,weights_only=False);resume_job(store,row['id'],saved['attempts'],sha(cp))
        else:resume_job(store,row['id'],0,None)

def _failure(store,error):
    from .orchestration.coordinator import control_config
    p=store.get('active')
    if not p:raise error
    if store.get('confirmation_started'):
        write(store.directory/'sealed'/f"{p['id']}-incomplete.json",dict(status='incomplete_confirmation',error=repr(error),jobs=[dict(r) for r in store.db.execute('SELECT id,status,error FROM jobs')],no_research_feedback=True),True);store.db.execute("UPDATE confirmations SET status='incomplete' WHERE family=?",(p['id'],));store.set('phase','confirmation_incomplete');store.control('stopped');return
    n=store.get('development_tier') or 2;seeds=read(store.directory/'development.json')['seeds'][:n];tasks=read(store.directory/'development.json')['tasks'];coverage=[]
    for label,c in [(p['id'],p['config']),('matched-control',control_config(store,p))]:
        for seed in seeds:
            for task in tasks:
                id='target-'+fingerprint(dict(config=c,seed=seed,task=task,phase='development'))[:24];row=store.db.execute('SELECT status,error FROM jobs WHERE id=?',(id,)).fetchone();coverage.append(dict(condition=label,seed=seed,task_id=task['task_id'],status=row['status'] if row else 'not_run_blocked',reason=row['error'] if row else str(error)))
    result=dict(id=p['id'],phase='development',classification='technical_or_budget_inconclusive',development_feasible=False,ablation_supported=False,n_network_seeds=n,error=repr(error),coverage=coverage,all_missing_and_failed_cells_preserved=True)
    write(store.directory/'candidates'/p['id']/'incomplete_outcome.json',result,True);store.result(p['id']+'-incomplete',result);store.set('latest_result',result);store.set('phase','result_critic')

def loop(store):
    from .orchestration.coordinator import advance
    from .orchestration.worker import once
    from .reporting import report
    from .orchestration.cpu import controller_budget
    while store.get('status')=='running':
        try:
            with controller_budget(store):
                integrity(store)
                if time.time()-(store.get('last_report') or 0)>=86400:report(store)
                pending=store.db.execute("SELECT 1 FROM jobs WHERE status='queued'").fetchone()
                if pending:
                    result=once(store)
                    if result and not result.get('passed'):
                        if result.get('status')=='paused_or_stopped':
                            if store.get('status')=='running':store.control('paused')
                            break
                        _failure(store,RuntimeError(str(result)))
                else:advance(store)
        except BudgetError as ex:
            store.event('limit_or_checkpoint',str(ex))
            if store.get('status') not in ['stopped','paused','manual_checkpoint']:store.control('budget_exhausted')
            break
        except Exception as ex:
            store.event('controller_error',repr(ex))
            if store.get('active') and store.get('phase')=='evaluate':_failure(store,ex)
            else:store.control('manual_checkpoint');break
    report(store,final=store.get('status') in ['stopped','budget_exhausted']);return store.state()
