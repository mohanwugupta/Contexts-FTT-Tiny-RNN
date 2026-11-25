"""Trusted single-worker native execution; no LLM dispatch or confirmation access."""
import contextlib
import io
import gzip
import json
import os
from pathlib import Path
import sys
import time
import traceback

if __package__ in [None,'']:
    __package__='experiments.cloud_native_trial_v1'


def guarded_runtime(base):
    from research_search.orchestration.budget import BudgetError
    class CleanBoundaryRuntime(base):
        def check(self):
            self.store.check()
            folder=getattr(self.store,'directory',None)
            if folder is not None and (Path(folder)/'STOP_REQUESTED').exists():raise BudgetError('Owner stop before next update')
            if time.process_time()>=self.cpu_deadline or self.deadline and time.monotonic()>=self.deadline:
                raise BudgetError('Clean resource boundary; complete state retained; no automatic continuation')
    return CleanBoundaryRuntime


def schedule(store,assets,identity):
    from research_search.common import fingerprint,sha,read
    from research_search.orchestration.coordinator import _enqueue
    from .control import configs
    target_ids=[];competence_ids=[]
    for arm,config in zip(['control','candidate'],configs()):
        for seed in [21000,21001]:
            library=json.loads(gzip.decompress((assets/f'seed{seed}.json.gz').read_bytes()))
            key=fingerprint(dict(model=config['model'],prior=config['prior'],seed=seed,library_sha=sha(assets/f'seed{seed}.json.gz')))[:24]
            pid='prior-'+key
            status=_enqueue(store,pid,arm,dict(kind='prior',config=config,seed=seed,identity=identity,tasks=library['background']))
            if status!='completed':continue
            pf=store.directory/'jobs'/pid/'pretrained.pt';prior=dict(prior_file=str(pf),prior_sha=sha(pf))
            for j,task in enumerate(library['competence']):
                jid=f'competence-{key}-{j}';competence_ids.append(jid)
                _enqueue(store,jid,arm,dict(kind='competence',config=config,seed=seed,identity=identity,tasks=[task],**prior))
            for task in read(assets/'development.json')['tasks']:
                jid='target-'+fingerprint(dict(config=config,seed=seed,task=task,phase='development'))[:24];target_ids.append(jid)
                _enqueue(store,jid,arm,dict(kind='target',config=config,seed=seed,identity=identity,task=task,phase='development',**prior))
    return target_ids,competence_ids


def report(store):
    import numpy as np
    import pandas as pd
    from research_search.common import read,write,clean,sha
    from experiments.meta_objective_balance.statistics import interval
    from .control import META_CEILING
    rows=[];coverage=[];competence=[];meta=0;checkpoints=[]
    for job in store.db.execute('SELECT * FROM jobs ORDER BY rowid'):
        payload=json.loads(job['payload']);folder=store.directory/'jobs'/job['id']
        coverage.append(dict(job=job['id'],arm=job['candidate'],seed=payload['seed'],kind=payload['kind'],task_id=payload.get('task',{}).get('task_id'),status=job['status'],error=job['error']))
        if (folder/'competence.json').exists():competence.append(dict(job=job['id'],arm=job['candidate'],seed=payload['seed'],**read(folder/'competence.json')))
        if payload['kind']=='prior' and (folder/'progress.pt').exists():
            import torch
            cp=torch.load(folder/'progress.pt',weights_only=False);meta+=cp['extra']['episode']
        for name in ['progress.pt','states.pt','pretrained.pt']:
            if (folder/name).is_file():checkpoints.append(dict(path=str((folder/name).relative_to(store.directory)),sha256=sha(folder/name)))
        if (folder/'evaluation.json').exists():
            row=read(folder/'evaluation.json')
            row.update(arm=job['candidate'],joint_four_outcome=bool(row['original_early_phenotype'] and all(row[c+'_useful'] for c in ['A','B','C','Mixed','NoContext'])))
            rows.append(row)
    if meta>META_CEILING:raise ValueError('Reptile outer-operation ceiling exceeded')
    f=pd.DataFrame(rows);pd.DataFrame(coverage).to_csv(store.directory/'coverage.csv',index=False);f.to_csv(store.directory/'seed_task_results.csv',index=False)
    write(store.directory/'background_competence.json',competence);write(store.directory/'checkpoint_hashes.json',checkpoints)
    metrics=['D_A','D_B','net_A','net_B','A_to_B','B_to_A','sustained_cost','familiar_ce','C_ce','C_accuracy','Mixed_ce','Mixed_accuracy','NoContext_ce','NoContext_accuracy','joint_four_outcome']
    intervals=[]
    if rows:
        for metric in metrics:
            means={}
            for arm in ['control','candidate']:
                g=f[f.arm.eq(arm)].groupby('seed')[metric].agg(['mean','count']).reindex([21000,21001]);x=g['mean'].where(g['count'].eq(4)).to_numpy(float);means[arm]=x
                intervals.append(dict(arm=arm,metric=metric,**interval(x,[21000,21001])))
            intervals.append(dict(arm='paired_candidate_minus_control',metric=metric,**interval(means['candidate']-means['control'],[21000,21001])))
    pd.DataFrame(intervals).to_csv(store.directory/'seed_intervals.csv',index=False)
    complete=len(rows)==16 and len(competence)==12 and all(r['status']=='completed' for r in coverage)
    adequate=all(r['all_practiced_contexts_useful'] for r in competence) if len(competence)==12 else None
    uncertain=store.db.execute("SELECT count(*) FROM steps WHERE status!='completed'").fetchone()[0]
    result=dict(complete=complete,background_adequate=adequate,status='completed_development_only' if complete and adequate else 'background_inconclusive' if complete else 'incomplete_inconclusive',
      target_results=len(rows),planned_target_results=16,planned_competence=12,coverage=coverage,
      actual=store.totals(),confirmed_reptile_interpolations=meta,reptile_interpolations=None if uncertain else meta,uncertain_optimizer_attempts=uncertain,combined_optimizer_and_meta_operations=None if uncertain else store.totals()['optimizer_steps']+meta,
      new_agent_calls=0,provider_tokens=None,old_token_holds=117000,parent_agent_isolation_certified=False,confirmation_evaluations=0,late_functional_status='unassessed',screen_status='unassessed')
    write(store.directory/'RESULT.json',result)
    (store.directory/'REPORT.md').write_text('# Native Cloud trial\n\n'+json.dumps(clean(result),indent=2)+'\n\nAll target results and missing jobs are retained. Task averages precede seed bootstrap (n=2); intervals are descriptive. The frozen scorer owns historical flags. NoContext is an additional joint requirement, not a replacement for the original measures. Background failure makes mechanism rejection inconclusive. No result triggers more training, confirmation or agent calls.\n')
    if len(rows):
        from .figures import draw
        draw(store.directory,coverage)
    return result


def run_native(root=Path('/trial'),assets=Path('/workspace/trial_assets'),cpu_begin=None):
    from research_search.common import read,write
    from research_search.orchestration.store import Store
    from research_search.orchestration.queue import claim,finish,worker_lock
    from research_search.orchestration.budget import BudgetError
    from research_search.runners import evaluate
    from .control import LIMITS,validate_definition
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_definition(read('/workspace/definition.json'))
    identity=read(assets/'provenance.json')['runner_hash']
    # Versioned clean-stop guard only. Construction, update, checkpoint and scoring code remain frozen.
    evaluate.NativeRuntime=guarded_runtime(evaluate.NativeRuntime)
    write(root/'budgets.json',LIMITS,True)
    store=Store(root,LIMITS);start=time.process_time() if cpu_begin is None else cpu_begin;cpu_id='native-'+str(time.time_ns());cpu_reserved=False
    with worker_lock(root):
        try:
            store.control('running') # refuses uncertain steps, terminal budgets and unknown call usage
            # Only explicit rerun reaches here. Full-state identity/cursor/hash checked by NativeRuntime.
            for row in store.db.execute("SELECT id FROM jobs WHERE status='paused'").fetchall():
                store.db.execute("UPDATE jobs SET status='queued' WHERE id=?",(row['id'],))
            remaining=LIMITS['cpu_seconds']-store.totals()['cpu_seconds']
            store.reserve_cpu(cpu_id,remaining);cpu_reserved=True
            while True:
                store.check()
                if (root/'STOP_REQUESTED').exists():store.control('stopped');break
                schedule(store,assets,identity)
                job=claim(store)
                if job is None:store.control('stopped');break
                left=remaining-(time.process_time()-start)
                wall=LIMITS['wall_seconds']-(time.time()-store.get('started'))
                os.environ['RESEARCH_CPU_SECONDS']=str(left);os.environ['RESEARCH_WALL_SECONDS']=str(wall)
                try:
                    r=evaluate.execute(root,job)
                    if r.get('passed') is not True:raise ValueError('Native job did not pass')
                    write(root/'worker_receipts'/job['id']/(cpu_id+'.json'),r,True);finish(store,job['id'],'completed')
                except BudgetError as error:
                    evaluate.partial(root,job);finish(store,job['id'],'paused',str(error));store.control('stopped' if (root/'STOP_REQUESTED').exists() else 'paused');break
                except BaseException as error:
                    evaluate.partial(root,job);finish(store,job['id'],'failed',repr(error));store.control('manual_checkpoint')
                    write(root/'FAILURE.json',dict(reason=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));break
        finally:
            result=report(store)
            if cpu_reserved:store.reconcile_cpu(cpu_id,time.process_time()-start)
            result['actual']=store.totals()
            write(root/'RESULT.json',result)
            store.close()
    return result


def main():
    cpu_begin=time.process_time()
    action=sys.argv[1]
    os.environ.clear();os.environ.update(PATH='/usr/local/bin:/usr/bin:/bin',HOME='/tmp',TMPDIR='/tmp',LANG='C.UTF-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mpl',PYTHONDONTWRITEBYTECODE='1')
    sys.path[:0]=['/workspace','/workspace/src']
    from .control import sha,save
    if action=='export':
        from .export import collect
        print(json.dumps(collect('/trial',Path('/output')/'bundle')));return
    if action=='stop':
        Path('/trial/STOP_REQUESTED').write_text('Owner requested terminal stop\n');return
    from experiments.worker_isolation_preflight_v1.container_probe import probe
    from experiments.worker_isolation_preflight_v1.preflight import verify_checks
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):probe()
    proof=json.loads(stream.getvalue());verify_checks(proof['checks'])
    hashes=json.loads(Path('/workspace/STAGED_FILES.json').read_text())
    for rel,h in hashes.items():
        if sha(Path('/workspace')/rel)!=h:raise ValueError('Staged source/task identity changed: '+rel)
    try:Path('/workspace/STAGED_FILES.json').write_text('DENIED')
    except OSError as error:
        if error.errno not in [13,30]:raise
    else:raise ValueError('Actual source mount writable')
    proof['actual_source_write_denied']=True;save('/output/WORKER_ISOLATION.json',proof)
    if action=='fixture':
        # Import/contract/filesystem fixture only: no model construction or update.
        from research_search.runners.candidate import validate,DEFAULT
        from research_search.runners.evaluate import execute
        from .control import configs
        a,b=configs()
        if a!=DEFAULT or validate(a)!=a or validate(b)!=b:raise ValueError('Native contract mismatch')
        p=Path('/trial/private_result.json');p.write_text('{"synthetic":true}\n');p.chmod(0o600)
        save('/output/RECEIPT.json',dict(passed=True,new_optimizer_steps=0,new_model_forwards=0,checks=19));return
    if action!='run':raise ValueError('Unsupported action')
    result=run_native(cpu_begin=cpu_begin)
    for rel,h in hashes.items():
        if sha(Path('/workspace')/rel)!=h:raise ValueError('Source or registered tasks changed')
    save('/output/RECEIPT.json',result)
    if not result['complete']:raise SystemExit(3)

if __name__=='__main__':main()
