"""One global CPU worker, hard subprocess CPU/wall bounds and durable receipts."""
import json,math,os,sys,time
from pathlib import Path
from research_search.common import *
from research_search.orchestration import queue
from research_search.orchestration.isolation import run
from research_search.orchestration.budget import BudgetError
from research_search.orchestration.agent_backend import subscription_env

def once(store):
    with queue.worker_lock(PACKAGE/'database'):
        job=queue.claim(store)
        if not job:return None
        payload=json.loads(job['payload']);request={k:job[k] for k in ['id','candidate','payload']};write(store.directory/'requests'/f"{job['id']}.json",request,True)
        remaining=store.limits['cpu_seconds']-store.totals()['cpu_seconds']-store.totals()['reserved_cpu_seconds'];cap=min(600,math.floor(remaining))
        if cap<1:queue.finish(store,job['id'],'paused','CPU ceiling');raise BudgetError('Insufficient remaining CPU reserve')
        cpu_id=f"job-{job['id']}-{int(time.time()*1e6)}";store.reserve_cpu(cpu_id,cap)
        remaining_wall=store.limits['wall_seconds']-(time.time()-(store.get('started') or time.time()));wall=min(1200,max(.1,remaining_wall))
        env=subscription_env();env.update(PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(store.directory/'tmp'),RESEARCH_CPU_SECONDS=str(cap),RESEARCH_WALL_SECONDS=str(wall),RESEARCH_ATTEMPT_ID=cpu_id);(store.directory/'tmp').mkdir(exist_ok=True)
        argv=[sys.executable,'-m','experiments.research_search_pilot_v2.worker_cli','--campaign',str(store.directory),'--job',job['id']]
        try:r=run(argv,store.directory,[ROOT/'results',ROOT/'outputs',ROOT/'data'],timeout=wall,cpu_seconds=cap,env=env)
        except BaseException:
            queue.finish(store,job['id'],'uncertain','Worker launch/accounting failed; CPU reservation retained');raise
        store.reconcile_cpu(cpu_id,r['cpu_seconds']);write(store.directory/'logs'/f'{cpu_id}.json',r,True)
        result_path=store.directory/'worker_receipts'/job['id']/f'{cpu_id}.json'
        if not result_path.exists():queue.finish(store,job['id'],'uncertain','No complete receipt; no automatic retry');return dict(passed=False,status='uncertain',job=job['id'])
        receipt=read(result_path)
        if receipt.get('passed'):queue.finish(store,job['id'],'completed')
        elif receipt.get('status')=='paused_or_stopped':queue.finish(store,job['id'],'paused',receipt.get('reason'))
        else:queue.finish(store,job['id'],'failed',receipt.get('reason'))
        if r['timed_out'] or r['returncode'] not in [0,3]:queue.finish(store,job['id'],'uncertain','Killed or unrecognized worker exit')
        return receipt|dict(job=job['id'])
