"""Single isolated no-network cached worker; hard CPU/wall limits, no retries."""
import json,os,sys,time
from research_search.common import ROOT,PACKAGE,write,read
from research_search.orchestration import queue
from research_search.orchestration.isolation import run
from research_search.orchestration.agent_backend import subscription_env
from .protocol import OUT


def once(store):
    with queue.worker_lock(PACKAGE/'database'):
        job=queue.claim(store)
        if not job:raise RuntimeError('No queued cached analysis')
        payload=json.loads(job['payload'])
        if payload['kind']!='cached_integration':raise ValueError('Only cached analysis authorized')
        cap=120;store.reserve_cpu('cached-worker',cap)
        argv=[sys.executable,'-m','experiments.research_search_integration_v1','worker','--campaign',str(store.directory),'--analysis',payload['analysis']]
        env=subscription_env();env.update(PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(store.directory/'mpl'))
        remaining=store.limits['wall_seconds']-(time.time()-store.get('started'))
        if remaining<=0:raise RuntimeError('Elapsed limit including approvals reached')
        protected=[ROOT/'results',ROOT/'outputs',ROOT/'data',PACKAGE/'database/benchmark_banks']
        try:r=run(argv,store.directory,protected,timeout=min(300,remaining),cpu_seconds=cap,env=env)
        except BaseException:
            queue.finish(store,job['id'],'uncertain','Worker launch failure; no automatic retry');raise
        store.reconcile_cpu('cached-worker',r['cpu_seconds']);write(store.directory/'worker_receipt.json',r,True)
        dest=store.directory/'jobs/integration1-cache'
        passed=r['returncode']==0 and not r['timed_out'] and (dest/'complete.json').exists()
        queue.finish(store,job['id'],'completed' if passed else 'failed',None if passed else r['stderr'][-1000:])
        if not passed:
            store.set('stopped_at',time.time());store.set('phase','compute_failed');store.control('stopped');return dict(passed=False,receipt=r)
        packet=read(dest/'evaluation_result.json');write(store.directory/'evaluation_result.json',packet,True)
        store.result('integration1-cache',packet);store.set('phase','critic');store.control('manual_checkpoint')
        return dict(passed=True,result=packet,CPU_seconds=r['cpu_seconds'],wall_seconds=r['wall_seconds'])
