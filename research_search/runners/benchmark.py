"""One bounded native fixture benchmark, unrelated sequence; never a candidate screen."""
import copy,time,resource
from research_search.common import PACKAGE,write,read,fingerprint
from research_search.orchestration.store import Store
from research_search.orchestration.queue import worker_lock
from .candidate import DEFAULT,build_model,run_target_learning
from .train import NativeRuntime
from experiments.context_experience_prior.core import TARGET_PATH

FIXTURE_PATH=[0,0,1,0,2,3,1,3]

def benchmark():
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    receipt=PACKAGE/'audit/native_benchmark.json'
    if receipt.exists():return read(receipt)
    assert FIXTURE_PATH!=TARGET_PATH
    limits=dict(tokens=1000,cpu_seconds=60,optimizer_steps=6,wall_seconds=300,proposals=2,active_branches=2)
    state=PACKAGE/'database/benchmark';s=Store(state,limits)
    if s.totals()['optimizer_steps']:raise RuntimeError('Interrupted benchmark cannot be silently restarted')
    c=copy.deepcopy(DEFAULT);c['prior']['episodes']=0;m=build_model(c,170001);r=NativeRuntime(s,'native-benchmark',state/'run',c,170001,fingerprint(c),fixture_steps=6);r.attach(m);start=time.monotonic();cpu=time.process_time()
    with worker_lock(PACKAGE/'database'):
        run_target_learning(m,c,dict(task_id='non-scientific-benchmark',path=FIXTURE_PATH,context_assignment=['A','B']))
    elapsed=time.monotonic()-start;used=time.process_time()-cpu;result=dict(passed=True,fixture_only=True,native_steps=6,meta_steps=0,wall_seconds=elapsed,active_cpu_seconds=used,steps_per_cpu_second=6/max(used,1e-9),hardware=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sequence=FIXTURE_PATH,not_canonical=True)
    write(receipt,result,True);s.close();return result
