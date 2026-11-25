"""Fresh, bounded selected-analysis job; no agent calls and no retry path."""
import argparse,datetime,fcntl,hashlib,json,os,resource,signal,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def deadline_seconds(value,now=None):
    dt=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('Explicit timezone required')
    return max(0.,min(300.,dt.timestamp()-(time.time() if now is None else now)))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def child(args):
    from .sandbox import lockdown,proof
    out=args.out.resolve();tmp=out/'tmp';tmp.mkdir()
    if (ROOT/'research_search/database/benchmark_banks/sealed').exists():raise RuntimeError('Unexpected private confirmation bank present; worker access is not authorized')
    manifest=json.loads((ROOT/'CLOUD_INPUT_MANIFEST.json').read_text())
    scope=lockdown(ROOT,out,tmp);checks=proof(args.source_fixture,args.private_fixture,out)
    save(out/'WORKER_ISOLATION.json',scope|dict(checks=checks,passed=True))
    from experiments.research_search_cloud_v1.port import verify
    verify(ROOT,manifest)
    import jsonschema,pandas as pd,numpy as np,torch
    from unittest.mock import patch
    from research_search.common import read
    from experiments.a06_novel_cue_followup.verify import no_scientific_execution
    from experiments.research_search_integration_v1_1.protocol import schema,validate_proposal
    selection=read(args.selection);proposal=read(args.proposal)
    jsonschema.validate(selection,schema('coordinator_select'));validate_proposal(proposal,'initialization')
    if selection['selected']!=proposal['id'] or proposal['analysis']!='confidence_vs_correctness':raise ValueError('Wrong selected proposal')
    from experiments.research_search_integration_v1_1.cached import execute
    from .analysis import TIMES,tables,seed_intervals
    request=out/'request';request.mkdir();save(request/'definition.json',dict(sources=manifest['native_consumed_inputs']))
    def forbidden(*a,**k):raise RuntimeError('No checkpoint/model loading authorized')
    with no_scientific_execution(),patch.object(torch,'load',forbidden):
        execute(request,'confidence_vs_correctness',out/'native')
        source=ROOT/'research_search/database/controlled_pilot_v2';frames=[]
        tasks={t['task_id']:t['path'] for t in read(source/'development.json')['tasks']}
        for condition,jobs in read(source/'paired_jobs.json').items():
            for job in jobs:
                p=source/'jobs'/job;meta=read(p/'evaluation.json')
                for view,name in [('native','positions.csv'),('quarter','static_readout_positions.csv')]:
                    f=pd.read_csv(p/name,float_precision='round_trip');f=f[f['update'].isin(TIMES)].copy()
                    path=tasks[meta['task_id']]
                    if not np.array_equal(f.target.to_numpy(),np.asarray(path)[f.position.to_numpy(int)]) or not np.array_equal(f.token.to_numpy(),np.asarray(path)[f.position.to_numpy(int)-1]):raise ValueError('Wrong frozen token/target indexing')
                    frames.append(f.assign(condition=condition,seed=meta['seed'],task_id=meta['task_id'],view=view))
        observations=pd.concat(frames,ignore_index=True);products=tables(observations)
        observations.to_csv(out/'fixed_time_positions.csv',index=False)
        for name,f in products.items():f.to_csv(out/(name+'.csv'),index=False)
        for name,dims in [('within_readout',['condition','update','cue']),('initialization',['view','update','cue'])]:
            means,intervals=seed_intervals(products[name+'_metrics'],dims)
            means.to_csv(out/(name+'_seed_means.csv'),index=False);intervals.to_csv(out/(name+'_seed_intervals.csv'),index=False)
    from .figures import render
    render(out)
    verify(ROOT,manifest)
    save(out/'RESULT.json',dict(status='completed_cached_analysis',selected=selection['selected'],proposal_sha256=sha(args.proposal),selection_sha256=sha(args.selection),fixed_time_positions=len(observations),paired_position_rows_per_comparison=4480,independent_seeds=2,new_model_forwards=0,new_optimizer_steps=0,native_summary='native/evaluation_result.json',scientific_scope='Readout expression and saved developmental comparisons only; no natural retrieval or competent-prior mechanism established.',full_parent_agent_isolation_certified=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--proposal',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline-utc',required=True);p.add_argument('--child',action='store_true');p.add_argument('--source-fixture',type=Path);p.add_argument('--private-fixture',type=Path);a=p.parse_args()
    if a.child:
        try:child(a)
        except BaseException as ex:save(a.out/'BLOCKED.json',dict(status='blocked_or_failed',error_type=type(ex).__name__,reason=str(ex),no_automatic_retry=True));raise
        return
    for k in ['selection','proposal','out']:setattr(a,k,getattr(a,k).resolve())
    wall=deadline_seconds(a.deadline_utc)
    if wall<=0:raise RuntimeError('Explicit pilot execution deadline passed')
    if a.out.exists():raise RuntimeError('Existing job must not be replayed or overwritten')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.out.mkdir()
        fixture=a.out.parent/'source_fixture.txt'
        if fixture.exists():raise RuntimeError('Prior execution fixture exists; inspect before any reuse')
        fixture.write_text('UNCHANGED_CONTROL');vault=Path(tempfile.mkdtemp(prefix='srn-private-control-',dir='/tmp'));secret=vault/'fixture.txt';secret.write_text('PRIVATE_CONTROL')
        save(a.out/'REQUEST.json',dict(selected_proposal=sha(a.proposal),selection=sha(a.selection),deadline=a.deadline_utc,CPU_cap_seconds=120,wall_cap_seconds=wall,code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        env=dict(PATH=os.defpath,HOME=str(a.out/'tmp'),TMPDIR=str(a.out/'tmp'),PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(a.out/'tmp/mpl'),LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        def limits():resource.setrlimit(resource.RLIMIT_CPU,(120,120));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        argv=[sys.executable,'-m','experiments.cloud_confidence_selected_v1.driver','--child','--selection',str(a.selection),'--proposal',str(a.proposal),'--out',str(a.out),'--deadline-utc',a.deadline_utc,'--source-fixture',str(fixture),'--private-fixture',str(secret)]
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,preexec_fn=limits)
        timeout=False
        try:stdout,stderr=proc.communicate(timeout=wall)
        except subprocess.TimeoutExpired:timeout=True;os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
        finally:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        after=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        (a.out/'stdout.log').write_text(stdout);(a.out/'stderr.log').write_text(stderr)
        passed=proc.returncode==0 and not timeout and cpu<=120 and (a.out/'RESULT.json').exists() and fixture.read_text()=='UNCHANGED_CONTROL' and secret.read_text()=='PRIVATE_CONTROL'
        receipt=dict(passed=passed,returncode=proc.returncode,timed_out=timeout,wall_seconds=time.monotonic()-start,CPU_seconds=cpu,CPU_cap_seconds=120,deadline=a.deadline_utc,new_agent_calls=0,reported_model_token_usage=None,full_parent_agent_isolation_certified=False)
        save(a.out/'EXECUTION_RECEIPT.json',receipt)
        save(a.out/'FILES.json',{str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file() and p.name!='FILES.json'})
        print(json.dumps(receipt,indent=2))
        if not passed:raise SystemExit(3)
if __name__=='__main__':main()
