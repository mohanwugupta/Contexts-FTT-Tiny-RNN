import argparse,json,time,resource,signal,uuid
_COMMAND_CPU_START=time.process_time()
_COMMAND_CHILD_START=resource.getrusage(resource.RUSAGE_CHILDREN)
from pathlib import Path
from research_search.common import clean,write
from research_search.orchestration.store import Store
from .protocol import DIRECTORY,OUT,LIMITS


def main():
    p=argparse.ArgumentParser(description='Bounded integration; no full campaign command')
    p.add_argument('action',choices=['prepare','call','compute','status','report','stop','worker','estimate'])
    p.add_argument('--ticket');p.add_argument('--approval');p.add_argument('--campaign',type=Path,default=DIRECTORY);p.add_argument('--analysis')
    args=p.parse_args()
    if args.action=='worker':
        from experiments.a06_novel_cue_followup.verify import no_scientific_execution
        from .cached import execute
        with no_scientific_execution():r=execute(args.campaign,args.analysis,args.campaign/'jobs/integration11-cache')
    elif args.action=='estimate':
        from .costs import estimate
        r=estimate()
    else:
        from .controller import initialize,integrity,prepare,call
        s=initialize() if not (args.campaign/'definition.json').exists() else Store(args.campaign,LIMITS)
        metered=args.action in ['prepare','call','compute','report'] and s.get('status') not in ['stopped','budget_exhausted']
        prior_status=s.get('status');cpu_id=None;recorded_before=s.totals()['cpu_seconds']
        if metered:
            s.control('ready');cpu_id='owner-'+uuid.uuid4().hex
            s.reserve_cpu(cpu_id,60)
            # Owner CPU only; subprocesses have their own durable reserves and
            # RLIMITs. Include imports when reconciling the whole command.
            signal.signal(signal.SIGPROF,lambda *_:(_ for _ in ()).throw(RuntimeError('Owner command CPU ceiling')))
            signal.setitimer(signal.ITIMER_PROF,60)
        try:
            if args.action=='prepare':r=prepare(s)
            elif args.action=='call':r=call(s,args.ticket,args.approval)
            elif args.action=='compute':
                from .worker import once
                integrity(s)
                if s.get('phase')!='compute_ready':raise RuntimeError('Selection required before execution')
                s.control('running');r=once(s)
            elif args.action=='stop':s.set('stopped_at',time.time());s.control('stopped');r=s.state()
            elif args.action=='report':
                from .reporting import report
                r=report(s)
            else:r=s.state()|dict(phase=s.get('phase'))
        finally:
            if metered:
                signal.setitimer(signal.ITIMER_PROF,0)
                after=resource.getrusage(resource.RUSAGE_CHILDREN)
                measured=time.process_time()-_COMMAND_CPU_START+after.ru_utime+after.ru_stime-_COMMAND_CHILD_START.ru_utime-_COMMAND_CHILD_START.ru_stime
                accounted=s.totals()['cpu_seconds']-recorded_before
                s.reconcile_cpu(cpu_id,max(0.,measured-accounted))
                if args.action in ['prepare','report'] and s.get('status')=='ready' and prior_status=='manual_checkpoint':s.control('manual_checkpoint')
            s.close()
    print(json.dumps(clean(r),indent=2))
if __name__=='__main__':main()
