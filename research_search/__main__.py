"""Public command line. Search launch is always an explicit user action."""
import argparse,json
from .common import *

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','verify','init','launch','pause','resume','stop','status','report','probe-backend','reconcile']);p.add_argument('--campaign',default=str(PACKAGE/'database/campaign'));p.add_argument('--manual-checkpoints',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--call');p.add_argument('--usage-file');args=p.parse_args()
    if args.command=='preflight':
        from .preflight import acceptance,readiness
        result=readiness() if (PACKAGE/'audit/readiness_local.json').exists() else acceptance()
    elif args.command=='verify':
        from .preflight import verify_artifacts
        result=verify_artifacts()
    elif args.command=='probe-backend':
        from .backend_probe import run
        result=run()
    elif args.dry_run:
        from .preflight import verify_seal,readiness
        result=dict(command=args.command,campaign=str(Path(args.campaign).resolve()),dry_run=True,dispatched=False,verification=verify_seal(),readiness=readiness(),manual_required=True)
    else:
        from .orchestration.store import Store
        from .campaign import initialize,start,loop,recover_paused
        if args.command in ['init','launch','resume']:s=initialize(args.campaign)
        else:s=Store(Path(args.campaign).resolve(),read(Path(args.campaign)/'budgets.json'))
        try:
            if args.command=='init':result=s.state()
            elif args.command in ['launch','resume']:
                if not args.manual_checkpoints:p.error('Current CLI requires --manual-checkpoints; a hard token ceiling inside a call cannot be guaranteed')
                from .preflight import readiness
                if not readiness()['live_backend_verified']:raise RuntimeError('Live backend validation remains blocked/unverified. Obtain approval for the fixed JSON-only probe first; no campaign dispatched.')
                if args.command=='resume':recover_paused(s)
                start(s,True);result=loop(s)
            elif args.command in ['pause','stop']:s.control('paused' if args.command=='pause' else 'stopped');result=s.state()
            elif args.command=='status':result=s.state()|dict(phase=s.get('phase'),latest_quota=s.get('latest_quota'))
            elif args.command=='report':
                from .reporting import report
                result=dict(report=report(s,final=True))
            elif args.command=='reconcile':
                if not args.call or not args.usage_file:p.error('--call and --usage-file must contain actual provider aggregate usage, never estimates')
                s.reconcile_call(args.call,read(args.usage_file));result=s.state()
        finally:s.close()
    print(json.dumps(clean(result),indent=2))
if __name__=='__main__':main()
