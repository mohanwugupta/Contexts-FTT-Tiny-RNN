"""Explicit entry point: sealed work permits only read-only reproduction/verification."""
import argparse,json,time
from .protocol import OUT
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['inspect','run','analyze','report','figures','verify','reproduce','seal']);p.add_argument('--destination');p.add_argument('--resume',action='store_true');p.add_argument('--wall-time-limit-minutes',type=float);args=p.parse_args()
    if (OUT/'manifest.json').exists() and args.stage not in ['verify','reproduce']:p.error('Experiment is sealed and stopped. Use verify or reproduce into an empty external directory.')
    if args.stage=='inspect':
        from .protocol import inspect
        result=inspect()
    elif args.stage=='run':
        from .execution import run
        if not args.resume:p.error('Explicit --resume required; never replay uncertain operations')
        result=run(time.monotonic()+60*args.wall_time_limit_minutes if args.wall_time_limit_minutes else None)
    elif args.stage in ['analyze','report','figures']:
        from . import analysis,reporting,figures
        result={'analyze':analysis.main,'report':reporting.main,'figures':figures.main}[args.stage](args.destination)
        result={'passed':True,'stage':args.stage}
    elif args.stage=='reproduce':
        from .regenerate import main as regenerate
        if not args.destination:p.error('--destination required')
        result=regenerate(args.destination)
    else:
        from . import verify
        result=verify.seal() if args.stage=='seal' else (verify.sealed() if (OUT/'manifest.json').exists() else verify.main())
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
