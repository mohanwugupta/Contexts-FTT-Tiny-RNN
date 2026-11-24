"""Registered CLI; report/verify never call a training stage."""
import argparse,time,json
from .common import *
def main():
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['inspect','preflight','pilot','select','confirm','report','verify','run']);parser.add_argument('--config',default='configs/joint_phenotype_hypothesis_screen.yaml');parser.add_argument('--phase',choices=['checkpoint','pretraining']);parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--wall-time-limit-minutes',type=float);parser.add_argument('--through',default='confirm',choices=['confirm']);args=parser.parse_args();configure()
 config=json.loads((ROOT/args.config).read_text());assert config['experiment']=='joint_phenotype_hypothesis_screen' and config['source']['scale']==5 and config['cohort']['additional_scales']==[] and config['budget']==dict(max_meta_inner_updates=124800,max_target_updates=15180,max_auxiliary_updates=5000)
 if (OUT/'manifest.json').exists() and args.command not in ['verify','inspect']:raise RuntimeError('Sealed program: only read-only verification/inspection; reproduce figures in a new external directory')
 from . import inspect,preflight,run,analysis,engine
 deadline=None if args.wall_time_limit_minutes is None else time.monotonic()+60*args.wall_time_limit_minutes
 try:
  if args.command=='inspect':
   inspect.main(args.dry_run)
   if args.dry_run:print((OUT/'registry/inspection.json').read_text())
  elif args.command=='preflight':preflight.main(deadline)
  elif args.command=='pilot':
   if args.phase is None:parser.error('pilot requires --phase')
   run.pilot(args.phase,deadline)
  elif args.command=='select':analysis.select()
  elif args.command=='confirm':run.confirm(deadline)
  elif args.command=='report':
   from .report import main as report;report()
  elif args.command=='verify':
   from .verify import main as verify;verify()
  else:
   inspect.main();preflight.main(deadline);run.pilot('checkpoint',deadline);run.pilot('pretraining',deadline);analysis.select();run.confirm(deadline)
 except engine.CleanStop as e:print(json.dumps(dict(status='clean_time_limit_stop',reason=str(e),resume=True)),flush=True)
if __name__=='__main__':main()
