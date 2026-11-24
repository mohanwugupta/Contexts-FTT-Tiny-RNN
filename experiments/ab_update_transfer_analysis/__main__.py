import argparse
from . import common

def after_seal(stage,output_dir,sealed):
 if sealed and (stage in ['inspect','preflight','run','summarize'] or stage=='figures' and output_dir is None):raise RuntimeError('Scientific budgets are stopped and evidence sealed. Use verify or figures with a new external destination.')

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['inspect','preflight','run','summarize','figures','verify']);p.add_argument('--config',default='configs/ab_update_transfer_analysis.yaml');p.add_argument('--resume',action='store_true');p.add_argument('--wall-time-limit-minutes',type=float);p.add_argument('--output-dir');a=p.parse_args();c=common.config(a.config);after_seal(a.stage,a.output_dir,(common.OUT/'manifest.json').exists());common.configure()
 if a.stage=='inspect':
  from .sources import inspect
  inspect()
 elif a.stage in ['preflight','run']:
  from .engine import run
  run(a.stage=='preflight',a.wall_time_limit_minutes if a.wall_time_limit_minutes is not None else c['execution']['optional_wall_time_limit_minutes'])
 elif a.stage=='summarize':
  from .analysis import main
  main()
 elif a.stage=='figures':
  from .figures import main
  main(output_dir=a.output_dir)
 else:
  from .verify import main
  main()
if __name__=='__main__':main()
