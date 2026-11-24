"""Whole-seed paired effects at every fixed block, switch, and C checkpoint."""
import argparse
from .common import *
from .analysis import paired_table

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);stage=p.parse_args().stage;seeds=SCREEN if stage=='screen' else CONFIRM;candidates=['B','C'] if stage=='screen' else [json.loads((OUT/'audit/selection_seal.json').read_text())['selected']]
 specs=[('switch',read(stage+'_switch_costs'),['dose','direction','boundary'],['raw_cost','adjusted_cost']),('block',read(stage+'_blocks'),['dose','start','end','practiced'],['practiced_gain','unpracticed_gain','specificity']),('C_trajectory',read(stage+'_evaluations').query("cue=='C'"),['dose','update'],['ce','gap']),('update_diagnostic',read(stage+'_learning_effects'),['dose','update'],['D_A','D_B','mean_specificity'])]
 for label,f,groups,metrics in specs:
  f=f[f.arm.isin(['A']+candidates)];raw,ci=paired_table(f,groups,metrics,seeds,candidates);write_frame(OUT/f'tables/{stage}_{label}_paired_seeds.csv',raw);write_frame(OUT/f'tables/{stage}_{label}_paired_intervals.csv',ci)
 dump(OUT/f'audit/{stage}_trajectory_pairs_complete.json',dict(passed=True,complete_seed_pairing=True,bootstrap=10000,pointwise=True,screen_confirmation_separate=True));print(stage,'paired trajectories complete',flush=True)
if __name__=='__main__':main()
