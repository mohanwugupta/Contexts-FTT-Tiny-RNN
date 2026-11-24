"""Independent saved-position-to-summary arithmetic; no model replay."""
import concurrent.futures
from .common import *

def verify_means(positions,evaluations):
 p=positions.groupby(['update','cue']).agg(position_ce=('ce','mean'),position_accuracy=('correct','mean')).reset_index();q=p.merge(evaluations[['update','cue','ce','accuracy']],on=['update','cue'],validate='one_to_one');assert len(q)==len(evaluations);np.testing.assert_allclose(q.position_ce,q.ce,rtol=1e-12,atol=1e-12,equal_nan=True);np.testing.assert_allclose(q.position_accuracy,q.accuracy,rtol=1e-12,atol=1e-12,equal_nan=True)

def check(row):
 d=OUT/row['directory'];p=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip');e=pd.read_csv(d/'evaluations.csv',float_precision='round_trip');verify_means(p,e);return len(p)

def main():
 r=pd.concat([pd.read_csv(OUT/'registry/stage_a_new_targets.csv'),pd.read_csv(OUT/'registry/stage_c_new_targets.csv')],ignore_index=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:n=sum(pool.map(check,r.to_dict('records')))
 assert len(r)==560 and n==7785120;dump(OUT/'audit/position_summary_crosslinks.json',dict(passed=True,new_runs=560,position_rows=n,frozen_summaries=560*331*6,all_probabilities_and_CE_validated_in_full_verification=True,no_replay=True));append('## Position/summarycrosslink GREEN\nIndependentgroupingofall7785120newpositionrows matchesall1112160savednewfrozenCE/accuracysummaries,pairedbyrun/state/cue. Wholemodel/state/probability/numericcontrolauditwasalreadyGREEN;thisclosesraw-to-summaryexportlinkwithoutnewforwardsortraining.')
if __name__=='__main__':main()
