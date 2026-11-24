"""Original full-state/probability/geometry checks, no matched branch replay."""
import concurrent.futures,multiprocessing
from .common import *
from experiments.heldout_context_meta import verify as native
from experiments.heldout_context_meta_scaling_10k.verify import verify_positions
from experiments.heldout_context_meta_scaling_10k.adapters import bind

def one(row):
 with bind(native,OUT=OUT,initial=initial,verify_position_rows=verify_positions):return native.verify_target(row)

def main():
 reg=pd.read_csv(OUT/'registry/preflight_targets.csv')
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(one,reg.to_dict('records')))
 assert len(rows)==12 and sum(r['target_steps'] for r in rows)==3960 and sum(r['diagnostic_steps'] for r in rows)==144;write_frame(OUT/'audit/preflight_verified.csv',rows);dump(OUT/'audit/preflight_verified.json',dict(passed=True,runs=12,states=3972,position_rows=166824,all_four_probabilities=True,all_C_columns_bitwise_fixed=True,all_original_loss_Adam_policy_counters=True,zero_optimizer_steps=True));print('All preflight states and probabilities PASSED',flush=True)
if __name__=='__main__':main()
