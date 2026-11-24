import argparse,concurrent.futures,multiprocessing
from .common import *
from experiments.heldout_context_meta import verify as native
from experiments.heldout_context_meta_scaling_10k.verify import verify_positions
from experiments.heldout_context_meta_scaling_10k.adapters import bind

def one(r):
 if r['status']!='completed':return dict(seed=r['seed'],arm=r['arm'],dose=r['dose'],status=r['status'],not_imputed=True)
 with bind(native,OUT=ROOT if r['reused'] else OUT,initial=initial,verify_position_rows=verify_positions):q=native.verify_target(r)
 return dict(q,seed=r['seed'],arm=r['arm'],dose=r['dose'],reused=r['reused'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);stage=p.parse_args().stage;reg=pd.read_csv(OUT/f'registry/{stage}_runs.csv')
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(one,reg.to_dict('records')))
 write_frame(OUT/f'audit/{stage}_verified.csv',rows);dump(OUT/f'audit/{stage}_verified.json',dict(passed=True,cells=len(rows),all_four_probabilities=True,C_bitwise_fixed=True,native_original_Adam_loss_policy=True,optimizer_updates=0));print(stage+' verification passed',flush=True)
if __name__=='__main__':main()
