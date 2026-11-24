"""Independently recreate scalar branch measurements from compressed arrays only."""
from .common import *
from . import design

def main(destination=None):
 destination=Path(destination or '/tmp/ab_update_numeric_regeneration');destination.mkdir(parents=True,exist_ok=True);assert not list(destination.iterdir())
 branches=pd.read_csv(OUT/'registry/branches.csv');rows=[]
 for r in branches[branches.status.eq('completed')].itertuples():
  path=(OUT/r.result).parent/'raw.npz';assert sha(path)==r.raw_sha256
  with np.load(path) as x:
   ce=x['ce'].mean(-1);acc=x['accuracy'].mean(-1);margin=x['margin'].mean(-1);targetp=x['probabilities'][:,:,np.arange(7),core.TARGET].mean(-1);own=0 if r.source_context=='A' else 1;other=1-own
   for k in range(len(ce)):
    own_gain=ce[0,own]-ce[k,own];cross_gain=ce[0,other]-ce[k,other]
    for j,c in enumerate(design.PROBES):rows.append(dict(trajectory_id=r.trajectory_id,target_update=r.target_update,source_context=r.source_context,branch_type=r.branch_type,branch_step=k,probe_context=c,pre_ce=ce[0,j],ce=ce[k,j],gain=ce[0,j]-ce[k,j],accuracy=acc[k,j],margin=margin[k,j],target_probability=targetp[k,j],own_gain=own_gain,cross_gain=cross_gain,specificity=own_gain-cross_gain))
 regenerated=pd.DataFrame(rows);keys=['trajectory_id','target_update','source_context','branch_type','branch_step','probe_context'];reported=pd.read_parquet(OUT/'tables/branch_metrics.parquet');joined=regenerated.merge(reported,on=keys,suffixes=('_regenerated','_reported'),validate='one_to_one');assert len(joined)==len(regenerated)==len(reported)
 metrics=['pre_ce','ce','gain','accuracy','margin','target_probability','own_gain','cross_gain','specificity'];errors={}
 for field in metrics:
  x=joined[field+'_regenerated'].to_numpy(float);y=joined[field+'_reported'].to_numpy(float);np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12);errors[field]=float(abs(x-y).max())
 effects=pd.read_csv(OUT/'tables/update_effects.csv',float_precision='round_trip');joined2=effects.merge(regenerated,on=keys,suffixes=('_reported','_regenerated'),validate='one_to_one');assert len(joined2)==len(effects)
 for field in ['pre_ce','gain','own_gain','cross_gain','specificity']:np.testing.assert_allclose(joined2[field+'_reported'],joined2[field+'_regenerated'],rtol=1e-12,atol=1e-12)
 csv(destination/'branch_metrics_recreated.csv',regenerated);receipt=dict(passed=True,scalar_rows=len(regenerated),endpoint_rows=len(effects),metrics=metrics,max_errors=errors,optimizer_operations=0,new_model_rollouts=0,external_destination=str(destination),external_table_sha256=sha(destination/'branch_metrics_recreated.csv'))
 path=OUT/'audit/numeric_regeneration.json'
 if path.exists():
  expected=json.loads(path.read_text())
  for key in ['passed','scalar_rows','endpoint_rows','metrics','max_errors','optimizer_operations','new_model_rollouts','external_table_sha256']:assert expected[key]==receipt[key],key
 elif not (OUT/'manifest.json').exists():dump(path,receipt)
 else:raise RuntimeError('Sealed numeric reproduction receipt is missing')
 dump(destination/'validation.json',receipt)
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir');main(parser.parse_args().output_dir)
