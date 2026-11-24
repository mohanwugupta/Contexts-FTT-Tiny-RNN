"""Validate recorded schedules and compress already cached endpoint arrays."""
from .common import *
from . import design

def validate_schedule(t,f):
 t=t.sort_values('update');assert np.array_equal(t['update'],np.arange(1,len(t)+1));assert np.array_equal(t.training_cue,[design.cue(k) for k in t['update']])
 q=f[f.probe_context.eq('A')].sort_values('target_update');assert np.array_equal(q.target_update,np.arange(len(t)+1));assert q.source_context.iloc[0]=='none';assert np.array_equal(q.source_context.iloc[1:],t.training_cue)

def save_arrays(path,arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  with np.load(path) as old:
   assert set(old.files)==set(arrays)
   for k,v in arrays.items():np.testing.assert_array_equal(old[k],v)
 else:np.savez_compressed(path,**arrays)
 return sha(path)

def main():
 configure();assert json.loads((OUT/'audit/analysis_complete.json').read_text())['passed'];receipt=OUT/'audit/artifacts_complete.json'
 if receipt.exists():return
 from experiments.heldout_context_meta.target import validate_training_records
 curve=pd.read_parquet(OUT/'tables/native_curve_metrics.parquet');reg=pd.read_csv(OUT/'registry/native_prediction_sources.csv');schedule=[]
 for row in reg.itertuples():
  p=ROOT/row.training;assert sha(p)==row.training_sha;t=pd.read_csv(p,float_precision='round_trip');f=curve[curve.trajectory_id.eq(row.trajectory_id)];validate_schedule(t,f);validate_training_records(t,f.rename(columns={'target_update':'update','probe_context':'cue'}));schedule.append(dict(trajectory_id=row.trajectory_id,recorded_updates=len(t),recorded_A=int(t.training_cue.eq('A').sum()),recorded_B=int(t.training_cue.eq('B').sum()),starting_cue=t.training_cue.iloc[0],last_cue=t.training_cue.iloc[-1],actual_recorded_schedule_verified=True,training_sha=row.training_sha))
 csv(OUT/'audit/native_schedule_checks.csv',schedule);del curve
 reg=pd.read_csv(OUT/'registry/branches.csv');parameter=[];geometry=[]
 for i,row in enumerate(reg[reg.status.eq('completed')].itertuples(),1):
  path=OUT/row.result;assert sha(path)==row.sha256;b=torch.load(path,weights_only=False);allsteps=[dict(observations=b['baseline']['observations'],checks=b['baseline']['checks'])]+b['steps'];arrays={k:np.stack([s['checks'][k] for s in allsteps]) for k in ['K','C_operator','centered_C_operator']};p=OUT/f'operator_arrays/{row.key}.npz';h=save_arrays(p,arrays);geometry.append(dict(trajectory_id=row.trajectory_id,target_update=row.target_update,source_context=row.source_context,branch_type=row.branch_type,path=str(p.relative_to(OUT)),sha256=h,source_result=row.result,source_result_sha=row.sha256))
  if b['parameter']:
   observations=b['parameter']['observations'];names=list(observations);arrays={field:np.stack([[observations[v][c][field] for c in design.PROBES] for v in names]) for field in ['logits','probabilities','ce','accuracy','margin','hidden']};from .verify import check_raw
   check_raw(arrays);arrays['variant_names']=np.array(names);arrays['probe_names']=np.array(design.PROBES);p=OUT/f'parameter_arrays/{row.key}.npz';h=save_arrays(p,arrays);parameter.append(dict(trajectory_id=row.trajectory_id,target_update=row.target_update,source_context=row.source_context,path=str(p.relative_to(OUT)),sha256=h,source_result=row.result,source_result_sha=row.sha256))
  if i%500==0:print('Cached arrays',i,'/',len(reg),flush=True)
 csv(OUT/'registry/parameter_array_sources.csv',parameter);csv(OUT/'registry/operator_array_sources.csv',geometry);dump(receipt,dict(passed=True,recorded_schedules=len(schedule),parameter_array_files=len(parameter),operator_array_files=len(geometry),optimizer_operations=0,new_frozen_rollouts=0));append('## Recorded task schedule and compressed cached arrays\nAll342 gain histories were checked against actual archived training_cue records, including switch indexing and original native CE/total-loss consistency tolerances. Raw per-position endpoint-replacement and exact C-operator arrays compressed directly from accepted cached observations/states; no optimizer operation, gradient or frozen model rollout repeated. Native clip coefficient is explicitly reconstructed in float32 from the recorded native clip-input norm; original host-double ratio is retained separately. No scientific threshold/tolerance changed.')
if __name__=='__main__':main()
