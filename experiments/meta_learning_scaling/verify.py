"""Complete frozen source/state/table audit, without new forward evaluations."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .data import registry
from .target import validate_training_records
from experiments.context_experience_prior.verify import verify_position_rows,verify_component_rows

def require_grid(f,conditions,doses,updates,cues):
 keys=['seed','condition','dose','update','cue'];exp=pd.MultiIndex.from_product([range(20),conditions,doses,updates,cues],names=keys);actual=pd.MultiIndex.from_frame(f[keys])
 if actual.has_duplicates or len(actual)!=len(exp) or set(actual)!=set(exp):raise ValueError('Incomplete/duplicated planned grid')

def verify_shard(row):
 initialize_process();path=ROOT/row['path'];assert sha(path)==row['sha256'];b=torch.load(path,weights_only=False);assert len(b['states'])==331;c=b['C_initial'];good=0;unavailable=0
 for k,st in enumerate(b['states']):
  assert st['update']==k
  if st['status']!='completed':unavailable+=1;assert st['model'] is None;continue
  good+=1;assert torch.equal(st['model']['recurrent.weight_ih_l0'][:,6],c)
  if k==0:
   assert not st['optimizer']['state'] and all(torch.count_nonzero(v)==0 for v in st['policy'].values());init,_,_=source_initial(row['seed'],row['prior'],row['organization'],row['dose']);assert all(torch.equal(v,st['model'][n]) for n,v in init.items())
  for state in st['optimizer']['state'].values():
   assert int(state['step'])==k
   if tuple(state['exp_avg'].shape)==(8,7):assert torch.count_nonzero(state['exp_avg'][:,6])==torch.count_nonzero(state['exp_avg_sq'][:,6])==0
 dest=OUT/row['directory'];f=pd.read_csv(dest/'evaluations.csv');t=pd.read_csv(dest/'training.csv');validate_training_records(t[t.status=='completed'],f);assert len(f)==331*6 and len(t)==330;d=pd.read_csv(dest/'diagnostics.csv');assert len(d)==24 and d.main_unchanged.all();dg=d[d.diagnostic_status=='completed'];assert dg.optimizer_restored.all() and dg.main_start_hash.eq(dg.clone_start_hash).all()
 positions=pd.read_csv(dest/'position_predictions.csv.gz',float_precision='round_trip');assert len(positions)==331*6*7;valid=positions[positions.status=='completed'];verify_position_rows(valid);assert np.array_equal(valid.token, np.asarray(old.TARGET_PATH)[valid.position.to_numpy()-1]) and np.array_equal(valid.target,old.TARGET[valid.position.to_numpy()-1]);co=pd.read_csv(dest/'mathematical_checks.csv.gz');assert len(co)==good*7 and co.passed.all() and (co.double_operator_logit_error<=co.double_tolerance).all() and (co.float32_logit_error<=co.float32_logit_allowance+co.double_tolerance).all();dc=pd.read_csv(dest/'decomposition.csv.gz',float_precision='round_trip');assert len(dc)==good*7;verify_component_rows(dc)
 return dict(run_id=row['run_id'],states=331,finite_states=good,unavailable_states=unavailable,positions=len(positions),new_target_steps=int(t.optimizer_step_executed.sum()),new_diagnostic_steps=int(d.diagnostic_update_executed.sum()/2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['a','full'],required=True);stage=p.parse_args().stage;dest=OUT/'stage_a' if stage=='a' else OUT;dest.mkdir(exist_ok=True);r=registry('a' if stage=='a' else 'full');new=r[~r.reused];rows=[];start=time.monotonic()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for row in pool.map(verify_shard,new.to_dict('records')):rows.append(row)
 verify_parent();write_frame(dest/'verified_targets.csv',rows);dump(dest/'verification.json',dict(passed=True,new_runs=len(rows),reused_runs=int(r.reused.sum()),finite_states=sum(x['finite_states'] for x in rows),unavailable_states=sum(x['unavailable_states'] for x in rows),positions=sum(x['positions'] for x in rows),new_target_updates=sum(x['new_target_steps'] for x in rows),new_diagnostic_updates=sum(x['new_diagnostic_steps'] for x in rows),source_unchanged=True,wall_seconds=time.monotonic()-start));print('Frozen integrity verified '+stage,flush=True)
if __name__=='__main__':main()
