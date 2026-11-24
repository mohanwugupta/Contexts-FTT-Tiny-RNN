"""Read-only checkpoint/budget/source/table verification; never trains or rolls out."""
import argparse
from types import SimpleNamespace
import pandas as pd
from .core import *
from .analysis import bootstrap,blocks,quality

def validate_registry(r):
 if r.duplicated(['seed','architecture','checkpoint']).any() or set(r[['seed','architecture','checkpoint']].itertuples(index=False,name=None))!={(s,a,k) for s in range(20) for a in ARCHS for k in STEPS}:raise ValueError('All240 checkpoint slots required')
 return True

def hashes(root,d):
 for n,h in d.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError('Hash mismatch: '+n)

def verify(out=OUT):
 out=Path(out);read=lambda n:pd.read_csv(out/n,float_precision='round_trip');load=lambda n:json.loads((out/n).read_text());gate=load('audit/pretraining_gate.json');hashes(ROOT,gate['inputs']);hashes(ROOT,gate['code']);seal=load('audit/training_seal.json');hashes(out,seal['checkpoint_hashes']);hashes(ROOT,seal['code']);assert seal['protected_evaluations']==0
 normal=ROOT/'results/recurrent_learning_transfer/v1';old=json.loads((normal/'manifest.json').read_text());hashes(normal,old['outputs']);hashes(ROOT,old['code'])
 registry=read('tables/checkpoint_registry.csv');validate_registry(registry)
 for r in registry[registry.status=='completed'].itertuples():
  payload=torch.load(out/r.checkpoint_path,map_location='cpu',weights_only=False);assert (payload['seed'],payload['architecture'],payload['update'],payload['config'])==(r.seed,r.architecture,r.checkpoint,CONFIG)
  wrap=lambda d:SimpleNamespace(state_dict=lambda:d)
  assert state_hash(wrap(payload['model_state_dict']),wrap(payload['policy_state_dict']))==payload['model_policy_hash']
  opt=payload['optimizer_state_dict'];assert len(opt['param_groups'][0]['params'])==8 and len(opt['state'])==(8 if r.checkpoint else 0)
  for v in opt['state'].values():assert int(v['step'])==r.checkpoint
  assert not torch.count_nonzero(payload['model_state_dict']['recurrent.weight_ih_l0'][:,6])
  if r.architecture=='RNN-Tanh' and r.checkpoint==170:
   cp=torch.load(ROOT/f'outputs/confirmatory_srn_final/confirmatory_checkpoints/High_monotonic_seed_{r.seed}.pt',weights_only=False,map_location='cpu')
   assert all(torch.equal(v,cp['model_state_dict'][n]) for n,v in payload['model_state_dict'].items()) and all(torch.equal(v,cp['policy_state_dict'][n]) for n,v in payload['policy_state_dict'].items())
 training=read('training_measurements.csv');l=read('learning_trajectories.csv');assert len(training)==13200 and len(l)==26480;assert int(training.optimizer_step_executed.sum())==seal['scientific_updates_executed'];assert all(r.training_cue==training_cue(r.update) for r in training.itertuples())
 pd.testing.assert_frame_equal(read('splitting.csv'),blocks(l));v=read('generalization.csv');assert len(v)==680 and not v.duplicated(['seed','architecture','checkpoint','cue']).any()
 pos=read('tables/transition_predictions.csv')
 for key,f in pos.groupby(['seed','architecture','checkpoint','cue']):
  f=f.sort_values('position');assert f.position.tolist()==list(range(1,8));z=f[[f'logit_{j}' for j in range(4)]].to_numpy();mx=z.max(1);ce=mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(7),TARGET];prob=np.exp(z-mx[:,None]);prob/=prob.sum(1,keepdims=True);np.testing.assert_allclose(prob,f[[f'probability_{j}' for j in range(4)]],atol=1e-12,rtol=1e-10);np.testing.assert_allclose(ce,f.ce,atol=1e-10,rtol=1e-10)
  row=v[(v.seed==key[0])&(v.architecture==key[1])&(v.checkpoint==key[2])&(v.cue==key[3])].iloc[0];np.testing.assert_allclose([ce.mean(),(z.argmax(1)==TARGET).mean()],[row.ce,row.accuracy],atol=1e-10,rtol=1e-10)
 q=read('tables/endpoint_seed_summary.csv');assert len(q)==200
 for r in q.itertuples():
  flags=quality(r.A_ce,r.B_ce,r.A_accuracy,r.B_accuracy,r.Mixed_ce,r.Mixed_accuracy)
  for n,x in flags.items():assert getattr(r,n)==x
  np.testing.assert_allclose(r.Mixed_gap,r.Mixed_ce-(r.A_ce+r.B_ce)/2,atol=1e-10)
 for r in read('tables/endpoint_intervals.csv').itertuples():
  f=q[(q.architecture==r.architecture)&(q.checkpoint==r.checkpoint)];x=bootstrap(f,r.metric);np.testing.assert_allclose([r.mean,r.ci_low,r.ci_high],[x[k] for k in ['mean','ci_low','ci_high']],atol=1e-10,rtol=1e-10,equal_nan=True);assert r.n_valid==x['n_valid']
 controls=read('tables/linear_controls.csv');bounds=read('tables/linear_position_bounds.csv');assert len(controls)==100 and controls.passed.all() and len(bounds)==700 and bounds.passed.all();assert (bounds.ce_excess<=bounds.ce_excess_upper_bound).all()
 parity=read('tables/canonical_training_parity.csv');assert len(parity)==20 and parity.model_tensors_bitwise_equal.all() and parity.policy_tensors_bitwise_equal.all() and parity.preupdate_losses_bitwise_equal.all()
 if (out/'manifest.json').exists():
  m=load('manifest.json');hashes(out,m['outputs']);hashes(ROOT,m['code'])
 return dict(passed=True,scientific_updates=seal['scientific_updates_executed'],checkpoints=240,canonical170_exact_reproductions=20,Linear_midpoint_controls=100,all20_seeds_retained=True,original_checkpoints_unchanged=True,verification_training_updates=0,verification_model_rollouts=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--regenerate',type=Path);a=p.parse_args();r=verify()
 if a.regenerate:
  from .figures import generate
  files=generate(OUT,a.regenerate)
  for n in files:assert sha(a.regenerate/n)==sha(OUT/'figures'/n),n
  r['byte_identical_regenerated_figures']=len(files)
 print(json.dumps(r,indent=2))

if __name__=='__main__':main()
