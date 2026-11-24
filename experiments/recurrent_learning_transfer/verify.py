"""Read-only integrity, budget and independent saved-table checks; never trains."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from .design import ROOT,OUT,CONFIG,ARCHS,DIAG,ENDPOINTS,TARGET,SEQ,hash_file,training_cue
from .learning import state_hash

def verify_hashes(root,expected):
 root=Path(root).resolve()
 for name,digest in expected.items():
  p=(root/name).resolve()
  if not p.is_relative_to(root) or not p.is_file() or hash_file(p)!=digest:raise ValueError(f'Integrity failure: {name}')
 return len(expected)

def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-11)

def verify(out=OUT):
 out=Path(out);read=lambda n:pd.read_csv(out/n);load=lambda n:json.loads((out/n).read_text())
 gate=load('audit/pretraining_gate.json');verify_hashes(ROOT,gate['inputs'])
 for stage in ['pretraining_gate','training_seal','probes_complete','analysis_complete']:verify_hashes(ROOT,load(f'audit/{stage}.json')['code'])
 seal=load('audit/training_seal.json');verify_hashes(out,seal['checkpoint_hashes']);verify_hashes(out,load('audit/report_products.json'))
 assert seal['protected_evaluations']==0 and seal['scientific_updates_executed']==13200 and seal['diagnostic_updates_executed']==640
 payload_hashes={}
 for name in seal['checkpoint_hashes']:
  p=torch.load(out/name,map_location='cpu',weights_only=False);assert p['config']==CONFIG
  def wrapped(d):return SimpleNamespace(state_dict=lambda:d)
  h=state_hash(wrapped(p['model_state']),wrapped(p['policy_state']),wrapped(p['optimizer_state']));assert h==p['scientific_state_hash']
  k=p['update'];assert k in DIAG;assert len(p['optimizer_state']['param_groups'][0]['params'])==8
  assert len(p['optimizer_state']['state'])==(8 if k else 0)
  for s in p['optimizer_state']['state'].values():assert int(s['step'])==k
  assert sum(v.numel() for v in p['model_state'].values())==172
  assert p['model_state']['wi'].shape==(8,7) and torch.equal(p['model_state']['wi'][:,6],torch.zeros(8))
  payload_hashes[(p['seed'],p['architecture'],k)]=h
 assert len(payload_hashes)==320
 l=read('learning_trajectories.csv');u=read('tables/training_history.csv');d=read('learning_transfer.csv')
 assert len(l)==26480 and len(u)==13200 and len(d)==1280
 assert not l.duplicated(['seed','architecture','update','cue']).any()
 assert not u.duplicated(['seed','architecture','update']).any()
 assert u.status.eq('completed').all() and u.optimizer_step_executed.all()
 assert all(r.training_cue==training_cue(r.update) for r in u.itertuples())
 assert len(d.drop_duplicates(['seed','architecture','checkpoint','train_context']))==640
 assert d.scientific_state_unchanged.all() and d.optimizer_restored.all() and d.diagnostic_update_executed.all()
 close(d.loss_before-d.loss_after,d.improvement)
 base=l.set_index(['seed','architecture','update','cue']).ce
 for r in d.itertuples():
  assert r.clone_start_hash==r.scientific_start_hash==payload_hashes[(r.seed,r.architecture,r.checkpoint)]
  close(r.loss_before,base.loc[(r.seed,r.architecture,r.checkpoint,r.evaluation_context)])
 w=read('tables/direct_specificity.csv').set_index(['seed','architecture','checkpoint'])
 for key,g in d.groupby(['seed','architecture','checkpoint']):
  v=g.set_index(['train_context','evaluation_context']).improvement;r=w.loc[key]
  da=v.loc[('A','A')]-v.loc[('A','B')];db=v.loc[('B','B')]-v.loc[('B','A')]
  close([r.D_A,r.D_B,r.mean_specificity],[da,db,(da+db)/2])
 b=read('tables/switch_costs.csv');assert len(b)==1320
 close(b.raw_cost,b.incoming_ce-b.outgoing_ce);close(b.adjusted_cost,b.raw_cost-b.initial_cue_difference)
 for r in b.itertuples():
  close(r.raw_cost,base.loc[(r.seed,r.architecture,r.boundary,r.incoming_context)]-base.loc[(r.seed,r.architecture,r.boundary,r.outgoing_context)])
  close(r.initial_cue_difference,base.loc[(r.seed,r.architecture,0,r.incoming_context)]-base.loc[(r.seed,r.architecture,0,r.outgoing_context)])
 s=read('tables/splitting_classification.csv');assert len(s)==80
 for r in s.itertuples():
  f=b[(b.seed==r.seed)&(b.architecture==r.architecture)].set_index('boundary')
  cy=np.array([.5*(f.loc[k,'raw_cost']+f.loc[k+10,'raw_cost']) for k in range(10,r.endpoint-9,20)])
  assert len(cy)==(8 if r.endpoint==170 else 16)==r.n_complete_cycles
  close(cy.mean(),r.mean_cycle_cost)
  boundary=cy.mean()>.05 and (cy>.05).mean()>=.5
  v=w.loc[(r.seed,r.architecture,r.endpoint)]
  direct=v.D_A>0 and v.D_B>0 and v.mean_specificity>.001 and v.mean_own_improvement>1e-4 and min(v.delta_A_A,v.delta_B_B)>-1e-6
  assert r.functional_split==bool(boundary and direct)
 g=read('generalization.csv');t=read('tables/transition_predictions.csv');assert len(g)==640 and len(t)==4480
 assert not g.duplicated(['seed','architecture','endpoint','cue']).any()
 gi=g.set_index(['seed','architecture','endpoint','cue'])
 for key,f in t.groupby(['seed','architecture','endpoint','cue']):
  f=f.sort_values('transition');assert len(f)==7;assert np.array_equal(f.target,TARGET) and np.array_equal(f.token,SEQ[:-1])
  z=f[[f'logit_{j}' for j in range(4)]].to_numpy();a=z.max(1);ce=a+np.log(np.exp(z-a[:,None]).sum(1))-z[np.arange(7),TARGET]
  prob=np.exp(z-a[:,None]);prob/=prob.sum(1)[:,None]
  close(prob,f[[f'probability_{j}' for j in range(4)]]);close(ce,f.ce);close(ce.mean(),gi.loc[key,'ce'])
  close((z.argmax(1)==TARGET).mean(),gi.loc[key,'accuracy'])
  close(ce,gi.loc[key,[f'ce_position_{j}' for j in range(1,8)]].astype(float))
 for r in g.itertuples():
  lf=.5*(gi.loc[(r.seed,r.architecture,r.endpoint,'A'),'ce']+gi.loc[(r.seed,r.architecture,r.endpoint,'B'),'ce'])
  close(r.familiar_ce,lf);close(r.gap,r.ce-lf);close(r.minus_uniform,r.ce-np.log(4))
 controls=read('tables/linear_controls.csv');assert len(controls)==40 and controls.passed.all() and controls.float32_bound_passed.all()
 links=read('tables/D1_source_linkage.csv');assert len(links)==120 and links.linked.all()
 dc=read('tables/tanh_D1_decomposition.csv');assert len(dc)==280 and dc.Torch_source_linked.all()
 assert dc.hidden_identity_residual.max()<1e-10 and dc.output_identity_residual.max()<1e-10
 assert (dc.observed_float32_residual<=dc.roundoff_bound).all()
 q=read('tables/endpoint_summary.csv');assert len(q)==80
 paired=read('tables/paired_architecture_effects.csv')
 for k in ENDPOINTS:
  f=q[q.endpoint==k].set_index(['seed','architecture']);p=paired[paired.endpoint==k].set_index('seed')
  for seed in range(20):close(p.loc[seed,'Mixed_gap'],f.loc[(seed,'RNN-Tanh'),'Mixed_gap']-f.loc[(seed,'RNN-Linear'),'Mixed_gap'])
 # Independently reconstruct seed intervals from the saved per-network estimands.
 for filename,frame,has_arch in [('tables/endpoint_intervals.csv',q,True),('tables/paired_intervals.csv',paired,False)]:
  for r in read(filename).itertuples():
   f=frame[frame.endpoint==r.endpoint]
   if has_arch:f=f[f.architecture==r.architecture]
   if r.qualification!='all':
    col={'strong':'strong_familiar','strong_split':'strong_split','common_strong':'common_strong','common_strong_split':'common_strong_split'}[r.qualification]
    f=f[f[col].eq(True)]
   x=f.sort_values('seed')[r.metric].to_numpy(float);x=x[np.isfinite(x)];assert len(x)==r.n_valid
   if not len(x):assert np.isnan(r.mean) and np.isnan(r.ci_low) and np.isnan(r.ci_high);continue
   ix=np.random.default_rng(20260925).integers(0,len(x),(10000,len(x)));low,high=np.quantile(x[ix].mean(axis=1),[.025,.975]);close([r.mean,r.ci_low,r.ci_high],[x.mean(),low,high])
 if (out/'manifest.json').exists():
  m=load('manifest.json');verify_hashes(out,m['outputs']);verify_hashes(ROOT,m['code'])
 return dict(passed=True,scientific_updates=13200,isolated_diagnostic_updates=640,checkpoints=320,protected_rows=640,transitions=4480,original_checkpoint_hashes_unchanged=True,training_replayed=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--regenerate',type=Path);a=parser.parse_args();result=verify()
 if a.regenerate:
  from .figures import generate
  names=generate(OUT,a.regenerate)
  for n in names:assert hash_file(a.regenerate/n)==hash_file(OUT/'figures'/n),n
  result['byte_identical_regenerated_figures']=len(names)
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
