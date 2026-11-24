"""Read-only source/cache/table verification and table-only figure regeneration."""
import argparse
from .core import *
from .analysis import aggregate_within_seed,paired330

def assert_archive_reuse(v,old):
 n=0
 for r in v[v.cue.isin(['A','B','Mixed'])].itertuples():
  f=old[(old.seed==r.seed)&(old.architecture==r.architecture)&(old.checkpoint==r.checkpoint)&(old.cue==r.cue)]
  assert len(f)==1;np.testing.assert_array_equal([r.ce,r.accuracy],[f.iloc[0].ce,f.iloc[0].accuracy]);n+=1
 return n

def hashes(root,items):
 for n,h in items.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError('Hash mismatch: '+n)

def verify(out=OUT):
 out=Path(out);load=lambda n:json.loads((out/n).read_text());read=lambda n:pd.read_csv(out/'tables'/n,float_precision='round_trip');audit=load('audit/evaluation_complete.json');assert guard_source()==audit['source_manifest_sha'];assert sha(REGISTRY_PATH)==audit['registry_sha'];hashes(ROOT,audit['code']);v=read('seed_values.csv');pos=read('position_predictions.csv');assert len(v)==640 and len(pos)==4480
 from .figures import validate
 validate(v);old=pd.read_csv(SOURCE/'generalization.csv',float_precision='round_trip');assert assert_archive_reuse(v,old)==240
 cp=read('checkpoint_hashes.csv');assert len(cp)==80;linear_payloads={}
 for r in cp.itertuples():
  assert sha(ROOT/r.path)==r.sha256;p=torch.load(ROOT/r.path,map_location='cpu',weights_only=False);assert not torch.count_nonzero(p['model_state_dict']['recurrent.weight_ih_l0'][:,6])
  if r.architecture=='RNN-Linear':linear_payloads[(r.seed,r.checkpoint)]=p['model_state_dict']
 archived_positions=pd.read_csv(SOURCE/'tables/transition_predictions.csv',float_precision='round_trip');keys=['seed','architecture','checkpoint','cue','position'];columns=keys+['token','target','ce','correct']+[f'hidden_{j}' for j in range(8)]+[f'logit_{j}' for j in range(4)]+[f'probability_{j}' for j in range(4)]
 left=pos[pos.cue.isin(['A','B','Mixed'])][columns].sort_values(keys).reset_index(drop=True);right=archived_positions[archived_positions.checkpoint.isin(STEPS)][columns].sort_values(keys).reset_index(drop=True);pd.testing.assert_frame_equal(left,right,check_exact=True)
 for key,f in pos.groupby(['seed','architecture','checkpoint','cue']):
  f=f.sort_values('position');assert f.position.tolist()==list(range(1,8));np.testing.assert_array_equal(f.token,SEQ[:-1]);np.testing.assert_array_equal(f.target,TARGET);z=f[[f'logit_{j}' for j in range(4)]].to_numpy();mx=z.max(1);ce=mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(7),TARGET];prob=np.exp(z-mx[:,None]);prob/=prob.sum(1,keepdims=True);np.testing.assert_allclose(prob,f[[f'probability_{j}' for j in range(4)]],atol=1e-12,rtol=1e-11);np.testing.assert_allclose(ce,f.ce,atol=1e-10,rtol=1e-10);np.testing.assert_array_equal(z.argmax(1),f.predicted_token)
  r=v[(v.seed==key[0])&(v.architecture==key[1])&(v.checkpoint==key[2])&(v.cue==key[3])].iloc[0];np.testing.assert_allclose([r.ce,r.accuracy],[ce.mean(),(z.argmax(1)==TARGET).mean()],atol=1e-10,rtol=1e-10)
 for (seed,a,k),f in v.groupby(['seed','architecture','checkpoint']):
  p=f.set_index('cue');lf=(p.loc['A','ce']+p.loc['B','ce'])/2;good=strong(p.loc['A','ce'],p.loc['B','ce'],p.loc['A','accuracy'],p.loc['B','accuracy']);np.testing.assert_allclose(f.gap,f.ce-lf,atol=1e-10,rtol=1e-10);assert f.strong_familiar.eq(good).all() and np.array_equal(f.above_chance,f.ce>CHANCE)
 aggregate=aggregate_within_seed(v);recorded=read('registered_aggregate_values.csv');pd.testing.assert_frame_equal(aggregate,recorded[aggregate.columns]);combined=read('all_cue_and_aggregate_values.csv');pairs,common=paired330(combined);pd.testing.assert_frame_equal(pairs,read('common_strong330_pairs.csv'));assert common==load('audit/config.json')['common_strong330_seeds']
 for r in read('common_strong330_paired_intervals.csv').itertuples():
  f=pairs[pairs.cue==r.cue];x=interval(f,r.metric,common);np.testing.assert_allclose([r.mean,r.ci_low,r.ci_high],[x[k] for k in ['mean','ci_low','ci_high']],atol=1e-9,rtol=1e-11);assert r.n_valid==len(common)
 for r in read('intervals.csv').itertuples():
  f=combined[(combined.architecture==r.architecture)&(combined.checkpoint==r.checkpoint)&(combined.cue==r.cue)].copy()
  if r.qualification!='all':
   mask=f.strong_familiar if r.qualification=='strong_familiar' else f.seed.isin(common);f[r.metric]=f[r.metric].where(mask,np.nan)
  x=interval(f,r.metric);np.testing.assert_allclose([r.mean,r.ci_low,r.ci_high],[x[k] for k in ['mean','ci_low','ci_high']],atol=1e-9,rtol=1e-11,equal_nan=True);assert r.n_valid==x['n_valid']
 controls=read('linear_affine_double.csv');bounds=read('linear_affine_float32.csv');assert len(controls)==320 and controls.passed.all() and len(bounds)==2240 and bounds.passed.all();assert (controls.hidden_max_error<=controls.tolerance).all() and (controls.logit_max_error<=controls.tolerance).all()
 for seed in range(20):
  for k in STEPS:
   d={}
   for c in NAMES:
    with np.load(out/'cache'/f'{seed}_RNN-Linear_{k}_{c}_float64.npz',allow_pickle=False) as cache:d[c]={key:cache[key].copy() for key in ['hidden','logits','ce']}
   for c in NAMES:assert affine_double(d,c)['passed']
   from types import SimpleNamespace
   st=linear_payloads[(seed,k)];mock=SimpleNamespace(recurrent=SimpleNamespace(weight_ih_l0=st['recurrent.weight_ih_l0'],weight_hh_l0=st['recurrent.weight_hh_l0'],bias_ih_l0=st['recurrent.bias_ih_l0'],bias_hh_l0=st['recurrent.bias_hh_l0']),sequence_head=SimpleNamespace(weight=st['sequence_head.weight'],bias=st['sequence_head.bias']));native={}
   for c in NAMES:
    f=pos[(pos.seed==seed)&(pos.architecture=='RNN-Linear')&(pos.checkpoint==k)&(pos.cue==c)].sort_values('position');native[c]=dict(hidden=f[[f'hidden_{j}' for j in range(8)]].to_numpy(),logits=f[[f'logit_{j}' for j in range(4)]].to_numpy())
   for c in NAMES:
    actual=pd.DataFrame(affine_float(mock,native,c));recorded=bounds[(bounds.seed==seed)&(bounds.checkpoint==k)&(bounds.cue==c)].sort_values('position');pd.testing.assert_frame_equal(actual.reset_index(drop=True),recorded[actual.columns].reset_index(drop=True))
 assert audit['new_float32_rollouts']==400 and audit['new_float64_math_rollouts']==320 and audit['reused_archived_cue_cells']==240
 if (out/'audit/legacy_native_linkage_source.json').exists():
  legacy=load('audit/legacy_native_linkage_source.json');assert sha(ROOT/legacy['source_path'])==legacy['source_sha']
 if (out/'manifest.json').exists():
  m=load('manifest.json');hashes(out,m['outputs']);hashes(ROOT,m['code'])
 return dict(passed=True,outcome_rows=640,position_rows=4480,unchanged_archived_cue_cells=240,common_strong330_n=len(common),Linear_affine_checks=320,position_bounds=2240,original_checkpoints_unchanged=True,training_updates=0,verification_model_replays=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--regenerate',type=Path);a=p.parse_args();r=verify()
 if a.regenerate:
  from .figures import generate
  files=generate(OUT,a.regenerate)
  for n in files:assert sha(a.regenerate/n)==sha(OUT/'figures'/n),n
  r['byte_identical_regenerated_figures']=len(files)
 print(json.dumps(r,indent=2))

if __name__=='__main__':main()
