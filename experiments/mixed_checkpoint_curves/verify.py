"""Read-only table/cached-logit verification; never replays a model."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import ROOT,SOURCE,OUT,sha,source_guard,summary,STEPS,ARCHS,CHANCE

def hashes(root,items):
 for n,h in items.items():
  p=(root/n).resolve()
  if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError('Hash mismatch: '+n)

def verify(out=OUT):
 out=Path(out);audit=json.loads((out/'audit/evaluation_complete.json').read_text());assert source_guard()==audit['source_manifest_sha'];hashes(SOURCE,audit['inputs']);hashes(ROOT,audit['source_code']);read=lambda n:pd.read_csv(out/'tables'/n,float_precision='round_trip')
 v=read('seed_values.csv');s=read('summary.csv');pd.testing.assert_frame_equal(s,summary(v));assert len(v)==360
 from .figure import validate
 validate(v);checkpoints=read('checkpoint_hashes.csv');assert len(checkpoints)==120
 for r in checkpoints.itertuples():assert sha(ROOT/r.path)==r.sha256
 controls=read('linear_midpoint_controls.csv');bounds=read('linear_float32_position_bounds.csv');assert len(controls)==60 and controls.passed.all() and len(bounds)==420 and bounds.passed.all()
 assert (controls.double_hidden_error<=controls.double_tolerance).all() and (controls.double_logit_error<=controls.double_tolerance).all() and (controls.double_ce_gap<=2*controls.double_tolerance).all()
 assert (bounds.ce_excess<=bounds.ce_excess_upper_bound).all()
 p=read('available_position_predictions.csv');assert len(p)==1540
 for key,f in p.groupby(['seed','architecture','checkpoint','cue']):
  f=f.sort_values('position');assert f.position.tolist()==list(range(1,8));z=f[[f'logit_{j}' for j in range(4)]].to_numpy();mx=z.max(1);y=f.target.to_numpy(int);ce=mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(7),y];np.testing.assert_allclose(ce,f.ce,atol=1e-12,rtol=1e-10)
  r=v[(v.seed==key[0])&(v.architecture==key[1])&(v.checkpoint==key[2])&(v.cue==key[3])].iloc[0];np.testing.assert_allclose([ce.mean(),(z.argmax(1)==y).mean()],[r.ce,r.accuracy],atol=1e-12,rtol=1e-10)
 old=pd.read_csv(SOURCE/'generalization.csv',float_precision='round_trip');l=pd.read_csv(SOURCE/'learning_trajectories.csv',float_precision='round_trip')
 for r in v.itertuples():
  if r.checkpoint==170:f=old[(old.seed==r.seed)&(old.architecture==r.architecture)&(old.endpoint==170)&(old.cue==r.cue)]
  elif r.cue!='Mixed':f=l[(l.seed==r.seed)&(l.architecture==r.architecture)&(l['update']==r.checkpoint)&(l.cue==r.cue)]
  else:continue
  assert len(f)==1;np.testing.assert_array_equal([r.ce,r.accuracy],[f.iloc[0].ce,f.iloc[0].accuracy])
 assert audit['counts']['new_float32_rollouts']==160 and audit['counts']['new_float64_math_rollouts']==120 and audit['counts']['reused_170_cue_cells']==120
 competence=read('familiar_competence.csv');assert len(competence)==120
 for r in competence.itertuples():
  f=v[(v.seed==r.seed)&(v.architecture==r.architecture)&(v.checkpoint==r.checkpoint)].set_index('cue')
  strong=bool(f.loc['A','ce']<=.5 and f.loc['B','ce']<=.5 and f.loc['A','accuracy']>=6/7 and f.loc['B','accuracy']>=6/7)
  assert r.strong_familiar==strong and r.Mixed_above_chance==bool(f.loc['Mixed','ce']>CHANCE)
  assert r.strong_and_Mixed_above_chance==bool(strong and r.Mixed_above_chance)
 counts=read('competence_counts.csv')
 for r in counts.itertuples():
  f=competence[(competence.architecture==r.architecture)&(competence.checkpoint==r.checkpoint)]
  assert r.n_seeds==20 and r.strong_familiar==int(f.strong_familiar.sum()) and r.strong_and_Mixed_above_chance==int(f.strong_and_Mixed_above_chance.sum())
 if (out/'manifest.json').exists():
  m=json.loads((out/'manifest.json').read_text());hashes(out,m['outputs']);hashes(ROOT,m['code'])
 return dict(passed=True,cue_cells=360,seeds_per_architecture=20,linear_checkpoint_controls=60,linear_position_bounds=420,original_checkpoint_hashes_unchanged=True,training_updates=0,verification_model_replays=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--regenerate',type=Path);a=p.parse_args();r=verify()
 if a.regenerate:
  from .figure import generate
  names=generate(OUT,a.regenerate)
  for n in names:assert sha(a.regenerate/n)==sha(OUT/'figures'/n),n
  r['byte_identical_regenerated_figures']=len(names)
 print(json.dumps(r,indent=2))

if __name__=='__main__':main()
