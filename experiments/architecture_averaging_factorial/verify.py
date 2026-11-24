"""Independent saved-table checks, immutable seal and read-only figure regeneration."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
import pandas as pd
from .protocol import *


def check_output_files(out,files):
 out=Path(out)
 for name,h in files.items():
  if not (out/name).is_file() or hash_file(out/name)!=h:raise ValueError('Missing/modified output: '+name)
 actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','.DS_Store']}
 if actual!=set(files):raise ValueError('Unexpected/missing scientific output bytes')
 return True

def _check_mapping(base,files):
 for name,h in files.items():
  if hash_file(Path(base)/name)!=h:raise ValueError('Source/checkpoint/code changed: '+name)

def table_checks(out=OUT):
 out=Path(out);q=pd.read_csv(out/'seed_level.csv',float_precision='round_trip');t=pd.read_csv(out/'transition_level.csv',float_precision='round_trip');reg=pd.read_csv(out/'tables/checkpoint_registry.csv');history=pd.read_csv(out/'tables/training_history.csv');g=pd.read_csv(out/'tables/geometry_by_position.csv',float_precision='round_trip')
 if len(q)!=320 or len(t)!=17920 or q.duplicated(['seed','architecture','step']).any() or len(history)!=80000 or not history.optimizer_step_executed.eq(True).all():raise ValueError('Coverage/scientific budget mismatch')
 if not q.status.eq('completed').all() or not reg.status.eq('completed').all():raise ValueError('Unexpected incomplete run')
 for (seed,a),f in history.groupby(['seed','architecture']):
  if f['update'].tolist()!=list(range(1,1001)) or f.training_cue.tolist()!=['A' if i%2 else 'B' for i in range(1,1001)]:raise ValueError('Training schedule/leakage/budget mismatch')
 z=t[[f'logit_{i}' for i in range(4)]].to_numpy();y=t.target.to_numpy(int);mx=z.max(1);ce=mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(len(z)),y]
 if np.max(abs(ce-t.raw_ce))>1e-12 or not np.array_equal(z.argmax(1)==y,t.correct):raise ValueError('Saved prediction score mismatch')
 model=q.set_index(['seed','architecture','step']);lookup=g.set_index(['seed','architecture','step','transition'])
 for (seed,a,k),f in t.groupby(['seed','architecture','step']):
  row=model.loc[(seed,a,k)]
  for cue,gg in f.groupby('cue'):
   gg=gg.sort_values('transition')
   if gg.transition.tolist()!=list(range(1,8)) or not np.array_equal(gg.token,SEQ[:-1]) or not np.array_equal(gg.target,TARGET):raise ValueError('Incorrect scored sequence')
   if not np.array_equal(gg[[f'cue_{i}' for i in range(3)]].to_numpy(),np.tile(CUES[cue],(7,1))):raise ValueError('Current/future physical cue mismatch')
   if abs(gg.raw_ce.mean()-row[cue+'_ce'])>1e-12 or abs(gg.correct.mean()-row[cue+'_accuracy'])>1e-12:raise ValueError('Seed aggregation mismatch')
  aa=f[f.cue=='A'].sort_values('transition');bb=f[f.cue=='B'].sort_values('transition');mm=f[f.cue=='Mixed'].sort_values('transition');hs=[x[[f'hidden_{i}' for i in range(8)]].to_numpy() for x in [aa,bb,mm]];zs=[x[[f'logit_{i}' for i in range(4)]].to_numpy() for x in [aa,bb,mm]];eh=hs[2]-(hs[0]+hs[1])/2;ez=zs[2]-(zs[0]+zs[1])/2;ez-=ez.mean(1,keepdims=True)
  for j in range(7):
   rr=lookup.loc[(seed,a,k,j+1)]
   if abs(rr.hidden_midpoint_residual-np.linalg.norm(eh[j]))>1e-12 or abs(rr.centered_logit_midpoint_residual-np.linalg.norm(ez[j]))>1e-12:raise ValueError('Midpoint geometry mismatch')
  if abs(row.Mixed_gap-(row.Mixed_ce-(row.A_ce+row.B_ce)/2))>1e-12:raise ValueError('Absolute loss and gap conflated')
 linear=q[q.activation=='identity']
 if len(linear)!=160 or not linear.linear_control_passed.eq(True).all() or (linear.double_ce_gap>2*linear.double_tolerance).any():raise ValueError('Affine control failure')
 c=pd.read_csv(out/'tables/paired_contrasts.csv',float_precision='round_trip');pis=pd.read_csv(out/'tables/paired_intervals.csv',float_precision='round_trip');cis=pd.read_csv(out/'tables/architecture_intervals.csv',float_precision='round_trip');primary=q[q.step==1000]
 gap=primary.pivot(index='seed',columns='architecture',values='Mixed_gap');rec=gap['RNN-Tanh']-gap['FF-Tanh'];inter=rec-(gap['RNN-Linear']-gap['FF-Linear']);saved=c[c.step==1000].set_index('seed').sort_index()
 if not np.allclose(rec,saved.recurrence_tanh,atol=1e-12,rtol=0) or not np.allclose(inter,saved.interaction,atol=1e-12,rtol=0):raise ValueError('Factorial contrast mismatch')
 draw=np.random.default_rng(20260925).integers(0,20,(10000,20))
 def interval_check(x,s):
  lo,hi=np.quantile(np.asarray(x)[draw].mean(1),[.025,.975])
  if not np.allclose([np.mean(x),lo,hi],[s['mean'],s.ci_low,s.ci_high],atol=1e-12,rtol=0):raise ValueError('Independent seed-bootstrap mismatch')
 for metric,x in [('recurrence_tanh',rec),('interaction',inter)]:interval_check(x,pis[(pis.step==1000)&(pis.qualification=='all')&(pis.metric==metric)].iloc[0])
 for arch,frame in primary.groupby('architecture'):
  for metric in ['familiar_ce','Mixed_ce','Mixed_gap']:interval_check(frame.sort_values('seed')[metric],cis[(cis.step==1000)&(cis.architecture==arch)&(cis.qualification=='all')&(cis.metric==metric)].iloc[0])
 return dict(passed=True,models=80,checkpoints=320,scientific_updates=80000,transition_rows=17920,linear_controls=160,independent_primary_intervals=14,all_sources_frozen=True)

def run(out=OUT,seal=False,regenerate=None):
 out=Path(out);gate=json.loads((out/'audit/pretraining_gate.json').read_text());train=json.loads((out/'audit/training_seal.json').read_text());evaluation=json.loads((out/'audit/evaluation_complete.json').read_text())
 _check_mapping(ROOT,gate['inputs'])
 for code in [gate['code'],train['code'],evaluation['code']]:_check_mapping(ROOT,code)
 _check_mapping(out,train['checkpoint_hashes'])
 if seal:
  if (out/'manifest.json').exists():raise FileExistsError('Immutable final seal')
  result=table_checks(out);dump(out/'audit/final_table_checks.json',result);files={str(p.relative_to(out)):hash_file(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','.DS_Store']};dump(out/'manifest.json',dict(status='completed_without_posthoc_changes',inputs=gate['inputs'],code=codes(),files=files,scientific_updates=80000,original_model_updates=0,primary=1000,seeds=20,architectures=4))
 m=json.loads((out/'manifest.json').read_text());_check_mapping(ROOT,m['inputs']);_check_mapping(ROOT,m['code']);check_output_files(out,m['files'])
 if regenerate:
  from .figures import generate
  destination=Path(regenerate)
  if destination.exists() and any(destination.iterdir()):raise FileExistsError('Empty new regeneration destination required')
  names=generate(out,destination)
  for name in names:
   if hash_file(destination/name)!=hash_file(out/'figures'/name):raise ValueError('Table-only regeneration differs: '+name)
  print(json.dumps(dict(regenerated_figures=4,files=8,all_hashes_identical=True,scientific_model_calls=0)),flush=True)
 print(json.dumps(dict(passed=True,files=len(m['files']),scientific_updates=80000,original_model_updates=0,checkpoints=320)),flush=True);return m

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));p.add_argument('--seal',action='store_true');p.add_argument('--regenerate');a=p.parse_args();run(a.output,a.seal,a.regenerate)
