"""Saved-table audit only: no model or training imports."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'results/context_mixing_sequence_continuity/v1'
DEFAULT=ROOT/'results/familiar_acquisition_generalization/v1'
SEQ=(0,1,2,0,3,1,2,3)
CUES=['A','B','Mixed','NoContext','Novel30_A','Novel30_B','Novel60_A','Novel60_B']
ARMS=['Original','B','I','S']
CHANCE=float(np.log(4))
H=[f'hidden_{i}' for i in range(8)];Z=[f'logit_{i}' for i in range(4)]
POLICY={'bootstrap_draws':10000,'bootstrap_seed':20260925,'strong_familiar_max_ce_each':.5,'strong_familiar_min_accuracy_each':6/7,'substantial_familiar_change':-.2,'retained_hidden_and_logit_ratio':.8,'success_max_gap':.1,'success_min_accuracy':6/7,'uniform_ce':CHANCE}


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(name):return pd.read_csv(PRIOR/'tables'/f'{name}.csv',float_precision='round_trip')

def validate(f):
 expected={(s,a,c,t) for s in range(20) for a in ARMS for c in CUES for t in range(1,8)}
 keys=['seed','arm','cue','transition']
 if f.duplicated(keys).any() or set(f[keys].itertuples(index=False,name=None))!=expected:raise ValueError('Missing/duplicate/wrong-family seed-arm-cue-position cells')
 if not ((f.step==np.where(f.arm=='Original',170,330))).all():raise ValueError('Wrong checkpoint update')
 if not np.array_equal(f.token,np.array(SEQ)[f.transition.to_numpy()-1]) or not np.array_equal(f.target,np.array(SEQ)[f.transition.to_numpy()]):raise ValueError('Wrong token/target indexing')
 if not np.isfinite(f[['raw_ce',*H,*Z]].to_numpy()).all():raise ValueError('Missing outcomes')
 if not f.groupby(['seed','arm']).checkpoint_sha.nunique().eq(1).all():raise ValueError('Checkpoint mixing')
 return True


def separation(a,b):
 a=a.sort_values('transition');b=b.sort_values('transition')
 dh=a[H].to_numpy()-b[H].to_numpy();za=a[Z].to_numpy();zb=b[Z].to_numpy();dz=(za-za.mean(1,keepdims=True))-(zb-zb.mean(1,keepdims=True))
 return float(np.linalg.norm(dh,axis=1).mean()),float(np.linalg.norm(dz,axis=1).mean())


def aggregate(f):
 validate(f);rows=[]
 for (seed,arm),g in f.groupby(['seed','arm']):
  r=dict(seed=seed,arm=arm,step=int(g.step.iloc[0]),checkpoint_sha=g.checkpoint_sha.iloc[0])
  for cue,c in g.groupby('cue'):
   c=c.sort_values('transition');r['ce_'+cue]=c.raw_ce.mean();r['accuracy_'+cue]=float((c[Z].to_numpy().argmax(1)==c.target.to_numpy()).mean())
   for q in c.itertuples():r[f'ce_{cue}_position_{q.transition}']=q.raw_ce
  for family,names in [('NovelMean',CUES[4:]),('Novel30Mean',CUES[4:6]),('Novel60Mean',CUES[6:])]:
   for metric in ['ce','accuracy']:r[f'{metric}_{family}']=np.mean([r[f'{metric}_{c}'] for c in names])
   for pos in range(1,8):r[f'ce_{family}_position_{pos}']=np.mean([r[f'ce_{c}_position_{pos}'] for c in names])
  r['familiar_ce']=(r['ce_A']+r['ce_B'])/2;r['familiar_accuracy']=(r['accuracy_A']+r['accuracy_B'])/2
  for probe in ['Mixed','NoContext','NovelMean','Novel30Mean','Novel60Mean']:
   r['gap_'+probe]=r['ce_'+probe]-r['familiar_ce'];r['minus_uniform_'+probe]=r['ce_'+probe]-CHANCE
  r['hidden_separation'],r['logit_separation']=separation(g[g.cue=='A'],g[g.cue=='B']);rows.append(r)
 out=pd.DataFrame(rows)
 original=out[out.arm=='Original'].set_index('seed')
 for metric in ['hidden_separation','logit_separation']:
  denominator=out.seed.map(original[metric]);out[metric.replace('_separation','_retention')]=out[metric]/denominator.where(denominator>1e-12)
 return out


def classify(r):
 strong=r['ce_A']<=.5 and r['ce_B']<=.5 and r['accuracy_A']>=6/7 and r['accuracy_B']>=6/7
 retained=r['hidden_retention']>=.8 and r['logit_retention']>=.8
 poor=r['ce_Mixed']>=CHANCE;success=r['ce_Mixed']<CHANCE and r['accuracy_Mixed']>=6/7 and r['gap_Mixed']<=.1
 return dict(strong_familiar=strong,perfect_familiar=strong and r['accuracy_A']==1 and r['accuracy_B']==1,retained_separation=retained,poor_mixed=poor,successful_mixed=success,strong_retained_poor=strong and retained and poor,strong_retained_success=strong and retained and success)


def compare(q):
 original=q[q.arm=='Original'].set_index('seed');rows=[]
 metrics=['familiar_ce','familiar_accuracy','ce_Mixed','gap_Mixed','ce_NoContext','gap_NoContext','ce_NovelMean','gap_NovelMean','hidden_separation','logit_separation']
 for r in q[q.arm!='Original'].to_dict('records'):
  b=original.loc[r['seed']];row=dict(seed=r['seed'],arm=r['arm'],original_checkpoint_sha=b.checkpoint_sha,final_checkpoint_sha=r['checkpoint_sha'])
  for m in metrics:row['delta_'+m]=r[m]-b[m]
  row['substantial_familiar_improvement']=row['delta_familiar_ce']<=-.2;row['mixed_absolute_improvement']=row['delta_ce_Mixed']<0;rows.append(row)
 return pd.DataFrame(rows)


def bootstrap(f,value='effect'):
 if sorted(f.seed.tolist())!=list(range(20)):raise ValueError('Exactly20 unique whole-seed values required; no pseudoreplication')
 x=f.sort_values('seed')[value].to_numpy(float)
 if not np.isfinite(x).all():return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n=20,status='missing_value')
 draw=np.random.default_rng(20260925).integers(0,20,(10000,20));lo,hi=np.quantile(x[draw].mean(1),[.025,.975])
 return dict(mean=x.mean(),ci_low=lo,ci_high=hi,n=20,status='whole_seed_bootstrap')


def sources():
 seal=json.loads((PRIOR/'manifests/sealed_manifest.json').read_text());names=['d0_native_replay','d3_natural_transition','d3_natural_seed','d3_separation_by_position','d3_model_registry','d0_checkpoint_audit','d4_original_separation_seed'];inputs={str((PRIOR/'tables'/f'{n}.csv').relative_to(ROOT)):seal['files'][f'tables/{n}.csv'] for n in names}
 for name,h in seal['code_hashes'].items():inputs[name]=h
 originals=read('d0_checkpoint_audit');registry=read('d3_model_registry')
 if sorted(originals.seed)!=list(range(20)) or not originals.condition.eq('High_monotonic').all() or not originals.checkpoint_step.eq(170).all():raise ValueError('Original cohort identity')
 for r in originals.itertuples():inputs[r.path]=r.checkpoint_sha256
 if len(registry)!=60 or registry.duplicated(['seed','arm']).any() or not registry.status.eq('completed').all() or not registry.fixed_endpoint.eq(330).all() or not registry.updates_executed.eq(160).all():raise ValueError('Continuation endpoint mismatch')
 original_sha=originals.set_index('seed').checkpoint_sha256.to_dict()
 for r in registry.itertuples():
  if r.original_checkpoint_sha!=original_sha[r.seed] or r.arm not in ['B','I','S']:raise ValueError('Unpaired original')
  inputs[str((PRIOR/r.checkpoint_path).relative_to(ROOT))]=r.checkpoint_sha
 inputs[str((PRIOR/'manifests/sealed_manifest.json').relative_to(ROOT))]=sha(PRIOR/'manifests/sealed_manifest.json')
 for name,h in inputs.items():
  if sha(ROOT/name)!=h:raise ValueError('Source/checkpoint hash mismatch: '+name)
 return inputs,originals,registry


def run(out=DEFAULT):
 out=Path(out);(out/'tables').mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
 if (out/'tables/seed_level.csv').exists():raise FileExistsError('Immutable completed audit')
 inputs,originals,registry=sources();a=read('d0_native_replay');b=read('d3_natural_transition');a['arm']='Original';a['step']=170;b['step']=330
 if not a.cohort.eq('canonical').all() or not a.condition.eq('High_monotonic').all():raise ValueError('Wrong original evaluation family')
 for r in registry.itertuples():
  g=b[(b.seed==r.seed)&(b.arm==r.arm)]
  if not g.checkpoint_sha.eq(r.checkpoint_sha).all() or not g.original_checkpoint_sha.eq(r.original_checkpoint_sha).all():raise ValueError('Evaluation checkpoint mismatch')
 f=pd.concat([a,b],ignore_index=True);validate(f);f['correct']=(f[Z].to_numpy().argmax(1)==f.target.to_numpy());z=f[Z].to_numpy();f['max_logit_tie']=((z==z.max(1,keepdims=True)).sum(1)>1)
 expected_cues={'A':[1,0,0],'B':[0,1,0],'Mixed':[.5,.5,0],'NoContext':[0,0,0],'Novel30_A':[np.sqrt(3)/2,-.5,0],'Novel30_B':[-.5,np.sqrt(3)/2,0],'Novel60_A':[.5,-np.sqrt(3)/2,0],'Novel60_B':[-np.sqrt(3)/2,.5,0]}
 for cue,v in expected_cues.items():
  if not np.allclose(f.loc[f.cue==cue,['cue_0','cue_1','cue_2']],v,atol=1e-14,rtol=0):raise ValueError('Physical cue mismatch')
 # Independently recompute CE from the entire saved current logit vector.
 maxz=z.max(1);ce=maxz+np.log(np.exp(z-maxz[:,None]).sum(1))-z[np.arange(len(z)),f.target.to_numpy(int)]
 if np.max(abs(ce-f.raw_ce))>1e-12:raise ValueError('Saved CE/head mismatch')
 q=aggregate(f);flags=pd.DataFrame([classify(r) for r in q.to_dict('records')]);q=pd.concat([q,flags],axis=1);changes=compare(q)
 check=[];oldsep=read('d3_separation_by_position').groupby(['seed','arm'])[['hidden_separation','task_separation']].mean()
 oldorig=read('d4_original_separation_seed').set_index('seed');oldce=read('d3_natural_seed')
 for r in q.itertuples():
  old=oldorig.loc[r.seed] if r.arm=='Original' else oldsep.loc[(r.seed,r.arm)]
  dh=abs(r.hidden_separation-old.hidden_separation);dz=abs(r.logit_separation-old.task_separation)
  if dh>1e-12 or dz>1e-4:raise ValueError('Separation definition mismatch')
  check.append(dict(seed=r.seed,arm=r.arm,hidden_error=dh,recorded_logits_vs_double_head_error=dz))
 for r in oldce.itertuples():
  val=q[(q.seed==r.seed)&(q.arm==r.arm)]['ce_'+r.cue].iloc[0]
  if abs(val-r.raw_ce)>1e-12:raise ValueError('Existing seed evaluation mismatch')
 stats=[]
 for arm,g in q.groupby('arm'):
  for m in ['familiar_ce','familiar_accuracy','ce_A','ce_B','ce_Mixed','accuracy_Mixed','gap_Mixed','minus_uniform_Mixed','ce_NoContext','gap_NoContext','ce_NovelMean','gap_NovelMean','hidden_separation','logit_separation']:
   stats.append(dict(arm=arm,type='endpoint',metric=m,**bootstrap(g,m)))
 for arm,g in changes.groupby('arm'):
  for m in [x for x in changes if x.startswith('delta_')]:stats.append(dict(arm=arm,type='paired_change',metric=m,**bootstrap(g,m)))
 stats=pd.DataFrame(stats);assoc=[]
 for arm,g in changes.groupby('arm'):
  x=g.delta_familiar_ce;y=g.delta_ce_Mixed
  pearson=x.corr(y) if x.std()>0 and y.std()>0 else np.nan;spearman=x.rank().corr(y.rank()) if x.std()>0 and y.std()>0 else np.nan
  sub=g[g.substantial_familiar_improvement];assoc.append(dict(arm=arm,n=20,pearson_change=pearson,spearman_change=spearman,n_substantial=len(sub),n_substantial_mixed_improved=int(sub.mixed_absolute_improvement.sum()),mean_delta_mixed_substantial=sub.delta_ce_Mixed.mean(),mean_delta_familiar_substantial=sub.delta_familiar_ce.mean()))
 counts=q.groupby('arm')[list(flags)].sum().reset_index();sensitivity=[]
 for arm,g in q.groupby('arm'):
  for threshold in [.25,.5,1.]:
   for retention in [.5,.8,1.]:
    good=(g.ce_A<=threshold)&(g.ce_B<=threshold)&(g.accuracy_A>=6/7)&(g.accuracy_B>=6/7);retain=(g.hidden_retention>=retention)&(g.logit_retention>=retention)
    sensitivity.append(dict(arm=arm,ce_each_max=threshold,retention_min=retention,n_good=int(good.sum()),n_good_retained_poor=int((good&retain&g.poor_mixed).sum()),n_good_retained_success=int((good&retain&g.successful_mixed).sum())))
 outputs={'seed_level':q,'paired_changes':changes,'bootstrap_summary':stats,'descriptive_associations':pd.DataFrame(assoc),'operational_counts':counts,'threshold_sensitivity':pd.DataFrame(sensitivity),'separation_reuse_checks':pd.DataFrame(check),'transition_level':f}
 for name,df in outputs.items():df.to_csv(out/'tables'/f'{name}.csv',index=False)
 (out/'audit/source_hashes.json').write_text(json.dumps(inputs,indent=2)+'\n');(out/'audit/fixed_policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
 from .figures import generate
 generate(out)
 for name,h in inputs.items():
  if sha(ROOT/name)!=h:raise ValueError('Source/checkpoint changed during audit')
 info={'models':80,'original_models':20,'continued_endpoints':60,'transition_rows':len(f),'saved_ce_max_error':float(np.max(abs(ce-f.raw_ce))),'ties':int(f.max_logit_tie.sum()),'checkpoint_files_verified':80,'model_loads':0,'reevaluations':0,'training_updates':0,'max_output_separation_roundoff':max(x['recorded_logits_vs_double_head_error'] for x in check)}
 (out/'audit/verification.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
 return q,changes,stats,pd.DataFrame(assoc),counts

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT));args=p.parse_args();run(Path(args.output))
