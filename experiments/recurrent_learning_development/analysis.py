"""Saved-table-only descriptive learning development; no model or training imports."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/recurrent_learning_transfer/v1'
OUT=ROOT/'results/recurrent_learning_development/v1'
ARCHS=['RNN-Linear','RNN-Tanh'];FLOOR=.05;DIAG=[0,10,20,40,80,120,170,330]
EPOCHS=[('initial_0_40',0,40),('developing_40_120',40,120),('primary_120_170',120,170),('secondary_170_330',170,330)]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
 return h.hexdigest()

def blocks(learning,ends=range(10,331,10)):
 if set(learning.cue)!= {'A','B'} or learning.duplicated(['seed','architecture','update','cue']).any():raise ValueError('Unique paired A/B frozen evaluations required')
 rows=[]
 for (seed,arch),g in learning.groupby(['seed','architecture']):
  p=g.pivot(index='update',columns='cue',values='ce')
  for end in ends:
   start=end-10;practiced='A' if (end//10)%2 else 'B';other='B' if practiced=='A' else 'A'
   if start not in p.index or end not in p.index or not np.isfinite(p.loc[[start,end],['A','B']]).all().all():raise ValueError('Missing paired endpoints')
   own=float(p.loc[start,practiced]-p.loc[end,practiced]);cross=float(p.loc[start,other]-p.loc[end,other]);sa,sb,ea,eb=[float(p.loc[k,c]) for k,c in [(start,'A'),(start,'B'),(end,'A'),(end,'B')]]
   during=p.loc[(p.index>=start)&(p.index<=end),['A','B']]
   rows.append(dict(seed=seed,architecture=arch,block=end//10,start=start,end=end,practiced=practiced,other=other,A_start=sa,B_start=sb,A_end=ea,B_end=eb,own_start=float(p.loc[start,practiced]),other_start=float(p.loc[start,other]),own=own,cross=cross,specificity=own-cross,own_floor_limited=bool(p.loc[start,practiced]<FLOOR),both_start_floor=bool(max(sa,sb)<FLOOR),both_end_floor=bool(max(ea,eb)<FLOOR),whole_block_below_marker=bool((during<FLOOR).all().all()),floor_crossing=bool(max(sa,sb)>=FLOOR and max(ea,eb)<FLOOR)))
 return pd.DataFrame(rows)

def interval(frame,metric):
 if sorted(frame.seed.tolist())!=list(range(20)):raise ValueError('20 unique seed rows required, missing values retained')
 x=frame.sort_values('seed')[metric].to_numpy(float);x=x[np.isfinite(x)];n=len(x)
 if n==0:return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n_valid=0,n_planned=20)
 draws=np.random.default_rng(20260925).integers(n,size=(10000,n));lo,hi=np.quantile(x[draws].mean(1),[.025,.975])
 return dict(mean=float(x.mean()),ci_low=float(lo),ci_high=float(hi),n_valid=n,n_planned=20)

def align_diagnostics(b,d):
 if d.duplicated(['seed','architecture','checkpoint','train_context','evaluation_context']).any():raise ValueError('Duplicate diagnostic effect')
 rows=[]
 for r in b.itertuples():
  for alignment,k in [('start',r.start),('end',r.end)]:
   f=d[(d.seed==r.seed)&(d.architecture==r.architecture)&(d.checkpoint==k)&(d.train_context==r.practiced)]
   if f.empty:continue
   if len(f)!=2 or set(f.evaluation_context)!={'A','B'}:raise ValueError('Missing diagnostic direction')
   v=f.set_index('evaluation_context').improvement;own=float(v[r.practiced]);cross=float(v[r.other]);spec=own-cross
   rows.append(dict(seed=r.seed,architecture=r.architecture,block=r.block,start=r.start,end=r.end,practiced=r.practiced,alignment=alignment,checkpoint=k,block_own=r.own,block_cross=r.cross,block_specificity=r.specificity,diagnostic_own=own,diagnostic_cross=cross,diagnostic_specificity=spec,same_strict_sign=bool(r.specificity*spec>0),opposite_strict_sign=bool(r.specificity*spec<0),both_start_floor=r.both_start_floor,own_floor_limited=r.own_floor_limited))
 return pd.DataFrame(rows)

def geometry_join(b,g,d):
 if g.duplicated(['seed','architecture','endpoint']).any():raise ValueError('Duplicate geometry endpoint')
 return g.merge(b,left_on=['seed','architecture','endpoint'],right_on=['seed','architecture','end'],how='left',validate='one_to_one').merge(d.rename(columns={'checkpoint':'endpoint'}),on=['seed','architecture','endpoint'],how='left',validate='one_to_one')

def summarized(frame,by,metrics):
 rows=[]
 for key,g in frame.groupby(by,sort=True):
  key=key if isinstance(key,tuple) else (key,)
  for metric in metrics:rows.append(dict(zip(by,key),metric=metric,**interval(g,metric)))
 return pd.DataFrame(rows)

def source_guard():
 m=json.loads((SOURCE/'manifest.json').read_text())
 for root,items in [(SOURCE,m['outputs']),(ROOT,m['code'])]:
  for name,h in items.items():
   if sha(root/name)!=h:raise ValueError('Sealed predecessor changed: '+name)
 return sha(SOURCE/'manifest.json')

def run():
 if (OUT/'audit/analysis_complete.json').exists():raise FileExistsError('Completed reanalysis cannot be replaced')
 source_manifest=source_guard();names=['learning_trajectories.csv','learning_transfer.csv','representational_geometry.csv','tables/switch_costs.csv','tables/direct_specificity.csv','tables/splitting_classification.csv','audit/config.json']
 inputs={n:sha(SOURCE/n) for n in names};read=lambda n:pd.read_csv(SOURCE/n,float_precision='round_trip')
 l=read(names[0]);d=read(names[1]);g=read(names[2]);sw=read(names[3]);wide=read(names[4]);original=read(names[5]);b=blocks(l)
 assert len(l)==26480 and len(d)==1280 and len(b)==1320 and len(g)==80
 expected={(s,a,e) for s in range(20) for a in ARCHS for e in range(10,331,10)}
 assert set(b[['seed','architecture','end']].itertuples(index=False,name=None))==expected
 # Switch costs must match the identical saved weight state, not different-time/cue branches.
 for r in sw.itertuples():
  x=l[(l.seed==r.seed)&(l.architecture==r.architecture)].pivot(index='update',columns='cue',values='ce')
  raw=x.loc[r.boundary,r.incoming_context]-x.loc[r.boundary,r.outgoing_context];baseline=x.loc[0,r.incoming_context]-x.loc[0,r.outgoing_context]
  np.testing.assert_allclose([r.raw_cost,r.adjusted_cost],[raw,raw-baseline],atol=1e-12,rtol=1e-12)
 # Identity: each practiced block's specificity equals endpoint minus start practiced advantage.
 for r in b.itertuples():
  before=r.other_start-r.own_start;after=(r.B_end-r.A_end) if r.practiced=='A' else (r.A_end-r.B_end)
  np.testing.assert_allclose(r.specificity,after-before,atol=1e-12,rtol=1e-12)
 aligned=align_diagnostics(b,d);geom=geometry_join(b,g,wide)
 save=lambda frame,n:frame.to_csv(OUT/'tables'/n,index=False)
 save(b,'block_learning.csv');save(aligned,'block_diagnostic_alignment.csv');save(geom,'geometry_learning.csv')
 for f,n in [(l,'original_learning_trajectories.csv'),(d,'original_learning_transfer.csv'),(sw,'original_switch_costs.csv'),(wide,'original_direct_specificity.csv'),(original,'original_splitting_classification.csv')]:save(f,n)
 save(summarized(b,['architecture','end'],['own','cross','specificity']),'block_intervals.csv')
 non=b.copy();non.loc[non.own_floor_limited,['own','cross','specificity']]=np.nan
 save(summarized(non,['architecture','end'],['own','cross','specificity']),'nonfloor_block_intervals.csv')
 save(summarized(sw,['architecture','boundary'],['raw_cost','adjusted_cost']),'switch_intervals.csv')
 save(summarized(l,['architecture','update','cue'],['ce']),'loss_intervals.csv')
 save(summarized(d,['architecture','checkpoint','train_context','evaluation_context'],['improvement']),'diagnostic_intervals.csv')
 save(summarized(wide,['architecture','checkpoint'],['D_A','D_B','mean_specificity','mean_own_improvement']),'specificity_intervals.csv')
 floors=b.groupby(['architecture','end'])[['own_floor_limited','both_start_floor','both_end_floor','whole_block_below_marker','floor_crossing']].sum().reset_index();floors['denominator']=20;save(floors,'floor_counts.csv')
 epoch=[]
 for label,start,end in EPOCHS:
  for (seed,arch),f in b[(b.start>=start)&(b.end<=end)].groupby(['seed','architecture']):
   row=dict(seed=seed,architecture=arch,epoch=label,start=start,end=end,n_blocks=len(f))
   for metric in ['own','cross','specificity','own_floor_limited','both_start_floor','whole_block_below_marker']:row[metric]=f[metric].mean()
   row['fraction_positive_specificity']=(f.specificity>0).mean();row['fraction_negative_cross']=(f.cross<0).mean();epoch.append(row)
 ep=pd.DataFrame(epoch);save(ep,'epoch_seed_summary.csv');save(summarized(ep,['architecture','epoch'],['own','cross','specificity','own_floor_limited','both_start_floor','whole_block_below_marker','fraction_positive_specificity','fraction_negative_cross']),'epoch_intervals.csv')
 directional=[]
 for label,start,end in EPOCHS:
  f=b[(b.start>=start)&(b.end<=end)].groupby(['seed','architecture','practiced'])[['own','cross','specificity']].mean().reset_index();f['epoch']=label;directional.append(f)
 direction=pd.concat(directional,ignore_index=True);save(direction,'epoch_direction_seed_summary.csv');save(summarized(direction,['architecture','epoch','practiced'],['own','cross','specificity']),'epoch_direction_intervals.csv')
 differences=[]
 for (seed,arch),f in ep.groupby(['seed','architecture']):
  x=f.set_index('epoch');differences.append(dict(seed=seed,architecture=arch,specificity_late_minus_early=x.loc[EPOCHS[-1][0],'specificity']-x.loc[EPOCHS[0][0],'specificity']))
 diff=pd.DataFrame(differences);save(diff,'paired_development_seed.csv');save(summarized(diff,['architecture'],['specificity_late_minus_early']),'paired_development_intervals.csv')
 agreement=aligned.groupby(['architecture','alignment','checkpoint','practiced']).agg(n=('seed','size'),same_sign=('same_strict_sign','sum'),opposite_sign=('opposite_strict_sign','sum'),own_floor_limited=('own_floor_limited','sum'),block_specificity_mean=('block_specificity','mean'),diagnostic_specificity_mean=('diagnostic_specificity','mean')).reset_index();save(agreement,'diagnostic_agreement.csv')
 save(summarized(geom,['architecture','endpoint'],['hidden_centroid_distance','position_matched_hidden_separation','centered_logit_separation','AB_probability_tv','specificity','mean_specificity']),'geometry_intervals.csv')
 correlations=[]
 for (arch,end),f in geom.groupby(['architecture','endpoint']):
  for x in ['hidden_centroid_distance','centered_logit_separation']:
   for y in ['specificity','mean_specificity']:correlations.append(dict(architecture=arch,endpoint=end,geometry=x,learning=y,pearson=f[x].corr(f[y]),spearman=f[x].rank().corr(f[y].rank()),n=20,interpretation='descriptive_only'))
 save(pd.DataFrame(correlations),'geometry_associations.csv')
 assert source_guard()==source_manifest
 audit=dict(passed=True,source=str(SOURCE.relative_to(ROOT)),source_manifest_sha=source_manifest,inputs=inputs,n_seeds_per_architecture=20,blocks=1320,diagnostic_effect_rows=1280,geometry_rows=80,floor_marker=FLOOR,splitting_criterion_unchanged=True,checkpoint_files_loaded=0,model_evaluations=0,optimizer_updates=0,bootstrap_draws=10000,bootstrap_seed=20260925,epochs=EPOCHS,code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments','tests'] for p in (ROOT/folder/'recurrent_learning_development').glob('*.py')})
 with (OUT/'audit/analysis_complete.json').open('x') as f:json.dump(audit,f,indent=2)
 print(json.dumps({k:v for k,v in audit.items() if k not in ['code','inputs']},indent=2))

if __name__=='__main__':run()
