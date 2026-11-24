"""Saved training measurements and same-weight A/B evaluations; no models imported."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'results/recurrent_learning_transfer/v1';OUT=ROOT/'results/original_learning_curves/v1';ARCHS=['RNN-Linear','RNN-Tanh']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def paired(l):
 if set(l.cue)!={'A','B'} or l.duplicated(['seed','architecture','update','cue']).any():raise ValueError('Unique A/B evaluations required')
 p=l.pivot(index=['seed','architecture','update'],columns='cue',values='ce')
 if not np.isfinite(p[['A','B']]).all().all():raise ValueError('Missing paired evaluation')
 return p

def actual_training(h,l):
 p=paired(l);rows=[]
 for r in h.itertuples():
  if not 1<=r.update<=330:raise ValueError('Outside saved run')
  cue='A' if ((r.update-1)//10)%2==0 else 'B'
  if r.training_cue!=cue:raise ValueError('Wrong A10/B10 cue schedule')
  key=(r.seed,r.architecture,r.update-1)
  if key not in p.index:raise ValueError('Missing pre-update frozen state')
  reference=float(p.loc[key,cue]);error=abs(r.ce_before-reference)
  if not np.isfinite(r.ce_before) or error>1e-5+2e-6*abs(reference):raise ValueError('Actual predictive loss does not match practiced pre-update state')
  rows.append(dict(seed=r.seed,architecture=r.architecture,repetition=r.update,context=cue,ce_before=r.ce_before,frozen_state=r.update-1,frozen_practiced_ce=reference,source_precision_difference=r.ce_before-reference,block=(r.update-1)//10+1,is_first_after_switch=r.update>1 and (r.update-1)%10==0))
 return pd.DataFrame(rows)

def gaps(l):
 p=paired(l).reset_index().rename(columns={'update':'state','A':'CE_A','B':'CE_B'});p['signed_B_minus_A']=p.CE_B-p.CE_A;p['gap_AB']=p.signed_B_minus_A.abs();p['A_to_B_boundary']=(p.state>=10)&(p.state%20==10);p['executed_A_to_B_switch']=p.A_to_B_boundary&(p.state<330);p['end_censored']=p.state.eq(330);return p

def pooling(l,ends=range(10,331,20)):
 p=paired(l);rows=[]
 for (seed,a),g in p.groupby(level=[0,1]):
  for end in ends:
   if end%20!=10:raise ValueError('A-ending blocks only')
   try:start=g.loc[(seed,a,end-10)];finish=g.loc[(seed,a,end)]
   except KeyError as e:raise ValueError('Missing A-block endpoints') from e
   ga=float(start.A-finish.A);gb=float(start.B-finish.B);cost=float(finish.B-finish.A)
   rows.append(dict(seed=seed,architecture=a,start=end-10,end=end,CE_A_start=float(start.A),CE_B_start=float(start.B),CE_A_end=float(finish.A),CE_B_end=float(finish.B),gain_A=ga,gain_B=gb,specificity_A=ga-gb,same_weight_A_to_B_cost=cost,executed_next_B=end<330,end_censored=end==330,positive_A_gain_and_B_transfer_and_switch_cost=bool(ga>0 and gb>0 and cost>0)))
 return pd.DataFrame(rows)

def summaries(frame,time,metrics):
 rows=[]
 for a,f in frame.groupby('architecture'):
  times=sorted(f[time].unique());seed_order=list(range(20));draws=np.random.default_rng(20260925).integers(0,20,(10000,20));weights=np.stack([np.bincount(x,minlength=20) for x in draws])/20
  for metric in metrics:
   if f.duplicated(['seed',time]).any():raise ValueError('Repeated measurements cannot be bootstrap networks')
   p=f.pivot(index='seed',columns=time,values=metric).reindex(index=seed_order,columns=times)
   if not np.isfinite(p).all().all():raise ValueError('All20 seeds must be retained')
   vals=p.to_numpy();means=vals.mean(0);lo,hi=np.quantile(weights@vals,[.025,.975],axis=0)
   for k,mean,low,high in zip(times,means,lo,hi):rows.append(dict(architecture=a,**{time:k},metric=metric,mean=float(mean),ci_low=float(low),ci_high=float(high),n_seeds=20))
 return pd.DataFrame(rows)

def source_guard():
 manifest=json.loads((SOURCE/'manifest.json').read_text())
 for root,items in [(SOURCE,manifest['outputs']),(ROOT,manifest['code'])]:
  for n,h in items.items():
   if sha(root/n)!=h:raise ValueError('Changed sealed source: '+n)
 return sha(SOURCE/'manifest.json')

def run():
 if (OUT/'audit/analysis_complete.json').exists():raise FileExistsError('Already completed')
 initial=source_guard();paths=['tables/training_history.csv','learning_trajectories.csv','tables/splitting_classification.csv'];inputs={n:sha(SOURCE/n) for n in paths};h=pd.read_csv(SOURCE/paths[0],float_precision='round_trip');l=pd.read_csv(SOURCE/paths[1],float_precision='round_trip')
 assert len(h)==13200 and h.status.eq('completed').all() and len(l)==26480
 train=actual_training(h,l);gap=gaps(l);pool=pooling(l)
 assert len(train)==13200 and len(gap)==13240 and len(pool)==680
 expected={(s,a,r) for s in range(20) for a in ARCHS for r in range(1,331)};assert set(train[['seed','architecture','repetition']].itertuples(index=False,name=None))==expected
 # Actual switch jump compares observations at two successive model states; retain separately.
 tr=train.set_index(['seed','architecture','repetition']);switch=[]
 for r in pool.itertuples():
  before=float(tr.loc[(r.seed,r.architecture,r.end),'ce_before']);after=float(tr.loc[(r.seed,r.architecture,r.end+1),'ce_before']) if r.executed_next_B else np.nan
  switch.append(dict(seed=r.seed,architecture=r.architecture,boundary=r.end,next_B_repetition=r.end+1 if r.executed_next_B else np.nan,last_A_training_ce=before,first_B_training_ce=after,actual_practiced_loss_jump=after-before if r.executed_next_B else np.nan,same_weight_A_to_B_cost=r.same_weight_A_to_B_cost,end_censored=r.end_censored))
 sw=pd.DataFrame(switch);pool=pool.merge(sw[['seed','architecture','boundary','actual_practiced_loss_jump']],left_on=['seed','architecture','end'],right_on=['seed','architecture','boundary'],validate='one_to_one').drop(columns='boundary')
 save=lambda df,n:df.to_csv(OUT/'tables'/n,index=False)
 save(train,'figure_1_training_values.csv');s=summaries(train,'repetition',['ce_before']);s['context']=np.where(((s.repetition-1)//10)%2==0,'A','B');save(s,'figure_1_training_summary.csv')
 save(gap,'figure_2_gap_values.csv');s=summaries(gap,'state',['gap_AB','signed_B_minus_A']);s['A_to_B_boundary']=(s.state>=10)&(s.state%20==10);s['executed_A_to_B_switch']=s.A_to_B_boundary&(s.state<330);s['end_censored']=s.state.eq(330);save(s,'figure_2_gap_summary.csv')
 save(pool,'figure_3_pooling_values.csv');save(summaries(pool,'end',['gain_A','gain_B']),'figure_3_pooling_summary.csv');save(sw,'A_to_B_switch_values.csv');save(summaries(sw[~sw.end_censored],'boundary',['actual_practiced_loss_jump','same_weight_A_to_B_cost']),'A_to_B_switch_summary.csv')
 counts=pool.groupby(['architecture','end']).agg(n_seeds=('seed','size'),positive_A_gain=('gain_A',lambda x:int((x>0).sum())),positive_B_transfer=('gain_B',lambda x:int((x>0).sum())),positive_same_weight_cost=('same_weight_A_to_B_cost',lambda x:int((x>0).sum())),coexistence=('positive_A_gain_and_B_transfer_and_switch_cost','sum'),end_censored=('end_censored','first')).reset_index();save(counts,'A_to_B_coexistence_counts.csv')
 assert source_guard()==initial
 audit=dict(passed=True,source_manifest_sha=initial,inputs=inputs,training_rows=13200,frozen_gap_rows=13240,A_block_rows=680,seeds_per_architecture=20,source_precision_max_abs_difference=float(train.source_precision_difference.abs().max()),bootstrap_draws=10000,bootstrap_seed=20260925,criterion_changed=False,network_evaluations=0,training_updates=0,code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments','tests'] for p in (ROOT/folder/'original_learning_curves').glob('*.py')})
 with (OUT/'audit/analysis_complete.json').open('x') as f:json.dump(audit,f,indent=2)
 print(json.dumps({k:v for k,v in audit.items() if k not in ['inputs','code']},indent=2))

if __name__=='__main__':run()
