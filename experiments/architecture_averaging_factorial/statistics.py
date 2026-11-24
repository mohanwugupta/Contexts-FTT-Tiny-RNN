"""Whole-trained-seed inference and paired factorial contrasts."""
import numpy as np
import pandas as pd
from .protocol import *


def bootstrap(f,value='effect'):
 if sorted(f.seed.tolist())!=list(range(20)):raise ValueError('20 unique seed rows required, including missing values')
 x=f.sort_values('seed')[value].to_numpy(float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n_valid=0,n_planned=20,status='not_estimable')
 draw=np.random.default_rng(20260925).integers(0,n,(10000,n));lo,hi=np.quantile(x[draw].mean(1),[.025,.975]);status='complete_seed_support' if n==20 else ('conditional_qualified_seeds' if n>1 else 'one_seed_not_population_interval')
 return dict(mean=float(x.mean()),ci_low=float(lo),ci_high=float(hi),n_valid=n,n_planned=20,status=status)

def contrasts(f):
 rows=[]
 for step,g in f.groupby('step'):
  if g.duplicated(['seed','architecture']).any() or len(g)!=80:raise ValueError('Missing/duplicated factorial cells')
  for seed in range(20):
   t=g[g.seed==seed].set_index('architecture')
   if set(t.index)!=set(ARCHS):raise ValueError('Missing architecture pair')
   a,b,c,d=[t.loc[n] for n in ARCHS];rows.append(dict(seed=seed,step=step,recurrence_tanh=d.Mixed_gap-b.Mixed_gap,recurrence_linear=c.Mixed_gap-a.Mixed_gap,nonlinearity_FF=b.Mixed_gap-a.Mixed_gap,nonlinearity_RNN=d.Mixed_gap-c.Mixed_gap,interaction=(d.Mixed_gap-b.Mixed_gap)-(c.Mixed_gap-a.Mixed_gap),recurrence_tanh_familiar=d.familiar_ce-b.familiar_ce,tanh_pair_qualified=bool(b.strong_familiar and d.strong_familiar),all_four_qualified=bool(t.strong_familiar.eq(True).all())))
 return pd.DataFrame(rows)

def run():
 q=pd.read_csv(OUT/'seed_level.csv');c=contrasts(q);c.to_csv(OUT/'tables/paired_contrasts.csv',index=False);summary=[]
 metrics=['A_ce','B_ce','A_accuracy','B_accuracy','familiar_ce','familiar_accuracy','Mixed_ce','Mixed_accuracy','Mixed_gap','NoContext_ce','NoContext_gap','NoContext_accuracy','NovelMean_ce','NovelMean_gap','NovelMean_accuracy','Novel30_A_ce','Novel30_A_gap','Novel30_A_accuracy','Novel30_B_ce','Novel30_B_gap','Novel30_B_accuracy','Novel60_A_ce','Novel60_A_gap','Novel60_A_accuracy','Novel60_B_ce','Novel60_B_gap','Novel60_B_accuracy','hidden_separation','logit_separation','AB_probability_tv','mean_hidden_midpoint_residual','mean_centered_logit_midpoint_residual','Mixed_saturation_fraction']
 for (step,architecture),g in q.groupby(['step','architecture']):
  for qualification in ['all','strong','perfect']:
   mask=np.ones(len(g),bool) if qualification=='all' else g['strong_familiar' if qualification=='strong' else 'perfect_familiar'].eq(True).to_numpy()
   for metric in metrics:
    f=g[['seed',metric]].copy();f.loc[~mask,metric]=np.nan;summary.append(dict(step=step,architecture=architecture,qualification=qualification,metric=metric,**bootstrap(f,metric)))
 pd.DataFrame(summary).to_csv(OUT/'tables/architecture_intervals.csv',index=False)
 paired=[]
 for step,g in c.groupby('step'):
  for qualification in ['all','tanh_pair','all_four']:
   mask=np.ones(len(g),bool) if qualification=='all' else g['tanh_pair_qualified' if qualification=='tanh_pair' else 'all_four_qualified'].eq(True).to_numpy()
   for metric in ['recurrence_tanh','recurrence_linear','nonlinearity_FF','nonlinearity_RNN','interaction','recurrence_tanh_familiar']:
    f=g[['seed',metric]].copy();f.loc[~mask,metric]=np.nan;paired.append(dict(step=step,qualification=qualification,metric=metric,**bootstrap(f,metric)))
 pd.DataFrame(paired).to_csv(OUT/'tables/paired_intervals.csv',index=False)
 flags=['strong_familiar','perfect_familiar','mixed_success','positive_gap','material_gap','substantial_context_dependence','competent_positive_gap','competent_material_gap','competent_context_success']
 counts=q.groupby(['step','architecture'])[flags].sum().reset_index();counts['planned_seeds']=20;counts.to_csv(OUT/'tables/competence_counts.csv',index=False)
 selected=q[(q.step==1000)&q.strong_familiar.eq(True)&(q.material_gap.eq(True)|q.competent_context_success.eq(True))];selected.to_csv(OUT/'tables/qualified_examples_all.csv',index=False)
 onset=[]
 for (step,a),g in q.groupby(['step','architecture']):
  for space in ['hidden','logit']:
   for position in [1,2,3,4,5,6,7,None]:
    n=int(g['first_'+space+'_deviation'].isna().sum()) if position is None else int(g['first_'+space+'_deviation'].eq(position).sum());onset.append(dict(step=step,architecture=a,space=space,first_position=position,count=n,threshold=1e-8))
 pd.DataFrame(onset).to_csv(OUT/'tables/onset_counts.csv',index=False)
 sensitivity=[]
 for a,g in q[q.step==1000].groupby('architecture'):
  for threshold in [.5,1.,2.]:
   for tv in [.05,.1,.2]:
    substantial=(g.hidden_separation>=1)&(g.logit_separation>=threshold)&(g.AB_probability_tv>=tv);sensitivity.append(dict(architecture=a,output_separation_threshold=threshold,probability_tv_threshold=tv,n_strong_context_success=int((g.strong_familiar.eq(True)&substantial&g.mixed_success.eq(True)).sum()),n_strong_context_material_gap=int((g.strong_familiar.eq(True)&substantial&g.material_gap.eq(True)).sum())))
 pd.DataFrame(sensitivity).to_csv(OUT/'tables/context_dependence_sensitivity.csv',index=False)
 dump(OUT/'audit/statistics_complete.json',dict(primary_endpoint=1000,seed_unit=True,bootstrap_draws=10000,all_seeds_primary=True,qualified_results_conditional=True));print('Whole-seed summaries and factorial contrasts complete',flush=True)

if __name__=='__main__':run()
