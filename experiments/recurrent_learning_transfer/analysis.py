"""Primary170-first seed inference; functional learning and geometry remain distinct."""
import json
import numpy as np
import pandas as pd
from .design import *
from .measures import *


def run():
 if (OUT/'audit/analysis_complete.json').exists():raise FileExistsError('Analysis already completed')
 learning=pd.read_csv(OUT/'learning_trajectories.csv',float_precision='round_trip');transfer=pd.read_csv(OUT/'learning_transfer.csv',float_precision='round_trip');general=pd.read_csv(OUT/'generalization.csv',float_precision='round_trip');geo=pd.read_csv(OUT/'representational_geometry.csv',float_precision='round_trip')
 wide=[]
 for (seed,a,k),f in transfer.groupby(['seed','architecture','checkpoint']):
  if len(f)!=4 or f.duplicated(['train_context','evaluation_context']).any():raise ValueError('Incomplete/crossed diagnostic cells')
  p=f.set_index(['train_context','evaluation_context']).improvement
  r=dict(seed=seed,architecture=a,checkpoint=k,delta_A_A=p.loc[('A','A')],delta_B_A=p.loc[('A','B')],delta_B_B=p.loc[('B','B')],delta_A_B=p.loc[('B','A')]);r.update(specificity(r));wide.append(r)
 wide=pd.DataFrame(wide);wide.to_csv(OUT/'tables/direct_specificity.csv',index=False);boundary=boundaries(learning);boundary.to_csv(OUT/'tables/switch_costs.csv',index=False);initial=initial_transfer(learning);initial.to_csv(OUT/'tables/initial_transfer.csv',index=False);split=split_summary(learning,wide);split.to_csv(OUT/'tables/splitting_classification.csv',index=False)
 cycles=[]
 for (seed,a),f in boundary.groupby(['seed','architecture']):
  for end in range(20,331,20):
   aa=f[f.boundary==end-10].iloc[0];bb=f[f.boundary==end].iloc[0];raw=(aa.raw_cost+bb.raw_cost)/2;adj=(aa.adjusted_cost+bb.adjusted_cost)/2
   if abs(raw-adj)>1e-12:raise ValueError('Static cue baseline failed to cancel')
   cycles.append(dict(seed=seed,architecture=a,cycle_end=end,A_to_B_boundary=end-10,B_to_A_boundary=end,cycle_cost=adj))
 cycles=pd.DataFrame(cycles);cycles.to_csv(OUT/'tables/balanced_cycle_costs.csv',index=False)
 rows=[];order=[]
 for endpoint in ENDPOINTS:
  for a in ARCHS:
   for seed in range(20):
    f=general[(general.seed==seed)&(general.architecture==a)&(general.endpoint==endpoint)];r=dict(seed=seed,architecture=a,endpoint=endpoint,status=f.status.iloc[0]);p=f.set_index('cue')
    for cue in CUES:r[cue+'_ce']=p.loc[cue,'ce'];r[cue+'_accuracy']=p.loc[cue,'accuracy']
    r['familiar_ce']=(r['A_ce']+r['B_ce'])/2;r['familiar_accuracy']=(r['A_accuracy']+r['B_accuracy'])/2
    for cue in ['Mixed','NoContext','Novel30_A','Novel30_B','Novel60_A','Novel60_B']:r[cue+'_gap']=r[cue+'_ce']-r['familiar_ce']
    for family,variants in [('NovelMean',['Novel30_A','Novel30_B','Novel60_A','Novel60_B']),('Novel30Mean',['Novel30_A','Novel30_B']),('Novel60Mean',['Novel60_A','Novel60_B'])]:
     for metric in ['ce','accuracy']:r[family+'_'+metric]=np.mean([r[c+'_'+metric] for c in variants])
     r[family+'_gap']=r[family+'_ce']-r['familiar_ce']
    r['strong_familiar']=bool(r['A_ce']<=.5 and r['B_ce']<=.5 and r['A_accuracy']>=6/7 and r['B_accuracy']>=6/7);r['perfect_familiar']=bool(r['strong_familiar'] and r['A_accuracy']==1 and r['B_accuracy']==1);r['useful_mixed_prediction']=bool(r['Mixed_ce']<=.5 and r['Mixed_accuracy']>=6/7 and r['Mixed_gap']<=.1);r['learned_skill_generalization']=bool(r['strong_familiar'] and r['useful_mixed_prediction']);r['positive_gap']=r['Mixed_gap']>1e-6;r['material_gap']=r['Mixed_gap']>.1
    sp=split[(split.seed==seed)&(split.architecture==a)&(split.endpoint==endpoint)].iloc[0];r.update({k:sp[k] for k in split.columns if k not in ['seed','architecture','endpoint']});w=wide[(wide.seed==seed)&(wide.architecture==a)&(wide.checkpoint==endpoint)].iloc[0]
    for k in ['delta_A_A','delta_B_A','delta_B_B','delta_A_B']:r[k]=w[k]
    ge=geo[(geo.seed==seed)&(geo.architecture==a)&(geo.endpoint==endpoint)].iloc[0];r.update({k:ge[k] for k in geo if k not in ['seed','architecture','endpoint','status']});r['strong_split']=bool(r['strong_familiar'] and r['functional_split']);r['strong_split_positive_gap']=bool(r['strong_split'] and r['positive_gap']);r['strong_split_material_gap']=bool(r['strong_split'] and r['material_gap']);r['strong_split_useful']=bool(r['strong_split'] and r['useful_mixed_prediction']);r['direct_uninformative']=r['direct_status']=='little_or_nonbeneficial_own_learning';rows.append(r)
  order.append(dict(endpoint=endpoint,role='primary' if endpoint==170 else 'secondary',models_processed=40))
 q=pd.DataFrame(rows);q.to_csv(OUT/'tables/endpoint_summary.csv',index=False);metrics=['A_ce','B_ce','A_accuracy','B_accuracy','familiar_ce','Mixed_ce','Mixed_accuracy','Mixed_gap','NoContext_ce','NoContext_gap','NovelMean_ce','NovelMean_gap','mean_specificity','mean_own_improvement','mean_cross_improvement','mean_cycle_cost','hidden_centroid_distance','position_matched_hidden_separation','centered_logit_separation','AB_probability_tv','functional_split','strong_familiar'];intervals=[];paired=[];paired_ci=[]
 for endpoint in ENDPOINTS:
  f=q[q.endpoint==endpoint];p=f.pivot(index='seed',columns='architecture');common_strong=p['strong_familiar'].eq(True).all(1);common_split=p['strong_split'].eq(True).all(1)
  for a in ARCHS:
   g=f[f.architecture==a].sort_values('seed')
   for qualifier in ['all','strong','strong_split']:
    mask=np.ones(20,bool) if qualifier=='all' else g[qualifier+'_familiar' if qualifier=='strong' else 'strong_split'].eq(True).to_numpy()
    for metric in metrics:
     v=g[['seed',metric]].copy();v.loc[~mask,metric]=np.nan;intervals.append(dict(endpoint=endpoint,architecture=a,qualification=qualifier,metric=metric,**bootstrap(v,metric)))
  for seed in range(20):
   r=dict(seed=seed,endpoint=endpoint,common_strong=bool(common_strong.loc[seed]),common_strong_split=bool(common_split.loc[seed]))
   for metric in metrics:r[metric]=float(p[metric].loc[seed,'RNN-Tanh'])-float(p[metric].loc[seed,'RNN-Linear'])
   paired.append(r)
  pp=pd.DataFrame([r for r in paired if r['endpoint']==endpoint])
  for qualifier in ['all','common_strong','common_strong_split']:
   mask=np.ones(20,bool) if qualifier=='all' else pp[qualifier].eq(True).to_numpy()
   for metric in metrics:
    v=pp[['seed',metric]].copy();v.loc[~mask,metric]=np.nan;paired_ci.append(dict(endpoint=endpoint,qualification=qualifier,metric=metric,**bootstrap(v,metric)))
 pd.DataFrame(intervals).to_csv(OUT/'tables/endpoint_intervals.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'tables/paired_architecture_effects.csv',index=False);pd.DataFrame(paired_ci).to_csv(OUT/'tables/paired_intervals.csv',index=False)
 flags=['strong_familiar','perfect_familiar','direct_evidence','direct_uninformative','boundary_evidence','functional_split','trajectory_specific_learning','strong_split','positive_gap','material_gap','strong_split_positive_gap','strong_split_material_gap','learned_skill_generalization','strong_split_useful'];counts=q.groupby(['endpoint','architecture'])[flags].sum().reset_index();counts['denominator']=20;counts.to_csv(OUT/'tables/counts.csv',index=False)
 diagnostic_intervals=[]
 for (a,k),f in wide.groupby(['architecture','checkpoint']):
  for metric in ['delta_A_A','delta_B_A','delta_B_B','delta_A_B','D_A','D_B','mean_specificity','mean_own_improvement','mean_cross_improvement']:diagnostic_intervals.append(dict(architecture=a,checkpoint=k,metric=metric,**bootstrap(f,metric)))
 pd.DataFrame(diagnostic_intervals).to_csv(OUT/'tables/diagnostic_intervals.csv',index=False)
 initial_ci=[]
 for (a,d),f in initial.groupby(['architecture','direction']):
  for metric in ['trained_improvement','cross_improvement','asymmetry','untrained_context_loss_before_block','untrained_context_loss_after_block']:initial_ci.append(dict(architecture=a,direction=d,metric=metric,**bootstrap(f,metric)))
 pd.DataFrame(initial_ci).to_csv(OUT/'tables/initial_transfer_intervals.csv',index=False)
 correlations=[]
 for (k,a),f in q.groupby(['endpoint','architecture']):
  for metric in ['hidden_centroid_distance','position_matched_hidden_separation','centered_logit_separation','AB_probability_tv']:
   x=f[metric];y=f.mean_specificity;correlations.append(dict(endpoint=k,architecture=a,geometry=metric,pearson=x.corr(y) if x.std()>0 and y.std()>0 else np.nan,spearman=x.rank().corr(y.rank()) if x.std()>0 and y.std()>0 else np.nan,n=20,causal_mediation=False))
 pd.DataFrame(correlations).to_csv(OUT/'tables/geometry_learning_correlations.csv',index=False)
 sensitivity=[]
 for endpoint in ENDPOINTS:
  for a in ARCHS:
   f=q[(q.endpoint==endpoint)&(q.architecture==a)]
   for direct_threshold in [.0001,.001,.01]:
    for cycle_threshold in [.01,.05,.1]:
     count=0
     for r in f.itertuples():
      cy=cycles[(cycles.seed==r.seed)&(cycles.architecture==a)&(cycles.cycle_end<=endpoint)].cycle_cost;boundary=cy.mean()>cycle_threshold and (cy>cycle_threshold).mean()>=.5;direct=r.D_A>0 and r.D_B>0 and r.mean_specificity>direct_threshold and r.mean_own_improvement>1e-4 and min(r.delta_A_A,r.delta_B_B)>-1e-6;count+=int(boundary and direct)
     sensitivity.append(dict(endpoint=endpoint,architecture=a,direct_threshold=direct_threshold,cycle_threshold=cycle_threshold,functional_split_count=count,denominator=20,primary_cell=direct_threshold==.001 and cycle_threshold==.05))
 pd.DataFrame(sensitivity).to_csv(OUT/'tables/splitting_sensitivity.csv',index=False)
 dump(OUT/'audit/analysis_complete.json',dict(code=codes(),analysis_order=order,bootstrap_draws=10000,seed_unit=True,geometry_used_for_splitting=False,thresholds_changed=False));append('## Primary-first analysis complete\nFunctional classifications combine signed current diagnostic effects with baseline-cancelled cycle costs. Primary170 inferred before secondary330; qualified intersections and near-stationary/discordant states remain explicit. All20 seeds per arm retained; no geometry-only splitting, pooled-clone inference or threshold tuning.')
 print('Functional-learning analysis complete; primary170 before secondary330',flush=True)

if __name__=='__main__':run()
