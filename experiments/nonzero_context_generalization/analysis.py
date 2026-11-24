"""Probe-within-network aggregation followed by paired trained-seed inference."""
from .core import *
from experiments.centered_context_linear.core import interval
from experiments.centered_context_linear.analysis import time_intervals

def validate_vectors(f,last=330,only_updates=None):
 keys=['seed','arm','update','alpha','vector'];updates=list(range(last+1)) if only_updates is None else only_updates;scales=sorted(f.alpha.unique());expected=pd.MultiIndex.from_product([range(20),ARMS,updates,scales,range(20)],names=keys);a=pd.MultiIndex.from_frame(f[keys])
 if a.has_duplicates or len(a)!=len(expected) or set(a)!=set(expected):raise ValueError('Missing/duplicate trained seed,context vector,scale orstate')

def reduce_network(f):
 keys=['seed','arm','update','alpha'];g=f.groupby(keys,sort=True);s=g.agg(n_vectors=('vector','size'),mean_ce=('ce','mean'),median_ce=('ce','median'),mean_accuracy=('accuracy','mean'),mean_margin=('margin','mean'),R=('success','mean'),above_chance_fraction=('above_chance','mean'),delta_zero=('delta_zero','mean'),gap=('gap','mean'),margin_change=('margin_change','mean'),novel_contribution_norm=('novel_contribution_norm','mean'),native_delta_norm=('native_delta_norm','mean'),fixed_competitor_margin=('fixed_competitor_margin','mean'),competitor_switch_penalty=('competitor_switch_penalty','mean'),strong_familiar=('strong_familiar','first')).reset_index()
 if not s.n_vectors.eq(20).all():raise ValueError('Every network must retain exactly20 probes')
 return s

def paired(f,metric):
 keys=['seed','update','alpha'];w=f.pivot(index=keys,columns='arm',values=metric).reset_index();w['normalization']=((w.NI-w.HI)+(w.NC-w.HC))/2;w['centering']=((w.HC-w.HI)+(w.NC-w.NI))/2;w['interaction']=(w.NC-w.NI)-(w.HC-w.HI);w['NC_minus_NI']=w.NC-w.NI;return w

def main():
 if (OUT/'audit/analysis_complete.json').exists():raise RuntimeError('Completed analysis is immutable')
 seed=pd.read_csv(OUT/'tables/seed_trajectories.csv',float_precision='round_trip');assert len(seed)==132400 and not seed.duplicated(['seed','arm','update','alpha']).any();ep=seed[seed['update'].isin(PROBES)].copy();ep.to_csv(OUT/'tables/seed_endpoints.csv',index=False);summary=[];counts=[]
 metrics=['mean_ce','median_ce','mean_accuracy','R','mean_margin','delta_zero','gap','margin_change','novel_contribution_norm']
 for (arm,k,alpha),f in ep.groupby(['arm','update','alpha']):
  for subset in ['all','strong_familiar']:
   x=f if subset=='all' else f[f.strong_familiar];counts.append(dict(arm=arm,update=k,alpha=alpha,subgroup=subset,n=len(x),zero_robustness=int(x.R.eq(0).sum()),all20_success=int(x.R.eq(1).sum()),at_least_one_success=int(x.R.gt(0).sum()),above_chance_network_means=int(x.mean_ce.gt(CHANCE).sum())))
   for metric in metrics:
    mean,lo,hi=interval(x[metric]);summary.append(dict(arm=arm,update=k,alpha=alpha,subgroup=subset,metric=metric,n=len(x),mean=mean,median_across_seeds=x[metric].median(),ci_low=lo,ci_high=hi))
 pd.DataFrame(summary).to_csv(OUT/'tables/endpoint_summary.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'tables/endpoint_counts.csv',index=False)
 primary=seed[seed.alpha==1].copy();primary.to_csv(OUT/'tables/primary_seed_trajectories.csv',index=False);time_intervals(primary,['arm'],'update',['mean_ce','median_ce','mean_accuracy','R','mean_margin','margin_change','novel_contribution_norm','fixed_competitor_margin','competitor_switch_penalty']).to_csv(OUT/'tables/primary_intervals.csv',index=False)
 refs=pd.read_csv(OUT/'tables/reference_evaluations.csv',float_precision='round_trip');random=primary[['seed','arm','update','mean_ce','mean_accuracy']].rename(columns={'mean_ce':'ce','mean_accuracy':'accuracy'});random['cue']='C-random';combined=pd.concat([refs,random],ignore_index=True);combined.to_csv(OUT/'tables/learning_curves.csv',index=False);time_intervals(combined,['arm','cue'],'update',['ce']).to_csv(OUT/'tables/learning_intervals.csv',index=False)
 mechanism=pd.read_csv(OUT/'tables/mechanism_trajectories.csv',float_precision='round_trip');mechanism=mechanism.merge(primary[['seed','arm','update','mean_ce','R','margin_change','novel_contribution_norm']],on=['seed','arm','update'],validate='one_to_one');mechanism.to_csv(OUT/'tables/primary_mechanism.csv',index=False);time_intervals(mechanism,['arm'],'update',['K_raw_norm','K_centered_norm','recurrent_spectral_norm','u_norm','v_norm','baseline_margin','hidden_separation','centered_logit_separation']).to_csv(OUT/'tables/mechanism_intervals.csv',index=False)
 contrasts=[];ci=[];conditional=[]
 for metric in ['mean_ce','median_ce','mean_accuracy','R','delta_zero','gap','margin_change']:
  z=paired(ep,metric);z['metric']=metric;contrasts.append(z)
  for (k,alpha),f in z.groupby(['update','alpha']):
   for effect in ['normalization','centering','interaction','NC_minus_NI']:
    mean,lo,hi=interval(f[effect]);ci.append(dict(update=k,alpha=alpha,metric=metric,effect=effect,n=20,mean=mean,median_across_seeds=f[effect].median(),ci_low=lo,ci_high=hi))
   q=ep[(ep['update']==k)&(ep.alpha==alpha)].pivot(index='seed',columns='arm',values='strong_familiar');ids=q.index[q.NI&q.NC];four=q.index[q.all(1)]
   for subset,ss in [('both_normalized_strong',ids),('all_four_strong',four)]:
    x=f[f.seed.isin(ss)];mean,lo,hi=interval(x.NC_minus_NI);conditional.append(dict(update=k,alpha=alpha,metric=metric,subgroup=subset,n=len(x),mean=mean,ci_low=lo,ci_high=hi))
 pd.concat(contrasts,ignore_index=True).to_csv(OUT/'tables/factorial_seed_contrasts.csv',index=False);pd.DataFrame(ci).to_csv(OUT/'tables/factorial_intervals.csv',index=False);pd.DataFrame(conditional).to_csv(OUT/'tables/conditional_NC_minus_NI.csv',index=False)
 # A distribution of400 probes is never passed to an architecture CI routine.
 dump(OUT/'audit/analysis_complete.json',dict(code=code_hashes(),seed_unit=20,vectors_per_seed=20,bootstrap_draws=10000,bootstrap_seed=20260925,primary_alpha=1,training_updates=0,selection='none; familiar-only reporting strata'))
 append('## Hierarchical analysis complete\nAll20 vectors reduced within each of20 trained seeds before paired contrasts/bootstrap. Alpha1 remains primary;everysecondaryscale retained. All-cohort/qualified denominators,medians,negativeCEchanges andzeroR preserved. Original functional splitting unassessed;source behavioral/geometry evidence retained separately.')
if __name__=='__main__':main()
