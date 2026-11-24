"""Predeclared paired factorial arithmetic and whole-seed inference."""
from .core import *
from experiments.centered_context_linear.analysis import block_table,switch_table,time_intervals,position_geometry
from experiments.centered_context_linear.core import interval

def assert_complete(f,arms=ARMS):
 keys=['seed','arm','update','cue'];expected=pd.MultiIndex.from_product([range(20),arms,range(331),list(CUES)],names=keys);actual=pd.MultiIndex.from_frame(f[keys])
 if actual.has_duplicates or len(actual)!=len(expected) or set(actual)!=set(expected):raise ValueError('Incomplete/duplicate seed-state-cue grid')

def factorial_contrasts(w):
 keys=[x for x in w.columns if x not in ARMS]
 if w.duplicated(keys).any():raise ValueError('Seeds cannot be resampled as independent observations')
 z=w[keys].copy();z['normalization']=((w.NI-w.HI)+(w.NC-w.HC))/2;z['centering']=((w.HC-w.HI)+(w.NC-w.NI))/2;z['interaction']=(w.NC-w.NI)-(w.HC-w.HI);return z

def endpoint_table(f):
 keys=['seed','arm','update'];ce=f.pivot(index=keys,columns='cue',values='ce');ac=f.pivot(index=keys,columns='cue',values='accuracy');q=ce.index.to_frame(index=False);q['familiar_ce']=(ce.A.to_numpy()+ce.B.to_numpy())/2;q['strong_familiar']=((ce[['A','B']]<=.5).all(1)&(ac[['A','B']]>=6/7).all(1)).to_numpy();q['delta_mixed']=(ce['Mixed-both']-ce['Mixed-average']).to_numpy();q['delta_C']=(ce.C-ce['Mixed-both']).to_numpy();out=f.merge(q,on=keys,validate='many_to_one');out['gap']=out.ce-out.familiar_ce;out['success']=(out.ce<=.5)&(out.accuracy>=6/7);out['above_chance']=out.ce>CHANCE;return out

def summarize_endpoints(part):
 base=OUT/part/'tables';f=pd.read_csv(base/'evaluations.csv',float_precision='round_trip');ep=endpoint_table(f[f['update'].isin(PROBES)]);ep.to_csv(base/'endpoints.csv',index=False);summary=[];counts=[]
 for (arm,k,cue),ff in ep.groupby(['arm','update','cue']):
  for subset in ['all','strong_familiar']:
   x=ff if subset=='all' else ff[ff.strong_familiar];counts.append(dict(arm=arm,update=k,cue=cue,subgroup=subset,n=len(x),successes=int(x.success.sum()),above_chance=int(x.above_chance.sum())))
   for metric in ['ce','accuracy','gap','familiar_ce','delta_mixed','delta_C']:
    mean,lo,hi=interval(x[metric]);summary.append(dict(arm=arm,update=k,cue=cue,subgroup=subset,metric=metric,n=len(x),mean=mean,median=x[metric].median(),ci_low=lo,ci_high=hi))
 pd.DataFrame(summary).to_csv(base/'endpoint_summary.csv',index=False);pd.DataFrame(counts).to_csv(base/'endpoint_counts.csv',index=False)
 return ep

def analyze_part_a():
 ep=summarize_endpoints('part_a');f=pd.read_csv(OUT/'part_a/tables/evaluations.csv');assert_complete(f,['Standard','Centered']);time_intervals(f,['arm','cue'],'update',['ce','accuracy']).to_csv(OUT/'part_a/tables/evaluation_intervals.csv',index=False)
 # Per-seed encoding comparisons, not independent cue groups.
 ep[ep.cue=='C'][['seed','arm','update','familiar_ce','strong_familiar','delta_mixed','delta_C']].to_csv(OUT/'part_a/tables/encoding_comparisons.csv',index=False)

def analyze_part_b():
 base=OUT/'part_b/tables';f=pd.read_csv(base/'evaluations.csv',float_precision='round_trip');assert_complete(f);ep=summarize_endpoints('part_b');b=block_table(f);s=switch_table(f);b.to_csv(base/'block_transfer.csv',index=False);s.to_csv(base/'switch_costs.csv',index=False)
 gap=f[f.cue.isin(['A','B'])].pivot(index=['seed','arm','update'],columns='cue',values='ce').reset_index();gap['signed_A_minus_B']=gap.A-gap.B;gap['absolute_gap']=abs(gap.signed_A_minus_B);gap.to_csv(base/'performance_gaps.csv',index=False)
 time_intervals(f,['arm','cue'],'update',['ce','accuracy']).to_csv(base/'evaluation_intervals.csv',index=False)
 for name,frame,groups,time,metrics_ in [('block_intervals',b,['arm','practiced'],'end',['practiced_gain','unpracticed_gain','specificity']),('switch_intervals',s,['arm','direction'],'boundary',['switch_cost','baseline_adjusted_cost'])]:time_intervals(frame,groups,time,metrics_).to_csv(base/(name+'.csv'),index=False)
 geo=pd.read_csv(base/'geometry.csv');time_intervals(geo,['arm'],'update',['hidden_separation','centered_logit_separation','hidden_centroid_distance','g_norm','d_norm','u_norm','v_norm','recurrent_spectral_norm','baseline_margin','actual_avg_margin_change','u_fixed_competitor_margin']).to_csv(base/'geometry_intervals.csv',index=False)
 # All continuous outcomes and success indicators, each contrast computed within seed.
 seed=ep[ep.cue=='C'][['seed','arm','update','familiar_ce','delta_mixed','delta_C','strong_familiar']].copy();seed['strong_familiar']=seed.strong_familiar.astype(float)
 for cue in CUES:
  x=ep[ep.cue==cue].set_index(['seed','arm','update']);i=pd.MultiIndex.from_frame(seed[['seed','arm','update']]);seed[cue+'_ce']=x.ce.reindex(i).to_numpy();seed[cue+'_accuracy']=x.accuracy.reindex(i).to_numpy();seed[cue+'_success']=x.success.reindex(i).to_numpy(float);seed[cue+'_joint_familiar_success']=seed[cue+'_success']*seed.strong_familiar
 seed.to_csv(base/'seed_outcomes.csv',index=False);contrasts=[];intervals=[]
 for metric in seed.columns[3:]:
  wide=seed.pivot(index=['seed','update'],columns='arm',values=metric).reset_index();z=factorial_contrasts(wide);z['metric']=metric;contrasts.append(z)
  for k,ff in z.groupby('update'):
   for effect in ['normalization','centering','interaction']:
    mean,lo,hi=interval(ff[effect]);intervals.append(dict(update=k,metric=metric,effect=effect,n=20,mean=mean,median=ff[effect].median(),ci_low=lo,ci_high=hi))
 pd.concat(contrasts,ignore_index=True).to_csv(base/'factorial_seed_contrasts.csv',index=False);pd.DataFrame(intervals).to_csv(base/'factorial_intervals.csv',index=False)
 # Prespecified descriptive common-qualified comparisons, explicit paired denominators.
 common=[]
 for k,x in seed.groupby('update'):
  w=x.pivot(index='seed',columns='arm',values='strong_familiar');ids=w.index[(w==1).all(1)].tolist()
  for s0 in range(20):common.append(dict(seed=s0,update=k,all_four_strong_familiar=s0 in ids))
 pd.DataFrame(common).to_csv(base/'common_familiar_registry.csv',index=False);conditional=[]
 for pairname,a,bb in [('norm_independent','HI','NI'),('norm_centered','HC','NC'),('center_high','HI','HC'),('center_normalized','NI','NC')]:
  for k,x in seed.groupby('update'):
   q=x.pivot(index='seed',columns='arm',values='strong_familiar');ids=q.index[(q[a]==1)&(q[bb]==1)];common4=q.index[(q==1).all(1)]
   for subset,ss in [('both_strong',ids),('all_four_strong',common4)]:
    for metric in ['familiar_ce','C_ce','Mixed-average_ce','Mixed-both_ce']:
     w=x.pivot(index='seed',columns='arm',values=metric).loc[ss];v=w[bb]-w[a];mean,lo,hi=interval(v);conditional.append(dict(comparison=pairname,update=k,subgroup=subset,n=len(v),metric=metric,mean=mean,ci_low=lo,ci_high=hi))
 pd.DataFrame(conditional).to_csv(base/'conditional_paired.csv',index=False)
 windows=[]
 for key,x in b.groupby(['seed','arm','window','practiced']):windows.append(dict(zip(['seed','arm','window','practiced'],key),practiced_gain=x.practiced_gain.mean(),cross_gain=x.unpracticed_gain.mean(),specificity=x.specificity.mean(),floor_blocks=int(x.floor_at_start.sum()),blocks=len(x),positive_above_floor_blocks=int(x.directional_sign_evidence.sum())))
 windows=pd.DataFrame(windows);windows.to_csv(base/'window_seed_effects.csv',index=False);summary=[]
 for key,x in windows.groupby(['arm','window','practiced']):
  for metric in ['practiced_gain','cross_gain','specificity']:
   mean,lo,hi=interval(x[metric]);summary.append(dict(zip(['arm','window','practiced'],key),metric=metric,n=20,mean=mean,ci_low=lo,ci_high=hi))
 pd.DataFrame(summary).to_csv(base/'window_intervals.csv',index=False)
 dump(OUT/'audit/analysis_complete.json',dict(code=code_hashes(),bootstrap_draws=10000,bootstrap_seed=20260925,unit='paired whole seed',original_splitting_gate='unassessed',floor=.05,selection='none'))
 append('## PartB descriptive analysis complete\nAll-seed mean and median preserved; factorial effects computed withinseed before bootstrap. Familiar-qualified strata and paired denominators explicit. No mediation interpretation, no criterion change. Block/switch effects signed; original splitting gate unassessed.')

if __name__=='__main__':
 import sys
 analyze_part_a() if '--part-a' in sys.argv else analyze_part_b()
