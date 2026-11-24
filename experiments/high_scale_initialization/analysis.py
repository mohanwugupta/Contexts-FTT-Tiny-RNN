"""Allseed primary/secondary summaries, signed pooling and separated source cohorts."""
import pandas as pd
from .core import *

def bootstrap(f,metric):
 if sorted(f.seed.tolist())!=list(range(20)):raise ValueError('20 distinct seed slots required')
 x=f.sort_values('seed')[metric].to_numpy(float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n_valid=0,n_planned=20)
 ix=np.random.default_rng(20260925).integers(0,n,(10000,n));lo,hi=np.quantile(x[ix].mean(1),[.025,.975]);return dict(mean=float(x.mean()),ci_low=float(lo),ci_high=float(hi),n_valid=n,n_planned=20)

def quality(a,b,aa,ba,m,ma):
 if not np.isfinite([a,b,aa,ba,m,ma]).all():return dict(strong_familiar=None,poor_Mixed=None,good_familiar_poor_Mixed=None,useful_Mixed=None)
 strong=bool(a<=.5 and b<=.5 and aa>=6/7 and ba>=6/7);poor=bool(m>CHANCE);gap=m-(a+b)/2
 return dict(strong_familiar=strong,poor_Mixed=poor,good_familiar_poor_Mixed=strong and poor,useful_Mixed=bool(strong and m<=.5 and ma>=6/7 and gap<=.1))

def blocks(l,ends=range(10,331,10)):
 rows=[]
 if l.duplicated(['seed','architecture','update','cue']).any():raise ValueError('Duplicate frozen states')
 for (seed,a),g in l.groupby(['seed','architecture']):
  p=g.pivot(index='update',columns='cue',values='ce')
  for end in ends:
   start=end-10;owncue='A' if end%20==10 else 'B';crosscue='B' if owncue=='A' else 'A';sa=float(p.loc[start,owncue]);sb=float(p.loc[start,crosscue]);ea=float(p.loc[end,owncue]);eb=float(p.loc[end,crosscue]);valid=bool(np.isfinite([sa,sb,ea,eb]).all());own=sa-ea;cross=sb-eb
   rows.append(dict(seed=seed,architecture=a,start=start,end=end,practiced=owncue,own_start_ce=sa,other_start_ce=sb,own_end_ce=ea,other_end_ce=eb,own=own,cross=cross,specificity=own-cross,valid=valid,both_start_floor=bool(sa<.05 and sb<.05) if valid else None,raw_incoming_minus_outgoing=eb-ea,baseline_adjusted_switch_cost=(eb-ea)-(float(p.loc[0,crosscue])-float(p.loc[0,owncue])),actual_next_block_executed=end<330,original_joint_gate='not_assessed_no_one_update_diagnostics'))
 return pd.DataFrame(rows)

def summarize(frame,keys,metrics):
 rows=[]
 for key,g in frame.groupby(keys):
  key=key if isinstance(key,tuple) else (key,)
  for metric in metrics:rows.append(dict(zip(keys,key),metric=metric,**bootstrap(g,metric)))
 return pd.DataFrame(rows)

def run():
 if (OUT/'audit/analysis_complete.json').exists():raise FileExistsError('Already analyzed')
 assert (OUT/'audit/probes_complete.json').exists();read=lambda n:pd.read_csv(OUT/n,float_precision='round_trip');v=read('generalization.csv');l=read('learning_trajectories.csv');t=read('training_measurements.csv');g=read('tables/geometry.csv');b=blocks(l);b.to_csv(OUT/'splitting.csv',index=False)
 gaps=l.pivot(index=['seed','architecture','update'],columns='cue',values='ce').reset_index().rename_axis(columns=None);gaps['signed_B_minus_A']=gaps.B-gaps.A;gaps['absolute_gap']=gaps.signed_B_minus_A.abs();gaps['A_to_B_boundary']=(gaps['update']>=10)&(gaps['update']%20==10);gaps.to_csv(OUT/'tables/AB_gap.csv',index=False)
 rows=[]
 # Primary170 first,330 second; earlier checkpoints are fixed descriptive states.
 for k in [170,330,80,100,120]:
  for a in ARCHS:
   for seed in range(20):
    f=v[(v.checkpoint==k)&(v.architecture==a)&(v.seed==seed)].set_index('cue');vals={c:float(f.loc[c,'ce']) for c in ['A','B','Mixed']};acc={c:float(f.loc[c,'accuracy']) for c in vals};lf=(vals['A']+vals['B'])/2
    row=dict(seed=seed,architecture=a,checkpoint=k,A_ce=vals['A'],B_ce=vals['B'],Mixed_ce=vals['Mixed'],A_accuracy=acc['A'],B_accuracy=acc['B'],Mixed_accuracy=acc['Mixed'],familiar_ce=lf,Mixed_gap=vals['Mixed']-lf,**quality(vals['A'],vals['B'],acc['A'],acc['B'],vals['Mixed'],acc['Mixed']));rows.append(row)
 q=pd.DataFrame(rows).merge(g.drop(columns=['checkpoint_sha','status']),on=['seed','architecture','checkpoint'],how='left',validate='one_to_one');q.to_csv(OUT/'tables/endpoint_seed_summary.csv',index=False)
 metrics=['A_ce','B_ce','familiar_ce','Mixed_ce','Mixed_accuracy','Mixed_gap','hidden_centroid_distance','centered_logit_separation','saturation_95','saturation_99','recurrent_spectral_norm','hidden_absmax'];summarize(q,['architecture','checkpoint'],metrics).to_csv(OUT/'tables/endpoint_intervals.csv',index=False)
 counts=[]
 for (a,k),f in q.groupby(['architecture','checkpoint']):counts.append(dict(architecture=a,checkpoint=k,n_planned=20,n_valid=int(np.isfinite(f.Mixed_ce).sum()),strong_familiar=int(f.strong_familiar.eq(True).sum()),Mixed_above_chance=int(f.poor_Mixed.eq(True).sum()),good_familiar_poor_Mixed=int(f.good_familiar_poor_Mixed.eq(True).sum()),useful_Mixed=int(f.useful_Mixed.eq(True).sum())))
 pd.DataFrame(counts).to_csv(OUT/'tables/counts.csv',index=False)
 conditional=q.copy();conditional.loc[~conditional.strong_familiar.eq(True),['Mixed_ce','Mixed_gap','Mixed_accuracy']]=np.nan;summarize(conditional,['architecture','checkpoint'],['Mixed_ce','Mixed_gap','Mixed_accuracy']).to_csv(OUT/'tables/competence_qualified_intervals.csv',index=False)
 summarize(t,['architecture','update'],['ce_before']).to_csv(OUT/'tables/training_intervals.csv',index=False);summarize(gaps,['architecture','update'],['absolute_gap','signed_B_minus_A']).to_csv(OUT/'tables/gap_intervals.csv',index=False);summarize(b,['architecture','end'],['own','cross','specificity','raw_incoming_minus_outgoing','baseline_adjusted_switch_cost']).to_csv(OUT/'tables/block_intervals.csv',index=False)
 paired=[]
 for k in PROBES:
  f=q[q.checkpoint==k].set_index(['seed','architecture'])
  for seed in range(20):paired.append(dict(seed=seed,checkpoint=k,**{m:float(f.loc[(seed,'RNN-Tanh'),m]-f.loc[(seed,'RNN-Linear'),m]) for m in ['familiar_ce','Mixed_ce','Mixed_gap']}))
 paired=pd.DataFrame(paired);paired.to_csv(OUT/'tables/paired_new_seed_effects.csv',index=False);summarize(paired,['checkpoint'],['familiar_ce','Mixed_ce','Mixed_gap']).to_csv(OUT/'tables/paired_new_intervals.csv',index=False)
 comparisons=[]
 norm=pd.read_csv(ROOT/'results/recurrent_learning_transfer/v1/tables/endpoint_summary.csv',float_precision='round_trip')
 for r in norm[norm.endpoint.isin([170,330])].itertuples():comparisons.append(dict(seed=r.seed,checkpoint=r.endpoint,group='Normalized '+r.architecture,familiar_ce=r.familiar_ce,Mixed_ce=r.Mixed_ce,reference=False,paired_to_new=False))
 for r in q[q.checkpoint.isin([170,330])].itertuples():comparisons.append(dict(seed=r.seed,checkpoint=r.checkpoint,group='Unnormalized '+r.architecture,familiar_ce=r.familiar_ce,Mixed_ce=r.Mixed_ce,reference=False,paired_to_new=True))
 old=pd.read_csv(ROOT/'results/recurrent_learning_transfer/v1/tables/historical_original_trace.csv',float_precision='round_trip')
 for seed,f in old.groupby('seed'):
  losses=f.groupby('cue').raw_ce.mean();comparisons.append(dict(seed=seed,checkpoint=170,group='Canonical Tanh reference',familiar_ce=float((losses.A+losses.B)/2),Mixed_ce=float(losses.Mixed),reference=True,paired_to_new=False))
 comp=pd.DataFrame(comparisons);comp.to_csv(OUT/'tables/initialization_comparison.csv',index=False);summarize(comp,['group','checkpoint'],['familiar_ce','Mixed_ce']).to_csv(OUT/'tables/comparison_intervals.csv',index=False)
 source=old[old.cue.isin(['A','B','Mixed'])].groupby(['seed','cue']).raw_ce.mean();replay=[]
 for r in v[(v.architecture=='RNN-Tanh')&(v.checkpoint==170)].itertuples():replay.append(dict(seed=r.seed,cue=r.cue,new_torch_ce=r.ce,archived_numpy_ce=float(source.loc[(r.seed,r.cue)]),ce_difference=float(r.ce-source.loc[(r.seed,r.cue)]),same_checkpoint_parameters=True))
 pd.DataFrame(replay).to_csv(OUT/'tables/canonical_frozen_eval_linkage.csv',index=False)
 # Fixed early/middle/late block summaries, allseeds and both practiced directions retained.
 epochs=[]
 for name,start,end in [('early',0,40),('middle',40,170),('late',170,330)]:
  f=b[(b.start>=start)&(b.end<=end)].groupby(['seed','architecture','practiced'])[['own','cross','specificity']].mean().reset_index();f['epoch']=name;epochs.append(f)
 ep=pd.concat(epochs);ep.to_csv(OUT/'tables/block_epoch_seed_summary.csv',index=False);summarize(ep,['architecture','practiced','epoch'],['own','cross','specificity']).to_csv(OUT/'tables/block_epoch_intervals.csv',index=False)
 dump(OUT/'audit/analysis_complete.json',dict(primary_first=True,all_seed_slots_retained=True,bootstrap_draws=10000,bootstrap_seed=20260925,normalization_comparison_paired=False,canonical_reference_independent=False,new_joint_splitting_gate_assessed=False));append('## Allseed inference complete\nPrimary170 precedes330; failures/missing notzero. Original strong-familiar andpoorMixed standards unchanged. Block effects signed/symmetric,sourcegeometry independent of functional gate. Normalizedcomparison unpaired descriptive; canonicalreference notindependent frommatchedreproducednewTanh.');print('Analysis complete; allseed and qualified results retained')

if __name__=='__main__':run()
