"""Allseed/qualified inference, paired common330 and within-seed novelty contrasts."""
from .core import *

def aggregate_within_seed(v):
 rows=[]
 for (seed,a,k),f in v.groupby(['seed','architecture','checkpoint']):
  p=f.set_index('cue')
  for name,cues in REGISTRY['aggregates'].items():
   x=p.loc[cues];rows.append(dict(seed=seed,architecture=a,checkpoint=k,cue=name,ce=float(x.ce.mean()),accuracy=float(x.accuracy.mean()),gap=float(x.gap.mean()),strong_familiar=bool(x.strong_familiar.iloc[0])))
 return pd.DataFrame(rows)

def paired330(v):
 f=v[v.checkpoint==330];flags=f.groupby(['seed','architecture']).strong_familiar.first().unstack();common=sorted(flags.index[flags.get('RNN-Linear',False)&flags.get('RNN-Tanh',False)].tolist());rows=[]
 for seed in common:
  for c,g in f[f.seed==seed].groupby('cue'):
   p=g.set_index('architecture')
   if set(p.index)!=set(ARCHS):raise ValueError('Missing paired architecture')
   rows.append(dict(seed=seed,cue=c,**{m+'_Tanh_minus_Linear':float(p.loc['RNN-Tanh',m]-p.loc['RNN-Linear',m]) for m in ['ce','gap','accuracy']}))
 return pd.DataFrame(rows,columns=['seed','cue','ce_Tanh_minus_Linear','gap_Tanh_minus_Linear','accuracy_Tanh_minus_Linear']),common

def run():
 if (OUT/'audit/analysis_complete.json').exists():raise FileExistsError('Already analyzed')
 assert (OUT/'audit/evaluation_complete.json').exists();v=pd.read_csv(OUT/'tables/seed_values.csv',float_precision='round_trip');agg=aggregate_within_seed(v);agg['familiar_ce']=agg.ce-agg.gap;agg['above_chance']=agg.ce>CHANCE;combined=pd.concat([v,agg],ignore_index=True);paired,common=paired330(combined);config=json.loads((OUT/'audit/config.json').read_text());assert common==config['common_strong330_seeds'];combined['common_strong330']=(combined.checkpoint==330)&combined.seed.isin(common);agg.to_csv(OUT/'tables/registered_aggregate_values.csv',index=False);combined.to_csv(OUT/'tables/all_cue_and_aggregate_values.csv',index=False);paired.to_csv(OUT/'tables/common_strong330_pairs.csv',index=False)
 intervals=[];counts=[]
 for (a,k,c),f in combined.groupby(['architecture','checkpoint','cue']):
  for qual in ['all','strong_familiar']+(['common_strong330'] if k==330 else []):
   mask=np.ones(20,bool) if qual=='all' else f.strong_familiar.to_numpy(bool) if qual=='strong_familiar' else f.seed.isin(common).to_numpy();ff=f.copy();n=int(mask.sum())
   for metric in ['ce','accuracy','gap']:
    ff[metric]=ff[metric].where(mask,np.nan);intervals.append(dict(architecture=a,checkpoint=k,cue=c,qualification=qual,metric=metric,**interval(ff,metric)))
   selected=f.loc[mask];counts.append(dict(architecture=a,checkpoint=k,cue=c,qualification=qual,n_valid=n,n_planned=20,above_chance=int((selected.ce>CHANCE).sum()),good_cue_prediction=int(((selected.ce<=.5)&(selected.accuracy>=6/7)).sum())))
 pd.DataFrame(intervals).to_csv(OUT/'tables/intervals.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'tables/counts.csv',index=False)
 pair_ci=[]
 for c,f in paired.groupby('cue'):
  for metric in ['ce_Tanh_minus_Linear','gap_Tanh_minus_Linear','accuracy_Tanh_minus_Linear']:pair_ci.append(dict(checkpoint=330,cue=c,metric=metric,**interval(f,metric,common)))
 pd.DataFrame(pair_ci).to_csv(OUT/'tables/common_strong330_paired_intervals.csv',index=False)
 contrasts=[]
 definitions=[('60_minus_30_A','Novel60_A','Novel30_A'),('60_minus_30_B','Novel60_B','Novel30_B'),('B_minus_A_30','Novel30_B','Novel30_A'),('B_minus_A_60','Novel60_B','Novel60_A')]
 for (seed,a,k),f in v.groupby(['seed','architecture','checkpoint']):
  p=f.set_index('cue')
  for label,plus,minus in definitions:contrasts.append(dict(seed=seed,architecture=a,checkpoint=k,contrast=label,strong_familiar=bool(f.strong_familiar.iloc[0]),**{'delta_'+m:float(p.loc[plus,m]-p.loc[minus,m]) for m in ['ce','gap','accuracy']}))
 contrasts=pd.DataFrame(contrasts);contrasts.to_csv(OUT/'tables/novelty_side_seed_contrasts.csv',index=False);contrast_ci=[]
 for (a,k,c),f in contrasts.groupby(['architecture','checkpoint','contrast']):
  for qual in ['all','strong_familiar']+(['common_strong330'] if k==330 else []):
   ff=f.copy();mask=np.ones(20,bool) if qual=='all' else f.strong_familiar.to_numpy(bool) if qual=='strong_familiar' else f.seed.isin(common).to_numpy()
   for metric in ['delta_ce','delta_gap','delta_accuracy']:
    ff[metric]=f[metric].where(mask,np.nan);contrast_ci.append(dict(architecture=a,checkpoint=k,contrast=c,qualification=qual,metric=metric,**interval(ff,metric)))
 pd.DataFrame(contrast_ci).to_csv(OUT/'tables/novelty_side_intervals.csv',index=False)
 joint=[]
 geometry=pd.read_csv(SOURCE/'tables/geometry.csv').set_index(['seed','architecture','checkpoint'])
 for (seed,a,k),f in v.groupby(['seed','architecture','checkpoint']):
  p=f.set_index('cue');nov=p.loc[NOVEL];good=bool(((nov.ce<=.5)&(nov.accuracy>=6/7)).all());familiar=bool(f.strong_familiar.iloc[0]);geo=geometry.loc[(seed,a,k)];joint.append(dict(seed=seed,architecture=a,checkpoint=k,strong_familiar=familiar,all_four_novel_good=good,strong_familiar_and_all_four_novel_good=familiar and good,any_novel_above_chance=bool((nov.ce>CHANCE).any()),all_four_novel_perfect_accuracy=bool((nov.accuracy==1).all()),hidden_centroid_distance=geo.hidden_centroid_distance,centered_logit_separation=geo.centered_logit_separation))
 joint=pd.DataFrame(joint);joint.to_csv(OUT/'tables/joint_familiar_novel.csv',index=False);joint.groupby(['architecture','checkpoint'])[['strong_familiar','all_four_novel_good','strong_familiar_and_all_four_novel_good','any_novel_above_chance','all_four_novel_perfect_accuracy']].sum().reset_index().to_csv(OUT/'tables/joint_counts.csv',index=False)
 dump(OUT/'audit/analysis_complete.json',dict(passed=True,common_strong330_n=len(common),common_strong330_seeds=common,all_seed_denominator=20,bootstrap_draws=10000,bootstrap_seed=20260925,probe_selection=False,changed_thresholds=False));append('## Fixed allseed and qualified analysis\nAll20 remainprimary. Ownstrong subsets andcommonstrong330 intersection reported withactualn; paired contrastsuseTanhminusLinear andsharedseedresampling. Angles/sides contrastedwithinseed beforeinference,aggregatecue means computedwithinseed. Jointfamiliar+all4novel standardisdescriptive,not a newskillgate or humanRTreplication.');print('Analysis complete; pairedcommonstrong330 n='+str(len(common)))

if __name__=='__main__':run()
