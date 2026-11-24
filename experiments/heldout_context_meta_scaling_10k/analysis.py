"""Continuous C gap and unchanged functional criteria, paired seed inference."""
from .common import *
from .data import collect
from experiments.meta_learning_scaling.analysis import interval,summarize
from experiments.meta_learning_scaling.analysis import matched_acquisition as source_matching
from experiments.centered_context_linear.analysis import block_table,time_intervals
from experiments.recurrent_learning_transfer.measures import specificity,boundaries,split_summary

KEY=['seed','condition','arm','dose','update']
def gap_records(f):
 p=f.pivot(index=KEY,columns='cue',values='ce');a=f.pivot(index=KEY,columns='cue',values='accuracy');q=p[['A','B']].copy();q['familiar_ce']=(q.A+q.B)/2;q['strong_familiar']=((q.A<=.5)&(q.B<=.5)&(a.A>=6/7)&(a.B>=6/7)).astype(float);q.loc[q.A.isna()|q.B.isna(),'strong_familiar']=np.nan;q=q[['familiar_ce','strong_familiar']].reset_index();f=f.merge(q,on=KEY,validate='many_to_one');f['gap']=f.ce-f.familiar_ce;f['success']=((f.ce<=.5)&(f.accuracy>=6/7)).astype(float);f['strict_ce']=(f.ce<=.1).astype(float);f['strict_accuracy']=(f.accuracy==1).astype(float);f['strict_gap']=(f.gap<=.1).astype(float);f['strict_all']=(f.strict_ce.eq(1)&f.strict_accuracy.eq(1)&f.strict_gap.eq(1)).astype(float)
 for n in ['success','strict_ce','strict_accuracy','strict_gap','strict_all']:f.loc[f.ce.isna()|f.accuracy.isna(),n]=np.nan
 rows=[]
 for keys,g in f[f.cue=='C'].groupby(['seed','condition','arm','dose']):
  g=g.sort_values('update');h=g[g.strong_familiar==1]['update'];first=h.min() if len(h) else np.nan;after=g[g['update']>=first] if len(h) else g.iloc[:0];rows.append(dict(zip(['seed','condition','arm','dose'],keys),first_familiar=first,post_acquisition_gap=after.gap.mean(),post_acquisition_C_CE=after.ce.mean(),post_acquisition_familiar_CE=after.familiar_ce.mean(),post_acquisition_states=len(after),status='acquired' if len(h) else 'never_acquired'))
 return f,pd.DataFrame(rows)

def paired_effects(f,groups,metrics):
 rows=[]
 for key,g in f.groupby(groups):
  key=key if isinstance(key,tuple) else (key,)
  for metric in metrics:
   p=g.pivot(index='seed',columns=['condition','arm','dose'],values=metric).reindex(range(20));doses=sorted(set(g.dose)-{0})
   for c in CONDS:
    for dose in doses:
     for aa,bb,name in [((c,'Transfer',dose),(c,'Seen',dose),'Transfer-Seen'),((c,'Transfer',dose),(c,'Historical',0),'Transfer-Historical'),((c,'Seen',dose),(c,'Historical',0),'Seen-Historical')]:
      if aa not in p or bb not in p:continue
      for seed,v in (p[aa]-p[bb]).items():rows.append(dict(zip(groups,key),seed=seed,condition=c,dose=dose,contrast=name,metric=metric,effect=v))
   for org in ['I','C']:
    for dose in doses:
     tags=[(c+'-'+org,a,dose) for c in ['P1','P2'] for a in ['Transfer','Seen']]
     if not all(x in p for x in tags):continue
     e1=p[('P1-'+org,'Transfer',dose)]-p[('P1-'+org,'Seen',dose)];e2=p[('P2-'+org,'Transfer',dose)]-p[('P2-'+org,'Seen',dose)]
     for arm_name in ['Transfer','Seen']:
      for seed,v in (p[('P2-'+org,arm_name,dose)]-p[('P1-'+org,arm_name,dose)]).items():rows.append(dict(zip(groups,key),seed=seed,condition='PriorDiff-'+org,dose=dose,contrast=arm_name+':P2-P1',metric=metric,effect=v))
     ha=('P1-'+org,'Historical',0);hb=('P2-'+org,'Historical',0)
     if ha in p and hb in p:
      for seed,v in (p[('P2-'+org,'Transfer',dose)]-p[hb]-p[('P1-'+org,'Transfer',dose)]+p[ha]).items():rows.append(dict(zip(groups,key),seed=seed,condition='PriorDiff-'+org,dose=dose,contrast='Transfer_improvement:P2-P1',metric=metric,effect=v))
     for condition,ef,name in [('PriorDiff-'+org,e2-e1,'P2_response-P1_response'),('PriorAverage-'+org,(e1+e2)/2,'Transfer-Seen_withinseed_prior_mean')]:
      for seed,v in ef.items():rows.append(dict(zip(groups,key),seed=seed,condition=condition,dose=dose,contrast=name,metric=metric,effect=v))
   for prior in ['P1','P2']:
    for dose in doses:
     for arm in ['Transfer','Seen']:
      a=(prior+'-C',arm,dose);b=(prior+'-I',arm,dose)
      if a in p and b in p:
       for seed,v in (p[a]-p[b]).items():rows.append(dict(zip(groups,key),seed=seed,condition=prior+':Center-I',dose=dose,contrast=arm+':Centered-Independent',metric=metric,effect=v))
 return pd.DataFrame(rows)

def tag(f):return f.condition+'|'+f.arm+'|'+f.dose.astype(str)
def untag(f,col):
 f=f.rename(columns={col:'tag'});v=f.tag.str.split('|',regex=False,expand=True);f['condition']=v[0];f['arm']=v[1];f['dose']=v[2].astype(int);return f.drop(columns='tag')

def main():
 (OUT/'tables').mkdir(parents=True,exist_ok=True)
 assert (OUT/'audit/targets_complete.json').exists();frames,pos,registry=collect();f,acq=gap_records(frames['evaluations']);g=frames['geometry'];t=frames['training'];d=frames['diagnostics'];assert len(f)==1200*331*6 and not f.duplicated(KEY+['cue']).any();bf=f.copy();bf['arm_tag']=tag(bf);b=untag(block_table(bf.rename(columns={'arm':'query_arm','arm_tag':'arm'})),'arm');lf=f[f.cue.isin(['A','B'])].copy();lf['architecture']=tag(lf);s=untag(boundaries(lf),'architecture');wr=[]
 for key,x in d.groupby(KEY):
  v={(r.train_context,r.evaluation_context):r.improvement for r in x.itertuples()};r=dict(zip(KEY,key),delta_A_A=v['A','A'],delta_B_A=v['A','B'],delta_B_B=v['B','B'],delta_A_B=v['B','A']);wr.append(dict(**r,**specificity(r)))
 wide=pd.DataFrame(wr);ww=wide.rename(columns={'update':'checkpoint'}).copy();ww['architecture']=tag(ww);gate=untag(split_summary(lf,ww),'architecture').rename(columns={'endpoint':'update'}).drop(columns=['trajectory_specific_learning','n_prior_direct_positive']);gate['available']=gate.direct_status.ne('unavailable');gate['functional_split']=gate.functional_split.astype(float);gate.loc[~gate.available,'functional_split']=np.nan
 mb=b.copy();mb['condition']=mb.condition+'|'+mb.arm;m=source_matching(mb);v=m.condition.str.split('|',regex=False,expand=True);m['condition']=v[0];m['arm']=v[1]
 end=f[f['update'].isin([0,40,80,120,170,330])];sumep=summarize(end,['condition','arm','dose','update','cue'],['ce','familiar_ce','gap','accuracy','success','strict_ce','strict_accuracy','strict_gap','strict_all','strong_familiar']);epfx=paired_effects(end,['update','cue'],['gap','ce','familiar_ce','accuracy','success','strict_ce','strict_accuracy','strict_gap','strict_all','strong_familiar']);epci=summarize(epfx,['condition','dose','contrast','update','cue','metric'],['effect']);af=acq.copy();af['summary']='post_acquisition';acf=paired_effects(af,['summary'],['post_acquisition_gap','first_familiar']);aci=summarize(acf,['condition','dose','contrast','summary','metric'],['effect']);mf=paired_effects(m,['performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain','start_A_CE','start_B_CE']);mi=summarize(mf,['condition','dose','contrast','performance_range','stratum','metric'],['effect'])
 joint=[]
 for key,x in end[end['update'].isin([170,330])].groupby(KEY):
  seed,c,arm,dose,k=key;x=x.set_index('cue');gt=gate[(gate.seed==seed)&(gate.condition==c)&(gate.arm==arm)&(gate.dose==dose)&(gate['update']==k)].iloc[0];bb=b[(b.seed==seed)&(b.condition==c)&(b.arm==arm)&(b.dose==dose)&(b.end<=k)];early=bb[bb.end<=100];before=bb[(~bb.floor_at_start)&(bb.practiced_gain>0)&(bb.specificity>0)];sign=all(before.practiced.eq(v).any() for v in ['A','B']);valid=bool(gt.available and np.isfinite(x.ce).all());joint.append(dict(zip(KEY,key),available=valid,strong_familiar=x.loc['C','strong_familiar'],C_success=x.loc['C','success'],C_strict=x.loc['C','strict_all'],Mixed_both_success=x.loc['Mixed-both','success'],functional_split=gt.functional_split,original_joint=bool(x.loc['C','strong_familiar']==1 and x.loc['C','success']==1 and x.loc['Mixed-both','success']==1 and gt.functional_split==1) if valid else None,descriptive_coexistence=bool(x.loc['C','strong_familiar']==1 and x.loc['C','success']==1 and x.loc['Mixed-both','success']==1 and sign) if valid else None,early_specificity=early.specificity.mean(),early_own=early.practiced_gain.mean(),early_cross=early.unpracticed_gain.mean(),early_floor=early.floor_at_start.mean()))
 j=pd.DataFrame(joint);curve=f[f.cue=='C'].merge(acq[['seed','condition','arm','dose','first_familiar']],on=['seed','condition','arm','dose']);curve['gap_after_acquisition']=curve.gap.where(curve['update']>=curve.first_familiar)
 poststat=summarize(acq,['condition','arm','dose'],['first_familiar','post_acquisition_gap','post_acquisition_C_CE','post_acquisition_familiar_CE']);ps=summarize(pos,['condition','arm','dose','update','position'],['ce','correct','margin']);errors=[]
 for key,x in pos.groupby(['seed','condition','arm','dose','update']):
  v=x.ce.to_numpy(float);total=v.sum();ordered=np.sort(v)[::-1];errors.append(dict(zip(['seed','condition','arm','dose','update'],key),C_total_CE=total,C_incorrect=int((~x.correct.astype(bool)).sum()),worst_position=int(x.iloc[np.argmax(v)].position) if np.isfinite(v).all() else np.nan,top1_CE_share=ordered[0]/total if total>0 else np.nan,top2_CE_share=ordered[:2].sum()/total if total>0 else np.nan,zero_loss=bool(total==0)))
 cg=f[f.cue=='C'];pp=cg.pivot(index=['seed','condition','dose','update'],columns='arm',values='gap').reset_index();pp=pp[pp.dose.gt(0)].copy();pp['gap_difference']=pp.Transfer-pp.Seen
 tables=dict(gap_difference=pp,evaluations=f,geometry=g,training=t,diagnostics=d,block_transfer=b,switch_costs=s,learning_effects=wide,functional_splitting=gate,matched_acquisition=m,matched_summary=summarize(m,['condition','arm','dose','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain','start_A_CE','start_B_CE']),matched_seed_effects=mf,matched_intervals=mi,endpoints=end,endpoint_summary=sumep,paired_seed_effects=epfx,paired_intervals=epci,acquisition=acq,post_acquisition_summary=poststat,post_acquisition_seed_effects=acf,post_acquisition_intervals=aci,post_acquisition_curve=curve,joint_outcomes=j,joint_summary=summarize(j,['condition','arm','dose','update'],['functional_split','original_joint','descriptive_coexistence','early_specificity','early_own','early_cross','early_floor']),C_positions=pos,C_position_summary=ps,C_error_concentration=pd.DataFrame(errors),physical_run_registry=registry)
 for n,ff in tables.items():write_frame(OUT/'tables'/f'{n}.csv',ff)
 for n,ff,groups,tt,metrics in [('gap_difference_intervals',pp,['condition','dose'],'update',['gap_difference']),('training_intervals',t,['condition','arm','dose'],'update',['ce_before']),('evaluation_intervals',f,['condition','arm','dose','cue'],'update',['ce','gap']),('post_acquisition_curve_intervals',curve,['condition','arm','dose'],'update',['gap_after_acquisition']),('block_intervals',b,['condition','arm','dose','practiced'],'end',['practiced_gain','unpracticed_gain','specificity']),('switch_intervals',s,['condition','arm','dose','direction'],'boundary',['raw_cost','adjusted_cost']),('diagnostic_intervals',wide,['condition','arm','dose'],'update',['mean_own_improvement','mean_cross_improvement','mean_specificity'])]:write_frame(OUT/'tables'/f'{n}.csv',time_intervals(ff,groups,tt,metrics))
 for name,ff,groups,metrics in [('block',b,['end','practiced'],['specificity','practiced_gain','unpracticed_gain']),('diagnostic',wide,['update'],['mean_specificity','mean_own_improvement','mean_cross_improvement']),('mechanism',g[g['update'].isin([0,170,330])],['update'],['baseline_ce','C_ce','C_norm','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm','C_minus_avg_centered_norm','C_minus_avg_CE','C_minus_baseline_CE','C_common_cosine','C_common_weight_distance','hidden_separation','centered_logit_separation'])]:
  ef=paired_effects(ff,groups,metrics);write_frame(OUT/'tables'/f'{name}_seed_effects.csv',ef);write_frame(OUT/'tables'/f'{name}_paired_intervals.csv',summarize(ef,['condition','dose','contrast',*groups,'metric'],['effect']))
 write_frame(OUT/'tables/geometry_summary.csv',summarize(g[g['update'].isin([0,40,80,120,170,330])],['condition','arm','dose','update'],[k for k in g if k not in ['seed','condition','arm','dose','update','physical_run_id','reused','prior','organization'] and pd.api.types.is_numeric_dtype(g[k])]))
 extended_tables(tables);dump(OUT/'audit/analysis_complete.json',dict(passed=True,models=1200,primary='Transfer10000-960 and Transfer-Seen10000; independent P1/P2 separate20pairedseeds',bootstrap=10000,rng=20260925,post_acquisition_inclusive=True,never_acquired_missing=True,selection=False));append('## Analysescomplete\nContinuousC−familiargaps/end330andpostacquisitioninclusive,strictdescriptives,positionerrors,unchangedfunctionalgate,fixedmatchedranges/missingn andpairedblock/probe/mechanismeffectssaved. Wholepairedseeds10kbootstrap,prioraveragen20not40,separateprimaryP1/P2I. No thresholds changed.')


# Additional fixed scaling/first20 descriptions; all original gate/matching helpers above remain unchanged.
def primary_filter(e):
 return e[e.condition.isin(['P1-I','P2-I'])&e.dose.eq(10000)&e['update'].eq(330)&e.cue.eq('C')&e.metric.eq('gap')&e.contrast.isin(['Transfer:10000-960','Transfer-Seen'])]

def scaling_effects(f,groups,metrics):
 rows=[]
 for key,g in f.groupby(groups):
  key=(key,) if not isinstance(key,tuple) else key
  for metric in metrics:
   p=g.pivot(index='seed',columns=['condition','arm','dose'],values=metric).reindex(range(20)).astype(float)
   for c in CONDS:
    doses=sorted({d for cc,a,d in p if cc==c and d>0})
    for dose in doses:
     for arm in ['Transfer','Seen']:
      a=(c,arm,dose);b=(c,arm,960)
      if dose>960 and a in p and b in p:
       for seed,v in (p[a]-p[b]).items():rows.append(dict(zip(groups,key),seed=seed,condition=c,dose=dose,contrast=f'{arm}:{dose}-960',metric=metric,effect=v))
     a=(c,'Transfer',dose);b=(c,'Seen',dose)
     if a in p and b in p:
      for seed,v in (p[a]-p[b]).items():rows.append(dict(zip(groups,key),seed=seed,condition=c,dose=dose,contrast='Transfer-Seen',metric=metric,effect=v))
    for dose in doses:
     tags=[(c,a,d) for a,d in [('Transfer',dose),('Transfer',960),('Seen',dose),('Seen',960)]]
     if dose>960 and all(t in p for t in tags):
      interaction=(p[tags[0]]-p[tags[1]])-(p[tags[2]]-p[tags[3]])
      for seed,v in interaction.items():rows.append(dict(zip(groups,key),seed=seed,condition=c,dose=dose,contrast=f'query_arm_scale_interaction:{dose}-960',metric=metric,effect=v))
    for before,after in zip(doses[:-1],doses[1:]):
     for arm in ['Transfer','Seen']:
      a=(c,arm,after);b=(c,arm,before)
      if a in p and b in p:
       for seed,v in (p[a]-p[b]).items():rows.append(dict(zip(groups,key),seed=seed,condition=c,dose=after,contrast=f'{arm}:adjacent{after}-{before}',metric=metric,effect=v))
 return pd.DataFrame(rows)

def early_updates(f):
 p=f[f.cue.isin(['A','B'])&f['update'].le(20)].pivot(index=KEY,columns='cue',values='ce');rows=[]
 for keys,g in p.reset_index().groupby(['seed','condition','arm','dose']):
  g=g.set_index('update').sort_index()
  for end in range(1,21):
   if end not in g.index or end-1 not in g.index:continue
   before=g.loc[end-1];after=g.loc[end];cue='A' if end<=10 else 'B';other='B' if cue=='A' else 'A';own=before[cue]-after[cue];cross=before[other]-after[other]
   rows.append(dict(zip(['seed','condition','arm','dose'],keys),update=end,practiced=cue,start_A_CE=before.A,start_B_CE=before.B,end_A_CE=after.A,end_B_CE=after.B,own_gain=own,cross_gain=cross,specificity=own-cross,floor_at_start=bool(before.A<=.05 and before.B<=.05)))
 return pd.DataFrame(rows)

def median_interval(x):
 import warnings
 x=np.asarray(x,float);idx=np.random.default_rng(20260925).integers(0,20,(10000,20));assert x.shape==(20,)
 if not np.isfinite(x).any():return dict(median_ci_low=np.nan,median_ci_high=np.nan)
 with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);v=np.nanmedian(x[idx],axis=1)
 lo,hi=np.nanquantile(v,[.025,.975]);return dict(median_ci_low=lo,median_ci_high=hi)

def extended_tables(tables):
 f=tables['evaluations'];end=tables['endpoints'];b=tables['block_transfer'];j=tables['joint_outcomes'];acq=tables['acquisition'];g=tables['geometry'];early=early_updates(f);metrics=['gap','ce','familiar_ce','accuracy','success','strict_ce','strict_accuracy','strict_gap','strict_all','strong_familiar'];fx=scaling_effects(end,['update','cue'],metrics);ci=summarize(fx,['condition','dose','contrast','update','cue','metric'],['effect']);primary=primary_filter(fx);pr=[]
 for key,x in primary.groupby(['condition','dose','contrast','update','cue','metric']):
  v=x.set_index('seed').effect.reindex(range(20)).to_numpy(float);pr.append(dict(zip(['condition','dose','contrast','update','cue','metric'],key),**interval(v),**median_interval(v),n_negative=int((v<0).sum()),n_positive=int((v>0).sum())))
 first20=b[b.end.le(20)].copy();es=first20.groupby(['seed','condition','arm','dose']).agg(early20_specificity=('specificity','mean'),early20_own=('practiced_gain','mean'),early20_cross=('unpracticed_gain','mean'),early20_floor=('floor_at_start','mean')).reset_index();flags=[]
 for key,x in first20.groupby(['seed','condition','arm','dose']):
  positive=(x.specificity>0)&(x.practiced_gain>0)&(~x.floor_at_start)
  valid=len(x)==2 and np.isfinite(x[['practiced_gain','unpracticed_gain','specificity']].to_numpy(float)).all();flags.append(dict(zip(['seed','condition','arm','dose'],key),early20_bidirectional=float(set(x.loc[positive,'practiced'])=={'A','B'}) if valid else np.nan))
 es=es.merge(pd.DataFrame(flags),on=['seed','condition','arm','dose']);jj=j.merge(es,on=['seed','condition','arm','dose'],validate='many_to_one');jj['early20_coexistence']=(jj.strong_familiar==1)&(jj.C_success==1)&(jj.Mixed_both_success==1)&jj.early20_bidirectional.eq(1)
 jj['early20_strict_coexistence']=(jj.strong_familiar==1)&(jj.C_strict==1)&(jj.Mixed_both_success==1)&jj.early20_bidirectional.eq(1)
 for col in ['early20_coexistence','early20_strict_coexistence']:jj.loc[~jj.available,col]=np.nan
 # Keep legacy descriptive_coexistence and unchanged functional_split beside the explicit first20 flag.
 efx=scaling_effects(es.assign(window='first20'),['window'],['early20_specificity','early20_own','early20_cross','early20_floor','early20_bidirectional']);mfx=scaling_effects(tables['matched_acquisition'],['performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain','start_A_CE','start_B_CE']);af=scaling_effects(acq.assign(window='post_acquisition'),['window'],['post_acquisition_gap','first_familiar']);mf=scaling_effects(g[g['update'].isin([0,170,330])],['update'],['baseline_ce','C_ce','C_norm','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm','C_minus_avg_centered_norm','C_minus_avg_CE','C_minus_baseline_CE','C_common_cosine','hidden_separation','centered_logit_separation']);med=[]
 for key,x in end[(end.cue=='C')&end['update'].eq(330)].groupby(['condition','arm','dose']):med.append(dict(zip(['condition','arm','dose'],key),**median_interval(x.set_index('seed').gap.reindex(range(20)).to_numpy(float))))
 pfx=scaling_effects(tables['C_positions'],['update','position'],['ce','correct','margin'])
 exports=dict(position_scaling_seed_effects=pfx,position_scaling_intervals=summarize(pfx,['condition','dose','contrast','update','position','metric'],['effect']),directional_probe_summary=summarize(tables['learning_effects'],['condition','arm','dose','update'],['D_A','D_B','delta_A_A','delta_B_A','delta_B_B','delta_A_B']),scaling_seed_effects=fx,scaling_intervals=ci,primary_seed_effects=primary,primary_intervals=pd.DataFrame(pr),early_updates=early,early20_seed=es,early20_summary=summarize(es,['condition','arm','dose'],['early20_specificity','early20_own','early20_cross','early20_floor','early20_bidirectional']),early20_seed_effects=efx,early20_intervals=summarize(efx,['condition','dose','contrast','window','metric'],['effect']),matched_scaling_seed_effects=mfx,matched_scaling_intervals=summarize(mfx,['condition','dose','contrast','performance_range','stratum','metric'],['effect']),post_scaling_seed_effects=af,post_scaling_intervals=summarize(af,['condition','dose','contrast','window','metric'],['effect']),mechanism_scaling_seed_effects=mf,mechanism_scaling_intervals=summarize(mf,['condition','dose','contrast','update','metric'],['effect']),joint_early20=jj,joint_early20_summary=summarize(jj,['condition','arm','dose','update'],['early20_coexistence','early20_strict_coexistence','early20_bidirectional','functional_split','original_joint','descriptive_coexistence']),gap_median_intervals=pd.DataFrame(med))
 for n,x in exports.items():write_frame(OUT/'tables'/f'{n}.csv',x)
 write_frame(OUT/'tables/early_update_intervals.csv',time_intervals(early,['condition','arm','dose','practiced'],'update',['own_gain','cross_gain','specificity']))

if __name__=='__main__':main()
