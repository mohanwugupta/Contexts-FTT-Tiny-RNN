"""Whole-seed repeated-dose inference and floor-aware functional descriptions."""
import argparse
from .common import *
from .protocol import DOSES,MATCH_BINS
from .data import collect
from experiments.centered_context_linear.analysis import block_table,time_intervals
from experiments.recurrent_learning_transfer.measures import boundaries,specificity,split_summary

def paired_interval(a,b):return interval(np.asarray(a,float)-np.asarray(b,float))
def interval(x):
 x=np.asarray(x,float);assert x.shape==(20,);valid=np.isfinite(x);idx=np.random.default_rng(20260925).integers(0,20,(10000,20));a=x[idx];count=np.isfinite(a).sum(1);values=np.divide(np.nansum(a,axis=1),count,out=np.full(10000,np.nan),where=count>0);lo,hi=np.nanquantile(values,[.025,.975]) if valid.any() else [np.nan,np.nan];return dict(mean=float(np.nanmean(x)) if valid.any() else np.nan,median=float(np.nanmedian(x)) if valid.any() else np.nan,ci_low=lo,ci_high=hi,n=int(valid.sum()),n_planned=20)

def matched_acquisition(b):
 rows=[]
 for (seed,c,dose),g in b.groupby(['seed','condition','dose']):
  g=g.sort_values('start');mean=(g.practiced_start_ce+g.unpracticed_start_ce)/2
  for low,high in MATCH_BINS:
   eligible=g[(mean>=low)&((mean<=high) if high==4 else (mean<high))]
   for stratum in ['all','A','B']:
    x=eligible if stratum=='all' else eligible[eligible.practiced==stratum];r=dict(seed=seed,condition=c,dose=dose,performance_range=f'CE{low:g}-{high:g}',stratum=stratum,available=bool(len(x)))
    if len(x):
     q=x.iloc[0];r.update({k:q[k] for k in ['start','end','practiced','practiced_gain','unpracticed_gain','specificity','floor_at_start']});r.update(start_A_CE=q.practiced_start_ce if q.practiced=='A' else q.unpracticed_start_ce,start_B_CE=q.practiced_start_ce if q.practiced=='B' else q.unpracticed_start_ce,mean_start_CE=(q.practiced_start_ce+q.unpracticed_start_ce)/2)
    else:r.update({k:np.nan for k in ['start','end','practiced_gain','unpracticed_gain','specificity','start_A_CE','start_B_CE','mean_start_CE']});r.update(practiced='',floor_at_start=None)
    rows.append(r)
 return pd.DataFrame(rows)

def tag(f):return f.condition+'@'+f.dose.astype(str)
def untag(f,col):
 f=f.rename(columns={col:'tag'});f['condition']=f.tag.str.split('@').str[0];f['dose']=f.tag.str.split('@').str[1].astype(int);return f.drop(columns='tag')

def summarize(f,groups,metrics):
 out=[]
 for key,g in f.groupby(groups):
  key=(key,) if not isinstance(key,tuple) else key
  for metric in metrics:
   x=g.set_index('seed')[metric].reindex(range(20)).to_numpy(float);r=dict(zip(groups,key),**interval(x));r['statistic' if 'metric' in groups else 'metric']=metric;out.append(r)
 return pd.DataFrame(out)

def analyze(stage):
 start=time.monotonic();dest=OUT/'stage_a' if stage=='a' else OUT;td=dest/'tables';td.mkdir(parents=True,exist_ok=True);frames,r=collect('a' if stage=='a' else 'full');f=frames['evaluations'];g=frames['geometry'];t=frames['training'];dg=frames['diagnostics'];doses=DOSES[:6] if stage=='a' else DOSES;keys=['seed','condition','dose','update'];expected=20*4*len(doses)*331*6;assert len(f)==expected and not f.duplicated(keys+['cue']).any();assert set(f.dose)==set(doses)
 p=f.pivot(index=keys,columns='cue',values='ce');a=f.pivot(index=keys,columns='cue',values='accuracy');q=p[['A','B']].copy();q['familiar_ce']=(q.A+q.B)/2;q['signed_A_minus_B']=q.A-q.B;q['absolute_AB_gap']=abs(q.A-q.B);q['strong_familiar']=((q.A<=.5)&(q.B<=.5)&(a.A>=6/7)&(a.B>=6/7)).astype(float);q.loc[q.A.isna()|q.B.isna(),'strong_familiar']=np.nan;q=q.drop(columns=['A','B']).reset_index();g=q[keys].merge(g,on=keys,how='left',validate='one_to_one');f=f.merge(q,on=keys,validate='many_to_one');f['gap']=f.ce-f.familiar_ce;f['success']=((f.ce<=.5)&(f.accuracy>=6/7)).astype(float);f.loc[f.ce.isna()|f.accuracy.isna(),'success']=np.nan
 bf=f.copy();bf['arm']=tag(bf);b=untag(block_table(bf),'arm');lf=f[f.cue.isin(['A','B'])].copy();lf['architecture']=tag(lf);s=untag(boundaries(lf),'architecture');wr=[]
 for (seed,c,dose,k),gg in dg.groupby(keys):
  dd={(x.train_context,x.evaluation_context):x.improvement for x in gg.itertuples()};rr=dict(seed=seed,condition=c,dose=dose,update=k,delta_A_A=dd['A','A'],delta_B_A=dd['A','B'],delta_B_B=dd['B','B'],delta_A_B=dd['B','A']);wr.append(dict(**rr,**specificity(rr)))
 wide=pd.DataFrame(wr);ww=wide.rename(columns={'update':'checkpoint'}).copy();ww['architecture']=tag(ww);gate=untag(split_summary(lf,ww),'architecture').rename(columns={'endpoint':'update'}).drop(columns=['trajectory_specific_learning','n_prior_direct_positive']);gate['gate_available']=gate.direct_status.ne('unavailable');gate.loc[~gate.gate_available,'functional_split']=np.nan;m=matched_acquisition(b);end=f[f['update'].isin([0,80,100,120,170,330])].copy();joint=[];timing=[]
 for (seed,c,dose),gg in q.groupby(['seed','condition','dose']):
  hit=gg.loc[gg.strong_familiar==1,'update'];tm=dict(seed=seed,condition=c,dose=dose,first_familiar=float(hit.min()) if len(hit) else np.nan,familiar_success_states=len(hit));allf=f[(f.seed==seed)&(f.condition==c)&(f.dose==dose)]
  for cue in ['C','Mixed-both']:
   hh=allf[(allf.cue==cue)&(allf.success==1)]['update'];tm['first_'+cue]=float(hh.min()) if len(hh) else np.nan
  timing.append(tm)
  bb=b[(b.seed==seed)&(b.condition==c)&(b.dose==dose)]
  for k in [170,330]:
   x=allf[allf['update']==k].set_index('cue');gt=gate[(gate.seed==seed)&(gate.condition==c)&(gate.dose==dose)&(gate['update']==k)].iloc[0];available=bool(np.isfinite(x.ce).all() and gt.direct_status!='unavailable');past=bb[(bb.end<=k)&(~bb.floor_at_start)&(bb.practiced_gain>0)&(bb.specificity>0)];flags={cue:bool(past.practiced.eq(cue).any()) for cue in ['A','B']};early=bb[bb.end<=100];qq=gg[gg['update']==k].iloc[0];jr=dict(seed=seed,condition=c,dose=dose,update=k,available=available,strong_familiar=qq.strong_familiar,C_success=x.loc['C','success'],Mixed_both_success=x.loc['Mixed-both','success'],functional_split=bool(gt.functional_split) if available else None,joint_original=bool(qq.strong_familiar==1 and x.loc['C','success']==1 and x.loc['Mixed-both','success']==1 and gt.functional_split) if available else None,earlier_nonfloor_A=flags['A'],earlier_nonfloor_B=flags['B'],early_specificity=early.specificity.mean(),early_own_gain=early.practiced_gain.mean(),early_cross_gain=early.unpracticed_gain.mean(),early_floor_fraction=early.floor_at_start.mean(),nonfloor_coexistence=bool(qq.strong_familiar==1 and x.loc['C','success']==1 and x.loc['Mixed-both','success']==1 and all(flags.values())) if available else None);joint.append(jr)
 joint=pd.DataFrame(joint);timing=pd.DataFrame(timing);summary=summarize(end,['condition','dose','update','cue'],['ce','accuracy','gap','success','strong_familiar']);js=summarize(joint,['condition','dose','update'],['joint_original','functional_split','nonfloor_coexistence','early_specificity','early_own_gain','early_cross_gain','early_floor_fraction']);ms=summarize(m,['condition','dose','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain','start_A_CE','start_B_CE']);ac=summarize(timing,['condition','dose'],['first_familiar']);gs=summarize(g[g['update'].isin([0,170,330])],['condition','dose','update'],['baseline_ce','C_ce','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm','A_norm','B_norm','C_norm','g_norm','d_norm','hidden_separation','centered_logit_separation','baseline_margin','C_margin'])
 effects=[]
 for (k,cue),xx in end.groupby(['update','cue']):
  for metric in ['ce','accuracy','success','gap','strong_familiar']:
   pp=xx.pivot(index='seed',columns=['condition','dose'],values=metric).reindex(range(20))
   for org in ['I','C']:
    for dose in doses:
     for aa,bb,name in [(('P2-'+org,dose),('P1-'+org,dose),'P2-P1'),(('P1-'+org,dose),('P1-'+org,0),'P1-P0'),(('P2-'+org,dose),('P2-'+org,0),'P2-P0')]:
      for seed,v in (pp[aa]-pp[bb]).items():effects.append(dict(seed=seed,organization=org,contrast=name,dose=dose,update=k,cue=cue,metric=metric,effect=v))
    if stage!='a':
     for prior in ['P1','P2']:
      for seed,v in (pp[(prior+'-'+org,1920)]-pp[(prior+'-'+org,240)]).items():effects.append(dict(seed=seed,organization=org,contrast=prior+':1920-240',dose=1920,update=k,cue=cue,metric=metric,effect=v))
     dd=pp[('P2-'+org,1920)]-pp[('P1-'+org,1920)]-pp[('P2-'+org,240)]+pp[('P1-'+org,240)]
     for seed,v in dd.items():effects.append(dict(seed=seed,organization=org,contrast='change_P2-P1:1920-240',dose=1920,update=k,cue=cue,metric=metric,effect=v))
 ef=pd.DataFrame(effects);es=summarize(ef,['organization','contrast','dose','update','cue','metric'],['effect']);matched_effect=[]
 for (perf,stratum),gg in m.groupby(['performance_range','stratum']):
  for metric in ['specificity','practiced_gain','unpracticed_gain']:
   pp=gg.pivot(index='seed',columns=['condition','dose'],values=metric).reindex(range(20))
   for org in ['I','C']:
    for dose in doses:
     pairs=[(('P2-'+org,dose),('P1-'+org,dose),'P2-P1')]
     for prior in ['P1','P2']:pairs.extend([((prior+'-'+org,dose),(prior+'-'+org,0),prior+':dose-0'),((prior+'-'+org,dose),(prior+'-'+org,240),prior+':dose-240')])
     for aa,bb,name in pairs:
      for seed,v in (pp[aa]-pp[bb]).items():matched_effect.append(dict(seed=seed,organization=org,dose=dose,contrast=name,performance_range=perf,stratum=stratum,metric=metric,effect=v))
 me=pd.DataFrame(matched_effect);mi=summarize(me,['organization','dose','contrast','performance_range','stratum','metric'],['effect'])
 tables=dict(evaluations=f,training=t,geometry=g,diagnostics=dg,block_transfer=b,switch_costs=s,learning_effects=wide,functional_splitting=gate,matched_acquisition=m,matched_summary=ms,performance_gaps=q,endpoints=end,endpoint_summary=summary,joint_outcomes=joint,joint_summary=js,acquisition_timing=timing,acquisition_summary=ac,geometry_summary=gs,paired_seed_effects=ef,paired_intervals=es,matched_seed_effects=me,matched_intervals=mi,physical_run_registry=r)
 for name,frame in tables.items():write_frame(td/f'{name}.csv',frame)
 # Pointwise temporal bootstrap reuses the same20seed resamples in every repeated panel.
 for name,frame,groups,tt,metrics in [('training_intervals',t,['condition','dose'],'update',['ce_before']),('evaluation_intervals',f,['condition','dose','cue'],'update',['ce']),('block_intervals',b,['condition','dose','practiced'],'end',['practiced_gain','unpracticed_gain','specificity']),('switch_intervals',s,['condition','dose','direction'],'boundary',['raw_cost','adjusted_cost']),('diagnostic_intervals',wide,['condition','dose'],'update',['mean_own_improvement','mean_cross_improvement','mean_specificity'])]:write_frame(td/f'{name}.csv',time_intervals(frame,groups,tt,metrics))
 dump(dest/'analysis_complete.json',dict(doses=doses,physical_runs=len(r),logical_references=20*4*len(doses),complete_seeds=20,bootstrap=10000,rng=20260925,wall_seconds=time.monotonic()-start,matched_conditional=True,original_gate='unchanged',selection=False));print('Analysis complete '+stage,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['a','full'],required=True);analyze(p.parse_args().stage)
