"""Fixed outcomes and original functional measures, whole paired seeds only."""
import argparse
from .common import *
from .protocol import SCREEN,CONFIRM
from .statistics import summarize,time_summary
from experiments.centered_context_linear.analysis import block_table
from experiments.meta_learning_scaling.analysis import matched_acquisition
from experiments.recurrent_learning_transfer.measures import boundaries,specificity,split_summary
KEY=['seed','arm','dose'];FULL=KEY+['update']

def gaps(f):
 p=f.pivot(index=FULL,columns='cue',values='ce');a=f.pivot(index=FULL,columns='cue',values='accuracy');q=pd.DataFrame(index=p.index);q['familiar_ce']=(p.A+p.B)/2;q['strong_familiar']=((p.A<=.5)&(p.B<=.5)&(a.A>=6/7)&(a.B>=6/7)).astype(float);q.loc[p.A.isna()|p.B.isna(),'strong_familiar']=np.nan;f=f.merge(q.reset_index(),on=FULL,validate='many_to_one');f['gap']=f.ce-f.familiar_ce;f['success']=((f.ce<=.5)&(f.accuracy>=6/7)).astype(float);f['strict_all']=((f.ce<=.1)&f.accuracy.eq(1)&(f.gap<=.1)).astype(float);f.loc[f.ce.isna(),['success','strict_all']]=np.nan;rows=[]
 for key,g in f[f.cue=='C'].groupby(KEY):
  x=g[g.strong_familiar.eq(1)]['update'];first=x.min() if len(x) else np.nan;after=g[g['update']>=first];rows.append(dict(zip(KEY,key),first_familiar=first,post_acquisition_gap=after.gap.mean(),post_C_ce=after.ce.mean(),post_familiar_ce=after.familiar_ce.mean(),post_states=len(after),status='acquired' if len(x) else 'never_acquired'))
 return f,pd.DataFrame(rows)

def paired_table(f,groups,metrics,seeds,candidates):
 if not set(f.seed).issubset(seeds):raise ValueError('Wrong screening/confirmation seed cohort')
 rows=[]
 for key,g in f.groupby(groups,dropna=False):
  key=key if isinstance(key,tuple) else (key,)
  for metric in metrics:
   p=g.pivot(index='seed',columns='arm',values=metric).reindex(seeds)
   for candidate in candidates:
    if candidate not in p or 'C' not in p:continue
    for seed,v in (p[candidate].astype(float)-p.C.astype(float)).items():rows.append(dict(zip(groups,key),seed=seed,candidate=candidate,contrast=candidate+'-C',metric=metric,effect=v))
 raw=pd.DataFrame(rows)
 if not len(raw):return raw,pd.DataFrame()
 result=[]
 from .statistics import interval
 for key,g in raw.groupby([*groups,'candidate','contrast','metric'],dropna=False):
  result.append(dict(zip([*groups,'candidate','contrast','metric'],key),**interval(g.set_index('seed').effect.reindex(seeds).to_numpy(float),seeds)))
 return raw,pd.DataFrame(result)

def read_dir(path,name):return pd.read_csv(path/name,float_precision='round_trip',low_memory=False)
def collect(stage):
 seeds=SCREEN if stage=='screen' else CONFIRM;reg=read_dir(OUT/'registry',stage+'_targets.csv');rows=reg.to_dict('records');old=read_dir(SCALE/'tables','physical_run_registry.csv');held=read_dir(OUT/'registry',stage+'_heldout.csv');oldheld=read_dir(SOURCE/'registry','heldout_diagnostics_v2.csv');registry=[];frames={n:[] for n in ['evaluations','training','geometry','diagnostics','C_positions']};hs=[];hb=[];hw=[]
 for seed in seeds:
  r=old[(old.seed==seed)&old.condition.eq('P2-I')&old.arm.eq('Transfer')&old.dose.eq(960)].iloc[0].to_dict();rows.append(dict(r,arm='Baseline',dose=0,reused=True))
 for row in rows:
  s=int(row['seed']);a=row['arm'];dose=int(row['dose']);reused=a=='Baseline';d=ROOT/row['directory'] if reused else OUT/row['directory'];assert d.exists();registry.append(dict(seed=s,arm=a,dose=dose,reused=reused,directory=str(d.relative_to(ROOT)),path=str((d/'states.pt').relative_to(ROOT)),sha256=sha(d/'states.pt')))
  meta=dict(seed=s,arm=a,dose=dose)
  for n in ['evaluations','training','diagnostics']:frames[n].append(read_dir(d,n+'.csv').assign(**meta))
  pos=read_dir(d,'position_predictions.csv.gz');gm=read_dir(d,'geometry.csv');margin=pos.groupby(['update','cue']).margin.mean().unstack('cue');gm=gm.merge(margin[['A','B']].rename(columns={'A':'A_margin','B':'B_margin'}).reset_index(),on='update',validate='one_to_one');frames['geometry'].append(gm.assign(**meta));frames['C_positions'].append(pos[pos.cue.eq('C')&pos['update'].isin([0,20,40,80,120,170,330])].assign(**meta))
  if reused:
   hrow=oldheld[(oldheld.seed==s)&oldheld.condition.eq('P2-I')&oldheld.arm.eq('Transfer')&oldheld.dose.eq(960)].iloc[0];hd=SOURCE/hrow.directory;wd=SCALE/f'conflict_wrong_cue/s{s:02d}_P2_I_Transfer_d960'
  else:
   hrow=held[(held.seed==s)&held.arm.eq(a)&held.dose.eq(dose)].iloc[0];hd=OUT/hrow.directory;wd=OUT/f'conflict_wrong_cue/{hrow.run_id}'
  tasks=read_dir(hd,'tasks.csv');hp=read_dir(hd,'positions.csv');blocks=read_dir(hd,'blocks.csv');wrong=read_dir(wd,'cells.csv');assert len(tasks)==12 and len(wrong)==24
  hs.append(dict(**meta,**{n:tasks[n].astype(float).mean() for n in ['ce','accuracy','conflict_accuracy','appropriate_context_change','prefix_centered_separation']},conflict_ce=hp[hp.position.eq(3)].ce.mean(),n_tasks=int(tasks.ce.notna().sum())))
  hb.append(dict(**meta,own_gain=blocks.own_gain.mean(),cross_gain=blocks.cross_gain.mean(),specificity=blocks.specificity.mean()))
  hw.append(dict(**meta,**{n:wrong[n].mean() for n in ['correct_CE_before','correct_CE_after','wrong_CE_before','wrong_CE_after','correct_gain','wrong_gain','context_benefit','context_gain_advantage','conflict_correct_ce','conflict_wrong_ce','conflict_context_benefit']}))
 return {n:pd.concat(v,ignore_index=True) for n,v in frames.items()},pd.DataFrame(hs),pd.DataFrame(hb),pd.DataFrame(hw),pd.DataFrame(registry)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);stage=p.parse_args().stage;seeds=SCREEN if stage=='screen' else CONFIRM;assert json.loads((OUT/f'audit/{stage}_targets_verified.json').read_text())['passed'];assert json.loads((OUT/f'audit/{stage}_heldout_complete.json').read_text())['passed'];frames,held,hblocks,wrong,registry=collect(stage);f,acq=gaps(frames['evaluations']);t=frames['training'];d=frames['diagnostics'];geometry=frames['geometry'];expected=(42 if stage=='screen' else 98)*331*6;assert len(f)==expected and not f.duplicated(FULL+['cue']).any();assert set(f.seed)==set(seeds)
 tag=lambda x:x.arm+'@'+x.dose.astype(str)
 def untag(x,col):
  v=x[col].str.split('@',expand=True);x=x.drop(columns=col);x['arm']=v[0];x['dose']=v[1].astype(int);return x
 bf=f.copy();bf['arm']=tag(f);blocks=untag(block_table(bf),'arm');lf=f[f.cue.isin(['A','B'])].copy();lf['architecture']=tag(lf);switches=untag(boundaries(lf),'architecture');wr=[]
 for key,g in d.groupby(FULL):
  v={(r.train_context,r.evaluation_context):r.improvement for r in g.itertuples()};r=dict(zip(FULL,key),delta_A_A=v['A','A'],delta_B_A=v['A','B'],delta_B_B=v['B','B'],delta_A_B=v['B','A']);wr.append(dict(**r,**specificity(r)))
 wide=pd.DataFrame(wr);ww=wide.rename(columns={'update':'checkpoint'}).copy();ww['architecture']=tag(ww);gate=untag(split_summary(lf,ww),'architecture').rename(columns={'endpoint':'update'});mb=blocks.copy();mb['condition']=mb.arm;matched=matched_acquisition(mb).rename(columns={'condition':'arm'});early=[]
 for key,g in blocks[blocks.end.le(20)].groupby(KEY):
  a=g[g.practiced.eq('A')].iloc[0];b=g[g.practiced.eq('B')].iloc[0];early.append(dict(zip(KEY,key),D_A=a.specificity,D_B=b.specificity,own_A=a.practiced_gain,cross_A=a.unpracticed_gain,own_B=b.practiced_gain,cross_B=b.unpracticed_gain,early_specificity=g.specificity.mean(),early_own=g.practiced_gain.mean(),early_cross=g.unpracticed_gain.mean(),floor_A=a.floor_at_start,floor_B=b.floor_at_start,nonfloor_both=bool(not a.floor_at_start and not b.floor_at_start and a.specificity>0 and b.specificity>0 and a.practiced_gain>0 and b.practiced_gain>0)))
 early=pd.DataFrame(early);end=f[f.cue.eq('C')&f['update'].eq(330)].drop(columns=['cue','status'],errors='ignore');out=end.merge(acq,on=KEY,validate='one_to_one').merge(early,on=KEY,validate='one_to_one').merge(held.rename(columns={'ce':'conflict_overall_ce','accuracy':'conflict_overall_accuracy'}),on=KEY,validate='one_to_one');both=f[f.cue.eq('Mixed-both')&f['update'].eq(330)][KEY+['success','ce']].rename(columns={'success':'Mixed_both_success','ce':'Mixed_both_ce'});out=out.merge(both,on=KEY).merge(gate[gate['update'].eq(330)][KEY+['functional_split']],on=KEY);out['joint']=out.strong_familiar.eq(1)&out.success.eq(1)&out.Mixed_both_success.eq(1)&out.nonfloor_both;out['strict_joint']=out.joint&out.strict_all.eq(1)
 tables=dict(evaluations=f,training=t,geometry=geometry,diagnostics=d,C_positions=frames['C_positions'],acquisition=acq,blocks=blocks,switch_costs=switches,learning_effects=wide,functional_splitting=gate,matched_acquisition=matched,early=early,heldout_seed=held,heldout_block_seed=hblocks,wrong_cue_seed=wrong,seed_outcomes=out,physical_registry=registry)
 candidates=[c for c in out.arm.unique() if c not in ['Baseline','C']];metrics=['gap','ce','familiar_ce','accuracy','success','strict_all','strong_familiar','post_acquisition_gap','first_familiar','D_A','D_B','early_specificity','early_own','early_cross','nonfloor_both','conflict_accuracy','conflict_ce','conflict_overall_ce','appropriate_context_change','Mixed_both_ce','Mixed_both_success','joint','strict_joint','functional_split'];summary=summarize(out,['arm','dose'],metrics,seeds);tables['outcome_summary']=summary
 pairs,pci=paired_table(out,['dose'],metrics,seeds,candidates);tables['paired_seed_effects']=pairs;tables['paired_intervals']=pci
 mf,mc=paired_table(matched,['dose','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain'],seeds,candidates);tables['matched_seed_effects']=mf;tables['matched_intervals']=mc
 for label,frame,groups,timecol,mets in [('training',t,['arm','dose'],'update',['ce_before']),('evaluations',f,['arm','dose','cue'],'update',['ce','gap']),('blocks',blocks,['arm','dose','practiced'],'end',['practiced_gain','unpracticed_gain','specificity']),('switch',switches,['arm','dose','direction'],'boundary',['raw_cost','adjusted_cost']),('diagnostic',wide,['arm','dose'],'update',['D_A','D_B','mean_specificity','mean_own_improvement','mean_cross_improvement'])]:tables[label+'_intervals']=time_summary(frame,groups,timecol,mets,seeds)
 tables['matched_summary']=summarize(matched,['arm','dose','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain'],seeds)
 tables['heldout_summary']=summarize(held,['arm','dose'],['ce','accuracy','conflict_accuracy','conflict_ce','appropriate_context_change'],seeds);tables['wrong_cue_summary']=summarize(wrong,['arm','dose'],[c for c in wrong if c not in KEY],seeds)
 if stage=='confirm':
  geom=geometry[geometry['update'].isin([0,170,330])];mets=[c for c in geom if c not in FULL and pd.api.types.is_numeric_dtype(geom[c])];tables['geometry_summary']=summarize(geom,['arm','dose','update'],mets,seeds);gg,gc=paired_table(geom,['dose','update'],mets,seeds,candidates);tables['mechanism_seed_effects']=gg;tables['mechanism_intervals']=gc
 for n,frame in tables.items():write_frame(OUT/'tables'/f'{stage}_{n}.csv',frame)
 dump(OUT/f'audit/{stage}_analysis_complete.json',dict(passed=True,physical_target_histories=len(registry),frozen_cue_cells=len(f),all_seeds=seeds,new_target_runs=36 if stage=='screen' else 84,reused_baseline_runs=len(seeds),bootstrap=10000,paired_seed_inference=True,screen_and_confirmation_not_pooled=True));append('## '+stage+' analyses complete\nSignedfirstA/Bown/crosseffects/floors,unchangedlatecriterion,allmatchedmissingbins,postacquisitioninclusiveCgap,conflict/wrongcueeffects,allseedjointflagsandpairedmean/median10kbootstrapintervalsretained. Nogeometrysubstitutedforlearning. Scientificcohortsremainseparate.')
if __name__=='__main__':main()
