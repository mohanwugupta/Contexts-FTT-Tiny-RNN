"""Original functional/transfer metrics at every fixed phase/dose; six paired seeds."""
from .common import *
from .analysis import first_blocks
from experiments.structured_overlap_meta.analysis import gaps
from experiments.centered_context_linear.analysis import block_table
from experiments.recurrent_learning_transfer.measures import boundaries,specificity,split_summary
from experiments.meta_learning_scaling.analysis import matched_acquisition
from experiments.meta_objective_balance.statistics import summarize,time_summary,interval
KEY=['seed','arm','dose'];FULL=KEY+['update']

def decorate(f):
 f=f.copy();f['condition']=f.arm.str.split('-').str[0];f['phase']=f.arm.str.split('-').str[1].fillna('Initial');return f

def paired(f,groups,metrics):
 if not set(f.seed).issubset(SEEDS):raise ValueError("Wrong paired-seed cohort")
 rows=[];cis=[]
 for key,g in decorate(f).groupby(groups,dropna=False):
  key=key if isinstance(key,tuple) else (key,)
  if set(g.condition)!={'U','P'}:continue
  for m in metrics:
   p=g.pivot(index='seed',columns='condition',values=m).reindex(SEEDS);v=p.P.astype(float)-p.U.astype(float)
   for seed,x in v.items():rows.append(dict(zip(groups,key),seed=seed,metric=m,effect=x,contrast='P-U'))
   cis.append(dict(zip(groups,key),metric=m,contrast='P-U',**interval(v.to_numpy(float),SEEDS)))
 return pd.DataFrame(rows),pd.DataFrame(cis)

def main():
 assert json.loads((OUT/'audit/targets_verified.json').read_text())['passed'] and json.loads((OUT/'audit/heldout_verified.json').read_text())['passed'];reg=pd.read_csv(OUT/'registry/targets.csv');pre=pd.read_csv(OUT/'registry/preflight_targets.csv');reg=pd.concat([reg,pre[pre.arm.eq('High')]],ignore_index=True);hr=pd.read_csv(OUT/'registry/heldout.csv');frames={n:[] for n in ['evaluations','training','geometry','diagnostics']};pos=[];dec=[];maths=[];held=[];wrong=[];physical=[];sources=[]
 for r in reg.itertuples():
  meta=dict(seed=r.seed,arm=r.arm,dose=r.dose);d=OUT/r.directory;physical.append(dict(**meta,directory=str(d.relative_to(ROOT)),path=str((d/'states.pt').relative_to(ROOT)),sha256=sha(d/'states.pt'),reused_preflight=r.arm=='High'))
  for n in frames:frames[n].append(pd.read_csv(d/f'{n}.csv',float_precision='round_trip').assign(**meta))
  for name,target in [('position_predictions',pos),('decomposition',dec)]:
   q=pd.read_csv(d/f'{name}.csv.gz',float_precision='round_trip');target.append(q[q['update'].isin([0,10,20,40,80,120,170,330])].assign(**meta))
  maths.append(pd.read_csv(d/'mathematical_checks.csv.gz',float_precision='round_trip').assign(**meta));sources.append(dict(**meta,position_source=str((d/'position_predictions.csv.gz').relative_to(ROOT)),position_sha=sha(d/'position_predictions.csv.gz'),decomposition_source=str((d/'decomposition.csv.gz').relative_to(ROOT)),decomposition_sha=sha(d/'decomposition.csv.gz')));rr=hr[hr.seed.eq(r.seed)&hr.arm.eq(r.arm)&hr.dose.eq(r.dose)].iloc[0];hd=OUT/rr.directory;h=pd.read_csv(hd/'tasks.csv');hp=pd.read_csv(hd/'positions.csv');hb=pd.read_csv(hd/'blocks.csv');w=pd.read_csv(OUT/f'conflict_wrong_cue/{rr.run_id}/cells.csv');held.append(dict(**meta,conflict_accuracy=h.conflict_accuracy.mean(),conflict_ce=hp[hp.position.eq(3)].ce.mean(),appropriate_context_change=h.appropriate_context_change.mean(),conflict_overall_ce=h.ce.mean(),conflict_overall_accuracy=h.accuracy.mean(),conflict_own_gain=hb.own_gain.mean(),conflict_cross_gain=hb.cross_gain.mean(),conflict_specificity=hb.specificity.mean()));wrong.append(dict(**meta,**{m:w[m].mean() for m in ['correct_CE_before','correct_CE_after','wrong_CE_before','wrong_CE_after','context_benefit','context_gain_advantage','conflict_correct_ce','conflict_wrong_ce','conflict_context_benefit']}))
 frames={n:pd.concat(v,ignore_index=True) for n,v in frames.items()};f,acq=gaps(frames['evaluations']);tag=lambda q:q.arm+'@'+q.dose.astype(str)
 def untag(q,col):
  v=q[col].str.rsplit('@',n=1,expand=True);q=q.drop(columns=col);q['arm']=v[0];q['dose']=v[1].astype(int);return q
 tagged=f.copy();tagged['arm']=tag(f);blocks=untag(block_table(tagged),'arm');lf=f[f.cue.isin(['A','B'])].copy();lf['architecture']=tag(lf);sw=untag(boundaries(lf),'architecture');wr=[]
 for key,g in frames['diagnostics'].groupby(FULL):
  v={(r.train_context,r.evaluation_context):r.improvement for r in g.itertuples()};q=dict(zip(FULL,key),delta_A_A=v['A','A'],delta_B_A=v['A','B'],delta_B_B=v['B','B'],delta_A_B=v['B','A']);wr.append(dict(**q,**specificity(q)))
 wide=pd.DataFrame(wr);ww=wide.rename(columns={'update':'checkpoint'}).copy();ww['architecture']=tag(ww);late=untag(split_summary(lf,ww),'architecture').rename(columns={'endpoint':'update'});mb=blocks.assign(condition=blocks.arm);matched=matched_acquisition(mb).rename(columns={'condition':'arm'});tb=blocks.copy();ts=sw.copy();tb['arm']=tag(tb);ts['arm']=tag(ts);early=untag(first_blocks(tb,ts),'arm');end=f[f.cue.eq('C')&f['update'].eq(330)].drop(columns=['cue','status'],errors='ignore');out=end.merge(early,on=KEY,validate='one_to_one').merge(pd.DataFrame(held),on=KEY,validate='one_to_one').merge(acq,on=KEY,validate='one_to_one');out['early_phenotype']=out.strong_familiar.eq(1)&~out.floor_A&~out.floor_B&out.own_A.gt(.05)&out.own_B.gt(.05)&out.D_A.gt(.05)&out.D_B.gt(.05)&out.cycle.gt(.05)
 for cue,label in [('Mixed-average','Mixed_average'),('Mixed-both','Mixed_both'),('NoContext','NoContext')]:
  q=f[f.cue.eq(cue)&f['update'].eq(330)][KEY+['ce','accuracy','success']].rename(columns={m:label+'_'+m for m in ['ce','accuracy','success']});out=out.merge(q,on=KEY)
 out=out.merge(late[late['update'].eq(330)][KEY+['functional_split']],on=KEY);tables=dict(evaluations=f,training=frames['training'],geometry=frames['geometry'],diagnostics=frames['diagnostics'],learning_effects=wide,blocks=blocks,switch_costs=sw,matched_acquisition=matched,functional_splitting=late,seed_outcomes=out,heldout_seed=pd.DataFrame(held),wrong_cue_seed=pd.DataFrame(wrong),physical_registry=pd.DataFrame(physical),source_tables=pd.DataFrame(sources),position_predictions=pd.concat(pos,ignore_index=True),decomposition=pd.concat(dec,ignore_index=True),mathematical_checks=pd.concat(maths,ignore_index=True));metrics=['D_A','D_B','own_A','cross_A','own_B','cross_B','cycle','familiar_ce','ce','gap','accuracy','strict_all','success','strong_familiar','early_phenotype','functional_split','conflict_accuracy','conflict_ce','appropriate_context_change','conflict_own_gain','conflict_cross_gain','Mixed_average_ce','Mixed_average_accuracy','Mixed_both_ce','Mixed_both_accuracy','Mixed_both_success','NoContext_ce','NoContext_accuracy'];tables['outcome_summary']=summarize(out,['arm','dose'],metrics,SEEDS);raw,ci=paired(out,['phase','dose'],metrics);tables['paired_seed_effects']=raw;tables['paired_intervals']=ci;tables['endpoint_cue_summary']=summarize(f[f['update'].eq(330)],['arm','dose','cue'],['ce','gap','accuracy','success','strict_all'],SEEDS)
 for label,frame,groups,time,mets in [('training',frames['training'],['arm','dose'],'update',['ce_before']),('evaluations',f,['arm','dose','cue'],'update',['ce','gap']),('blocks',blocks,['arm','dose','practiced'],'end',['practiced_gain','unpracticed_gain','specificity']),('switch',sw,['arm','dose','direction'],'boundary',['raw_cost','adjusted_cost'])]:tables[label+'_intervals']=time_summary(frame,groups,time,mets,SEEDS)
 tables['matched_summary']=summarize(matched,['arm','dose','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain'],SEEDS)
 for label,frame,groups,mets in [('switch',sw,['phase','dose','direction','boundary'],['raw_cost','adjusted_cost']),('matched',matched,['phase','dose','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain']),('mechanism',frames['geometry'][frames['geometry']['update'].isin([0,170,330])],['phase','dose','update'],['hidden_separation','centered_logit_separation','baseline_ce','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm'])]:
  r,c=paired(frame,groups,mets);tables[label+'_paired_seeds']=r;tables[label+'_paired_intervals']=c
 for name,frame in tables.items():write_frame(OUT/'tables'/f'meta_{name}{".csv.gz" if name in ["position_predictions","decomposition","mathematical_checks"] else ".csv"}',decorate(frame) if 'arm' in frame else frame)
 aliases=[]
 for s in SEEDS:
  for a in ['U','P']:
   for phase,dose,physical_arm,physical_dose in [('R',0,'High',0),('F',0,a+'-R',1920)]:aliases.append(dict(seed=s,condition=a,phase=phase,dose=dose,physical_arm=physical_arm,physical_dose=physical_dose,independent_target_run=False))
 write_frame(OUT/'registry/logical_checkpoint_aliases.csv',aliases);dump(OUT/'audit/final_analysis_complete.json',dict(passed=True,physical_histories=78,new_trained_target_histories=72,reused_preflight_high=6,seed_unit=6,all_prespecified_checkpoints=True,bootstrap=10000,no_favorable_checkpoint_selection=True));append('## All fixed-checkpoint analyses complete\nAll78physical main histories (72new/6preflight High reused), allfixedphase/dose comparisons, signed own/cross and directional switches, original170/330 gates, missing matched bins, sixcue absolute/relative losses, conflict/wrongcue capabilities and exact Linear geometry retained. Whole6seed paired mean/median bootstraps, no task/transition resampling or favorable checkpoint selection. F0/R0 alias registry avoids duplicate training.')
if __name__=='__main__':main()
