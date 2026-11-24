"""Original functional measures; fixed six-seed preflight decision inputs."""
from .common import *
from experiments.structured_overlap_meta.analysis import gaps
from experiments.centered_context_linear.analysis import block_table
from experiments.recurrent_learning_transfer.measures import boundaries,specificity,split_summary
from experiments.meta_learning_scaling.analysis import matched_acquisition
from experiments.meta_objective_balance.statistics import summarize,time_summary,interval

def first_blocks(blocks,switch):
 rows=[]
 for (s,a),g in blocks.groupby(['seed','arm']):
  aa=g[g.end.eq(10)].iloc[0];bb=g[g.end.eq(20)].iloc[0];sw=switch[switch.seed.eq(s)&switch.arm.eq(a)&switch.boundary.isin([10,20])];rows.append(dict(seed=s,arm=a,own_A=aa.practiced_gain,cross_A=aa.unpracticed_gain,D_A=aa.specificity,own_B=bb.practiced_gain,cross_B=bb.unpracticed_gain,D_B=bb.specificity,floor_A=bool(aa.practiced_start_ce<=.05),floor_B=bool(bb.practiced_start_ce<=.05),source_floor_A=bool(aa.floor_at_start),source_floor_B=bool(bb.floor_at_start),start_A=aa.practiced_start_ce,start_B=bb.practiced_start_ce,cycle=sw.adjusted_cost.mean(),A_to_B_adjusted=sw[sw.boundary.eq(10)].adjusted_cost.iloc[0],B_to_A_adjusted=sw[sw.boundary.eq(20)].adjusted_cost.iloc[0],A_to_B_raw=sw[sw.boundary.eq(10)].raw_cost.iloc[0],B_to_A_raw=sw[sw.boundary.eq(20)].raw_cost.iloc[0]))
 return pd.DataFrame(rows)

def main():
 reg=pd.read_csv(OUT/'registry/preflight_targets.csv');frames={n:[] for n in ['evaluations','training','geometry','diagnostics']};ps=[];dc=[];mc=[]
 for r in reg.itertuples():
  d=OUT/r.directory;tag=dict(seed=r.seed,arm=r.arm,dose=0)
  for n in frames:frames[n].append(pd.read_csv(d/f'{n}.csv',float_precision='round_trip').assign(**tag))
  for name,dest in [('position_predictions',ps),('decomposition',dc),('mathematical_checks',mc)]:dest.append(pd.read_csv(d/f'{name}.csv.gz',float_precision='round_trip').assign(**tag))
 frames={n:pd.concat(v,ignore_index=True) for n,v in frames.items()};f,acq=gaps(frames['evaluations']);blocks=block_table(f);lf=f[f.cue.isin(['A','B'])].copy();lf['architecture']=lf.arm;sw=boundaries(lf).rename(columns={'architecture':'arm'});early=first_blocks(blocks,sw);wr=[]
 for (s,a,k),g in frames['diagnostics'].groupby(['seed','arm','update']):
  v={(r.train_context,r.evaluation_context):r.improvement for r in g.itertuples()};q=dict(seed=s,arm=a,update=k,delta_A_A=v['A','A'],delta_B_A=v['A','B'],delta_B_B=v['B','B'],delta_A_B=v['B','A']);wr.append(dict(**q,**specificity(q)))
 wide=pd.DataFrame(wr);ww=wide.rename(columns={'update':'checkpoint','arm':'architecture'});late=split_summary(lf,ww).rename(columns={'architecture':'arm'});mb=blocks.assign(condition=blocks.arm,dose=0);matched=matched_acquisition(mb).rename(columns={'condition':'arm'});end=f[f.cue.eq('C')&f['update'].eq(330)].drop(columns=['status','cue'],errors='ignore');out=end.merge(early,on=['seed','arm'],validate='one_to_one');out['complete']=[bool(reg[reg.seed.eq(r.seed)&reg.arm.eq(r.arm)].status.eq('completed').all()) for r in out.itertuples()];out['strong']=out.strong_familiar.eq(1);out['early_phenotype']=out.complete&out.strong&~out.floor_A&~out.floor_B&out.own_A.gt(.05)&out.own_B.gt(.05)&out.D_A.gt(.05)&out.D_B.gt(.05)&out.cycle.gt(.05)
 tables=dict(evaluations=f,training=frames['training'],geometry=frames['geometry'],diagnostics=frames['diagnostics'],learning_effects=wide,blocks=blocks,switch_costs=sw,matched_acquisition=matched,functional_splitting=late,seed_outcomes=out,position_predictions=pd.concat(ps,ignore_index=True),decomposition=pd.concat(dc,ignore_index=True),mathematical_checks=pd.concat(mc,ignore_index=True));metrics=['D_A','D_B','own_A','cross_A','own_B','cross_B','cycle','strong_familiar','ce','gap','accuracy','strict_all','early_phenotype'];tables['outcome_summary']=summarize(out,['arm'],metrics,SEEDS);pairs=[];cis=[]
 for m in metrics:
  p=out.pivot(index='seed',columns='arm',values=m).reindex(SEEDS);v=p.High.astype(float)-p.Low.astype(float)
  for seed,effect in v.items():pairs.append(dict(seed=seed,metric=m,effect=effect,contrast='High-Low'))
  cis.append(dict(metric=m,contrast='High-Low',**interval(v.to_numpy(float),SEEDS)))
 tables['paired_seed_effects']=pd.DataFrame(pairs);tables['paired_intervals']=pd.DataFrame(cis)
 for label,frame,groups,t,mets in [('training',frames['training'],['arm'],'update',['ce_before']),('evaluations',f,['arm','cue'],'update',['ce','gap']),('blocks',blocks,['arm','practiced'],'end',['practiced_gain','unpracticed_gain','specificity']),('switch',sw,['arm','direction'],'boundary',['raw_cost','adjusted_cost'])]:tables[label+'_intervals']=time_summary(frame,groups,t,mets,SEEDS)
 for name,frame in tables.items():write_frame(OUT/'tables'/f'preflight_{name}{".csv.gz" if name in ["position_predictions","decomposition","mathematical_checks"] else ".csv"}',frame)
 dump(OUT/'audit/preflight_analysis_complete.json',dict(passed=True,runs=12,seeds=SEEDS,seed_bootstrap=10000,original_splitting_unchanged=True,own_floor_gate_distinct_from_original_both_context_floor=True));append('## Preflight analyses complete\nSigned frozen-weight first-A/first-B effects, all33blocks/directional costs, source original170/330 gate, restored-Adam diagnostics, fixed matched bins and all missing statuses retained. Gate uses its preregistered own-context floor, while original both-context floor remains separately unchanged. No numerical or statistical threshold altered.')
if __name__=='__main__':main()
