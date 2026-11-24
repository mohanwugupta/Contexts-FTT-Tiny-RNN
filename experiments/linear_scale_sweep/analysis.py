import argparse
from .common import *
from experiments.structured_overlap_meta.analysis import gaps
from experiments.centered_context_linear.analysis import block_table
from experiments.recurrent_learning_transfer.measures import boundaries,specificity,split_summary
from experiments.meta_learning_scaling.analysis import matched_acquisition
from experiments.high_scale_constrained_meta.analysis import first_blocks
from experiments.meta_objective_balance.statistics import summarize,time_summary,interval
KEY=['seed','arm','dose'];FULL=KEY+['update']

def transition_summary(sw):
 rows=[]
 for key,g in sw.groupby(KEY):
  early=g[g.boundary.le(60)];later=g[g.boundary.between(30,60)];cycles=[g[g.boundary.isin([k-10,k])].adjusted_cost.mean() for k in [20,40,60]];rows.append(dict(zip(KEY,key),early_cost=early.adjusted_cost.mean(),sustained_cost=later.adjusted_cost.mean(),A_to_B=early[early.boundary.mod(20).eq(10)].adjusted_cost.mean(),B_to_A=early[early.boundary.mod(20).eq(0)].adjusted_cost.mean(),early_raw=early.raw_cost.mean(),headroom=int(early.incoming_ce.gt(.05).sum()),later_headroom=int(later.incoming_ce.gt(.05).sum()),positive_later_cycles=sum(v>0 for v in cycles[1:]),**{f'cycle_{i+1}':v for i,v in enumerate(cycles)}))
 return pd.DataFrame(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','replicate'],required=True);stage=p.parse_args().stage;seeds=list(range(6)) if stage=='screen' else list(range(6,20));reg=pd.read_csv(OUT/f'registry/{stage}_runs.csv');fs=[];ts=[];gs=[];ds=[];pc=[]
 for r in reg.to_dict('records'):
  meta=dict(seed=r['seed'],arm=r['arm'],dose=r['dose']);d=(ROOT if r['reused'] else OUT)/r['directory']
  if r['status']!='completed' and not (d/'evaluations.csv').exists():
   fs.append(pd.DataFrame([dict(**meta,update=k,cue=c,ce=np.nan,accuracy=np.nan,status=r['status']) for k in range(331) for c in CUES]));continue
  f=pd.read_csv(d/'evaluations.csv',float_precision='round_trip');fs.append(f.assign(**meta));ts.append(pd.read_csv(d/'training.csv',float_precision='round_trip').assign(**meta));gs.append(pd.read_csv(d/'geometry.csv',float_precision='round_trip').assign(**meta));ds.append(pd.read_csv(d/'diagnostics.csv',float_precision='round_trip').assign(**meta));pos=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip')
  for k in range(10,331,10):
   old='A' if k%20==10 else 'B';new='B' if old=='A' else 'A';a=pos[pos['update'].eq(k)&pos.cue.eq(old)].sort_values('position');b=pos[pos['update'].eq(k)&pos.cue.eq(new)].sort_values('position');a0=pos[pos['update'].eq(0)&pos.cue.eq(old)].sort_values('position');b0=pos[pos['update'].eq(0)&pos.cue.eq(new)].sort_values('position')
   for j in range(7):pc.append(dict(**meta,boundary=k,position=j+1,outgoing_context=old,incoming_context=new,incoming_ce=b.ce.iloc[j],outgoing_ce=a.ce.iloc[j],raw_cost=b.ce.iloc[j]-a.ce.iloc[j],adjusted_cost=(b.ce.iloc[j]-a.ce.iloc[j])-(b0.ce.iloc[j]-a0.ce.iloc[j]),incoming_accuracy=float(b.correct.iloc[j]),outgoing_accuracy=float(a.correct.iloc[j])))
 f,acq=gaps(pd.concat(fs,ignore_index=True));t=pd.concat(ts,ignore_index=True);geom=pd.concat(gs,ignore_index=True);diag=pd.concat(ds,ignore_index=True);tag=lambda q:q.arm+'@'+q.dose.astype(str)
 def untag(q,col):
  v=q[col].str.rsplit('@',n=1,expand=True);q=q.drop(columns=col);q['arm']=v[0];q['dose']=v[1].astype(int);return q
 z=f.copy();z['arm']=tag(z);blocks=untag(block_table(z),'arm');lf=f[f.cue.isin(['A','B'])].copy();lf['architecture']=tag(lf);sw=untag(boundaries(lf),'architecture');positions=pd.DataFrame(pc);acc=positions.groupby(KEY+['boundary'])[['incoming_accuracy','outgoing_accuracy']].mean().reset_index();sw=sw.merge(acc,on=KEY+['boundary'],how='left');sw['accuracy_cost']=sw.outgoing_accuracy-sw.incoming_accuracy;summary=transition_summary(sw);bb=blocks.copy();ss=sw.copy();bb['arm']=tag(bb);ss['arm']=tag(ss);early=untag(first_blocks(bb,ss),'arm');wide=[]
 for key,g in diag.groupby(FULL):
  v={(r.train_context,r.evaluation_context):r.improvement for r in g.itertuples()};q=dict(zip(FULL,key),delta_A_A=v['A','A'],delta_B_A=v['A','B'],delta_B_B=v['B','B'],delta_A_B=v['B','A']);wide.append(dict(**q,**specificity(q)))
 wide=pd.DataFrame(wide);ww=wide.rename(columns={'update':'checkpoint'}).copy();ww['architecture']=tag(ww);late=untag(split_summary(lf,ww),'architecture').rename(columns={'endpoint':'update'});matched=matched_acquisition(blocks.assign(condition=blocks.arm)).rename(columns={'condition':'arm'});out=f[f.cue.eq('C')&f['update'].eq(330)].drop(columns=['cue','status'],errors='ignore').merge(early,on=KEY).merge(summary,on=KEY).merge(acq,on=KEY)
 for cue,label in [('Mixed-average','Mixed_average'),('Mixed-both','Mixed_both'),('NoContext','NoContext')]:out=out.merge(f[f.cue.eq(cue)&f['update'].eq(330)][KEY+['ce','accuracy','success']].rename(columns={v:label+'_'+v for v in ['ce','accuracy','success']}),on=KEY)
 init=f[f['update'].eq(0)&f.cue.isin(['A','B'])].pivot(index=KEY,columns='cue',values=['ce','accuracy']);init.columns=['initial_'+a+'_'+b for a,b in init.columns];out=out.merge(init.reset_index(),on=KEY);out['complete']=[reg[reg.seed.eq(r.seed)&reg.arm.eq(r.arm)&reg.dose.eq(r.dose)].status.eq('completed').all() for r in out.itertuples()];out['early_phenotype']=out.complete&out.strong_familiar.eq(1)&~out.floor_A&~out.floor_B&out.own_A.gt(.05)&out.own_B.gt(.05)&out.D_A.gt(.05)&out.D_B.gt(.05)&out.cycle.gt(.05);out['sustained']=out.complete&out.sustained_cost.gt(0)&out.later_headroom.ge(2)&out.positive_later_cycles.eq(2);out=out.merge(late[late['update'].eq(330)][KEY+['functional_split']],on=KEY);out.loc[~out.complete,'functional_split']=np.nan
 tables=dict(evaluations=f,training=t,geometry=geom,diagnostics=diag,blocks=blocks,switch_costs=sw,position_switch_costs=positions,matched=matched,learning_effects=wide,functional_splitting=late,outcomes=out);metrics=['early_cost','sustained_cost','A_to_B','B_to_A','D_A','D_B','own_A','cross_A','own_B','cross_B','ce','gap','familiar_ce','accuracy','strong_familiar','strict_all','early_phenotype','sustained','Mixed_both_ce','Mixed_both_accuracy','Mixed_both_success','Mixed_average_ce','NoContext_ce','headroom'];tables['summary']=summarize(out,['arm','dose'],metrics,seeds);raw=[];ci=[]
 for arm,g in out.groupby('arm'):
  for s in sorted(g.dose.unique()):
   if s==2:continue
   a=g[g.dose.eq(s)].set_index('seed');b=g[g.dose.eq(2)].set_index('seed')
   for m in metrics:
    v=(a[m].astype(float)-b[m].astype(float)).reindex(seeds)
    raw.extend(dict(seed=i,arm=arm,dose=s,metric=m,effect=x) for i,x in v.items());ci.append(dict(arm=arm,dose=s,metric=m,**interval(v.to_numpy(float),seeds)))
 tables['paired_seeds']=pd.DataFrame(raw);tables['paired_intervals']=pd.DataFrame(ci);mr=[]
 for (arm,bin,stratum),g in matched.groupby(['arm','performance_range','stratum']):
  p=g.pivot(index='seed',columns='dose',values='specificity').reindex(seeds)
  for s in p:
   if s==2:continue
   v=p[s]-p[2];mr.append(dict(arm=arm,dose=s,performance_range=bin,stratum=stratum,**interval(v.to_numpy(float),seeds)))
 tables['matched_pairs']=pd.DataFrame(mr)
 for label,frame,groups,time,ms in [('training',t,['arm','dose'],'update',['ce_before']),('evaluations',f,['arm','dose','cue'],'update',['ce','gap']),('switch',sw,['arm','dose','direction'],'boundary',['raw_cost','adjusted_cost'])]:tables[label+'_intervals']=time_summary(frame,groups,time,ms,seeds)
 for n,v in tables.items():write_frame(OUT/f'tables/{stage}_{n}.csv',v)
 dump(OUT/f'audit/{stage}_analysis.json',dict(passed=True,cells=len(out),seeds=seeds,no_meta_updates=True));append('## '+stage+' analysis complete\nAll frozen-weight directional costs/accuracies/seven position contributions,firstthreecycle balance,persistence/heads, signed own/cross, original late gates, missing bins, acquisition/failure and sixcue outcomes retained. Whole planned paired seeds only; no transition/position resampling.')
if __name__=='__main__':main()
