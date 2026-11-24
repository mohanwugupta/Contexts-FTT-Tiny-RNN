"""Saved-table reanalysis of full factorial; no network evaluations or updates."""
from .common import *
from .measures import extended_boundaries
from experiments.structured_overlap_meta.analysis import gaps
from experiments.centered_context_linear.analysis import block_table
from experiments.recurrent_learning_transfer.measures import specificity,split_summary
from experiments.high_scale_constrained_meta.analysis import first_blocks
from experiments.linear_scale_sweep.analysis import transition_summary
from experiments.meta_learning_scaling.analysis import matched_acquisition
from experiments.meta_objective_balance.statistics import interval,summarize,time_summary
KEY=['seed','scale','phase','dose'];PROBES=[0,100,170,330,660,990]

def read_dir(d,n):return pd.read_csv(d/n,float_precision='round_trip',low_memory=False)
def analyze_one(r):
 d=OUT/r['directory'];s=int(r['seed']);scale=int(r['scale']);phase=r['phase'];dose=int(r['dose']);meta=dict(seed=s,arm=f'{scale}-{phase}',dose=dose)
 f,acq=gaps(read_dir(d,'evaluations.csv').assign(**meta));t=read_dir(d,'training.csv').assign(**meta);g=read_dir(d,'geometry.csv').assign(**meta);diag=read_dir(d,'diagnostics.csv').assign(**meta);pos=read_dir(d,'position_predictions.csv.gz');b=block_table(f);b['dose']=dose;lf=f[f.cue.isin(['A','B'])].copy();lf['architecture']=lf.arm;sw=extended_boundaries(lf).rename(columns={'architecture':'arm'});sw['dose']=dose
 early=first_blocks(b,sw).iloc[0].drop(['seed','arm']).to_dict();tr=transition_summary(sw).iloc[0].drop(['seed','arm','dose']).to_dict();first=acq.iloc[0].drop(['seed','arm','dose']).to_dict();wide=[]
 for k,z in diag.groupby('update'):
  v={(x.train_context,x.evaluation_context):x.improvement for x in z.itertuples()};q=dict(**meta,update=k,delta_A_A=v['A','A'],delta_B_A=v['A','B'],delta_B_B=v['B','B'],delta_A_B=v['B','A']);wide.append(dict(**q,**specificity(q)))
 wide=pd.DataFrame(wide);ww=wide.rename(columns={'update':'checkpoint','arm':'architecture'});late=split_summary(lf[lf['update'].le(330)],ww).rename(columns={'architecture':'arm','endpoint':'update'});late['dose']=dose;matched_frames=[]
 for horizon in ([330,990] if int(f['update'].max())==990 else [330]):
  mb=b[b.end.le(horizon)];matched_frames.append(matched_acquisition(mb.assign(condition=mb.arm)).rename(columns={'condition':'arm'}).assign(horizon=horizon))
 matched=pd.concat(matched_frames,ignore_index=True);ends=[];cycles=[];pc=[]
 for k in PROBES:
  q=f[f['update'].eq(k)].set_index('cue')
  if q.empty:continue
  row=dict(**meta,update=k,**early,**tr,**first,familiar_ce=q.loc['C','familiar_ce'],strong_familiar=q.loc['C','strong_familiar'],gap=q.loc['C','gap'],strict_C=q.loc['C','strict_all'])
  for cue,label in [('A','A'),('B','B'),('C','C'),('Mixed-average','Mixed_average'),('Mixed-both','Mixed_both'),('NoContext','NoContext')]:
   for metric in ['ce','accuracy','gap','success']:row[label+'_'+metric]=q.loc[cue,metric]
  start=f[f['update'].eq(0)].set_index('cue');row.update(initial_ce_A=start.loc['A','ce'],initial_ce_B=start.loc['B','ce'],initial_accuracy_A=start.loc['A','accuracy'],initial_accuracy_B=start.loc['B','accuracy'],nonfloor_first_both=bool(not early['floor_A'] and not early['floor_B']),positive_D_A=early['D_A']>0,positive_D_B=early['D_B']>0,sustained_positive=tr['sustained_cost']>0,original_gate_available=k in [170,330],functional_split=bool(late[late['update'].eq(k)].functional_split.iloc[0]) if k in [170,330] else np.nan)
  ends.append(row)
 for end in range(20,int(f['update'].max())+1,20):
  a=sw[sw.boundary.eq(end-10)].iloc[0];bb=sw[sw.boundary.eq(end)].iloc[0];q=f[f['update'].isin([end-10,end])&f.cue.isin(['A','B'])];cycles.append(dict(**meta,end=end,adjusted_cost=(a.adjusted_cost+bb.adjusted_cost)/2,raw_cost=(a.raw_cost+bb.raw_cost)/2,A_to_B=a.adjusted_cost,B_to_A=bb.adjusted_cost,headroom=int(a.incoming_ce>.05)+int(bb.incoming_ce>.05),both_boundaries_familiar=bool(q.strong_familiar.eq(1).all()),both_boundaries_floor=bool(q.ce.le(.05).all()),after_first_acquisition=bool(np.isfinite(first['first_familiar']) and end-10>=first['first_familiar'])))
 for row in sw.itertuples():
  def pp(k,c):return pos[pos['update'].eq(k)&pos.cue.eq(c)].sort_values('position')
  a=pp(row.boundary,row.outgoing_context);bb=pp(row.boundary,row.incoming_context);a0=pp(0,row.outgoing_context);b0=pp(0,row.incoming_context)
  for j in range(7):pc.append(dict(**meta,boundary=row.boundary,position=j+1,outgoing_context=row.outgoing_context,incoming_context=row.incoming_context,incoming_ce=bb.ce.iloc[j],outgoing_ce=a.ce.iloc[j],raw_cost=bb.ce.iloc[j]-a.ce.iloc[j],adjusted_cost=bb.ce.iloc[j]-a.ce.iloc[j]-(b0.ce.iloc[j]-a0.ce.iloc[j]),incoming_accuracy=float(bb.correct.iloc[j]),outgoing_accuracy=float(a.correct.iloc[j]),incoming_margin=bb.margin.iloc[j],outgoing_margin=a.margin.iloc[j]))
 pc=pd.DataFrame(pc);accuracy=pc.groupby(['seed','arm','dose','boundary'])[['incoming_accuracy','outgoing_accuracy']].mean().reset_index();sw=sw.merge(accuracy,on=['seed','arm','dose','boundary'],validate='one_to_one');sw['accuracy_cost']=sw.outgoing_accuracy-sw.incoming_accuracy
 # Weight-only extraction fills registered RMS fields in inherited prefixes.
 from experiments.high_scale_constrained_meta.audit import metrics as weight_metrics
 from experiments.high_scale_constrained_meta.runtime import scale_metrics
 saved=torch.load(ROOT/r['path'],weights_only=False)['states'];weight_rows=[]
 for st in saved:
  if st['update'] in PROBES:
   q=scale_metrics(source_model(st['model']));weight_rows.append(dict(**meta,update=st['update'],**weight_metrics(st['model']),centered_contrast=float(q['centered_contrast'].mean())))
 margins=pos[pos['update'].isin(PROBES)].groupby(['update','cue']).margin.mean().reset_index().assign(**meta)
 tables=dict(target_checkpoint_weights=pd.DataFrame(weight_rows),evaluations=f,training=t,geometry=g,diagnostics=diag,learning_effects=wide,blocks=b,switch_costs=sw,position_switch_costs=pc,balanced_cycles=pd.DataFrame(cycles),functional_splitting=late,matched_acquisition=matched,outcomes=pd.DataFrame(ends),position_margin_summary=margins)
 for name,v in tables.items():
  v=v.drop(columns=['arm'],errors='ignore');v['scale']=scale;v['phase']=phase;v['dose']=dose;tables[name]=v
 return tables

def paired_effects(out,metrics):
 raw=[];cis=[];keys=['seed','update'];seeds=list(range(6))
 for scale,g in out.groupby('scale'):
  conditions=g[['phase','dose']].drop_duplicates().itertuples(index=False,name=None)
  for phase,dose in conditions:
   refs=[('Fresh',0)] if phase=='R' else ([('R',1920)] if phase in ['T','B','O1','O2'] else [])
   if phase in ['T','B','O1','O2']:refs +=[(other,dose) for other in ['T','B','O1','O2'] if other<phase]
   a=g[g.phase.eq(phase)&g.dose.eq(dose)]
   for ref,rdose in refs:
    b=g[g.phase.eq(ref)&g.dose.eq(rdose)];paired=a.merge(b,on=keys,suffixes=('_a','_b'));contrast=f'{phase}{dose}-{ref}{rdose}'
    for update,z in paired.groupby('update'):
     for metric in metrics:
      v=(z.set_index('seed')[metric+'_a'].astype(float)-z.set_index('seed')[metric+'_b'].astype(float)).reindex(seeds);raw.extend(dict(seed=s,scale=scale,update=update,phase=phase,dose=dose,reference=ref,reference_dose=rdose,contrast=contrast,metric=metric,effect=x) for s,x in v.items());cis.append(dict(scale=scale,update=update,phase=phase,dose=dose,reference=ref,reference_dose=rdose,contrast=contrast,metric=metric,**interval(v.to_numpy(float),seeds)))
 return pd.DataFrame(raw),pd.DataFrame(cis)

def main():
 assert json.loads((OUT/'audit/diagnostics_complete.json').read_text())['passed'];tables={};reg=pd.concat([read_dir(OUT/'registry',x+'.csv') for x in ['fresh','reptile_targets','fomaml_targets']],ignore_index=True);assert len(reg)==288
 for i,r in enumerate(reg.to_dict('records'),1):
  for n,v in analyze_one(r).items():tables.setdefault(n,[]).append(v)
  if i%18==0:print('Table-only analysis',i,'/288',flush=True)
 tables={n:pd.concat(v,ignore_index=True) for n,v in tables.items()};out=tables['outcomes'];metrics=['D_A','D_B','own_A','cross_A','own_B','cross_B','early_cost','sustained_cost','A_to_B','B_to_A','familiar_ce','strong_familiar','C_ce','C_accuracy','gap','strict_C','Mixed_average_ce','Mixed_both_ce','Mixed_both_success','NoContext_ce','first_familiar','functional_split'];tables['outcome_summary']=summarize(out,['scale','phase','dose','update'],metrics,list(range(6)));tables['paired_seed_effects'],tables['paired_intervals']=paired_effects(out,metrics)
 # Scale comparisons use the same Gaussian direction and identical procedure/dose/target endpoint.
 sr=[];si=[]
 for (phase,dose,update),z in out.groupby(['phase','dose','update']):
  for metric in metrics:
   p=z.pivot(index='seed',columns='scale',values=metric).reindex(range(6))
   for a,b in [(5,4),(6,4),(6,5)]:
    v=p[a].astype(float)-p[b].astype(float);sr.extend(dict(seed=s,phase=phase,dose=dose,update=update,contrast=f'{a}-{b}',metric=metric,effect=x) for s,x in v.items());si.append(dict(phase=phase,dose=dose,update=update,contrast=f'{a}-{b}',metric=metric,**interval(v.to_numpy(float),list(range(6)))))
 tables['scale_paired_seeds']=pd.DataFrame(sr);tables['scale_paired_intervals']=pd.DataFrame(si)
 held=read_dir(OUT/'tables','conflict_seed.csv');tables['conflict_summary']=summarize(held,['scale','phase','dose'],['conflict_accuracy','critical_ce','ce','context_benefit','conflict_context_benefit','appropriate_context_change','own_gain','cross_gain','specificity'],list(range(6)))
 tables['conflict_paired_seeds'],tables['conflict_paired_intervals']=paired_effects(held.assign(update=-1),['conflict_accuracy','critical_ce','ce','context_benefit','conflict_context_benefit','appropriate_context_change','own_gain','cross_gain','specificity'])
 tables['joint_components']=out.merge(held,on=KEY,how='left',suffixes=('','_conflict'));tables['matched_summary']=summarize(tables['matched_acquisition'],['scale','phase','dose','horizon','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain'],list(range(6)))
 # Match acquisition exposure as well as fixed loss bin/stratum.
 matched_raw=[];matched_ci=[]
 for key,z in tables['matched_acquisition'].groupby(['horizon','performance_range','stratum'],dropna=False):
  raw,ci=paired_effects(z.assign(update=key[0]),['specificity','practiced_gain','unpracticed_gain','start_A_CE','start_B_CE'])
  for frame,dest in [(raw,matched_raw),(ci,matched_ci)]:
   frame['horizon']=key[0];frame['performance_range']=key[1];frame['stratum']=key[2];dest.append(frame)
 tables['matched_paired_seeds']=pd.concat(matched_raw,ignore_index=True);tables['matched_paired_intervals']=pd.concat(matched_ci,ignore_index=True)
 # Registered meta state geometry, not a mediation analysis.
 rows=[]
 from experiments.high_scale_constrained_meta.audit import metrics as weight_metrics
 from experiments.high_scale_constrained_meta.runtime import scale_metrics
 for seed in range(6):
  for scale in [4,5,6]:
   for phase,doses in [('Fresh',[0])]+[(a,[240,960,1920]) for a in ['R','T','B','O1','O2']]:
    for dose in doses:
     state,src=initial(seed,'P2','I',f'{scale}-{phase}',dose);q=scale_metrics(source_model(state));wi=state['recurrent.weight_ih_l0'].double().numpy();rows.append(dict(seed=seed,scale=scale,phase=phase,dose=dose,source=str(src.relative_to(ROOT)),source_sha=sha(src),**weight_metrics(state),centered_contrast=float(q['centered_contrast'].mean()),AB_column_difference=float(np.linalg.norm(wi[:,4]-wi[:,5]))))
 tables['meta_geometry']=pd.DataFrame(rows)
 for label,frame,groups,tt,ms in [('fresh_training',tables['training'].query("phase=='Fresh'"),['scale'],'update',['ce_before']),('fresh_evaluations',tables['evaluations'].query("phase=='Fresh'"),['scale','cue'],'update',['ce','accuracy']),('fresh_switches',tables['switch_costs'].query("phase=='Fresh'"),['scale','direction'],'boundary',['raw_cost','adjusted_cost']),('fresh_cycles',tables['balanced_cycles'].query("phase=='Fresh'"),['scale'],'end',['adjusted_cost','raw_cost'])]:tables[label+'_intervals']=time_summary(frame,groups,tt,ms,list(range(6)))
 for n,v in tables.items():write_frame(OUT/f'tables/{n}.csv',v)
 dump(OUT/'audit/analysis_complete.json',dict(passed=True,physical_targets=288,seeds=6,bootstrap=10000,signed_values_retained=True,original_gate_endpoints=[170,330],extended_gate='not_redefined',training=False));append('## Analysis complete\nAll288 physical target trajectories, registered endpoints, direction/position/accuracy costs, every block and unchanged late gates retained. Pairing uses common seed at identical target update; FOMAML+0 aliases R1920. Conditional/missing bins and all negative values preserved. No joint score, causal mediation or post hoc ranking.');print('Analysis PASSED',flush=True)
if __name__=='__main__':main()
