"""All-seed paired descriptive inference; original functional gate unchanged."""
from .core import *
from experiments.centered_context_linear.analysis import block_table,time_intervals
from experiments.recurrent_learning_transfer.measures import specificity,boundaries,split_summary

def validate_grid(f,conditions=CONDS,updates=range(331),cues=list(CUES)):
 keys=['seed','condition','update','cue'];expected=pd.MultiIndex.from_product([range(20),conditions,updates,cues],names=keys);actual=pd.MultiIndex.from_frame(f[keys])
 if actual.has_duplicates or len(actual)!=len(expected) or set(actual)!=set(expected):raise ValueError('Incomplete or duplicated all-seed grid')

def seed_interval(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,median=np.nan,ci_low=np.nan,ci_high=np.nan,n=0)
 idx=np.random.default_rng(20260925).integers(n,size=(10000,n));lo,hi=np.quantile(x[idx].mean(1),[.025,.975]);return dict(mean=float(x.mean()),median=float(np.median(x)),ci_low=float(lo),ci_high=float(hi),n=n)

def paired_contrasts(f):
 rows=[]
 for (k,cue),g in f.groupby(['update','cue']):
  for metric in ['ce','accuracy','gap','success','strong_familiar']:
   if metric not in g:continue
   p=g.pivot(index='seed',columns='condition',values=metric).astype(float).reindex(range(20))
   for a,b in [('P2','P1'),('P1','P0'),('P2','P0')]:
    effects={}
    for org in ORG:
     if a+'-'+org not in p or b+'-'+org not in p:continue
     effects[org]=p[a+'-'+org]-p[b+'-'+org]
     rows.extend(dict(seed=s,update=k,cue=cue,metric=metric,organization=org,contrast=a+'-'+b,effect=v) for s,v in effects[org].items())
    if len(effects)==2:rows.extend(dict(seed=s,update=k,cue=cue,metric=metric,organization='C-minus-I',contrast=a+'-'+b,effect=v) for s,v in (effects['C']-effects['I']).items())
 return pd.DataFrame(rows)

def endpoint_gate(learning,wide):
 # Exact source endpoint gate. Secondary trajectory criterion explicitly dropped:
 # different registered diagnostic times cannot reproduce it identically.
 return split_summary(learning,wide).drop(columns=['trajectory_specific_learning','n_prior_direct_positive'])

def main():
 assert (OUT/'audit/target_complete.json').exists()
 read=lambda n:pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip')
 f=read('evaluations');validate_grid(f);g=read('geometry');t=read('training');dg=read('diagnostics');b=block_table(f.rename(columns={'condition':'arm'})).rename(columns={'arm':'condition'})
 learning=f[f.cue.isin(['A','B'])].rename(columns={'condition':'architecture'});s=boundaries(learning).rename(columns={'architecture':'condition'});wr=[]
 for (seed,c,k),gg in dg.groupby(['seed','condition','update']):
  d={(r.train_context,r.evaluation_context):r.improvement for r in gg.itertuples()};r=dict(seed=seed,architecture=c,checkpoint=k,delta_A_A=d['A','A'],delta_B_A=d['A','B'],delta_B_B=d['B','B'],delta_A_B=d['B','A']);wr.append(dict(**r,**specificity(r)))
 wide=pd.DataFrame(wr);gate=endpoint_gate(learning,wide).rename(columns={'architecture':'condition','endpoint':'update'});wide=wide.rename(columns={'architecture':'condition','checkpoint':'update'})
 q=[]
 for (seed,c,k),gg in f.groupby(['seed','condition','update']):
  d=gg.set_index('cue');q.append(dict(seed=seed,condition=c,update=k,strong_familiar=bool((d.loc[['A','B'],'ce']<=.5).all() and (d.loc[['A','B'],'accuracy']>=6/7).all()),familiar_ce=d.loc[['A','B'],'ce'].mean(),signed_A_minus_B=d.loc['A','ce']-d.loc['B','ce']))
 q=pd.DataFrame(q);q['absolute_AB_gap']=abs(q.signed_A_minus_B);f=f.merge(q,on=['seed','condition','update'],validate='many_to_one');f['gap']=f.ce-f.familiar_ce;f['success']=(f.ce<=.5)&(f.accuracy>=6/7);ep=f[f['update'].isin(PROBES)].copy();summary=[];counts=[]
 for (c,k,cue),gg in ep.groupby(['condition','update','cue']):
  for subgroup in ['all','strong_familiar']:
   x=gg if subgroup=='all' else gg[gg.strong_familiar]
   for metric in ['ce','accuracy','gap','success']:
    summary.append(dict(condition=c,update=k,cue=cue,subgroup=subgroup,metric=metric,**seed_interval(x[metric])))
   counts.append(dict(condition=c,update=k,cue=cue,subgroup=subgroup,n=len(x),strong_familiar=int(x.strong_familiar.sum()),success=int(x.success.sum()),above_chance=int(x.above_chance.sum())))
 joint=[]
 for r in q[q['update'].isin(PROBES)].itertuples():
  x=ep[(ep.seed==r.seed)&(ep.condition==r.condition)&(ep['update']==r.update)].set_index('cue');gt=gate[(gate.seed==r.seed)&(gate.condition==r.condition)&(gate['update']==r.update)];split=bool(gt.iloc[0].functional_split) if len(gt) else None
  joint.append(dict(seed=r.seed,condition=r.condition,update=r.update,strong_familiar=r.strong_familiar,C_success=bool(x.loc['C','success']),Mixed_average_success=bool(x.loc['Mixed-average','success']),Mixed_both_success=bool(x.loc['Mixed-both','success']),functional_split=split,functional_gate_status='exact_original_endpoint' if split is not None else 'not_registered_at_this_state',joint_behavioral=(None if split is None else bool(r.strong_familiar and x.loc['C','success'] and x.loc['Mixed-both','success'] and split))))
 paired=paired_contrasts(ep);ps=[]
 for keys,gg in paired.groupby(['update','cue','metric','organization','contrast']):ps.append(dict(zip(['update','cue','metric','organization','contrast'],keys),**seed_interval(gg.effect)))
 # Scatter descriptors use one seed, one condition; early learning summary is prespecified first100.
 early=b[b.end<=100].groupby(['seed','condition']).agg(early_specificity=('specificity','mean'),early_own_gain=('practiced_gain','mean'),early_cross_gain=('unpracticed_gain','mean'),early_floor_fraction=('floor_at_start','mean')).reset_index();sc=ep[ep.cue=='C'].merge(g,on=['seed','condition','update','organization','prior'],suffixes=('','_geometry')).merge(early,on=['seed','condition']).merge(wide,on=['seed','condition','update'],how='left');sw=s[s.boundary<=100].groupby(['seed','condition']).agg(early_switch_cost=('raw_cost','mean'),early_adjusted_switch_cost=('adjusted_cost','mean')).reset_index();sc=sc.merge(sw,on=['seed','condition'])
 frames={'block_transfer':b,'switch_costs':s,'learning_effects':wide,'functional_splitting':gate,'performance_gaps':q,'endpoints':ep,'endpoint_summary':pd.DataFrame(summary),'endpoint_counts':pd.DataFrame(counts),'joint_outcomes':pd.DataFrame(joint),'paired_seed_effects':paired,'paired_intervals':pd.DataFrame(ps),'transfer_associations':sc}
 frames['evaluation_intervals']=time_intervals(f,['condition','cue'],'update',['ce','accuracy'])
 frames['training_intervals']=time_intervals(t,['condition'],'update',['ce_before'])
 frames['geometry_intervals']=time_intervals(g,['condition'],'update',['hidden_separation','centered_logit_separation','recurrent_spectral_norm','g_norm','d_norm','C_norm','u_norm','v_norm','C_response_norm','K_centered_norm','baseline_ce','C_ce','baseline_margin','C_margin'])
 frames['block_intervals']=time_intervals(b,['condition','practiced'],'end',['practiced_gain','unpracticed_gain','specificity'])
 frames['switch_intervals']=time_intervals(s,['condition','direction'],'boundary',['raw_cost','adjusted_cost'])
 frames['gap_intervals']=time_intervals(q,['condition'],'update',['signed_A_minus_B','absolute_AB_gap'])
 frames['diagnostic_intervals']=time_intervals(wide,['condition'],'update',['mean_specificity','D_A','D_B','mean_own_improvement','mean_cross_improvement'])
 for name,x in frames.items():
  assert not (OUT/'tables'/f'{name}.csv').exists();x.to_csv(OUT/'tables'/f'{name}.csv',index=False)
 dump(OUT/'audit/analysis_complete.json',dict(bootstrap_draws=10000,bootstrap_seed=20260925,unit='paired whole seed',original_endpoint_gate='identical source at170/330',secondary_trajectory_gate='unassessed; diagnostic_times_differ',floor=.05,hyperparameter_selection=False,all20=True));append('## Analysis complete\nAll120 targetnetworks retained;6probe states,whole-seed10k CIs,means/medians,qualifieddenominators,withinseedpriorcontrasts andcentered-minus-independent differences. Exactendpointgate170/330 retained;secondarytrajectorygate unassessed. Geometrynever entersgate;floorlimitedgains notsharedlearningproof.')
if __name__=='__main__':main()
