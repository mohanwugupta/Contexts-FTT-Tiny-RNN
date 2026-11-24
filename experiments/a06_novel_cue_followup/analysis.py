"""Saved arrays → all cues/positions, unchanged learning rules and seed statistics."""
from .protocol import *
from .execution import original_cache
from experiments.joint_phenotype_hypothesis_screen.analysis import seed_summary
from experiments.recurrent_learning_transfer.measures import boundaries

def static(z,bias):return .25*(np.asarray(z)-np.asarray(bias))+np.asarray(bias)
def rows(z,seed,condition,n,c,status,path):
 finite=z is not None and np.isfinite(z).all() and status=='completed';s=core.score(z) if finite else None;positions=[]
 for t in range(7):
  positions.append(dict(seed=seed,cohort='reused_pilot' if seed<2 else 'new_exploratory',condition=condition,update=n,cue=c,position=t+1,token=core.TARGET_PATH[t],target=int(core.TARGET[t]),predicted_class=int(np.argmax(z[t])) if finite else np.nan,correct=bool(s['accuracy'][t]) if finite else np.nan,ce=float(s['ce'][t]) if finite else np.nan,target_probability=float(s['probabilities'][t,core.TARGET[t]]) if finite else np.nan,margin=float(s['margin'][t]) if finite else np.nan,**{f'logit_{j}':float(z[t,j]) if finite else np.nan for j in range(4)},**{f'probability_{j}':float(s['probabilities'][t,j]) if finite else np.nan for j in range(4)},status=status,source=path))
 ce=float(s['ce'].mean()) if finite else np.nan;correct=int(s['accuracy'].sum()) if finite else np.nan
 return dict(seed=seed,cohort='reused_pilot' if seed<2 else 'new_exploratory',condition=condition,update=n,cue=c,ce=ce,correct_count=correct,accuracy=correct/7,margin=float(s['margin'].mean()) if finite else np.nan,target_probability=float(np.mean([r['target_probability'] for r in positions])) if finite else np.nan,useful=bool(ce<=.5 and correct>=6) if finite else np.nan,strict=bool(ce<=.1 and correct==7) if finite else np.nan,above_chance=ce>np.log(4) if finite else np.nan,status=status),positions
def distribution(g):
 valid=g[g.ce.notna()];x=valid.similarity_C if 'similarity_C' in valid else pd.Series(dtype=float)
 def corr(y):return float(x.rank().corr(valid[y].rank())) if len(valid)>1 and x.nunique()>1 and valid[y].nunique()>1 else np.nan
 return dict(n_cues=len(valid),n_planned=len(g),mean_ce=valid.ce.mean(),median_ce=valid.ce.median(),mean_accuracy=valid.accuracy.mean(),best_accuracy=valid.accuracy.max(),median_accuracy=valid.accuracy.median(),worst_accuracy=valid.accuracy.min(),fraction_useful=valid.useful.astype(float).mean(),similarity_C_spearman_ce=corr('ce'),similarity_C_spearman_accuracy=corr('accuracy'))

def main(destination=None):
 dest=Path(destination) if destination else OUT
 configure();metrics=[];positions=[];practices=[];bank=json.loads((OUT/'registry/cue_vectors.json').read_text());oldstatic={s:np.load(SCREEN/f'runs/A06_static_s{s}/observations.npz') for s in [0,1]};oldtimes=[0,10,20,30,40,50,60,100,170,330];static_checks=[]
 for seed in SEEDS:
  A00=original_cache(seed,'A00');originals={'A00':A00,'A06':original_cache(seed,'A06')};maps={c:{int(k):i for i,k in enumerate(arr['updates'])} for c,arr in originals.items()}
  for c,arr in originals.items():
   source_path=load_sources()['controls'][str(seed)]['positions'] if c=='A00' else (str((SCREEN/f'runs/A06_s{seed}/observations.npz').relative_to(ROOT)) if seed<2 else f'results/a06_novel_cue_followup/v1/runs/A06_s{seed}/original.npz')
   for n in range(331):
    for j,cue_ in enumerate(PROBES):
     i=maps[c].get(n);z=None if i is None else arr['logits'][i,j];status='missing_archived_state' if i is None else str(arr['status'][i,j]);m,p=rows(z,seed,c,n,cue_,status,source_path);metrics.append(m);positions.extend(p)
   path=ROOT/load_sources()['controls'][str(seed)]['training'] if c=='A00' else (SCREEN/f'runs/A06_s{seed}/practice.csv' if seed<2 else OUT/f'runs/A06_s{seed}/practice.csv');g=pd.read_csv(path,float_precision='round_trip');g=g[g['update'].le(330)].copy();g['condition']=c;g['seed']=seed;practices.append(g)
  for i,n in enumerate(A00['updates']):
   for j,cue_ in enumerate(PROBES):
    z=static(A00['logits'][i,j],A00['output_bias'][i]);path=f'results/a06_novel_cue_followup/v1/original_cache/A00_s{seed}.npz + matching-bias quarter view'
    if seed<2 and n in oldtimes:
     oi=oldtimes.index(n);np.testing.assert_array_equal(z,oldstatic[seed]['logits'][oi,j]);z=oldstatic[seed]['logits'][oi,j];path=str((SCREEN/f'runs/A06_static_s{seed}/observations.npz').relative_to(ROOT));static_checks.append(dict(seed=seed,update=int(n),cue=cue_,logits_exact=True))
    m,p=rows(z,seed,'A00-static',int(n),cue_,'completed',path);metrics.append(m);positions.extend(p)
  for n in NOVEL_TIMES:
   for c in ['A00','A06']:
    path=OUT/f'novel_cache/{c}_s{seed}/update{n:03}.npz';arr=dict(np.load(path)) if path.exists() else None
    for j,reg in enumerate(bank):
     z=arr['logits'][j] if arr is not None else None;status=str(arr['status'][j]) if arr is not None else 'missing_state';m,p=rows(z,seed,c,n,reg['cue'],status,str(path.relative_to(ROOT)));metrics.append(m);positions.extend(p)
     if c=='A00':
      zs=static(z,arr['output_bias']) if arr is not None else None;m,p=rows(zs,seed,'A00-static',n,reg['cue'],status,str(path.relative_to(ROOT))+' + matching-bias quarter view');metrics.append(m);positions.extend(p)
 f=pd.DataFrame(metrics);p=pd.DataFrame(positions)
 familiar=f[f.cue.isin(['A','B'])].groupby(['seed','condition','update']).agg(familiar_ce=('ce','mean'),n_familiar=('ce','count'),strong_familiar=('useful',lambda x:bool(x.notna().all() and x.astype(bool).all()))).reset_index();familiar.loc[familiar.n_familiar.ne(2),['familiar_ce','strong_familiar']]=np.nan;f=f.merge(familiar,on=['seed','condition','update'],validate='many_to_one');f['gap']=f.ce-f.familiar_ce;f['historical_strict_gap']=f.strict.eq(True)&f.gap.le(.1);f.loc[f.ce.isna(),'historical_strict_gap']=np.nan
 registry=pd.read_csv(OUT/'cue_registry.csv',float_precision='round_trip');f=f.merge(registry,on='cue',how='left',validate='many_to_one');f.loc[f.cue.isin(PROBES),'kind']='original';f.loc[f.cue.eq('C'),'similarity_C']=1.;f.loc[f.cue.isin(['A','B']),'similarity_C']=0.
 original=f[f.cue.isin(PROBES)].copy();novel=f[f['update'].isin(NOVEL_TIMES)].copy();learning=[];switches=[];reference=json.loads((SCREEN/'reference_thresholds.json').read_text())
 for (seed,c),g in original[original.condition.ne('A00-static')].groupby(['seed','condition']):
  valid=g[g.status.eq('completed')];r=seed_summary(valid.rename(columns={'condition':'candidate'}),c,int(seed),'reused_pilot' if seed<2 else 'new_exploratory',reference);r['condition']=c;r['screen_joint']=r['joint'];r['original_early_joint_raw']=bool(r['original_early_phenotype'] and r['C_useful'] and r['Mixed_useful']);r['original_early_joint_not_interference_dominated']=bool(r['original_early_joint_raw'] and r['net_A']>0 and r['net_B']>0);learning.append(r)
  sw=boundaries(g[g.cue.isin(['A','B'])].assign(architecture=c,seed=seed));sw['condition']=c
  observed_states=set(valid['update'])
  for ix,row in sw.iterrows():
   k=int(row.boundary)
   if k not in observed_states:sw.loc[ix,'reacquisition_status']='missing_frozen_boundary_state';sw.loc[ix,'updates_to_gain_0_05']=np.nan
   elif any(j not in observed_states for j in range(k+1,min(k+10,330)+1)):sw.loc[ix,'reacquisition_status']='incomplete_next_block_states';sw.loc[ix,'updates_to_gain_0_05']=np.nan
  switches.append(sw)
 learning=pd.DataFrame(learning);sw=pd.concat(switches,ignore_index=True);observed=novel[novel.kind.ne('original')];dist=[]
 for (seed,c,n),g in observed.groupby(['seed','condition','update']):
  summary=distribution(g);known=novel[novel.seed.eq(seed)&novel.condition.eq(c)&novel['update'].eq(n)].set_index('cue');cc=known.loc['C'];nn=known.loc['NoContext'];summary.update(C_ce=cc.ce,C_accuracy=cc.accuracy,NoContext_ce=nn.ce,NoContext_accuracy=nn.accuracy,C_minus_median_novel_ce=cc.ce-summary['median_ce'],C_minus_median_novel_accuracy=cc.accuracy-summary['median_accuracy'],fraction_novel_ce_lower_than_C=float((g.ce<cc.ce).mean()),fraction_novel_accuracy_higher_than_C=float((g.accuracy>cc.accuracy).mean()),fraction_novel_accuracy_below_NoContext=float((g.accuracy<nn.accuracy).mean()),fraction_novel_ce_above_NoContext=float((g.ce>nn.ce).mean()),fraction_random_useful=g[g.kind.eq('random_sphere')].useful.astype(float).mean(),fraction_C_arc_useful=g[g.kind.eq('C_arc')].useful.astype(float).mean());dist.append(dict(seed=seed,condition=c,update=n,**summary))
 dist=pd.DataFrame(dist);paired=[]
 def add_pairs(frame,groups,fields,first='A06',second='A00'):
  for key,g in frame.groupby(groups,dropna=False):
   key=key if isinstance(key,tuple) else (key,);a=g[g.condition.eq(first)].set_index('seed');b=g[g.condition.eq(second)].set_index('seed')
   for field in fields:
    for seed in SEEDS:
     av=float(a.loc[seed,field]) if seed in a.index else np.nan;bv=float(b.loc[seed,field]) if seed in b.index else np.nan;paired.append(dict(**dict(zip(groups,key)),contrast=first+'-minus-'+second,metric=field,seed=seed,first=av,second=bv,effect=av-bv))
 for second in ['A00','A00-static']:
  add_pairs(original[original['update'].eq(330)],['cue'],['ce','accuracy','gap','useful','margin'],second=second)
  add_pairs(dist[dist['update'].isin(NOVEL_TIMES)].assign(probe_group='novel_within_seed_mean'),['probe_group','update'],['mean_ce','mean_accuracy','fraction_useful','median_ce','worst_accuracy','similarity_C_spearman_ce'],second=second)
 for field in ['D_A','D_B','Q_A','Q_B','E','E_normalized','A_to_B','B_to_A','sustained_cost','original_early_phenotype','original_early_joint_raw','screen_joint','net_A','net_B']:
  add_pairs(learning.assign(probe_group='learning'),['probe_group'],[field])
 pairs=pd.DataFrame(paired);stats=[]
 for key,g in pairs.groupby(['contrast','metric','cue','probe_group','update'],dropna=False):
  x=g.set_index('seed').effect.reindex(SEEDS).to_numpy(float);r=paired_interval(x);stats.append(dict(zip(['contrast','metric','cue','probe_group','update'],key),**r))
 csv(dest/'original_cue_results.csv',original);csv(dest/'unfamiliar_cue_results.csv',novel);csv(dest/'position_predictions.csv',p);csv(dest/'static_readout_controls.csv',f[f.condition.eq('A00-static')]);csv(dest/'learning_measures.csv',learning);csv(dest/'switch_costs.csv',sw);csv(dest/'practice.csv.gz',pd.concat(practices,ignore_index=True));csv(dest/'cue_distribution_by_seed.csv',dist);csv(dest/'paired_seed_results.csv',pairs);csv(dest/'paired_summary.csv',stats);csv(dest/'registry/static_reuse_checks.csv',static_checks)
 # Historical observations/classifications verified without rerunning optimization.
 old=pd.read_csv(SCREEN/'summary.csv',float_precision='round_trip');pilot=learning[learning.seed.isin([0,1])]
 for row in pilot.itertuples():
  rr=old[old.seed.eq(row.seed)&old.candidate.eq(row.condition)].iloc[0]
  for field in ['D_A','D_B','Q_A','Q_B','E','E_normalized','C_ce','NoContext_ce']:np.testing.assert_allclose(getattr(row,field),rr[field],atol=1e-12,rtol=1e-12)
  assert row.joint==rr.joint and row.original_early_phenotype==rr.original_early_phenotype
 put(dest/'audit/analysis_complete.json',dict(passed=True,original_rows=len(original),unfamiliar_experiment_rows=len(novel),position_rows=len(p),missing_original_cells=int(original.ce.isna().sum()),all6_seeds=True,bootstrap_draws=10000,bootstrap_seed=20260925,unit='trained seed, after within-seed cue summaries',old_pilot_measures_unchanged=True,optimizer_steps=0))
 return original,novel,learning,dist,pairs,pd.DataFrame(stats)
