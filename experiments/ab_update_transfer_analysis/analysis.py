"""Cached-table extraction and paired inference; no optimizer operations."""
import gzip,pickle
from .common import *
from . import design
from .statistics import interval
META=['analysis_version','trajectory_id','seed','scale','regime','objective','meta_stage','meta_episodes','cohort']
GROUP=['scale','regime','objective','meta_stage','meta_episodes','cohort']
BANDS=[(2,4),(1,2),(.5,1),(.1,.5)]
def parquet(path,frame):
 path=Path(path);frame=pd.DataFrame(frame)
 if path.exists():pd.testing.assert_frame_equal(pd.read_parquet(path),frame,check_exact=True)
 else:
  pending=path.with_suffix('.pending');frame.to_parquet(pending,index=False,compression='zstd',engine='pyarrow');os.replace(pending,path)

def loss_band(x):
 for lo,hi in BANDS:
  if x>=lo and (x<=hi if hi==4 else x<hi):return f'{lo:g}-{hi:g}'
 return 'outside_registered_bands'
def effect(before,after,cross_before,cross_after):
 own=before-after;cross=cross_before-cross_after
 return dict(gain=own,own_gain=own,cross_gain=cross,specificity=own-cross,own_floor=before<=.05,probe_floor=before<=.05,own_deterioration=own<0,cross_deterioration=cross<0)
def realized_intervals(f):
 rows=[]
 for c,g in f.groupby('probe_context'):
  g=g.sort_values('target_update');lookup=g.set_index('target_update').ce
  for lo,hi in [(0,10),(10,20),(20,60),(60,170),(170,330),(330,990)]:
   available=all(k in lookup.index for k in range(lo,hi+1))
   for source in ['A','B']:
    ks=[k for k in range(lo+1,hi+1) if design.cue(k)==source];good=[k for k in ks if k in lookup.index and k-1 in lookup.index]
    rows.append(dict(probe_context=c,interval=f'{lo}-{hi}',interval_start=lo,interval_end=hi,source_context=source,n_A_updates=sum(design.cue(k)=='A' for k in range(lo+1,hi+1)),n_B_updates=sum(design.cue(k)=='B' for k in range(lo+1,hi+1)),n_source_updates=len(good),planned_source_updates=len(ks),coverage='complete' if available else 'missing_history',realized_gain=sum(lookup[k-1]-lookup[k] for k in good) if available else np.nan,starting_ce=lookup.get(lo,np.nan),remaining_ce=lookup.get(hi,np.nan)))
 return pd.DataFrame(rows)
def meta(r):return dict(analysis_version=design.VERSION,trajectory_id=r['trajectory_id'],seed=r['seed'],scale=r['scale'],regime=r['regime'],objective=r['objective'],meta_stage=r['meta_stage'],meta_episodes=r['dose'],cohort=r['cohort'])
def metrics(obs):return dict(ce=float(obs['ce'].mean()),accuracy=float(obs['accuracy'].mean()),margin=float(obs['margin'].mean()),target_probability=float(obs['probabilities'][np.arange(7),core.TARGET].mean()))

def curves(q):
 from .engine import actual_gains
 from .sources import checked
 frames=[];realized=[];provenance=[];manifest=json.loads((SOURCE/'manifest.json').read_text())
 for i,r in enumerate(q.to_dict('records'),1):
  folder=(ROOT/r['path']).parent;e_path=folder/'evaluations.csv';p_path=folder/'position_predictions.csv.gz';t_path=folder/'training.csv';eh=checked(e_path,manifest);ph=checked(p_path,manifest);th=checked(t_path,manifest);path=OUT/f'curve_cache/{r["trajectory_id"]}.pkl.gz'
  if path.exists():
   with gzip.open(path,'rb') as f:b=pickle.load(f)
   assert b['hashes']==[eh,ph,th];f=b['frame']
  else:
   ev=pd.read_csv(e_path,float_precision='round_trip');ev=ev[ev.cue.isin(design.PROBES)].rename(columns={'update':'target_update','cue':'probe_context'});pos=pd.read_csv(p_path,float_precision='round_trip');pos=pos[pos.cue.isin(design.PROBES)].copy();pos['target_probability']=pos[[f'probability_{j}' for j in range(4)]].to_numpy()[np.arange(len(pos)),pos.target.to_numpy(int)];m=pos.groupby(['update','cue']).agg(margin=('margin','mean'),target_probability=('target_probability','mean')).reset_index().rename(columns={'update':'target_update','cue':'probe_context'});f=ev.merge(m,on=['target_update','probe_context'],validate='one_to_one');f['practiced_cue_previous']=[design.cue(k) if k else 'none' for k in f.target_update];f['next_cue']=[design.cue(k+1) for k in f.target_update];f['probe_floor']=f.ce.le(.05);f['source_context']=f.practiced_cue_previous;f['branch_type']='original_history';f['branch_step']=0
   for k,v in meta(r).items():f[k]=v
   path.parent.mkdir(exist_ok=True)
   with gzip.open(path,'xb') as g:pickle.dump(dict(frame=f,hashes=[eh,ph,th]),g,protocol=5)
  pivot=f.pivot(index='target_update',columns='probe_context',values='ce').sort_index();assert np.array_equal(pivot.index,np.arange(r['total']+1));g=actual_gains(pivot.to_numpy(float));assert len(g)==r['total'];realized.append(realized_intervals(f).assign(**meta(r)));frames.append(f);provenance.append(dict(**meta(r),evaluations=str(e_path.relative_to(ROOT)),evaluation_sha=eh,position_predictions=str(p_path.relative_to(ROOT)),position_sha=ph,training=str(t_path.relative_to(ROOT)),training_sha=th))
  if i%36==0:print('Saved histories',i,'/342',flush=True)
 csv(OUT/'registry/native_prediction_sources.csv',provenance);csv(OUT/'tables/native_curve_metrics.csv.gz',pd.concat(frames,ignore_index=True));parquet(OUT/'tables/native_curve_metrics.parquet',pd.concat(frames,ignore_index=True));csv(OUT/'tables/realized_gain_by_source.csv',pd.concat(realized,ignore_index=True));return pd.concat(frames,ignore_index=True)

def motions(pre,post):
 from .adapter import masks
 ms=masks(pre['model']);out={}
 for name,entries in ms.items():out['motion_'+name]=float(np.sqrt(sum(float(((post['model'][k].double()-pre['model'][k].double())[mask]**2).sum()) for k,mask in entries.items())))
 out['policy_motion']=float(np.sqrt(sum(float(((post['policy'][k].double()-v.double())**2).sum()) for k,v in pre['policy'].items())))
 return out

def contribution(base,obs,check,tag,k):
 z=obs['C']['logits']-obs['NoContext']['logits'];center=z-z.mean(-1,keepdims=True);gC=base['C']['ce'].mean()-obs['C']['ce'].mean();gN=base['NoContext']['ce'].mean()-obs['NoContext']['ce'].mean();rows=[]
 for t in range(7):
  rows.append(dict(tag,branch_step=k,position=t+1,target=int(core.TARGET[t]),C_ce=obs['C']['ce'][t],NoContext_ce=obs['NoContext']['ce'][t],C_minus_NoContext_ce=obs['C']['ce'][t]-obs['NoContext']['ce'][t],C_margin=obs['C']['margin'][t],NoContext_margin=obs['NoContext']['margin'][t],margin_difference=obs['C']['margin'][t]-obs['NoContext']['margin'][t],C_gain_minus_NoContext_gain=gC-gN,centered_C_norm=float(np.linalg.norm(center[t])),**{f'raw_C_contribution_{j}':z[t,j] for j in range(4)},**{f'centered_C_contribution_{j}':center[t,j] for j in range(4)},**{f'double_C_operator_{j}':check['C_operator'][t,j] for j in range(4)}))
 return rows

def extract_branches(q):
 metrics_rows=[];effects=[];params=[];cue=[];registries=[];missing=[];timing=[]
 all_receipts={}
 for p in (OUT/'branches').glob('*/complete.json'):
  r=json.loads(p.read_text());key=(r['trajectory_id'],r['target_update'],r['source_context'],r['branch_type']);assert key not in all_receipts;all_receipts[key]=(p,r)
 for row in q.to_dict('records'):
  ns=design.PRIMARY+(design.SECONDARY if row['cohort']=='final' else [])
  for n in ns:
   for source in ['A','B']:
    for kind in ['native']+(['zero_gradient_adam'] if row['cohort']=='final' and n in [10,20] else []):
     key=(row['trajectory_id'],n,source,kind);h=10 if kind=='native' and row['cohort']=='final' and n in design.PRIMARY else 1;tag=dict(meta(row),target_update=n,source_context=source,branch_type=kind)
     if key not in all_receipts:
      failure=[json.loads(p.read_text()) for p in (OUT/'branches').glob('*/failure.json')];rr=next((r for r in failure if (r['trajectory_id'],r['target_update'],r['source_context'],r['branch_type'])==key),{});missing.append(dict(tag,horizon=h,status=rr.get('status','unavailable_not_completed'),reason=rr.get('reason','No accepted complete branch; wall cap or unavailable source')));registries.append(dict(tag,horizon=h,status=missing[-1]['status'],reason=missing[-1]['reason']));continue
     path,receipt=all_receipts[key];assert receipt['horizon']==h and sha(path.parent/'result.pt')==receipt['sha256'] and sha(path.parent/'raw.npz')==receipt['raw_sha256'];b=torch.load(path.parent/'result.pt',weights_only=False);assert b['key']==receipt['key'];registries.append(dict(tag,**{k:v for k,v in receipt.items() if k not in tag}));base=b['baseline']['observations'];own=source;cross='B' if own=='A' else 'A';baseline_model=b['baseline']['state']['model'];checkpoints=[dict(step=0,observations=base,checks=b['baseline']['checks'])]+b['steps']
     for step in checkpoints:
      k=step['step'];obs=step['observations'];gown=float(base[own]['ce'].mean()-obs[own]['ce'].mean());gcross=float(base[cross]['ce'].mean()-obs[cross]['ce'].mean())
      for c in design.PROBES:
       v=metrics(obs[c]);initial=metrics(base[c]);r=dict(tag,branch_step=k,horizon=h,probe_context=c,**v,pre_ce=initial['ce'],pre_margin=initial['margin'],pre_target_probability=initial['target_probability'],gain=initial['ce']-v['ce'],post_ce=v['ce'],probability_gain=v['target_probability']-initial['target_probability'],margin_gain=v['margin']-initial['margin'],own_gain=gown,cross_gain=gcross,specificity=gown-gcross,own_start_ce=float(base[own]['ce'].mean()),cross_start_ce=float(base[cross]['ce'].mean()),own_floor=bool(base[own]['ce'].mean()<=.05),probe_floor=bool(initial['ce']<=.05),own_deterioration=gown<0,cross_deterioration=gcross<0,headroom_band=loss_band(float(base[own]['ce'].mean())),model_hash=step.get('model_hash',digest(baseline_model)),policy_hash=step.get('policy_hash',digest(b['baseline']['state']['policy'])),Adam_hash=step.get('Adam_hash',digest(b['baseline']['state']['optimizer'])),optimizer_counter=n+k if k else n,source_state_hash=receipt['source_state_hash'],reused_optimizer_step=step.get('reused_optimizer_step',False),status='completed');metrics_rows.append(r)
       if k in {1,h}:effects.append(dict(r,post_ce=v['ce'],probability_gain=v['target_probability']-initial['target_probability'],margin_gain=v['margin']-initial['margin']))
      cue.extend(contribution(base,obs,step['checks'],tag,k))
      if k:
       prev=b['baseline']['state'] if k==1 else b['steps'][k-2]['state'];post=step['state'];timing.append(dict(tag,branch_step=k,**{j:step[j] for j in ['update_seconds','probe_seconds','math_seconds','replay','reused_optimizer_step']},pre_model_hash=digest(prev['model']),post_model_hash=digest(post['model']),pre_policy_hash=digest(prev['policy']),post_policy_hash=digest(post['policy']),pre_Adam_hash=digest(prev['optimizer']),post_Adam_hash=digest(post['optimizer']),native_clip_factor_float32=float(torch.clamp(torch.tensor(1.)/(torch.tensor(step['update_info']['clipped_input_norm'],dtype=torch.float32)+1e-6),max=1.)),Adam_counters=json.dumps([float(v['step']) for v in post['optimizer']['state'].values()]),**motions(prev,post),**step['update_info']))
     part=b['parameter']
     if part:
      for variant,obs in part['observations'].items():
       for c in design.PROBES:
        g=float(base[c]['ce'].mean()-obs[c]['ce'].mean());params.append(dict(tag,branch_step=1,probe_context=c,variant=variant,gain=g,pre_ce=float(base[c]['ce'].mean()),post_ce=float(obs[c]['ce'].mean()),pre_margin=float(base[c]['margin'].mean()),post_margin=float(obs[c]['margin'].mean()),structural_check=c in ['C','NoContext'] and variant in ['cue_only','practiced_cue_only','other_cue_only'],**{f'motion_{k}':v for k,v in part['motion'].items()}))
      for c in design.PROBES:
       gain=lambda variant:float(base[c]['ce'].mean()-part['observations'][variant][c]['ce'].mean());params.append(dict(tag,branch_step=1,probe_context=c,variant='interaction',gain=gain('full')-gain('shared_only')-gain('cue_only'),pre_ce=float(base[c]['ce'].mean()),post_ce=np.nan,structural_check=False))
 csv(OUT/'registry/branches.csv',registries);csv(OUT/'tables/branch_metrics.csv.gz',metrics_rows);parquet(OUT/'tables/branch_metrics.parquet',metrics_rows);csv(OUT/'tables/update_effects.csv',effects);csv(OUT/'tables/parameter_interventions.csv',params);csv(OUT/'tables/cue_contributions.csv.gz',cue);parquet(OUT/'tables/cue_contributions.parquet',cue);csv(OUT/'tables/update_audits.csv',timing);csv(OUT/'tables/missingness.csv',missing if missing else [dict(status='no_missing_branches',reason='',planned_branches=len(registries))]);return pd.DataFrame(effects)

def summarize_rows(frame,groups,metrics):
 out=[]
 for key,g in frame.groupby(groups,dropna=False):
  assert not g.seed.duplicated().any(),key;tag=dict(zip(groups,key if isinstance(key,tuple) else (key,)))
  for metric in metrics:out.append(dict(tag,metric=metric,**interval(g.set_index('seed')[metric].reindex(design.SEEDS).to_numpy(float))))
 return pd.DataFrame(out)

def statistics(effects):
 e=effects;groups=GROUP+['target_update','source_context','branch_type','branch_step','probe_context'];metrics_=['gain','pre_ce','post_ce','own_gain','cross_gain','specificity','margin_gain','probability_gain'];summary=summarize_rows(e,groups,metrics_);csv(OUT/'tables/seed_summaries.csv',summary);pairs=[];cis=[]
 key=['scale','objective','meta_stage','meta_episodes','target_update','source_context','branch_type','branch_step','probe_context']
 for tag,g in e.groupby(key):
  if set(g.regime)>={'U','P'}:
   for metric in metrics_:
    p=g.pivot(index='seed',columns='regime',values=metric).reindex(design.SEEDS);d=p.P-p.U
    for seed,x in d.items():pairs.append(dict(zip(key,tag),contrast='P-U',seed=seed,metric=metric,effect=x))
    cis.append(dict(zip(key,tag),contrast='P-U',metric=metric,**interval(d.to_numpy(float))))
 # Exact lineage contrasts at the common one-step checkpoints only.
 native=e[e.branch_type.eq('native')&e.branch_step.eq(1)&e.target_update.isin(design.PRIMARY)]
 joins=['seed','scale','target_update','source_context','probe_context']
 for row in native[native.meta_stage.ne('fresh')].to_dict('records'):
  candidates=[('F10000-F1920','fomaml',1920,row['regime'],row['objective'])] if row['meta_episodes']==10000 else []
  candidates+=([('R1920-Fresh','fresh',0,'Shared','none')] if row['meta_stage']=='reptile' else [('F-ownR1920','reptile',1920,row['regime'],'none')])
  for contrast,stage,dose,reg,obj in candidates:
   ref=native[native.meta_stage.eq(stage)&native.meta_episodes.eq(dose)&native.regime.eq(reg)&native.objective.eq(obj)]
   for k in joins:ref=ref[ref[k].eq(row[k])]
   assert len(ref)<=1
   for metric in metrics_:pairs.append(dict({k:row[k] for k in key},regime=row['regime'],contrast=contrast,seed=row['seed'],metric=metric,effect=row[metric]-ref.iloc[0][metric] if len(ref) else np.nan))
 pf=pd.DataFrame(pairs);stagepairs=pf[pf.contrast.ne('P-U')]
 if len(stagepairs):
  for tag,g in stagepairs.groupby(key+['regime','contrast','metric'],dropna=False):cis.append(dict(zip(key+['regime','contrast','metric'],tag),**interval(g.set_index('seed').effect.reindex(design.SEEDS).to_numpy(float))))
 csv(OUT/'tables/paired_seed_effects.csv',pf);csv(OUT/'tables/paired_intervals.csv',cis)
 # Native vs momentum-only conditional update. Held gains are exactly0 and retain countersn.
 z=e[e.branch_type.eq('zero_gradient_adam')];nat=e[e.branch_type.eq('native')&e.branch_step.eq(1)];j=META+['target_update','source_context','branch_step','probe_context'];a=z.merge(nat,on=j,suffixes=('_zero','_native'),validate='one_to_one');a['native_minus_zero_gain']=a.gain_native-a.gain_zero;a['held_gain']=0.;a['held_ce']=a.pre_ce_native;a['held_counter']=a.target_update;np.testing.assert_array_equal(a.pre_ce_zero,a.pre_ce_native);np.testing.assert_allclose(a.native_minus_zero_gain,a.post_ce_zero-a.post_ce_native,atol=1e-12,rtol=1e-12);csv(OUT/'tables/adam_controls.csv',a);csv(OUT/'tables/adam_control_intervals.csv',summarize_rows(a,GROUP+['target_update','source_context','probe_context'],['gain_native','gain_zero','native_minus_zero_gain','pre_ce_native','held_gain']))
 # First eligible checkpoint per seed/condition/bin; never select by gains.
 match=[]
 for tag,g in nat[nat.target_update.isin(design.PRIMARY)].groupby(GROUP+['seed','source_context','probe_context']):
  for lo,hi in BANDS:
   label=f'{lo:g}-{hi:g}';eligible=g[g.headroom_band.eq(label)].sort_values('target_update');rr=dict(zip(GROUP+['seed','source_context','probe_context'],tag),headroom_band=label,available=bool(len(eligible)))
   rr.update({k:eligible.iloc[0][k] if len(eligible) else np.nan for k in ['target_update','gain','own_gain','cross_gain','specificity','pre_ce','own_start_ce']});match.append(rr)
 m=pd.DataFrame(match);csv(OUT/'tables/matched_headroom.csv',m);mg=GROUP+['source_context','probe_context','headroom_band'];csv(OUT/'tables/matched_headroom_summary.csv',summarize_rows(m,mg,['gain','own_gain','cross_gain','specificity','pre_ce','own_start_ce']));mc=[]
 for tag,g in m.groupby([k for k in mg if k!='regime']):
  if set(g.regime)>={'U','P'}:
   for metric in ['gain','own_gain','cross_gain','specificity','pre_ce','own_start_ce']:
    pivot=g.pivot(index='seed',columns='regime',values=metric).reindex(design.SEEDS);mc.append(dict(zip([k for k in mg if k!='regime'],tag),metric=metric,contrast='P-U_conditional_first_eligible_own_loss',**interval((pivot.P-pivot.U).to_numpy(float))))
 csv(OUT/'tables/matched_headroom_paired.csv',mc)

def wide_gains(f,keys=None):
 keys=keys or META+['target_update','branch_type','branch_step'];p=f.pivot(index=keys,columns=['source_context','probe_context'],values='gain').reindex(columns=pd.MultiIndex.from_product([['A','B'],design.PROBES]));p.columns=[f'G_{source}_{probe.replace("-","_")}' for source,probe in p.columns];p=p.reset_index();p['D_A']=p.G_A_A-p.G_A_B;p['D_B']=p.G_B_B-p.G_B_A;return p

def export_gain_matrices():
 e=pd.read_csv(OUT/'tables/update_effects.csv',float_precision='round_trip');w=wide_gains(e);csv(OUT/'tables/gain_matrices.csv',w);csv(OUT/'tables/gain_matrix_intervals.csv',summarize_rows(w,GROUP+['target_update','branch_type','branch_step'],[c for c in w if c.startswith('G_') or c in ['D_A','D_B']]))

def check_early(e,original):
 for source,n in [('A',0),('B',10)]:
  q=e[e.source_context.eq(source)&e.target_update.eq(n)&e.branch_step.eq(10)&e.probe_context.eq('C')].set_index('trajectory_id');old=original.set_index('trajectory_id').reindex(q.index)
  assert old['D_'+source].notna().all()
  for new,name in [('specificity','D_'),('own_gain','own_'),('cross_gain','cross_')]:np.testing.assert_allclose(q[new],old[name+source],atol=1e-4,rtol=1e-5)

def original_early(q,e):
 old=pd.read_csv(SOURCE/'tables/outcomes.csv',float_precision='round_trip');old=old[old['update'].eq(330)];joined=q[['trajectory_id','seed','scale','condition','phase','dose']].merge(old,on=['seed','scale','condition','phase','dose'],validate='one_to_one');assert len(joined)==342
 keep=['trajectory_id','seed','scale','condition','phase','dose','own_A','cross_A','D_A','own_B','cross_B','D_B','floor_A','floor_B','source_floor_A','source_floor_B','early_cost','A_to_B','B_to_A','sustained_cost','functional_split','original_gate_available','familiar_ce','strong_familiar']
 csv(OUT/'tables/original_early_splitting.csv',joined[keep]);check_early(e[e.cohort.eq('final')&e.branch_type.eq('native')],joined)

def main():
 configure();receipt=OUT/'audit/analysis_complete.json'
 if receipt.exists():return
 assert json.loads((OUT/'audit/inspect_complete.json').read_text())['passed'];q=pd.read_csv(OUT/'registry/trajectories.csv');curve=curves(q);effects=extract_branches(q);original_early(q,effects);statistics(effects)
 cue=pd.read_parquet(OUT/'tables/cue_contributions.parquet');cg=META+['target_update','source_context','branch_type','branch_step'];cue_seed=cue.groupby(cg)[['C_minus_NoContext_ce','C_gain_minus_NoContext_gain','margin_difference','centered_C_norm']].mean().reset_index();csv(OUT/'tables/cue_seed_effects.csv',cue_seed);csv(OUT/'tables/cue_effect_intervals.csv',summarize_rows(cue_seed,GROUP+['target_update','source_context','branch_type','branch_step'],['C_minus_NoContext_ce','C_gain_minus_NoContext_gain','margin_difference','centered_C_norm']))
 import importlib.metadata
 env={k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','matplotlib','pyarrow']};path=OUT/'registry/analysis_environment.json'
 if path.exists():assert json.loads(path.read_text())==env
 else:dump(path,env)
 from .storage import Ledger
 counts=Ledger(OUT/'registry/optimizer_events.jsonl').counts();reg=pd.read_csv(OUT/'registry/branches.csv');completed=reg[reg.status.eq('completed')];exposure=dict(new_diagnostic_control_steps=counts['diagnostic'],new_reconstruction_steps=counts['reconstruction'],new_fixture_steps=counts['fixture'],new_meta_steps=0,archived_branch_optimizer_steps_reused=int(completed.reused_optimizer_steps.sum()),completed_branches=len(completed),planned_branches=len(reg),missing_or_failed_branches=int(reg.status.ne('completed').sum()),logical_completed_branch_steps=int(completed.horizon.sum()),failed_or_uncertain_attempts=counts['diagnostic']-int(completed.new_optimizer_steps.sum()))
 csv(OUT/'registry/exposure.csv',[exposure]);dump(OUT/'registry/storage_schema.json',dict(format='Required Parquet (existing pyarrow), exact CSV/gzip companions; native tensor state PT and compressed NumPy NPZ raw arrays',no_new_dependencies=True,parquet_dependency='existing installed pyarrow (version recorded separately)',positive_gain='baseline CE minus endpoint CE',position_index='1..7',probability_class='0..3',floor='inherited CE<=.05',headroom='lower inclusive; upper exclusive except4 inclusive',primary_metrics='native core.trace score on float32 logits; float64 aggregation; training CE_before separately nativefloat32',raw_npz_axes='branch_step(includingheld0),probe[A,B,C,NoContext,Mixed-average],position[1..7],class[0..3] or hidden[0..7]',row_keys=META+['target_update','source_context','branch_type','branch_step','probe_context','position']))
 dump(receipt,dict(passed=True,all_conditions_retained=True,optimizer_operations_in_analysis=0,**exposure));append('## Saved-table analysis completed\nActual gains telescope, realized exposure remains source-labelled, and original signed/floor measurements retained. Seed-level matched branches, masked endpoints, C−NoContext geometry and narrow Adam history controls exported. Inference uses identical20261122 whole-seed draws, no new score. Exact exposure and all missing/failed branches are explicit. No optimizer operations in summarization.')
