"""Cached-only A00/static/A06 comparison. No model is constructed or executed."""
import argparse,contextlib,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiments.context_experience_prior import core
from experiments.joint_phenotype_hypothesis_screen import engine as source
from experiments.joint_phenotype_hypothesis_screen.common import sha,csv,put
ROOT=core.ROOT;SOURCE=ROOT/'results/joint_phenotype_hypothesis_screen/v1';OUT=ROOT/'results/a06_cached_readout_comparison/v1';ARMS=['A00','A00-static','A06'];PAIRS=[('A00-static','A00'),('A06','A00'),('A06','A00-static')]

def static_logits(z,b):
 z=np.asarray(z);b=np.asarray(b)
 if z.ndim!=4 or b.shape!=(z.shape[0],4):raise ValueError('Each labeled checkpoint requires its own four-class output bias')
 return .25*(z-b[:,None,None,:])+b[:,None,None,:]
def layout(frame,shape,updates,probes):
 if frame[['update','cue']].duplicated().any() or frame['update'].drop_duplicates().tolist()!=list(updates):raise ValueError('Missing, duplicate or wrong checkpoint labels; never infer updates')
 if shape!=(len(updates),len(probes),7,4):raise ValueError('Unlabeled/misaligned array dimensions')
 for _,g in frame.groupby('update',sort=False):
  if g.cue.tolist()!=list(probes):raise ValueError('Missing or wrong physical probe labels')
 return list(updates),list(probes)
def common_updates(available):return sorted(set.intersection(*(set(v) for v in available.values()))) if available else []
def score_rows(z,seed,update,cue,arm,source):
 d=core.score(z);rows=[]
 for i in range(7):rows.append(dict(seed=int(seed),scale=5,regime='U',update=int(update),cue=cue,arm=arm,position=i+1,token=int(core.TARGET_PATH[i]),target=int(core.TARGET[i]),predicted_class=int(np.argmax(z[i])),correct=bool(d['accuracy'][i]),target_probability=float(d['probabilities'][i,core.TARGET[i]]),margin=float(d['margin'][i]),ce=float(d['ce'][i]),**{f'logit_{j}':float(z[i,j]) for j in range(4)},**{f'probability_{j}':float(d['probabilities'][i,j]) for j in range(4)},source=source))
 return rows
def aggregate(p):
 return p.groupby(['seed','scale','regime','update','cue','arm'],sort=False,as_index=False).agg(ce=('ce','mean'),correct_count=('correct','sum'),mean_margin=('margin','mean'),mean_target_probability=('target_probability','mean')).assign(accuracy=lambda f:f.correct_count/7)
def position_differences(p):
 rows=[];keys=['seed','update','cue','position','target']
 for first,second in PAIRS:
  a=p[p.arm.eq(first)];b=p[p.arm.eq(second)];m=a.merge(b,on=keys,suffixes=('_first','_second'),validate='one_to_one')
  for r in m.itertuples():
   transition=('stayed_correct' if r.correct_second else 'became_correct') if r.correct_first else ('became_incorrect' if r.correct_second else 'stayed_incorrect')
   rows.append(dict(**{k:getattr(r,k) for k in keys},contrast=first.replace('-','_')+'_minus_'+second.replace('-','_'),first_arm=first,second_arm=second,ce_difference=r.ce_first-r.ce_second,ce_contribution=(r.ce_first-r.ce_second)/7,correct_difference=int(r.correct_first)-int(r.correct_second),transition=transition,target_probability_difference=r.target_probability_first-r.target_probability_second,margin_difference=r.margin_first-r.margin_second))
 return pd.DataFrame(rows)
def paired(f):
 rows=[]
 for first,second in PAIRS:
  a=f[f.arm.eq(first)];b=f[f.arm.eq(second)];m=a.merge(b,on=['seed','update','cue'],suffixes=('_first','_second'),validate='one_to_one')
  for r in m.itertuples():rows.append(dict(seed=r.seed,update=r.update,cue=r.cue,first_arm=first,second_arm=second,ce_difference=r.ce_first-r.ce_second,correct_count_difference=int(r.correct_count_first)-int(r.correct_count_second),mean_margin_difference=r.mean_margin_first-r.mean_margin_second))
 return pd.DataFrame(rows)
def preserve_early(f,static_matches=None):
 q=f[f.candidate.isin(['A00','A06','A06_static'])&f.seed.isin([0,1])].copy();q['arm']=q.candidate.replace({'A06_static':'A00-static'});q['has_practice_history']=q.candidate.ne('A06_static')
 if static_matches is not None:
  for idx,r in q[q.candidate.eq('A06_static')].iterrows():
   if not static_matches.get(int(r.seed),False):q.loc[idx,'arm']='original_A06_static'
 return q
def registered_fallback(records):
 rows=[]
 for seed,name,frame in records:
  g=frame.copy();g['seed']=seed;g['candidate']=name;rows.append(g)
 return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['seed','candidate','update','cue','ce','correct_count'])

@contextlib.contextmanager
def cached_only():
 calls=[]
 def forbidden(*args,**kwargs):calls.append('prohibited');raise AssertionError('Cached-only analysis: model execution, backward, optimization and fitting are prohibited')
 with contextlib.ExitStack() as stack:
  for obj,name in [(core,'step'),(core,'trace'),(core,'objective'),(core,'optimizer'),(core,'ExperienceLinear'),(torch.nn.Module,'_call_impl'),(torch.Tensor,'backward'),(torch.autograd,'grad'),(torch.optim.Adam,'step'),(torch.optim.SGD,'step')]:stack.enter_context(patch.object(obj,name,forbidden))
  yield calls

class Inputs:
 def __init__(self):
  self.hashes={};self.missing=[];self.notes=[];self.parent=json.loads((SOURCE/'manifest.json').read_text());self.hashes[str((SOURCE/'manifest.json').relative_to(ROOT))]=sha(SOURCE/'manifest.json')
 def pin(self,path,key=None):
  p=Path(path)
  if not p.exists():self.missing.append(dict(path=str(p.relative_to(ROOT)),key=key,reason='file unavailable; no rollout/reconstruction'));return False
  rel=str(p.relative_to(ROOT));expected=self.parent['outputs'].get(str(p.relative_to(SOURCE))) if p.is_relative_to(SOURCE) else self.parent['code'].get(rel,self.parent['inputs'].get(rel));assert expected and sha(p)==expected,'Changed sealed source '+rel;self.hashes[rel]=expected;return True
 def read(self,name):
  p=SOURCE/name
  if not self.pin(p):return None
  return pd.read_csv(p,float_precision='round_trip') if p.suffix=='.csv' or p.suffix=='.gz' else json.loads(p.read_text())
 def run(self,seed,name,config):
  folder=SOURCE/f'runs/{name}_s{seed}';path=folder/'observations.npz'
  if not self.pin(path):return None
  with np.load(path,allow_pickle=False) as bundle:arrays={k:bundle[k].copy() for k in bundle.files}
  if 'logits' not in arrays:self.missing.append(dict(path=str(path.relative_to(ROOT)),key='logits',reason='essential key unavailable'));return None
  metrics=self.read(f'runs/{name}_s{seed}/metrics.csv')
  if metrics is None:return None
  updates=config['target']['probe_updates'];probes=[{'Mixed_average':'Mixed-average'}.get(v,v) for v in config['target']['probes']];assert updates==source.TIMES and probes==source.PROBES
  layout(metrics,arrays['logits'].shape,updates,probes)
  if 'output_bias' not in arrays:
   p=folder/'checkpoints.pt'
   if self.pin(p,'model.sequence_head.bias'):
    saved=torch.load(p,weights_only=False)
    try:arrays['output_bias']=np.array([saved[k]['model']['sequence_head.bias'].numpy() for k in updates]);self.notes.append(dict(path=str(p.relative_to(ROOT)),status='bias recovered only; no model constructed'))
    except (KeyError,TypeError):self.missing.append(dict(path=str(p.relative_to(ROOT)),key='model.sequence_head.bias',reason='bias unavailable'))
  return dict(arrays=arrays,updates=list(updates),probes=probes,source=str(path.relative_to(ROOT)),metrics=metrics)

def build(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);inputs=Inputs()
 for path in ['experiments/context_experience_prior/core.py','experiments/joint_phenotype_hypothesis_screen/engine.py','experiments/joint_phenotype_hypothesis_screen/common.py','experiments/joint_phenotype_hypothesis_screen/rules.py']:inputs.pin(ROOT/path)
 config=inputs.read('registry/resolved_config.json');assert config['source']['scale']==5 and config['source']['regime']=='U';derived=inputs.read('registry/derived_controls.csv');registered=inputs.read('metrics.csv');summaries=inputs.read('summary.csv');ref=inputs.read('reference_thresholds.json');routes=inputs.read('selection_route_components.csv');selection=inputs.read('selection.json');practice=inputs.read('practice.csv.gz');inputs.pin(SOURCE/'registry/optimizer_events.jsonl');controls=inputs.read('registry/control_sources.json')
 positions=[];available=[];static_checks=[];metric_frames=[]
 for seed in [0,1]:
  cache={arm:inputs.run(seed,arm,config) for arm in ['A00','A06']};base=cache['A00'];old=None
  row=derived[derived.seed.eq(seed)] if derived is not None else pd.DataFrame()
  if len(row)==1:
   original=str(row.iloc[0].source);assert original==f'runs/A06_static_s{seed}' and row.iloc[0].physical_model==f'A00_s{seed}'
   old=inputs.run(seed,'A06_static',config)
   if old:assert sha(SOURCE/original/'observations.npz')==row.iloc[0].raw_sha and sha(SOURCE/original/'metrics.csv')==row.iloc[0].metrics_sha
  else:inputs.missing.append(dict(path=str((SOURCE/'registry/derived_controls.csv').relative_to(ROOT)),key=f'seed{seed}',reason='derived view not uniquely registered'))
  if base and 'output_bias' in base['arrays']:
   z=static_logits(base['arrays']['logits'],base['arrays']['output_bias']);arrays=source.from_logits(z,base['arrays']['output_bias']);consistent=False;error=None
   if old:
    error=float(abs(old['arrays']['logits']-z).max());tolerance=1e-10+1e-12*abs(z).max();consistent=bool(old['updates']==base['updates'] and old['probes']==base['probes'] and error<=tolerance)
   static_checks.append(dict(seed=seed,original_name='A06_static',original_source=old['source'] if old else None,cached_formula_matches=consistent,max_logit_error=error,formula='0.25*(z_A00-b_A00_checkpoint)+b_A00_checkpoint',action='reuse verified cached view' if consistent else 'preserve original source view and derive PRD view separately'))
   cache['A00-static']=dict(arrays=old['arrays'] if consistent else arrays,updates=base['updates'],probes=base['probes'],source=(old['source'] if consistent else base['source']+' + checkpoint_bias_formula'))
  else:
   inputs.missing.append(dict(path=str((SOURCE/f'runs/A00_s{seed}/observations.npz').relative_to(ROOT)),key='output_bias',reason='A00-static unavailable; do not replace with0.25*z'))
  for arm in ['A00','A06']:
   if cache.get(arm):metric_frames.append((seed,arm,cache[arm]['metrics']))
  if old:metric_frames.append((seed,'A06_static',old['metrics']))
  common=common_updates({k:v['updates'] for k,v in cache.items() if v});all_updates=sorted({u for v in cache.values() if v for u in v['updates']})
  for arm in ARMS:
   run=cache.get(arm)
   available.append(dict(seed=seed,arm=arm,available=run is not None,updates=run['updates'] if run else [],mutual_updates=common,unmatched_updates=[u for u in all_updates if not run or u not in run['updates']]))
   if not run:continue
   for index,k in enumerate(run['updates']):
    if k not in common:continue
    for j,c in enumerate(run['probes']):positions.extend(score_rows(run['arrays']['logits'][index,j],seed,k,c,arm,run['source']))
 p=pd.DataFrame(positions);f=aggregate(p);d=paired(f);pdiff=position_differences(p)
 for first,second,tag in [('A00-static','A00','static_minus_A00'),('A06','A00','A06_minus_A00'),('A06','A00-static','A06_minus_static')]:
  diff=d[d.first_arm.eq(first)&d.second_arm.eq(second)][['seed','update','cue','ce_difference','correct_count_difference']].rename(columns={'ce_difference':tag+'_ce','correct_count_difference':tag+'_correct_count'});f=f.merge(diff,on=['seed','update','cue'],how='left',validate='many_to_one')
 # Acceptance: all source metrics and all position scores agree with archived measurements.
 accepted=0
 if registered is None:registered=registered_fallback(metric_frames)
 for arm,oldname in [('A00','A00'),('A06','A06'),('A00-static','A06_static')]:
  q=f[f.arm.eq(arm)].merge(registered[registered.candidate.eq(oldname)&registered.seed.isin([0,1])],on=['seed','update','cue'],suffixes=('_new','_source'),validate='one_to_one')
  if arm!='A00-static' or all(r['cached_formula_matches'] for r in static_checks):
   np.testing.assert_allclose(q.ce_new,q.ce_source,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(q.correct_count_new,q.correct_count_source);accepted+=len(q)
 for r in d.itertuples():
  z=pdiff[(pdiff.seed==r.seed)&(pdiff['update']==r.update)&(pdiff.cue==r.cue)&(pdiff.first_arm==r.first_arm)&(pdiff.second_arm==r.second_arm)];assert len(z)==7;np.testing.assert_allclose(z.ce_contribution.sum(),r.ce_difference,atol=1e-12,rtol=1e-12);assert z.correct_difference.sum()==r.correct_count_difference
 early=preserve_early(summaries,{r['seed']:r['cached_formula_matches'] for r in static_checks}) if summaries is not None else pd.DataFrame(columns=['candidate','seed','arm','has_practice_history'])
 if practice is None:
  rows=[]
  for seed in [0,1]:
   for arm in ['A00','A06']:
    g=inputs.read(f'runs/{arm}_s{seed}/practice.csv')
    if g is not None:g=g.copy();g['candidate']=arm;g['seed']=seed;rows.append(g)
  practice=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['candidate','seed','update','training_cue','ce_before'])
 practice=practice[practice.candidate.isin(['A00','A06'])&practice.seed.isin([0,1])].copy()
 for (seed,arm),g in practice.groupby(['seed','candidate']):assert g['update'].tolist()==list(range(1,331)) and g.training_cue.tolist()==['A' if ((n-1)//10)%2==0 else 'B' for n in range(1,331)]
 for row in early.itertuples():
  ff=f[f.seed.eq(row.seed)&f.arm.eq(row.arm)]
  if ff.empty:continue
  r=source.rules.summarize(ff)
  for key in ['own_A','cross_A','D_A','Q_A','own_B','cross_B','D_B','Q_B','E','E_normalized','A_to_B','B_to_A','sustained_cost']:np.testing.assert_allclose(r[key],getattr(row,key),atol=1e-12,rtol=1e-12,equal_nan=True)
 csv(out/'comparison.csv',f);csv(out/'positions.csv',p);csv(out/'paired.csv',d);csv(out/'position_differences.csv',pdiff);csv(out/'early.csv',early);csv(out/'practice.csv.gz',practice);csv(out/'endpoint.csv',f[f['update'].eq(330)])
 wide_tables(out,p,pdiff)
 put(out/'missing_inputs.json',dict(missing=inputs.missing,notes=inputs.notes,coverage=available));put(out/'source_audit.json',dict(inputs=inputs.hashes,static_views=static_checks,targets=core.TARGET.tolist(),tokens=core.TARGET_PATH[:-1],probe_vectors={k:core.CUES[k] for k in source.PROBES},label_sources=['recorded per-run metrics.csv','resolved_config target.probe_updates','pinned source engine.TIMES/PROBES writer contract'],scale=5,regime='U',seeds=[0,1],no_checkpoint_recovery_needed=not bool(inputs.notes)))
 validation=dict(passed=True,aggregate_rows=len(f),position_rows=len(p),paired_rows=len(d),position_difference_rows=len(pdiff),archived_metric_matches=accepted,all_four_logits_and_probabilities=True,original_early_measures_unchanged=summaries is not None,original_classifications_copied=summaries is not None,source_selection_unchanged=selection is not None and selection['selected']==[],new_model_rollouts=0,new_optimizer_steps=0,new_seeds=0,parameter_fits=0)
 return f,p,d,pdiff,early,practice,ref,routes,validation

def wide_tables(out,p,d):
 for seed in [0,1]:
  for cue in ['C','NoContext']:
   q=p[p.seed.eq(seed)&p['update'].eq(330)&p.cue.eq(cue)];keys=['position','token','target'];w=q.pivot(index=keys,columns='arm',values=['predicted_class','correct','target_probability','margin','ce']);w.columns=['__'.join(v) for v in w.columns];w=w.reset_index()
   for first,second in PAIRS:
    v=d[d.seed.eq(seed)&d['update'].eq(330)&d.cue.eq(cue)&d.first_arm.eq(first)&d.second_arm.eq(second)][['position','transition','ce_contribution']];label=first+'_minus_'+second;w=w.merge(v.rename(columns={'transition':label+'__transition','ce_contribution':label+'__ce_contribution'}),on='position',how='left')
   csv(out/f'seed{seed}_{cue}_positions.csv',w)
