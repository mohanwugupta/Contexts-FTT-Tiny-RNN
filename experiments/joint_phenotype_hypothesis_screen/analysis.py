"""Saved-table summaries, original early measures and fixed pilot selection."""
from .common import *
from . import rules,engine

def seed_summary(f,candidate,seed,role,ref):
 r=dict(candidate=candidate,seed=seed,role=role,status='completed',**rules.summarize(f));end=f[f['update'].eq(330)].set_index('cue');r['familiar']=all(rules.useful(end.loc[c,'ce'],end.loc[c,'correct_count']) for c in ['A','B']);r['familiar_ce']=float(end.loc[['A','B'],'ce'].mean())
 for label,c in [('A','A'),('B','B'),('C','C'),('Mixed','Mixed-average'),('NoContext','NoContext')]:
  v=end.loc[c];r.update({label+'_ce':float(v.ce),label+'_correct':int(v.correct_count),label+'_accuracy':int(v.correct_count)/7,label+'_margin':float(v.margin),label+'_gap':float(v.ce-r['familiar_ce']),label+'_useful':rules.useful(v.ce,v.correct_count),label+'_strict':rules.strict(v.ce,v.correct_count),label+'_historical_strict_gap':bool(v.ce<=.1 and v.correct_count==7 and v.ce-r['familiar_ce']<=.1)})
 r['early']=rules.early_gate(r,ref);r['reference_status']='resolved' if ref['resolved'] else 'reference_unresolved';r['interference_dominated']=bool(r['net_A']<=0 or r['net_B']<=0);r['joint']=bool(r['familiar'] and r['C_useful'] and r['Mixed_useful'] and r['early']);r['strict_joint']=bool(r['joint'] and r['C_strict']);r['strict_Mixed']=r['Mixed_strict'];r['original_early_phenotype']=bool(r['familiar'] and not r['floor_A'] and not r['floor_B'] and min(r['own_A'],r['own_B'],r['D_A'],r['D_B'],r['cycle'])>.05);r['original_late_functional']=np.nan;r['original_late_status']='unassessed_without_new_counterfactual_updates'
 ratios=[r['Q_A']/ref['Q_A'],r['Q_B']/ref['Q_B'],r['E_normalized']/ref['E_normalized']] if ref['resolved'] else [np.nan];r['minimum_retention']=float(min(ratios)) if np.isfinite(ratios).all() else np.nan
 # Source function, unchanged: cross-check the explicit six-boundary screen.
 from experiments.linear_scale_sweep.analysis import transition_summary
 sw=rules.switches(f).rename(columns={'update':'boundary','raw_switch':'raw_cost','adjusted_switch':'adjusted_cost'});p=f.pivot(index='update',columns='cue',values='ce');sw['incoming_ce']=[p.loc[x.boundary,x.incoming] for x in sw.itertuples()];sw['seed']=seed;sw['arm']=candidate;sw['dose']=0
 old=transition_summary(sw).iloc[0]
 for k in ['early_cost','sustained_cost','A_to_B','B_to_A']:assert r[k]==old[k],'Changed original early/sustained measure'
 return r

def classify(r,b,background_adequate=True):
 if r['joint']:return 'joint_feasible'
 if not background_adequate:return 'budget_inconclusive'
 transfer=(b['C_ce']-r['C_ce']>=.05 or r['C_correct']>b['C_correct'])
 splitting=(r['E_normalized']-b['E_normalized']>=.01 and r['Q_A']>0 and r['Q_B']>0)
 splitting_loss=(b['Q_A']-r['Q_A']>.05 or b['Q_B']-r['Q_B']>.05 or b['E_normalized']-r['E_normalized']>.01 or (b['familiar'] and not r['familiar']))
 transfer_loss=(r['C_ce']-b['C_ce']>.05 or r['C_correct']<b['C_correct'])
 if (transfer and splitting_loss) or (splitting and transfer_loss):return 'tradeoff'
 if transfer or splitting or r['early'] or r['C_useful'] or (r['familiar'] and not b['familiar']):return 'partial_signal'
 return 'no_useful_change'

def collect():
 ref=json.loads((OUT/'reference_thresholds.json').read_text());metrics=[];summaries=[];practices=[];background=[]
 for dest in sorted((OUT/'runs').iterdir()):
  receipt=dest/'complete.json'
  if not receipt.exists():continue
  c=json.loads(receipt.read_text());f=pd.read_csv(dest/'metrics.csv',float_precision='round_trip');metrics.append(f);summaries.append(seed_summary(f,c['candidate'],int(c['seed']),c['role'],ref))
  if c['candidate']!='A06_static':
   t=pd.read_csv(dest/'practice.csv',float_precision='round_trip');t['candidate']=c['candidate'];t['seed']=c['seed'];t['role']=c['role'];practices.append(t)
 for p in sorted((OUT/'background').glob('*/complete.json')):background.append(json.loads(p.read_text()))
 summary=pd.DataFrame(summaries);lookup={(r['candidate'],r['seed']):r for r in summaries};bg={(r['candidate'],r['seed'],r['fresh']):r for r in background}
 for idx,row in summary.iterrows():
  candidate=row.candidate;seed=int(row.seed);control='A00' if candidate.startswith('A') else 'B00';adequate=bg.get(('B00',seed,False),dict(background_competent=True,adaptation_viable=True));adequate=adequate['background_competent'] and adequate['adaptation_viable']
  if candidate in rules.ORDER:
   base=lookup[(control,seed)];summary.loc[idx,'classification']=classify(row,base,adequate if candidate.startswith('B') else True);summary.loc[idx,'acquisition_plus_transfer']=bool(row.familiar and not base['familiar']);summary.loc[idx,'matched_control']=control
   for field in ['C_ce','C_correct','E_normalized','Q_A','Q_B','familiar_ce','NoContext_ce']:summary.loc[idx,'delta_'+field]=row[field]-base[field]
  else:summary.loc[idx,'classification']='reference' if candidate=='F00' else ('frozen_expression_control' if candidate=='A06_static' else 'matched_control')
  summary.loc[idx,'background_adequate_control']=adequate if candidate.startswith('B') else np.nan
 # Every registered pilot row receives an explicit outcome, including unavailable branches.
 registry=pd.read_csv(OUT/'candidate_registry_initial.csv');selection=json.loads((OUT/'selection.json').read_text()) if (OUT/'selection.json').exists() else None
 for idx,row in registry.iterrows():
  if (row.candidate,int(row.seed)) in lookup:registry.loc[idx,'status']='completed_reused' if row.candidate in ['F00','A00','B00'] else 'completed'
  elif row.candidate=='A07':registry.loc[idx,'status']='not_triggered' if not json.loads((OUT/'registry/A07_trigger.json').read_text())['triggered'] else row.status
  elif row.candidate=='A04' and not json.loads((OUT/'registry/sgd_calibration.json').read_text())['passed']:registry.loc[idx,'status']='calibration_inconclusive'
  else:
   files=[OUT/f'runs/{row.candidate}_s{row.seed}/{n}' for n in ['failure.json','meta_failure.json']]
   failure=next((p for p in files if p.exists()),None)
   if failure:registry.loc[idx,'status']=json.loads(failure.read_text())['status']
   elif row.role=='reserved':registry.loc[idx,'status']='pending_confirmation' if selection and row.candidate in selection['selected'] else 'not_selected'
  if row.role=='pilot' and row.candidate in rules.ORDER and (row.candidate,int(row.seed)) not in lookup:
   summaries_missing=dict(candidate=row.candidate,seed=int(row.seed),role='pilot',status=registry.loc[idx,'status'],classification=registry.loc[idx,'status'],joint=False,strict_joint=False)
   summary=pd.concat([summary,pd.DataFrame([summaries_missing])],ignore_index=True)
 return dict(metrics=pd.concat(metrics,ignore_index=True),summary=summary,practice=pd.concat(practices,ignore_index=True),background=pd.DataFrame(background),registry=registry)

def select():
 engine.require_identity();assert all(json.loads((OUT/f'audit/pilot_{p}_complete.json').read_text())['passed'] for p in ['checkpoint','pretraining']);f=collect();s=f['summary'];rows=[]
 for candidate in rules.ORDER:
  a=s[s.candidate.eq(candidate)&s.seed.isin([0,1])].sort_values('seed');control='A00' if candidate.startswith('A') else 'B00';b=s[s.candidate.eq(control)&s.seed.isin([0,1])].sort_values('seed')
  if len(a)!=2 or not a.status.eq('completed').all():
   rows.append(dict(id=candidate,eligible=False,status=a.status.iloc[0] if len(a) else 'source_unavailable',strict_count=0,retention=None,C_ce=None));continue
  static=s[s.candidate.eq('A06_static')&s.seed.isin([0,1])] if candidate=='A06' else None;r=rules.route(a,b,candidate,static);retention=float(a.minimum_retention.min()) if a.minimum_retention.notna().all() else -np.inf
  rows.append(dict(id=candidate,status='completed',**r,strict_count=int(a.strict_joint.sum()),joint_count=int(a.joint.sum()),retention=retention,C_ce=float(a.C_ce.mean()),control=control,pilot_seeds=[0,1],classification=a.classification.tolist()))
 selected=rules.rank(rows)[:2];finite=[dict(r,eligible=True) for r in rows if r['status']=='completed' and r['id'] not in selected];partial=rules.rank(finite)[:2]
 # Immutable input views pin exactly what selection saw, before confirmation.
 csv(OUT/'registry/pilot_summary_at_selection.csv',s);csv(OUT/'registry/pilot_metrics_at_selection.csv',f['metrics']);result=dict(passed=True,selected=selected,eligible_ranking=rules.rank(rows),partial_display=partial,rows=rows,frozen_runtime_sha=sha(OUT/'registry/runtime_identity.json'),configuration_sha=sha(ROOT/'configs/joint_phenotype_hypothesis_screen.yaml'),reference_sha=sha(OUT/'reference_thresholds.json'),pilot_summary_sha=sha(OUT/'registry/pilot_summary_at_selection.csv'),pilot_metrics_sha=sha(OUT/'registry/pilot_metrics_at_selection.csv'),selection_seeds=[0,1],reserved_confirmation_seeds=[2,3,4,5],ranking='strict_joint_count descending; worst minimum Fresh retention descending; mean C CE ascending; registry order',partial_display_label='descriptive only; not promoted or winners',procedures_unchanged=True)
 put(OUT/'selection.json',result);append('## Selection frozen\nBoth pilot batches completed before applying the registered rules once. Selected='+str(selected)+'. Ineligible, negative and undefined components retained. Reserved seeds were not used for selection. Descriptive partial-display ranking does not promote candidates.');print(json.dumps(clean(result)),flush=True);return result
