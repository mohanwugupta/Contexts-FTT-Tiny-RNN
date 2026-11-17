"""R1 assembles the exact inherited-state/current-cue factorial from P4 only."""
import numpy as np
import pandas as pd
from .core import *
from .audit import pool,validate_rows


def history_prestate(hidden,k):return np.zeros(8,np.float32) if k==1 else np.asarray(hidden[k-2],np.float32)

def validate_factorial(f):
 validate_rows(f,records());groups=f.groupby(['seed','k','source_context','branch'])
 if len(groups)!=360:raise ValueError('Missing factorial cells')
 for (seed,k,g,c),q in groups:
  if sorted(q.transition.tolist())!=list(range(k,8)):raise ValueError('Missing/duplicated steps')
 if not f.groupby(['seed','k','source_context']).prestate_hash.nunique().eq(1).all():raise ValueError('Current-cue branches changed inherited prestate')
 return True

def summarize(t,reference):
 q=future_summary(t,reference)
 return dict(q,current_ce=float(t[t.lag==0].raw_ce.iloc[0]),full_remaining_ce=float(t.raw_ce.mean()))

def enrich(t):
 t=t.copy();z=t[Z].to_numpy();y=t.target.to_numpy(int);wrong=z.copy();wrong[np.arange(len(z)),y]=-np.inf
 t['correct_target_margin']=z[np.arange(len(z)),y]-wrong.max(1)
 for j in range(4):t[f'centered_logit_{j}']=z[:,j]-z.mean(1)
 return t

def run(out):
 f=pool();validate_factorial(f);nat=read(out,'r0_native_replay');seedrows=[];contrasts=[]
 f=enrich(f)
 for (seed,k,g,c),t in f.groupby(['seed','k','source_context','branch'],sort=True):
  t=t.sort_values('transition');q=summarize(t,refs(nat,seed));q.update(seed=seed,k=k,source_context=g,branch=c,schedule='pulse',future_cue='Mixed',status='exact_reuse',checkpoint_sha256=t.checkpoint_sha256.iloc[0],prestate_hash=t.prestate_hash.iloc[0],source_sample_id=t.sample_id.iloc[0])
  error=t[t.correct_target_margin<0];q['first_error_lag']=int(error.lag.iloc[0]) if len(error) else np.nan;q['current_correct']=bool(t.correct_target_margin.iloc[0]>0)
  q['later_failure']=bool((t[t.lag>0].correct_target_margin<0).any());seedrows.append(q)
 cells=pd.DataFrame(seedrows)
 for seed in range(20):
  for k in K:
   def cell(g,c):return cells[(cells.seed==seed)&(cells.k==k)&(cells.source_context==g)&(cells.branch==c)].iloc[0]
   for endpoint in ['current_ce','future_ce']:
    both=[]
    for g in ['A','B']:
     h=cell(g,'Mixed')[endpoint]-cell('Mixed','Mixed')[endpoint];both.append(h)
     for name,val in [('history_'+g,h),('cue_'+g,cell(g,'Mixed')[endpoint]-cell(g,g)[endpoint]),('interaction_'+g,(cell(g,'Mixed')[endpoint]-cell(g,g)[endpoint])-(cell('Mixed','Mixed')[endpoint]-cell('Mixed',g)[endpoint])),('swapped_'+g,cell(g,'B' if g=='A' else 'A')[endpoint]-cell(g,g)[endpoint])]:
      contrasts.append(dict(seed=seed,k=k,endpoint=endpoint,contrast=name,effect=val,n_references=1))
    contrasts.append(dict(seed=seed,k=k,endpoint=endpoint,contrast='history_mean_AB',effect=np.mean(both),n_references=2))
 con=pd.DataFrame(contrasts);stats=[]
 for (k,e,c),q in con.groupby(['k','endpoint','contrast']):stats.append(dict(k=k,endpoint=e,contrast=c,**bootstrap(q)))
 save(out,'r1_branch_rollouts',f);save(out,'r1_cells',cells);save(out,'r1_seed_contrasts',con);save(out,'r1_bootstrap',pd.DataFrame(stats))
 cases=[]
 for k in K:
  for seed in range(20):
   t=cells[(cells.seed==seed)&(cells.k==k)&cells.source_context.isin(['A','B'])&(cells.branch=='Mixed')]
   cases.append(dict(seed=seed,k=k,n_references=2,n_viable=int(t.viable_continuation.sum()),any_viable=bool(t.viable_continuation.any()),any_current_correct=bool(t.current_correct.any()),any_current_correct_then_error=bool((t.current_correct&t.later_failure).any())))
 save(out,'r1_reference_cases',pd.DataFrame(cases))
 report(out,'R1_STATE_X_CUE_REPORT.md','# R1 — inherited state × one current cue\n\n360/360 cells and1440/1440 transition rows exactly reused;0 scientific rollout recomputations. Every seed and allnine cells retained at k3/k5. Source-checkpoint, prestate, current token, four-class head and Mixed future schedule were verified in R0. Natural A/B at k equals the familiar current prediction; its future here is deliberately Mixed, never called full-A/B performance.\n\n'+table(pd.DataFrame(stats))+'\n\nAll effects are signed CE differences, negative=improvement. Primary k3 state-history contrasts change only inherited state while current and future cues stay Mixed. Cue effects change only the current cue; interactions subtract the same cue contrast from natural Mixed history. Both A/B references are retained independently of viability. Margins, first-error lag and subsequent failures are descriptive; neither proves memory loss. Future excludes current k, full-remaining includes it only as a secondary endpoint. Familiar-reference viability is imported unchanged, +.10 with equality passing. Bootstrap draws trained seeds; k5 is descriptive secondary.\n')
 return dict(reused_cells=360,new_scientific_rollouts=0,primary_viable_reference_seeds=int(pd.DataFrame(cases).query('k==3').any_viable.sum()),all20_present=True,model_updates=0)
