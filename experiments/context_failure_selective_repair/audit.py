"""R0 source fidelity checks and exact-reuse registry; no scientific factorial rerun."""
import inspect,json
import numpy as np
import pandas as pd
from context_sequence.provenance.hashing import sha256
from experiments.output_compatible_trajectories.continuation import natural_trace,digest_state
from experiments.output_compatible_trajectories.cue_access import cue_branch
from experiments.context_mixing_sequence_continuity.audit import verify_prior_scientific
from .core import *


def validate_cues(c):
 for g,v in CONTRACT['cue'].items():
  if g not in c or not np.array_equal(c[g],v):raise ValueError('Physical cue registry mismatch: '+g)
 return True

def validate_rows(df,recs):
 ids={r['seed']:r['checkpoint_sha256'] for r in recs}
 for r in df.itertuples():
  if r.seed not in ids or r.checkpoint_sha256!=ids[r.seed] or r.cohort!='canonical' or r.condition!='High_monotonic':raise ValueError('Checkpoint mismatch')
  if r.source_context not in NAMES or r.branch not in NAMES or r.schedule!='pulse' or r.cue!='Mixed' or r.intervention_type!='cue_only' or r.model_optimizer_updates!=0:raise ValueError('Incompatible pulse cell')
  if r.k not in K or r.transition not in range(r.k,8) or r.lag!=r.transition-r.k or r.token!=SEQUENCE[r.transition-1] or r.target!=SEQUENCE[r.transition]:raise ValueError('Off-by-one scoring window')
  expect=CONTRACT['cue'][r.branch if r.lag==0 else 'Mixed']
  if not np.array_equal([r.cue_0,r.cue_1,r.cue_2],expect):raise ValueError('Wrong current/future cue')
 return True

def fidelity(a,b):
 cols=[c for c in a if c.startswith(('hidden_','logit_','probability_')) and c not in ['hidden_norm']]
 if len(a)!=len(b) or not len(a):raise ValueError('Missing fidelity evidence')
 err=float(np.max(np.abs(a[cols].to_numpy(float)-b[cols].to_numpy(float))))
 if not np.isfinite(err) or err>1e-12:raise ValueError('recompute_required: source replay mismatch')
 return dict(passed=True,max_abs_error=err,tolerance=1e-12)

def pool():
 f=pd.read_csv(P4/'tables/p4_same_prestate_branch_long.csv',float_precision='round_trip')
 return f[(f.cue=='Mixed')&f.k.isin(K)&f.source_context.isin(NAMES)&f.branch.isin(NAMES)&(f.schedule=='pulse')&(f.intervention_type=='cue_only')].copy()

def inputs():
 # Verify every sealed scientific byte, excluding only explicitly named Finder metadata.
 allfiles={};aud=[]
 for base in [ROOT/'results/context_cue_access/v1',D]:
  extra=verify_prior_scientific(base);m=json.loads((base/'manifests/sealed_manifest.json').read_text());allfiles.update(m['inputs']);allfiles.update(m['code_hashes'])
  allfiles.update({str((base/p).relative_to(ROOT)):h for p,h in m['files'].items()});allfiles[str((base/'manifests/sealed_manifest.json').relative_to(ROOT))]=sha256(base/'manifests/sealed_manifest.json')
 m=json.loads((P4/'manifest.json').read_text());check_files(ROOT,m['code_hashes']);check_files(P4,m['files']);allfiles.update(m['code_hashes'])
 allfiles.update({str((P4/p).relative_to(ROOT)):h for p,h in m['files'].items()});allfiles[str((P4/'manifest.json').relative_to(ROOT))]=sha256(P4/'manifest.json')
 for p in [ROOT/'PRD_SRN_CONTEXT_FAILURE_SELECTIVE_REPAIR.md',*list((ROOT/'phase1k_cue_generation').glob('*.py')),ROOT/'analysis/phase1k/PHASE1K_CUE_GENERATION_REPORT.md']:
  allfiles[str(p.relative_to(ROOT))]=sha256(p)
 for r in records():allfiles[r['path']]=r['checkpoint_sha256']
 return allfiles

def run(out):
 validate_cues(cues());f=pool();validate_rows(f,records());keys=['seed','k','source_context','branch'];groups=f.groupby(keys)
 if len(groups)!=360 or len(f)!=1440:raise ValueError('Incomplete exact-reuse pool')
 old=pd.read_csv(D/'tables/d0_native_replay.csv',float_precision='round_trip');nat=[];checks=[];reg=[];models=[];calls=0
 for rec in records():
  m=load_original(ROOT,rec);before=m.digest()
  if m.wih.shape!=(8,7) or m.whh.shape!=(8,8) or m.output.shape!=(4,8) or np.any(m.wih[:,6]!=0):raise ValueError('Wrong canonical source')
  saved={}
  with frozen_weight_guard(m):
   for g in NAMES:
    t=natural_trace(m,cues()[g]);calls+=1;saved[g]=t
    checks.append(dict(seed=rec['seed'],k=0,source_context=g,branch=g,kind='natural_full',**fidelity(t,native(old,rec['seed'],g))))
    t['seed']=rec['seed'];t['cue']=g;t['checkpoint_sha256']=rec['checkpoint_sha256'];nat.append(t)
   for k in K:
    # Every reused native baseline and familiar same-cue current output is checked.
    for g in NAMES:
     reused=select(f,rec['seed'],k,g,g)
     a=saved[g].iloc[k-1:].copy() if g=='Mixed' else saved[g].iloc[[k-1]].copy()
     b=reused if g=='Mixed' else reused.iloc[[0]]
     checks.append(dict(seed=rec['seed'],k=k,source_context=g,branch=g,kind='native_tail_or_current',**fidelity(a,b)))
    if rec['seed']==0:
     for g in NAMES:
      pre=saved[g][H].iloc[k-2].to_numpy(np.float32)
      for c in NAMES:
       if g==c=='Mixed':continue # already full-tail replayed
       t=cue_branch(m,pre,k,cues()[c],cues()['Mixed'],persistent=False)
       checks.append(dict(seed=0,k=k,source_context=g,branch=c,kind='representative_factorial_fidelity',**fidelity(t,select(f,0,k,g,c))));calls+=1
  models.append(dict(**rec,tensor_digest=before,weights_unchanged=m.digest()==before))
  for k in K:
   for g in NAMES:
    pre=saved[g][H].iloc[k-2].to_numpy(np.float32);want=digest_state(pre)
    for c in NAMES:
     q=select(f,rec['seed'],k,g,c)
     if q.transition.tolist()!=list(range(k,8)) or q.prestate_hash.nunique()!=1 or q.prestate_hash.iloc[0]!=want:raise ValueError('Missing/duplicated history cell')
     reg.append(dict(seed=rec['seed'],k=k,source_context=g,current_cue=c,schedule='pulse',future_cue='Mixed',input_token=SEQUENCE[k-1],current_target=SEQUENCE[k],checkpoint_sha256=rec['checkpoint_sha256'],prestate_hash=want,source_path=str((P4/'tables/p4_same_prestate_branch_long.csv').relative_to(ROOT)),source_sha256=sha256(P4/'tables/p4_same_prestate_branch_long.csv'),source_sample_id=q.sample_id.iloc[0],available_fields='hidden8|logit4|probability4|raw_ce|cue3|token|target|transition',indexing='1-based post-update; future excludes k',status='exact_reuse'))
 save(out,'r0_native_replay',pd.concat(nat,ignore_index=True));save(out,'r0_source_fidelity',pd.DataFrame(checks));save(out,'r0_checkpoints',pd.DataFrame(models));save(out,'r0_reuse_registry',pd.DataFrame(reg))
 leads=[]
 for p in [ROOT/'analysis/phase1k/phase1k_local_cue_branches.csv',ROOT/'analysis/phase1k/phase1k_pulse_transition_metrics.csv']:
  unavailable=not p.is_file() or bool(p.stat().st_flags&0x40000000)
  leads.append(dict(path=str(p.relative_to(ROOT)),status='unavailable_cloud_optional' if unavailable else 'available_not_needed',observed_sha=None if unavailable else sha256(p),reason='Exact P4 supersedes; Phase1K local geometry alone is not CE; native-restoring pulse is not common Mixed future.'))
 save(out,'r0_legacy_sources',pd.DataFrame(leads))
 dump(out/'audit/future_summary_contract.json',dict(path=inspect.getsourcefile(future_summary),line=inspect.getsourcelines(future_summary)[1],source=inspect.getsource(future_summary),sha256=sha256(inspect.getsourcefile(future_summary)),formula='mean(CE[k+1:7]) <= mean((A.raw_ce[k:]+B.raw_ce[k:])/2)+0.10'))
 means=pd.concat(nat).groupby('cue').raw_ce.mean().reset_index()
 report(out,'audit/R0_SOURCE_AND_REUSE_MAP.md','# R0 — actual source and reuse audit\n\nAll20 canonical High_monotonic seeds0–19, original170, are SHA-verified. Eight tanh coordinates, four one-hot tokens, four-class sequence head, three raw context channels; third input column exactly zero. Mixed=(.5,.5,0), not the sum (1,1,0). Source evaluator `src/context_sequence/evaluation/rollout.py:Model.step` executes float32 affine/tanh/head. It folds the original two recurrent biases in float32; prior D0 established Torch/NumPy source tolerances. The execution policy is separate and not scored.\n\n'+table(means)+'\n\nAll360 cells/1440 transition rows exactly reused from P4 named A/B/Mixed pulse branches, never native_source/target_native aliases or persistent schedules. Zero new factorial scientific evaluations. Fidelity only:60 natural full rollouts plus16 representative nonnative tails; every reused (M,M) baseline and familiar same-cue current output verified at1e-12 CSV roundoff. All prestate hashes verified. Every future consumes the same remaining teacher-forced tokens with physical Mixed.\n\nActual source leads: `results/output_compatible_trajectories/v1_20261111/reports/{FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT,P4_RESULTS,P2_RESULTS}.md` and its tables/manifests; `analysis/phase1k/PHASE1K_CUE_GENERATION_REPORT.md` plus `phase1k_cue_generation/interventions.py`; `results/context_cue_access/v1/{FINAL_CONTEXT_CUE_ACCESS_REPORT,S2_CUE_HISTORY_RESULTS,S3_LOCAL_CONTROL_RESULTS}.md`; `results/context_mixing_sequence_continuity/v1/{FINAL_CONTEXT_MIXING_SEQUENCE_REPORT,D0_SOURCE_AUDIT,D1_MIXING_BREAKDOWN}.md`. All available scientific predecessor seals verified. Phase1K CSVs are cloud placeholders, no observed hash invented and no substitute geometric CE. P4 contains exact locally available required outcomes. S3 cue Jacobians/CE gradients do not supply R2 state-to-logit label-free Jacobians. D1 e=j+r localized a nonlinear discrepancy but its correction was inconclusive; it is not repeated here. P2 full-right SVD and all-four-probability/roundoff audit are imported unchanged.\n\nCriterion imported unchanged: future positions k+1…7, familiar bound +.10; equality passes. Current k excluded. k1 has no history effect. Two optional legacy files are unavailable, not blockers because exact required P4 evidence exists. Scratchpad template unavailable; PRD sections used transparently.\n')
 return dict(checkpoints=20,reused_cells=360,reused_rows=1440,fidelity_only_rollouts=calls,scientific_factorial_rollouts=0,model_updates=0,passed=True)
