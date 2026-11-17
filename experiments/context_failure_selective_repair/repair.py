"""R3: one natural-gap projection, fixed matched controls, and fixed reverse sign."""
import hashlib,json
import numpy as np
import pandas as pd
from experiments.output_compatible_trajectories.state_geometry import valid_state,ray_interval,assert_equivalent
from experiments.output_compatible_trajectories.nullspace import source_head_audit
from experiments.output_compatible_trajectories.continuation import continue_from,row
from .factorial import enrich,summarize
from .core import *


def fixed_policy(overrides=None):
 if overrides:raise ValueError('No outcome-driven protocol overrides in v1')
 return CONTRACT

def directions(d,U,W,N,seed,k,reference):
 du=U@(U.T@d);norm=np.linalg.norm(du);w=W@(W.T@d)
 def unit(space,kind):
  identity=json.dumps([CONTRACT['random_seed'],int(seed),int(k),reference,kind],separators=(',',':'));s=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big');v=space@np.random.default_rng(s).normal(size=space.shape[1]);n=np.linalg.norm(v)
  return v/n if n>1e-12 else np.full(len(d),np.nan),s
 if np.linalg.norm(w)<=1e-12:w,ws=unit(W,'W_fallback')
 else:w=w/np.linalg.norm(w);ws=None
 random,rs=unit(N,'random')
 return {'U':du,'W':w*norm,'random':random*norm},dict(W_fallback_rng=ws,random_rng=str(rs))

def joint_scale(anchor,vectors):
 if any(not np.isfinite(v).all() or np.linalg.norm(v)<=1e-12 for v in vectors):return np.nan,'not_identifiable'
 intervals=[ray_interval(anchor,v,CONTRACT['margin']) for v in vectors];lo=max(a for a,b in intervals);hi=min(1.,*(b for a,b in intervals));a=.95*hi
 if not np.isfinite(a) or hi<lo or a<lo-1e-12 or a<=1e-12 or not all(valid_state(anchor+a*v) for v in vectors):return np.nan,'infeasible'
 return a,'valid'

def assert_matched(vectors):
 norms=np.array([np.linalg.norm(v) for v in vectors])
 if not np.isfinite(norms).all() or norms.min()<=1e-12 or norms.max()-norms.min()>1e-10:raise ValueError('Primary displacement norms not matched')
 return True

def fixed_reverse(reference,direction,a):return np.asarray(reference)-a*np.asarray(direction)

def patch_trace(model,state,k,cue_name):
 if not valid_state(state):raise ValueError('Infeasible intervention; clipping forbidden')
 cue=CONTRACT['cue'][cue_name];t=continue_from(model,state,k,cue)
 z=np.asarray(state,np.float32)@model.output.T+model.output_bias
 # Actual float32 head for the current score; exact float64 placement separately retained.
 current=row(state,z,k,k,SEQUENCE,cue)
 for key,value in current.items():t.at[0,key]=value
 return enrich(t)

def selective_gate(stats,softmax_violations):
 if softmax_violations:return False
 for c in ['U_minus_W','U_minus_random']:
  q=stats[stats.contrast==c]
  if len(q)!=1 or int(q.n_valid.iloc[0])!=20 or not np.isfinite(q.ci_high.iloc[0]) or q.ci_high.iloc[0]>=0:return False
 return True

def seed_contrasts(frame):
 rows=[]
 for k in sorted(frame.k.unique()):
  for seed in range(20):
   f=frame[(frame.k==k)&(frame.seed==seed)];base='natural' if 'natural' in frame.arm.unique() else 'baseline'
   for name,a,b in [('U_minus_natural','U',base),('U_minus_W','U','W'),('U_minus_random','U','random'),('visible_minus_natural','visible',base),('silent_minus_natural','silent',base),('reference_minus_natural','reference',base)]:
    vals=[]
    for side in ['A','B']:
     x=f[(f.reference==side)&(f.arm==a)&(f.status=='valid')];y=f[(f.reference==side)&(f.arm==b)&(f.status=='valid')]
     if len(x)==1 and len(y)==1 and np.isfinite(x.future_ce.iloc[0]) and np.isfinite(y.future_ce.iloc[0]):vals.append(x.future_ce.iloc[0]-y.future_ce.iloc[0])
    rows.append(dict(seed=seed,k=k,contrast=name,effect=np.mean(vals) if vals else np.nan,n_references=len(vals)))
 return pd.DataFrame(rows)

def _basis(df,seed,k,name):
 f=df[(df.seed==seed)&(df.k==k)&(df.space==name)];return f.pivot(index='coordinate',columns='axis',values='value').sort_index().to_numpy()

def run(out):
 fixed_policy();nat=read(out,'r0_native_replay');fac=read(out,'r1_branch_rollouts');gaps=read(out,'r2_natural_gap_components');gaps=gaps[gaps.reference_kind=='factorial'];bases=read(out,'r2_bases');forward=[];reverse=[];familiar=[];long=[];nr=0;violations=0
 for rec in records():
  m=load_original(ROOT,rec);before=m.digest()
  with frozen_weight_guard(m):
   for gap in gaps[gaps.seed==rec['seed']].itertuples():
    seed=rec['seed'];k=gap.k;side=gap.reference;h=np.array([getattr(gap,f'anchor_{j}') for j in range(8)]);r=np.array([getattr(gap,f'reference_state_{j}') for j in range(8)]);d=r-h
    U,W,N=(_basis(bases,seed,k,g) for g in ['U','W','N']);vectors,rng=directions(d,U,W,N,seed,k,side);a,joint=joint_scale(h,[vectors[q] for q in ['U','W','random']]);matchedstatus=joint if gap.geometry_status=='identifiable' else 'not_identifiable'
    if joint=='valid':assert_matched([a*vectors[q] for q in ['U','W','random']])
    dv=np.array([getattr(gap,f'visible_{j}') for j in range(8)]);dn=np.array([getattr(gap,f'silent_{j}') for j in range(8)]);vectors.update(visible=dv,silent=dn)
    target=SEQUENCE[k];reference=refs(nat,seed);base_t=select(fac,seed,k,'Mixed','Mixed');ref_t=select(fac,seed,k,side,'Mixed')
    def evaluate(arm,state,anchor,group,ctx,status='valid',cached=None,scale=a,sign=1):
     nonlocal nr,violations
     q=dict(seed=seed,k=k,reference=side,arm=arm,group=group,context=ctx,sign=sign,amplitude=scale,joint_status=joint,geometry_status=gap.geometry_status,status=status,future_cue=ctx,n_future=7-k,n_references_planned=2,checkpoint_sha256=rec['checkpoint_sha256'],reference_original_viable=bool(gap.reference_viable),reference_interior=valid_state(r),box_margin=1-np.max(abs(state)) if state is not None else np.nan,requested_norm=np.linalg.norm(state-anchor) if state is not None else np.nan,realized_norm=np.nan,state_applied=False,current_softmax_valid=None,**rng)
     for name in ['current_ce','future_ce','worst_future_ce','full_remaining_ce','full_sequence_ce','viable_continuation','viable_strict','softmax_max_difference','source_float32_softmax_difference']:q[name]=np.nan
     if state is not None:q.update({f'state_{j}':v for j,v in enumerate(state)})
     if status=='valid':
      if cached is None and not valid_state(state):q['status']='infeasible'
      else:
       if arm in ['U','W','random','silent']:
        try:
         proof=assert_equivalent(m,anchor,state,target=target);roundoff=source_head_audit(m,anchor,state,target);q.update(proof,**roundoff);q['current_softmax_valid']=True
        except ValueError as e:
         violations+=1;q['status']='invalid_softmax';q['invalid_reason']=str(e);q['current_softmax_valid']=False
       if q['status']=='valid':
        t=cached.copy() if cached is not None else patch_trace(m,state,k,ctx)
        if cached is None:nr+=1
        t=enrich(t);q.update(summarize(t,reference));q['realized_norm']=np.linalg.norm(state-anchor);q['state_applied']=cached is None;q['observed_reused']=cached is not None
        prefix_context='Mixed' if group=='forward' else (side if group=='reverse' else ctx)
        prefix=native(nat,seed,prefix_context).raw_ce.iloc[:k-1].sum();q['full_sequence_ce']=(prefix+t.raw_ce.sum())/7
        if group=='reverse':q['prefix_contract']='natural reference-side before k, counterfactual current-Mixed and patched future'
        for key,v in dict(seed=seed,k=k,reference=side,arm=arm,group=group,context=ctx,sign=sign).items():t[key]=v
        long.append(t)
     return q
    forward.append(evaluate('natural',h,h,'forward','Mixed',cached=base_t,scale=0))
    # Exact observed R1 upper bound retained even if tanh boundary prevents strict-interior patch.
    fref=evaluate('reference',r,h,'forward','Mixed',cached=ref_t,scale=1);fref['reference_observed_boundary']=not valid_state(r);fref['strict_patch_feasible']=valid_state(r);forward.append(fref)
    for arm in ['visible','silent','U','W','random']:
     state=h+a*vectors[arm] if np.isfinite(a) else None;status=matchedstatus if arm in ['U','W','random'] else joint
     forward.append(evaluate(arm,state,h,'forward','Mixed',status=status))
    # Reverse rows remain explicit for all80 references, including originally nonviable cases.
    eligible=bool(gap.reference_viable)
    reverse.append(evaluate('baseline',r,r,'reverse','Mixed',cached=ref_t,scale=0,status='valid' if eligible else 'not_originally_viable'))
    reverse_feasible=matchedstatus=='valid' and all(valid_state(fixed_reverse(r,vectors[arm],a)) for arm in ['U','W','random'])
    for arm in ['U','W','random']:
     state=fixed_reverse(r,vectors[arm],a) if np.isfinite(a) else None
     status='not_originally_viable' if not eligible else (matchedstatus if matchedstatus!='valid' else ('valid' if reverse_feasible else 'infeasible_matched_reverse'))
     reverse.append(evaluate(arm,state,r,'reverse','Mixed',status=status,sign=-1))
    for context in ['A','B']:
     anchor=native(nat,seed,context)[H].iloc[k-1].to_numpy();cached=native(nat,seed,context).iloc[k-1:].copy();cached['lag']=cached.transition-k
     for sign in [-1,1]:
      familiar.append(evaluate('baseline',anchor,anchor,'familiar',context,cached=cached,scale=0,sign=sign))
      ok=matchedstatus=='valid' and all(valid_state(anchor+sign*a*vectors[arm]) for arm in ['U','W','random'])
      for arm in ['U','W','random']:
       state=anchor+sign*a*vectors[arm] if np.isfinite(a) else None;status=matchedstatus if matchedstatus!='valid' else ('valid' if ok else 'infeasible_matched_familiar')
       familiar.append(evaluate(arm,state,anchor,'familiar',context,status=status,sign=sign))
  if m.digest()!=before:raise ValueError('Frozen weights changed')
 f=pd.DataFrame(forward);rv=pd.DataFrame(reverse);fam=pd.DataFrame(familiar);allcon=[]
 for name,context,sign,frame in [('forward','Mixed',1,f),('reverse','Mixed',-1,rv)]+[('familiar',g,s,fam[(fam.context==g)&(fam.sign==s)]) for g in ['A','B'] for s in [-1,1]]:
  c=seed_contrasts(frame);c['group']=name;c['context']=context;c['sign']=sign;allcon.append(c)
 con=pd.concat(allcon,ignore_index=True);stats=[]
 for (group,context,sign,k,c),q in con.groupby(['group','context','sign','k','contrast']):stats.append(dict(group=group,context=context,sign=sign,k=k,contrast=c,**bootstrap(q)))
 bs=pd.DataFrame(stats);primary=bs[(bs.group=='forward')&(bs.k==3)];passed=selective_gate(primary,violations)
 cov=pd.concat([f,rv,fam],ignore_index=True).groupby(['group','context','sign','k','arm','status'],dropna=False).agg(n_reference_cells=('seed','size'),n_seeds=('seed','nunique')).reset_index()
 # Every arm also has an all20 denominator and retained per-seed0/1/2 reference coverage.
 seedcov=[]
 for (group,context,sign,k,arm),g in pd.concat([f,rv,fam]).groupby(['group','context','sign','k','arm']):
  for seed in range(20):
   q=g[g.seed==seed];good=q[q.status=='valid'];seedcov.append(dict(group=group,context=context,sign=sign,k=k,arm=arm,seed=seed,n_planned=2,n_valid=len(good),n_viable=int(good.viable_continuation.eq(True).sum()),n_strict=int(good.viable_strict.eq(True).sum()),any_viable=bool(good.viable_continuation.eq(True).any())))
 for name,df in [('r3_interventions',f),('r3_reverse',rv),('r3_familiar_controls',fam),('r3_seed_contrasts',con),('r3_bootstrap',bs),('r3_coverage',cov),('r3_seed_coverage',pd.DataFrame(seedcov)),('r3_transition_rollouts',pd.concat(long,ignore_index=True))]:save(out,name,df)
 verdict=dict(selective_mechanism_gate=passed,softmax_violations=violations,new_patch_rollouts=nr,planned_forward_cells=len(f),planned_reverse_cells=len(rv),planned_familiar_cells=len(fam),training_updates=0,optimization_steps=0,primary='Mixed k3',all20_retained=True)
 dump(out/'audit/r3_decision.json',verdict)
 report(out,'R3_SELECTIVE_REPAIR_REPORT.md','# R3 — selective natural-gap repair and fixed reverse controls\n\n'+json.dumps(verdict,indent=2)+'\n\n'+table(primary)+'\n\nDirections and amplitudes use only geometry. U=projection of actual Mixed→fixed reference gap into the two familiar-derived sensitive kernel directions. W is the complementary gap rescaled to the same norm (hash fallback only if zero); random is a hash-seeded kernel vector with the same norm. Shared a=.95×largest joint feasible scale≤1; no clipping, outcome-dependent sign, reference choice, rank or scale search. Projection arms use the same a where feasible. If no joint amplitude exists, the arms remain explicit missing outcomes, never zero CE. Exact cached natural/full-reference outcomes are shown as observed references; boundary references are not claimed strict-interior patch placements.\n\nAll actual prediction scores use source float32 recurrence/head; exact double kernel identity and float32 head roundoff audits are separate. Every silent intervention checks allfour probabilities before future scoring. Full-sequence CE uses the actual natural pre-k prefix, current score and unassisted future; it is not the primary endpoint. Reverse tests only originally viable references, fixed negative sign and same forward a; unmatched reverse feasibility is explicit. Familiar controls retain both signs and their own A/B future cue.\n\nAll20 seed rows, both reference sides, failed feasibility, adverse effects and rank/zero statuses are exported. Continuous intervals condition on estimable seeds when coverage is incomplete; this does not support a full-cohort mechanism claim. Primary selective gate needs both U−W and U−random upper95% seed limits<0, full20 estimable seeds and no current-softmax violation. Reverse evidence is required separately for bidirectionality. No useful full-state bound or raw component norm alone identifies a selective cause.\n')
 return verdict
