"""S0 source provenance, native replay and exact imported P3 viability."""
from pathlib import Path
import inspect
import json
import numpy as np
import pandas as pd
from context_sequence.provenance.hashing import sha256
from context_sequence.dynamics.closure import references
from experiments.output_compatible_trajectories.source_audit import load_sources
from experiments.output_compatible_trajectories.verify import run as verify_prior
from experiments.output_compatible_trajectories.continuation import future_summary,frozen_weight_guard
from .config import ROOT,PRIOR,CONTRACT,SEQUENCE,PROBES
from .core import load_checkpoint,source_trace,metrics
from .io import save,dump,report,table


def sources():
    b=load_sources(ROOT);b['records']=sorted([r for r in b['records'] if r['condition']=='High_monotonic'],key=lambda r:r['seed'])
    if len(b['records'])!=20 or [r['seed'] for r in b['records']]!=list(range(20)):raise ValueError('blocked_missing_source: canonical High cohort identity')
    return b


def frozen_inputs():
    verify_prior(PRIOR)
    prior=json.loads((PRIOR/'manifest.json').read_text());paths=set(prior['code_hashes'])
    paths.update(x['path'] for x in prior['source_config_checkpoint_identities'])
    for p in ['manifest.json','tables/p0_natural_trace.csv','tables/p3_oracle_restart_summary.csv','tables/p3_oracle_best_seed.csv','tables/p4_cue_access_seed.csv',
       'reports/FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md','reports/RESEARCHER_HANDOFF.md',*[f'reports/{x}' for x in ['P0_SOURCE_AUDIT.md','P1_RESULTS.md','P2_RESULTS.md','P3_RESULTS.md','P4_RESULTS.md']]]:paths.add(str((PRIOR/p).relative_to(ROOT)))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'docs/context_cue_access').glob('*.md'))
    paths.update(['results/dynamical_generalization/dg_v1_20261103/FINAL_DYNAMICAL_GENERALIZATION_REPORT.md','results/dynamical_generalization/dg_v1_20261103/E2_RESULTS.md'])
    result={}
    for path in sorted(paths):
        p=ROOT/path
        if not p.is_file():raise FileNotFoundError('blocked_missing_source: '+path)
        result[path]=sha256(p)
    return result


def domain(u,v):
    tol=CONTRACT['domain_tolerance'];norm=float(np.hypot(u,v))
    return {'u':float(u),'v':float(v),'w':0.,'alpha':float(u-v),'beta':float(u+v),'cue_norm':norm,'normalized_cue_norm':norm,
      'familiar_line':bool(u>=-tol and v>=-tol and abs(u+v-1)<=tol),'familiar_square':bool(-tol<=u<=1+tol and -tol<=v<=1+tol),
      'extended_box':bool(-1.5-tol<=u<=1.5+tol and -1.5-tol<=v<=1.5+tol),'signed_extrapolation':bool(u<0 or v<0)}


def input_registry(cues):
    rows=[]
    for i,u in enumerate(np.linspace(-1.5,1.5,13)):
        for j,v in enumerate(np.linspace(-1.5,1.5,13)):rows.append({'cue_id':f'grid_{i:02d}_{j:02d}','kind':'grid','anchor_name':'',**domain(u,v)})
    for name,vector in cues.items():
        if vector[2]!=0:raise ValueError('Frozen cue has active third coordinate')
        rows.append({'cue_id':'anchor_'+name,'kind':'exact_anchor','anchor_name':name,**domain(*vector[:2])})
    return pd.DataFrame(rows)


def prior_groups():
    f=pd.read_csv(PRIOR/'tables/p3_oracle_restart_summary.csv');rows=[]
    for (seed,cue,k),g in f.groupby(['seed','cue','k']):
        valid=g[g.feasible];rows.append({'seed':seed,'cue':cue,'k':k,'p3_reference_status':'available' if len(valid) else 'no_feasible_prior_reference',
          'p3_any_viable':bool(valid.viable_continuation.any()),'p3_far_viable':bool(valid.far_viable.any()),'p3_far_viable_midpoint_failed':bool(valid.far_viable_midpoint_failed.any()),
          'p3_viable_state_ids':'|'.join(valid.loc[valid.viable_continuation,'sample_id'].tolist())})
    return pd.DataFrame(rows)


def run(root,out):
    b=sources();registry=input_registry(b['cues']);save(out,'input_registry',registry)
    old=pd.read_csv(PRIOR/'tables/p4_cue_access_seed.csv');old=old[old.branch.eq('target_native')&old.schedule.eq('pulse')&old.source_context.eq(old.cue)]
    historical=pd.read_csv(PRIOR/'tables/p0_natural_trace.csv');parts=[];checks=[];native=[];modelrows=[]
    for r in b['records']:
        m=load_checkpoint(root,r);before=m.digest()
        if m.wih.shape!=(8,7) or m.output.shape!=(4,8) or m.whh.shape!=(8,8) or np.any(m.wih[:,6]!=0):raise ValueError('blocked_source_variant: canonical shape/third channel mismatch')
        a,bb,_=references(m)
        with frozen_weight_guard(m):
            for cue in PROBES:
                t=source_trace(m,np.tile(b['cues'][cue],(7,1)));known=historical[historical.condition.eq('High_monotonic')&historical.seed.eq(r['seed'])&historical.cue.eq(cue)]
                for prefix,n in [('hidden_',8),('logit_',4),('probability_',4)]:
                    col=[prefix+str(i) for i in range(n)];error=np.abs(t[col].to_numpy()-known[col].to_numpy()).max()
                    if error>1e-12:raise ValueError('blocked_replay_mismatch '+prefix)
                    checks.append({'seed':r['seed'],'cue':cue,'field':prefix,'max_abs_error':error,'passed':True})
                for k in [1,3,5]:
                    q=metrics(t,k,(a,bb));known=old[old.seed.eq(r['seed'])&old.cue.eq(cue)&old.k.eq(k)]
                    if len(known)!=1 or abs(q['future_ce']-float(known.future_ce.iloc[0]))>1e-12 or q['viable_continuation']!=bool(known.viable_continuation.iloc[0]):raise ValueError('blocked_metric_contract: P4 native disagreement')
                    native.append({'seed':r['seed'],'condition':r['condition'],'checkpoint_sha256':r['checkpoint_sha256'],'cue':cue,'k':k,**q})
                for key,value in {'seed':r['seed'],'cue':cue,'checkpoint_sha256':r['checkpoint_sha256']}.items():t[key]=value
                parts.append(t)
        modelrows.append({**r,'state_dict_digest':before,'weights_unchanged':m.digest()==before,'third_channel_inactive':True,'next_token_output_only':True})
    save(out,'native_replay',pd.concat(parts,ignore_index=True));save(out,'native_seed',pd.DataFrame(native));save(out,'source_replay_checks',pd.DataFrame(checks));save(out,'checkpoint_registry',pd.DataFrame(modelrows))
    save(out,'prior_p3_groups',prior_groups())
    index=[]
    for k in [1,3,5]:
        for t in range(1,8):index.append({'k':k,'transition':t,'token':SEQUENCE[t-1],'target':SEQUENCE[t],'prestate_index':t-1,'poststate_index':t,'included_in_future_ce':t>k,
            's2_role':'outside_history_window' if k==1 or t<k-2 else ('prep_early' if t==k-2 else 'prep_late' if t==k-1 else 'fixed_final_target' if t==k else 'fixed_future_target')})
    save(out,'index_contract',pd.DataFrame(index));dump(out/'audit/source_replay.json',{'passed':True,'models':20,'native_cases':360,'metric_function':inspect.getsourcefile(future_summary),'metric_line':inspect.getsourcelines(future_summary)[1],
       'formula':'future_ce <= mean((A.raw_ce[k:]+B.raw_ce[k:])/2)+0.10','position_specific_reference':True,'model_updates':0,'hidden_placements':0})
    dump(out/'manifests/config.json',CONTRACT)
    report(out,'INDEXING_DIAGRAM.md','# Fixed sequence indexing\n\n'+table(pd.DataFrame(index))+'\n\nS1: natural h_(k−1) → [token k, candidate cue] → h_k → fixed target-cue future; score k+1…7.\n\nS2 k3: same zero h0 → token1/prep early → token2/prep late → token3/fixed target → score4…7.\n\nS2 k5: same natural-target h2 → token3/prep early → token4/prep late → token5/fixed target → score6…7. No current/prep CE enters future viability.\n')
    report(out,'S0_SOURCE_AUDIT.md','# S0 — frozen source, indexing and viability\n\nWe audited the canonical High seeds0–19 used by P3/P4, including the historical monotonic head but scoring only next-token logits. All have eight tanh states, four logits, four token inputs and three cue inputs, with third-cue weights exactly zero. Checkpoint and prior sealed identities passed; no source, model or historical output was modified. Native states, logits and complete probabilities reproduce P0; all360 target/position native future losses and viability decisions reproduce P4 within1e-12 CSV roundoff.\n\nThe viability function is imported unchanged from `experiments/output_compatible_trajectories/continuation.py:future_summary`: future mean CE ≤ mean((A.raw_ce[k:]+B.raw_ce[k:])/2)+0.10 nats. The familiar reference is position-specific and excludes current/preparatory steps. Strict-boundary tests pass.\n\nEvery rollout starts at zero and calls the original source step; k5 history arms share the natural-target h2, and k3 arms share zero h0. The indexing diagram and per-step registry define tokens, states and targets exactly. This establishes faithful frozen evaluation, not successful cue access. S1 may proceed after its tests pass.\n')
    return {'models':20,'native_cases':360,'model_updates':0,'hidden_placements':0,'metric_contract':'exact_imported_P3','native_replay_passed':True}
