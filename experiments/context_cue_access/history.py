"""S2 matched final probe: equal two-parameter early, late and repeated preparation."""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from context_sequence.dynamics.closure import references
from experiments.output_compatible_trajectories.common import Stream
from experiments.output_compatible_trajectories.continuation import frozen_weight_guard
from .config import PROBES,CONTRACT
from .audit import sources
from .core import load_checkpoint
from .simulation import evaluate,schedules,source_select
from .search import optimize
from .single import replay_attempts,add_meta
from .io import save,report,identity


def parameter_count(arm):return {'N0':0,'E':2,'L':2,'R':2,'I':4}[arm]


def bindings_for(arm,k):
    if k not in [3,5]:raise ValueError('S2 final probe position must be3 or5')
    return {'N0':[],'E':[(k-3,0)],'L':[(k-2,0)],'R':[(k-3,0),(k-2,0)],'I':[(k-3,0),(k-2,2)]}[arm]


def candidate_values(arm,registry,target):
    u=registry[['u','v']].to_numpy(float);ids=registry.cue_id.tolist()
    if arm=='N0':return np.zeros((1,2)),['native_target']
    if arm!='I':return u,ids
    sobol=qmc.Sobol(d=4,scramble=True,seed=CONTRACT['sobol_seed']).random_base2(9)*3-1.5
    vals=[sobol];labels=[f'sobol_{i:03d}' for i in range(512)]
    for kind,block in [('target_u',np.c_[np.tile(target[:2],(len(u),1)),u]),('u_u',np.c_[u,u]),('u_target',np.c_[u,np.tile(target[:2],(len(u),1))])]:
        vals.append(block);labels.extend(kind+'_'+x for x in ids)
    return np.concatenate(vals),labels


def oracle_starts(arm,cues,target,best):
    base=[np.asarray(cues[x])[:2] for x in ['A','B','Mixed','NoContext']]+[np.asarray(target)[:2]]
    if arm=='I':base=[np.r_[x,x] for x in base]
    return np.stack([*base,np.asarray(best,float)])


def normalize_dimensions(frame,prefixes):
    for prefix in prefixes:
        for j in range(4):
            if f'{prefix}{j}' not in frame:frame[f'{prefix}{j}']=np.nan
    return frame


def run(root,out):
    b=sources();registry=pd.read_csv(out/'tables/input_registry.csv');native=pd.read_csv(out/'tables/native_seed.csv');prior=pd.read_csv(out/'tables/prior_p3_groups.csv')
    candidates_stream=Stream(out,'s2_candidate_schedules');histories=Stream(out,'s2_oracle_history');attempt_stream=Stream(out,'s2_oracle_attempts');selected_trace=Stream(out,'s2_selected_transition');allrows=[]
    sobol=candidate_values('I',registry,b['cues']['Mixed'])[0][:512];save(out,'s2_sobol_pairs',pd.DataFrame(sobol,columns=['early_u','early_v','late_u','late_v']))
    for number,r in enumerate(b['records'],1):
        m=load_checkpoint(root,r);a,bb,_=references(m)
        with frozen_weight_guard(m):
            for cue in PROBES:
                target=b['cues'][cue];template=np.tile(target,(7,1))
                for k in [3,5]:
                    meta={'seed':r['seed'],'cue':cue,'k':k,'condition':r['condition'],'checkpoint_sha256':r['checkpoint_sha256']}
                    nat=native[native.seed.eq(r['seed'])&native.cue.eq(cue)&native.k.eq(k)].iloc[0];pg=prior[prior.seed.eq(r['seed'])&prior.cue.eq(cue)&prior.k.eq(k)].iloc[0]
                    common={**meta,'native_future_ce':nat.future_ce,'native_viable':nat.viable_continuation,'p3_reference_status':pg.p3_reference_status,
                      'p3_oracle_positive_native_failing':bool(pg.p3_any_viable and not nat.viable_continuation) if pg.p3_reference_status=='available' else np.nan,'p3_far_viable':pg.p3_far_viable}
                    anchor=None
                    for arm in ['N0','E','L','R','I']:
                        values,ids=candidate_values(arm,registry,target);binding=bindings_for(arm,k)
                        metrics,long=evaluate(m,template,binding,values,k,(a,bb));plan=schedules(template,binding,values)
                        if metrics.prep_initial_hash.nunique()!=1:raise ValueError('S2 variable inherited anchors')
                        this_anchor=metrics.prep_initial_hash.iloc[0]
                        if anchor is None:anchor=this_anchor
                        if anchor!=this_anchor:raise ValueError('S2 arms differ in inherited anchor')
                        if not np.array_equal(plan[:,k-1:],np.broadcast_to(template[k-1:],(len(values),8-k,3))):raise ValueError('S2 changed final target or future cue')
                        metrics['candidate_id']=ids;metrics['arm']=arm;metrics['cue_parameters']=parameter_count(arm)
                        for t in range(7):
                            metrics[f'cue_{t+1}_u']=plan[:,t,0];metrics[f'cue_{t+1}_v']=plan[:,t,1]
                        metrics['intervention_type']='native_physical_cue' if arm=='N0' else 'fixed_preparatory_cue_intervention';add_meta(metrics,meta);candidates_stream.write(metrics)
                        idx=source_select(metrics);bestfixed=metrics.loc[idx].copy()
                        if arm=='N0' and abs(bestfixed.future_ce-nat.future_ce)>1e-12:raise ValueError('S2 native replay differs')
                        row={**common,**bestfixed.to_dict(),'policy':'native' if arm=='N0' else 'fixed_candidates','status':'completed','selection_future_targets':arm!='N0','sample_id':identity({**meta,'arm':arm,'policy':'fixed','candidate_id':bestfixed.candidate_id})}
                        for j in range(4):row[f'selected_{j}']=values[idx,j] if j<values.shape[1] and arm!='N0' else np.nan
                        allrows.append(row)
                        trace=long[long.candidate_index.eq(idx)].copy();add_meta(trace,{**meta,'arm':arm,'policy':row['policy'],'sample_id':row['sample_id'],'intervention_type':row['intervention_type']});selected_trace.write(trace)
                        if arm=='N0':continue
                        starts=oracle_starts(arm,b['cues'],target,values[idx]);history,attempts=optimize(m,template,binding,starts,k)
                        normalize_dimensions(history,['cue_']);add_meta(history,{**meta,'arm':arm});histories.write(history)
                        attempts,_=replay_attempts(m,template,binding,attempts,k,(a,bb));normalize_dimensions(attempts,['requested_initial_','actual_initial_','selected_']);add_meta(attempts,{**meta,'arm':arm});attempts['intervention_type']='cue_oracle_future_informed';attempt_stream.write(attempts)
                        j=source_select(attempts)
                        chosen=attempts.loc[j].copy() if j is not None else pd.Series({'future_ce':np.nan,'viable_continuation':np.nan,'status':'no_valid_oracle_start'})
                        selected={**common,**chosen.to_dict(),'policy':'bounded_oracle','arm':arm,'selection_future_targets':True,'sample_id':identity({**meta,'arm':arm,'policy':'oracle','restart':chosen.get('restart_index',-1)})}
                        allrows.append(selected)
                        if j is not None:
                            d=parameter_count(arm);values_chosen=chosen[[f'selected_{x}' for x in range(d)]].to_numpy(float)[None]
                            _,t=evaluate(m,template,binding,values_chosen,k,(a,bb));add_meta(t,{**meta,'arm':arm,'policy':'bounded_oracle','sample_id':selected['sample_id'],'intervention_type':'cue_oracle_future_informed'});selected_trace.write(t)
        print(f'S2 model {number}/20',flush=True)
    summary=pd.DataFrame(allrows);save(out,'s2_seed_summary',summary)
    paired=[]
    for (seed,cue,k),g in summary.groupby(['seed','cue','k']):
        for policy in ['fixed_candidates','bounded_oracle']:
            f=g[g.policy.eq(policy)].set_index('arm');rr=f.loc['R'];ll=f.loc['L'];ee=f.loc['E'];ii=f.loc['I']
            paired.append({'seed':seed,'cue':cue,'k':k,'policy':policy,'R_minus_L_future_ce':rr.future_ce-ll.future_ce,'E_minus_L_future_ce':ee.future_ce-ll.future_ce,'I_minus_R_future_ce':ii.future_ce-rr.future_ce,
               'R_minus_L_viability':float(rr.viable_continuation)-float(ll.viable_continuation),'R_success_L_failed':(bool(rr.viable_continuation and not ll.viable_continuation) if pd.notna(rr.viable_continuation) and pd.notna(ll.viable_continuation) else np.nan),
               'L_success_R_failed':(bool(ll.viable_continuation and not rr.viable_continuation) if pd.notna(rr.viable_continuation) and pd.notna(ll.viable_continuation) else np.nan),'I_success_R_failed':(bool(ii.viable_continuation and not rr.viable_continuation) if pd.notna(ii.viable_continuation) and pd.notna(rr.viable_continuation) else np.nan),
               'native_viable':rr.native_viable,'p3_oracle_positive_native_failing':rr.p3_oracle_positive_native_failing,'p3_reference_status':rr.p3_reference_status})
    paired=pd.DataFrame(paired);save(out,'s2_paired_effects',paired)
    if candidates_stream.n!=240*1575 or attempt_stream.n!=240*4*6 or histories.n!=240*4*486 or len(summary)!=240*9:raise ValueError('S2 fixed-budget/registry mismatch')
    report(out,'S2_CUE_HISTORY_RESULTS.md','# S2 — same-final-cue preparation\n\nEvery arm processed identical tokens from the same natural anchor (zero h0 at k3; natural-target h2 at k5), received the same physical target cue at k and the same subsequent cues, and scored only k+1…7. N0 was unmodified; E/L/R each controlled two scalars, I four. R tied the two prep inputs.\n\nAll20 seeds were evaluated, independent of S1. E/L/R shared177 fixed candidates and six starts×80 updates; I had512 fixed Sobol pairs plus531 explicit nesting controls and the same oracle evaluation/backprop budget. I has both more parameters and a larger finite menu, so its advantage is secondary. Original float32 source replay determines all CE/viability. Every attempt and per-step loss is retained.\n\nPrimary R−L at k3 is in tables/s2_paired_effects.csv (negative CE contrast favors repeated preparation); k5 is separate. Crossing viability with R when L fails supports finite-budget equal-parameter history access, not proof that all untested single cues fail or that cues are naturally selected. S3 will audit actual local control derivatives, including null outcomes.\n')
    return {'models':20,'fixed_schedule_cases':candidates_stream.n,'oracle_attempts':attempt_stream.n,'oracle_history_rows':histories.n,'seed_summary_rows':len(summary),'model_updates':0,'hidden_placements':0,'same_final_target_verified':True}
