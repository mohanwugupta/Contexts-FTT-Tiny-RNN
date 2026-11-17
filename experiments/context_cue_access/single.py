"""S1 single current cue, identical natural prestate and future target inputs."""
import numpy as np
import pandas as pd
from context_sequence.dynamics.closure import references
from experiments.output_compatible_trajectories.common import Stream
from experiments.output_compatible_trajectories.continuation import frozen_weight_guard
from .config import PROBES,CONTRACT
from .audit import sources
from .core import load_checkpoint
from .simulation import evaluate,source_select
from .search import optimize
from .io import save,report,table,identity


def pulse_bindings(k):return [(k-1,0)]


def replay_attempts(model,template,bindings,attempts,k,refs):
    attempts=attempts.copy();cols=[f'selected_{i}' for i in range(int(attempts.cue_parameters.iloc[0]))]
    valid=np.isfinite(attempts[cols].to_numpy()).all(1);traces=[]
    for i in attempts.index[valid]:
        f,t=evaluate(model,template,bindings,attempts.loc[[i],cols].to_numpy(),k,refs)
        for name,value in f.iloc[0].items():
            if name!='candidate_index':attempts.loc[i,name]=value
        t['restart_index']=attempts.loc[i,'restart_index'];traces.append(t)
    if 'future_ce' not in attempts:attempts['future_ce']=np.nan
    attempts['source_minus_surrogate_ce']=attempts.future_ce-attempts.best_smooth_loss
    return attempts,pd.concat(traces,ignore_index=True) if traces else pd.DataFrame()


def add_meta(frame,meta):
    for key,value in meta.items():frame[key]=value
    return frame


def run(root,out):
    b=sources();registry=pd.read_csv(out/'tables/input_registry.csv');native=pd.read_csv(out/'tables/native_seed.csv');prior=pd.read_csv(out/'tables/prior_p3_groups.csv')
    states=Stream(out,'s1_grid_seed');traces=Stream(out,'s1_grid_transition');histories=Stream(out,'s1_oracle_history');attempt_stream=Stream(out,'s1_oracle_attempts');oracle_trace=Stream(out,'s1_oracle_transition');bestrows=[]
    for number,r in enumerate(b['records'],1):
        m=load_checkpoint(root,r);a,bb,_=references(m)
        with frozen_weight_guard(m):
            for cue in PROBES:
                template=np.tile(b['cues'][cue],(7,1))
                for k in [1,3,5]:
                    meta={'seed':r['seed'],'cue':cue,'k':k,'condition':r['condition'],'checkpoint_sha256':r['checkpoint_sha256']};bindings=pulse_bindings(k)
                    summary,long=evaluate(m,template,bindings,registry[['u','v']].to_numpy(),k,(a,bb))
                    if summary.prestate_hash.nunique()!=1:raise ValueError('S1 candidate prestates differ')
                    f=pd.concat([registry.reset_index(drop=True),summary],axis=1);add_meta(f,meta);f['intervention_type']='fixed_cue_intervention';f['selection_future_targets']=False
                    f['historical_p4_menu']=f.kind.eq('exact_anchor');states.write(f)
                    long['cue_id']=long.candidate_index.map(f.cue_id);long['intervention_type']='fixed_cue_intervention';add_meta(long,meta);traces.write(long)
                    grid=f[f.kind.eq('grid')];best_grid=grid.loc[source_select(grid)];starts=np.array([b['cues'][x][:2] for x in ['A','B','Mixed','NoContext']]+[b['cues'][cue][:2],best_grid[['u','v']].to_numpy(float)])
                    history,attempts=optimize(m,template,bindings,starts,k);add_meta(history,meta);histories.write(history)
                    attempts,t=replay_attempts(m,template,bindings,attempts,k,(a,bb));add_meta(attempts,meta);attempts['intervention_type']='cue_oracle_future_informed';attempt_stream.write(attempts)
                    if len(t):add_meta(t,meta);t['intervention_type']='cue_oracle_future_informed';oracle_trace.write(t)
                    nat=native[native.seed.eq(r['seed'])&native.cue.eq(cue)&native.k.eq(k)].iloc[0]
                    pg=prior[prior.seed.eq(r['seed'])&prior.cue.eq(cue)&prior.k.eq(k)].iloc[0]
                    subgroup=bool(pg.p3_any_viable and not nat.viable_continuation) if pg.p3_reference_status=='available' else np.nan
                    common={**meta,'native_future_ce':nat.future_ce,'native_viable':nat.viable_continuation,'p3_reference_status':pg.p3_reference_status,'p3_oracle_positive_native_failing':subgroup,'p3_far_viable':pg.p3_far_viable}
                    selections=[('native',f[f.anchor_name.eq(cue)].iloc[0])]
                    for label,col in [('best_familiar_line','familiar_line'),('best_familiar_square','familiar_square'),('best_extended_grid','extended_box')]:
                        domain_grid=grid[grid[col]];selections.append((label,domain_grid.loc[source_select(domain_grid)]))
                    idx=source_select(attempts);selected=attempts.loc[idx] if idx is not None else pd.Series({'future_ce':np.nan,'viable_continuation':np.nan,'status':'no_valid_oracle_start'})
                    selected=selected.copy();selected['u']=selected.get('selected_0',np.nan);selected['v']=selected.get('selected_1',np.nan);selections.append(('bounded_oracle',selected))
                    for policy,row in selections:
                        result={**common,**row.to_dict(),'policy':policy,'selection_future_targets':policy!='native','intervention_type':'native_physical_cue' if policy=='native' else 'target_informed_cue_selection'}
                        result.update(meta);result['status']=row.get('status','completed');result['sample_id']=identity({**meta,'policy':policy,'u':result.get('u'),'v':result.get('v')});bestrows.append(result)
        print(f'S1 model {number}/20',flush=True)
    best=pd.DataFrame(bestrows);save(out,'s1_best_cues',best)
    if states.n!=20*6*3*177 or attempt_stream.n!=20*6*3*6 or histories.n!=20*6*3*486:raise ValueError('S1 budget/registry mismatch')
    counts=best[best.k.eq(1)].groupby(['cue','policy'],as_index=False).agg(n_seeds=('seed','nunique'),viable=('viable_continuation','sum'),mean_future_ce=('future_ce','mean'));save(out,'s1_primary_counts',counts)
    report(out,'S1_SINGLE_CUE_RESULTS.md','# S1 — current-cue access\n\nWe changed only the cue at k, holding the natural prestate, token and all future target cues fixed. All20 canonical High seeds received169 grid candidates and eight exact anchors at k1/3/5. Each case also received six future-target-informed cue starts and80 updates, with every selected cue replayed through the original source recurrence. No hidden state was placed or trained.\n\nPrimary k1 counts/CE are in tables/s1_primary_counts.csv; all domain partitions and subgroup denominators remain in s1_best_cues.csv. Fixed-grid and oracle discoveries are selected using future performance and do not establish natural retrieval. Extended signed-box cues are mathematical inputs, not necessarily plausible stimuli. The remaining question is whether preparation helps with the same final physical target cue; S2 runs for all seeds regardless of S1 success.\n')
    return {'models':20,'grid_cases':states.n,'oracle_attempts':attempt_stream.n,'oracle_history_rows':histories.n,'model_updates':0,'hidden_placements':0}
