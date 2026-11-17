"""P1: reuse complete native lag rows; replay only missing cue schedules."""
import numpy as np
import pandas as pd
from context_sequence.dynamics.closure import references
from .config import CONTRACT,PROBES,subset
from .source_audit import load_sources,verify_checkpoint
from .continuation import continue_from,add_geometry,natural_trace,frozen_weight_guard
from .state_geometry import output_values
from .common import metadata,save,report,table,unique,bootstrap_seed,add_novel_means
from .figures import figure_A

METRICS=['raw_ce','natural_ce','raw_rescue','midpoint_hidden_distance','midpoint_task_distance']


def paired_rows(patched,natural):
    if not np.array_equal(patched.transition.to_numpy(),natural.transition.to_numpy()):raise ValueError('Mismatched future token/transition baseline')
    f=patched.copy();f['natural_ce']=natural.raw_ce.to_numpy();f['raw_rescue']=f.natural_ce-f.raw_ce
    return f


def replay_pair(model,cue,k,continuation):
    a,b,mid=references(model);sep=np.linalg.norm(a['hidden'].astype(float)-b['hidden'].astype(float),axis=1)
    natural=natural_trace(model,cue);h=natural.iloc[k-1][[f'hidden_{i}' for i in range(8)]].to_numpy(float)
    future={'D0':cue,'D1':[0,0,0],'D2':[.5,.5,0]}[continuation]
    base=continue_from(model,h,k,cue,future_cue=future)
    patched=continue_from(model,mid[k-1],k,cue,future_cue=future)
    return add_geometry(paired_rows(patched,base),model,mid,sep)


def lag_seed(frame):
    unique(frame,['cohort','condition','seed','cue','continuation','k','transition'])
    keys=['cohort','condition','seed','cue','continuation','lag']
    out=frame.groupby(keys,as_index=False)[METRICS].mean()
    counts=frame.groupby(keys,as_index=False).k.nunique().rename(columns={'k':'n_positions'})
    out=out.merge(counts,on=keys,validate='one_to_one')
    return add_novel_means(out,[k for k in keys if k!='cue'],METRICS+['n_positions'])


def run(root,out,smoke=False):
    bundle=load_sources(root);records=subset(bundle['records'],'p1',smoke);cues=bundle['cues']
    reuse=pd.read_csv(root/'results/dynamical_generalization/dg_v1_20261103/tables/e1_single_patch_replay.csv')
    # Smoke stages use the full P0 audit as their immutable reference.
    natural_path=out/'tables/p0_natural_trace.csv'
    if not natural_path.is_file():natural_path=out.parent/'tables/p0_natural_trace.csv'
    natural_source=pd.read_csv(natural_path)
    parts=[];fixed=[];aliases=[];registry=[]
    for i,r in enumerate(records,1):
        m=verify_checkpoint(root,r);meta=metadata(r);a,b,mid=references(m);sep=np.linalg.norm(a['hidden'].astype(float)-b['hidden'].astype(float),axis=1)
        old=reuse[reuse.condition.eq(r['condition'])&reuse.seed.eq(r['seed'])]
        source=natural_source[natural_source.condition.eq(r['condition'])&natural_source.seed.eq(r['seed'])]
        with frozen_weight_guard(m):
            for cue in PROBES:
                native=source[source.cue.eq(cue)].sort_values('transition')
                for k in range(1,8):
                    oldp=old[old.probe.eq(cue)&old.position.eq(k)].sort_values('transition')
                    columns=['transition','lag','token','target','raw_ce','hidden_norm'] if 'hidden_norm' in oldp else ['transition','lag','token','target','raw_ce']
                    columns += [f'hidden_{j}' for j in range(8)]+[f'logit_{j}' for j in range(4)]+[f'probability_{j}' for j in range(4)]
                    d0=oldp[columns].copy();d0['natural_ce']=native.set_index('transition').raw_ce.loc[d0.transition].values;d0['raw_rescue']=d0.natural_ce-d0.raw_ce
                    d0=add_geometry(d0,m,mid,sep)
                    current=output_values(m,mid[k-1],target=CONTRACT['sequence'][k])
                    for continuation in ['D0','D1','D2']:
                        p=d0.copy() if continuation=='D0' else replay_pair(m,cues[cue],k,continuation)
                        p['hidden_norm']=np.linalg.norm(p[[f'hidden_{j}' for j in range(8)]].to_numpy(),axis=1)
                        difference=float(np.max(np.abs(p.iloc[0][[f'probability_{j}' for j in range(4)]].to_numpy(float)-current['probabilities'])))
                        if difference>1e-10:raise ValueError('P1 immediate midpoint softmax mismatch')
                        for key,value in meta.items():p[key]=value
                        p['cue']=cue;p['k']=k;p['continuation']=continuation;p['branch']='single_midpoint';p['patch_count']=1;p['repeat_id']=0
                        p['n_future']=7-k;p['exogenous_patch']=p.lag.eq(0);p['same_output_now']=p.lag.eq(0);p['current_softmax_residual']=difference
                        p['source_reused']='frozen_E1_native_rows' if continuation=='D0' else 'source_replay_missing_schedule'
                        p['alias_of']='D0' if (cue=='Mixed' and continuation=='D2') or (cue=='NoContext' and continuation=='D1') else ''
                        parts.append(p)
                        registry.append({**meta,'stage':'p1','cue':cue,'k':k,'branch':continuation,'repeat_id':0,'status':'completed','output_path':'tables/p1_lag_long.csv'})
                        if p.alias_of.iloc[0]=='D0':
                            error=float(np.max(np.abs(p.raw_ce.to_numpy()-d0.raw_ce.to_numpy())))
                            if error>1e-10:raise ValueError('P1 continuation alias failed')
                            aliases.append({**meta,'cue':cue,'k':k,'continuation':continuation,'max_ce_difference':error,'passed':True})
                    if k in [1,3]:
                        for branch in ['natural','single_midpoint','repeated_midpoint']:
                            if branch=='single_midpoint':q=d0.copy()
                            elif branch=='natural':
                                q=native[native.transition.ge(k)].copy();q['lag']=q.transition-k;q=add_geometry(q,m,mid,sep)
                            else:q=add_geometry(continue_from(m,mid[k-1],k,cues[cue],repeated_target=mid),m,mid,sep)
                            for key,value in meta.items():q[key]=value
                            q['cue']=cue;q['k']=k;q['branch']=branch
                            q['natural_ce']=native.set_index('transition').raw_ce.loc[q.transition].values;q['raw_rescue']=q.natural_ce-q.raw_ce
                            fixed.append(q)
        if i%10==0:print(f'P1 models {i}/{len(records)}',flush=True)
    long=pd.concat(parts,ignore_index=True);seed=lag_seed(long)
    fixed=pd.concat(fixed,ignore_index=True)
    keys=['cohort','condition','seed','k','branch','transition','lag']
    fs=fixed.groupby(keys+['cue'],as_index=False)[METRICS].mean();fs=add_novel_means(fs,keys,METRICS)
    boot=bootstrap_seed(seed,METRICS,['cohort','condition','cue','continuation','lag']);boot['analysis']='available_positions_within_seed'
    fb=bootstrap_seed(fs,METRICS,['cohort','condition','cue','k','branch','lag']);fb['analysis']='fixed_k'
    for name,f in [('p1_lag_long',long),('p1_lag_seed',seed),('p1_lag_bootstrap',pd.concat([boot,fb],ignore_index=True)),('p1_fixed_k',fs),('p1_run_registry',pd.DataFrame(registry))]:save(out,name,f)
    pd.DataFrame(aliases).to_csv(out/'audit/p1_continuation_aliases.csv',index=False)
    figure_A(out,fs)
    primary=boot[boot.condition.eq('High_monotonic')&boot.cue.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])&boot.continuation.eq('D0')&boot.metric.eq('raw_rescue')&boot.lag.isin([0,1,6])]
    report(out,'P1_RESULTS.md','# P1 — what follows one state placement?\n\nWe asked how prediction and midpoint drift evolve after one external placement. We reused the complete E1 native rows, and replayed only missing zero/Mixed continuation schedules against unpatched controls with the same future cues. Model weights and teacher-forced tokens remained fixed.\n\n'+table(primary[['cue','lag','mean','ci_low','ci_high','n_seeds']])+'\n\nIntervals resample trained seeds after averaging available injection positions within seed and cue. Position support changes from seven positions at lag zero to one at lag six. Figure A therefore also shows fixed-k=1/3 trajectories, with the present external placement and repeated-patch reference explicitly labeled. All current patched softmax vectors and continuation aliases passed.\n\nThis tests state-placement persistence, not natural generation or causal mediation by midpoint distance. The remaining question is whether output-equivalent states away from the midpoint can support a different future.\n')
    return {'integrity_passed':True,'implementation_status':'validated','scientific_status':'completed','n_models':len(records),'n_lag_rows':len(long),'model_optimizer_updates':0,'native_rows_reused_from_E1':True,'legacy_optional_tables_status':'unavailable; complete E1 source used'}
