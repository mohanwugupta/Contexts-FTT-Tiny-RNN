"""P2: matched-current-distribution random interventions under frozen recurrence."""
import json
import numpy as np
import pandas as pd
from context_sequence.dynamics.closure import references
from .config import CONTRACT,PROBES,K,rng_for,subset
from .source_audit import load_sources,verify_checkpoint
from .state_geometry import output_spaces,bounded_step,fixed_radius_step,valid_state,assert_equivalent,output_values
from .continuation import continue_many,continue_from,future_summary,add_geometry,frozen_weight_guard
from .common import metadata,state_id,Stream,save,report,table,bootstrap_seed,unique


def source_head_audit(model,anchor,state,target):
    """Audit actual float32 head rounding against a priori forward-error bounds."""
    w=model.output.astype(np.float32);bias=model.output_bias.astype(np.float32)
    def values(h):
        z=(np.asarray(h,np.float32)@w.T+bias).astype(float)
        p=np.exp(z-z.max());p/=p.sum()
        return p,float(z.max()+np.log(np.exp(z-z.max()).sum())-z[target])
    p,ce=values(anchor);q,qe=values(state)
    magnitude=max(np.max(np.abs(w.astype(float))@np.abs(anchor)+np.abs(bias)),np.max(np.abs(w.astype(float))@np.abs(state)+np.abs(bias)))
    bound=float(32*np.finfo(np.float32).eps*(1+magnitude))
    dp=float(np.max(np.abs(p-q)));dc=abs(ce-qe)
    if dp>bound or dc>2*bound:raise ValueError('Float32 current head difference exceeds the frozen arithmetic-error bound')
    return {'source_float32_softmax_difference':dp,'source_float32_ce_difference':dc,'source_float32_probability_bound':bound,'source_float32_ce_bound':2*bound}


def generate_states(model,anchor,distance,seed,k,anchor_kind,smoke=False):
    space=output_spaces(model);n=space['N'];r=space['R'];rows=[]
    count=CONTRACT['p2_directions'] if anchor_kind=='midpoint' else CONTRACT['p2_natural_directions']
    radii=CONTRACT['p2_radii'] if anchor_kind=='midpoint' else CONTRACT['p2_natural_radii']
    if smoke:count=2;radii=radii[:1]
    for direction_index in range(count):
        coeff=rng_for('P2',seed,k,anchor_kind,direction_index).normal(size=n.shape[1]);coeff/=np.linalg.norm(coeff)
        direction=n@coeff
        rc=rng_for('P2-visible',seed,k,anchor_kind,direction_index).normal(size=r.shape[1])
        visible=r@rc;visible=visible/np.linalg.norm(visible) if np.linalg.norm(visible) else np.zeros(len(anchor))
        for sign in [-1,1]:
            for fraction in radii:
                q=bounded_step(anchor,sign*direction,fraction*distance)
                pair=state_id({'seed':int(seed),'k':int(k),'anchor':anchor_kind,'direction':direction_index,'sign':sign,'radius_fraction':fraction})
                base={'anchor_kind':anchor_kind,'direction_index':direction_index,'sign':sign,'radius_fraction':fraction,'target_radius':fraction*distance,'pair_id':pair,
                    'kernel_direction':(sign*coeff).tolist(),'feasible_max_radius':q['feasible_max_radius'],'minimum_feasible_radius':q['minimum_radius']}
                rows.append({**base,'kind':'nullspace','state':q['state'],'unit_direction':sign*direction,'realized_radius':q['realized_radius'],'feasible':q['feasible'],'zero_radius':q['zero_radius'],'norm_match_request':q['realized_radius']})
                v=fixed_radius_step(anchor,sign*visible,q['realized_radius'])
                rows.append({**base,'kind':'visible_control','state':v['state'],'unit_direction':sign*visible,'realized_radius':q['realized_radius'],'feasible':bool(q['feasible'] and v['feasible'] and r.shape[1]>0),'zero_radius':q['zero_radius'],'norm_match_request':q['realized_radius']})
    return pd.DataFrame(rows)


def score_states(model,anchor,candidates,k,cue,midpoints,refs):
    """Validate every current state before invoking any candidate future score."""
    f=candidates.copy().reset_index(drop=True);audits=[];valid=[];m=output_spaces(model)['M']
    target=CONTRACT['sequence'][k];anchor_output=output_values(model,anchor,target)
    # All validation precedes recurrence. Rejected proposals remain planned rows only.
    for i,r in f.iterrows():
        q={'sample_index':i,'feasible':bool(r.feasible),'passed':None,'centered_logit_residual':np.nan,'softmax_max_difference':np.nan,'immediate_ce_difference':np.nan,'argmax_equivalent':None}
        if r.feasible:
            if not valid_state(r.state):raise ValueError('A purported valid intervention leaves the fixed hidden box')
            if r.kind=='nullspace':
                q.update(assert_equivalent(model,anchor,r.state,target=target))
                q.update(source_head_audit(model,anchor,r.state,target))
            valid.append(i)
        if r.kind=='nullspace':audits.append(q)
    base=continue_many(model,np.asarray([anchor]),k,cue)
    baseline=future_summary(base,refs);nt=8-k
    defaults=['future_ce','worst_future_ce','future_ce_effect','absolute_future_ce_effect','future_softmax_max_difference','first_criterion_loss_lag','familiar_bound','worse_endpoint_bound','initial_ce','current_softmax_residual','current_centered_logit_residual']
    defaults += [f'initial_probability_{i}' for i in range(4)]
    for name in defaults:f[name]=np.nan
    for name in ['viable_continuation','viable_strict','viable_worse_endpoint']:f[name]=False
    f['n_future']=7-k;f['anchor_future_ce']=baseline['future_ce'];f['state_applied']=f.feasible
    f['relative_radius']=f.realized_radius/(np.linalg.norm(refs[0]['hidden'][k-1]-refs[1]['hidden'][k-1])+1e-8)
    f['midpoint_distance']=[float(np.linalg.norm(h-midpoints[k-1])) for h in f.state]
    distance=np.linalg.norm(refs[0]['hidden'][k-1].astype(float)-refs[1]['hidden'][k-1].astype(float))
    f['relative_midpoint_distance']=f.midpoint_distance/(distance+1e-8)
    f['far']=f.feasible & f.relative_midpoint_distance.ge(.25)
    f['box_margin']=[1-float(np.max(np.abs(h))) for h in f.state]
    f['null_residual']=[float(np.max(np.abs(m@(h-anchor)))) for h in f.state]
    f['current_visible_change']=False
    for i in valid:
        now=output_values(model,f.at[i,'state'],target)
        f.at[i,'initial_ce']=now['ce'];f.at[i,'current_softmax_residual']=np.max(np.abs(now['probabilities']-anchor_output['probabilities']))
        f.at[i,'current_centered_logit_residual']=np.max(np.abs(now['centered_logits']-anchor_output['centered_logits']))
        f.at[i,'current_visible_change']=bool(f.at[i,'current_centered_logit_residual']>1e-10)
        for j in range(4):f.at[i,f'initial_probability_{j}']=now['probabilities'][j]
    if not valid:return f,pd.DataFrame(),pd.DataFrame(audits)
    future=continue_many(model,np.stack(f.loc[valid,'state'].tolist()),k,cue)
    ce=future.raw_ce.to_numpy().reshape(len(valid),nt);prob=future[[f'probability_{i}' for i in range(4)]].to_numpy().reshape(len(valid),nt,4)
    basece=base.raw_ce.to_numpy();baseprob=base[[f'probability_{i}' for i in range(4)]].to_numpy()
    means=ce[:,1:].mean(1);end=(refs[0]['raw_ce'][k:]+refs[1]['raw_ce'][k:])/2
    worse=np.maximum(refs[0]['raw_ce'][k:],refs[1]['raw_ce'][k:]).mean()
    for j,i in enumerate(valid):
        lost=ce[j,1:]>end+.10
        values={'future_ce':means[j],'worst_future_ce':ce[j,1:].max(),'future_ce_effect':means[j]-baseline['future_ce'],'absolute_future_ce_effect':abs(means[j]-baseline['future_ce']),
            'future_softmax_max_difference':np.max(np.abs(prob[j,1:]-baseprob[1:])),
            'familiar_bound':end.mean(),'worse_endpoint_bound':worse,'viable_continuation':means[j]<=end.mean()+.10,'viable_strict':means[j]<=end.mean(),'viable_worse_endpoint':means[j]<=worse+.10,
            'first_criterion_loss_lag':int(np.flatnonzero(lost)[0]+1) if lost.any() else np.nan}
        for key,value in values.items():f.at[i,key]=value
    future['sample_index']=np.repeat(valid,nt)
    future['anchor_raw_ce']=np.tile(basece,len(valid));future['raw_ce_effect']=future.raw_ce-future.anchor_raw_ce
    future['softmax_max_difference_from_anchor']=np.max(np.abs(prob-baseprob[None,:,:]),axis=2).ravel()
    sep=np.linalg.norm(refs[0]['hidden'].astype(float)-refs[1]['hidden'].astype(float),axis=1)
    future=add_geometry(future,model,midpoints,sep)
    return f,future,pd.DataFrame(audits)


def strict_novel_means(frame,keys,metrics):
    frames=[frame]
    for name,variants in [('Novel30Mean',['Novel30_A','Novel30_B']),('Novel60Mean',['Novel60_A','Novel60_B'])]:
        f=frame[frame.cue.isin(variants)]
        if not len(f):continue
        if not f.groupby(keys,dropna=False).size().eq(2).all():raise ValueError('Incomplete paired Novel direction groups')
        # min_count retains an undefined constituent instead of silently dropping it.
        v=f.groupby(keys,dropna=False)[metrics].sum(min_count=2).div(2).reset_index();v['cue']=name;frames.append(v)
    return pd.concat(frames,ignore_index=True)


def summarize_states(states,detailed=False):
    keys=['cohort','condition','seed','cue','k','anchor_kind','kind']
    if detailed:keys+=['radius_fraction','sign']
    rows=[]
    for key,g in states.groupby(keys,dropna=False,sort=False):
        v=g[g.feasible];n=len(g)
        rows.append({**dict(zip(keys,key)),'n_planned':n,'n_valid':len(v),'n_geometry_infeasible':n-len(v),'n_zero_radius':int(g.zero_radius.sum()),
            'mean_future_ce':v.future_ce.mean(),'anchor_future_ce':g.anchor_future_ce.iloc[0],'mean_effect':v.future_ce_effect.mean(),'mean_absolute_effect':v.absolute_future_ce_effect.mean(),
            'mean_realized_relative_radius':v.relative_radius.mean(),'fraction_improved_of_planned':float((v.future_ce_effect<-.01).sum()/n),
            'fraction_worsened_of_planned':float((v.future_ce_effect>.01).sum()/n),'fraction_indistinguishable_of_planned':float((v.future_ce_effect.abs()<=.01).sum()/n),
            'fraction_viable_of_planned':float(v.viable_continuation.sum()/n),'fraction_far_viable_of_planned':float((v.viable_continuation&v.far).sum()/n),
            'fraction_far_of_planned':float(v.far.sum()/n),'mean_future_softmax_difference':v.future_softmax_max_difference.mean()})
    return pd.DataFrame(rows)


def complete_bootstrap(frame,metrics,keys):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for metric in metrics:
            if g[metric].notna().all():
                result=bootstrap_seed(g,[metric],keys)
                result['status']='complete_seed_support';result['n_seeds_with_valid_value']=len(g);rows.append(result)
            else:
                rows.append(pd.DataFrame([{**dict(zip(keys,key)),'metric':metric,'n_seeds':len(g),'mean':np.nan,'ci_low':np.nan,'ci_high':np.nan,'resampling_unit':'trained_seed','bootstrap_seed':20260925,'status':'not_estimable_for_complete_seed_cohort','n_seeds_with_valid_value':int(g[metric].notna().sum())}]))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

TRACE_COLUMNS=['cohort','condition','seed','checkpoint_sha256','cue','k','sample_id','anchor_kind','kind','direction_index','radius_fraction','sign','branch','effect_reference','is_repeated_control',
 'transition','lag','token','target','raw_ce','hidden_norm',*[f'hidden_{i}' for i in range(8)],*[f'logit_{i}' for i in range(4)],*[f'probability_{i}' for i in range(4)],*[f'cue_{i}' for i in range(3)],
 'anchor_raw_ce','raw_ce_effect','softmax_max_difference_from_anchor','midpoint_hidden_distance','midpoint_task_distance','AB_hidden_separation']


def export_states(frame,meta):
    f=frame.copy()
    for i in range(8):f[f'state_{i}']=[h[i] for h in f.state];f[f'direction_{i}']=[h[i] for h in f.unit_direction]
    f['paired_null_kernel_direction']=f.kernel_direction.map(json.dumps)
    f=f.drop(columns=['state','unit_direction','kernel_direction'])
    for key,value in meta.items():f[key]=value
    f['sample_index']=np.arange(len(f))
    f['sample_id']=[state_id({**meta,'anchor_kind':r.anchor_kind,'kind':r.kind,'direction':int(r.direction_index),'sign':int(r.sign),'radius':float(r.radius_fraction)}) for r in f.itertuples()]
    f['status']=np.where(f.feasible,np.where(f.zero_radius,'zero_radius_completed','completed'),'geometrically_infeasible_not_applied')
    return f


def run(root,out,smoke=False):
    bundle=load_sources(root);records=subset(bundle['records'],'p2',smoke)
    probes=CONTRACT['smoke']['probes'] if smoke else PROBES;positions=CONTRACT['smoke']['k'] if smoke else K
    state_stream=Stream(out,'p2_nullspace_states');future_stream=Stream(out,'p2_future_transition');audit_stream=Stream(out,'p2_invariance_audit');register_stream=Stream(out,'p2_run_registry')
    summary=[];detail=[];reference_rows=[];reconcile=[];seen=set();max_softmax=0.;max32=0.
    for model_number,r in enumerate(records,1):
        model=verify_checkpoint(root,r);a,b,mid=references(model);sep=np.linalg.norm(a['hidden'].astype(float)-b['hidden'].astype(float),axis=1)
        count=valid_count=trace_count=0
        with frozen_weight_guard(model):
            for cue in probes:
                vector=bundle['cues'][cue]
                native=__import__('context_sequence.dynamics.closure',fromlist=['natural']).natural(model,vector)
                for k in positions:
                    meta={**metadata(r),'cue':cue,'k':k}
                    midbase=continue_from(model,mid[k-1],k,vector)
                    for anchor_kind,anchor in [('midpoint',mid[k-1]),('natural',native['hidden'][k-1].astype(float))]:
                        planned=generate_states(model,anchor,sep[k-1],r['seed'],k,anchor_kind,smoke)
                        states,future,audits=score_states(model,anchor,planned,k,vector,mid,(a,b))
                        states=export_states(states,meta)
                        if seen.intersection(states.sample_id):raise ValueError('Repeated planned intervention identity')
                        seen.update(states.sample_id);unique(states,['sample_id'])
                        count+=len(states);valid_count+=int(states.feasible.sum())
                        state_stream.write(states)
                        summary.append(summarize_states(states));detail.append(summarize_states(states,detailed=True))
                        lookup=states.set_index('sample_index')
                        if len(future):
                            for key,value in meta.items():future[key]=value
                            for key in ['sample_id','anchor_kind','kind','direction_index','radius_fraction','sign']:future[key]=future.sample_index.map(lookup[key])
                            future['branch']='single_output_equivalent' if anchor_kind=='midpoint' else 'single_natural_anchor_perturbation'
                            future['effect_reference']=anchor_kind;future['is_repeated_control']=False
                            future_stream.write(future.reindex(columns=TRACE_COLUMNS));trace_count+=len(future)
                        for key,value in meta.items():audits[key]=value
                        audits['sample_id']=audits.sample_index.map(lookup.sample_id);audits['expected_equivalence']=True
                        # Fixed audit schema includes unexecuted proposals and numeric rounding checks.
                        auditcols=['cohort','condition','seed','checkpoint_sha256','cue','k','sample_id','sample_index','feasible','expected_equivalence','passed','centered_logit_residual','softmax_max_difference','immediate_ce_difference','argmax_equivalent',
                            'source_float32_softmax_difference','source_float32_ce_difference','source_float32_probability_bound','source_float32_ce_bound']
                        audit_stream.write(audits.reindex(columns=auditcols))
                        good=audits[audits.feasible]
                        if len(good):
                            if not good.passed.all():raise ValueError('P2 current-output gate failed')
                            max_softmax=max(max_softmax,float(good.softmax_max_difference.max()));max32=max(max32,float(good.source_float32_softmax_difference.max()))
                        register=states[['cohort','condition','seed','cue','k','sample_id','anchor_kind','kind','direction_index','radius_fraction','sign','status']].copy()
                        register['stage']='p2';register['branch']=register.kind;register['restart_index']=-1;register['output_path']='tables/p2_nullspace_states.csv';register_stream.write(register)
                    for branch,h,repeated in [('midpoint',mid[k-1],False),('natural_probe',native['hidden'][k-1],False),('A_endpoint',a['hidden'][k-1],False),('B_endpoint',b['hidden'][k-1],False),('repeated_midpoint',mid[k-1],True)]:
                        f=continue_from(model,h,k,vector,repeated_target=mid if repeated else None);f=add_geometry(f,model,mid,sep)
                        metrics=future_summary(f,(a,b));sample=state_id({**meta,'reference':branch})
                        reference_rows.append({**meta,'sample_id':sample,'branch':branch,'is_repeated_control':repeated,**metrics,
                            'viable_continuation':metrics['viable_continuation'] if not repeated else np.nan,
                            'single_placement_eligible':not repeated,**{f'state_{i}':float(v) for i,v in enumerate(h)}})
                        for key,value in meta.items():f[key]=value
                        f['sample_id']=sample;f['anchor_kind']='reference';f['kind']='reference_repeated' if repeated else 'reference_single';f['direction_index']=-1;f['radius_fraction']=0.;f['sign']=0;f['branch']=branch;f['effect_reference']='midpoint';f['is_repeated_control']=repeated
                        f['anchor_raw_ce']=midbase.raw_ce.to_numpy();f['raw_ce_effect']=f.raw_ce-f.anchor_raw_ce
                        f['softmax_max_difference_from_anchor']=np.max(np.abs(f[[f'probability_{j}' for j in range(4)]].to_numpy()-midbase[[f'probability_{j}' for j in range(4)]].to_numpy()),axis=1)
                        future_stream.write(f.reindex(columns=TRACE_COLUMNS));trace_count+=len(f)
        expected_per_group=16 if smoke else 240
        expected=len(probes)*len(positions)*expected_per_group
        if count!=expected:raise ValueError('P2 planned direction/sign/radius counts do not reconcile')
        reconcile.append({**metadata(r),'planned_states':count,'expected_planned_states':expected,'valid_states_scored':valid_count,'infeasible_states_retained':count-valid_count,'future_rows_including_references':trace_count,'passed':count==expected})
        print(f'P2 models {model_number}/{len(records)}; planned {count}, applicable {valid_count}',flush=True)
    pooled=pd.concat(summary,ignore_index=True);detailed=pd.concat(detail,ignore_index=True)
    basekeys=['cohort','condition','seed','k','anchor_kind','kind']
    pooled=strict_novel_means(pooled,basekeys,[c for c in pooled.select_dtypes('number') if c not in basekeys])
    detailkeys=basekeys+['radius_fraction','sign']
    detailed=strict_novel_means(detailed,detailkeys,[c for c in detailed.select_dtypes('number') if c not in detailkeys])
    save(out,'p2_primary_seed',pooled);save(out,'p2_seed_summary',detailed);save(out,'p2_reference_seed',pd.DataFrame(reference_rows));save(out,'p2_reconciliation',pd.DataFrame(reconcile))
    metrics=['mean_effect','mean_absolute_effect','fraction_viable_of_planned','fraction_far_viable_of_planned','fraction_improved_of_planned','fraction_worsened_of_planned']
    boot=complete_bootstrap(pooled,metrics,['cohort','condition','cue','k','anchor_kind','kind']);save(out,'p2_effect_bootstrap',boot)
    _plot(out,pooled)
    primary=boot[boot.condition.eq('High_monotonic')&boot.cue.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])&boot.k.eq(1)&boot.anchor_kind.eq('midpoint')&boot.kind.eq('nullspace')&boot.metric.eq('mean_absolute_effect')]
    report(out,'P2_RESULTS.md',f'# P2 — same present prediction, different future\n\nWe asked whether currently invisible hidden information affects future predictions. We changed only one bounded post-update state along the centered output kernel, then used identical teacher-forced future inputs and the actual frozen recurrence. All {state_stream.n:,} planned null/visible-control states were retained, including geometrically infeasible proposals. No model parameter changed.\n\nMaximum complete current-softmax discrepancy in float64: {max_softmax:.4g}. Original float32-head discrepancy: {max32:.4g}, checked against an a priori machine-roundoff bound; this rounding audit is distinct from the exact centered-map identity.\n\nPrimary k=1 absolute future CE changes, averaged within seed over fixed random interventions:\n\n'+table(primary[['cue','mean','ci_low','ci_high','n_seeds','n_seeds_with_valid_value','status']])+'\n\nInference resamples trained seeds; random directions are repeated measures. Raw signed effects, both radii/signs, realized radii, complete future probabilities, natural-anchor controls and norm-matched visible controls remain in the tables. Where a seed has no admissible direction, its missing CE is explicit and a complete-cohort CE interval is not estimated. Success fractions over planned interventions include invalid proposals as non-successes, not fabricated zero losses.\n\nThis is a causal hidden-state intervention inside the SRN, not evidence that natural cues retrieve these states. The remaining question is whether a bounded, far alternative sustains viable continuation across the whole remaining horizon.\n')
    return {'integrity_passed':True,'implementation_status':'validated','scientific_status':'completed','n_models':len(records),'planned_states':state_stream.n,'future_rows':future_stream.n,'max_current_softmax_difference_float64':max_softmax,'max_current_softmax_difference_float32':max32,'model_optimizer_updates':0}


def _plot(out,summary):
    import matplotlib.pyplot as plt
    from .figures import save_figure
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    f=summary[summary.condition.eq('High_monotonic')&summary.k.eq(1)&summary.anchor_kind.eq('midpoint')&summary.cue.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
    for cue,g in f.groupby('cue'):
        n=g[g.kind.eq('nullspace')];axes[0].scatter(n.anchor_future_ce,n.mean_future_ce,s=22,alpha=.7,label=cue)
        axes[1].scatter(n.mean_realized_relative_radius,n.mean_absolute_effect,s=22,alpha=.7,label=cue)
    bounds=axes[0].get_xlim();axes[0].plot(bounds,bounds,color='gray',ls='--')
    axes[0].set(xlabel='Midpoint future CE (nats)',ylabel='Mean random-null-state future CE',title='k=1; six unpatched future predictions')
    axes[1].set(xlabel='Mean realized radius / familiar separation',ylabel='Mean absolute future CE change',title='One point per seed and cue family');axes[1].legend(fontsize=8)
    save_figure(fig,out,'fig_p2_identical_now_different_later')
