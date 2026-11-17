"""D1 exact nonlinear mixing/propagation, with observed float32 rounding explicit."""
import numpy as np
import pandas as pd
from experiments.output_compatible_trajectories.continuation import row,frozen_weight_guard
from .config import CONTRACT,ROOT,SEQUENCE
from .source_adapter import records,load_original
from .replay import natural_trace
from .statistics import seed_bootstrap
from .io import save,report,table


def validate_identity(e,j,r,tolerance=1e-11):
    if not np.isfinite(np.r_[e,j,r]).all() or np.max(np.abs(e-j-r))>tolerance:raise ValueError('Exact mixing identity failed')
    return True


def exact_components(aA,aB,propagated_input,activation=np.tanh):
    bar=(aA+aB)/2;mid=(activation(aA)+activation(aB))/2
    j=activation(bar)-mid;r=activation(bar+propagated_input)-activation(bar);e=activation(bar+propagated_input)-mid
    validate_identity(e,j,r)
    return {'j':j,'r':r,'e':e}


def decompose(model):
    a=natural_trace(model,[1,0,0]);b=natural_trace(model,[0,1,0]);m=natural_trace(model,[.5,.5,0])
    cols=[f'hidden_{i}' for i in range(8)];ha=a[cols].to_numpy();hb=b[cols].to_numpy();hm=m[cols].to_numpy();mid=(ha+hb)/2
    pre=[np.r_[np.zeros((1,8)),h[:-1]] for h in [ha,hb,hm]];w=model.output.astype(float);pw=w-w.mean(0,keepdims=True);wh=model.whh.astype(float);rows=[]
    for t,token in enumerate(SEQUENCE[:-1]):
        aa=a[[f'preactivation64_{i}' for i in range(8)]].iloc[t].to_numpy(float);ab=b[[f'preactivation64_{i}' for i in range(8)]].iloc[t].to_numpy(float);am=m[[f'preactivation64_{i}' for i in range(8)]].iloc[t].to_numpy(float)
        previous=pre[2][t]-(pre[0][t]+pre[1][t])/2;q=exact_components(aa,ab,wh@previous);ideal=np.tanh(am)-(np.tanh(aa)+np.tanh(ab))/2;actual=hm[t]-mid[t]
        affine=float(np.max(np.abs(am-(aa+ab)/2-wh@previous)))
        identity=float(np.max(np.abs(ideal-q['j']-q['r'])))
        if affine>1e-11 or identity>1e-11:raise ValueError('Exact affine/tanh identity failed')
        bounds=[]
        for previous_h,cue in zip(pre,[[1,0,0],[0,1,0],[.5,.5,0]]):
            inp=np.r_[np.eye(4)[token],cue];magnitude=np.abs(model.wih.astype(float))@np.abs(inp)+np.abs(wh)@np.abs(previous_h[t])+np.abs(model.bias.astype(float))
            bounds.append(64*np.finfo(np.float32).eps*(1+magnitude))
        bound=float(np.max(bounds[2]+(bounds[0]+bounds[1])/2));observed=float(np.max(np.abs(actual-q['j']-q['r'])))
        if observed>bound:raise ValueError('Recorded float32 deviation exceeds declared forward-rounding bound')
        target=SEQUENCE[t+1];z=m[[f'logit_{i}' for i in range(4)]].iloc[t].to_numpy(float);z[target]=-np.inf;competitor=int(z.argmax());margin_direction=w[target]-w[competitor]
        vectors={'j':q['j'],'r':q['r'],'e':actual,'e_ideal':ideal};data={'transition':t+1,'token':token,'target':target,'competitor':competitor,'mixed_ce':m.raw_ce.iloc[t],'A_ce':a.raw_ce.iloc[t],'B_ce':b.raw_ce.iloc[t],
            'familiar_ce':(a.raw_ce.iloc[t]+b.raw_ce.iloc[t])/2,'mixed_excess_ce':m.raw_ce.iloc[t]-(a.raw_ce.iloc[t]+b.raw_ce.iloc[t])/2,
            'affine_residual':affine,'hidden_identity_residual':identity,'output_identity_residual':float(np.max(np.abs(pw@(ideal-q['j']-q['r'])))),
            'observed_float32_residual':observed,'roundoff_bound':bound,'observed_task_roundoff_residual':float(np.max(np.abs(pw@(actual-q['j']-q['r']))))}
        for name,v in vectors.items():
            data[name+'_hidden_norm']=float(np.linalg.norm(v));data[name+'_task_norm']=float(np.linalg.norm(pw@v));data[name+'_signed_target_margin']=float(margin_direction@v)
            data.update({f'{name}_hidden_{j}':float(x) for j,x in enumerate(v)});data.update({f'{name}_task_{j}':float(x) for j,x in enumerate(pw@v)})
        rows.append(data)
    return pd.DataFrame(rows)


def corrected_rollout(model,k):
    if k not in range(1,7):raise ValueError('Registered corrections k1…6 only')
    a=natural_trace(model,[1,0,0]);b=natural_trace(model,[0,1,0]);mid=(a[[f'hidden_{i}' for i in range(8)]].to_numpy()+b[[f'hidden_{i}' for i in range(8)]].to_numpy())/2
    previous=np.zeros(8,np.float32);rows=[];cue=np.array([.5,.5,0.])
    for t,token in enumerate(SEQUENCE[:-1]):
        h,z=model.step(token,cue,previous)
        if t==k-1:h=mid[t].copy();z=model.output.astype(float)@h+model.output_bias.astype(float)
        f=row(h,z,t+1,k,SEQUENCE,cue);f.update(patch_k=k,exogenous_assistance=t==k-1,patch_count=1,intervention_type='one_time_midpoint_state_correction')
        idealz=model.output.astype(float)@mid[t]+model.output_bias.astype(float);delta=np.asarray(z,float)-idealz;f['centered_logit_midpoint_distance']=np.linalg.norm(delta-delta.mean());f['raw_logit_midpoint_distance']=np.linalg.norm(delta)
        rows.append(f);previous=h
    return pd.DataFrame(rows)


def ensembles(model):
    a=natural_trace(model,[1,0,0]);b=natural_trace(model,[0,1,0]);hcols=[f'hidden_{i}' for i in range(8)];pcols=[f'probability_{i}' for i in range(4)]
    ha=a[hcols].to_numpy();hb=b[hcols].to_numpy();w=model.output.astype(float);bias=model.output_bias.astype(float)
    za=ha@w.T+bias;zb=hb@w.T+bias;mid=(ha+hb)/2@w.T+bias;avg=(za+zb)/2;pm=(a[pcols].to_numpy()+b[pcols].to_numpy())/2
    def softmax(z):
        p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)
    pl=softmax(avg);ph=softmax(mid);target=np.array(SEQUENCE[1:]);actual_avg=(a[[f'logit_{i}' for i in range(4)]].to_numpy()+b[[f'logit_{i}' for i in range(4)]].to_numpy())/2
    f=pd.DataFrame({'transition':np.arange(1,8),'target':target,'probability_ensemble_ce':-np.log(pm[np.arange(7),target]),'logit_average_ce':-np.log(pl[np.arange(7),target]),'midpoint_readout_ce':-np.log(ph[np.arange(7),target]),
      'affine_logit_error':np.max(np.abs(mid-avg),axis=1),'recorded_float32_head_roundoff':np.max(np.abs(actual_avg-avg),axis=1),'probability_vs_logit_average_difference':np.max(np.abs(pm-pl),axis=1),'externally_constructed':True})
    for j in range(4):f[f'probability_average_{j}']=pm[:,j];f[f'logit_average_probability_{j}']=pl[:,j]
    if f.affine_logit_error.max()>1e-11:raise ValueError('Affine midpoint/head identity failed')
    return f


def run(root,out):
    decomposition=[];patches=[];baselines=[];summary=[];local=[]
    old=pd.read_csv(ROOT/'results/dynamical_generalization/dg_v1_20261103/tables/e1_single_patch_replay.csv');checks=[]
    for r in records():
        model=load_original(root,r)
        with frozen_weight_guard(model):
            f=decompose(model);e=ensembles(model);natural=natural_trace(model,[.5,.5,0]);f['seed']=r['seed'];f['checkpoint_sha']=r['checkpoint_sha256'];f['config_sha']=CONTRACT['sha256'];decomposition.append(f);e['seed']=r['seed'];baselines.append(e)
            for k in range(1,7):
                p=corrected_rollout(model,k);p['seed']=r['seed'];p['natural_mixed_ce']=natural.raw_ce.to_numpy();p['raw_rescue']=p.natural_mixed_ce-p.raw_ce;p['checkpoint_sha']=r['checkpoint_sha256'];patches.append(p)
                known=old[old.condition.eq('High_monotonic')&old.seed.eq(r['seed'])&old.probe.eq('Mixed')&old.position.eq(k)].sort_values('transition');actual=p[p.transition.ge(k)]
                error=float(np.max(np.abs(actual.raw_ce.to_numpy()-known.raw_ce.to_numpy())))
                if error>1e-10:raise ValueError('Prior exact single-patch replay mismatch')
                checks.append({'seed':r['seed'],'patch_k':k,'max_abs_ce_error':error,'passed':True})
                summary.append({'seed':r['seed'],'patch_k':k,'n_future':7-k,'natural_future_ce':float(natural.raw_ce.iloc[k:].mean()),'corrected_future_ce':float(p.raw_ce.iloc[k:].mean()),'correction_minus_natural_future_ce':float(p.raw_ce.iloc[k:].mean()-natural.raw_ce.iloc[k:].mean())})
            a=natural_trace(model,[1,0,0]);b=natural_trace(model,[0,1,0]);mid=(a[[f'hidden_{i}' for i in range(8)]].to_numpy()+b[[f'hidden_{i}' for i in range(8)]].to_numpy())/2
            for t,token in enumerate(SEQUENCE[:-1]):
                pre=np.zeros(8) if t==0 else mid[t-1];h,z=model.step(token,[.5,.5,0],pre);q=row(h,z,t+1,1,SEQUENCE,[.5,.5,0]);q.update(seed=r['seed'],diagnostic='one_step_from_familiar_midpoint_prestate_not_natural_trajectory');local.append(q)
    long=pd.concat(decomposition,ignore_index=True);p=pd.concat(patches,ignore_index=True);e=pd.concat(baselines,ignore_index=True);s=pd.DataFrame(summary)
    for name,frame in [('d1_decomposition',long),('d1_correction_transition',p),('d1_ensemble_baselines',e),('d1_correction_seed',s),('d1_local_no_prior_deviation',pd.DataFrame(local)),('d1_prior_patch_replay_checks',pd.DataFrame(checks))]:save(out,name,frame)
    boot=seed_bootstrap(s,['correction_minus_natural_future_ce','natural_future_ce','corrected_future_ce'],['patch_k']);save(out,'d1_correction_intervals',boot)
    primary=boot[boot.patch_k.eq(1)&boot.metric.eq('correction_minus_natural_future_ce')]
    report(out,'D1_MIXING_BREAKDOWN.md','# D1 — nonlinear mixing and inherited deviation\n\nAt the observed source prestates we separated new nonlinear mismatch j=tanh(mean familiar preactivation)−mean(tanh familiar preactivation) from propagation r=tanh(mean preactivation+W_h e_previous)−tanh(mean preactivation). Their double-arithmetic identity is exact to declared1e-11; the difference from recorded float32 states is bounded and exported, never attributed to a mechanism. First-step propagation is exactly zero.\n\nMaximum double hidden identity residual='+str(long.hidden_identity_residual.max())+'; maximum observed float32 residual='+str(long.observed_float32_residual.max())+'. Output norms and signed target-margin projections are separate, not additive CE explanations.\n\nThe fixed first-position correction changed the remaining six-step mean CE (corrected−natural; negative improves):\n\n'+table(primary[['mean','ci_low','ci_high','n_seeds']])+'\n\nAll later k2…6 corrections and failed/adverse seeds remain. All120 correction cases matched the prior E1 source. Probability averaging and affine midpoint/logit averaging are external baselines, not natural recurrent computations. This identifies an early numerical mismatch and tests the effect of one initial correction; it does not prove that initial mixing explains every later error or that midpoint is necessary.\n')
    return {'models':20,'decomposition_rows':len(long),'correction_rows':len(p),'max_exact_hidden_residual':float(long.hidden_identity_residual.max()),'max_observed_float32_residual':float(long.observed_float32_residual.max()),'optimizer_steps':0}
