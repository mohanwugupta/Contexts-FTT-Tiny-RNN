"""P3: bounded future-target-informed hidden-state oracle; never model training."""
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from context_sequence.dynamics.closure import references
from .config import CONTRACT,PROBES,K,SEQUENCE,rng_for,subset
from .state_geometry import output_spaces,bounded_step,valid_state,assert_equivalent,output_values
from .continuation import continue_from,future_summary,add_geometry,frozen_weight_guard
from .source_audit import load_sources,verify_checkpoint
from .common import metadata,state_id,Stream,save,report,table
from .nullspace import source_head_audit,strict_novel_means,complete_bootstrap


class FrozenObjective:
    """Smooth double recurrence with copied constant source weights and no Parameters."""
    def __init__(self,model,midpoint,N,k,cues):
        self.n_future=7-k
        if self.n_future<1:raise ValueError('No oracle inference at k=7')
        self.future_targets=torch.tensor(SEQUENCE[k+1:],dtype=torch.long)
        self.N=torch.tensor(N,dtype=torch.float64);self.mid=torch.tensor(midpoint,dtype=torch.float64)
        self.wh=torch.tensor(model.whh.astype(float));self.wo=torch.tensor(model.output.astype(float))
        self.bias=torch.tensor(model.bias.astype(float));self.bo=torch.tensor(model.output_bias.astype(float))
        inputs=[]
        for cue in cues:
            row=[]
            for token in SEQUENCE[k:-1]:
                x=np.eye(4,dtype=np.float32)[token]
                if model.context_u is None:drive=model.wih@np.r_[x,np.asarray(cue,np.float32)]
                else:drive=model.wih@x+model.drive(cue)
                row.append(drive.astype(float))
            inputs.append(row)
        self.inputs=torch.tensor(np.asarray(inputs),dtype=torch.float64)
        self.constants=(self.N,self.mid,self.wh,self.wo,self.bias,self.bo,self.inputs,self.future_targets)
        if any(t.requires_grad for t in self.constants):raise ValueError('A frozen model buffer requires gradients')

    def __call__(self,a):
        h=self.mid+a@self.N.T;zs=[]
        for j in range(self.n_future):
            h=torch.tanh(self.inputs[:,j]+h@self.wh.T+self.bias)
            zs.append(h@self.wo.T+self.bo)
        z=torch.stack(zs,1)
        return F.cross_entropy(z.reshape(-1,4),self.future_targets.repeat(len(a)),reduction='none').reshape(len(a),self.n_future).mean(1)


def _step_inside(a,da,mid,N,caps):
    h=mid+a@N.T;dh=da@N.T;bound=1-CONTRACT['box_margin'];limit=np.ones(len(a))
    for j in range(h.shape[1]):
        positive=dh[:,j]>0;negative=dh[:,j]<0;q=np.full(len(a),np.inf)
        np.divide(bound-h[:,j],dh[:,j],out=q,where=positive)
        np.divide(-bound-h[:,j],dh[:,j],out=q,where=negative)
        limit=np.minimum(limit,q)
    finite=np.isfinite(caps);A=(da*da).sum(1);B=2*(a*da).sum(1)
    C=(a*a).sum(1)-np.where(finite,caps,0)**2
    take=finite&(A>0);root=np.full(len(a),np.inf)
    root[take]=(-B[take]+np.sqrt(np.maximum(0,B[take]**2-4*A[take]*C[take])))/(2*A[take])
    limit=np.maximum(0,np.minimum(limit,root));limit=np.minimum(1,limit)
    limit=np.where(limit<1,.99*limit,limit)
    new=a+limit[:,None]*da
    # A fixed numerical backoff is permitted only for constraint roundoff, never loss.
    for _ in range(24):
        nh=mid+new@N.T;radius=np.linalg.norm(new@N.T,axis=1)
        bad=(np.max(np.abs(nh),axis=1)>bound+1e-10)|(radius>caps+1e-10)
        if not bad.any():break
        limit[bad]*=.5;new[bad]=a[bad]+limit[bad,None]*da[bad]
    nh=mid+new@N.T;bad=(np.max(np.abs(nh),axis=1)>bound+1e-10)|(np.linalg.norm(new@N.T,axis=1)>caps+1e-10)
    new[bad]=a[bad];limit[bad]=0
    return new,limit


def search(model,midpoint,distance,k,cues,seed,smoke=False):
    """All fixed starts and all fixed iterations, including invalid/stalled records."""
    N=output_spaces(model)['N'];M=output_spaces(model)['M'];nd=N.shape[1]
    starts=CONTRACT['smoke']['oracle_starts'] if smoke else 9;steps=CONTRACT['smoke']['oracle_steps'] if smoke else 150
    planned=[];initial=[]
    for cue,vector in cues.items():
        for constraint in ['free','familiar_near']:
            cap=np.inf if constraint=='free' else .5*distance
            for restart in range(starts):
                if restart==0:
                    a=np.zeros(nd);feasible=valid_state(midpoint);target_radius=0.
                else:
                    direction=rng_for('P3',int(seed),int(k),restart).normal(size=nd);direction/=np.linalg.norm(direction)
                    target_radius=CONTRACT['oracle_random_radii'][restart-1]*distance
                    q=bounded_step(midpoint,N@direction,target_radius);a=direction*q['realized_radius'];feasible=q['feasible'] and np.linalg.norm(N@a)<=cap+1e-10
                planned.append({'cue':cue,'constraint_set':constraint,'restart_index':restart,'radius_cap':cap,'target_start_radius':target_radius,'feasible':bool(feasible),'status':'budget_exhausted' if feasible else 'infeasible_start_not_replaced','vector':vector})
                initial.append(a)
    plan=pd.DataFrame(planned);initial=np.stack(initial);B=len(plan);valid=np.flatnonzero(plan.feasible.to_numpy());nv=len(valid)
    loss_hist=np.full((steps+1,B),np.nan);grad_hist=loss_hist.copy();null_hist=loss_hist.copy();box_hist=loss_hist.copy();radius_hist=loss_hist.copy();p_hist=loss_hist.copy();ce_hist=loss_hist.copy();fraction_hist=loss_hist.copy()
    state_hist=np.full((steps+1,B,8),np.nan);a_hist=np.full((steps+1,B,nd),np.nan)
    best_a=initial.copy();best_loss=np.full(B,np.nan);best_iter=np.full(B,-1);last_loss=np.full(B,np.nan)
    before=model.digest();target=SEQUENCE[k];now=output_values(model,midpoint,target)
    if nv:
        objective=FrozenObjective(model,midpoint,N,k,plan.iloc[valid].vector.tolist())
        a=torch.tensor(initial[valid],dtype=torch.float64,requires_grad=True);moment=np.zeros((nv,nd));variance=np.zeros_like(moment);caps=plan.iloc[valid].radius_cap.to_numpy(float);incoming=np.full(nv,np.nan)
        minima=np.full(nv,np.inf)
        with frozen_weight_guard(model):
            for iteration in range(steps+1):
                coordinates=a.detach().numpy();h=midpoint+coordinates@N.T
                null=np.max(np.abs((h-midpoint)@M.T),axis=1);margin=1-np.max(np.abs(h),axis=1);excess=np.where(np.isfinite(caps),np.maximum(0,np.linalg.norm(h-midpoint,axis=1)-caps),0.)
                z=h@model.output.astype(float).T+model.output_bias.astype(float);centered=z-z.mean(1,keepdims=True)
                p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
                pdelta=np.max(np.abs(p-now['probabilities']),axis=1)
                ce=z.max(1)+np.log(np.exp(z-z.max(1,keepdims=True)).sum(1))-z[:,target]
                cedelta=np.abs(ce-now['ce'])
                if (null>1e-10).any() or (margin<1e-6-1e-10).any() or (excess>1e-10).any() or (pdelta>1e-10).any() or (cedelta>1e-10).any():raise ValueError('Oracle iterate violated a frozen nullspace/box/radius/current-output constraint')
                loss=objective(a)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite hidden-coordinate objective')
                value=loss.detach().numpy();loss_hist[iteration,valid]=value;null_hist[iteration,valid]=null;box_hist[iteration,valid]=margin;radius_hist[iteration,valid]=excess;p_hist[iteration,valid]=pdelta;ce_hist[iteration,valid]=cedelta;fraction_hist[iteration,valid]=incoming
                state_hist[iteration,valid]=h;a_hist[iteration,valid]=coordinates
                improved=value<minima;minima[improved]=value[improved];best_a[valid[improved]]=coordinates[improved];best_iter[valid[improved]]=iteration;best_loss[valid]=minima;last_loss[valid]=value
                if iteration==steps:break
                gradient=torch.autograd.grad(loss.sum(),a)[0].detach().numpy()
                if not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite hidden-coordinate gradient')
                grad_hist[iteration,valid]=np.linalg.norm(gradient,axis=1)
                moment=.9*moment+.1*gradient;variance=.999*variance+.001*gradient**2
                delta=-.02*(moment/(1-.9**(iteration+1)))/(np.sqrt(variance/(1-.999**(iteration+1)))+1e-8)
                updated,incoming=_step_inside(coordinates,delta,midpoint,N,caps)
                a=torch.tensor(updated,dtype=torch.float64,requires_grad=True)
    if model.digest()!=before:raise RuntimeError('Oracle altered a trained-model tensor')
    history=pd.DataFrame({'iteration':np.repeat(np.arange(steps+1),B),'cue':np.tile(plan.cue,steps+1),'constraint_set':np.tile(plan.constraint_set,steps+1),'restart_index':np.tile(plan.restart_index,steps+1),
        'feasible':np.tile(plan.feasible,steps+1),'loss':loss_hist.ravel(),'gradient_norm':grad_hist.ravel(),'null_residual':null_hist.ravel(),'box_margin':box_hist.ravel(),'radius_excess':radius_hist.ravel(),'current_softmax_residual':p_hist.ravel(),'current_ce_difference':ce_hist.ravel(),'incoming_step_fraction':fraction_hist.ravel()})
    for j in range(8):history[f'state_{j}']=state_hist[:,:,j].ravel()
    for j in range(nd):history[f'a_{j}']=a_hist[:,:,j].ravel()
    result=plan.drop(columns='vector').copy();result['initial_smooth_loss']=loss_hist[0];result['best_smooth_loss']=best_loss;result['last_smooth_loss']=last_loss;result['selected_iteration']=best_iter;result['optimization_steps_executed']=np.where(plan.feasible,steps,0)
    result['state']=[midpoint+N@best_a[i] if plan.feasible.iloc[i] else np.full(8,np.nan) for i in range(B)]
    result['initial_state']=[midpoint+N@initial[i] for i in range(B)]
    for i in valid:
        if np.ptp(loss_hist[max(0,steps-9):,i])<=1e-8:result.at[i,'status']='budget_exhausted_stalled'
    audit=result[['cue','constraint_set','restart_index','feasible','status']].copy();audit['model_tensors_requiring_grad']=0;audit['model_weights_unchanged']=model.digest()==before
    for name,array,operation in [('max_null_residual',null_hist,'max'),('minimum_box_margin',box_hist,'min'),('max_radius_excess',radius_hist,'max'),('max_current_softmax_difference',p_hist,'max')]:
        audit[name]=[getattr(np, 'nan'+operation)(array[:,i]) if plan.feasible.iloc[i] else np.nan for i in range(B)]
    return history,result,audit


def _source_replay(model,result,midpoint,k,cues,refs,meta,stream):
    result=result.copy();a,b=refs;distance=np.linalg.norm(a['hidden'][k-1].astype(float)-b['hidden'][k-1].astype(float))
    sep=np.linalg.norm(a['hidden'].astype(float)-b['hidden'].astype(float),axis=1)
    for name in ['future_ce','initial_actual_future_ce','worst_future_ce','relative_midpoint_distance','midpoint_distance','source_minus_smooth_ce','current_softmax_residual','source_float32_softmax_difference','source_float32_ce_difference','distance_to_A','distance_to_B','distance_to_observed_familiar_states']:
        result[name]=np.nan
    for name in ['far','viable_continuation','viable_strict','viable_worse_endpoint','far_viable','far_viable_midpoint_failed','practical_gain']:result[name]=False
    baselines={}
    for cue,vector in cues.items():
        trace=continue_from(model,midpoint,k,vector);baselines[cue]=future_summary(trace,refs)
    result['sample_id']=[state_id({**meta,'cue':r.cue,'constraint_set':r.constraint_set,'restart':int(r.restart_index)}) for r in result.itertuples()]
    for i,r in result.iterrows():
        base=baselines[r.cue]
        result.at[i,'midpoint_future_ce']=base['future_ce'];result.at[i,'midpoint_viable']=base['viable_continuation'];result.at[i,'familiar_bound']=base['familiar_bound'];result.at[i,'worse_endpoint_bound']=base['worse_endpoint_bound'];result.at[i,'n_future']=7-k
        if not r.feasible:continue
        for snapshot,h in [('initial',r.initial_state),('selected',r.state)]:
            q=assert_equivalent(model,midpoint,h,target=SEQUENCE[k]);precision=source_head_audit(model,midpoint,h,SEQUENCE[k])
            trace=continue_from(model,h,k,cues[r.cue]);trace=add_geometry(trace,model,(a['hidden'].astype(float)+b['hidden'].astype(float))/2,sep)
            metric=future_summary(trace,refs)
            for key,value in {**meta,'cue':r.cue,'constraint_set':r.constraint_set,'restart_index':r.restart_index,'sample_id':r.sample_id,'snapshot':snapshot,'k':k,'intervention_type':'state_oracle_reference' if snapshot=='selected' else 'random_unoptimized_reference','oracle_uses_future_targets':snapshot=='selected','model_weights_frozen':True}.items():trace[key]=value
            stream.write(trace)
            if snapshot=='initial':result.at[i,'initial_actual_future_ce']=metric['future_ce']
            else:
                for key,value in metric.items():result.at[i,key]=value
                radius=np.linalg.norm(h-midpoint);relative=radius/(distance+1e-8)
                far=relative>=.25;viable=bool(metric['viable_continuation'])
                values={'midpoint_distance':radius,'relative_midpoint_distance':relative,'far':far,'far_viable':far and viable,'far_viable_midpoint_failed':far and viable and not base['viable_continuation'],
                    'practical_gain':metric['future_ce']<=base['future_ce']-.20,'source_minus_smooth_ce':metric['future_ce']-r.best_smooth_loss,'current_softmax_residual':q['softmax_max_difference'],
                    'source_float32_softmax_difference':precision['source_float32_softmax_difference'],'source_float32_ce_difference':precision['source_float32_ce_difference'],
                    'distance_to_A':np.linalg.norm(h-a['hidden'][k-1]),'distance_to_B':np.linalg.norm(h-b['hidden'][k-1]),
                    'distance_to_observed_familiar_states':float(np.linalg.norm(np.vstack([a['hidden'],b['hidden']]).astype(float)-h,axis=1).min())}
                for key,value in values.items():result.at[i,key]=value
    for j in range(8):result[f'state_{j}']=[h[j] for h in result.state];result[f'initial_state_{j}']=[h[j] for h in result.initial_state]
    for key,value in meta.items():result[key]=value
    result['oracle_future_targets_used']=True;result['model_optimizer_updates']=0
    return result.drop(columns=['state','initial_state'])


def _best_rows(restarts):
    rows=[];keys=['cohort','condition','seed','cue','k','constraint_set']
    for key,g in restarts.groupby(keys,sort=False):
        valid=g[g.feasible];base=dict(zip(keys,key))
        if len(valid):
            best=valid.sort_values(['future_ce','restart_index']).iloc[0].to_dict();row={**best,**base}
            row['best_unoptimized_future_ce']=float(valid.initial_actual_future_ce.min())
            random=valid[valid.restart_index.gt(0)];row['mean_random_initial_future_ce']=float(random.initial_actual_future_ce.mean()) if len(random) else np.nan
        else:
            row={**g.iloc[0].to_dict(),**base,'status':'search_limited_inconclusive_no_feasible_start','future_ce':np.nan,'sample_id':'not_established'}
            row['best_unoptimized_future_ce']=np.nan;row['mean_random_initial_future_ce']=np.nan
        row['n_restarts_planned']=len(g);row['n_feasible_starts']=len(valid)
        row['any_far_viable']=bool(valid.far_viable.any());row['any_far_viable_midpoint_failed']=bool(valid.far_viable_midpoint_failed.any())
        row['n_far_viable_restarts']=int(valid.far_viable.sum());row['n_viable_restarts']=int(valid.viable_continuation.sum())
        row['oracle_gain']=row['midpoint_future_ce']-row['future_ce'] if len(valid) else np.nan
        row['existence_status']='far_viable_found' if row['any_far_viable'] else ('viable_near_only_found' if valid.viable_continuation.any() else 'search_limited_inconclusive')
        rows.append(row)
    return pd.DataFrame(rows)


def run(root,out,smoke=False):
    bundle=load_sources(root);records=subset(bundle['records'],'p3',smoke)
    probes=CONTRACT['smoke']['probes'] if smoke else PROBES;positions=CONTRACT['smoke']['k'] if smoke else K
    cues={c:bundle['cues'][c] for c in probes}
    history_stream=Stream(out,'p3_oracle_restart_long');future_stream=Stream(out,'p3_future_transition');all_results=[];audits=[]
    for number,r in enumerate(records,1):
        model=verify_checkpoint(root,r);a,b,mid=references(model)
        with frozen_weight_guard(model):
            for k in positions:
                distance=float(np.linalg.norm(a['hidden'][k-1].astype(float)-b['hidden'][k-1].astype(float)))
                h,res,audit=search(model,mid[k-1],distance,k,cues,r['seed'],smoke)
                meta={**metadata(r),'k':k}
                for key,value in meta.items():h[key]=value;audit[key]=value
                h['oracle_uses_future_targets']=True;h['model_optimizer_updates']=0
                history_stream.write(h)
                res=_source_replay(model,res,mid[k-1],k,cues,(a,b),meta,future_stream)
                lookup={(x.cue,x.constraint_set,x.restart_index):x.sample_id for x in res.itertuples()}
                audit['sample_id']=[lookup[(x.cue,x.constraint_set,x.restart_index)] for x in audit.itertuples()]
                all_results.append(res);audits.append(audit)
        print(f'P3 {"smoke" if smoke else "full"} model {number}/{len(records)}; fixed {len(positions)} positions',flush=True)
    restarts=pd.concat(all_results,ignore_index=True);best=_best_rows(restarts);audit=pd.concat(audits,ignore_index=True)
    planned=len(records)*len(positions)*len(probes)*2*(2 if smoke else 9)
    if len(restarts)!=planned:raise ValueError('Oracle restart registry/budget mismatch')
    expected_history=planned*(11 if smoke else 151)
    if history_stream.n!=expected_history:raise ValueError('Oracle loss trajectory counts do not reconcile')
    for name,frame in [('p3_oracle_restart_summary',restarts),('p3_oracle_best_seed',best),('p3_gradient_feasibility_audit',audit)]:save(out,name,frame)
    metrics=['future_ce','midpoint_future_ce','oracle_gain','far_viable','any_far_viable','any_far_viable_midpoint_failed','viable_continuation','n_feasible_starts']
    numeric=best[['cohort','condition','seed','cue','k','constraint_set',*metrics]].copy()
    for c in metrics:numeric[c]=pd.to_numeric(numeric[c])
    numeric=strict_novel_means(numeric,['cohort','condition','seed','k','constraint_set'],metrics)
    boot=complete_bootstrap(numeric,metrics,['cohort','condition','cue','k','constraint_set']);save(out,'p3_viability_counts',boot)
    registry=restarts[['cohort','condition','seed','cue','k','constraint_set','restart_index','sample_id','status','optimization_steps_executed']].copy()
    registry['stage']='p3';registry['branch']='hidden_state_oracle';registry['direction_index']=-1;registry['output_path']='tables/p3_oracle_restart_summary.csv';save(out,'p3_run_registry',registry)
    _plot(out,best)
    headline=boot[boot.cue.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])&boot.k.eq(1)&boot.constraint_set.eq('free')&boot.metric.eq('any_far_viable_midpoint_failed')]
    found=int(restarts.far_viable_midpoint_failed.sum())
    report(out,'P3_RESULTS.md',f'# P3 — oracle existence of alternative viable trajectories\n\nWe asked whether a distant, output-equivalent starting state can sustain the entire remaining sequence. Only kernel coordinates were optimized, using future labels; all recurrent/input/output weights remained fixed. Every planned free/near case retained nine starts and 150 updates per feasible start (smoke: two starts, ten updates). Infeasible and stalled starts were retained without replacement.\n\nAcross all registered positions/cues, {found} restart endpoints were far and viable while their midpoint failed. The independent unit is the seed, not a restart. Primary k=1, free-oracle discovery rates:\n\n'+table(headline[['cue','mean','ci_low','ci_high','n_seeds','status']])+'\n\nEvery accepted iterate satisfied the interior box, kernel and radius constraints and current four-class softmax identity. Selected states were replayed under the original float32 recurrence; smooth-surrogate discrepancies and every iteration are retained. The mean-familiar+0.10-nat criterion, zero-slack and worse-endpoint sensitivities are all exported.\n\nA found far state disproves exact midpoint necessity for that frozen model/cue/horizon. It is a future-target-informed existence result, not learned retrieval or a natural-performance estimate. Failure to find one is search-limited and does not prove impossibility. The remaining question is whether the tested physical cue routes can generate viable continuation from natural prestates.\n')
    return {'integrity_passed':True,'implementation_status':'validated','scientific_status':'completed','existence_outcome':'far_viable_alternative_found' if restarts.far_viable.any() else 'search_limited_inconclusive','n_models':len(records),'planned_restarts':len(restarts),'feasible_restarts':int(restarts.feasible.sum()),'restart_history_rows':history_stream.n,'far_viable_midpoint_failed_restart_endpoints':found,'hidden_coordinate_updates':int(restarts.optimization_steps_executed.sum()),'model_optimizer_updates':0,'oracle_future_target_information':True}


def _plot(out,best):
    import matplotlib.pyplot as plt
    from .figures import save_figure
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');planned=int(best.n_restarts_planned.max())
    for ax,cue in zip(axes.ravel(),PROBES):
        f=best[best.k.eq(1)&best.cue.eq(cue)]
        if not len(f):
            ax.axis('off');ax.text(.1,.5,cue+'\nNot run in technical smoke',transform=ax.transAxes);continue
        for kind,marker in [('free','o'),('familiar_near','s')]:
            g=f[f.constraint_set.eq(kind)&f.feasible]
            ax.scatter(g.midpoint_future_ce,g.future_ce,c=np.where(g.far_viable,'#208b59','#888888'),marker=marker,s=28,alpha=.7,label=kind)
        limits=ax.get_xlim();ax.plot(limits,limits,color='gray',ls='--',lw=.7)
        ax.set(title=cue,xlabel='Midpoint future CE (nats)',ylabel=f'Best-of-{planned} bounded oracle CE');ax.legend(fontsize=7)
    fig.suptitle('k=1: six unpatched future predictions; green = far and viable. Oracle uses future targets.')
    save_figure(fig,out,'fig_p3_midpoint_vs_alternative')
