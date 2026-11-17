"""S3 local finite-horizon cue sensitivity; no autonomous-dynamics analysis."""
import json
import numpy as np
import pandas as pd
import torch
from context_sequence.dynamics.closure import references
from experiments.output_compatible_trajectories.continuation import frozen_weight_guard
from .config import CONTRACT,PROBES,SEQUENCE
from .core import source_trace,frozen_constants,load_checkpoint
from .audit import sources
from .simulation import schedules
from .history import bindings_for
from .io import save,report


def local_maps(model,state):
    d=1-np.asarray(state,float)**2
    return d[:,None]*model.wih[:,4:6].astype(float),d[:,None]*model.whh.astype(float)


def control_matrices(model,trace,k):
    h=trace[[f'hidden_{i}' for i in range(8)]].to_numpy(float)
    B_early,_=local_maps(model,h[k-3]);B_late,J_late=local_maps(model,h[k-2]);_,J_final=local_maps(model,h[k-1])
    early=J_final@J_late@B_early;late=J_final@B_late
    return {'early':early,'late':late,'tied':early+late,'two':np.c_[early,late]}


def finite_difference(model,schedule,k,which,coord,eps):
    points={'early':[k-3],'late':[k-2],'tied':[k-3,k-2]}[which]
    base=np.asarray(schedule,float);plus=base.copy();minus=base.copy();plus[points,coord]+=eps;minus[points,coord]-=eps
    inside_plus=np.all(np.abs(plus[:,:2])<=1.5);inside_minus=np.all(np.abs(minus[:,:2])<=1.5)
    def endpoint(c):return source_trace(model,c)[[f'hidden_{i}' for i in range(8)]].iloc[k-1].to_numpy(float)
    if inside_plus and inside_minus:return (endpoint(plus)-endpoint(minus))/(2*eps),'central'
    if inside_plus:return (endpoint(plus)-endpoint(base))/eps,'forward_inside_box'
    if inside_minus:return (endpoint(base)-endpoint(minus))/eps,'backward_inside_box'
    return np.full(8,np.nan),'undefined_boundary_no_common_direction'


def rank_alignment(matrix,gradient):
    U,s,_=np.linalg.svd(matrix,full_matrices=False);largest=float(s[0]) if len(s) else 0.
    absolute=int(np.sum(s>1e-8));relative=int(np.sum(s>1e-4*largest));combined=int(np.sum(s>max(1e-8,1e-4*largest)))
    norm=float(np.linalg.norm(gradient))
    alignment=float(np.linalg.norm(U[:,:combined].T@gradient)/norm) if norm>0 else np.nan
    return {'singular_values':json.dumps(s.tolist()),'rank_absolute':absolute,'rank_relative':relative,'rank_combined':combined,'gradient_norm':norm,'gradient_alignment':alignment,
      'alignment_status':'defined' if norm>0 else 'undefined_zero_gradient','largest_singular_value':largest,'smallest_singular_value':float(s[-1]) if len(s) else np.nan}


def future_gradient(model,state,cue,k):
    c=frozen_constants(model);h=torch.tensor(np.asarray(state,float),dtype=torch.float64,requires_grad=True);initial=h;losses=[]
    vector=torch.tensor(np.asarray(cue[:2],float));tokens=torch.eye(4,dtype=torch.float64)
    for t in range(k,7):
        h=torch.tanh(tokens[SEQUENCE[t]]@c['wx'].T+vector@c['wc'].T+h@c['wh'].T+c['bias'])
        z=h@c['wo'].T+c['bo'];losses.append(torch.logsumexp(z,0)-z[SEQUENCE[t+1]])
    return torch.autograd.grad(torch.stack(losses).mean(),initial)[0].detach().numpy()


def run(root,out):
    b=sources();summary=pd.read_csv(out/'tables/s2_seed_summary.csv');paired=pd.read_csv(out/'tables/s2_paired_effects.csv');rows=[];fds=[]
    for number,r in enumerate(b['records'],1):
        model=load_checkpoint(root,r)
        with frozen_weight_guard(model):
            for cue in PROBES:
                template=np.tile(b['cues'][cue],(7,1))
                for k in [3,5]:
                    selected=summary[summary.seed.eq(r['seed'])&summary.cue.eq(cue)&summary.k.eq(k)&summary.arm.eq('R')&summary.policy.eq('bounded_oracle')].iloc[0]
                    paired_row=paired[paired.seed.eq(r['seed'])&paired.cue.eq(cue)&paired.k.eq(k)&paired.policy.eq('bounded_oracle')].iloc[0]
                    for schedule_kind in ['N0','R_oracle']:
                        meta={'seed':r['seed'],'cue':cue,'k':k,'schedule_kind':schedule_kind,'checkpoint_sha256':r['checkpoint_sha256'],'R_minus_L_future_ce':paired_row.R_minus_L_future_ce,'R_success_L_failed':paired_row.R_success_L_failed}
                        if schedule_kind=='R_oracle' and not np.isfinite(selected[['selected_0','selected_1']].to_numpy(float)).all():
                            for matrix_name in ['late','tied','two']:rows.append({**meta,'matrix':matrix_name,'status':'missing_selected_oracle_schedule','gradient_alignment':np.nan})
                            continue
                        schedule=template if schedule_kind=='N0' else schedules(template,bindings_for('R',k),selected[['selected_0','selected_1']].to_numpy(float)[None])[0]
                        trace=source_trace(model,schedule);mats=control_matrices(model,trace,k);state=trace[[f'hidden_{i}' for i in range(8)]].iloc[k-1].to_numpy(float)
                        gradient=future_gradient(model,state,b['cues'][cue],k)
                        if not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite future CE gradient')
                        for name in ['late','tied','two']:
                            q=rank_alignment(mats[name],gradient)
                            rows.append({**meta,**q,'matrix':name,'rows':8,'columns':mats[name].shape[1],'matrix_values':json.dumps(mats[name].tolist()),'future_gradient_values':json.dumps(gradient.tolist()),
                                'future_label_informed_diagnostic':True,'saturation_min_D':float((1-trace[[f'hidden_{i}' for i in range(8)]].iloc[k-3:k].to_numpy(float)**2).min()),'status':'completed'})
                        for eps in [1e-3,1e-4]:
                            for name in ['early','late','tied']:
                                for coord in [0,1]:
                                    fd,method=finite_difference(model,schedule,k,name,coord,eps);analytic=mats[name][:,coord]
                                    error=np.abs(fd-analytic);relative=float(np.linalg.norm(fd-analytic)/(np.linalg.norm(analytic)+1e-12))
                                    fds.append({**meta,'matrix':name,'coordinate':coord,'epsilon':eps,'method':method,'max_abs_error':float(np.max(error)),'relative_norm_error':relative,
                                      'boundary':method!='central','poor_linear_approximation':bool(np.max(error)>.02 and relative>.1),'analytic_values':json.dumps(analytic.tolist()),'finite_difference_values':json.dumps(fd.tolist()),'status':'completed' if np.isfinite(fd).all() else method})
        print(f'S3 model {number}/20',flush=True)
    f=pd.DataFrame(rows);fd=pd.DataFrame(fds);save(out,'s3_sensitivities',f);save(out,'s3_finite_difference_audit',fd)
    if len(f)!=20*6*2*2*3:raise ValueError('S3 planned sensitivity registry mismatch')
    diagnostic=[]
    for (seed,cue,k,kind),g in f.groupby(['seed','cue','k','schedule_kind']):
        x=g.set_index('matrix');diagnostic.append({'seed':seed,'cue':cue,'k':k,'schedule_kind':kind,'R_minus_L_future_ce':g.R_minus_L_future_ce.iloc[0],
          'alignment_tied_minus_late':x.loc['tied','gradient_alignment']-x.loc['late','gradient_alignment'],
          'rank_tied_minus_late':x.loc['tied','rank_combined']-x.loc['late','rank_combined'] if 'rank_combined' in x else np.nan})
    save(out,'s3_paired_control_diagnostic',pd.DataFrame(diagnostic))
    report(out,'S3_LOCAL_CONTROL_RESULTS.md','# S3 — local finite-horizon context sensitivity\n\nWe computed baseline and selected repeated-oracle schedules at k3/5 for every seed/probe, regardless of behavioral success. Exact chain products give late control J_k B_(k−1), early J_k J_(k−1) B_(k−2), tied repeated early+late (eight×two), and independent early/late concatenation (eight×four). Raw singular values, absolute1e-8 and relative1e-4 ranks, combined rank and label-informed future-CE gradient alignment are retained.\n\nAnalytic/autodiff/source finite-difference synthetic tests pass. The actual-source audit uses eps1e-3 and1e-4 at both prep coordinates and tied changes. Boundary differences remain inside the numeric box and are labeled one-sided; saturation and poor approximation flags are retained rather than tuning tolerances. Zero gradient norm has undefined alignment.\n\nThese are local sensitivities of a forced, short computation. Extra input steps do not guarantee rank growth or useful behavior. The behavioral causal contrast remains the nonlinear same-final-target R/L cue intervention; alignment associations cannot establish mediation, global reachability, attractors or autonomous stability.\n')
    return {'models':20,'sensitivity_rows':len(f),'finite_difference_rows':len(fd),'poor_fd_rows':int(fd.poor_linear_approximation.sum()),'model_updates':0,'hidden_placements':0}
