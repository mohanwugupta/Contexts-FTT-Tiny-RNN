"""P4 physical cue branches. Generation never uses an oracle or future labels."""
import numpy as np
import pandas as pd
from context_sequence.dynamics.closure import natural,references
from .config import CONTRACT,PROBES,K,SEQUENCE,subset
from .continuation import continue_from,row,digest_state,future_summary,add_geometry,frozen_weight_guard
from .common import metadata,state_id,Stream,save,report,table
from .source_audit import load_sources,verify_checkpoint
from .nullspace import strict_novel_means,complete_bootstrap


def branch_vectors(cues,source,target):
    return {name:np.asarray(cues[key]).copy() for name,key in [('native_source',source),('A','A'),('B','B'),('Mixed','Mixed'),('zero','NoContext'),('target_native',target)]}


def cue_branch(model,prestate,k,current_cue,target_cue,persistent=False):
    """The only generation action is one original recurrent step with a physical cue."""
    pre=np.asarray(prestate,np.float32).copy();identity=digest_state(pre);before=model.digest()
    h,z=model.step(SEQUENCE[k-1],current_cue,pre)
    future=current_cue if persistent else target_cue
    t=continue_from(model,h,k,current_cue,future_cue=future)
    # Preserve the source float32 head at the actual cue-generated current update.
    current=row(h,z,k,k,SEQUENCE,current_cue)
    for key,value in current.items():t.at[0,key]=value
    if digest_state(pre)!=identity or model.digest()!=before:raise RuntimeError('Cue branch mutated prestate or weights')
    t['prestate_hash']=identity
    return t


def viable_distance(state,states):
    if not len(states):return np.nan,'not_established'
    return float(np.linalg.norm(np.asarray(states)-state,axis=1).min()),'finite_found_set'


def cue_only(frame):return frame[frame.intervention_type.eq('cue_only')].copy()


def run(root,out,smoke=False):
    bundle=load_sources(root);records=subset(bundle['records'],'p4',smoke)
    probes=CONTRACT['smoke']['probes'] if smoke else PROBES;positions=CONTRACT['smoke']['k'] if smoke else K
    base=out.parent if smoke else out
    restarts=pd.read_csv(base/'tables/p3_oracle_restart_summary.csv');best=pd.read_csv(base/'tables/p3_oracle_best_seed.csv')
    stream=Stream(out,'p4_same_prestate_branch_long');oracle_stream=Stream(out,'p4_state_oracle_transition');rows=[];controls=[]
    for number,r in enumerate(records,1):
        model=verify_checkpoint(root,r);a,b,mid=references(model);sep=np.linalg.norm(a['hidden'].astype(float)-b['hidden'].astype(float),axis=1);meta=metadata(r)
        naturals={c:natural(model,bundle['cues'][c],SEQUENCE) for c in CONTRACT['p4_sources']}
        with frozen_weight_guard(model):
            for k in positions:
                for target in probes:
                    p3=restarts[restarts.seed.eq(r['seed'])&restarts.k.eq(k)&restarts.cue.eq(target)]
                    good=p3[p3.feasible&p3.viable_continuation]
                    viable=good[[f'state_{i}' for i in range(8)]].to_numpy()
                    # Oracle controls are a distinct intervention type, once per case.
                    for constraint in ['free','familiar_near']:
                        q=best[best.seed.eq(r['seed'])&best.k.eq(k)&best.cue.eq(target)&best.constraint_set.eq(constraint)].iloc[0]
                        control={**meta,'k':k,'cue':target,'source_context':'oracle_state','branch':constraint,'schedule':'pulse','intervention_type':'state_oracle_reference','oracle_uses_future_targets':True,'status':q.status,'future_ce':q.future_ce,'viable_continuation':bool(q.viable_continuation),'sample_id':q.sample_id}
                        controls.append(control)
                        if q.n_feasible_starts:
                            state=q[[f'state_{i}' for i in range(8)]].to_numpy(float)
                            t=add_geometry(continue_from(model,state,k,bundle['cues'][target]),model,mid,sep)
                            t['prestate_hash']='not_applicable_state_oracle'
                            for key,value in control.items():
                                if key not in ['future_ce','viable_continuation']:t[key]=value
                            t['model_optimizer_updates']=0;oracle_stream.write(t)
                    for source in CONTRACT['p4_sources']:
                        pre=np.zeros(8,np.float32) if k==1 else naturals[source]['hidden'][k-2].copy()
                        vectors=branch_vectors(bundle['cues'],source,target)
                        baseline=cue_branch(model,pre,k,vectors['target_native'],bundle['cues'][target],False)
                        baseline_ce=future_summary(baseline,(a,b))['future_ce']
                        for branch,vector in vectors.items():
                            for schedule in ['pulse','persistent']:
                                t=cue_branch(model,pre,k,vector,bundle['cues'][target],schedule=='persistent')
                                # A same-cue continuation must replay the unmodified historical source exactly.
                                if np.array_equal(vector,bundle['cues'][source]) and (schedule=='persistent' or np.array_equal(vector,bundle['cues'][target])):
                                    np.testing.assert_array_equal(t[[f'hidden_{i}' for i in range(8)]],naturals[source]['hidden'][k-1:])
                                    np.testing.assert_array_equal(t[[f'logit_{i}' for i in range(4)]],naturals[source]['logits'][k-1:])
                                metric=future_summary(t,(a,b));state=t[[f'hidden_{i}' for i in range(8)]].iloc[0].to_numpy(float)
                                distance,status=viable_distance(state,viable)
                                ident={**meta,'k':k,'cue':target,'source_context':source,'branch':branch,'schedule':schedule}
                                record={**ident,**metric,'sample_id':state_id(ident),'status':'completed','intervention_type':'cue_only','oracle_uses_future_targets':False,
                                    'prestate_hash':digest_state(pre),'inherited_history':k>1,'current_ce':float(t.raw_ce.iloc[0]),'baseline_target_pulse_future_ce':baseline_ce,'raw_future_rescue':baseline_ce-metric['future_ce'],
                                    'midpoint_distance':np.linalg.norm(state-mid[k-1]),'relative_midpoint_distance':np.linalg.norm(state-mid[k-1])/(sep[k-1]+1e-8),
                                    'distance_to_viable_set':distance,'viable_set_status':status,'n_viable_oracle_reference_states':len(viable),'model_optimizer_updates':0}
                                rows.append(record)
                                t=add_geometry(t,model,mid,sep)
                                for key,value in record.items():
                                    if key not in metric and key not in ['current_ce','midpoint_distance']:t[key]=value
                                stream.write(t)
        print(f'P4 {"smoke" if smoke else "full"} model {number}/{len(records)}',flush=True)
    seeds=pd.DataFrame(rows);oracle=pd.DataFrame(controls)
    expected=len(records)*len(positions)*len(probes)*8*6*2
    if len(seeds)!=expected or seeds.duplicated(['seed','k','cue','source_context','branch','schedule']).any():raise ValueError('Cue branch count/identity mismatch')
    save(out,'p4_cue_access_seed',seeds);save(out,'p4_state_oracle_reference',oracle)
    metrics=['future_ce','worst_future_ce','viable_continuation','viable_strict','viable_worse_endpoint','raw_future_rescue','current_ce','relative_midpoint_distance']
    numeric=seeds[['cohort','condition','seed','cue','k','source_context','branch','schedule',*metrics]].copy()
    for name in metrics:numeric[name]=pd.to_numeric(numeric[name])
    numeric=strict_novel_means(numeric,['cohort','condition','seed','k','source_context','branch','schedule'],metrics)
    boot=complete_bootstrap(numeric,metrics,['cohort','condition','cue','k','source_context','branch','schedule']);save(out,'p4_branch_bootstrap',boot)
    comparison=[]
    for (seed,k,cue),g in seeds.groupby(['seed','k','cue']):
        p=best[best.seed.eq(seed)&best.k.eq(k)&best.cue.eq(cue)]
        pulse=g[g.schedule.eq('pulse')];own=pulse[pulse.source_context.eq(cue)&pulse.branch.eq('target_native')].iloc[0]
        comparison.append({'seed':seed,'k':k,'cue':cue,'cohort':g.cohort.iloc[0],'condition':g.condition.iloc[0],
          'oracle_far_viable_found':bool(p.any_far_viable.any()),'oracle_far_viable_midpoint_failed':bool(p.any_far_viable_midpoint_failed.any()),
          'any_pulse_viable':bool(pulse.viable_continuation.any()),'native_viable':bool(own.viable_continuation),
          'oracle_found_no_tested_pulse_viable':bool(p.any_far_viable.any() and not pulse.viable_continuation.any()),
          'n_unique_prestates':pulse.prestate_hash.nunique(),'n_pulse_branches':len(pulse),'minimum_pulse_future_ce':pulse.future_ce.min()})
    comparison=pd.DataFrame(comparison);save(out,'p4_oracle_vs_cue',comparison)
    registry=seeds[['cohort','condition','seed','cue','k','source_context','branch','schedule','sample_id','status']].copy();registry['stage']='p4';registry['restart_index']=-1;registry['direction_index']=-1;registry['output_path']='tables/p4_cue_access_seed.csv';save(out,'p4_run_registry',registry)
    _plot(out,seeds,oracle)
    head=comparison[comparison.k.eq(1)].groupby('cue',as_index=False)[['oracle_far_viable_found','any_pulse_viable','native_viable','oracle_found_no_tested_pulse_viable']].sum()
    report(out,'P4_RESULTS.md','# P4 — finite physical cue access\n\nWe asked whether the registered physical cues can generate viable future behavior from identical natural prestates. We changed only the current cue for pulse arms; persistent arms also changed future cues and are a separate comparison. No cue-generation step used future labels or hidden-state optimization. All original weights remained frozen.\n\nPrimary k=1 counts out of '+str(len(records))+' seeds, each retaining all eight sources and six branches (at k=1 every source has the same zero prestate):\n\n'+table(head)+'\n\nSeed-bootstrap intervals, every individual branch, per-lag CE, and separate state-oracle references are exported. Oracle proximity is diagnostic only; where no viable oracle reference exists it is NA/not_established. Same-cue branches reproduced original source states and logits bit for bit. History comparisons are interpretable only at k=3/5. Cue success demonstrates accessibility for that finite cue/prestate/schedule. Failure establishes neither global unreachability nor a human mechanism. All futures are teacher-forced.\n')
    return {'integrity_passed':True,'implementation_status':'validated','scientific_status':'completed','n_models':len(records),'cue_only_branch_cases':len(seeds),'oracle_reference_cases':len(oracle),'transition_rows':stream.n,'model_optimizer_updates':0,'cue_generator_future_target_optimization':False}


def _plot(out,seeds,oracle):
    import matplotlib.pyplot as plt
    from .figures import save_figure
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,cue in zip(axes.ravel(),PROBES):
        g=seeds[seeds.k.eq(1)&seeds.source_context.eq('A')&seeds.cue.eq(cue)]
        if not len(g):ax.axis('off');continue
        names=CONTRACT['p4_branches'];x=np.arange(len(names))
        for schedule,offset,color in [('pulse',-.18,'#2878a8'),('persistent',.18,'#cc7940')]:
            f=g[g.schedule.eq(schedule)].groupby('branch').viable_continuation.mean().reindex(names)
            ax.bar(x+offset,f,width=.35,label=schedule,color=color)
        o=oracle[oracle.k.eq(1)&oracle.cue.eq(cue)&oracle.branch.eq('free')]
        ax.axhline(o.viable_continuation.mean(),ls='--',color='black',label='state oracle; future-target informed')
        ax.set(title=cue,ylim=(0,1.05),ylabel='Fraction viable (20 trained seeds)',xticks=x,xticklabels=names);ax.tick_params(axis='x',labelrotation=40)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('k=1; six future predictions. Pulse holds future target cue fixed; persistent changes it.')
    save_figure(fig,out,'fig_p4_generation_vs_maintenance')
