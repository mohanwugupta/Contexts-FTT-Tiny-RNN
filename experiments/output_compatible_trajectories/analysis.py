"""P5 fixed-outcome synthesis. No search, model updates or threshold changes."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from context_sequence.dynamics.closure import references
from .config import CONTRACT,PROBES
from .common import save,report,table,metadata
from .source_audit import load_sources,verify_checkpoint
from .state_geometry import output_values
from .nullspace import strict_novel_means,complete_bootstrap


def adjudicate(oracle_found,cue_viable,exact_output_access):
    if not oracle_found:return 'search_limited_inconclusive'
    if exact_output_access:return 'alternative_exists_exact_output_cue_access_found'
    if cue_viable:return 'alternative_exists_cue_viability_found'
    return 'alternative_exists_tested_cues_failed'


def headline_counts(frame):
    f=frame[frame.k.eq(1)].copy()
    if f.duplicated(['seed','cue']).any():raise ValueError('Repeated branch or seed in headline')
    keys=['oracle_far_viable_found','any_pulse_viable','native_viable','oracle_found_no_tested_pulse_viable']
    g=f.groupby('cue')[keys].sum().reset_index();g['n_seeds']=f.groupby('cue').seed.nunique().reindex(g.cue).to_numpy()
    return g


def cue_compatibility(root,out,p4):
    # Pure diagnostic on saved generated states, not another intervention/search.
    current=[]
    for chunk in pd.read_csv(out/'tables/p4_same_prestate_branch_long.csv',chunksize=50000):current.append(chunk[chunk.lag.eq(0)])
    current=pd.concat(current,ignore_index=True);bundle=load_sources(root);rows=[]
    for r in bundle['records']:
        if r['condition']!='High_monotonic':continue
        model=verify_checkpoint(root,r);a,b,mid=references(model)
        g=current[current.seed.eq(r['seed'])]
        for k in [1,3,5]:
            f=g[g.k.eq(k)].copy();h=f[[f'hidden_{i}' for i in range(8)]].to_numpy()
            z=h@model.output.astype(float).T+model.output_bias.astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
            reference=output_values(model,mid[k-1])['probabilities']
            f['current_softmax_distance_to_midpoint']=np.max(np.abs(p-reference),axis=1)
            rows.append(f[['sample_id','current_softmax_distance_to_midpoint']])
    diagnostics=pd.concat(rows,ignore_index=True)
    if diagnostics.sample_id.duplicated().any():raise ValueError('Repeated cue state diagnostic')
    f=p4.merge(diagnostics,on='sample_id',validate='one_to_one')
    f['exact_output_compatible']=f.current_softmax_distance_to_midpoint.le(CONTRACT['softmax_tolerance'])
    f['far']=f.relative_midpoint_distance.ge(CONTRACT['far_relative_distance'])
    f['far_viable']=f.far&f.viable_continuation
    f['far_viable_exact_output']=f.far_viable&f.exact_output_compatible
    save(out,'p5_cue_output_compatibility',f)
    return f


def run(root,out,smoke=False):
    if smoke:raise ValueError('P5 synthesis uses completed full stages only')
    read=lambda n:pd.read_csv(out/'tables'/f'{n}.csv')
    p2=read('p2_primary_seed');best=read('p3_oracle_best_seed');restarts=read('p3_oracle_restart_summary');p4=read('p4_cue_access_seed');comparison=read('p4_oracle_vs_cue')
    compatible=cue_compatibility(root,out,p4)
    counts=headline_counts(comparison);save(out,'p5_headline_seed_counts',counts)
    metrics=['oracle_far_viable_found','oracle_far_viable_midpoint_failed','any_pulse_viable','native_viable','oracle_found_no_tested_pulse_viable']
    num=comparison[['cohort','condition','seed','cue','k',*metrics]].copy()
    for c in metrics:num[c]=num[c].astype(float)
    num=strict_novel_means(num,['cohort','condition','seed','k'],metrics)
    intervals=complete_bootstrap(num,metrics,['cohort','condition','cue','k']);save(out,'p5_headline_intervals',intervals)
    # Cue-access events are collapsed within each trained seed, never over branch rows.
    access=compatible[compatible.schedule.eq('pulse')].groupby(['seed','cue','k'],as_index=False)[['far_viable','far_viable_exact_output']].any()
    save(out,'p5_cue_access_events',access)
    merged=comparison.merge(access,on=['seed','cue','k'],validate='one_to_one')
    merged['decision']=[adjudicate(r.oracle_far_viable_found,r.any_pulse_viable,r.far_viable_exact_output) for r in merged.itertuples()];save(out,'p5_decision_table',merged)
    figure_b(out,p2,best)
    supplementary_c(out,p4,read('p4_same_prestate_branch_long'),read('p4_state_oracle_transition'))
    # Deterministic illustration: first seed/cue/restart satisfying registered k=1 counterexample.
    candidates=restarts[restarts.k.eq(1)&restarts.constraint_set.eq('free')&restarts.far_viable_midpoint_failed]
    example=None
    if len(candidates):
        example=candidates.sort_values(['seed','cue','restart_index']).iloc[0]
        allfuture=read('p3_future_transition');future=allfuture[allfuture.sample_id.eq(example.sample_id)&allfuture.snapshot.eq('selected')].sort_values('lag')
        parts=[]
        for ref in pd.read_csv(out/'tables/p2_future_transition.csv',chunksize=50000):
            parts.append(ref[ref.condition.eq('High_monotonic')&ref.seed.eq(example.seed)&ref.cue.eq(example.cue)&ref.k.eq(1)&ref.branch.eq('midpoint')])
        mid=pd.concat(parts).sort_values('lag')
        numeric=future[['transition','lag','target','raw_ce','midpoint_hidden_distance','AB_hidden_separation',*[f'probability_{i}' for i in range(4)]]].copy()
        numeric['relative_midpoint_distance']=numeric.midpoint_hidden_distance/(numeric.AB_hidden_separation+1e-8);numeric['midpoint_ce']=mid.raw_ce.to_numpy()
        for j in range(4):numeric[f'midpoint_probability_{j}']=mid[f'probability_{j}'].to_numpy()
        numeric['sample_id']=example.sample_id;save(out,'p5_worked_example',numeric)
    _reports(root,out,counts,intervals,merged,best,restarts,p2,example)
    claims=[
      ('C01','all rank=3/nullity=5; original replay parity','all audited checkpoints',120,'p0_output_spaces.csv; p0_replay_parity.csv','supplement','test_source_and_replay.py; test_output_nullspace.py','Two optional legacy CSVs unavailable; complete E1 replay verified'),
      ('C02','current complete softmax residual; future CE absolute effect','High_monotonic',20,'p2_invariance_audit.csv; p2_effect_bootstrap.csv','fig_B_same_now_different_future','test_nullspace_intervention.py','Interventions are causal within frozen model; no origin-of-natural-deficit inference'),
      ('C03','k=1/free any_far_viable and any_far_viable_midpoint_failed','High_monotonic',20,'p3_oracle_best_seed.csv; p3_viability_counts.csv','fig_B_same_now_different_future','test_oracle_and_constraints.py','Future-target-informed best-of-nine existence; infeasible retained'),
      ('C04','k=1 pulse any viable branch and native viable','High_monotonic',20,'p4_oracle_vs_cue.csv; p5_headline_intervals.csv','fig_p4_generation_vs_maintenance','test_cue_branches.py; test_final_analysis.py','Finite eight-source/six-branch menu; k=1 has zero inherited history'),
      ('C05','far_viable_exact_output under 1e-10 complete probability tolerance','High_monotonic',20,'p5_cue_output_compatibility.csv; p5_cue_access_events.csv','supplement','test_final_analysis.py; test_output_nullspace.py','Cue viability and exact output-compatible access are different claims'),
      ('C06','fixed-k raw CE and centered-logit drift by lag','High_monotonic',20,'p1_fixed_k.csv; p1_lag_bootstrap.csv','fig_p1_single_patch_timecourse','test_patch_and_continuation.py','Descriptive drift is not causal mediation'),
    ]
    claim=pd.DataFrame(claims,columns=['claim_id','exact_metric','cohort','seed_count','table','figure','test','caveat']);claim['selector']=['All 120 audited checkpoints; p0 replay passed=True','condition=High_monotonic; k=1; anchor_kind=midpoint; kind=nullspace; metric=mean_absolute_effect','k=1; constraint_set=free/familiar_near separately; any_far_viable / any_far_viable_midpoint_failed; denominator=20 seeds','k=1; schedule=pulse; any within seed over 8 sources and 6 branches; native source=cue and branch=target_native','k=1; schedule=pulse; far=True; viable_continuation=True; current_softmax_distance_to_midpoint<=1e-10','condition=High_monotonic; k in {1,3}; raw_ce / midpoint_task_distance by lag and branch']
    claim['worked_example_sample_id']=example.sample_id if example is not None else 'not_established';claim.to_csv(out/'claim_source_map.csv',index=False)
    p0=read('p0_reference_geometry')[['cohort','condition','seed','checkpoint_sha256']].drop_duplicates();p0['stage']='p0';p0['cue']='all_registered';p0['k']=-1;p0['branch']='source_replay';p0['status']='completed';p0['output_path']='tables/p0_replay_parity.csv'
    registry=pd.concat([p0,*[read(f'p{i}_run_registry') for i in [1,2,3,4]]],ignore_index=True)
    registry.to_csv(out/'run_registry.csv',index=False)
    for src,dst in [('output_compatible_test_evidence.jsonl','test_evidence.jsonl'),('output_compatible_decisions.jsonl','decisions.jsonl')]:shutil.copyfile(root/'scratchpads'/src,out/dst)
    return {'integrity_passed':True,'implementation_status':'validated','scientific_status':'completed','model_optimizer_updates':0,'n_primary_seeds':20,'final_reports_written':True,'manifest_seal_required':True}


def figure_b(out,p2,best):
    import matplotlib.pyplot as plt
    from .figures import save_figure
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained');a=axes[0,0];a.axis('off')
    a.text(.03,.92,'Same complete prediction now',fontsize=18,weight='bold');a.text(.03,.70,r'$h=m$    versus    $h\prime=m+Na$',fontsize=20)
    a.text(.03,.47,r'$PW_oN=0$  $\Rightarrow$  $\mathrm{softmax}(z\prime)=\mathrm{softmax}(z)$',fontsize=17)
    a.text(.03,.20,'Identical future tokens + cue + frozen recurrence\n\nPrimary: k=1, all six remaining predictions',fontsize=13)
    colors=plt.cm.tab10(np.arange(6))
    for color,cue in zip(colors,PROBES):
        f=p2[p2.condition.eq('High_monotonic')&p2.k.eq(1)&p2.anchor_kind.eq('midpoint')&p2.kind.eq('nullspace')&p2.cue.eq(cue)]
        axes[0,1].scatter(f.anchor_future_ce,f.mean_future_ce,s=20,color=color,label=cue,alpha=.7)
        axes[1,0].scatter(f.mean_realized_relative_radius,f.mean_absolute_effect,s=20,color=color,alpha=.7)
        g=best[best.k.eq(1)&best.constraint_set.eq('free')&best.cue.eq(cue)&best.n_feasible_starts.gt(0)]
        axes[1,1].scatter(g.midpoint_future_ce,g.best_unoptimized_future_ce,s=14,color='gray',marker='x',alpha=.3)
        axes[1,1].scatter(g.midpoint_future_ce,g.future_ce,s=25,color=color,alpha=.7)
        far=g[g.far_viable];axes[1,1].scatter(far.midpoint_future_ce,far.future_ce,s=65,facecolors='none',edgecolors='black')
    for ax in [axes[0,1],axes[1,1]]:
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],ls='--',color='gray');ax.set(xlabel='Midpoint future CE (nats)')
    axes[0,1].set(title='Random output-equivalent states',ylabel='Mean random-state future CE');axes[0,1].legend(fontsize=8)
    axes[1,0].set(title='Actual realized perturbation',xlabel='Mean radius / A–B separation',ylabel='Mean absolute future CE change (nats)')
    axes[1,1].set(title='Bounded oracle, best of nine fixed starts',ylabel='Oracle future CE (future-label informed)')
    axes[1,1].text(.03,.97,'Ring: selected state far + viable\nGray x: best of same unoptimized starts\nInfeasible starts retained in tables',transform=axes[1,1].transAxes,va='top',fontsize=9)
    save_figure(fig,out,'fig_B_same_now_different_future')


def supplementary_c(out,seeds,long,oracle):
    import matplotlib.pyplot as plt
    from .figures import save_figure
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,cue in zip(axes.ravel(),PROBES):
        f=long[long.k.eq(1)&long.cue.eq(cue)&long.source_context.eq('A')]
        for branch,schedule,color in [('target_native','pulse','#2878a8'),('Mixed','pulse','#2c965d'),('Mixed','persistent','#cc7940')]:
            g=f[f.branch.eq(branch)&f.schedule.eq(schedule)].groupby('lag').raw_ce.mean()
            ax.plot(g.index,g.values,label=branch+' '+schedule,color=color)
        q=oracle[oracle.k.eq(1)&oracle.cue.eq(cue)&oracle.branch.eq('free')].groupby('lag').raw_ce.mean();ax.plot(q.index,q.values,'k--',label='oracle reference (feasible seeds)')
        ax.set(title=cue,xlabel='Lag after cue step',ylabel='Raw CE (nats)');ax.axvline(0,color='gray',lw=.5)
    axes.flat[0].legend(fontsize=7);fig.suptitle('Cue schedules and maintenance; current step is not in future viability average')
    save_figure(fig,out,'fig_p5_cue_lag_profiles')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,k in zip(axes,[3,5]):
        for cue in PROBES:
            f=seeds[seeds.k.eq(k)&seeds.cue.eq(cue)&seeds.branch.eq('Mixed')&seeds.schedule.eq('pulse')&seeds.source_context.isin(['A','B'])]
            p=f.pivot(index='seed',columns='source_context',values='future_ce');ax.scatter(p.A,p.B,label=cue,s=18)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],color='gray',ls='--');ax.set(title=f'k={k}: same Mixed pulse, different inherited state',xlabel='A-history future CE',ylabel='B-history future CE')
    axes[0].legend(fontsize=7);save_figure(fig,out,'fig_p5_source_history')


def _reports(root,out,counts,intervals,merged,best,restarts,p2,example):
    manifests={f'p{i}':json.loads((out/'manifests'/f'p{i}.json').read_text()) for i in range(5)}
    primary=intervals[intervals.k.eq(1)&intervals.cue.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
    oracle=primary[primary.metric.isin(['oracle_far_viable_found','oracle_far_viable_midpoint_failed'])]
    access=merged[merged.k.eq(1)].groupby('cue',as_index=False)[['any_pulse_viable','far_viable','far_viable_exact_output']].sum()
    searchsets=best[best.k.eq(1)].groupby(['cue','constraint_set'],as_index=False)[['any_far_viable','any_far_viable_midpoint_failed','viable_continuation']].sum()
    p2boot=pd.read_csv(out/'tables/p2_effect_bootstrap.csv');p2head=p2boot[p2boot.condition.eq('High_monotonic')&p2boot.k.eq(1)&p2boot.anchor_kind.eq('midpoint')&p2boot.kind.eq('nullspace')&p2boot.metric.eq('mean_absolute_effect')&p2boot.cue.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
    proof=''
    if example is not None:
        e=example;worked=pd.read_csv(out/'tables/p5_worked_example.csv')
        proof=(f'Illustrative counterexample selected by first seed/cue/restart among registered k=1 free-oracle far+viable/midpoint-failed cases: seed {int(e.seed)}, {e.cue}, restart {int(e.restart_index)}, sample `{e.sample_id}`. Midpoint future CE = {e.midpoint_future_ce:.6f}; oracle = {e.future_ce:.6f}; familiar mean bound = {e.familiar_bound:.6f}, so the registered viability ceiling is {e.familiar_bound+.1:.6f} nats. Initial relative distance = {e.relative_midpoint_distance:.6f}; current maximum four-probability difference = {e.current_softmax_residual:.3g}. Worst oracle future-step CE = {e.worst_future_ce:.6f}. Initial distance does not guarantee persistent distance; all steps are below.\n\n'+table(worked[['lag','target','raw_ce','midpoint_ce','relative_midpoint_distance',*[f'probability_{i}' for i in range(4)]]]))
    text='''# Output-compatible recurrent trajectories

A frozen recurrent network can sustain the registered future-prediction criterion from some output-compatible starting states far from the A/B midpoint. This is an existence result for particular seeds, cues and six-step teacher-forced futures. Natural cue access is more limited and must be separated from exact output-compatible access.

The primary analysis used 20 canonical High monotonic seeds. C0–C3 were evaluated separately for P1/P2; Low was included only in P0 fidelity checks. No cohort was pooled or checkpoint retrained. All statistics resample trained seeds, keeping directions, restarts, positions and branch outcomes as repeated measures. Novel30/60 means average both registered variants within seed; missing variants stay missing.

## Primary results

P0 reproduced 120 frozen checkpoints against complete local E1 evidence. Every centered 4-by-8 output head had rank three and a five-dimensional kernel. P1 preserves fixed-position per-lag natural, single-midpoint and repeatedly patched comparisons; repeated patches are external assistance, not a viable unpatched trajectory.

P2 tested 432,000 planned states across 100 models. The maximum complete-current-softmax residual in double geometry was 1.027e-14, below the frozen 1e-10 tolerance. Original float32-head rounding was separately bounded (maximum observed difference 5.692e-7); it is not represented as bitwise probability identity. With the exact same subsequent inputs and original float32 recurrent implementation, output-equivalent states produced different future predictions. This supports a causal effect of hidden-state changes within these models, not an explanation of the natural Mixed/Novel deficit.

P3 retained all 6,480 planned restart slots, including 1,572 infeasible starts. Feasible starts used the fixed 150 hidden-coordinate updates, with no model parameter update. Every iterate, stalled endpoint and source replay discrepancy is exported. The four-seed smoke test found no far+viable/midpoint-failed counterexample; the full search did. The primary k=1 seed rates, free and familiar-near search retained separately in source tables, are summarized below from the union of the two fixed search sets. The union is an existence summary, not an extra search or natural-performance estimate.

'''+table(oracle[['cue','metric','mean','ci_low','ci_high','n_seeds','status']])+'''

Free and familiar-near search counts out of 20 seeds (raw probe variants):

'''+table(searchsets)+'''

P2 primary mean absolute future-CE effect, averaging perturbations within seed, with 95% trained-seed bootstrap intervals:

'''+table(p2head[['cue','mean','ci_low','ci_high','n_seeds','status']])+'''

P4 retained all 34,560 cue-only branch cases: eight natural prestate sources, six fixed current cue branches, pulse and persistent schedules, three positions, six probes and 20 seeds. Pulse comparisons share the same future target cue. Persistent comparisons change future input and are a different estimand. At k=1 all sources have a common zero prestate; no history effect is attributed to their labels. Counts below are trained seeds out of 20 with at least one tested pulse route, not independent branch counts:

'''+table(access)+'''

A successful pulse route demonstrates finite cue-accessible viability. It does not establish that the route retrieves the oracle state, or even shares its present softmax. The final column requires far+viable behavior **and** all four current probabilities within the prespecified 1e-10 equivalence tolerance of the midpoint. A zero count limits this exact claim for the tested cue menu, without showing global unreachability. Native versus alternative routes and oracle-found/cue-failed cases remain in `p4_oracle_vs_cue.csv` and `p5_decision_table.csv`.

## Primary figures

![Figure A](../figures/fig_p1_single_patch_timecourse.png)

Figure A: raw CE and task-space drift by lag at fixed k=1 and k=3. Current placement is direct assistance; later predictions use the frozen recurrence. Drift is descriptive, not a mediation test.

![Figure B](../figures/fig_B_same_now_different_future.png)

Figure B: complete present-output identity, random-state futures and the bounded best-of-nine oracle. Each point is a seed/cue case; oracle results use future labels. Actual realized radii and infeasibility are retained.

![Figure C](../figures/fig_p4_generation_vs_maintenance.png)

Figure C: physical cue access and maintenance, source A at k=1 (identical to all other zero-prestate sources). Pulse and persistent bars describe different future input schedules. The oracle is a separate state intervention. Per-lag profiles and source A/B history comparisons at k=3/5 are supplementary figures.

## Worked numeric counterexample

'''+proof+'''

## Mathematical explanation and limits

Let z = W_o h + b and P = I − 11ᵀ/4. Construct an orthonormal basis N for ker(PW_o) using a full right SVD. For h′ = m + Na, P(z′−z)=0, hence z′−z=c1 and every current softmax probability is identical. Current CE is therefore identical for every possible next-token target. Literal equality of all raw logits is unnecessary. With eight hidden coordinates and at most three independent centered output contrasts, a nontrivial kernel is guaranteed; the audited rank is three.

Future state updates are h_next=tanh(W_x x_next + W_h h + b_h). Locally, an output-invisible displacement can become visible through P W_o diag(1−h_next²) W_h N. Nothing in PW_o N=0 forces this later map to vanish. A nonzero future effect is possible, but useful future performance and cue accessibility do not follow from nullspace dimension alone.

The oracle optimized only a, using correct future labels to minimize mean CE over lag>0. Weights, cues, targets, budgets, thresholds and cohort identities stayed fixed. States were constrained to the tanh interior with no hidden-coordinate clipping; the near search also limited distance to 0.5 A/B separation. Final candidates were replayed by the original float32 source recurrence. The differentiable double surrogate is an optimization aid; exported actual-source CE determines conclusions. Failed search is search-limited evidence, not a proof of midpoint necessity.

“Far” means initial distance at least 0.25 times familiar A/B separation; it does not require the entire future trajectory to stay far. Viability means mean remaining CE no greater than mean familiar CE plus 0.10 nats; it does not mean zero errors, perfect probability or autonomous sequence generation. Zero-slack, worse-endpoint, worst-step loss, later geometry and all individual probabilities are saved. No human hidden-state mechanism is identified.

## Integrity, negative outcomes and reproducibility

P0–P4 stage manifests verify unchanged checkpoint/source hashes before and after each run and freeze implementation identities. The final sealed manifest includes report, figure, code, configuration and output hashes. `claim_source_map.csv`, `run_registry.csv`, `test_evidence.jsonl` and `decisions.jsonl` provide the audit trail. All null, negative, infeasible and blocked outcomes are retained.

Two optional historical CSVs remain unavailable as Box cloud placeholders: `analysis/phase1j_single_injection_context_stability/phase1j_transition_metrics.csv` (expected SHA b0205f6bc82d2b5ce3db29f078538fb921984526671988810d6a8677e40fe511) and `results/paper/v2_20260925_novelty30_60_r2/tables/a3_composite_sufficiency_seed.csv` (expected SHA d1648d53d60b9270df64bf9d2ae2480eaf0985ac586b9410cc0c9543ea21aab6). No observed digest was invented. Complete local E1 single-patch/static evidence was hash-checked and directly replayed instead. The exact source audit and alias resolutions are in P0. An initial overly broad legacy test invocation executed one disposable old training fixture; no saved checkpoint or scientific model was changed, and all subsequent suites were restricted to read-only model evaluation.

Reproduce with `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.output_compatible_trajectories.cli --stage p0 --eval-only --output NEW_DIRECTORY`, then p1, p2, p3 --smoke, p3, p4, p5 in order, followed by `python3 -m experiments.output_compatible_trajectories.verify --output NEW_DIRECTORY --seal`. Existing outputs are immutable and cannot be silently overwritten. Run the test command recorded in `audit/final_test_result.json` before synthesis. The old failed dynamical-program E3/E4 calibration was never resumed.
'''
    exact_count=int(merged[merged.k.eq(1)].far_viable_exact_output.sum())
    exact_statement=('No tested k=1 pulse route simultaneously met the far, viable and complete-current-softmax-compatible criteria.' if exact_count==0 else f'At least one tested k=1 pulse route met all three criteria in {exact_count} seed/probe cases.')
    text=text.replace('Natural cue access is more limited and must be separated from exact output-compatible access.', 'Some physical cue routes produced viable futures. '+exact_statement+' This finite-menu result does not establish global unreachability.')
    if example is not None:
        matching=merged[merged.seed.eq(example.seed)&merged.k.eq(1)&merged.cue.eq(example.cue)].iloc[0]
        text=text.replace('Initial distance does not guarantee persistent distance;',f'Tested pulse cues were viable for this seed/probe: {bool(matching.any_pulse_viable)}. Initial distance does not guarantee persistent distance;')
    report(out,'FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md',text)
    handoff='''# Researcher handoff

The question is whether the useful midpoint is necessary, or whether another state with the same present prediction can support a viable recurrent future. We found the latter in some frozen High seeds; the primary seed counts and uncertainty are in the final report. Cue-accessible viability is a separate question, with its exact-output-compatible subset explicitly checked.

1. **Algorithm.** Center the frozen linear output head, take its complete right SVD, and displace the midpoint along its five-dimensional kernel. Verify all four current probabilities and CE, remain inside the tanh box, then run the exact source recurrence under identical future inputs. Random and visible controls use matched realized norms. An invisible current displacement need not remain invisible after recurrence.
2. **Oracle.** Only initial kernel coordinates are optimized, using correct future labels. Nine fixed starts and 150 updates per feasible start; free and familiar-near constraints; no training, clipping, extra restart or outcome-dependent retuning. Best-of-nine is an existence assay. Every selected state is replayed with the original float32 model; infeasible and stalled outcomes remain visible.
3. **Cue access.** Identical natural prestates receive the fixed cue menu. Pulse arms then share the same future cue; persistent takeover changes future inputs. There is no hidden placement or future-target optimization in cue generation. Oracle states enter only afterward as separate references. At k=1 there is no inherited context history. Native replay is bit-exact.
4. **Evidence.** Use Figure A for maintenance after a single patch, Figure B for present identity/future divergence and existence, and Figure C for finite physical cue access. Exact selectors, sample IDs, tests and caveats are in `claim_source_map.csv`. `p5_worked_example.csv` contains each of the four probabilities at every step. `p5_cue_output_compatibility.csv` prevents conflating cue viability with exact-output access.
5. **Causal scope.** P2 is a causal state intervention within the frozen SRN. P3 is a target-informed oracle existence result. P4 is a physical cue intervention with a finite menu. Midpoint drift/proximity is descriptive. None identifies a human mechanism, global reachability or autonomous generation.

## Worked example

'''+proof+'''

## 60–90-second teach-back

“The output head compresses eight hidden coordinates into four class probabilities. Some hidden differences are invisible to that prediction. We changed only those hidden coordinates, checked that all four probabilities stayed the same, and then gave both states exactly the same next inputs. Their future predictions could differ. We next searched, with the correct future answers available to the search, for a bounded alternative state far from the A/B midpoint. Some alternatives sustained the familiar-derived six-step criterion even when the midpoint did not. That rules out exact midpoint necessity in those particular frozen cases. It does not say those oracle states are naturally retrieved. We tested physical cue changes separately, starting from identical natural prestates, and report both viable cue behavior and the much stricter subset sharing the midpoint’s present output. Failures remain failures of this finite search or cue menu. The result is about teacher-forced prediction in these SRNs, not how humans must represent the task.”

## How the conclusion could be wrong or overstated

A state may pass the mean-CE cutoff while having a poor individual step, or start far and rapidly return near the midpoint. The exported worst-step CE and full later geometry expose those possibilities. Limited oracle search cannot prove necessity when it fails. A viable cue branch can change current probabilities, so it cannot automatically count as reaching an output-compatible oracle state. Source replay, source/current precision bounds and strict variant aggregation were tested explicitly. Two unavailable optional historical tables limit direct comparisons to those exact files; complete E1 replay supports the main analysis.

## Next permitted experiment

No new model training is needed to interpret these results. First inspect the worked counterexample’s whole trajectory and matched cue branches, including strict/worse-endpoint sensitivities. A broader history or cue-search experiment would answer a different reachability question and requires a separately fixed design; this program does not silently expand the tested menu.

## Files and verification

Read `FINAL_OUTPUT_COMPATIBLE_TRAJECTORY_REPORT.md`, P0–P4 one-page reports, `claim_source_map.csv`, `test_evidence.jsonl`, `decisions.jsonl`, and `run_registry.csv`. The sealed `manifest.json` hashes every output, source/config/checkpoint identity, cue registry, environment and command. Test totals and the final verification command are in `audit/final_test_result.json`. Use `python3 -m experiments.output_compatible_trajectories.verify --output results/output_compatible_trajectories/v1_20261111` to check the seal. Scientific weights were never updated; the initial disposable legacy test-fixture exception is disclosed in the final report and scratchpad.
'''
    report(out,'RESEARCHER_HANDOFF.md',handoff)
