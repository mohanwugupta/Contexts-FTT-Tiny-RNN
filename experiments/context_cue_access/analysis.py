"""S4 table-driven seed inference, claim ledger and researcher reports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiments.output_compatible_trajectories.nullspace import strict_novel_means,complete_bootstrap
from .config import PROBES,CONTRACT
from .io import save,report,table,dump


def truth(series):return series.isin([True,'True','true',1])


def novel_means(frame,keys,metrics):
    f=frame.copy()
    for c in metrics:f[c]=pd.to_numeric(f[c])
    return strict_novel_means(f,['seed',*keys],metrics)


def seed_intervals(frame,metrics,keys):return complete_bootstrap(frame,metrics,keys)


def subgroup_summary(frame):
    keys=[c for c in ['cue','k','policy','arm'] if c in frame];rows=[]
    for key,g in frame.groupby(keys,dropna=False):
        if g.seed.duplicated().any():raise ValueError('Subgroup repeated measures not collapsed within trained seed')
        eligible=truth(g.p3_oracle_positive_native_failing);sub=g[eligible];valid=pd.notna(sub.viable_continuation)
        rows.append({**dict(zip(keys,key)),'n_full':g.seed.nunique(),'n_full_valid':int(pd.notna(g.viable_continuation).sum()),'n_subgroup':len(sub),'n_subgroup_valid':int(valid.sum()),
            'n_prior_unavailable':int(g.p3_reference_status.ne('available').sum()),'n_viable_subgroup':int(truth(sub.viable_continuation).sum()),
            'subgroup_viability_rate':float(truth(sub.viable_continuation).mean()) if len(sub) and valid.all() else np.nan,'status':'complete' if len(sub) and valid.all() else 'no_eligible_cases' if not len(sub) else 'incomplete_source_outcomes'})
    return pd.DataFrame(rows)


def source_claim(surrogate_loss,actual_loss,ceiling,oracle):
    if not np.isfinite(ceiling):return 'not_established_missing_reference'
    if not np.isfinite(actual_loss) or actual_loss>ceiling:return 'source_replay_failed_viability'
    return 'oracle_selected_source_confirmed' if oracle else 'native_source_confirmed'


def paired_history(summary):
    rows=[]
    for (seed,cue,k),g in summary.groupby(['seed','cue','k']):
        for policy in ['fixed_candidates','bounded_oracle']:
            x=g[g.policy.eq(policy)]
            if x.arm.duplicated().any() or set(x.arm)!=set(['E','L','R','I']):raise ValueError('Missing/duplicate matched history arms')
            f=x.set_index('arm');r=f.loc['R'];l=f.loc['L'];e=f.loc['E'];i=f.loc['I']
            def crossing(a,b):return float(bool(a) and not bool(b)) if pd.notna(a) and pd.notna(b) else np.nan
            rows.append({'seed':seed,'cue':cue,'k':k,'policy':policy,'R_minus_L_future_ce':r.future_ce-l.future_ce,'E_minus_L_future_ce':e.future_ce-l.future_ce,'I_minus_R_future_ce':i.future_ce-r.future_ce,
                'R_minus_L_viability':float(r.viable_continuation)-float(l.viable_continuation),'R_success_L_failed':crossing(r.viable_continuation,l.viable_continuation),
                'R_success_both_singles_failed':crossing(r.viable_continuation,bool(e.viable_continuation or l.viable_continuation)) if pd.notna(e.viable_continuation) and pd.notna(l.viable_continuation) else np.nan,
                'L_success_R_failed':crossing(l.viable_continuation,r.viable_continuation),'I_success_R_failed':crossing(i.viable_continuation,r.viable_continuation),
                'native_viable':r.native_viable,'p3_oracle_positive_native_failing':r.p3_oracle_positive_native_failing,'p3_reference_status':r.p3_reference_status})
    return pd.DataFrame(rows)


def derived_tables(data):
    data=Path(data);s1=pd.read_csv(data/'tables/s1_best_cues.csv',float_precision='round_trip');s2=pd.read_csv(data/'tables/s2_seed_summary.csv',float_precision='round_trip')
    p=paired_history(s2);old=pd.read_csv(data/'tables/s2_paired_effects.csv',float_precision='round_trip')
    joined=p.merge(old,on=['seed','cue','k','policy'],suffixes=('_reconstructed','_original'),validate='one_to_one')
    for c in ['R_minus_L_future_ce','E_minus_L_future_ce','I_minus_R_future_ce','R_minus_L_viability','R_success_L_failed','L_success_R_failed','I_success_R_failed']:
        np.testing.assert_allclose(joined[c+'_reconstructed'],joined[c+'_original'],atol=1e-12,rtol=0,equal_nan=True)
    v1=s1[['seed','cue','k','policy','future_ce','viable_continuation']].copy();v1=novel_means(v1,['k','policy'],['future_ce','viable_continuation'])
    v2=s2[['seed','cue','k','policy','arm','future_ce','viable_continuation']].copy();v2=novel_means(v2,['k','policy','arm'],['future_ce','viable_continuation'])
    metrics=['R_minus_L_future_ce','E_minus_L_future_ce','I_minus_R_future_ce','R_minus_L_viability','R_success_L_failed','R_success_both_singles_failed','L_success_R_failed','I_success_R_failed']
    vp=novel_means(p[['seed','cue','k','policy',*metrics]],['k','policy'],metrics)
    return {'s4_s1_intervals':seed_intervals(v1,['future_ce','viable_continuation'],['cue','k','policy']),
        's4_s2_intervals':seed_intervals(v2,['future_ce','viable_continuation'],['cue','k','policy','arm']),
        's4_paired_intervals':seed_intervals(vp,metrics,['cue','k','policy']),
        's4_s1_subgroups':subgroup_summary(s1),'s4_s2_subgroups':subgroup_summary(s2),'s4_paired_reconstructed':p}


def decision_ledger(s1,s2):
    rows=[]
    for (seed,cue,k),g in s2.groupby(['seed','cue','k']):
        pulse=s1[s1.seed.eq(seed)&s1.cue.eq(cue)&s1.k.eq(k)];native=bool(g.native_viable.iloc[0]);single=bool(truth(pulse[pulse.policy.isin(['best_extended_grid','bounded_oracle'])].viable_continuation).any())
        fixed=g[g.policy.eq('fixed_candidates')].set_index('arm');oracle=g[g.policy.eq('bounded_oracle')].set_index('arm')
        def known_cross(frame):
            a=frame.loc['R','viable_continuation'];b=frame.loc['L','viable_continuation']
            return bool(a and not b) if pd.notna(a) and pd.notna(b) else np.nan
        all_values=pd.concat([pulse.viable_continuation,g.viable_continuation]);unknown=all_values.isna().any()
        p3_available=g.p3_reference_status.iloc[0]=='available';p3positive=bool(g.p3_oracle_positive_native_failing.iloc[0]) if p3_available else np.nan
        rows.append({'seed':seed,'cue':cue,'k':k,'native_viable':native,'alternative_single_cue_rescue':bool(single and not native),'repeated_fixed_success_late_failed':known_cross(fixed),
           'repeated_oracle_success_late_failed':known_cross(oracle),'early_oracle_viable':oracle.loc['E','viable_continuation'],
           'only_independent_oracle_viable':bool(oracle.loc['I','viable_continuation'] and not truth(oracle.loc[['E','L','R'],'viable_continuation']).any()) if not oracle.viable_continuation.isna().any() else np.nan,
           'finite_bounded_access_gap':bool(p3positive and not truth(all_values).any()) if p3_available and not unknown else np.nan,'prior_status':g.p3_reference_status.iloc[0],
           'unknown_search_outcomes':bool(unknown),'S1_S2_same_estimand':False})
    return pd.DataFrame(rows)


def subgroup_intervals(frame):
    keys=[c for c in ['cue','k','policy','arm'] if c in frame];rows=[]
    for key,g in frame.groupby(keys):
        g=g[truth(g.p3_oracle_positive_native_failing)];base=dict(zip(keys,key))
        if not len(g):
            for metric in ['future_ce','viable_continuation']:rows.append({**base,'metric':metric,'n_seeds':0,'mean':np.nan,'ci_low':np.nan,'ci_high':np.nan,'status':'no_eligible_cases'})
        else:
            g=g.copy();g['viable_continuation']=pd.to_numeric(g.viable_continuation)
            rows.extend(seed_intervals(g,['future_ce','viable_continuation'],keys).to_dict('records'))
    return pd.DataFrame(rows)


def run(root,out):
    read=lambda n:pd.read_csv(out/'tables'/f'{n}.csv',float_precision='round_trip')
    tables=derived_tables(out)
    for name,f in tables.items():save(out,name,f)
    s1=read('s1_best_cues');s2=read('s2_seed_summary');paired=tables['s4_paired_reconstructed'];decision=decision_ledger(s1,s2);save(out,'s4_decision_ledger',decision)
    save(out,'s4_s1_subgroup_intervals',subgroup_intervals(s1));save(out,'s4_s2_subgroup_intervals',subgroup_intervals(s2))
    for n in sorted(set(range(1,21))):
        p=out/'audit'/f'bootstrap_indices_n{n}.npy'
        with p.open('xb') as f:np.save(f,np.random.default_rng(20260925).integers(0,n,(10000,n)))
    confirmations=[]
    for stage in ['s1','s2']:
        attempts=read(stage+'_oracle_attempts');attempts['stage']=stage;attempts['confirmation']=[source_claim(r.best_smooth_loss,r.future_ce,r.familiar_bound+.1,True) for r in attempts.itertuples()]
        attempts['surrogate_would_pass']=attempts.best_smooth_loss.le(attempts.familiar_bound+.1);confirmations.append(attempts)
    confirmations=pd.concat(confirmations,ignore_index=True);save(out,'s4_oracle_source_confirmation',confirmations)
    from .plots import render_all
    render_all(out,out,tables)
    examples=worked_examples(s2,paired);save(out,'s4_worked_examples',examples)
    claims=pd.DataFrame([
      ('C01','k1 any viable grid cue vs native, separately line/square/extended','s1_best_cues.csv; s4_s1_intervals.csv','fig_A_single_cue_accessibility','T09–T17; T33–T37','Cue selection uses future labels; extended signed box is numeric capacity'),
      ('C02','k3 R−L future CE, paired by trained seed; fixed/oracle separate','s2_paired_effects.csv; s4_paired_intervals.csv','fig_B_cue_history','T18–T26; T33–T34','Same two parameters does not eliminate optimizer difficulty; all future inputs fixed'),
      ('C03','R viable/L fails; early-only and independent alternatives retained','s2_seed_summary.csv; s4_decision_ledger.csv','fig_B_cue_history','T19–T24; T35–T37','Finite search only; I has four parameters and larger fixed menu'),
      ('C04','late vs tied vs independent ranks and label-informed gradient alignment','s3_sensitivities.csv; s3_finite_difference_audit.csv','fig_C_local_control','T27–T32','Local sensitivity/association is not causal mediation or global reachability'),
      ('C05','prior P3-positive/native-failing subgroup denominator and bounded gap','s4_s1_subgroups.csv; s4_s2_subgroups.csv; s4_decision_ledger.csv','tables','T01–T08; T35','Raw Novel sides only for subgroups; unavailable references NA'),
      ('C06','native replay, source CE confirmations and before/after hashes','native_replay.csv; s4_oracle_source_confirmation.csv','audit','T01–T08; T16; T26; T37–T38','All reported behavioral conclusions use original float32 source'),
    ],columns=['claim_id','exact_metric','table','figure','tests','alternative_or_limit'])
    claims['cohort']='canonical High_monotonic';claims['n_independent_seeds']=20;claims['primary_positions']='S1 k1; S2 k3; k5 separate secondary';claims['uncertainty']='10000 paired seed-cluster bootstrap draws; all raw probe sides before within-seed Novel means'
    save(out,'claim_source_map',claims)
    _reports(out,s1,s2,paired,tables,decision,examples,confirmations)
    registry=[]
    for stage in ['s1','s2']:
        f=read(stage+'_oracle_attempts');cols=[c for c in ['seed','cue','k','arm','restart_index','status','updates_executed','objective_evaluations'] if c in f];g=f[cols].copy();g['stage']=stage;g['record_type']='cue_oracle_attempt';g['output_path']='tables/'+stage+'_oracle_attempts.csv';registry.append(g)
    for stage,name in [('s1','s1_grid_seed'),('s2','s2_candidate_schedules')]:
        f=read(name);cols=[c for c in ['seed','cue','k','arm','candidate_id','cue_id'] if c in f];g=f[cols].copy();g['stage']=stage;g['status']='completed';g['record_type']='fixed_input_intervention';g['output_path']='tables/'+name+'.csv';registry.append(g)
    pd.concat(registry,ignore_index=True).to_csv(out/'audit/run_registry.csv',index=False)
    return {'model_updates':0,'hidden_placements':0,'n_independent_seeds':20,'final_reports_written':True,'final_seal_required':True,'all_source_confirmations_recorded':True}


def worked_examples(s2,paired):
    choices=[]
    for policy in ['fixed_candidates','bounded_oracle']:
        f=paired[paired.k.eq(3)&paired.policy.eq(policy)&~truth(paired.native_viable)&truth(paired.R_success_L_failed)].sort_values(['seed','cue'])
        if len(f):
            first=f.iloc[0];g=s2[s2.seed.eq(first.seed)&s2.cue.eq(first.cue)&s2.k.eq(3)&(s2.policy.eq(policy)|s2.arm.eq('N0'))].copy();g['example_role']='history_advantage';choices.append(g);break
    f=paired[paired.k.eq(3)&paired.policy.eq('bounded_oracle')&paired.R_minus_L_future_ce.ge(0)].sort_values(['seed','cue'])
    if len(f):
        first=f.iloc[0];g=s2[s2.seed.eq(first.seed)&s2.cue.eq(first.cue)&s2.k.eq(3)&(s2.policy.eq('bounded_oracle')|s2.arm.eq('N0'))].copy();g['example_role']='negative_repeated_advantage';choices.append(g)
    return pd.concat(choices,ignore_index=True) if choices else pd.DataFrame([{'status':'no_case_met_prespecified_example_selectors'}])


def _reports(out,s1,s2,paired,tables,decision,examples,confirmations):
    s1counts=s1[s1.k.eq(1)].groupby(['cue','policy'],as_index=False).agg(n_seeds=('seed','nunique'),viable=('viable_continuation','sum'),mean_future_ce=('future_ce','mean'))
    primary=tables['s4_paired_intervals'];headline=primary[primary.k.eq(3)&primary.policy.eq('bounded_oracle')&primary.metric.eq('R_minus_L_future_ce')]
    rawheadline=headline[headline.cue.isin(PROBES)];fixed=primary[primary.k.eq(3)&primary.policy.eq('fixed_candidates')&primary.metric.eq('R_minus_L_future_ce')&primary.cue.isin(PROBES)]
    h=paired[paired.k.eq(3)&paired.policy.eq('bounded_oracle')].groupby('cue',as_index=False)[['R_success_L_failed','R_success_both_singles_failed','L_success_R_failed','I_success_R_failed']].sum()
    counts=s2[s2.k.eq(3)&(s2.policy.eq('bounded_oracle')|s2.arm.eq('N0'))].groupby(['cue','arm'],as_index=False).agg(n_seeds=('seed','nunique'),n_viable=('viable_continuation','sum'),mean_future_ce=('future_ce','mean'))
    fd=pd.read_csv(out/'tables/s3_finite_difference_audit.csv');sens=pd.read_csv(out/'tables/s3_sensitivities.csv')
    rank=sens[sens.k.eq(3)&sens.schedule_kind.eq('N0')].groupby('matrix',as_index=False)[['rank_absolute','rank_relative','rank_combined','gradient_alignment']].mean()
    informative=bool((rawheadline.ci_high<0).any());cposition='primary diagnostic' if informative else 'supplemental diagnostic; no raw-probe interval met the prespecified display gate'
    exampletext='No case met the prespecified worked-example selector; this remains inconclusive.'
    if 'example_role' in examples:
        parts=[]
        for role,g in examples.groupby('example_role',sort=False):
            r=g.iloc[0];cols=['arm','policy','future_ce','current_ce','full_sequence_ce','viable_continuation','familiar_bound','selected_0','selected_1','selected_2','selected_3']
            parts.append(f'### {role}: seed {int(r.seed)}, {r.cue}, k=3\n\n'+table(g[cols])+f'\n\nThe source criterion ceiling is {r.familiar_bound+.1:.6f} nats. Every arm receives the physical target cue at step3 and scores the same four predictions at steps4–7. Selected cues are found using future labels. '+('Early-only E also passes: this example does not establish that two updates are necessary.' if bool(g[g.arm.eq('E')].viable_continuation.iloc[0]) else 'Early and late single updates both fail in this example under their registered finite searches.')+'\n')
        exampletext='\n'.join(parts)
    discrepancy=float(confirmations.source_minus_surrogate_ce.abs().max());failed=confirmations[confirmations.status.ne('completed')];sat=[]
    for stage in ['s1','s2']:
        hist=pd.read_csv(out/'tables'/f'{stage}_oracle_history.csv',usecols=['saturated']);sat.append((stage,int(hist.saturated.sum())))
    subgroup=tables['s4_s1_subgroups'];denoms=subgroup[subgroup.k.eq(1)&subgroup.policy.eq('bounded_oracle')][['cue','n_full','n_subgroup','n_prior_unavailable','n_viable_subgroup','status']]
    docs=f'''# Context cues and access to viable recurrent computation

The frozen context pathway can support useful future behavior under alternative numeric cues. Short preparation also changes future performance with the final physical probe cue held fixed. The magnitude, viability crossings and adverse cases are reported separately for every probe and seed; there is no universal mechanism label for all networks.

## What was fixed and what changed

We evaluated the same20 canonical High monotonic checkpoints from P3/P4, scoring only the four-class next-token head. All weights, checkpoint bytes, sequence tokens, cue encoding and prior viability labels remained fixed. Every source rollout begins at zero and calls the original float32 recurrent step. No hidden state was placed, trained or used as a cue-optimization target. The three supplied PRD package files are preserved in docs/context_cue_access and hash-indexed in S0.

S1 changes only the current cue at k; its natural prestate and future target cues are identical. S2 begins from the same natural anchor, processes the same three prep/final tokens, and gives **every arm the same physical target cue at k and afterward**. At k3 that anchor is zero h0; at k5 it is natural-target h2. Only positions k+1…7 enter future viability: six/four/two predictions at k1/3/5. Current CE and full-sequence CE are exported separately. This is teacher-forced prediction, not autonomous generation.

Viability imports `experiments/output_compatible_trajectories/continuation.py:future_summary` unchanged: future CE ≤ position-specific mean familiar A/B future CE +0.10 nats. Strict zero slack, worse-endpoint sensitivity and worst-step CE remain in the source tables. No current-output equivalence constraint is imposed on cue-generated states.

## S1 — alternative single-cue capacity

All cases received169 prespecified grid points on [-1.5,1.5]² and eight exact named anchors. No viability interpolation or smoothing was used. The third input coordinate remains zero. Alpha=u−v, beta=u+v, norm and domain membership are exported separately. Grid points and anchors are repeated interventions within a trained seed, not new networks.

Primary k1 counts out of20 seeds:

'''+table(s1counts)+'''

The familiar line and square are subsets of the extended numeric box. Signed extrapolations establish mathematical input-pathway capacity, not psychological plausibility. The bounded cue oracle received six starts×80 Adam updates at LR0.04, with two cue parameters only. Starts were A/B/Mixed/zero/target/best-grid, even if values coincided. Both grid-best selection and oracle selection use future performance; neither demonstrates natural selection or retrieval. Inference uses all20 seeds. The prospectively defined P3-viable/native-failing subgroup remains secondary:

'''+table(denoms)+'''

All other k values and subgroup confidence intervals are in s4_s1_intervals and s4_s1_subgroup_intervals. Novel sides are retained before full-cohort within-seed means. Subgroup denominators differ across sides, so subgroup aggregates are shown only for raw cues.

## S2 — same-final-target history, matched parameter count

E changes the early preparatory cue; L the late cue; R uses **the same two scalar cue parameters at both preparatory steps**. I has four independently controlled scalars. N0 is unmodified. E/L/R each received the same177 fixed candidate values and six×80 oracle updates; I received512 deterministic Sobol pairs plus531 nesting controls and the same oracle updates. I therefore has both more parameters and a larger finite menu. Its advantage alone cannot isolate temporal accumulation.

Primary k3 bounded-oracle paired R−L mean future CE (negative favors repeated preparation), with95% trained-seed bootstrap intervals:

'''+table(headline[['cue','mean','ci_low','ci_high','n_seeds','status']])+'''

The matched finite candidate comparison, without continuous cue optimization, gives:

'''+table(fixed[['cue','mean','ci_low','ci_high','n_seeds','status']])+'''

Actual-source k3 viability and CE by arm:

'''+table(counts)+'''

Discordant cases under the bounded searches:

'''+table(h)+'''

A repeated-cue success while late-only fails demonstrates an advantage under the registered matched-parameter search, not that every possible single cue fails. Early-only success is an alternative route; independent-only success remains confounded by extra control and fixed-menu size. All adverse R−L cases, k5 secondary comparisons and failed searches are preserved. Intervals are descriptive per registered probe; no multiplicity-adjusted confirmation or universal-history claim is made. S1 pulses and S2 preparation are not pooled as one estimand.

## S3 — minimal local control audit

At each actual source state, B_t=diag(1−h_t²)W_c and J_t=diag(1−h_t²)W_h. Late sensitivity at h_k is J_k B_(k−1); early sensitivity is J_k J_(k−1) B_(k−2). Tied repetition has sensitivity early+late (8×2). Independently controlled preparation concatenates early and late (8×4). Extra time does not guarantee new usable directions. Cue norm is normalized by the familiar A norm (one), so normalized_cue_norm equals the raw Euclidean norm here.

Baseline k3 descriptive averages across registered cases (raw per-seed/cue singular values and ranks are exported):

'''+table(rank)+f'''

Absolute rank uses1e-8; relative rank uses1e-4 times the largest singular value; combined rank requires both. Gradient alignment is the fraction of future-CE gradient norm in the retained column span, and is undefined for zero norm. This label-informed diagnostic does not model natural cue selection.

The actual float32 source FD audit retained {len(fd)} rows at epsilon1e-3 and1e-4, with {int(fd.boundary.sum())} one-sided boundary checks and {int(fd.poor_linear_approximation.sum())} flagged poor approximations. The separate maxima were absolute error {fd.max_abs_error.max():.6g} and relative norm error {fd.relative_norm_error.max():.6g}; zero joint flags do not mean bitwise derivative equality, and large relative errors can occur for near-zero derivatives. Boundaries stay inside the fixed box. Saturation/roundoff errors are reported, not repaired by changing epsilon or rank thresholds. Analytic derivatives agreed with autodiff and both source FD scales on nonsaturated synthetic fixtures. FigureC is a {cposition}. Correlation of alignment or rank with behavioral benefit is not causal mediation or global controllability evidence. No fixed-point, attractor, Lyapunov or autonomous-eigenspectrum analysis was performed.

## Figures

![Single-cue map](figures/fig_A_single_cue_accessibility.png)

FigureA: unsmoothed k1 cohort cue maps; color is mean future CE, markers show the number of viable seeds at the exact grid input. Domain boundaries distinguish familiar mixtures, square and extrapolative numeric input. Bars retain all20 seeds and label future-informed selection. Per-seed exemplar selection uses the lowest numeric rescued seed and is supplementary.

![Matched history](figures/fig_B_cue_history.png)

FigureB: every k3 seed’s N0/E/L/R/I source losses, viability counts, paired R−L uncertainty and the same-final-cue schematic. I is explicitly four parameters. k5 is a separate figure.

![Local control diagnostic](figures/fig_C_local_control.png)

FigureC: local matrix ranks and within-seed control-alignment difference versus realized history benefit. This is explanatory sensitivity evidence only.

## Worked and negative cases

'''+exampletext+f'''

Some preparations worsen current or earlier predictions while improving the registered future. Current CE and full-sequence CE expose that cost; future viability is not a claim of better full-sequence accuracy.

## Integrity, null findings and uncertainty

S0 reproduced360 native cue/position outcomes and the full present probabilities/logits. S1 saved63,720 fixed-grid/anchor cases and2,160 oracle attempts. S2 saved378,000 schedules and5,760 attempts. All80 updates and81 objective evaluations per finite start were retained; nonfinite/saturated attempts were not replaced. There were {len(failed)} non-completed oracle attempt rows. Saturated history-row counts were {sat}. These are numerical diagnostics, not discarded cases. Maximum absolute selected source-minus-smooth CE discrepancy was {discrepancy:.6g} nats. Actual source CE determines every outcome even when it disagrees with the surrogate.

Bootstraps resample trained seed clusters with seed20260925 and10,000 draws. Full-cohort raw cue sides and within-seed Novel means are separate; no grid point, optimizer restart or transition is an independent network. Missing prior references and unknown source outcomes are explicit NA/status, never zeros. No old model or scientific output was overwritten. The prior dynamical program’s failed E3/E4 gate remains untouched.

The interpretation is finite-horizon, bounded-input capacity in this canonical High sample. Search-selected cues use future answers, signed cues may be implausible stimuli, and results may depend on the mean-CE cutoff or short horizon. An extra prep step may help through nonlinear saturation and cue-history geometry without introducing a new independent direction. Direct P3 state oracles are reference evidence only; no cue objective tried to match a P3 state. Finite search failure cannot establish global impossibility.

## Claim ledger and reproducibility

`tables/claim_source_map.csv` links exact outcomes, figures, tests, caveats and denominators. `CLAIM_LEDGER.md` is the one-page interpretation ledger. `audit/test_evidence.jsonl` and `tests/red_green_log.jsonl` preserve RED/GREEN/refactor commands and outputs; scratchpad.md is append-only. S0–S4 manifests record unchanged source/checkpoint hashes and implementation identities. Final test totals and the seal verification command are in audit/final_test_result.json. The final sealed manifest hashes every output, including null, negative, failed, bounded and audit records.

Reproduce in a fresh output directory after copying the supplied scratchpad template: `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.context_cue_access.cli --stage s0 --eval-only --output NEW_DIRECTORY`, then s1, s2, s3, s4 in order. Existing stage files are exclusive. Seal with `python3 -m experiments.context_cue_access.seal --output NEW_DIRECTORY --seal`; omit --seal to verify. Regenerate key summaries and figures without model evaluation using `python3 -m experiments.context_cue_access.regenerate --source RUN_DIRECTORY --destination EMPTY_DIRECTORY`; this checks sealed identities and writes only to the new directory.
'''
    nc=headline[headline.cue.eq('NoContext')].iloc[0];mx=headline[headline.cue.eq('Mixed')].iloc[0]
    mg=s1counts[s1counts.cue.eq('Mixed')&s1counts.policy.eq('best_extended_grid')].iloc[0];mn=s1counts[s1counts.cue.eq('Mixed')&s1counts.policy.eq('native')].iloc[0]
    short=f'At k1, the extended grid supported viable Mixed futures in{int(mg.viable)}/{int(mg.n_seeds)} seeds versus{int(mn.viable)}/{int(mn.n_seeds)} with the native cue. At k3, repeated minus late-only preparation gave NoContext mean CE {nc["mean"]:.3f} nats (95% seed interval {nc.ci_low:.3f} to {nc.ci_high:.3f}); Mixed was {mx["mean"]:.3f} ({mx.ci_low:.3f} to {mx.ci_high:.3f}), an inconclusive mean advantage. Early-only controls often also succeed, so repetition is not generally necessary.'
    docs=docs.replace('The frozen context pathway can support useful future behavior under alternative numeric cues.',short+'\n\nThe frozen context pathway can support useful future behavior under alternative numeric cues.')
    report(out,'FINAL_CONTEXT_CUE_ACCESS_REPORT.md',docs)
    ledger='''# One-page claim ledger

| Claim | Exact evidence | Alternative explanation / uncertainty |
| --- | --- | --- |
| Alternative current cues have input-pathway capacity | S1 k1 per-seed source CE/viability, familiar-line/square/extended grid and oracle separately | Future-answer-informed selection; signed-box cues need not be plausible; no natural retrieval claim |
| Preparation can alter access with a fixed final target | S2 k3 paired R−L source CE and discordant viability; E and I controls; k5 secondary | R/L equal parameters and budget, but optimizer landscapes differ; finite search cannot rule out untested single cues |
| Independent two-step input can provide a capacity ceiling | I source rollouts and explicit nesting registry | Four parameters and larger finite menu confound comparison with R/L |
| Local input sensitivities may explain a history effect | Raw late/tied/independent singular values, ranks, future-label gradient alignment and two-scale source FD | Saturation, numerical FD error and association do not prove mediation or global reachability |
| Failed bounded cue access may coexist with P3 state existence | Per-case prior-positive/native-failing denominator and all tested source outcomes | Prior hidden-state oracle used future labels; missing reference is NA; finite-search failure is not impossibility |

All primary results retain20 trained seeds per raw cue. Confidence intervals resample seeds, not cues/restarts/steps. Exact selectors and test IDs are in tables/claim_source_map.csv. No network parameter, threshold, learning rate or budget changed after outcome inspection.
'''
    report(out,'CLAIM_LEDGER.md',ledger)
    hand=f'''# Researcher handoff

## Question and hypothesis

Can an alternative bounded context cue, or a short history of cues, access a useful recurrent computation in the existing frozen High networks? The hypotheses are wrong current signal, history-dependent prepared-state access, and a finite bounded input-access gap; they can differ by seed/probe.

## Algorithm

S1 enumerated169 current-cue values plus eight exact anchors, holding natural prestate and all future target cues fixed. S2 processed identical tokens from a common natural anchor, varied early/late prep cues, then fixed the physical target cue at k and afterward. E/L/R expose two scalars; repeated R ties the same values at both prep steps; I exposes four. Continuous cue oracles used six starts×80 Adam updates, LR0.04, c=1.5*tanh(z), fixed betas/epsilon, and correct future targets. All selected cues were replayed with the original float32 source. The only optimized variables were cue scalars. P3 states were references, never matching objectives.

The imported P3 criterion is mean remaining CE no greater than position-specific mean familiar A/B CE+0.10. At k1/3/5 the future has six/four/two predictions. Current/prep CE and full-sequence CE are recorded separately. S3 calculated exact local chain maps: late=J_k B_(k−1), early=J_k J_(k−1) B_(k−2), tied=early+late, independent=[early late]. Absolute/relative rank, label-informed gradient alignment and two source FD scales were retained.

## What changed and what stayed fixed

Cues changed at the specified steps. Sequence tokens, frozen checkpoints, all learned weights/heads, input encoding, final S2 target cue, future S1/S2 cues and viability thresholds stayed fixed. Every actual rollout starts from zero and uses source recurrence; k5 arms share the same natural h2. No hidden-state intervention or model training was performed. S1 current-pulse capacity and S2 preparation are separate estimands.

## Primary full-cohort evidence

S1 k1 counts out of20, by raw probe/domain/policy, are in tables/s1_primary_counts.csv and FigureA. S2 k3 bounded-oracle R−L CE with95% seed intervals:

'''+table(headline[['cue','mean','ci_low','ci_high','n_seeds','status']])+'''

Every seed and adverse case is retained. Compare fixed candidates to bounded oracles; inspect E before attributing an advantage uniquely to repetition. I is four parameters and a larger finite menu. Per-case viability, subgroup denominators and current/per-step CE are in the source tables; k5 is separate.

## Worked example and negative example

'''+exampletext+f'''

## Assumptions and causal scope

These are short teacher-forced futures, a fixed numeric cue box and20 canonical High networks. Grid/oracle cue selection uses future answers unavailable to natural retrieval. Signed/extrapolative cues may not be plausible stimuli. Causal input interventions demonstrate behavior in these frozen models; local rank and gradient-alignment associations do not identify mediation or global reachability. The current-output equality constraint from the prior program is deliberately absent here. FD flagged {int(fd.poor_linear_approximation.sum())}/{len(fd)} under the joint absolute/relative criterion; separate max absolute error={fd.max_abs_error.max():.6g}, max relative error={fd.relative_norm_error.max():.6g}. All are retained. Maximum selected source/surrogate CE discrepancy was {discrepancy:.6g} nats; source replay always determines claims.

## 60–90-second researcher teach-back

“We asked whether these already-trained networks have a useful sequence computation that their context input can access. First we changed one current cue while keeping its starting state and everything afterward fixed. The broader cue grid found useful futures in more cases than ordinary target cues, but selecting those cues used future answers. We then asked a different question: can preparation help when the final physical target cue is identical? Every arm saw the same tokens and future inputs. Late-only and repeated preparation each had two free numbers, and every oracle got the same starts and update budget. Their paired loss and viability results tell us whether repeating a cue helps under this finite search. Earlier-only and four-parameter independent controls keep that interpretation honest. Finally we measured which local state directions each input schedule can influence and checked those derivatives against the actual recurrence. These results concern bounded input capacity and short forced predictions. They do not mean the network naturally chooses the optimized cue or that a failed search proves a state is unreachable.”

## Decisive uncertainty and smallest permitted next step

The strongest alternative explanation for oracle differences is unequal optimization difficulty despite identical parameter counts and update budgets; fixed-candidate results are the first check. A mean-CE criterion can also hide a bad step, so inspect ce_1…ce_7, worst_future_ce and full_sequence_ce for the worked and negative cases. The smallest next action is a read-only comparison of these already exported controls, not another training run. A changed cue domain, longer history or search budget requires a separately fixed protocol; no expansion is part of v1.

## Files and verification

Start with FINAL_CONTEXT_CUE_ACCESS_REPORT.md, CLAIM_LEDGER.md and FiguresA/B. FigureC is a {cposition}. Tests and source replay evidence are in audit/; original RED failures and GREEN/refactor logs are preserved. The final seal hashes every output and source/checkpoint identity. Use `PYTHONPATH=.:src python3 -m experiments.context_cue_access.seal --output results/context_cue_access/v1` to verify, and the regenerate command in the final report to rebuild figures/key summaries in a fresh directory without model evaluation.
'''
    report(out,'RESEARCHER_HANDOFF.md',hand)
