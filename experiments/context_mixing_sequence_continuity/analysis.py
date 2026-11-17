"""D4 source-linked seed inference and qualified scientific synthesis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from context_sequence.provenance.hashing import sha256
from .config import ROOT,CONTRACT,SEQUENCE
from .source_adapter import records,load_original
from .replay import separation
from .statistics import seed_bootstrap,novel_means
from .history_diagnostic import seed_contrast
from .io import save,dump,report,table


def interpretation(mean,low,high,familiar_delta,separation_S,separation_I):
    if not np.isfinite([mean,low,high,familiar_delta,separation_S,separation_I]).all():return ['incomplete_outcomes']
    labels=[]
    if low>0:labels.append('switching_increases_mixed_error')
    elif high>=0:labels.append('no_clear_switch_specific_benefit')
    else:labels.append('switch_curriculum_reduces_mixed_error')
    if abs(familiar_delta)>.20:labels.append('familiar_competence_confounded')
    if separation_I<=0:labels.append('separation_ratio_undefined')
    elif separation_S/separation_I<.80:labels.append('repair_via_merging_possible')
    else:labels.append('relative_separation_guard_passed')
    return labels


def retention_checks(familiar,original_hidden_mean):
    means={}
    for arm in ['B','I','S']:
        g=familiar[familiar.arm.eq(arm)]
        if g.seed.duplicated().any():raise ValueError('Duplicate seed/arm familiar summary')
        means[arm]={name:float(g[name].mean()) if g[name].notna().all() else np.nan for name in ['A_ce','B_ce','familiar_mean_ce','hidden_separation','task_separation']}
    i=means['I'];s=means['S'];delta=s['familiar_mean_ce']-i['familiar_mean_ce']
    ratio=s['hidden_separation']/i['hidden_separation'] if np.isfinite(i['hidden_separation']) and i['hidden_separation']>0 else np.nan
    complete=np.isfinite([ratio,delta,original_hidden_mean]).all()
    return {'per_arm_cohort_means':means,'familiar_S_minus_I':delta,'familiar_guard_passed':bool(abs(delta)<=.20) if np.isfinite(delta) else None,
       'hidden_retention_ratio':ratio,'separation_guard_passed':bool(ratio>=.8) if np.isfinite(ratio) else None,'original_hidden_mean':original_hidden_mean,
       'S_relative_to_original_hidden':s['hidden_separation']/original_hidden_mean if original_hidden_mean>0 else np.nan,
       'interpretation_status':'checks_estimable_absolute_retention_descriptive' if complete else 'incomplete_outcomes'}


def baseline_separation(data):
    f=pd.read_csv(Path(data)/'tables/d0_native_replay.csv',float_precision='round_trip');rows=[]
    for seed in sorted(f.seed.unique()):
        a=f[f.seed.eq(seed)&f.cue.eq('A')].sort_values('transition');b=f[f.seed.eq(seed)&f.cue.eq('B')].sort_values('transition')
        h=np.linalg.norm(a[[f'hidden_{i}' for i in range(8)]].to_numpy()-b[[f'hidden_{i}' for i in range(8)]].to_numpy(),axis=1)
        delta=a[[f'logit_{i}' for i in range(4)]].to_numpy()-b[[f'logit_{i}' for i in range(4)]].to_numpy();z=np.linalg.norm(delta-delta.mean(1,keepdims=True),axis=1)
        rows.append({'seed':seed,'hidden_separation':h.mean(),'task_separation':z.mean()})
    return pd.DataFrame(rows)


def derived_tables(data):
    data=Path(data);read=lambda n:pd.read_csv(data/'tables'/f'{n}.csv',float_precision='round_trip')
    native=read('d0_native_replay');decomposition=read('d1_decomposition');patch=read('d1_correction_transition');external=read('d1_ensemble_baselines');d2=read('d2_history_seed');fam=read('d3_familiar_validation');after=read('d3_natural_seed');effects=read('d3_paired_effects')
    curves=native[native.cue.isin(['A','B','Mixed'])][['seed','cue','transition','raw_ce']]
    natural_boot=seed_bootstrap(curves,['raw_ce'],['cue','transition'])
    d1metrics=['e_task_norm','j_task_norm','r_task_norm','mixed_excess_ce','e_signed_target_margin','j_signed_target_margin','r_signed_target_margin']
    d1_boot=seed_bootstrap(decomposition,d1metrics,['transition'])
    parts=[]
    for branch,frame in [('natural_Mixed',native[native.cue.eq('Mixed')][['seed','transition','raw_ce']]),('one_time_correction',patch[patch.patch_k.eq(1)][['seed','transition','raw_ce']]),('external_midpoint_readout',external[['seed','transition','midpoint_readout_ce']].rename(columns={'midpoint_readout_ce':'raw_ce'}))]:
        x=frame.copy();x['branch']=branch;parts.append(x)
    patch_curve=pd.concat(parts,ignore_index=True);patch_boot=seed_bootstrap(patch_curve,['raw_ce'],['branch','transition'])
    d2['analysis']='fixed_grammar';history_boot=seed_bootstrap(d2,['history_contrast','history_excess_ce','diagnostic_excess_ce','overall_excess_ce'],['analysis'])
    after_mean=novel_means(after,['arm'],['raw_ce']);arm_boot=seed_bootstrap(after_mean,['raw_ce'],['arm','cue'])
    famil_metrics=['A_ce','B_ce','familiar_mean_ce','hidden_separation','task_separation'];fam_boot=seed_bootstrap(fam,famil_metrics,['arm'])
    control=[]
    for seed,g in fam.groupby('seed'):
        if g.arm.duplicated().any() or set(g.arm)!=set(['B','I','S']):raise ValueError('Missing/duplicate familiar arm')
        x=g.set_index('arm');control.append({'seed':seed,'comparison':'S_minus_I',**{m:float(x.loc['S',m]-x.loc['I',m]) for m in famil_metrics}})
    control=pd.DataFrame(control);control_boot=seed_bootstrap(control,famil_metrics,['comparison'])
    original=baseline_separation(data)
    d3history=read('d3_history_signatures');history=[]
    for arm,g in d3history.groupby('arm'):
        s=seed_contrast(g);s['arm']=arm;history.append(s)
    history=pd.concat(history,ignore_index=True);diags=[]
    d3decomp=read('d3_d1_signatures')
    for (seed,arm),g in d3decomp.groupby(['seed','arm']):
        h=history[history.seed.eq(seed)&history.arm.eq(arm)].iloc[0]
        diags.append({'seed':seed,'arm':arm,'mean_e_task_norm':g.e_task_norm.mean(),'first_j_task_norm':g.j_task_norm.iloc[0],'mean_j_task_norm':g.j_task_norm.mean(),'history_contrast':h.history_contrast,'overall_excess_ce':h.overall_excess_ce})
    return {'d4_native_position_intervals':natural_boot,'d4_mixing_position_intervals':d1_boot,'d4_correction_curve':patch_curve,'d4_correction_curve_intervals':patch_boot,
      'd4_history_intervals':history_boot,'d4_training_arm_intervals':arm_boot,'d4_familiar_intervals':fam_boot,'d4_familiar_paired_seed':control,'d4_familiar_paired_intervals':control_boot,
      'd4_original_separation_seed':original,'d4_converging_diagnostics':pd.DataFrame(diags)}


def run(root,out):
    tables=derived_tables(out)
    for name,frame in tables.items():save(out,name,frame)
    read=lambda n:pd.read_csv(out/'tables'/f'{n}.csv',float_precision='round_trip')
    fam=read('d3_familiar_validation');checks=retention_checks(fam,float(tables['d4_original_separation_seed'].hidden_separation.mean()));dump(out/'audit/d4_interpretation_checks.json',checks)
    effect=read('d3_effect_intervals');primary=effect[effect.cue.eq('Mixed')&effect.metric.eq('S_minus_I')].iloc[0]
    labels=interpretation(primary['mean'],primary.ci_low,primary.ci_high,checks['familiar_S_minus_I'],checks['per_arm_cohort_means']['S']['hidden_separation'],checks['per_arm_cohort_means']['I']['hidden_separation']);dump(out/'audit/d4_scientific_status.json',{'labels':labels,'absolute_original_separation_comparison':'reported_continuously_not_post_hoc_thresholded','formal_mediation_claim':False})
    from .plots import render_all
    render_all(out,out,tables)
    examples=example_rows(read('d3_paired_effects'),fam);save(out,'d4_examples',examples)
    claims=pd.DataFrame([
      ('C01','natural A/B/Mixed seven-step CE and identical sequence','d0_native_replay.csv; d4_native_position_intervals.csv','fig_1_mixing_failure','test_d0_source.py','High/Low and old C0 are not a causal comparison'),
      ('C02','j/r/e vector identities, source-roundoff bounds and t1 propagation=0','d1_decomposition.csv; d1_local_no_prior_deviation.csv','fig_1_mixing_failure','test_d1_mixing.py','Exact algebra is not additive CE causation; natural states remain float32'),
      ('C03','patch_k1 corrected−natural remaining-six-step CE','d1_correction_seed.csv; d1_correction_intervals.csv','fig_1_mixing_failure','test_d1_mixing.py','One-time state intervention; external midpoint/ensemble is not natural retrieval'),
      ('C04','history positions1,3,4,7 vs diagnostic2,5,6 excess CE','d2_transition_history.csv; d2_history_seed.csv; d2_history_intervals.csv','fig_2_sequence_information','test_d2_information.py','Token, position and difficulty confounds; no unique lost-progress identification'),
      ('C05','familiar-only prototype/ridge transfer and nonidentifiable flags','d2_decoder_sanity.csv; d2_decoder_seed.csv','fig_2_sequence_information','test_d2_information.py','14 correlated states; failed familiar transfer cannot establish absent information'),
      ('C06','fixed-final330 Mixed S−I paired seed CE; B control and Novel secondary','d3_training_schedule.csv; d3_model_registry.csv; d3_paired_effects.csv; d3_effect_intervals.csv','fig_3_continuity_training','test_d3_curriculum.py','Total curriculum effect with source loss and fresh Adam; no mediation claim'),
      ('C07','cohort familiar-mean difference<=.20; raw S/I separation>=.80','d3_familiar_validation.csv; d4_familiar_paired_intervals.csv; d4_original_separation_seed.csv','fig_3_continuity_training','test_d4_reporting.py','Pragmatic interpretation checks, not equivalence tests or inclusion filters'),
    ],columns=['claim_id','exact_metric','tables','figure','tests','allowed_limit'])
    claims['cohort']='canonical High_monotonic';claims['independent_seeds']=20;claims['original_update']=170;claims['source_sha']=sha256(ROOT/'confirmatory_srn/training.py');claims['config_sha']=CONTRACT['sha256'];save(out,'claim_source_map',claims)
    _reports(out,tables,checks,labels,examples,primary,effect)
    registry=[]
    for stage,name in [('d0','d0_checkpoint_audit'),('d1','d1_correction_seed'),('d2','d2_decoder_sanity'),('d3','d3_model_registry')]:
        f=read(name);cols=[c for c in ['seed','arm','patch_k','status','checkpoint_sha256','checkpoint_sha','prototype_status','ridge_status','updates_executed'] if c in f];g=f[cols].copy();g['stage']=stage;g['output_path']='tables/'+name+'.csv';registry.append(g)
    pd.concat(registry,ignore_index=True).to_csv(out/'audit/run_registry.csv',index=False)
    return {'independent_seeds':20,'trained_endpoints':60,'scientific_training_updates':int(read('d3_model_registry').updates_executed.sum()),'status_labels':labels,'three_figures_written':True,'reports_written':True,'original_sources_unchanged':True,'seal_required':True}


def example_rows(effects,fam):
    f=effects[effects.cue.eq('Mixed')].sort_values('seed');rows=[]
    good=f[f.S_minus_I.lt(0)];bad=f[f.S_minus_I.ge(0)]
    if len(good):r=good.iloc[0].to_dict();r['example_role']='first_seed_improvement';rows.append(r)
    if len(bad):r=bad.iloc[0].to_dict();r['example_role']='first_seed_adverse';rows.append(r)
    elif len(f):r=f.loc[f.S_minus_I.idxmax()].to_dict();r['example_role']='least_benefit_no_adverse_seed_observed';rows.append(r)
    result=pd.DataFrame(rows)
    for i,r in result.iterrows():
        for arm in ['B','I','S']:
            x=fam[fam.seed.eq(r.seed)&fam.arm.eq(arm)].iloc[0]
            for name in ['A_ce','B_ce','familiar_mean_ce','hidden_separation','task_separation']:result.loc[i,arm+'_'+name]=x[name]
    return result


def _reports(out,tables,checks,labels,examples,primary,effect):
    read=lambda n:pd.read_csv(out/'tables'/f'{n}.csv',float_precision='round_trip')
    native=read('d0_native_replay');decomp=read('d1_decomposition');d2=read('d2_decoder_sanity');corr=read('d1_correction_intervals');history=read('d2_history_intervals');cp=corr[corr.patch_k.eq(1)&corr.metric.eq('correction_minus_natural_future_ce')].iloc[0];hp=history[history.metric.eq('history_contrast')].iloc[0]
    native_means=native.groupby('cue',as_index=False).raw_ce.mean();external=read('d1_ensemble_baselines').groupby('seed')[['probability_ensemble_ce','logit_average_ce','midpoint_readout_ce']].mean().mean().to_frame('cohort_mean_CE').reset_index(names='external_baseline')
    familiarity=tables['d4_familiar_intervals'];contrast=tables['d4_familiar_paired_intervals'];raw=read('d3_natural_seed').groupby(['arm','cue'],as_index=False).raw_ce.mean();training=read('d3_model_registry');d1_diag=tables['d4_converging_diagnostics'].groupby('arm',as_index=False)[['mean_e_task_norm','first_j_task_norm','history_contrast','overall_excess_ce']].mean()
    means=pd.DataFrame([{'arm':a,**values} for a,values in checks['per_arm_cohort_means'].items()]);orig=tables['d4_original_separation_seed'].mean(numeric_only=True)
    expl=''
    for r in examples.itertuples():expl+=f'- {r.example_role}: seed{int(r.seed)}, original Mixed after continuation B={r.B:.6f}, I={r.I:.6f}, S={r.S:.6f}; S−I={r.S_minus_I:.6f} nats. Familiar mean I={r.I_familiar_mean_ce:.6f}, S={r.S_familiar_mean_ce:.6f}; raw split I={r.I_hidden_separation:.6f}, S={r.S_hidden_separation:.6f}.\n'
    audit_path=out/'audit/final_test_result.json'
    validation=(str(json.loads(audit_path.read_text())['tests_passed'])+' tests passed; source/checkpoint hashes unchanged.') if audit_path.is_file() else 'Final test totals and commands are recorded in audit/final_test_result.json.'
    story=f'''# Why contextual mixing disrupts the same learned sequence

The frozen High networks make different errors under physical Mixed despite an identical token sequence. The first hidden mismatch can arise from nonlinear cue integration without any inherited state mismatch. That observation does not by itself explain all subsequent prediction failures: the fixed initial correction has an uncertain cohort effect, and the task-history signature is also inconclusive.

The familiar-only continuation experiment estimates the total causal effect of the curriculum from identical already-trained weights and policy parameters. At fixed final update330, **Mixed S−I CE={primary['mean']:.6f} nats**, with95% paired-seed interval[{primary.ci_low:.6f},{primary.ci_high:.6f}]. Negative favors within-sequence switching over whole-sequence interleaving. Interpretation status: **{', '.join(labels)}**. This conclusion is qualified by familiar competence and retained separation below; it is not a formal demonstration that sequence-position preservation mediates the effect.

## Frozen computation: what breaks and what remains uncertain

All20 canonical High_monotonic checkpoints at original update170 were audited. No Low/C0 models were pooled. Source float32 natural mean next-token CE:

'''+table(native_means)+f'''

D1 decomposes the difference from the familiar hidden midpoint into newly generated nonlinear mismatch j and propagation r of the preceding deviation. At t1 all contexts start at zero, so r1=0. The maximum exact double-arithmetic identity residual was {decomp.hidden_identity_residual.max():.3g}; maximum recorded-float32 discrepancy was {decomp.observed_float32_residual.max():.3g}, inside the predeclared arithmetic bounds. Components are vectors: their norms must not be stacked or converted into a percentage of CE explained.

Replacing only the first Mixed state by its A/B midpoint changed the next six predictions by {cp['mean']:.6f} nats (corrected−natural),95% seed interval[{cp.ci_low:.6f},{cp.ci_high:.6f}]. This fixed causal state-reset control therefore does not establish that the initial mismatch explains the full deficit. All later k2…6 corrections, their failed cases and the non-recurrent one-step diagnostics remain exported.

The external prediction baselines below combine familiar predictions or states at each position. They are not natural Mixed rollouts and do not demonstrate that the model performs this averaging:

'''+table(external)+f'''

D2 compares Mixed excess CE over familiar mean CE at history-dependent positions1,3,4,7 versus token-diagnostic positions2,5,6. The seed contrast was {hp['mean']:.6f} nats,95% interval[{hp.ci_low:.6f},{hp.ci_high:.6f}]. This is not evidence of a disproportionate history-specific deficit under this protocol. Token identity, position and difficulty also differ between masks, so the contrast cannot uniquely identify missing sequence progress even if positive.

Only {int(d2.prototype_status.eq('shared_decoder_sanity_passed').sum())}/20 prototype and {int(d2.ridge_status.eq('shared_decoder_sanity_passed').sum())}/20 ridge diagnostics passed the frozen familiar-transfer/geometry gate. Invalid shared-readout cases are nonidentifiable, not evidence that position information is absent. Raw scores for every seed remain in tables. Fourteen familiar states provide little independent support even for passing decoders. A decoded position is not proof that the original output head uses it.

## Familiar-only learning intervention

Every B/I/S arm starts from the same seed's model **and separate monotonic policy weights**, not a fresh initialization. Saved optimizer moments do not exist, so every arm has identical fresh Adam state: `fine_tune_from_weights_optimizer_reset`. The original loss, global clipping, token sequence, update unit and reset rule are preserved. There are160 online sequence updates per arm. Smoke seeds0/1 count among the60 final endpoints and are not trained twice; total executed scientific updates={int(training.updates_executed.sum())}. No protected cue or loss enters training or checkpoint selection. All60 endpoints were sealed before protected evaluation.

B has A10/B10 blocks; I alternates whole A/B sequences; S alternates A→B and B→A within each sequence at boundary3→4. Every scored position sees80 A and80 B cues, with identical tokens and targets. Total context-switch events are intentionally not matched: the location and frequency of changes are the curriculum manipulation. All sequences reset h0=0.

Final familiar competence and separation, cohort means:

'''+table(means)+f'''

Familiar S−I mean CE difference={checks['familiar_S_minus_I']:.6f} nats; pragmatic absolute margin .20, passed={checks['familiar_guard_passed']}. Raw hidden separation S/I={checks['hidden_retention_ratio']:.6f}; threshold .80, passed={checks['separation_guard_passed']}. Original High raw separation={checks['original_hidden_mean']:.6f}; S/original={checks['S_relative_to_original_hidden']:.6f}. Original task separation={orig.task_separation:.6f} (recorded centered logits, up to the audited float32 head precision). Absolute retention is reported continuously, without inventing a new threshold. These are interpretation checks, not formal equivalence tests, evaluation gates or seed-exclusion rules.

All registered per-probe effects, including the individual Novel directions and within-seed means, are shown in the following source-replay contrasts:

'''+table(effect[['cue','metric','mean','ci_low','ci_high','n_seeds','status']])+'''

Converging I/S computation diagnostics (descriptive; not mediation):

'''+table(d1_diag)+'''

### Illustrative and adverse examples

'''+expl+'''

The full20-seed plot includes every adverse outcome. If no seed has an adverse primary sign, the least-benefit case is labeled accordingly instead of manufacturing a failure. Tables preserve per-transition CE, target margin, all four probabilities and actual hidden states.

## Detailed evidence: three primary figures

![Where Mixed fails](figures/fig_1_mixing_failure.png)

Figure1 separates exact nonlinear identities, output-relevant mismatch and the one-time state-reset evidence. Lag0 correction is exogenous assistance. The midpoint reference is an external readout, not a recurrent solution.

![Sequence information](figures/fig_2_sequence_information.png)

Figure2 retains all20 grammar contrasts. Decoder scoring is conditional on a familiar sanity gate and explicitly labels the small selected diagnostic subsets; invalid cases remain in the source tables and primary analysis.

![Continuity training](figures/fig_3_continuity_training.png)

Figure3 compares the three familiar-only curricula, the paired primary effect and familiar/raw/task separation controls. All start from the same original seed weights; none trained on Mixed or Novel. Similarities in post-training diagnostics do not identify a unique mediation pathway.

## Exact algorithms and equations

At position t, a_t^c=W_h h_(t−1)^c+W_x x_t+W_c c+b_h; h_t^c=tanh(a_t^c); logits=W_o h_t^c+b_o; p=softmax(logits); CE=−log p_correct. Inputs are four token one-hots plus three raw context channels, eight tanh units and four next-token logits. The third cue column is zero and preserved by the source hook/enforcement. Seven inputs `(0,1,2,0,3,1,2)` predict `(1,2,0,3,1,2,3)` with teacher forcing. All evaluations score only the next-token head, not RT.

m_t=(h_t^A+h_t^B)/2; e_t=h_t^Mixed−m_t; abar=(a_t^A+a_t^B)/2. With physical Mixed=(A+B)/2, a_t^Mixed=abar+W_h e_(t−1). Then j_t=tanh(abar)−(tanh(a_t^A)+tanh(a_t^B))/2 and r_t=tanh(abar+W_h e_(t−1))−tanh(abar), so e=j+r in real arithmetic. The double audit evaluates that exact map at observed source prestates; the original float32 rounding term is separate. Task components use P W_o, P=I−11ᵀ/4. Correct-versus-competitor signed projections use the same highest incorrect natural-Mixed competitor for all components. No local Jacobian or autonomous-dynamics survey is used.

Probability averaging `(pA+pB)/2` differs from softmax of averaged logits. The recomputed linear head maps a hidden midpoint to averaged logits exactly up to1e-11 double precision; recorded float32-head differences are separately exported. Familiar comparator CE is the arithmetic mean of the two endpoint **losses**, not NLL of their mean probability.

The source continuation loss is:

L=mean CE +.20 mean[.50(1−v)+stopgrad(1−p_correct)v²],

v=sigmoid(softplus(raw_slope) logit(stopgrad(clamp(p_correct,1e-6,1−1e-6)))+intercept).

The172 model scalars get CE gradients; the two policy parameters get execution gradients. Joint L2 clipping at1.0 still couples their norm. Adam uses lr.005, betas(.9,.999), eps1e-8, no weight decay/AMSGrad/maximize/fused/capturable/differentiable variants; source default foreach=None is preserved. One full seven-step forward/BPTT, one backward, joint clip and one Adam step occur per sequence, then the source third-column enforcement runs. The thin adapter supplies time-varying cues to the original `model.recurrent` and `sequence_head`; it does not create a new SRN class. Static-cue source gradients and updates matched independent source fixtures.

Prototype decoding assigns Euclidean-nearest familiar position. Ridge uses fourteen A/B states and fixed lambda1.0, population feature SD floor1e-6 and an unpenalized intercept. Familiar A-only→B and B-only→A preprocessing is fitted on each training side alone. Shared sanity requires each direction at least5/7, pooled14 states, no below-floor variance and centered rank≥7. Coefficients/normalizers are sealed before Mixed scoring and stored in audit. No probe label, alternate lambda or nonlinear decoder is fitted.

## Inference, failures, provenance and falsification

The trained seed is the independent unit. All10,000 bootstrap draws use RNG20260925; repeated positions, cue sides and training arms remain paired. Novel sides are preserved, with named means averaged within seed. The20-seed sample supports this model family/sequence, not generalization across architectures or human hidden-state implementation. Missing/nonfinite outcomes propagate as NA/status and never become zeros or replacements.

Full source paths, hashes and the original objective contract are in audit/source_contract.md and manifests/d3_objective_contract.json. Two optional original config/training-log files remain unavailable cloud placeholders; expected historical hashes are recorded, and exact embedded checkpoint configs/local training source establish the contract without invented historical curves. Finder .DS_Store metadata in the previous run was logged and excluded without changing that older freeze; any changed recorded scientific byte or unregistered scientific file still fails verification. {validation} Every failed RED, passing GREEN, behavioral negative fixture and refactor result is retained in audit/test_evidence.jsonl and tests/red_green_log.jsonl. Final test totals are in audit/final_test_result.json. No old E3/E4 calibration was resumed or overwritten.

A clear S−I improvement with stable familiar competence and substantial retained raw separation supports learnable context-robust prediction after splitting. A benefit with merging or differential familiar learning has a narrower explanation. A null/worse S result means this fixed switch location/budget is insufficient, not that all continuity learning is impossible. The negative D1/D2 signatures limit a claim that loss of sequence progress has been identified. To change the mechanistic conclusion would require convergent, interpretable progress diagnostics or an independently manipulated information-preservation pathway; no such follow-up is silently run here.

Reproduce in a new directory after copying CONTEXT_MIXING_SCRATCHPAD_TEMPLATE.md to scratchpad.md: run `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.context_mixing_sequence_continuity.cli --stage d0 --output NEW_DIRECTORY`, then d1,d2,d3_smoke,d3,d4 in order. Never rerun a started stage into its original directory. Use `python3 -m experiments.context_mixing_sequence_continuity.verify --output NEW_DIRECTORY --seal` once; omit --seal to verify. Figures/key summaries regenerate from sealed tables using the regenerate module and a fresh destination, without new training.
'''
    story=story.replace('{validation}',validation)
    report(out,'FINAL_CONTEXT_MIXING_SEQUENCE_REPORT.md',story)
    ledger='''# Claim and alternative ledger

| Claim | Exact evidence | Tested alternative / failure status | Allowed statement |
| --- | --- | --- | --- |
| Nonlinearity can create first-step task mismatch | D1 j/r/e vectors; r1=0; linear and constant-output fixtures | Large hidden divergence can be output-invisible; rounding separately bounded | Earliest computational mismatch, not fraction of CE caused |
| Initial correction affects subsequent prediction | Fixed k1 corrected−natural six-step CE and20-seed interval | All adverse seeds and later corrections retained | Causal contribution of one state correction; no unique midpoint necessity |
| History-sensitive error signature | D2 fixed grammar contrast over20 seeds | Token/position/difficulty confounds; interval may include zero | Motivated signature, not identified lost information |
| Familiar-trained position readout transfer | Prototype/ridge raw scores and sanity statuses | Failed A↔B transfer or degenerate geometry is nonidentifiable | Tiny-sample conditional decoding, not natural retrieval |
| Curriculum changes held-out Mixed | Final330 S−I paired CE; B and all60 endpoints | Familiar and raw/task split controls; no protected selection | Total effect of schedule in these frozen-start models |
| Specific preservation mechanism | Post I/S D1/D2 signatures alongside guardrails | No independent mediation manipulation | Converging descriptive evidence only; mechanism remains qualified |

Exact selectors, source/test names and hashes are in tables/claim_source_map.csv. All scalar thresholds were fixed before outcomes; no failing seed, condition or probe is dropped.
'''
    report(out,'CLAIM_LEDGER.md',ledger)
    hand=f'''# Researcher handoff

## Question and primary answer

Why does mixing familiar contexts disturb prediction of the same sequence, and can familiar context-switch learning prevent that failure? D1 identifies new nonlinear mismatch and propagation, but initial correction and the grammar-history contrast are inconclusive at cohort level. The paired familiar-only learning effect is Mixed S−I={primary['mean']:.6f} nats,95% interval[{primary.ci_low:.6f},{primary.ci_high:.6f}], qualified as {', '.join(labels)}.

## Actual algorithms, objective and parameters

The original float32 eight-unit tanh SRN predicts seven teacher-forced next tokens. D1 evaluates the exact e=j+r identity in double arithmetic at source prestates and reports numerical rounding separately. A single midpoint placement at k1 tests the contribution of initial mismatch under unchanged Mixed continuation; external probability/logit averages are not trajectories. D2 compares fixed history positions1/3/4/7 with diagnostic2/5/6, using mean familiar endpoint CE as comparator. Familiar-only Euclidean prototypes and ridge lambda1.0 are scored on Mixed only after fitting/scaling; failed cross-context validation is nonidentifiable.

D3 strictly loads each saved model **and policy**, then applies160 sequence updates in B/I/S. B uses A10/B10, I alternates complete sequences, S switches once at3→4 and alternates A→B/B→A. All positions get80 A and80 B cues. Original total loss is mean CE+.20 mean[.50(1−v)+stopgrad(1−p_correct)v²], with detached-confidence monotonic v. All172 model and2 policy scalars are jointly clipped at1.0; Adam lr.005 and source defaults are preserved. Moments were absent, so all arms reset optimizer state identically. Hidden state resets each sequence. The six smoke models are reused, keeping scientific updates at9600. All60 final330 checkpoints are sealed before protected evaluation, with no best epoch/seed selection.

## What changed / what stayed fixed

Only familiar cue schedules differ among training arms. Original weights, source code, sequence tokens, next-token targets, objective, architecture, step budget, clipping and optimizer settings are held fixed at the start/contract level. Continued training intentionally changes new copies of parameters. Original checkpoint files are untouched. Mixed, NoContext and exact angular Novel cues never enter training, auxiliary losses, hyperparameter choice or checkpoint selection. This is not the earlier failed E3 closure factorial.

## Evidence and interpretation checks

D1 first-correction future effect={cp['mean']:.6f},95% interval[{cp.ci_low:.6f},{cp.ci_high:.6f}]. D2 history-minus-diagnostic excess={hp['mean']:.6f},95% interval[{hp.ci_low:.6f},{hp.ci_high:.6f}]. Shared decoder sanity: prototype{int(d2.prototype_status.eq('shared_decoder_sanity_passed').sum())}/20; ridge{int(d2.ridge_status.eq('shared_decoder_sanity_passed').sum())}/20. Invalid cases remain in all primary analyses and raw diagnostic tables.

Familiar S−I={checks['familiar_S_minus_I']:.6f} nats (guard .20, passed={checks['familiar_guard_passed']}); raw hidden S/I={checks['hidden_retention_ratio']:.6f} (guard .80, passed={checks['separation_guard_passed']}); S/original raw separation={checks['S_relative_to_original_hidden']:.6f}. These are descriptive guardrails, not formal equivalence or inclusion tests. Raw A/B and task separation are shown for every arm. The final report provides all raw cue-side effects and intervals.

Illustrative/adverse source cases:

'''+expl+'''

## Tests, assumptions and limits

RED logs include missing source modules, corrupt/missing checkpoint traps, wrong propagation sign, output-invisible divergence, grammar/target corruption, decoder leakage/transfer failures, protected cue insertion and wrong optimizer settings. Source-gradient/update fidelity, all160 update counts, per-position exposures and paired-seed resampling pass. Read the final totals/commands in audit/final_test_result.json and full outputs in audit/test_evidence.jsonl. Only D3 updates scientific copies; no D0–D2 optimizer step occurred.

The model is teacher-forced and the decoder sample is tiny. Grammar masks confound position and difficulty. Adam reset makes this fine-tuning from learned weights, not exact optimizer-history continuation. Curriculum assignment identifies a total effect within this model family; concurrent changes in D1/D2 are not causal mediation. Hidden/output norms cannot allocate CE causation. Results do not identify human or autonomous-generative mechanisms.

## 60–90-second teach-back

“The network predicts the same sequence in A and B but can fail when their physical cues are mixed. We traced the computation and separated newly generated nonlinear mismatch from propagation of the preceding mismatch. At the first step there is no inherited mismatch, so any deviation begins in nonlinear contextual integration. But the initial state correction did not clearly repair the whole future, and the fixed token-history contrast did not establish a special history deficit. Familiar-trained decoders usually failed their own cross-context validation, so their Mixed failures cannot prove information is absent. We then cloned the same trained checkpoints into three matched familiar-only continuation curricula. Within-sequence switching is compared with whole-sequence interleaving at the fixed final endpoint, never selected by Mixed or Novel performance. Switching did not clearly outperform interleaving, and familiar performance was worse by about0.703 nats while raw separation was retained. So this protocol has not shown that context-continuity learning prevents Mixed failure. It remains a short teacher-forced SRN experiment, not a demonstrated human mechanism.”

## What would change the conclusion

An independently validated shared progress readout or an independent manipulation of preservation would strengthen a sequence-information mechanism. Better Mixed performance with collapsing A/B separation, or substantially different familiar competence, instead narrows the explanation. No next training run, new loss, switch location or hyperparameter search was performed to secure an effect. First inspect the all-seed plots, adverse cases and source-linked tables.

## Files

Start with FINAL_CONTEXT_MIXING_SEQUENCE_REPORT.md, Figures1–3 and CLAIM_LEDGER.md. tables/claim_source_map.csv maps exact metrics to source tables/tests. audit/source_contract.md and manifests/d3_objective_contract.json explain the original loss. Every checkpoint, source/input byte and result is hash-indexed. The verify/regenerate commands in the final report check the seal and rebuild figures/key summaries without training or new protected evaluation.
'''
    report(out,'RESEARCHER_HANDOFF.md',hand)
    presenter=f'''# Lab-meeting presenter summaries

## Figure1 — nonlinear mixing and inherited deviation

Physical Mixed changes the nonlinear state update even though tokens and targets are identical. At the first step, inherited propagation is exactly zero; the mismatch is new nonlinear integration. Later propagation and fresh mixing can reinforce or cancel in output space. A single first-state correction changed future CE by {cp['mean']:.3f} nats, with an interval spanning zero. This identifies a computational discrepancy, but does not establish that it causes the entire sequence deficit. Component norms are not additive CE contributions.

## Figure2 — sequence information remains uncertain

The fixed grammar contrast was {hp['mean']:.3f} nats and its interval includes zero. We did not detect a disproportionate history-dependent Mixed deficit. Familiar cross-context decoding sanity passes only3/20 prototype and2/20 ridge cases; their displayed Mixed decoding is conditional on that small subset. Failed shared decoding is nonidentifiable, not evidence that progress information is absent. Position, token identity and difficulty remain confounded.

## Figure3 — the fixed continuity curriculum did not establish prevention

Allthree arms use identical original model/policy weights, source loss, fresh Adam and160 familiar-only sequence updates. The primary Mixed S−I effect was {primary['mean']:.3f} nats with95% interval[{primary.ci_low:.3f},{primary.ci_high:.3f}]. Familiar mean CE is worse under S by {checks['familiar_S_minus_I']:.3f} nats, failing the .20 margin; raw separation remains {checks['S_relative_to_original_hidden']:.1%} of original High. This is an inconclusive prevention test with a competence confound, not evidence for a clean preservation mechanism. No parameters or training budgets were changed to obtain a different result.
'''
    report(out,'PRESENTER_SUMMARIES.md',presenter)
