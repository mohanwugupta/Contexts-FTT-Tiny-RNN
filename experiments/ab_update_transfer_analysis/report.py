"""Plain-language scientific handoff from the fixed saved tables."""
from .common import *
from .statistics import interval
from . import design

def read(n):
 for suffix in ['.csv','.csv.gz']:
  p=OUT/'tables'/(n+suffix)
  if p.exists():return pd.read_csv(p,float_precision='round_trip')
 raise FileNotFoundError(n)
def tab(f):return f.to_markdown(index=False,floatfmt='.4g')
def val(x):
 d=interval(np.asarray(x,float));return f"{d['mean']:.4g} [{d['ci_low']:.4g}, {d['ci_high']:.4g}]"

def main():
 assert not (OUT/'manifest.json').exists(),'Sealed reports require an explicit new version'
 assert json.loads((OUT/'audit/analysis_complete.json').read_text())['passed'];assert (OUT/'figures/FIGURE_INDEX.json').exists();e=read('update_effects');curve=read('native_curve_metrics');primary=e[e.cohort.eq('final')&e.branch_type.eq('native')&e.branch_step.eq(10)&((e.target_update.eq(0)&e.source_context.eq('A'))|(e.target_update.eq(10)&e.source_context.eq('B')))];rows=[]
 for keys,g in primary.groupby(['scale','regime','objective','source_context']):
  c=g[g.probe_context.eq('C')].sort_values('seed');n=g[g.probe_context.eq('NoContext')].sort_values('seed');assert len(c)==len(n)==6
  rows.append(dict(zip(['scale','regime','objective','source_context'],keys),own_gain=val(c.own_gain),cross_gain=val(c.cross_gain),D=val(c.specificity),C_gain=val(c.gain),NoContext_gain=val(n.gain),C_start=c.pre_ce.mean(),NoContext_start=n.pre_ce.mean(),own_negative=int(c.own_gain.lt(0).sum()),cross_negative=int(c.cross_gain.lt(0).sum()),own_floor=int(c.own_floor.sum()),C_floor=int(c.probe_floor.sum()),NoContext_floor=int(n.probe_floor.sum())))
 early=pd.DataFrame(rows);csv(OUT/'tables/report_first_blocks.csv',early)
 decomposition=[]
 for (tid,c),g in curve[curve.probe_context.isin(['C','NoContext'])].groupby(['trajectory_id','probe_context']):
  g=g.sort_values('target_update');tag=g.iloc[0];v=g.set_index('target_update').ce;r={k:tag[k] for k in ['trajectory_id','seed','scale','regime','objective','meta_stage','meta_episodes','cohort']};r.update(probe_context=c,initial_error=v.loc[0],available_horizon=int(v.index.max()),CE60=v.get(60,np.nan),CE330=v.get(330,np.nan),CE990=v.get(990,np.nan),gain0_60=v.loc[0]-v.get(60,np.nan),gain60_330=v.get(60,np.nan)-v.get(330,np.nan),gain330_990=v.get(330,np.nan)-v.get(990,np.nan),gain60_990=v.get(60,np.nan)-v.get(990,np.nan));decomposition.append(r)
 decomp=pd.DataFrame(decomposition);csv(OUT/'tables/transfer_error_decomposition.csv',decomp);final=decomp[decomp.cohort.eq('final')];cohort=final.groupby(['scale','regime','objective','probe_context'])[['initial_error','gain0_60','gain60_330','gain330_990','CE990']].mean().reset_index();csv(OUT/'tables/report_error_decomposition.csv',cohort)
 cue=curve[curve.cohort.eq('final')&curve.target_update.isin([0,10,20,40,60,170,330,990])&curve.probe_context.isin(['C','NoContext'])].pivot(index=['trajectory_id','seed','scale','regime','objective','target_update'],columns='probe_context',values='ce').reset_index();cue['C_minus_NoContext']=cue.C-cue.NoContext;csv(OUT/'tables/report_C_cue_effect.csv',cue)
 adam=read('adam_controls');ag=adam.groupby(['scale','regime','objective','target_update','source_context','probe_context'])[['gain_native','gain_zero','native_minus_zero_gain','pre_ce_native']].mean().reset_index();csv(OUT/'tables/report_Adam_means.csv',ag);exposure=pd.read_csv(OUT/'registry/exposure.csv').iloc[0];reg=pd.read_csv(OUT/'registry/branches.csv');missing=read('missingness');earlyoriginal=read('original_early_splitting');ss=read('seed_summaries');paired=read('paired_intervals')
 statements=[]
 for probe in ['C','NoContext']:
  d=final[final.probe_context.eq(probe)];statements.append(f"For {probe}, early0–60 practice decreases loss in {int(d.gain0_60.gt(0).sum())}/144 model conditions. Later60–990 practice increases loss in {int(d.gain60_990.lt(0).sum())}/144. These are descriptive counts over repeated six-seed conditions, not144 independent networks or a new success criterion.")
 a0=primary[primary.probe_context.eq('C')];statements.insert(0,f"Across the registered first-A(0→10) and first-B(10→20) branches, own gain is positive in {int(a0.own_gain.gt(0).sum())}/{len(a0)} source-labelled model conditions, D is positive in {int(a0.specificity.gt(0).sum())}/{len(a0)}, and familiar cross gain is negative in {int(a0.cross_gain.lt(0).sum())}/{len(a0)}. Positive D must be read with both gains; it can partly reflect harm to the other familiar context.")
 for n in [0,60,330,990]:
  z=cue[cue.target_update.eq(n)];statements.append(f"At target{n}, C CE is lower than NoContext in {int(z.C_minus_NoContext.lt(0).sum())}/144 model conditions and higher in {int(z.C_minus_NoContext.gt(0).sum())}/144. C's column is fixed; shared updates can still change how that cue is expressed.")
 recommendation='The smallest supported follow-up is a frozen directional-derivative audit at the already registered target20 states, using all144 models and the cached native and zero-gradient-Adam displacements. Compute ∇CE_C·Δθ_shared and ∇CE_NoContext·Δθ_shared, compare their signs and magnitudes with the observed finite-step gains, and report the residual curvature. It requires no optimizer step, new model or meta-training. This would distinguish an unfavorable local update direction from nonlinear response to an otherwise favorable direction before choosing a new plasticity rule or meta objective. It is recommended only; it was not run under this PRD.'
 first_A=primary[primary.source_context.eq('A')&primary.probe_context.eq('C')];first_B=primary[primary.source_context.eq('B')&primary.probe_context.eq('C')];first_AN=primary[primary.source_context.eq('A')&primary.probe_context.eq('NoContext')];first_BN=primary[primary.source_context.eq('B')&primary.probe_context.eq('NoContext')]
 nmeans=cohort[cohort.probe_context.eq('NoContext')].CE990
 lead=f"A/B practice usually improves C, even while preferential familiar learning remains. The first A10 block improves C in {int(first_A.gain.gt(0).sum())}/144 model conditions; the first B10 block improves C in {int(first_B.gain.gt(0).sum())}/144. Own-minus-cross gain is positive in {int(first_A.specificity.gt(0).sum())}/144 and {int(first_B.specificity.gt(0).sum())}/144 respectively, with positive own gains in all first blocks. These counts describe repeated six-seed conditions, not independent replications or a new joint pass rule.\n\n"
 lead+=f"NoContext also benefits early: {int(first_AN.gain.gt(0).sum())}/144 first A blocks and {int(first_BN.gain.gt(0).sum())}/144 first B blocks improve it. Nevertheless, final NoContext condition-mean losses range from {nmeans.min():.3g} to {nmeans.max():.3g}nats. Thus weak endpoint transfer is not generally an absence of benefit from A/B learning. The supported pattern is a large initial error with incomplete recovery; later deterioration contributes in some models, while the learned C cue usually helps expression after acquisition. It does not identify how much each cause independently explains.\n\n"
 fresh=[]
 for n in [10,20]:
  for source in ['A','B']:
   z=adam[adam.target_update.eq(n)&adam.source_context.eq(source)&adam.probe_context.eq('C')];fresh.append(f"target{n}/{source}: {int(z.native_minus_zero_gain.gt(0).sum())}/144")
 lead+='Existing Adam history matters, but fresh gradients often add C benefit beyond the momentum-only control ('+'; '.join(fresh)+'). These are conditional native-versus-zero-gradient effects, not an additive division of learning. Small later familiar gains must be read with floor flags.\n\n'
 text='''# What A/B practice changes in A, B, C and NoContext

'''+lead+ '\n\n'.join(statements)+'''

The substantive explanation must be read across these components: early A/B updates can be selective while changing C and NoContext through shared entries; a fixed C column can help or obstruct expression depending on the shared recurrence/readout. Small late gains are often floor-limited. A native update also depends on Adam history; zero-gradient Adam is a moving control rather than an untouched baseline. The following condition-level results quantify each part, without interpreting a correlation or structural identity as mediation.

## 1. Does practice preferentially improve its own familiar context?

These are the original first A block from0 and first B block from10, each ten updates. These two windows have different starting histories; intrinsic A-versus-B comparisons require the matched common-checkpoint matrices, which show both alternative sources at every registered n. Entries are six-seed mean [pointwise95% interval]. Raw individual values, signed own/cross gains, one-step counterparts and all later checkpoints are in update_effects.csv. gain_matrices.csv supplies the explicit G_A_A…G_B_NoContext/Mixed_average columns and D_A/D_B, with gain_matrix_intervals.csv for paired-seed summaries. The source's original early metrics and floor flags are retained in original_early_splitting.csv and checked against the matched ten-step results under the inherited evaluation contract. The original170/330 criterion is unchanged; no990 gate is introduced.

'''+tab(early[['scale','regime','objective','source_context','own_gain','cross_gain','D','C_gain','NoContext_gain']])+'''

## 2–3. Do these updates help C and NoContext?

C_gain and NoContext_gain above use exactly the same practiced-source branches. Their starting errors and own probe floor flags remain beside them in report_first_blocks.csv and update_effects.csv. Figure2 gives both one-step and ten-step effects at all five acquisition checkpoints. Figure3 keeps specificity and transfer as separate coordinates, marking nonpositive own gains and familiar cross deterioration. NoContext is informative and is not an added behavioral pass requirement.

## 4–5. Development, starting error and later interference

Each row below averages only its six paired seeds. Positive gain means error decreased; negative gain means error increased. Starting error − gain0_60 − gain60_330 − gain330_990 = CE990. Full signed per-update values, position predictions, margins and A/B-specific realized contributions are available. The first60 updates have30 A/30 B trials; full990 has500 A/490 B. The intervals0–10,10–20,20–60,60–170,170–330,330–990 have their actual source counts in realized_gain_by_source.csv. No interpolated states fill a gap.

'''+tab(cohort)+'''

Floor flags use the inherited CE≤.05 definition. A near-zero familiar loss leaves little measurable own improvement. Such periods do not show that representations became shared or that separate-learning capacity disappeared. Compare a probe's own starting loss and margin: matching the practiced context's loss does not match C/NoContext headroom. matched_headroom.csv retains the first eligible checkpoint within each registered own-loss band (lower inclusive, upper exclusive except4 inclusive), including all missing bins and actual paired n. This is conditional description, not causal adjustment.

## C cue, parameter changes and optimizer history

The prediction parameter masks partition shared token/recurrent/readout/bias entries, familiar cueA, familiar cueB, and fixed cueC. Other-cue motion is retained when Adam momentum moves an unpracticed familiar column. Frozen endpoint replacements copy exact saved entries; they do not train a different frozen-pathway model. All six variants and the A/B interaction residual are in parameter_interventions.csv.

Cue-only replacements leave all C/NoContext logits unchanged; full/shared-only agree there. These are structural checks and do not establish a context-independent skill. The observed C−NoContext CE and margins show whether C helps expression. A larger centered cue response is not necessarily helpful.

Adam controls preserve current moments and parameter participation. Held performs no step. Zero-gradient Adam constructs the native gradients, replaces each non-None gradient with zero before clipping/stepping, and can move parameters through prior moments. Native gain − zero-gradient gain equals CE_zero − CE_native. It describes how the fresh gradient changes an update given history, including the changed moment denominator; it is not an additive history-free contribution. The source-labelled zero controls have identical model/policy/Adam endpoints for A/B because the native participation pattern is identical, while their recorded cue losses differ. They remain repeated measurements, not extra seeds. The control is restricted to10/20 and does not identify the cause of later60–990 deterioration. adam_control_intervals.csv gives paired intervals for native-minus-zero effects.

## 6. Regime, scale, objective and historical stage

All24 final factorial cells and all six seeds are present in the main overview/figures. P−U is the complete projected regime beginning in Reptile; it is not an isolated FOMAML projection effect. Targets have no norm projection. T/B/O1/O2 are complete procedures with different support/query exposure and cadence, including one outer update per two B episodes. Fresh18, Reptile36 and FOMAML1920 references144 are unique physical trajectories, not duplicated objective-labelled parents. Their five one-step checkpoint measurements, stage changes, loss matching, means/medians and sign counts are in the exact tables. Repeated scales/objectives/checkpoints and positions are not independent trained seeds.

All uncertainty uses10000 draws of the same six seed IDs with RNG20261122. Paired P−U and lineage contrasts reuse those draws, report mean/median intervals and actual n; missing values stay missing. Pointwise intervals are descriptive and not multiplicity-adjusted. A crossing-zero interval is inconclusive, not equivalence. No weighted joint winner or threshold revision is used.

## 7. One recommended next action

'''+recommendation+'''

## Native algorithm and equations

The native manual identity recurrence is h_t=R h_(t−1)+W_x x_t+W_c c+b_ih+b_hh, z_t=W_o h_t+b_o. Hidden state resets to0 for each seven-token teacher-forced sequence. Inputs [0,1,2,0,3,1,2], targets [1,2,0,3,1,2,3]. The advertised tanh RNN container is never executed. Native Adam0.005 and CE+.2 detached-confidence execution loss, policy gradients, joint norm1 clipping, parameter ordering/moments/counters are preserved. No C/NoContext practice, target projection or optimizer reset occurs.

G_s,e=L_e(before)−L_e(after); D_A=G_A,A−G_A,B and D_B=G_B,B−G_B,A. Baselines use identical model weights across probes. Ten-step branches are constant-source counterfactuals; their one-step prefixes are not recomputed. Frozen core.trace uses its original float64 score of native float32 logits; native training CE_before remains separately float32. Statistical aggregation is float64.

K_t=W_oΣ_(j=0)^(t−1)R^j, z_C−z_N=K_t w_C. With w_C fixed, response change=(K'_t−K_t)w_C. Class-centering removes only a common four-class logit offset; all raw logits/probabilities are retained. Mixed-average gives(z_A+z_B)/2 with the inherited positionwise CE Jensen bound, not an accuracy guarantee. It is an interpolation control, distinct from learned C. Geometry does not identify independent human memories.

## Coverage, numerical limitations and artifacts

Protocol-order qualification: source histories were sealed before branching, but this analysis consolidated StageA exports and ran telescoping checks after the diagnostic branches, contrary to the requested order. All342 subsequent checks passed with no outcome-dependent selection or retuning. registry/protocol_deviations.json records the deviation; no work was rerun to simulate compliance.

'''+f"Completed {int(exposure.completed_branches)}/{int(exposure.planned_branches)} branches; missing/failed {int(exposure.missing_or_failed_branches)}. Actual new diagnostic/control optimizer calls {int(exposure.new_diagnostic_control_steps)}, reconstructed {int(exposure.new_reconstruction_steps)}, synthetic-fixture {int(exposure.new_fixture_steps)}, meta0. Exactly {int(exposure.archived_branch_optimizer_steps_reused)} archived branch steps were reused, not new replications. Every attempted operation is charged before execution; complete per-step states resume, uncertain attempts quarantine.\n\n"+'''All original strict forward precision exceptions are retained (three distinct source cases), separately from inherited passing float32 propagation bounds and original gradient/Adam checks. No tolerance was relaxed. Scheduled preflight16 full-state replays match bitwise. Branch algebra checks cover allfour probabilities/logits across seven positions; source archives are immutable. RNG is in recovery sidecars, not each historical snapshot. The native path consumes no RNG; Torch sidecars agree along each archived trajectory. Python/NumPy were not recorded at each historical target snapshot and are unused by native target operations. Fresh spliced prefixes cannot identify the original worker RNG; this provenance qualification remains explicit. Matched branches nonetheless restore and isolate complete relevant state and current unused RNG. State/data hashes, exact cursor and scheduler semantics are audited.

[All six figure families and companions](figures/README.md), SOURCE_AUDIT.md, IMPLEMENTATION_AUDIT.md, REPRODUCIBILITY.md, CLAIM_LEDGER.csv and the append-only SCRATCHPAD.md locate all evidence. Required Parquet tables use the already installed pyarrow, with exact CSV companions for portable plotting. Native raw arrays are compressed NPZ; complete model/policy/Adam/RNG/cursor states and endpoint replacements remain cached. Read-only verification and figure regeneration never call an optimizer. Further experiments require a new protocol.
'''
 (OUT/'REPORT.md').write_text(text)
 handoff='''# Researcher handoff

An A/B practice update can help its own cue, help or hurt the other familiar cue, and independently change C/NoContext through shared sequence dynamics. We measured these effects from identical complete starting states, not from two differently trained evaluation models. C is learned during meta-training but fixed here. Removing C removes only that cue drive. Changing shared recurrence/readout can alter C's contribution despite a fixed C column.

'''+ '\n\n'.join(statements)+'\n\n'+recommendation+'''

Read the first-block table with own/cross gains and starting losses, then the initial-error/development decomposition, C−NoContext comparisons and native/zero-gradient history controls. A positive D partly caused by cross deterioration is qualified; a small gain at the inherited CE floor is uninformative about sharing. Shared-only agreement for C/NoContext is structural. No parameter norm or hidden distance substitutes for separate learning. P−U covers the complete regime; only six paired seeds supply uncertainty.

All scientific branches have stopped. Three inherited strict precision failures and the unused/missing historical RNG provenance qualifications remain. REPORT.md answers the seven questions; exact per-seed/position data and CLAIM_LEDGER.csv provide the evidence. Reproduce figures from saved numerical tables; do not repeat optimizer operations.
'''
 (OUT/'RESEARCHER_HANDOFF.md').write_text(handoff)
 claims=[]
 paths={'Early selective learning':['tables/update_effects.csv','tables/gain_matrices.csv','tables/gain_matrix_intervals.csv','tables/report_first_blocks.csv','tables/original_early_splitting.csv'],'Starting error and actual gains':['tables/transfer_error_decomposition.csv','tables/realized_gain_by_source.csv','tables/native_curve_metrics.parquet'],'C cue contribution':['tables/cue_contributions.parquet','tables/cue_effect_intervals.csv','tables/report_C_cue_effect.csv','registry/parameter_array_sources.csv','registry/operator_array_sources.csv'],'Parameter endpoint effects':['tables/parameter_interventions.csv','tables/update_audits.csv'],'Conditional Adam history':['tables/adam_controls.csv','tables/adam_control_intervals.csv','tables/report_Adam_means.csv'],'Paired uncertainty and missingness':['tables/seed_summaries.csv','tables/paired_intervals.csv','tables/matched_headroom_paired.csv','tables/missingness.csv','tables/replay_comparison_availability.csv'],'Source state and finite exposure':['registry/source_states.csv','registry/exposure.csv','registry/exposure_by_kind.csv','registry/runtime_identity.json','audit/preflight_review.json'],'Figures':['figures/FIGURE_INDEX.json']}
 for claim,ps in paths.items():
  for n in ps:claims.append(dict(claim=claim,source=n,sha256=sha(OUT/n),qualification='Six paired seeds; descriptive conditional effects, signed values/floors/missingness retained. No norm mediation or separate-human-memory claim.'))
 csv(OUT/'CLAIM_LEDGER.csv',claims);dump(OUT/'audit/report_complete.json',dict(passed=True,separate_questions=7,follow_up_recommended_only=True,all_conditions=True));append('## Scientific report and handoff\nAllseven questions answered with condition-level source-specific first-block effects, initial-error/realized-gain decomposition, cue and optimizer-history controls. No new threshold, weighted winner or meta-training. One smallest frozen derivative follow-up is recommended only, not executed. Original numerical/RNG provenance qualifications remain explicit.')

if __name__=='__main__':main()
