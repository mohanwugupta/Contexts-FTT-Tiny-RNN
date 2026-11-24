"""Scientific conclusions from the complete fixed factorial; no selection or tuning."""
from .common import *
from .statistics import interval
from .budget import plan
ATTRIBUTION='P−U estimates the complete projected regime beginning during Reptile. It does not isolate a FOMAML-only projection effect because the parents already differ.'
TEACHBACK='Preserving RMS constrains magnitude, not parameter directions or effective computation. Linear interpolation explains Mixed-average; it does not imply natural access to a useful C state. No oracle states are used, and an algebraic existence result is not natural retrieval.'

def paired_value(f,scale,arm,metric,update):
 q=f[f.scale.eq(scale)&f.phase.eq(arm)&f.dose.eq(10000)&f['update'].eq(update)];p=q.pivot(index='seed',columns='condition',values=metric).reindex(SEEDS);return interval((p['P'].astype(float)-p['U'].astype(float)).to_numpy(float))
def tab(f):return f.to_markdown(index=False,floatfmt='.4g')
def effect(r):return f"{r['mean']:+.4g} [{r['ci_low']:.4g}, {r['ci_high']:.4g}]; median {r['median']:+.4g}; n={r['n']}"

def explain_protocol():return '''The eight-unit Linear SRN executes the native identity forward with separate input/recurrent biases. Fresh token/all-three-context input and output matrices are the exact archived High tensors scaled by s/2. Recurrence was normalized once to spectral norm0.9 in that original initialization; it is not renormalized here.

Teacher-forced target practice predicts [1,2,0,3,1,2,3] from [0,1,2,0,3,1,2]. It uses alternating A10/B10, h0=0 each trial, fresh native model/policy Adam0.005, CE+0.2 detached-confidence execution loss and joint norm1 clipping. Detached confidence and uncertainty, policy gradients and their clipping contribution remain unchanged. C is fixed bitwise throughout target practice, although it is learned during meta-training.

Original P2 Reptile independently adapts a fresh inner Adam/policy for20 updates in four five-update support blocks. It interpolates prediction tensors only, with epsilon0.1→0.01 through episode240 and0.01 thereafter through1920. T takes first-order query gradients after the native20-step held-out-context support, preserving shared/conflicting task queries, one joint clip and outer Adam0.001 per episode. B independently adapts one shared and one conflicting task from identical starting weights, combines0.5 shared+0.5 all-position conflict gradients, clips once and takes one outer Adam step per pair. Its native beta0 auxiliary critical gradient is still computed and recorded.

O1/O2 process two reset sequence examples per inner update, X10/Y10 adaptation, with outer gradients0.2 early+0.5 familiar+0.3 novel. Familiar loss is0.75 all-position CE+0.25 critical-position CE when eligible, otherwise all-position CE. Novel queries use only registered identical-prefix/shared-target cells. O1's exception-count proportions0/1/2 are25/50/25%; O2's are50/40/10%. Full native blocks and the original query masks are unchanged. The final10000-dose partial block is consumed in native order; its realized exposure is reported rather than rebalanced after observing outcomes.'''

def cohort(out):
 rows=[]
 for key,g in out.groupby(['scale','condition','phase','dose','update']):
  assert len(g)==6 and not g.seed.duplicated().any();tag=dict(zip(['scale','condition','phase','dose','update'],key));rows.append(dict(tag,n=6,familiar=int(g.strong_familiar.sum()),strict_C=int(g.strict_C.sum()),original_C=int(g.C_success.sum()),Mixed_both=int(g.Mixed_both_success.sum()),Mixed_average=int(g.Mixed_average_success.sum()),original_gate=int(g.functional_split.sum()) if key[-1] in [170,330] else np.nan,D_A=g.D_A.mean(),D_B=g.D_B.mean(),balanced=g.early_cost.mean(),A_to_B=g.A_to_B.mean(),B_to_A=g.B_to_A.mean(),sustained=g.sustained_cost.mean(),familiar_CE=g.familiar_ce.mean(),C_CE=g.C_ce.mean(),C_gap=g.gap.mean(),C_gap_median=g.gap.median(),C_accuracy=g.C_accuracy.mean(),Mixed_both_CE=g.Mixed_both_ce.mean(),Mixed_average_CE=g.Mixed_average_ce.mean(),NoContext_CE=g.NoContext_ce.mean(),NoContext_accuracy=g.NoContext_accuracy.mean(),NoContext_minus_C=g.NoContext_minus_C.mean(),positive_D_A=int(g.positive_D_A.sum()),positive_D_B=int(g.positive_D_B.sum()),nonfloor_first_both=int(g.nonfloor_first_both.sum())))
 return pd.DataFrame(rows)

def main():
 if (OUT/'audit/report_complete.json').exists():return
 assert json.loads((OUT/'audit/analysis_complete.json').read_text())['passed'];read=lambda n:pd.read_csv(OUT/f'tables/{n}.csv',float_precision='round_trip');o=read('outcomes');p=read('paired_intervals');c=cohort(o);write_frame(OUT/'tables/report_cohort.csv',c);final=c[c.phase.isin(ARMS)&c.dose.eq(10000)&c['update'].isin([330,660,990])];r=c[c.phase.eq('R')&c.dose.eq(1920)&c['update'].isin([330,990])];primary=p[p.kind.eq('P-U')&p.phase.isin(ARMS)&p.dose.eq(10000)&p['update'].isin([330,990])];write_frame(OUT/'tables/final_primary_effects.csv',primary);geom=read('meta_geometry');proj=read('projection_summary');conf=read('conflict_seed');cs=read('conflict_paired_intervals');matched=read('matched_acquisition');cycles=read('balanced_cycles');post=[]
 for key,g in cycles.groupby(['seed','scale','condition','phase','dose']):
  z=g[g.after_first_acquisition&g.both_boundaries_familiar];non=z[~z.both_boundaries_floor];post.append(dict(zip(['seed','scale','condition','phase','dose'],key),post_acquisition_cycles=len(z),nonfloor_post_acquisition_cycles=len(non),mean_post_raw=z.raw_cost.mean(),mean_post_adjusted=z.adjusted_cost.mean(),mean_nonfloor_raw=non.raw_cost.mean()))
 write_frame(OUT/'tables/post_acquisition_cycles.csv',post);precision=[]
 for phase in ['B','C','D']:
  path=OUT/f'audit/phase_{phase}_precision.csv'
  if path.exists():precision.append(pd.read_csv(path).assign(verification_phase=phase))
 prec=pd.concat(precision,ignore_index=True) if precision else pd.DataFrame();fail=prec[~prec.strict_passed] if len(prec) else prec;write_frame(OUT/'tables/strict_precision_exceptions.csv',fail)
 # PhaseD P verification includes its full0–10000 prefix, so repeated audited traces are not independent failures.
 unique_fail=fail.drop_duplicates(['seed','scale','condition','arm','dose','part']) if len(fail) else fail
 comparisons=[];answers=[]
 for scale in SCALES:
  for arm in ARMS:
   d={m:paired_value(o,scale,arm,m,330) for m in ['D_A','D_B','early_cost','A_to_B','B_to_A','sustained_cost','gap','C_ce','Mixed_both_ce','NoContext_ce']};comparisons.append(dict(scale=scale,objective=arm,**{m:effect(v) for m,v in d.items()}));z=final[final.scale.eq(scale)&final.phase.eq(arm)&final['update'].eq(990)].set_index('condition');answers.append(f"Scale{scale}, {arm}: target990 familiar U/P={int(z.loc['U','familiar'])}/{int(z.loc['P','familiar'])} of6; strict C={int(z.loc['U','strict_C'])}/{int(z.loc['P','strict_C'])}; Mixed-both={int(z.loc['U','Mixed_both'])}/{int(z.loc['P','Mixed_both'])}. At matched target330, P−U first-A {effect(d['D_A'])}, first-B {effect(d['D_B'])}, sustained cost {effect(d['sustained_cost'])}, C-gap {effect(d['gap'])}.")
 comp=pd.DataFrame(comparisons);write_frame(OUT/'tables/report_primary_summary.csv',comp);headline=[]
 for m in ['D_A','D_B','early_cost','A_to_B','B_to_A','sustained_cost','gap','C_ce','Mixed_both_ce','NoContext_ce']:
  z=primary[primary['update'].eq(330)&primary.metric.eq(m)];assert len(z)==12;headline.append(f"For {m}, the12 prespecified scale×objective paired comparisons have{int(z.ci_low.gt(0).sum())} intervals wholly above0,{int(z.ci_high.lt(0).sum())} wholly below0 and{int((z.ci_low.le(0)&z.ci_high.ge(0)).sum())} spanning0. These are dependent feasibility comparisons, not independent replications or multiplicity-adjusted discoveries.")
 tg=read('target_geometry_paired_intervals');z=tg[tg.kind.eq('P-U')&tg.phase.isin(ARMS)&tg.dose.eq(10000)&tg['update'].eq(330)&tg.metric.eq('centered_contrast')];assert len(z)==12
 headline.append(f"For effective class-centered A/B contrast at the SAME frozen target330 states, P−U intervals are wholly positive in{int(z.ci_low.gt(0).sum())} comparisons and wholly negative in{int(z.ci_high.lt(0).sum())}. Absolute values and every seed are in same_state_behavior_geometry.csv; small norm differences and geometric separation do not establish separate learning.")
 # Transparent component co-occurrence, without a newly weighted success rule.
 joint=o[o.phase.isin(ARMS)&o.dose.eq(10000)&o['update'].isin([330,990])][['seed','scale','condition','phase','dose','update','strong_familiar','D_A','D_B','positive_D_A','positive_D_B','floor_A','floor_B','early_cost','A_to_B','B_to_A','sustained_cost','strict_C','Mixed_both_success','NoContext_ce','functional_split']];write_frame(OUT/'tables/final_joint_components.csv',joint)
 claim_rows=[]
 sources=[('Exact source and optimizer continuation',['registry/initializations.csv','registry/U_checkpoints.csv','audit/source_audit.json','audit/benchmark.json'],'Verified exact fresh tensors and original U1920 moments/counters; no historical retraining.'),('Minimum RMS throughout training',['tables/projection_summary.csv','registry/scale_histories.csv','audit/phase_D_verified.json'],'All radial moves independently reconstructed; R/bias/policy untouched by projection, moments preserved.'),('Functional splitting and floors',['tables/blocks.csv','tables/switch_costs.csv','tables/functional_splitting.csv','tables/post_acquisition_cycles.csv'],'Signed own/cross and both directions; original170/330 gate; no990 gate or floor-as-sharing inference.'),('Paired projected regime effect',['tables/final_primary_effects.csv','tables/paired_seed_effects.csv'],ATTRIBUTION),('Generalization and joint components',['tables/outcomes.csv','tables/final_joint_components.csv'],'C, Mixed-average, Mixed-both and NoContext separate; no posthoc weighted score.'),('Performance matching',['tables/matched_acquisition.csv','tables/matched_paired_intervals.csv'],'All original bins/strata, planned6 seeds and missing eligibility; conditional, not mediation.'),('Unseen conflict adaptation',['tables/conflict_seed.csv','tables/conflict_paired_intervals.csv'],'Twelve strictly unseen tasks averaged within each seed, no task pseudoreplication.'),('Effective computation',['tables/meta_geometry.csv','tables/effective_context_by_position.csv','tables/parameter_directions.csv','tables/same_state_behavior_geometry.csv','tables/target_geometry_paired_intervals.csv'],'Seven-position class-centered contrast and actual directions; geometry does not establish separate memories.'),('Numerical precision limits',['tables/strict_precision_exceptions.csv','audit/phase_D_verified.json'],'Strict failures retained distinctly from existing float32 propagation-bound consistency.'),('Complete computational exposure',['registry/exposure.csv','audit/exposure_verified.json'],'Fixed budgets; no optimization for reproduction; all fixtures accounted separately.')]
 for title,paths,limit in sources:
  for path in paths:
   target=OUT/path;claim_rows.append(dict(claim=title,source=path,sha256=sha(target) if target.exists() else 'pending_finalization',scope=limit))
 for row in primary.to_dict('records'):
  claim_rows.append(dict(claim='Registered P-U outcome',source='tables/final_primary_effects.csv',sha256=sha(OUT/'tables/final_primary_effects.csv'),scope=f"scale{row['scale']};{row['phase']};target{row['update']};{row['metric']}",mean=row['mean'],median=row['median'],ci_low=row['ci_low'],ci_high=row['ci_high'],n=row['n'],n_planned=row['n_planned']))
 write_frame(OUT/'CLAIM_LEDGER.csv',claim_rows)
 absolute=final[final['update'].eq(330)]
 late_answer='No final condition recovered the original late splitting criterion at target330.' if absolute.original_gate.sum()==0 else 'The original late splitting criterion is met by some final models at target330; the complete condition-level counts are shown below, without selecting a successful arm.'
 text=f'''# High-scale-constrained meta-learning through10,000 FOMAML episodes

All144 registered FOMAML trajectories reached10,000 episodes after1,920 Reptile episodes. The fixed P−U contrast tests whether preserving high initial RMS throughout both stages helps retain functional A/B-specific acquisition alongside transfer. It does not select a scale, objective, seed or checkpoint. {ATTRIBUTION}

The projection's numerical success and behavioral success are separate questions. The tables below report every condition, including negative effects, familiar failures, floor limitations and missing comparisons. Six paired seeds are an initial scaling study; pointwise intervals do not provide strong population confirmation.

## Direct findings

{late_answer}

'''+ '\n\n'.join(headline)+f'''

## Final acquisition and transfer

Counts are out of6 per row. CE values are arithmetic means across those same six trained seeds, without pooling scales or procedures. See cohort_summary.csv for mean/median95% intervals and all original intermediate doses.

{tab(final[['scale','condition','phase','update','familiar','strict_C','Mixed_both','familiar_CE','C_CE','C_gap','C_gap_median','C_accuracy','Mixed_both_CE','NoContext_CE']])}

Absolute C CE and the signed C gap are distinct. A smaller gap may accompany worse familiar learning. Mixed-average has its own interpolation advantage and remains in the full component tables; it is never merged with Mixed-both. The final target states330/660/990 belong to the same physical target history. At330 the original late gate is retained; at990 it is unassessed, not failed or passed.

The inherited strong-familiar criterion requires CE≤0.5 and accuracy≥6/7 separately in A and B. Original cue success requires that cue's CE≤0.5 and accuracy≥6/7. Strict C requires CE_C≤0.1, accuracy_C=1 and G_C≤0.1; familiar acquisition remains a separate component rather than being silently added to this flag.

## Does P retain stronger functional learning?

Absolute final effects at target330 are shown first; the original gate counts are out of6. First-block effects occur before the endpoint and must not be interpreted as evidence that late learning remains separate.

{tab(absolute[['scale','condition','phase','D_A','D_B','balanced','A_to_B','B_to_A','sustained','nonfloor_first_both','original_gate']])}

The following P−U comparisons use identical target330 weights within each frozen evaluation and paired seeds across arms. Each cell reports mean difference [pointwise95% CI], median difference and actual n. Raw gains, starting losses/accuracy, all directional costs, margins and original floor flags remain in their exact tables.

{tab(comp[['scale','objective','D_A','D_B','early_cost','A_to_B','B_to_A','sustained_cost']])}

Do not interpret a positive balanced cost as both directions being positive. Initial cue-asymmetry terms cancel in complete balanced cycles. At a CE floor, persistent opposite adjusted offsets can reflect the original baseline, rather than a measurable new learning cost. post_acquisition_cycles.csv separates nonfloor cycles after familiar acquisition; missing nonfloor cycles do not establish shared learning.

## Does stronger splitting coexist with transfer?

{tab(comp[['scale','objective','gap','C_ce','Mixed_both_ce','NoContext_ce']])}

final_joint_components.csv retains the actual seed-level components, with no new composite threshold or weighted ranking. All original170/330 gates are in functional_splitting.csv. Better C alone is not evidence of preserved independent skills. The primary same-dose P−U effect includes the changed Reptile parent; neither norm correlations nor target performance matching identify mediation.

Figure5 overlays × wherever the existing familiar A/B criterion is not met, while retaining every seed line and the unfiltered cohort mean. Missing criterion flags are not relabelled as failures.

## Changes already present after Reptile

{tab(r[['scale','condition','update','familiar','strict_C','Mixed_both','D_A','D_B','balanced','sustained','C_CE','C_gap','NoContext_CE']])}

P starts from the exact original fresh tensors, never from rescaled pretrained U. All P Reptile branches use the original1,920 tasks. Reptile240/960/1920 versus Fresh, and every FOMAML dose versus its own Reptile parent, are in paired_intervals.csv with individual signed changes in paired_seed_effects.csv. Phase C was an integrity checkpoint, not a behavioral selection gate: every arm continued. P-F1920 has only the registered330 target horizon; historical U-F1920 has990. Their990 paired comparison remains missing. A1920-to10000 P change at990 is likewise unregistered; no extended states were invented.

## Conflict adaptation and effective computation

{tab(conf[conf.phase.isin(ARMS)&conf.dose.eq(10000)].groupby(['scale','condition','phase'])[['conflict_accuracy','critical_ce','conflict_context_benefit','appropriate_context_change']].mean().reset_index())}

Each displayed value first averages12 reserved tasks within each seed, then six seeds. These tasks never occur in any training library. conflict_paired_intervals.csv retains all correct/wrong-cue, own/cross and1920-to10000 changes; no new network-level conflict success threshold is imposed.

The cue-benefit sign is CE_wrong−CE_correct: a positive value favors the appropriate contextual cue. Critical-position accuracy/CE are reported separately from the seven-position average, which can hide a local conflict error.

Projection was applied to five independent groups: token inputs, three cue columns and hidden-to-output weights. Complete pre-outer/raw/post-projection RMS, multipliers, flags and displacement histories are in tables/scale_histories/*.npz and their schema/index. projection_summary.csv reports every seed's floor compliance and trigger frequency. parameter_directions.csv and meta_geometry.csv retain direction changes, recurrent spectral norm and cue-column similarities. Effective A/B contrast is measured for all seven positions, not inferred from raw RMS.

Figure6 and meta_geometry.csv describe the meta initializer BEFORE target practice. same_state_behavior_geometry.csv joins every behavioral outcome to the EXACT matching frozen target state, and target_geometry_paired_intervals.csv gives its six-seed paired contrasts. Use those target330/990 rows to assess transfer alongside contextual sensitivity. No projection occurs during target practice, so a P target state's RMS need not still meet its meta-training floor. First-block D_A/D_B remain properties of the earlier learning trajectory, not a measurement of late separate learning.

## Algorithms and equations

The native algorithms are reused without changing their losses, masks or optimizer contracts. P additionally receives the radial operation specified below.

{explain_protocol()}

For each P group g, r_g=√mean(W_g²) is computed in float64 and the registered floor is its actual fresh RMS. After each Reptile interpolation or outer Adam step only, W_g←W_g max(1,r_min,g/r_g), with float32 multiplication. Outer moments/counters are unchanged. No inner or target projection, no recurrent/bias/policy projection and no renormalization are performed. Zero/nonfinite group norms stop the run. The original floor tolerance remains atol1e−7+rtol2e−7×floor.

U continuation restores its exact model/outer Adam/RNG/task cursor at FOMAML1920. P forks T/B/O1/O2 from identical constrained Reptile1920 weights with native fresh outer Adam. B performs one combined outer move per two episodes: its final counter is5000, versus10000 for the other objectives. Overlap uses two reset examples per inner update, so sequence and query exposures differ; these are complete-procedure comparisons.

For h_t=R h_(t−1)+W_x x_t+W_c c+b_ih+b_hh and h0=0,
K_t=W_o Σ_(j=0)^(t−1)R^j and z_A−z_B=K_t(w_A−w_B).
The centering operator Π removes each four-class common offset. A large input RMS need not preserve ‖ΠK_t(w_A−w_B)‖: directions, recurrence and output mapping can change. Geometry is not evidence of independent memories or causal mediation.

Mixed-average produces(z_A+z_B)/2 and has the positionwise Jensen CE bound L_M≤(L_A+L_B)/2. This does not guarantee its argmax accuracy. Mixed-both produces z_A+z_B−z_0 and has no such bound. C=(0,0,1) is a learned nonzero channel here; it is not NoContext. No optimized hidden states, cue search or oracle rescue are used.

D_A=(L_A,start−L_A,end)−(L_B,start−L_B,end), with the own/unpracticed labels reversed for D_B. Switch raw cost compares incoming versus outgoing CE at identical weights; adjustment subtracts the corresponding initial difference. The original loss bins2–4,1–2,.5–1,.1–.5 and all/A/B strata are retained through the registered330/990 horizons. Missing cells stay missing; ratios with tiny denominators are not needed.

The unchanged late gate combines mean complete-cycle cost>0.05 with at least50% of cycles>0.05, and the endpoint restored-Adam diagnostic: both signed one-update specificities>0, their mean>0.001, mean own improvement>1e−4 and minimum own improvement>−1e−6. This gate is evaluated only at170/330. It is distinct from first-block D_A/D_B and from geometric separation.

## Numerical and computational audit

Exactly{plan()['new_meta_episodes']:,} new meta episodes were authorized, with{plan()['new_inner_updates']:,} inner Adam updates,{plan()['new_outer_Adam_updates']:,} FOMAML outer Adam updates and34,560 Reptile interpolations. There are846 new target histories and288 exact historical references, plus522 new conflict diagnostic cells and90 archived references. The derived504-diagnostic arithmetic mistake was corrected before diagnostic execution; its original log and plan remain. Full actual support/query/target/diagnostic/fixture exposure is in IMPLEMENTATION_AUDIT.md and registry/exposure.csv.

The strict inherited forward reference has{len(unique_fail)} distinct newly audited exceptions (PhaseD repeats of P-prefix checks are not new observations). They remain strict failures. All accepted checks also satisfy the separately labelled pre-existing float32 propagation bound and original analytic-gradient/query-CE/Adam tolerances. The predecessor's original strict exception remains in its sealed history. A bound-consistency certificate does not erase a strict failure or prove every error is solely rounding. No tolerance, task, learning rate or budget was changed after outcomes.

## Figures, data and uncertainty

Seven PNG/SVG families:1 actual five-group RMS and recurrent norm;2 original pre-update A10/B10 practice curves (330/100/20/990);3 functional splitting and directions;4 C/Mixed/NoContext transfer;5 joint components;6 effective seven-position contextual computation;7 unseen conflict capability. Each objective has separate companion plots. Exact table-only sources and hashes are in figures/FIGURE_INDEX.json and REPRODUCIBILITY.md.

The [complete figure index](figures/README.md) links every companion. Fixed examples below use T, the first registered objective, and are not chosen for performance:

- [1: Reptile weight scales](figures/figure_1_scale_R.png).
- [2: Original early practice](figures/figure_2_T_20.png) and [full990 practice](figures/figure_2_T_990.png).
- [3: Reptile retention](figures/figure_3_Reptile.png) and [FOMAML retention](figures/figure_3_splitting_T.png).
- [4: C, Mixed and NoContext](figures/figure_4_transfer_T.png).
- [5: Joint components](figures/figure_5_joint_T.png).
- [6: Effective contextual computation](figures/figure_6_effective_T.png).
- [7: Unseen conflict adaptation](figures/figure_7_conflicts_T.png).

All measurements and claims are indexed in CLAIM_LEDGER.csv. Only six paired seeds were trained, and conditions reuse each seed and parent; conditions/checkpoints/tasks are not independent networks. Mean and median effects may differ. Pointwise intervals are not multiplicity-adjusted. Conditional performance matching and raw norm associations do not resolve causation. Natural retrieval is not established by any algebraic identity. All registered budgets are spent; further optimization requires a new protocol.
'''
 (OUT/'REPORT.md').write_text(text)
 (OUT/'RESEARCHER_HANDOFF.md').write_text('# Researcher handoff\n\n'+ATTRIBUTION+'\n\nTeach-back: begin with the original fresh high-scale Linear SRN. U reuses all historical1920 training and continues actual Adam; P learns from the same fresh tensors with radial minimum-RMS projection throughout Reptile and FOMAML. The projection changes only five group magnitudes after outer moves. Inner adaptations and target A10/B10 learning remain unchanged. All scales and objectives run to the fixed budget.\n\n'+TEACHBACK+'\n\nRead signed D_A and D_B with own/cross gains, both switch directions, initial headroom and floor flags. The original170/330 splitting criterion is unchanged; no990 gate is invented. C, Mixed-both and NoContext characterize distinct computations. Twelve unseen conflicts are averaged within each trained seed. Use the mean, median, interval and individual signs, not a weighted joint winner.\n\nActual final comparisons:\n\n'+'\n\n'.join(answers)+'\n\nEvery bracket is a pointwise six-seed paired bootstrap interval. Intervals spanning zero are inconclusive. Complete final components, match-bin eligibility and negative outcomes remain in the tables. See IMPLEMENTATION_AUDIT.md for failures, continuation identities and exposure; REPRODUCIBILITY.md for hash verification and table-only figures. Do not rerun training or evaluation to reproduce plots.\n')
 dump(OUT/'audit/report_complete.json',dict(passed=True,primary_scale_objective_cells=12,all_seeds=True,no_score=True,strict_distinct_exceptions=len(unique_fail),attribution=ATTRIBUTION));append('## Report and researcher handoff\nAll final scale/objective P−U comparisons, signed outcomes, criterion/floor limitations and explicit numerical exceptions retained. No winner, new990 gate, posthoc score or natural-retrieval claim. All claims trace to exact tables. Finalization must fill exposure hashes before sealing.')
if __name__=='__main__':main()
