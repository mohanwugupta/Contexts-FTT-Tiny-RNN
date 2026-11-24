"""Scientific report from fixed tables; retain nulls and incomplete evidence."""
from .common import *

def explain_protocol():return '''The eight-unit Linear SRN uses the original manual identity forward, with separate input/recurrent bias tensors. The nn.RNN container's tanh setting is not its executed activation. For each seed, multiply the original High token/all-three-context input matrix and output projection by s/2 at scales4/5/6. Recurrence is initialized once to spectral norm0.9; biases are unchanged. Original per-group High initialization floors are inherited before scaling; no new floors, projection, centering, hidden clipping, recurrent renormalization or numerical rescue occurs.

Target practice is the original seven-token input [0,1,2,0,3,1,2] predicting [1,2,0,3,1,2,3], A10/B10 for330 or990 updates, h0=0 each trial. Native Adam0.005 optimizes CE+0.2 execution loss, with joint model/policy norm1 clipping. Both execution confidence and uncertainty are detached; execution policy state and its contribution to joint clipping remain part of the contract. The C column is fixed bitwise during target practice; it is learned during meta-training. Fresh target Adam/policy is used at every distinct meta dose. Scale4/6 fresh targets reuse exact0–330 states and restore model/policy/Adam through990; target-trained weights never initialize Reptile.

Original P2 Reptile adapts four five-update blocks with a fresh policy/inner Adam,20 steps/episode, and interpolates only prediction tensors. Epsilon decays0.1→0.01 through episode240, then stays0.01 through1,920. Transfer T uses the native held-out shared-context query and original conflicting queries, first-order query gradients, one clip and outer Adam0.001 per episode. Balanced B independently adapts one shared and one conflict task from identical initial weights, combines0.5 shared+0.5 all-position conflict gradients, clips once, and takes960 paired outer updates for1,920 episodes. Its native audit also computes the beta0 unused critical-only gradient. No Reptile inner Adam is inherited.

O1/O2 use two reset examples per native inner update, X10/Y10 adaptation, and query gradients0.2 early+0.5 familiar+0.3 novel. Familiar loss is0.75 all-position CE+0.25 critical-position CE when critical cells exist, otherwise all-position CE. Novel Q queries include only source-registered identical-prefix/shared-target cells. O1 exception counts0/1/2 have proportions0.25/0.50/0.25; O2 has0.50/0.40/0.10. Thus effective critical-only outer mixture mass is0.09375 for O1 and0.0625 for O2, despite unchanged formula. Each has1,920 outer updates and40 example presentations per episode, versus20 in T/Reptile and40 per B pair. These are complete-procedure comparisons; objective ecology, cadence and exposure differ. Observational norm associations are not causal mediation.'''

def tab(x):return x.to_markdown(index=False,floatfmt='.5g')
def main():
 if (OUT/'audit/report_complete.json').exists():return
 assert json.loads((OUT/'audit/analysis_complete.json').read_text())['passed'];read=lambda n:pd.read_csv(OUT/f'tables/{n}.csv',float_precision='round_trip');o=read('outcomes');paired=read('paired_intervals');held=read('conflict_seed');precision=pd.read_csv(OUT/'audit/fomaml_precision_checks.csv');precision_failures=precision[~precision.strict_passed];cycles=read('balanced_cycles');gate=read('functional_splitting');summary=read('outcome_summary');matched=read('matched_paired_intervals');rows=[]
 for (scale,phase,dose,update),g in o.groupby(['scale','phase','dose','update']):
  rows.append(dict(scale=scale,phase=phase,meta_episodes=dose,target_update=update,n=len(g),familiar=int(g.strong_familiar.sum()),C_strict=int(g.strict_C.sum()),Mixed_both=int(g.Mixed_both_success.sum()),D_A=g.D_A.mean(),D_B=g.D_B.mean(),balanced=g.early_cost.mean(),sustained=g.sustained_cost.mean(),A_to_B=g.A_to_B.mean(),B_to_A=g.B_to_A.mean(),familiar_CE=g.familiar_ce.mean(),C_CE=g.C_ce.mean(),C_gap=g.gap.mean(),C_accuracy=g.C_accuracy.mean(),Mixed_both_CE=g.Mixed_both_ce.mean(),NoContext_CE=g.NoContext_ce.mean(),functional_gate=int(g.functional_split.sum()) if update in [170,330] else 'unassessed'))
 cohort=pd.DataFrame(rows);write_frame(OUT/'tables/report_cohort.csv',cohort);fresh=cohort[cohort.phase.eq('Fresh')&cohort.target_update.isin([330,660,990])];final=cohort[cohort.meta_episodes.eq(1920)&cohort.target_update.isin([330,990])];change=paired[paired.dose.eq(1920)&paired['update'].isin([330,990])&paired.reference.eq('R')&paired.metric.isin(['D_A','D_B','early_cost','sustained_cost','gap','C_ce','Mixed_both_ce'])];late=gate.groupby(['scale','phase','dose','update']).functional_split.agg(['sum','count']).reset_index();post=[]
 for key,g in cycles.groupby(['scale','phase','dose']):
  q=g[g.after_first_acquisition&g.both_boundaries_familiar];non=q[~q.both_boundaries_floor];post.append(dict(zip(['scale','phase','dose'],key),post_acquired_cycles=len(q),nonfloor_cycles=len(non),mean_post_raw=q.raw_cost.mean(),mean_post_adjusted=q.adjusted_cost.mean(),mean_nonfloor_raw=non.raw_cost.mean()))
 write_frame(OUT/'tables/post_acquisition_cycles.csv',post);claims=[
 ('Fresh acquisition with extension','tables/report_cohort.csv;tables/outcomes.csv','Report every seed at330/660/990; first criterion crossing is not permanent mastery.'),
 ('Functional learning specificity','tables/blocks.csv;tables/learning_effects.csv;tables/functional_splitting.csv','Signed own/cross gains and original170/330 gate; no late gate invented for990.'),
 ('Switch magnitude and persistence','tables/switch_costs.csv;tables/position_switch_costs.csv;tables/balanced_cycles.csv','At floor, opposite baseline offsets are not persistent behavioral difficulty; balanced raw=adjusted in complete cycles.'),
 ('Reptile changes','tables/paired_intervals.csv;tables/paired_seed_effects.csv','Compare each dose with Fresh at identical target endpoints; six seeds only.'),
 ('FOMAML versus shared parent','tables/paired_intervals.csv;tables/joint_components.csv','Common R1920 reused as +0, not four independent samples; complete-procedure causal contrast, not isolated objective coefficient.'),
 ('Matched-acquisition qualification','tables/matched_paired_intervals.csv;tables/matched_acquisition.csv','All bins/strata and missing eligibility retained; identical330/990 horizons; conditional matching is not mediation.'),
 ('Conflict capability','tables/conflict_seed.csv;tables/conflict_paired_intervals.csv','Twelve fixed unseen tasks averaged within seed; no invented network-level success threshold.'),
 ('Weight organization and scale','tables/meta_geometry.csv;tables/geometry.csv','Actual norms/contrasts are descriptive; geometry is not proof of independent memories.'),
 ('Strict precision exception','audit/fomaml_precision_checks.csv;audit/34_fomaml.log;audit/51_fomaml_readonly_verification_v2.log','The inherited fixed strict replay fails once; unchanged predecessor float32 bound and original gradient/CE/Adam checks pass. Bound consistency does not erase the strict failure.'),
 ('Source fidelity and compute','IMPLEMENTATION_AUDIT.md;registry/parent_hashes.json','All source/checkpoint SHA256, failure/correction and actual exposure retained.')]
 write_frame(OUT/'CLAIM_LEDGER.csv',[dict(claim=c,source=s,limit=l,status='established measurements; interpretation qualified') for c,s,l in claims]);answers=[]
 for scale in [4,5,6]:
  z=fresh[fresh.scale.eq(scale)];answers.append('Fresh scale '+str(scale)+': familiar acquisition '+', '.join(str(int(r.familiar))+'/6 at update'+str(int(r.target_update)) for r in z.itertuples())+'.')
 for scale in [4,5,6]:
  p=paired[paired.scale.eq(scale)&paired.phase.eq('R')&paired.dose.eq(1920)&paired.reference.eq('Fresh')&paired['update'].eq(990)]
  def reffect(metric):
   r=p[p.metric.eq(metric)].iloc[0];return f"{r['mean']:+.4g} [{r['ci_low']:.4g}, {r['ci_high']:.4g}]"
  answers.append(f"Reptile1920 minus Fresh, scale{scale} at target990: C-gap change {reffect('gap')}; first-A specificity {reffect('D_A')}; first-B {reffect('D_B')}; sustained cost {reffect('sustained_cost')}.")
 for scale in [4,5,6]:
  for arm in ['T','B','O1','O2']:
   g=final[final.scale.eq(scale)&final.phase.eq(arm)&final.target_update.eq(990)].iloc[0];p=change[change.scale.eq(scale)&change.phase.eq(arm)&change['update'].eq(990)]
   def eff(metric):
    r=p[p.metric.eq(metric)].iloc[0];return f"{r['mean']:+.4g} [{r['ci_low']:.4g}, {r['ci_high']:.4g}]"
   answers.append(f"Scale{scale}, {arm}: familiar{int(g.familiar)}/6, strict C{int(g.C_strict)}/6, Mixed-both{int(g.Mixed_both)}/6 at990. Versus its R parent, C-gap change {eff('gap')}; first-A specificity {eff('D_A')}; first-B {eff('D_B')}; sustained cost {eff('sustained_cost')}.")
 answers='\n\n'.join(answers)
 text=f'''# High-scale Linear SRN: full meta-learning progression

The complete fixed factorial is reported below. Six paired seeds support an initial mechanistic comparison, not strong population confirmation. The predecessor sweep established amplified balanced and first-block effects, not a fully reliable bidirectional splitting phenotype. This programme starts from that qualified baseline and re-assesses extended acquisition/persistence. Strong C transfer alone does not establish retained functional splitting. Read the signed directional effects, initial-confidence controls, matched-acquisition intervals and acquisition failures alongside the balanced means. No scale, seed, dose or objective was selected using these outcomes.

## Fresh acquisition through990

{tab(fresh)}

This table answers whether familiar acquisition at330 improves by660/990 without stopping any successful run early. All individual criterion crossings and seed endpoints are in outcomes.csv. The original late functional gate is assessed only170/330;990 is explicitly unassessed under that criterion. Complete raw/balanced cycles, original floor flags and post-acquisition cycles separate measurable late effects from loss-floor limitations.

## Reptile and four final FOMAML procedures

{tab(final)}

At each scale, every FOMAML arm starts from the exact same Reptile1920 prediction tensors and fresh outer Adam. Intermediate240/960 target branches stop at330; final Reptile/FOMAML branches run990. Reptile0 aliases Fresh; FOMAML0 aliases one R1920 parent. Absolute familiar/C/Mixed/NoContext losses and C gaps remain distinct. Mixed-average and Mixed-both are separate physical probes.

## Paired final changes from the common Reptile parent

{tab(change[['scale','phase','update','metric','n','mean','median','ci_low','ci_high','n_negative','n_positive']])}

Pointwise95% intervals use10,000 whole-seed paired bootstrap resamples with the original fixed RNG20260925. The six seed IDs are the sampling units; target repetitions, checkpoints, sequence positions and reserved tasks are not independent networks. Each comparison above uses the same target update in parent/child. Mean and median differences can disagree in a heterogeneous six-model cohort. Intervals spanning zero do not establish equivalence or absence of an effect. No multiplicity-adjusted population confirmation is claimed.

## Direct answers from the completed experiment

{answers}

Brackets above are pointwise paired95% intervals. Signs and intervals qualify each directional claim; spanning zero is inconclusive. All acquisition failures remain in the six-seed denominators. The remaining tables retain mean/median paired effects and individual signs.

## Functional interpretation and research questions

1. **Do fresh models eventually acquire A/B?** Use all330/660/990 rows and seed-level first_familiar values above. Failing endpoints remain in denominators, including networks that crossed the criterion transiently.
2. **Does splitting persist after acquisition?** Inspect directional raw and adjusted costs, balanced complete cycles, nonfloor post-acquisition cells and original170/330 gates. A larger early balanced effect or centroid distance does not establish sustained separate learning. Late opposite adjusted offsets at a CE floor are inherited initial-asymmetry constants.
3. **Does Reptile improve transfer while reducing specificity?** Read the R−Fresh paired intervals for D_A/D_B, directional/early/sustained costs, absolute C CE/gap, Mixed-both and NoContext. These are separate outcomes, not a composite score.
4. **Which FOMAML objective preserves splitting while transferring?** The final component table and parent/arm contrasts are the direct evidence. No single strongest joint phenotype is declared from a post hoc weighted ranking. Claim a directional improvement only with its actual interval and acquisition/headroom qualification. The data do not isolate one loss coefficient because procedures differ in cadence, task ecology and example exposure.
5. **Is context-dependent conflict adaptation retained?** Every seed has its critical accuracy/CE, correct-versus-wrong benefit, own/cross gains and number of appropriate task responses in conflict_seed.csv. These tasks are strictly unseen and average within seed; no network-level conflict success threshold was invented.

All continuous values and component flags coexist in joint_components.csv. Matched first-eligible CE bins2–4,1–2,.5–1,.1–.5 are supplied for all/A/B strata separately through330 and, where registered,990. Missing bins remain missing. These comparisons qualify initial-confidence explanations; they do not identify causal mediation. The original splitting gate is unchanged, with all individual results in functional_splitting.csv.

## Reserved conflict capability

{tab(held[held.dose.eq(1920)].groupby(['scale','phase'])[['conflict_accuracy','critical_ce','context_benefit','appropriate_context_change']].mean().reset_index())}

These are means of six seed-level task means. The source-to-parent paired intervals are in conflict_paired_intervals.csv; task repetitions are not independent models.

## Actual algorithms

{explain_protocol()}

## Mathematics and computation

For fixed prediction weights, h_t=R h_(t−1)+W_x x_t+W_c c+b_ih+b_hh and z_t=W_o h_t+b_o. With h0=0, K_t=W_o sum_(j=0)^(t−1)R^j, so z_A−z_B=K_t(w_A−w_B). z(c)=z(0)+K_t W_c c. Physical Mixed-average(.5,.5,0) gives(z_A+z_B)/2 and CE_M≤(CE_A+CE_B)/2 positionwise by convexity; this is not an accuracy guarantee. Mixed-both(1,1,0) gives z_A+z_B−z(0), so it lacks that Jensen guarantee. C=(0,0,1) is a learned third channel in this program and is not NoContext.

Fresh zero biases imply α² initial-logit scaling when input/output matrices both scale by α=s/2, while initial argmax accuracy is unchanged. Meta-learning can change R, biases, norms and directions, making later effects more than simple temperature rescaling. Actual RMS, centered effective contrast, context-column difference and context-free performance are saved. Their associations with behavior are not causal mediation, nor proof that representation geometry consists of separate skills.

D_A=(L_A,before−L_A,after)−(L_B,before−L_B,after); D_B reverses own/unpracticed labels. Switch raw cost=L_incoming(t)−L_outgoing(t) at identical weights. Adjusted cost subtracts the corresponding initial cue difference. Balanced early cost averages complete10/20,30/40,50/60 cycles; sustained early cost uses cycles2/3. Initial baseline terms cancel in complete balanced cycles, so balancing is not an independent initial-error control.

## Figures and exact data

''' + '\n\n'.join(f'![{name}](figures/{path}.png)' for name,path in [('Extended fresh acquisition','figure_1_full990'),('Fresh early view','figure_1_first20'),('Six physical cues','figure_2_six_contexts'),('Reptile progression','figure_3_reptile'),('FOMAML splitting versus C transfer','figure_4_fomaml_balanced'),('Joint behavioral components','figure_5_joint_components'),('Actual weight RMS','figure_6_RMS'),('Effective computation','figure_6_computation')])+'''

All six families have PNG/SVG exports. Figure1 has full990/first100/first20 and symlog close-up companions; Figure4 has separate first-A/first-B companions. All figures regenerate from tables only, with fixed SVG salt and dates omitted. Seed-level outcomes, per-block/switch/position coordinates, all native per-run logits/probabilities/states, cohort intervals and exact paired effects are retained. See REPRODUCIBILITY.md and CLAIM_LEDGER.csv.

## Integrity, negative findings and remaining uncertainty

Initial contracts failed on the predecessor scale registry's correct rejection of5; the new fixed scale4/5/6 transformation changes no predecessor. Numerical-reference tests exposed an inherited packer assumption of already-initialized outer Adam; a reporting-only empty-state adapter preserved fresh Adam and native gradients. A missed330 cue-helper guard stopped the initial extended-target attempt. Every failing log and finite partial state is retained; the explicit target_v2 correction restored exact model/policy/Adam state without repeating any update. The same guard inside the extended switch-table adapter was caught by RED25 and corrected before final analysis. No scientific threshold, learning rate, normalization, objective or budget changed.

The original fixed strict forward tolerance failed in {len(precision_failures)} archived query trace(s):

{tab(precision_failures[['seed','scale','arm','dose','part','max_float32_logit_error','max_logit_bound_ratio']])}

This failure is retained, not reclassified as a strict pass. A separate certificate uses the predecessor's existing64*float32.eps propagation bound and original double tolerance; all cases satisfy that unchanged bound. The original analytic-gradient, query-CE and Adam tolerances also pass unchanged. This establishes consistency with bounded float32 replay, not proof that every difference comes solely from rounding. No parameters, predictions or thresholds were altered, and no model forward branch or optimization was rerun for this algebraic audit.

All six seeds, failed acquisition, signed effects, floors and missing matched bins remain. Evidence cannot establish natural hidden-state retrieval from an optimized state because none was optimized here. Norm correlations do not identify why a particular procedure transfers. Six seeds, pointwise intervals, differing computational exposure and conditional performance matching limit population and mechanistic conclusions. The program stops at the registered budgets; further training requires a new protocol.
'''
 (OUT/'REPORT.md').write_text(text)
 handoff='''# Researcher handoff

Start with REPORT.md's final component tables and paired parent comparisons, then Figures3–5. The scientific question concerns simultaneous functional learning specificity and transfer; no weighted winner was selected.

Teach-back: the same six Gaussian directions produced fresh scale4/5/6 Linear SRNs. Their990-update target learning was diagnostic only. Each fresh model separately learned the original1920-episode P2 Reptile prior, then forked into four first-order procedures from identical prediction weights with fresh outer Adam. Target practice always used the original seven-token sequence, policy/loss/Adam/clip and physical cues, with C frozen. The Balanced procedure makes half as many outer updates; Overlap uses two examples per inner update and early/familiar/shared-prefix novel query gradients. This is a complete-procedure comparison.

D_A/D_B compare practiced versus unpracticed CE gains, while same-weight switch costs compare outgoing versus incoming contexts. Keep both signs, absolute losses, floor indicators and missing matched bins. An early effect is not sustained separate learning; representation distance is not functional learning. At floor, adjusted directional offsets can persist even when raw costs vanish. The original gate remains170/330; no new990 gate is invented.

Linear cue additivity gives a Mixed-average midpoint/Jensen CE bound, but not one for Mixed-both or C. Initially doubling both input/output weights quadruples logits without changing class argmax accuracy. Later norms/directions/recurrence can change, so use actual RMS and effective contrast. A norm association is not a causal explanation.

Every target/meta/conflict registry and checkpoint hash is indexed. Corrected-v2 diagnostic tasks are excluded from training and averaged within seed. Inspect conflict_seed.csv for appropriate-response task counts; no network-level success threshold is invented. Read all mean/median paired intervals and negative outcomes. Six seeds provide preliminary evidence, not strong population confirmation.

One archived Overlap query trace fails the inherited fixed strict forward tolerance; the original failure and thresholds remain. Its separate unchanged predecessor float32-bound certificate and original gradient/CE/Adam checks pass. Do not say every strict replay passed.

The initial330-horizon guard error and all finite partial states remain in audit/ and target/. The documented target_v2 correction continued them exactly; it did not restart Adam or replay target steps. Reproduce with hash verification and external table-only figures; never rerun scientific optimization for plots. Fresh0 and FOMAML+0 are explicit aliases, not extra independent networks.
'''
 (OUT/'RESEARCHER_HANDOFF.md').write_text(handoff+'\n## Actual completed results\n\n'+answers+'\n\nIntervals are paired pointwise95% whole-seed bootstrap intervals; any spanning zero remains inconclusive. See the report and conflict tables for capability components and all negative outcomes.\n')
 dump(OUT/'audit/report_complete.json',dict(passed=True,claims=len(claims),threshold_changes=0,selection=False));append('## Reports complete\nFull source-to-claim ledger, actual algorithm/equations, signed final/paired results, all corrections and scientific limitations recorded. No strongest joint score invented. Researcher teach-back distinguishes early effects, floor limitations, context additivity and procedure/exposure confounds.')
if __name__=='__main__':main()
