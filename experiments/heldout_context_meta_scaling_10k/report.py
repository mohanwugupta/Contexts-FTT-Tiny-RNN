"""Nine-question scientific report and teach-back from complete saved tables."""
from .common import *
from .protocol import DOSES,NEW_DOSES,AUDIT_EPISODES
QUESTIONS=[
'Does held-out-context meta-learning continue improving transfer from960 to10000 episodes?',
'Does the improvement plateau, accelerate, or deteriorate?',
'Does the Seen-context control improve similarly?',
'Does the C learning curve approach the familiar A/B curves?',
'Are improvements shared across most networks or concentrated among outliers?',
'Does functional A/B-specific learning survive additional pretraining?',
'Can the networks still learn genuinely context-dependent conflicting tasks?',
'How does the computation underlying C generalization change?',
'How close are the networks to reproducing the joint human pattern?']

def interpret_interval(lo,hi):
 if hi<0:return 'decrease supported by this pointwise interval'
 if lo>0:return 'increase supported by this pointwise interval'
 return 'uncertain direction; neither benefit nor equivalence established'

def sample_n(row):
 values=[row[k] for k in ['n','n_seeds'] if k in row]
 if not values or not all(np.isfinite(v) and v>=0 and float(v).is_integer() for v in values) or len(set(values))!=1:raise ValueError('Missing/conflicting actual seed denominator')
 return int(values[0])

def fmt(x):return f'{float(x):.5f}' if pd.notna(x) else 'missing'
def ci(r):return f"{fmt(r['mean'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n'
def filtered(f,**values):
 for k,v in values.items():f=f[f[k].isin(v) if isinstance(v,list) else f[k].eq(v)]
 return f

def scientific_takeaways():
 end=filtered(read('endpoints'),dose=10000,update=330,cue='C',arm='Transfer')
 rows=[]
 for c in ['P1-I','P2-I']:
  g=filtered(end,condition=c);rows.append(f"{c}: C CE {fmt(g.ce.mean())}, familiar CE {fmt(g.familiar_ce.mean())}, strict C success {int(g.strict_all.eq(1).sum())}/{len(g)}")
 held=read('heldout_summary');wrong=read('heldout_wrong_summary');changes=read('heldout_paired_intervals')
 conflict=[]
 for c in ['P1-I','P2-I']:
  a=filtered(held,condition=c,arm='Transfer',dose=10000,metric='conflict_accuracy').iloc[0]
  d=filtered(changes,condition=c,dose=10000,contrast='Transfer:10000-960',metric='conflict_accuracy').iloc[0]
  w=filtered(wrong,condition=c,arm='Transfer',dose=10000,metric='context_benefit').iloc[0]
  conflict.append(f"{c}: conflict-position accuracy {fmt(a['mean'])}, paired change from 960 {ci(d)}; correct-cue CE advantage over wrong cue {ci(w)}")
 return '''## Observed results and their limits

Scaling closes the large C deficit for the independent Transfer cohorts at the registered final target state: '''+'; '.join(rows)+'''. Familiar acquisition remains strong. All four primary paired gap intervals favor Transfer or additional training, but the much smaller median changes show that large mean gains partly reflect the difficult tail of seeds. This supports successful target-C transfer under this task ecology, not uniform large improvement or arbitrary novel-cue robustness.

Functional specialization is only partly retained. At 10000 episodes, P1-I Transfer has positive first-B specificity but its first-A interval spans zero; P2-I Transfer has positive first-A and first-B intervals. P1-I's mean first20 specificity decreases from 960 and relative to Seen; the corresponding P2-I intervals span zero, which cannot establish preservation. The inherited first20 floor flag is zero for every final-dose seed in every arm: early weakening cannot be explained by that measured floor. Late near-zero losses still limit endpoint probes. Fixed-bin comparisons have small and sometimes zero eligible paired samples; they cannot settle persistent separate learning.

The original functional gate passes no final-dose network at either target state 170 or 330. The separate early20 descriptive joint pattern occurs in 12/20 P1-I Transfer and 10/20 P2-I Transfer seeds. Mixed-both remains heterogeneous, and its paired mean-loss intervals for both independent Transfer-versus-Seen and 10000-versus-960 contrasts span zero. Successful C transfer therefore does not establish the full joint pattern.

Reserved conflict adaptation reveals a cost rather than clear preservation of contextual capability. '''+'; '.join(conflict)+'''. Both conflict-accuracy change intervals are below zero. A positive correct-versus-wrong cue advantage survives, so these networks are not wholly context-insensitive; nevertheless, accurate disambiguation of conflicting sequences has weakened. Large loss reductions during adaptation partly reflect very poor starting predictions and do not by themselves establish faster learning. These are secondary diagnostics, not a criterion used to select the training dose.

Seen also improves substantially. Seen and Transfer already differ at 960; the final arm comparison includes their cumulative histories. The secondary scale interaction does not show a general extra scaling benefit specific to Transfer. No plateau, causal mediation by weight geometry, persistent separate skills, or human-like joint behavior is established.

'''

def render():
 assert json.loads((OUT/'audit/analysis_complete.json').read_text())['passed'];primary=read('primary_intervals');end=read('endpoints');end=end[end['update'].eq(330)];effects=read('scaling_intervals');summary=read('endpoint_summary');acq=read('acquisition');post=read('post_scaling_intervals');b=read('block_transfer');matched=read('matched_scaling_intervals');joint=read('joint_early20');held=read('heldout_summary');mechanism=read('mechanism_scaling_intervals');position=read('position_scaling_intervals');early=read('early20_intervals');meta=json.loads((OUT/'audit/meta_complete.json').read_text());bench=json.loads((OUT/'audit/benchmark.json').read_text())
 text='''# Held-out-context FOMAML scaling to10,000 episodes

The registered endpoint is the signed context-C loss disadvantage, `G_C = CE_C − (CE_A+CE_B)/2`, after330 target updates. The co-primary question is functional A/B-specific learning during acquisition. All20 paired seeds remain in each P1/P2 × independent/initially centered condition and both Seen/Transfer arms. Dose counts refer only to FOMAML; the original1920 Reptile episodes precede them.

## What was executed

The160 exact episode960 prediction models **and outer Adam moments/step counters** continued for9040 episodes each. The architecture, objective, optimizer settings,20-step inner schedule, task roles and target task are unchanged. No checkpoint was chosen using C/Mixed results. Historical0 and existing240/960 target histories are reused, not new replications.

Each episode copies the initialization into a task model, resets inner Adam and the two-parameter execution policy, and performs20 support updates using CE+0.2 detached-confidence execution loss, Adam0.005, and joint norm1 clipping. Query CE is then differentiated with respect to the adapted prediction weights. Those gradients are mapped to the initialization, clipped at norm1, and applied by persistent outer Adam0.001 with the inherited moments. There is no derivative through Adam, query inner update, outer policy update, Reptile interpolation, recurrent projection, new encoder or architecture change.

Shared-task Transfer support excludes query context Q; Seen support includes Q plus a counterbalanced companion. Both query the same sequence under Q. The support-membership difference is inherent; complete-block marginal exposures are balanced. Conflict episodes are identical between arms: P2 stable assignments, P1 balanced ambiguity; query losses are means, not sums scaled by batch size.

All complete240-episode blocks retain120 shared/120 conflict tasks and exact contextual role/exposure balance. Episode10000 is the first160 episodes of block42, kept in its original order. Its realized task/context counts are saved; it is not retrospectively rebalanced. The corrected diagnostic full eight-symbol paths are reserved before new generation. Common subsequences remain allowed, and conflict prefixes are deliberately shared. New registries, exclusions, audit subset and storage choices were sealed before scientific training.

Each new dose1920/3840/5760/7680/10000 supplies160 fresh target runs:330 original A10/B10 updates, fresh policy/Adam, no C or Mixed practice, and a bitwise-fixed C column. All331 model/policy/Adam states, six cues, seven positions, logits, probabilities and hidden states are retained. Actual practiced loss at repetition k is measured before its update, at frozen state k−1. Same-weight A/B evaluations define learning transfer and switch costs; no free-running generation is tested.

'''
 text+=table(['Budget','Completed'],[['Continued meta runs',meta['runs']],['Additional meta episodes/outer updates',meta['outer_updates']],['Additional inner updates',meta['inner_updates']],['New target runs/updates','800 /264000'],['Target restored-Adam clone updates',9600],['New conflict clone updates',115200],['Historical target histories reused',400],['Total new scientific optimizer steps',30763200],['Separate implementation fixtures','180 inner +9 outer']])
 text+='''
The continuation RNG is the original PCG64 stream keyed by seed/block. The predecessor has no saved global torch RNG snapshot, but its update contains no stochastic operation: task models are copied, policies reset to zero, all data are registered, and constructor RNG is forked/restored. The benchmark verifies that the update leaves global torch RNG unchanged. No required task RNG state was approximated. New restart states include global torch/NumPy/Python snapshots and the sealed task-registry hash/cursor.

The storage protocol explicitly changes observation retention: every episode atomically commits a compact scalar/query record and checksummed complete restart state; full21-state inner traces are retained for a fixed subset, not every episode. Registered checkpoints and optimizer/RNG states are retained transactionally and atomically exported. The benchmark matches original and compact observation modes **bitwise** for successive model/Adam/query-gradient results. It does not change scientific operations or precision.

All inference uses10000 whole-seed paired bootstrap resamples with the original fixed RNG20260925. Effects are formed within seed before aggregation; repeated arms/doses retain the same resampling indices. Missing outcomes and missing matched bins are not zero. Intervals are pointwise, not simultaneous; no equivalence margin or multiplicity correction was registered. Median intervals and the count of beneficial paired seed changes accompany mean effects. A zero-spanning interval is inconclusive, not evidence of preservation, equivalence or a plateau.

'''
 for i,q in enumerate(QUESTIONS,1):
  text+=f'## {i}. {q}\n\n'
  if i==1:
   text+='The first registered primary contrast is Transfer10000−Transfer960, separately for independent P1 and P2. Negative effects favor additional training. Read C CE and familiar CE alongside the gap: worse familiar acquisition must not be mistaken for transfer.\n\n';x=primary[primary.contrast.eq('Transfer:10000-960')];text+=table(['Condition','Paired gap effect [95% CI]','Median effect [95% CI]','Improved seeds','n','Inference'],[[r.condition,ci(r),f'{fmt(r["median"])} [{fmt(r.median_ci_low)}, {fmt(r.median_ci_high)}]',int(r.n_negative),sample_n(r),interpret_interval(r.ci_low,r.ci_high)] for _,r in x.iterrows()]);rows=[]
   for keys,g in end[end.cue.eq('C')&end.dose.isin([960,10000])].groupby(['condition','arm','dose']):rows.append([*keys,fmt(g.ce.mean()),fmt(g.familiar_ce.mean()),fmt(g.gap.mean()),fmt(g.gap.median()),int(g.strong_familiar.eq(1).sum()),int(g.strict_all.eq(1).sum())])
   text+='\n'+table(['Family','Arm','FOMAML','C CE','Familiar CE','Gap mean','Gap median','Strong A/B /20','Strict C /20'],rows)
  elif i==2:
   text+='Figure3 shows every registered dose, including reversals. No power law was fitted. The terminal adjacent contrast10000−7680 is secondary; an uncertain terminal change does not establish saturation. The table gives paired gap changes and per1000-episode slopes.\n\n';x=effects[effects.metric.eq('gap')&effects.cue.eq('C')&effects['update'].eq(330)&effects.contrast.isin(['Transfer:adjacent10000-7680','Seen:adjacent10000-7680'])];text+=table(['Family','Contrast','Paired terminal change [95% CI]','Mean slope /1000 episodes','Interpretation'],[[r.condition,r.contrast,ci(r),fmt(r['mean']/2.32),interpret_interval(r.ci_low,r.ci_high)] for _,r in x.iterrows()]);rows=[]
   for (c,a,d),g in end[end.cue.eq('C')].groupby(['condition','arm','dose']):rows.append([c,a,d,fmt(g.ce.mean()),fmt(g.gap.mean()),fmt(g.gap.median()),int(g.strict_all.eq(1).sum())])
   text+='\nComplete empirical endpoint dose response:\n\n'+table(['Family','Arm','FOMAML','Mean C CE','Mean C gap','Median C gap','Strict C /20'],rows)
   qq=read('meta_query_summary');qq=qq[qq.dose.isin([960,10000])&qq.window.eq('last240')&qq.metric.eq('query_ce')];text+='\nSaved outer-query CE over each dose’s final240 episodes (tasks averaged within seed):\n\n'+table(['Family','Arm','Dose','Task kind','Query CE [95% CI]','Seed n'],[[r.condition,r.arm,r.dose,r.kind,ci(r),sample_n(r)] for _,r in qq.iterrows()]);text+='\nP1 conflict queries retain their intended ambiguity. These training losses use fixed windows and actual task counts; they are diagnostics, not target-generalization criteria or checkpoint-selection scores.\n'
  elif i==3:
   text+='The second registered primary contrast is Transfer−Seen at10000, separately for independent P1/P2. This compares the cumulative query objectives under the same optimization algorithm. The arms resume their own already-divergent960 states, sharing the original Reptile parent atFOMAML0. It is not a fresh randomization at960. The secondary scale interaction subtracts the960 arm difference. Historical comparisons additionally change optimizer and amount of experience.\n\n';x=primary[primary.contrast.eq('Transfer-Seen')];text+=table(['Condition','Paired gap effect [95% CI]','Median paired effect','Improved seeds','n'],[[r.condition,ci(r),fmt(r['median']),int(r.n_negative),sample_n(r)] for _,r in x.iterrows()]);x=effects[effects.dose.eq(10000)&effects['update'].eq(330)&effects.cue.eq('C')&effects.metric.isin(['ce','familiar_ce'])&effects.contrast.isin(['Transfer-Seen','Transfer:10000-960','Seen:10000-960'])];text+='\nUnderlying loss effects (all families, n and signs retained):\n\n'+table(['Family','Contrast','Metric','Paired effect [95% CI]','n'],[[r.condition,r.contrast,r.metric,ci(r),sample_n(r)] for _,r in x.iterrows()])
   xx=effects[effects.dose.eq(10000)&effects['update'].eq(330)&effects.cue.eq('C')&effects.metric.eq('gap')&effects.contrast.eq('query_arm_scale_interaction:10000-960')];text+='\nSecondary difference-in-differences of additional scaling, (Transfer10000−Transfer960)−(Seen10000−Seen960):\n\n'+table(['Family','Paired interaction [95% CI]','n'],[[r.condition,ci(r),sample_n(r)] for _,r in xx.iterrows()])
  elif i==4:
   text+='Figure2 retains all natural cue trajectories. Full-strength C is computationally distinct from NoContext and was never trained on this target. Strict descriptive targets remain CE≤0.1, perfect7/7 accuracy, signed gap≤0.1, and their conjunction. Original strong acquisition requires each familiar cue CE≤0.5 and accuracy≥6/7; original C success uses the same cue-wise rule. None is used for model selection.\n\n';x=summary[summary.dose.eq(10000)&summary['update'].eq(330)&summary.metric.eq('ce')];text+=table(['Family','Arm','Cue','CE [95% CI]','n'],[[r.condition,r.arm,r.cue,ci(r),sample_n(r)] for _,r in x.iterrows()]);x=post[post.dose.eq(10000)&post.metric.eq('post_acquisition_gap')&post.contrast.isin(['Transfer:10000-960','Transfer-Seen'])];text+='\nPost-acquisition disadvantage includes every state from first strong A/B acquisition through330, including later worsening. Never-acquired networks remain missing/censored.\n\n'+table(['Family','Contrast','Paired post-acquisition gap change [95% CI]','Acquired paired n'],[[r.condition,r.contrast,ci(r),sample_n(r)] for _,r in x.iterrows()]);text+='\nNever-acquired counts:\n\n'+table(['Family','Arm','Dose','Never acquired /20'],[[*keys,int(g.first_familiar.isna().sum())] for keys,g in acq.groupby(['condition','arm','dose'])])
  elif i==5:
   text+='The primary table reports mean and median paired effects and the count of negative changes. The negative-change count includes any negative sign, without a minimum practical magnitude. Large mean improvement alongside small median change indicates heterogeneous tail effects; it does not mean every seed improves. All seed observations remain in primary_seed_effects.csv and Figure5. Position changes below retain all seven transitions; they are not a selected difficult-position analysis.\n\n';x=position[position.dose.eq(10000)&position['update'].eq(330)&position.condition.isin(['P1-I','P2-I'])&position.metric.eq('ce')&position.contrast.isin(['Transfer:10000-960','Transfer-Seen'])];text+=table(['Family','Contrast','Position','Paired C CE effect [95% CI]','n'],[[r.condition,r.contrast,r.position,ci(r),sample_n(r)] for _,r in x.iterrows()]);err=read('C_error_concentration');rows=[]
   for key,g in err[err.dose.eq(10000)&err['update'].eq(330)].groupby(['condition','arm']):rows.append([*key,fmt(g.C_incorrect.mean()),fmt(g.top1_CE_share.mean()),fmt(g.top2_CE_share.mean()),int(g.zero_loss.sum())])
   text+='\nResidual error concentration (loss shares are missing if total CE is exactly zero):\n\n'+table(['Family','Arm','Mean wrong positions','Mean largest-position loss share','Mean largest-two share','Exact-zero losses /20'],rows)
  elif i==6:
   text+='Use continuous signed own/cross effects, first20 updates, matched-acquisition bins, and restored-Adam probes. Geometry is not separate-learning evidence. First A block is frozen states0→10; first B is10→20. Their full start/end A/B losses and each individual early update are retained. Small effects when both starting losses≤0.05 are floor-limited, not evidence of shared learning.\n\n';ms=read('block_intervals');x=ms[ms.dose.eq(10000)&ms.end.isin([10,20])&ms.metric.isin(['specificity','practiced_gain','unpracticed_gain'])];text+=table(['Family','Arm','Practice','Metric','First-block mean [95% CI]','n'],[[r.condition,r.arm,r.practiced,r.metric,ci(r),sample_n(r)] for _,r in x.iterrows()]);x=early[early.dose.eq(10000)&early.metric.isin(['early20_specificity','early20_own','early20_cross'])&early.contrast.isin(['Transfer:10000-960','Transfer-Seen'])];text+='\nCo-primary continuous first20 contrasts:\n\n'+table(['Family','Contrast','Metric','Paired effect [95% CI]','n'],[[r.condition,r.contrast,r.metric,ci(r),sample_n(r)] for _,r in x.iterrows()]);x=matched[matched.dose.eq(10000)&matched.stratum.eq('all')&matched.metric.eq('specificity')&matched.contrast.isin(['Transfer:10000-960','Transfer-Seen'])];text+='\nFixed-bin first eligible block, either practice direction; supplemental first-A/first-B remain separate. A missing paired bin cannot establish zero specificity or preservation. Matching is a conditional description, not randomized mediation.\n\n'+table(['Family','Contrast','Familiar CE range','Paired specificity change [95% CI]','Eligible paired n'],[[r.condition,r.contrast,r.performance_range,ci(r),sample_n(r)] for _,r in x.iterrows()]);x=joint[joint.dose.eq(10000)];text+='\nOriginal endpoint criterion, unchanged at170/330:\n\n'+table(['Family','Arm','Target state','Original functional /20','Original joint /20','Available n'],[[*key,int(g.functional_split.eq(1).sum()),int(g.original_joint.eq(1).sum()),int(g.available.sum())] for key,g in x.groupby(['condition','arm','update'])]);text+='\nThe exact criterion still requires both D_A,D_B>0, meanD>0.001, mean own gain>0.0001, minimum own gain>−0.000001, plus balanced-cycle mean adjusted switch cost>0.05 and at least half cycles>0.05. It is reported even when late loss floors make interpretation limited. No threshold was changed.\n'
  elif i==7:
   text+='The same corrected-v2 individually unseen12-task library is reserved through10000. Each initialization receives20 diagnostic updates per task with stable identifiable assignments for both priors. At position3, identical token prefixes require different targets under different contexts. This tests functional capability, not P1 training ambiguity. All tasks are averaged within each seed before inference; they are not independent networks.\n\n';x=held[held.metric.isin(['ce','conflict_ce','conflict_accuracy','appropriate_context_change'])];text+=table(['Family','Arm','Dose','Diagnostic','Mean [95% CI]','Seed n'],[[r.condition,r.arm,r.dose,r.metric,ci(r),sample_n(r)] for _,r in x.iterrows()]);x=read('heldout_paired_intervals');x=x[x.dose.eq(10000)&x.contrast.isin(['Transfer:10000-960','Transfer-Seen'])&x.metric.isin(['conflict_accuracy','appropriate_context_change','conflict_ce'])];text+='\nChanges in conflict capability (secondary):\n\n'+table(['Family','Contrast','Metric','Paired effect [95% CI]','n'],[[r.condition,r.contrast,r.metric,ci(r),sample_n(r)] for _,r in x.iterrows()]);text+='\nCorrect-context results are reused. Previously missing wrong-context inputs use the same frozen task state, tokens and targets. Initial correct CE is reconstructed from final CE plus all four signed own/cross block gains; no matched correct branch is replayed. Wrong-context probabilities/logits, correct/wrong learning gains and conflict-position CE are saved separately. Context-sensitive success on these tasks complements target learning measures but does not establish persistent original A/B skills.\n'
  elif i==8:
   text+='''For constant cue input and zero recurrent initial state,

`S_t = Σ(k=0…t−1) W_h^k`, `K_t = W_o S_t`.

`z_A=z_0+u+v`, `z_B=z_0+u−v`, `z_C=z_0+K_t w_C`,

where `u=K_t(w_A+w_B)/2` and `v=K_t(w_A−w_B)/2`. Full-strength Mixed-both has `z_Mixed-both=z_0+2u`. Therefore

`z_C−z_Mixed-average = K_t[w_C−(w_A+w_B)/2]`.

The decomposition is numerically checked at every position/state with independent double operators and the inherited bounded native arithmetic. Centered logits remove class-common offsets. Mixed-average has the convex CE bound relative to familiar A/B; neither C nor Mixed-both inherits it. Softmax probabilities need not equal an arithmetic average of familiar probabilities.

Changes in baseline CE and C-minus-baseline CE distinguish observed baseline and cue effects. Recurrent sensitivity and weight geometry describe the learned computation, but across-model correlations do not establish causal mediation. Smaller norms alone are not successful transfer. No new context-strength sweep, optimized hidden state, rescue search or architectural intervention was run.

''';x=mechanism[mechanism.dose.eq(10000)&mechanism['update'].eq(330)&mechanism.metric.isin(['baseline_ce','C_minus_baseline_CE','C_minus_avg_centered_norm','v_norm','C_norm','C_common_cosine','K_centered_norm','recurrent_spectral_norm'])&mechanism.contrast.isin(['Transfer:10000-960','Transfer-Seen'])];text+=table(['Family','Contrast','Mechanism','Paired change [95% CI]','n'],[[r.condition,r.contrast,r.metric,ci(r),sample_n(r)] for _,r in x.iterrows()])
  elif i==9:
   text+='Figure5 shows all networks against continuous C gap and early learning specificity. Filled markers indicate strong familiar acquisition and positive own gain/signed specificity in both first ten-update blocks, outside the inherited floor flag. These signs form a descriptive coexistence measure; they do not replace the original gate or establish reliable equivalence of specialization between arms. A missing first20 flag can reflect a floor-limited B block after rapid transfer from A; it is not proof of context-insensitive learning.\n\n';x=joint[joint['update'].eq(330)&joint.dose.isin([0,960,10000])];text+=table(['Family','Arm','Dose','Strong familiar /20','C success /20','Mixed-both /20','Early20 bidirectional /20','Descriptive joint /20','Strict-C descriptive joint /20','Original functional /20'],[[*key,int(g.strong_familiar.eq(1).sum()),int(g.C_success.eq(1).sum()),int(g.Mixed_both_success.eq(1).sum()),int(g.early20_bidirectional.eq(1).sum()),int(g.early20_coexistence.eq(1).sum()),int(g.early20_strict_coexistence.eq(1).sum()),int(g.functional_split.eq(1).sum())] for key,g in x.groupby(['condition','arm','dose'])]);text+='\nObserved coexistence, persistent separate skill learning and resemblance to human behavior are distinct claims. C was experienced on earlier sequences; the held-out association is C with this target sequence, not an entirely new sensory cue. Human response times and motor behavior were not fitted. Missing bins, floor-limited probes, heterogeneous seeds and any residual conflict deficits limit general claims. The experiment can establish component performance under the fixed task ecology; it cannot establish that capacity and strong cues inevitably create separate skills.\n'
 text+='''
## Retained table-only analysis correction

The first analysis pass exported42 base tables, then stopped because Boolean position-accuracy values cannot be subtracted directly. A regression test required signed ±1 differences. The corrected extension converts indicators to numeric0/1 while preserving missing values. All42 base-table hashes remained unchanged, and only the missing extensions were resumed. No model, checkpoint, native evaluation branch, threshold, primary contrast or training budget changed. The original failure, RED/GREEN logs and recovery receipts remain in audit/.

## Evidence, figures and reproducibility

- [Figure1: actual A10/B10 curves](figures/figure_1.png), [first100](figures/figure_1_early.png), [first20](figures/figure_1_first20.png).
- [Figure2: six natural cues](figures/figure_2.png), [early acquisition](figures/figure_2_early.png).
- [Figure3: empirical scaling](figures/figure_3.png), [centered conditions](figures/figure_3_centered.png), [paired arm differences](figures/figure_3_paired_effects.png).
- [Figure4: first A/B functional specificity](figures/figure_4.png), [own gain](figures/figure_4_own.png), [cross gain](figures/figure_4_cross.png), [matched acquisition](figures/figure_4_matched.png), [early switches](figures/figure_4_switches.png).
- [Figure5: joint transfer and learning](figures/figure_5.png), [descriptive and original gate rates](figures/figure_5_joint_rates.png).
- [Figure6: reserved conflict capability](figures/figure_6.png), [wrong-context comparisons](figures/figure_6_context_effects.png).

Every figure has SVG, exact seed values and interval tables. Supplemental doses, restored-Adam states, full block time courses and mechanism plots remain in figures/. CLAIM_LEDGER.csv maps claims to exact tables and limitations. SOURCE_REUSE_AUDIT.md, REPRODUCIBILITY.md and RESEARCHER_HANDOFF.md explain source integrity, algorithms and safe table-only regeneration. The append-only scratchpad and RED/GREEN logs retain failed tests and implementation choices. Full states and manifests are source-linked; reused data are never new replications. No predecessor sealed data were overwritten.
'''
 return text.replace('## What was executed',scientific_takeaways()+'## What was executed',1).replace('A missing first20 flag can reflect a floor-limited B block after rapid transfer from A; it is not proof of context-insensitive learning.','The first20 floor flag is zero for all final-dose seeds, so it does not explain their missing descriptive flags. A missing flag alone is not proof of context-insensitive learning.')

def prose_spacing(text):
 import re
 text=re.sub(r'(?<![A-Za-z0-9_])(\d[\d,]*(?:\.\d+)?)(?=[A-Za-z]{3})',r'\1 ',text)
 text=re.sub(r'\b(FOMAML|Reptile|Adam|episode|episodes|update|updates|state|states|target|through|from|to|at|after|before|for|all|All|each|every|first|last|norm|block|position|Figure)(?=\d)',r'\1 ',text)
 return text

def main():
 (OUT/'FINAL_HELDOUT_CONTEXT_META_SCALING_10K_REPORT.md').write_text(prose_spacing(render()));append('## Scientific report\nNinePRDquestionsansweredfromallseeds/doses, includingfourregisteredindependentprimarycontrasts, mean/median/seedheterogeneity,terminaladjacentchangeswithoutplateau/equivalenceclaims,continuousfirst20/matchedeffects/floors,unchangedgate, reservedconflictsandLinearcomputation. No checkpointselection orcontextstrengthsweep.')
if __name__=='__main__':main()
