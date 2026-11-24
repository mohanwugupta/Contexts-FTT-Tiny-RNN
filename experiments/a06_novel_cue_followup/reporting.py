"""Interpretation from fixed saved tables; no fitting or scientific execution."""
from .protocol import *
def counts(f):
 return f.groupby('condition',as_index=False).agg(n=('seed','size'),familiar=('familiar','sum'),C_useful=('C_useful','sum'),screen_joint=('screen_joint','sum'),original_early_joint=('original_early_joint_not_interference_dominated','sum'))
def main(destination=None):
 dest=Path(destination) if destination else OUT
 learning=pd.read_csv(dest/'learning_measures.csv');dist=pd.read_csv(dest/'cue_distribution_by_seed.csv');original=pd.read_csv(dest/'original_cue_results.csv');novel=pd.read_csv(dest/'unfamiliar_cue_results.csv');stats=pd.read_csv(dest/'paired_summary.csv');pairs=pd.read_csv(dest/'paired_seed_results.csv');cohort=counts(learning);csv(dest/'joint_counts.csv',cohort)
 diffs=[]
 for second in ['A00','A00-static']:
  a=novel[novel.condition.eq('A06')&novel.kind.ne('original')];b=novel[novel.condition.eq(second)&novel.kind.ne('original')];z=a.merge(b,on=['seed','update','cue'],suffixes=('_first','_second'),validate='one_to_one')
  for row in z.itertuples():diffs.append(dict(seed=row.seed,update=row.update,cue=row.cue,contrast='A06-minus-'+second,ce_difference=row.ce_first-row.ce_second,correct_count_difference=row.correct_count_first-row.correct_count_second,accuracy_difference=row.accuracy_first-row.accuracy_second,useful_difference=int(row.useful_first)-int(row.useful_second)))
 diffs=pd.DataFrame(diffs);csv(dest/'individual_cue_paired_changes.csv',diffs)
 sensitivity=[]
 for key,g in pairs[pairs.seed.ge(2)].groupby(['contrast','metric','cue','probe_group','update'],dropna=False):
  r=interval(g.set_index('seed').effect.reindex([2,3,4,5]).to_numpy(float),[2,3,4,5]);sensitivity.append(dict(zip(['contrast','metric','cue','probe_group','update'],key),**r,cohort='new exploratory seeds2–5; not independent confirmation'))
 csv(dest/'new_seed_sensitivity.csv',sensitivity)
 endpoint=[]
 for r in learning.itertuples():endpoint.append(dict(seed=r.seed,condition=r.condition,A_B_acquired=r.familiar,C_CE=r.C_ce,C_correct=r.C_correct,C_useful=r.C_useful,NoContext_CE=r.NoContext_ce,NoContext_correct=r.NoContext_correct,NoContext_useful=r.NoContext_useful,Mixed_useful=r.Mixed_useful,D_A=r.D_A,D_B=r.D_B,E_normalized=r.E_normalized,original_early=r.original_early_phenotype,original_early_joint=r.original_early_joint_not_interference_dominated,screen_joint=r.screen_joint))
 endpoint=pd.DataFrame(endpoint);csv(dest/'report_endpoints.csv',endpoint);d330=dist[dist['update'].eq(330)].copy()
 def effect(contrast,metric,cue_=None,group=None):
  g=stats[stats.contrast.eq(contrast)&stats.metric.eq(metric)]
  if cue_ is not None:g=g[g.cue.eq(cue_)]
  if group is not None:g=g[g.probe_group.eq(group)&g['update'].eq(330)]
  r=g.iloc[0];return f'{r["mean"]:+.6g} [{r.ci_low:+.6g}, {r.ci_high:+.6g}], median {r["median"]:+.6g}; improves/worsens/unchanged (signed+/−) {int(r.n_positive)}/{int(r.n_negative)}/{int(r.n_unchanged)}'
 primary=stats[((stats.cue.isin(['C','NoContext']))&stats.metric.isin(['ce','accuracy']))|((stats.probe_group=='novel_within_seed_mean')&stats['update'].eq(330)&stats.metric.isin(['mean_ce','mean_accuracy','fraction_useful']))];csv(dest/'report_primary_paired_effects.csv',primary)
 report='''# A06 output initialization and unfamiliar-context transfer

**Smaller initial output weights improve generalization on average, including unfamiliar-cue correctness beyond the frozen readout control. They do not produce reliable all-cue or joint success.** This is a six-seed exploratory follow-up to a selected pilot candidate, not an independent preregistered confirmation.

## 1. A/B learning, C and NoContext

All12 A00/A06 networks acquire A and B strongly and retain useful Mixed-average performance at330. All6 per condition satisfy the unchanged original early splitting flag, with positive own gains and positive familiar net gains in both first blocks. Thus the signed specificity is not driven primarily by damaging the other familiar context. However, raw D_A/D_B are smaller under A06 in all6 seeds; initial output strength also changes initial error/headroom. Normalized balanced switching has no consistent paired change. The stricter screen-preservation gate remains a distinct definition.

'''+cohort.to_markdown(index=False)+'''

Original-early joint here means the inherited early flag plus useful C and Mixed, in the same network, with positive familiar net gains. Original early+C/Mixed co-occurrence is4/6 A06 versus3/6 A00; the old QA/QB/normalized-E screen-joint is3/6 versus2/6. Neither is the original late counterfactual functional gate, which remains **unassessed** for new A06 because no extra diagnostic optimizer branches are authorized. Do not relabel an early routing gate as proof of independent human memories. NoContext is diagnostic, not another gate.

'''+endpoint.to_markdown(index=False,floatfmt='.6g')+'''

C loss improves in5/6 seeds but worsens slightly in seed4 (both remain7/7 with tiny loss); C accuracy improves in2, worsens in1 and is unchanged in3. C useful performance is4/6 versus3/6. NoContext CE improves in6/6, accuracy in3/6 with3 ties; only1/6 A06 networks meets its CE≤.5/≥6/7 useful criterion versus0/6 A00. Benefits are graded rather than universal. Seed1 retains the previously documented C loss-versus-accuracy tradeoff. No seeds or checkpoints were selected using these outcomes.

## Paired uncertainty (10,000 whole-seed resamples)

'''+primary[['contrast','cue','probe_group','metric','mean','median','ci_low','ci_high','n','n_positive','n_negative','n_unchanged']].to_markdown(index=False,floatfmt='.6g')+'''

Effects are first minus second. Negative CE favors A06; positive accuracy/useful fraction favors A06. Counts label arithmetic signs, not significance. Six paired trained seeds are the unit; every cue/position/checkpoint is a repeated observation. For unfamiliar cues, first average the fixed12 within each network before pairing seeds. The same six-seed resamples are used across metrics. Median intervals and all timepoints are in paired_summary.csv. New-seed2–5 sensitivity is reported separately in new_seed_sensitivity.csv and is still exploratory. Pointwise intervals are descriptive, not multiplicity-adjusted population confirmation.

## 2. Is original C uniquely difficult?

**Generally no.** Original C is a cue learned during ordinary background tasks, though the target sequence was held out. It is usually easier than the unfamiliar bank. In the two remaining C-failing A06 networks (seeds1/2), C CE is close to their unfamiliar-cue median; they have broad input sensitivity rather than one isolated catastrophic C direction. Seed1 does have6/12 unfamiliar cues with higher accuracy than C, but only1/12 meets the full usefulness rule; seed2 has2/12 with higher accuracy and3/12 useful. Some alternative directions work; that is not broad robustness.

'''+d330[['seed','condition','C_ce','median_ce','C_minus_median_novel_ce','C_accuracy','median_accuracy','best_accuracy','worst_accuracy','fraction_useful','fraction_novel_accuracy_higher_than_C','NoContext_accuracy','fraction_novel_accuracy_below_NoContext']].to_markdown(index=False,floatfmt='.6g')+'''

A06 mean unfamiliar accuracy rises in all6 seeds. Relative to A00, its six-seed mean improvement is+0.0892857 (8.93percentage points),95% interval[0.0257937,0.164683]; against A00-static it is+0.101190[0.0357143,0.176587]. The mean useful-cue fraction is24/72=33.3% for A06 versus10/72=13.9% A00 and12/72=16.7% static. These72 entries are6×12 repeated cue tests, not72 independent networks. No network succeeds on all12. Every cue remains in unfamiliar_cue_results.csv and individual_cue_paired_changes.csv, including unchanged or worsened predictions hidden by a positive within-seed mean.

The four unit-norm C-arc inputs yield12/24 useful A06 cases; the eight random sphere inputs yield12/48. Higher cosine similarity to C is associated with lower CE in all6 models (per-seed Spearman values in cue_distribution_by_seed.csv), but close cues do not guarantee success. One fixed great-circle arc is not a complete neighborhood sample, and8 random directions do not characterize the full sphere. NoContext outperforms some unfamiliar cues in5/6 A06 networks; seed4 remains poor there. Its zero input norm differs intentionally from the unit-norm bank and is not evidence of equal-magnitude invariance.

## 3. One next intervention

**Preregister broader cue-code coverage during ordinary pretraining while keeping A06 and target A/B practice fixed**, for example an episodewise orthogonal cue basis. This directly tests the remaining dependence on particular learned input directions. It is a hypothesis to test, not a demonstrated rescue: the earlier short B02 screen had inadequate background competence, so a new study would need an adequate matched ordinary-learning control. Do not choose favorable cues or add transfer/splitting rewards. No such training or rescue was launched here.

## Algorithms, provenance and limits

Same native H8 identity recurrence, two hidden biases, four token inputs/three cue inputs/four logits, zero hidden reset, target[0,1,2,0,3,1,2]→[1,2,0,3,1,2,3]. A06 changes W_out×.25 once at initialization, with unchanged b_out and all other tensors/policy. Native Adam.005, CE+.2 detached execution objective, joint norm1 clipping and full ordinary updates continue unchanged; no target norm projection. Target C column is bitwise fixed at every new update. Static z=.25(z_A00−b_A00_checkpoint)+b_A00_checkpoint uses the matching bias at every state, never rescales the saved A06 outputs, and has no separate practice history.

The12 vectors were sealed before scientific execution using independent PCG64 seed2026112306:8 normalized Gaussian directions and a predeclared perpendicular great-circle arc around C at15/30/45/60deg. All have unit norm (native float32 execution tolerance retained), same for every model. They are combinations of existing channels that were not discrete prior cues; this is contextual-input robustness, not a newly learned sensory context or an optimized C retrieval cue. The physical Mixed-average input is evaluated directly. Frozen evaluation leaves model, policy, optimizer, RNG, gradients, modes and cursor unchanged. Native float32 logits retain inherited float64 scoring. Older strict precision exceptions remain exceptions, not repaired passes.

Exactly1320 new optimizer calls:4×330 A06 targets, no A00 training, meta-training, auxiliary optimizer fixture or replay. All6 A00 histories reused, old A06 pilots reused. All331 states are available for new runs; the old A06 pilots saved only10 named states. Their642 missing states/3210 original-probe cells remain explicit, with no interpolation/reconstruction. All8 unfamiliar-probe checkpoints are available in every network, so the primary frozen comparison is complete.

The PRD frozen-cell arithmetic double-counts conditions:6×2×8×17=1632 unique ExperimentB cells (1152 new unfamiliar sequences+480 reused original cells), not3264. ExperimentA separately required6620 new dense original-cue predictions across4×331×5; total new frozen sequences7772. Static array transforms are not model forwards. This accounting is fixed in the before-run scratchpad and implementation audit; no duplicate probes were manufactured or training budget expanded.

All sources, checkpoint identities, full new continuation state, cue bank, failures/missingness, original criteria, bootstrap procedures and table-only regeneration are documented in IMPLEMENTATION_AUDIT.md, REPRODUCIBILITY.md and the final manifest. This result supports a broader average benefit of output initialization, while leaving substantial directional and joint-phenotype failures unresolved.
'''
 (dest/'REPORT.md').write_text(report)
 handoff='''# Researcher handoff

A06 gives a real accuracy benefit beyond the checkpoint-specific static quarter-readout control, but does not solve reliable C/joint or arbitrary-cue generalization. All6 A00 and all6 A06 acquire A/B/Mixed and pass the inherited early splitting flag. C useful:3/6→4/6; original early joint:3/6→4/6; stricter screen joint:2/6→3/6. The original late functional gate is unassessed for new A06, not a zero or a newly solved gate. Raw DA/DB shrink in every seed; normalized switching remains mixed. NoContext CE improves6/6 and accuracy3/6, yet useful performance is only1/6.

Across the fixed12 unit-norm unfamiliar cues, summarize within each trained seed: mean accuracy rises in6/6. Paired A06−A00 +8.93points CI[2.58,16.47], versus static +10.12points[3.57,17.66]. Useful cue tests:24/72 A06,10/72 A00,12/72 static; cues are repeated tests, not independent networks. No network passes all12. Original C is generally easier than the unfamiliar median; residual C failures resemble wider directional fragility. Higher C similarity often helps, but a single arc and8 random vectors do not identify a universal neighborhood. NoContext sometimes wins but has norm0 rather than1.

Teach-back: ordinary target training is unchanged; only initial readout W is quartered, bias preserved. Later learning can change all ordinary trainable entries. Frozen static scaling cannot explain every extra correct prediction, but this comparison does not isolate an internal representation mechanism. Context drive, recurrence/readout and optimization path may all contribute. No target C/new cue practice, cue fitting or new reward.

Next single intervention to preregister: broaden ordinary background cue-code coverage, holding A06 and final A/B-only target protocol fixed. Earlier B02 was budget-inconclusive; do not claim this will work. Require an adequate matched prior-learning control. It was not launched here.

Read report_endpoints.csv, learning_measures.csv, individual_cue_paired_changes.csv and cue_distribution_by_seed.csv before drawing conclusions. Primary unit6 trained paired seeds, with0/1 reused pilots and2–5 new exploratory cases, not independent confirmation. Exactly1320 optimizer steps,0 meta/auxiliary; all new states saved.642 old-pilot intermediate states missing, retained. All frozen8-checkpoint comparisons complete. Do not retrain for reproduction; use saved tables/arrays and hashes.
''';(dest/'RESEARCHER_HANDOFF.md').write_text(handoff)
 put(dest/'audit/report_complete.json',dict(passed=True,report_sha=sha(OUT/'REPORT.md'),joint_counts=cohort.to_dict('records'),new_optimizer_steps=0))
