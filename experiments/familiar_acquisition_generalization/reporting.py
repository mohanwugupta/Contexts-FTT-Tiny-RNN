"""Report and auxiliary summaries from saved audit tables only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .audit import DEFAULT,PRIOR,ROOT,CHANCE,bootstrap


def verify_seed_arithmetic(q):
 for probe in ['Mixed','NoContext','NovelMean']:
  if not np.allclose(q['gap_'+probe],q['ce_'+probe]-q.familiar_ce,atol=1e-12,rtol=0):raise ValueError('Absolute CE and relative gap conflated')
 return True

def md(df):
 def fmt(x):
  if isinstance(x,(float,np.floating)):return 'NA' if not np.isfinite(x) else f'{x:.4f}'
  return str(x)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))

def write(out=DEFAULT):
 out=Path(out);q=pd.read_csv(out/'tables/seed_level.csv',float_precision='round_trip');d=pd.read_csv(out/'tables/paired_changes.csv',float_precision='round_trip');s=pd.read_csv(out/'tables/bootstrap_summary.csv');a=pd.read_csv(out/'tables/descriptive_associations.csv');counts=pd.read_csv(out/'tables/operational_counts.csv');f=pd.read_csv(out/'tables/transition_level.csv');verify_seed_arithmetic(q)
 cue=[];position=[]
 for (arm,c),g in f.groupby(['arm','cue']):
  seed=g.groupby('seed',as_index=False).agg(ce=('raw_ce','mean'),accuracy=('correct','mean'))
  for m in ['ce','accuracy']:cue.append(dict(arm=arm,cue=c,metric=m,**bootstrap(seed,m)))
 for (arm,c,t),g in f.groupby(['arm','cue','transition']):
  for m in ['raw_ce','correct']:position.append(dict(arm=arm,cue=c,position=t,metric=m,**bootstrap(g,m)))
 pd.DataFrame(cue).to_csv(out/'tables/cue_summary.csv',index=False);pd.DataFrame(position).to_csv(out/'tables/position_summary.csv',index=False)
 conditional=[]
 for arm,g in q.groupby('arm'):
  for label,sub in [('all',g),('strong_familiar',g[g.strong_familiar]),('perfect_familiar',g[g.perfect_familiar])]:
   conditional.append(dict(arm=arm,subset=label,n=len(sub),familiar_ce=sub.familiar_ce.mean(),mixed_ce=sub.ce_Mixed.mean(),mixed_gap=sub.gap_Mixed.mean(),mixed_accuracy=sub.accuracy_Mixed.mean(),n_retained_poor=int(sub.strong_retained_poor.sum()),n_retained_success=int(sub.strong_retained_success.sum())))
 pd.DataFrame(conditional).to_csv(out/'tables/familiar_subgroups_descriptive.csv',index=False)
 examples=q[q.strong_retained_poor|q.strong_retained_success];examples.to_csv(out/'tables/identified_examples.csv',index=False)
 no_loss=[]
 for arm,g in d.groupby('arm'):
  subset=g[(g.delta_ce_Mixed<0)&(g.delta_hidden_separation>=0)&(g.delta_logit_separation>=0)];no_loss.append(dict(arm=arm,n_improved_mixed_without_either_separation_decrease=len(subset),seeds='|'.join(map(str,subset.seed.tolist()))))
 pd.DataFrame(no_loss).to_csv(out/'tables/improvement_without_separation_loss.csv',index=False)
 means=q.groupby('arm')[['familiar_ce','ce_Mixed','gap_Mixed','familiar_accuracy','accuracy_Mixed','hidden_separation','logit_separation']].mean().reindex(['Original','B','I','S']).reset_index()
 main=s[(s.type=='paired_change')&s.metric.isin(['delta_familiar_ce','delta_ce_Mixed','delta_gap_Mixed','delta_hidden_separation','delta_logit_separation'])][['arm','metric','mean','ci_low','ci_high']]
 probes=q.groupby('arm')[['ce_NoContext','gap_NoContext','accuracy_NoContext','ce_NovelMean','gap_NovelMean','accuracy_NovelMean']].mean().reindex(['Original','B','I','S']).reset_index()
 chance=s[(s.type=='endpoint')&(s.metric=='minus_uniform_Mixed')][['arm','mean','ci_low','ci_high']]
 perf=q[q.perfect_familiar].groupby('arm')[['strong_retained_poor','strong_retained_success']].sum().reset_index()
 examplecols=['seed','arm','ce_A','ce_B','accuracy_A','accuracy_B','ce_Mixed','accuracy_Mixed','gap_Mixed','hidden_retention','logit_retention']
 text=f'''# Familiar acquisition versus generalization in High-scale SRNs

Poor Mixed prediction is not confined to networks that have failed to learn familiar A/B sequences. Additional blocked and whole-sequence interleaved training substantially improves familiar learning and absolute Mixed prediction, but leaves large familiar-relative Mixed gaps. Individual continued networks can predict A/B perfectly, retain most of their original A/B separation, and still predict Mixed worse than uniform. There is also a network with good familiar performance, substantial retained separation and useful Mixed prediction. The evidence supports a persistent, heterogeneous generalization problem; it does not support treating every original update170 network as adequately trained, or treating improved Mixed loss as an eliminated generalization gap.

## Data and fixed analysis

All20 original canonical High_monotonic seeds at update170 and all60 matched B/I/S endpoints at update330 are included. Each endpoint is paired to its own original checkpoint; the original is one network, not three independent baselines. B continues blocked A/B training; I alternates whole familiar sequences; S switches familiar context within sequences. These are already trained endpoints, not new fits. Their historical optimizer was reset identically across arms; these are continuations from the same weights rather than restored optimizer moments.

The audit uses saved natural evaluations for8 physical cues and7 teacher-forced positions:4,480 transition rows. A/B hidden separation is the mean of7 matched-position Euclidean distances. Output separation is the corresponding distance between recorded centered four-class logits; class-common logit shifts cancel. Differences from the predecessor double-head separation are at most3.33e-7, a numerical convention difference rather than changed geometry. Accuracy uses argmax of the saved four logits; there are no maximum-logit ties. All saved CEs reproduce exactly from those logits. No checkpoints were loaded for inference, changed or retrained. All80 checkpoint file hashes were verified before and after extraction.

LF=(LA+LB)/2; Gprobe=Lprobe−LF. Absolute probe CE and its gap are separate columns throughout. Uniform prediction has CE ln4={CHANCE:.6f} nats (display1.386). NovelMean equally weights allfour registered Novel30_A/B and Novel60_A/B cues within each model. Every individual cue, per-cue accuracy and allseven position CEs remain in `tables/seed_level.csv`; transition accuracy/logits/states are in `tables/transition_level.csv`. Novel30/60 family means are also retained. No human, Low, C0–C3 or other architecture is pooled.

For descriptive identification only, “strong familiar” means CE≤.5 in **each** A/B context and accuracy≥6/7 in **each**. “Retained” means both hidden and centered-logit separation≥80% of that seed’s original. “Poor Mixed” means CE≥ln4. “Successful Mixed” requires CE<ln4, accuracy≥6/7 and gap≤.10. These pragmatic flags were fixed in the scratchpad before outcome extraction; they are not proof of convergence and do not select checkpoints or primary comparisons. A stricter perfect-familiar sensitivity and CE/retention threshold grid are supplied.

## 1. Do continued networks reach strong familiar performance?

Yes, especially B and I:14/20 B and16/20 I endpoints meet the strong-familiar definition;12 B and13 I endpoints have perfect A/B accuracy while meeting the CE bound. S does less well:5/20 strong and2/20 perfect. The originals have only4/20 strong and2/20 perfect networks. Thus the original cohort should not be described as uniformly adequate familiar learners.

{md(means)}

## 2. Does absolute Mixed performance improve?

Yes. Mean paired Mixed CE changes are B−1.031, I−.976 and S−1.075 nats; allthree95% seed-bootstrap intervals lie below zero. Cohort mean final Mixed loss is1.218,1.272 and1.173 respectively, versus2.248 originally. Final mean Mixed accuracies are.643,.671 and.636, versus.457 originally. These are meaningful improvements, not perfect transfer.

{md(main)}

All intervals use10,000 paired whole-seed bootstrap draws, RNG20260925,20 independent original seeds per schedule. Changes are330−170; negative CE/gap changes are improvements. Positions, probe variants and endpoints from the same seed are not independently resampled. Intervals are descriptive, unadjusted across multiple metrics.

The three continued cohort means lie below the uniform CE reference. Individual results remain heterogeneous:6/20 B,9/20 I and7/20 S endpoints remain at or above ln4, compared with11/20 originals. A mean below uniform is not evidence that every network generalizes. The table below reports mean Mixed−ln4 and its seed interval; it should not be confused with the familiar-relative gap.

{md(chance)}

## 3. Does the Mixed generalization gap shrink?

Not clearly for B or I. Their mean final gaps remain.922 and1.029 nats; paired changes−.140[−.452,.185] and−.032[−.387,.315] include zero. Their familiar learning and absolute Mixed improvement largely move together. S has a smaller final gap.228 and a clear mean paired decrease−.834[−1.338,−.333], but also much worse familiar performance than B/I. A smaller relative gap can coexist with a weaker familiar learner; it is not by itself a successful acquisition-and-transfer solution.

NoContext and NovelMean show the same need to separate absolute loss from gap. B/I improve absolute losses, yet NoContext gaps remain about1.8–1.9 and NovelMean gaps about1.7–1.8 nats. Full per-schedule paired estimates and intervals are in `tables/bootstrap_summary.csv`, individual cues in `tables/cue_summary.csv` and positions in `tables/position_summary.csv`.

{md(probes)}

## Familiar learning and Mixed improvement

Within B and I, all16/20 endpoints whose familiar CE improves by at least.20 nats also improve absolute Mixed CE. In S,7/8 such endpoints improve Mixed. Pearson correlations between paired familiar and Mixed CE changes are.563,.586 and.325 for B,I,S; Spearman values are.636,.510 and.229. These are descriptive associations, not causal mediation: shared starting difficulty, regression to the mean, training history and coupled outcomes remain alternatives. No regression on pooled80 networks is used to create an inflated independent sample size.

{md(a)}

## 4. Can a network learn A/B well, retain separation and still fail on Mixed?

Yes:1 B,4 I and2 S endpoints meet allthree fixed flags. These7 endpoints represent6 original seeds, because seed2 appears in both B and I. The strongest illustration is I seed9: A CE.008705, B CE.001150, perfect7/7 accuracy in both, but Mixed CE2.070113 with4/7 accuracy. Its Mixed gap is2.065186 nats; hidden separation retains97.75% and centered-logit separation96.26% of its original values. This counterexample survives much stricter familiar-learning and separation thresholds; it is not an artifact of the.5-CE cutoff. Two I endpoints retain separation and fail Mixed even under the perfect-familiar sensitivity. All qualifying cases are listed below and in `tables/identified_examples.csv`, rather than choosing a best reference/model for inference.

{md(examples[examplecols])}

## 5. Can a network learn A/B well, retain separation and successfully generalize?

Yes under the fixed useful-transfer criterion: B seed0. Its A/B CEs are.230407/.093623, accuracies6/7 and7/7, Mixed CE.195330 with6/7 accuracy, and gap.033315. It retains90.26% of hidden and81.50% of output separation. It therefore does not require collapse of A/B differentiation to produce useful Mixed predictions. This is one descriptive case, not a universal mechanism. Under the stricter requirement of perfect familiar accuracy, no endpoint satisfies the complete retained-separation/success flag. “Useful” here does not mean perfect sequence prediction.

{md(counts)}

Mixed improvement also occurs in the following numbers of endpoints with **neither** hidden nor output separation decreasing. Conversely, many B/I endpoints improve while separation decreases. The scatterplots therefore describe co-occurrence; they do not establish that reducing separation causes transfer or that a particular amount of separation is necessary/sufficient.

{md(pd.DataFrame(no_loss))}

## Figures

![Before and after](figures/fig_1_before_after.png)

Figure1. Paired seed-level familiar and absolute Mixed CE for B/I/S, with uniform reference. Black marks give cohort means and seed-bootstrap95% intervals; colored lines show all20 matched seeds.

![Acquisition and transfer](figures/fig_2_familiar_vs_mixed.png)

Figure2. Every original and continued network appears once in the left panel; faint lines connect paired checkpoints. Rings identify good familiar learners with retained separation and poor Mixed performance. The right panel shows60 paired changes, separated by schedule.

![Transfer and separation](figures/fig_3_transfer_vs_separation.png)

Figure3. Mixed CE changes against hidden and centered-logit separation changes, retaining all60 matched endpoints. No fitted line is interpreted as mediation.

## What this audit establishes, and what remains uncertain

The existing continuations establish that substantial additional familiar acquisition can coexist with a persistent Mixed generalization gap, including an especially clear perfect-familiar/retained-separation counterexample. They also establish substantial absolute Mixed improvement and at least one useful retained-separation success. These claims can both be true.

They do not establish an asymptotic deficit after all possible familiar learning, a unique mechanism of failure, or a causal role of separation. Only one fixed additional160-update endpoint per schedule is available. “Strong”, “retained” and “success” are operational descriptors; sensitivity counts and unthresholded values are provided. Conditional familiar-subgroup summaries in `tables/familiar_subgroups_descriptive.csv` are descriptive selections, not new randomized cohorts or primary exclusion rules. Seven teacher-forced predictions do not test autonomous generation or long-horizon stability. No model or hyperparameter was selected on protected probe performance.

Sources: `results/context_mixing_sequence_continuity/v1/tables/d0_native_replay.csv` (originals), `d3_natural_transition.csv` (endpoints), `d3_model_registry.csv`/`d0_checkpoint_audit.csv` (pairing/hashes), `d3_natural_seed.csv` (independent CE reuse check), `d3_separation_by_position.csv`/`d4_original_separation_seed.csv` (geometry reuse checks). Exact input/checkpoint/source-code hashes are in `audit/source_hashes.json`, numerical/coverage checks in `audit/verification.json`, and decisions/test failures in `scratchpad.md` and `audit/*red.log`. All scientific source tables and checkpoints are unchanged.

Reproduce tables/figures in a fresh initialized output with `PYTHONPATH=.:src python -m experiments.familiar_acquisition_generalization.audit --output NEW_DIRECTORY`, then generate this report/auxiliary summaries with `PYTHONPATH=.:src python -m experiments.familiar_acquisition_generalization.reporting --output NEW_DIRECTORY`. Use the existing scratchpad/policy; do not run historical training scripts. Six figure files (three PNG/SVG pairs) can also regenerate from saved audit tables alone via `figures.generate(SOURCE, EMPTY_DESTINATION)`.
'''
 p=out/'FINAL_FAMILIAR_ACQUISITION_GENERALIZATION_REPORT.md'
 with p.open('x') as h:h.write(text)
 print('Report written',p)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT));a=p.parse_args();write(Path(a.output))
