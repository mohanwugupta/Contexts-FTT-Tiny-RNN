"""Report and claim ledger, generated only from completed tables."""
import pandas as pd
from .core import *

def run():
 read=lambda n:pd.read_csv(OUT/n,float_precision='round_trip');q=read('tables/endpoint_seed_summary.csv');counts=read('tables/counts.csv');ci=read('tables/endpoint_intervals.csv');qualified=read('tables/competence_qualified_intervals.csv');parity=read('tables/canonical_training_parity.csv');controls=read('tables/linear_controls.csv')
 assert len(parity)==20
 cohort=q.groupby(['architecture','checkpoint'])[['A_ce','B_ce','familiar_ce','Mixed_ce','Mixed_accuracy','Mixed_gap']].mean().reset_index().merge(counts,on=['architecture','checkpoint']);cohort.to_csv(OUT/'tables/cohort_summary.csv',index=False)
 normraw=pd.read_csv(ROOT/'results/recurrent_learning_transfer/v1/tables/transition_predictions.csv');sat=[]
 for (a,k,seed),f in normraw[normraw.cue.isin(['A','B'])].groupby(['architecture','endpoint','seed']):
  h=f[[f'hidden_{j}' for j in range(8)]].to_numpy();sat.append(dict(seed=seed,architecture=a,checkpoint=k,initialization='normalized',abs_hidden_gt_095=float(np.mean(abs(h)>.95)),abs_hidden_gt_099=float(np.mean(abs(h)>.99)),is_bounded_tanh_saturation=a=='RNN-Tanh'))
 for r in q[q.checkpoint.isin([170,330])].itertuples():sat.append(dict(seed=r.seed,architecture=r.architecture,checkpoint=r.checkpoint,initialization='unnormalized',abs_hidden_gt_095=r.saturation_95,abs_hidden_gt_099=r.saturation_99,is_bounded_tanh_saturation=r.architecture=='RNN-Tanh'))
 sat=pd.DataFrame(sat);sat.to_csv(OUT/'tables/saturation_comparison.csv',index=False);normal_sat=sat[(sat.architecture=='RNN-Tanh')&(sat.initialization=='normalized')&(sat.checkpoint==170)].abs_hidden_gt_095.mean()
 rows=[]
 for r in cohort.itertuples():rows.append(f'| {r.architecture} | {r.checkpoint} | {r.A_ce:.6f} | {r.B_ce:.6f} | {r.Mixed_ce:.6f} | {r.Mixed_accuracy:.4f} | {r.Mixed_gap:.6f} | {r.strong_familiar}/20 | {r.Mixed_above_chance}/20 | {r.good_familiar_poor_Mixed}/20 |')
 table='\n'.join(rows)
 report=f'''# Does original High-scale initialization restore Mixed failure?

**Yes for the canonical Tanh recipe; the unnormalized Linear comparison is acquisition-limited.** New Tanh reproduces every archived seed's original update170 weights, execution policy and entire170-loss trajectory exactly. Its mean Mixed CE is2.248301 at170 versus.430016 in the normalized program. Linear retains affine averaging at all100 tested model/checkpoint combinations, but learns familiar A/B well in only2/20 seeds at170 and14/20 at330. Successful Linear learners generalize; the full Linear cohort does not reliably acquire the task within the fixed budget.

## Fixed cohorts and endpoints

20 paired seeds per new architecture, source Normal(0,2) input/recurrent/output tensors, original biases and neutral cue channel, no Wh rescaling/projection. Current-token4+cue3 inputs,8 hidden units, zero h0, same7 teacher-forced targets. Source CE+.20 detached-confidence policy objective, Adam.005 and jointclip1;330 A10/B10 updates. All40 runs completed13,200 scientific updates without nonfinite failure; all240 checkpoint slots were sealed before Mixed probes. Large finite losses are retained, not classified as numerical failures.

Primary170 was analyzed first;330 is the fixed secondary endpoint. Allstates were fixed before outcomes. Good familiar means the unchanged CE≤.5 and accuracy≥6/7 separately in A/B. “Poor Mixed” in the severe-failure count means CE>ln4=1.386294; the earlier useful-Mixed definition is also retained separately. All20 seeds remain in every primary estimate.

| Architecture | Update | Mean A CE | Mean B CE | Mean Mixed CE | Mixed accuracy | Mixed−familiar | Good familiar | Mixed>chance | Good familiar+Mixed>chance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table}

Full-precision values and pointwise95% seed-bootstrap intervals are in `tables/endpoint_intervals.csv`;10,000 whole-seed draws,RNG20260925. Updates, positions and canonical duplicates are not independent networks. Conditional summaries retain nvalid out of20 and are descriptive, not selected models.

## Answers to the seven questions

1. **Does unnormalized Linear learn reliably?** Not at170: only2/20 qualify;330 reaches14/20. Familiar CE means387.820 and55.092 are dominated by unsuccessful, often confidently wrong seeds. Initial Linear hidden magnitudes reached638,585 and pre-update CE743,971. No normalization, hidden clipping, budget extension or hyperparameter rescue followed. At330, the14 familiar-qualified Linear seeds have mean Mixed CE.000994 [95% CI .000406,.001678] and100% Mixed accuracy. The two qualified170 seeds also have100% Mixed accuracy, but n=2 is a weak population basis.

2. **Context dependence and incomplete transfer?** Yes descriptively: A/B hidden/logit separation persists and practiced blocks often improve their own context more than the other, with substantial negative cross effects retained. Late allseed Linear A/B block specificity means are538.24/570.64 nats; these large values reflect poorly learned/high-scale outliers and are not competence-matched evidence. Native Tanh late directional means are.163/.222, both with positive seed intervals. Shared improvement and interference can both occur. Geometry alone is not separate learning. No new one-update diagnostics were requested, so the earlier joint functional splitting gate is **not assessed**, not silently redefined. Near-floor loss reductions are not required to preserve a contextual representation.

3. **Linear averaging?** All100 model/state checks pass (20 seeds×80/100/120/170/330). Under the same tokens and initial state, h_M=(h_A+h_B)/2 and z_M=(z_A+z_B)/2 in real arithmetic; convex logsumexp(z)−z_target then gives CE_M≤(CE_A+CE_B)/2 positionwise and after averaging. This does not imply arithmetic softmax averaging. Double maximum hidden/logit residuals are{controls.double_hidden_error.max():.3e}/{controls.double_logit_error.max():.3e}, within original scale-aware tolerances. All700 positionwise float32 controls pass separately propagated64eps bounds. A loss bound relative to poor familiar predictions is not an absolute success guarantee: Linear Mixed CE326.264 at170 is below familiar387.820 yet still extremely poor.

4. **Does Tanh restore severe impairment?** Yes. At170 Mixed CE2.248301 [1.613604,2.908528],accuracy45.7%,gap1.061753 [.581578,1.554507],11/20 above chance. At330 Mixed improves to1.226759 [.849323,1.631413] while the gap remains.868721 [.467949,1.288988], with7/20 above chance. New Tanh170 is an exact training reproduction of the original canonical models, not20 new independent confirmations. Archived NumPy versus new native Torch frozen CE differs by at most3.01e−6, despite identical trained tensors.

5. **Penalty among adequate familiar learners?** Yes, with small-sample limits at the primary endpoint. The4 qualified Tanh170 seeds have Mixed CE.977921 [.601109,1.354732] versus familiar.171595, gap.806326 [.455938,1.156714]. None of those4 exceeds chance, but none meets the inherited useful-Mixed standard. At330,13 qualify; their Mixed CE is1.001375 [.596426,1.476442], gap.870858 [.466318,1.340456];3/13 exceed chance. The primary and secondary counts remain separate; no favorable endpoint replaces170.

6. **What changes with initialization?** Acquisition, magnitude and bounded Tanh saturation differ substantially. New Tanh170 has73.125% of familiar hidden coordinates with abs(h)>.95, versus{normal_sat:.3%} in normalized Tanh. New Tanh170 hidden centroid distance1.932 and centered-logit separation5.808 exceed normalized1.659/3.579 descriptively. Linear distances are huge and scale-dependent; abs(h)>.95 in an identity network is an amplitude marker, **not saturation**. The generic table field `saturation_95/99` must be read with this distinction. PCA is fit only to14 familiar states per model; alltrajectories are saved, not used as a splitting classifier.

7. **Nonlinearity or recurrent scale?** Affine recurrence structurally guarantees averaging whenever numerically valid; tanh need not preserve a cue midpoint. The restored canonical recipe reproduces the motivating failure, whereas the normalized recipe attenuated it. But the normalized program used a different RNG draw path and hand-unrolled Tanh arithmetic, and acquired A/B much better. Thus the cross-program comparison does **not** isolate Wh normalization alone or establish saturation as causal mediation. New arms are exactly paired within seed; cross-program contrasts are unpaired descriptive. A clean claim is: high-scale canonical Tanh dynamics reproduce substantial Mixed impairment; identity dynamics retain the averaging guarantee but have serious acquisition difficulty under that same fixed high-scale recipe.

## Figures and reproducibility

1. [Actual alternating training losses and first100 zoom](figures/figure_1_alternating_learning.png)
2. [Absolute A/B gap and signed B−A diagnostic](figures/figure_2_AB_gap.png)
3. [Symmetric A/B cross-context pooling](figures/figure_3_cross_context_pooling.png)
4. [A/B/Mixed across fixed learning stages](figures/figure_4_Mixed_learning_stages.png)
5. [Primary initialization comparison, canonical reference separated](figures/figure_5_initialization_generalization.png)

Each has SVG alongside PNG; every seed is retained. Symlog axes are labeled where needed to show large signed ranges and exact zeros without clipping. Source-to-claim links: `tables/claim_ledger.csv`. Root training/splitting/generalization CSVs preserve individual measurements; checkpoint registry, configuration, numerical controls, PCA, failure table(empty because allfinite), parity audit, source hashes, append-only scratchpad and reproduction guide document provenance. No follow-up experiments or protected-result tuning were added.
'''
 with (OUT/'REPORT.md').open('x') as f:f.write(report)
 claims=[('C1','All20 canonical170 training trajectories and model/policy tensors exactly reproduced','tables/canonical_training_parity.csv','Historical reference is not independent'),('C2','Linear acquisition unreliable at170 and incomplete at330','tables/counts.csv','Do not claim good relative gap means absolute success'),('C3','All100 Linear averaging/CE controls pass','tables/linear_controls.csv','Exact real arithmetic; float32 has separate bounded errors'),('C4','Severe Tanh Mixed failure restored atprimary170','tables/endpoint_intervals.csv','Original reproduction, not independent replication'),('C5','Tanh penalty remains among familiar-qualified networks','tables/competence_qualified_intervals.csv','Smallconditionaln, primary/secondary distinct'),('C6','Context-preferential block effects coexist with incomplete transfer','tables/block_epoch_intervals.csv','Geometry and allseed outlier effects do not prove distinctskills'),('C7','Normalized/new comparison differs in acquisition and implementation provenance','tables/initialization_comparison.csv','Unpaired descriptive, not isolated normalization effect'),('C8','Bounded Tanh saturation differs','tables/saturation_comparison.csv','Not causal mediation; identity amplitude not saturation')]
 pd.DataFrame([dict(claim_id=i,claim=c,source=n,sha256=sha(OUT/n),limitation=limit) for i,c,n,limit in claims]).to_csv(OUT/'tables/claim_ledger.csv',index=False)
 append('## Report and interpretive limits\nExactcanonicalTanhreproduction20/20; severeMixedprimaryrestored. Linear averaging100/100 holds butprimaryfamiliaracquisition2/20,secondary14/20; do notpromote boundtosuccess. Tanhconditionalprimaryn4 penaltybut0abovechance;secondaryn13 with3abovechance. No posthocpoorperformance threshold,changedgateorrescues. NormalizedcomparisonhasRNG/kernel/acquisitionconfounds andhistoricalreferenceisnonindependent. Fivefigures,fulltablesandclaimledgergenerated. Stopafterverification.')
 print('Report and8-claim ledger generated')

if __name__=='__main__':run()
