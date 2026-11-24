"""Evidence-linked conclusions from the registered, saved factorial tables."""
import json,platform,sys
import numpy as np
import pandas as pd
import torch,scipy,matplotlib
from .protocol import *
from .figures import generate


def recurrence_verdict(mean,low,high):
 if low>0:return 'supported_positive_paired_effect'
 if high<0:return 'supported_negative_paired_effect'
 return 'not_established'

def md(f):
 def fmt(x):
  if isinstance(x,(float,np.floating)):return 'NA' if not np.isfinite(x) else f'{x:.6f}'
  return str(x)
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in r)+' |' for r in f.itertuples(index=False,name=None))

def run():
 q=pd.read_csv(OUT/'seed_level.csv');g=pd.read_csv(OUT/'tables/geometry_by_position.csv');ci=pd.read_csv(OUT/'tables/architecture_intervals.csv');pair=pd.read_csv(OUT/'tables/paired_intervals.csv');counts=pd.read_csv(OUT/'tables/competence_counts.csv');hist=pd.read_csv(OUT/'tables/historical_reference.csv');d1=pd.read_csv(OUT/'tables/historical_d1_decomposition.csv');registry=pd.read_csv(OUT/'tables/checkpoint_registry.csv');init=pd.read_csv(OUT/'tables/initialization.csv');primary=q[q.step==1000];pcounts=counts[counts.step==1000]
 means=primary.groupby('architecture')[['familiar_ce','Mixed_ce','Mixed_gap','Mixed_accuracy','hidden_separation','logit_separation','AB_probability_tv']].mean().reindex(ARCHS).reset_index();gapci=ci[(ci.step==1000)&(ci.qualification=='all')&(ci.metric=='Mixed_gap')][['architecture','mean','ci_low','ci_high','n_valid']];pc=pair[(pair.step==1000)&(pair.qualification=='all')];r=pc[pc.metric=='recurrence_tanh'].iloc[0];interaction=pc[pc.metric=='interaction'].iloc[0]
 verdict=recurrence_verdict(r['mean'],r.ci_low,r.ci_high);linear=q[q.linear_control_passed.eq(True)];nonlinear=primary[primary.activation=='tanh'];position=g[g.step==1000].groupby(['architecture','transition'])[['hidden_midpoint_residual','centered_logit_midpoint_residual']].mean().reset_index();position.to_csv(OUT/'tables/primary_geometry_means.csv',index=False)
 substantial=q[q.competent_context_success.eq(True)];substantial.to_csv(OUT/'tables/context_success_all_fixed_checkpoints.csv',index=False)
 examples=primary[primary.competent_material_gap.eq(True)][['seed','architecture','A_ce','B_ce','Mixed_ce','Mixed_gap','Mixed_accuracy','hidden_separation','logit_separation','AB_probability_tv']]
 secondary=primary.groupby('architecture')[['NoContext_ce','NoContext_gap','NoContext_accuracy','NovelMean_ce','NovelMean_gap','NovelMean_accuracy']].mean().reindex(ARCHS).reset_index();secondary.to_csv(OUT/'tables/secondary_probe_summary.csv',index=False)
 saturation=primary[primary.activation=='tanh'].groupby('architecture')[['A_saturation_fraction','B_saturation_fraction','Mixed_saturation_fraction']].mean().reset_index();saturation.to_csv(OUT/'tables/tanh_saturation_summary.csv',index=False)
 d1summary=d1.groupby('transition')[['j_hidden_norm','r_hidden_norm','e_hidden_norm','mixed_excess_ce']].mean().reset_index();d1summary.to_csv(OUT/'tables/historical_d1_summary.csv',index=False)
 versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),training_device='CPU',training_torch_threads=1,deterministic_algorithms=True);dump(OUT/'audit/environment.json',versions)
 claims=[('C1','Allfour architectures acquired familiar A/B at primary1000','tables/competence_counts.csv','20/20 strong and20/20 perfect in each arm; earlier170 FF-Tanh14/20 is acquisition-confounded.'),('C2','Linear midpoint/CE bound validated algebraically','seed_level.csv','160 affine controls; do not interpret tiny negative Jensen gaps as tests of exact linearity.'),('C3','Recurrence is not necessary for a positive gap in this matched-input task','tables/qualified_examples_all.csv','FF-Tanh seeds10/11 have strong familiar learning and material positive gaps; this is not inevitable failure of all tanh models.'),('C4','Recurrence worsening the nonlinear gap is not established','tables/paired_intervals.csv','Paired95% interval crosses zero; architecture also changes optimization/learned representations.'),('C5','Nonlinear midpoint deviations appear at first position','tables/onset_counts.csv','40/40 primary nonlinear models cross fixed1e-8 double residual threshold at position1; magnitude is not CE mediation.'),('C6','Primary context-probability dependence is small despite raw separation','seed_level.csv','0/80 substantial-dependence flags at1000; earlier fixed checkpoints have qualified successes, not human-like phenotype evidence.'),('C7','Original nonlinear mismatch/propagation audit reused unchanged','tables/historical_d1_decomposition.csv','Historical current-token/block-trained SRNs are a different cohort and task; no new original-model rollout.')]
 claimrows=[]
 for cid,claim,source,limit in claims:claimrows.append(dict(claim_id=cid,claim=claim,source_table=source,source_sha256=hash_file(OUT/source),limit=limit))
 pd.DataFrame(claimrows).to_csv(OUT/'tables/claim_source_map.csv',index=False)
 context_counts=counts[['step','architecture','substantial_context_dependence','competent_context_success']]
 text=f'''# What prevents context averaging in SRNs? Registered recurrence × activation factorial

**Nonlinearity can break context averaging without recurrence. This run does not show recurrence worsening the nonlinear Mixed gap.** All80 networks acquired A/B extremely well by the primary endpoint1000. Both linear architectures obeyed the exact hidden/logit midpoint identities and convex-CE bound. FF-Tanh and RNN-Tanh had positive average Mixed gaps, but most networks still predicted Mixed well in absolute terms. The new matched-input task therefore demonstrates a nonlinear relative deficit, not the original canonical SRN's level of absolute Mixed failure.

## Primary results: all20 paired seeds per architecture, update1000

{md(means)}

Cross-entropy is in nats; uniform prediction is ln4=1.386294. Familiar CE=(A+B)/2; gap=Mixed−familiar. Every primary model has7/7 A and7/7 B accuracy and CE≤.5 in each context. No primary comparison is acquisition-confounded under the frozen80%-strong reliability descriptor. At170, FF-Tanh has only14/20 strong models, so unqualified early architecture comparisons can be acquisition-confounded. Update330 onward has20/20 strong and perfect familiar models in every arm. All intermediate checkpoints were fixed beforehand and are secondary.

{md(pcounts[['architecture','strong_familiar','perfect_familiar','mixed_success','positive_gap','material_gap','substantial_context_dependence']])}

“Positive” means gap>1e-6; “material” gap>.10. Useful Mixed success requires CE≤.5,accuracy≥6/7,gap≤.10. These descriptors were frozen before training, not tuned after results. All80 primary Mixed losses are below the uniform reference; relative gaps should not be described as uniformly poor absolute prediction.

## 1. Do allfour architectures adequately learn A/B?

Yes at1000:20/20 in each condition meet both the registered strong-familiar criterion and the stricter perfect-accuracy sensitivity. The strong and four-arm-common qualified primary summaries therefore use the same20 seeds as the all-seed results. Small continuous familiar differences remain: paired RNN-Tanh−FF-Tanh familiar CE is{pc[pc.metric=='recurrence_tanh_familiar']['mean'].iloc[0]:.6f},95% interval[{pc[pc.metric=='recurrence_tanh_familiar'].ci_low.iloc[0]:.6f},{pc[pc.metric=='recurrence_tanh_familiar'].ci_high.iloc[0]:.6f}]. This is not an exact familiar-performance matching experiment. All seeds/interim failures are retained; no endpoint was selected on protected cues.

## 2. Do both linear architectures satisfy exact midpoint averaging?

Yes. For fixed weights and identical initial states, induction gives h_M,t=(h_A,t+h_B,t)/2 for identity activation, because the complete current input is the average and the recurrence/output maps are affine. Then z_M,t=(z_A,t+z_B,t)/2. For the same target, convexity of CE in logits gives L_M,t≤(L_A,t+L_B,t)/2 at each position, hence over allseven. This does **not** imply p_M=(p_A+p_B)/2; softmax probabilities need not average. Training itself is nonlinear through CE/Adam, but the frozen forward map at a fixed checkpoint is affine.

All160 linear checkpoint controls pass. Maximum double hidden residual={linear.double_hidden_max_error.max():.3e}, logit residual={linear.double_logit_max_error.max():.3e}. Actual float32 maximum coordinate residuals are{linear.float32_hidden_max_error.max():.3e} and{linear.float32_logit_max_error.max():.3e}, both below predeclared propagated affine/head summand bounds. Float64 tolerance is1e-10+1e-12×scale; source float32 uses64eps32 summand errors and |Wh| propagation, with CE convexity tolerance2max(logit_bound). These are algebra/numerical controls, not conventional statistical tests of whether exact linearity holds. Their small negative gaps are compatible with Jensen's inequality. They do not guarantee NoContext/Novel performance because those cues are not necessarily between A and B.

## 3. Can tanh alone produce a positive gap without recurrence?

Yes. FF-Tanh mean gap=.053868 nats,95% seed-bootstrap interval[.001698,.140718];13/20 gaps are positive and2/20 exceed.10, all with perfect familiar accuracy. RNN-Tanh mean gap=.020441[.003778,.045719], with14/20 positive and1/20 above.10. The existence of competent FF-Tanh failures establishes that recurrence is not necessary for this **relative** deficit in the matched-window setup. Nonlinearity is not a guarantee of failure:18/20 FF-Tanh and19/20 RNN-Tanh satisfy useful-Mixed success.

{md(gapci)}

All material-gap cases are below. They are not chosen best models and are not exclusions from primary estimates. FF-Tanh seed10 has familiar CE≈.00463 and Mixed CE≈.71481: a clear relative gap, but still below uniform. Large geometric midpoint deviations can coexist with low CE.

{md(examples)}

## 4. Does recurrence increase the nonlinear gap?

Not established: RNN-Tanh−FF-Tanh={r['mean']:.6f},95% paired interval[{r.ci_low:.6f},{r.ci_high:.6f}]. The point estimate is negative rather than an amplification. The recurrence × activation interaction is{interaction['mean']:.6f}[{interaction.ci_low:.6f},{interaction.ci_high:.6f}], also crossing zero. Do not convert an interval spanning zero into proof of no effect or choose an earlier checkpoint to change this conclusion. Directional verdict: `{verdict}`.

{md(pc[['metric','mean','ci_low','ci_high','n_valid']])}

Intervals use10,000 whole-seed paired bootstrap draws,RNG20260925. The interaction is (RNN-Tanh−FF-Tanh)−(RNN-Linear−FF-Linear) on the gap. Checkpoints, positions and cues are repeated measures; the same seed supplies allfour arms. Qualified intersections and complete continuous familiar/Mixed values are in the saved tables; no80-network independent-arm bootstrap is used. Confidence intervals across secondary metrics are descriptive/unadjusted.

## 5. Where does the nonlinear deviation first appear, and how does it evolve?

At primary1000, all20 FF-Tanh and all20 RNN-Tanh models first cross the prespecified1e-8 double hidden and centered-logit residual threshold at position1. Alllinear cases remain numerical zero. The first recurrent state has zero prestate, so this first nonlinear discrepancy does not require propagation of prior mismatch. Later residual magnitudes fluctuate rather than showing uniform monotonic growth. Position means are in `tables/primary_geometry_means.csv`, individual vectors and all7 positions in `tables/geometry_by_position.csv` and `transition_level.csv`. Onset is a thresholded numerical diagnostic, not evidence of lost psychological sequence memory.

Tanh saturation (fraction |h|≥.95) is reported separately:

{md(saturation)}

Neither saturation nor curvature is isolated causally here. We did not replace activations within a single trained nonlinear model or optimize hidden states.

## 6. Can substantial context dependence coexist with successful generalization?

Yes at some **secondary fixed checkpoints**, but not under the functional criterion at1000. Substantial dependence requires raw hidden distance≥1, centered-logit distance≥1 AND mean A/B probability total variation≥.1. Competent, context-dependent useful-Mixed cases occur in4 FF-Tanh and2 RNN-Tanh networks at170, and1 FF-Tanh network at330. They are all exported in `tables/context_success_all_fixed_checkpoints.csv`; these checkpoints were not selected for the primary claim. Primary1000 has0/80 substantial-dependence flags. Maximum primary mean A/B TV is{primary.AB_probability_tv.max():.6f}, below.1.

{md(context_counts)}

Raw hidden and centered-logit separations remain large even in the successful linear conditions (mean hidden≈7.2–7.3; centered-logit≈6.6–14.2). Their probabilities are nearly context-independent because the correct class dominates; large contrasts among negligible wrong-class probabilities need not change prediction. Raw representation distance alone therefore does not establish functionally separate skills. Neither primary successes nor the earlier probability-dependent successes are labeled human-like splitting/transfer: the Cross/Split phenotype and human behavior were not tested. A sensitivity grid over output distance/TV thresholds is saved, with no primary threshold changes.

## 7. Architectural conclusion and limits

Within this specific task, affine feedforward and affine recurrent computations structurally preserve Mixed logit interpolation and preclude a positive Mixed CE gap except arithmetic error. Tanh is sufficient for observed interpolation failure without recurrence. This run does not support the proposed recurrent amplification. These claims concern the architecture/training ecology manipulated **before** learning, not a unique causal pathway inside the original frozen SRN. Architecture changes optimization and learned representations as well as forward computation.

The new15-feature causal window makes the task linearly solvable without recurrent memory; the original used only four current-token channels. Recurrent initialization is normalized to spectral norm.9 in both new recurrent arms, instead of the original unnormalized High recurrent matrix. Normalization is applied only at initialization, never projected after Adam updates. Training here is ABAB whole-sequence alternation for1000 updates, unlike original A10/B10 blocks for170. These controls make this a matched new cohort, **not** a direct original-model ablation or demonstration that original failure was caused by recurrence. Historical originals have mean familiar CE{hist.familiar_ce.mean():.6f},Mixed{hist.Mixed_ce.mean():.6f}, but those numbers are contextual references, not interchangeable factorial cells.

Secondary NoContext/Novel behavior is distinct from midpoint interpolation:

{md(secondary)}

Allfour individual Novel cues have absolute CE,accuracy and familiar-relative gap in `seed_level.csv` and `tables/architecture_intervals.csv`; NovelMean averages allfour within a model before seed inference. No protected probe entered training, parameter selection, model selection or the stopping rule.

## Complementary historical frozen D1 diagnostic

The actual previously audited D1 table is reused byte-for-byte. At observed familiar prestates, write a_bar=(a_A+a_B)/2 and prior mismatch e_prev=h_M,prev−(h_A,prev+h_B,prev)/2. The imported equations are j=tanh(a_bar)−(tanh(a_A)+tanh(a_B))/2 and r=tanh(a_bar+Wh e_prev)−tanh(a_bar), so ideal e=j+r. The source's observed float32 remainder is separately bounded. At first position r=0. The original D1 audit verified this decomposition and its existing single-correction test was inconclusive; neither is rerun here.

{md(d1summary)}

These are original-cohort descriptive norms and CE, not factorial results. Vector identities do not make norm sums additive or percentages of cross-entropy caused. No new decomposition or original checkpoint modification was introduced.

## Four primary figures

![Architecture design](figures/fig_1_architecture_design.png)

Figure1: only recurrence and hidden activation differ within the matched-input2×2 design; independent execution policy is used for source-fidelity training but never scores prediction.

![Learning and transfer](figures/fig_2_learning_transfer.png)

Figure2: all20 seed trajectories per architecture, fixed170/330/660/1000 checkpoints, familiar/Mixed CE and uniform reference. Symmetric-log scale is linear within±1e-4 and does not clip losses. Bands are whole-seed95% intervals.

![Midpoint deviation](figures/fig_3_midpoint_residuals.png)

Figure3: actual float32 residuals by position at1000; exact double affine controls are exported separately. Thin lines retain seeds. Hidden and centered-output geometry are not percentages of CE explained.

![Gap and acquisition](figures/fig_4_recurrence_gap.png)

Figure4: allseed gaps, familiar performance, paired nonlinear recurrence contrast and factorial interaction at the primary endpoint.

## Integrity and reproducibility

Exactly80,000 scientific updates completed;80 independent trained networks/320 fixed checkpoints, no numerical training failures and no retries. All80 training attempts and320 checkpoint slots were sealed before any protected evaluation. All320 seed/checkpoint rows and17,920 cue-position rows remain. Source/objective/Adam/policy-gradient parity fixtures and14 pretraining tests passed; subsequent evaluation/statistics/report tests retain behavioral RED logs. `audit/pretraining_gate.json`, `audit/training_seal.json`, `audit/evaluation_complete.json`, and final manifest bind source/code/checkpoints. Original20 checkpoint hashes remain unchanged. The pretraining source-map lookup correction is logged; no architecture/loss/budget/threshold changed after protected outcomes.

Model states, separate policies and optimizer moments for allfour fixed checkpoints are in `checkpoints/`. Reproducibility configuration is `audit/config.json`; paired initial hashes in `tables/initialization.csv`; full familiar-only histories in `tables/training_history.csv`; software versions in `audit/environment.json`. Exact mathematical fixtures, causal-input LP witness, negative leakage fixtures, per-step precision audits, source-to-claim map and scratchpad are retained. No additional scientific runs are proposed by this report.
'''
 with (OUT/'RESULTS.md').open('x') as f:f.write(text)
 products=generate();dump(OUT/'audit/report_products.json',{'RESULTS.md':hash_file(OUT/'RESULTS.md'),**{'figures/'+p:hash_file(OUT/'figures'/p) for p in products}})
 append('## Findings and reporting stopping decision\nAll80 primary models strong/perfect familiar. Linear160 checkpoint controls pass. FF-Tanh competent positive gaps13/20, material2/20; RNN-Tanh14/20, material1/20. Primary RNN−FF gap−.033427 CI[−.103218,.007831], no amplification claim. Primary probability-context dependence small; earlier fixed checkpoints retain qualified context-success cases. No architecture, budget, loss, threshold or checkpoint changed. Four saved-table figures and source-to-claim map generated; stop after verification, no new training.')
 print('Report and four figures complete',flush=True)

if __name__=='__main__':run()
