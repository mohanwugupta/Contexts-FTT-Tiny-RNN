"""Conclusions explicitly qualified by acquisition and functional-phenotype evidence."""
import json,sys,platform
import numpy as np
import pandas as pd
import torch,scipy,matplotlib
from .design import *
from .figures import generate


def critical_verdict(common_strong_split,positive_gap_effect):
 if common_strong_split==0:return 'critical_split_control_not_recovered'
 if not positive_gap_effect:return 'gap_effect_not_established'
 return 'conditional_support_requires_competence_and_optimization_caveats'

def md(df):
 def fmt(v):
  if isinstance(v,(float,np.floating)):return 'NA' if not np.isfinite(v) else f'{v:.6f}'
  return str(v)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def run():
 q=pd.read_csv(OUT/'tables/endpoint_summary.csv');counts=pd.read_csv(OUT/'tables/counts.csv');ci=pd.read_csv(OUT/'tables/endpoint_intervals.csv');pair=pd.read_csv(OUT/'tables/paired_intervals.csv');wide=pd.read_csv(OUT/'tables/direct_specificity.csv');ini=pd.read_csv(OUT/'tables/initial_transfer_intervals.csv');boundary=pd.read_csv(OUT/'tables/switch_costs.csv');linear=pd.read_csv(OUT/'tables/linear_controls.csv');res=pd.read_csv(OUT/'tables/midpoint_residuals.csv');d1=pd.read_csv(OUT/'tables/tanh_D1_decomposition.csv');links=pd.read_csv(OUT/'tables/D1_source_linkage.csv')
 primary=q[q.endpoint==170];secondary=q[q.endpoint==330];means=q.groupby(['endpoint','architecture'])[['familiar_ce','Mixed_ce','Mixed_gap','Mixed_accuracy','mean_specificity','mean_cycle_cost']].mean().reset_index();pc=counts[counts.endpoint==170];sc=counts[counts.endpoint==330]
 p=pair[(pair.endpoint==170)&(pair.qualification=='all')];gapp=p[p.metric=='Mixed_gap'].iloc[0];common=pair[(pair.endpoint==170)&(pair.qualification=='common_strong_split')&(pair.metric=='Mixed_gap')].iloc[0];verdict=critical_verdict(int(common.n_valid),gapp.ci_low>0)
 mainci=ci[(ci.endpoint==170)&(ci.qualification=='all')&ci.metric.isin(['familiar_ce','Mixed_ce','Mixed_gap','mean_specificity','mean_cycle_cost'])][['architecture','metric','mean','ci_low','ci_high','n_valid']]
 qualified=ci[(ci.metric=='Mixed_gap')][['endpoint','architecture','qualification','mean','ci_low','ci_high','n_valid','status']]
 direct=wide.groupby(['checkpoint','architecture'])[['delta_A_A','delta_B_A','delta_B_B','delta_A_B','D_A','D_B','mean_specificity','mean_own_improvement','direct_evidence']].mean().reset_index();direct.to_csv(OUT/'tables/diagnostic_means.csv',index=False)
 reacq=boundary.groupby(['architecture','reacquisition_status']).size().rename('boundary_cells').reset_index();reacq.to_csv(OUT/'tables/reacquisition_coverage.csv',index=False)
 geom=q.groupby(['endpoint','architecture'])[['hidden_centroid_distance','position_matched_hidden_separation','centered_logit_separation','AB_probability_tv','mean_specificity']].mean().reset_index();geom.to_csv(OUT/'tables/geometry_means.csv',index=False)
 initial=ini[ini.metric.isin(['trained_improvement','cross_improvement','asymmetry','untrained_context_loss_after_block'])][['architecture','direction','metric','mean','ci_low','ci_high']]
 signed_examples=q[(q.strong_familiar.eq(True))&(q.material_gap.eq(True))];signed_examples.to_csv(OUT/'tables/competent_gap_cases_all.csv',index=False)
 d1means=d1.groupby(['endpoint','transition'])[['j_hidden_norm','r_hidden_norm','e_hidden_norm','j_task_norm','r_task_norm']].mean().reset_index();d1means.to_csv(OUT/'tables/D1_means.csv',index=False)
 claims=[('C1','Linear recurrence can acquire the original seven-step task from current tokens only','tables/endpoint_summary.csv','20/20 strong/perfect at170; fixed-task result, not all sequences.'),('C2','Early practice is context-biased in both architectures','tables/initial_transfer_intervals.csv','Initial asymmetry positive; does not alone establish robust learned separate skills.'),('C3','Critical endpoint splitting control was not recovered','tables/splitting_classification.csv','0/20 joint positive cases in each arm at170/330; near-stationary updates are uninformative, not proof of shared learning.'),('C4','Linear midpoint/CE bound holds algebraically','tables/linear_controls.csv','40 trained endpoint controls; not a statistical test of exact linearity or proof of splitting.'),('C5','Tanh shows a positive Mixed gap, including competent-only subsets','tables/endpoint_intervals.csv','Unequal familiar losses and absent common strong+split support preclude the strongest under-splitting causal comparison.'),('C6','Geometric separation is insufficient to classify learning transfer','representational_geometry.csv','PCA/centroid/logit distances are secondary; do not establish separate human skills.'),('C7','D1 describes instantaneous and propagated deviation in frozen new Tanh models','tables/tanh_D1_decomposition.csv','Imported equations, audited NumPy/Torch linkage; norms do not partition CE causation.')]
 pd.DataFrame([dict(claim_id=i,claim=c,source_table=s,source_sha256=hash_file(OUT/s),limitation=l) for i,c,s,l in claims]).to_csv(OUT/'tables/claim_source_map.csv',index=False)
 dump(OUT/'audit/environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),device='CPU',training_threads=1,deterministic_algorithms=True))
 text=f'''# Does nonlinear recurrence cause context-mixing failure?

**The strongest proposed comparison is inconclusive because the registered endpoint functional-splitting phenotype was not recovered.** Both architectures show preferential early learning in the practiced context. Linear recurrence acquires the original current-token task and preserves exact Mixed averaging; Tanh has a positive Mixed gap, including among familiar-competent models. But neither arm meets the prespecified joint switch-cost plus bidirectional one-update criterion at170 or330. This is not evidence that all learning was shared: the linear models are nearly stationary by170, and late Tanh updates are directionally asymmetric under the restored Adam state. Geometry alone does not repair this missing control.

## Primary update170, before the fixed secondary330 analysis

{md(means[means.endpoint==170])}

{md(pc[['architecture','strong_familiar','perfect_familiar','direct_evidence','direct_uninformative','boundary_evidence','functional_split','trajectory_specific_learning','strong_split','material_gap','learned_skill_generalization']])}

There are20 paired seeds per architecture. Strong familiar means CE≤.5 and accuracy≥6/7 **separately** in A/B; the perfect sensitivity also requires7/7 each. Linear has20/20 strong and perfect models; Tanh has18/20 strong and13/20 perfect. Useful Mixed prediction additionally requires CE≤.5,accuracy≥6/7 and gap≤.10; a weak familiar learner is not counted as generalizing learned skills. Uniform CE is ln4=1.386294. Absolute Mixed loss and Mixed−familiar gap are always separate.

{md(mainci)}

These are10,000-draw whole-seed bootstrap95% intervals, RNG20260925. All20 seeds remain in primary descriptive estimates. Updates, cue positions and diagnostic clones are repeated measures, not independent networks. Secondary/qualified intervals are descriptive and unadjusted across multiple metrics.

## 1. Can a linear recurrent network learn the original sequence from current tokens?

Yes: all20 Linear models predict allseven familiar A/B targets correctly at170 and330. Mean familiar CE is.000516 at170 and.000185 at330 (exact tables above). Input is only current-token one-hot4 plus cue3; no history windows, position labels, embeddings or future tokens. The target-dependent ambiguity of the current token is resolved through recurrent state.

A separate deterministic8-state capacity witness was verified before training. It uses Wh=.9 times a shift matrix, bias=e1 and zero input weights, yielding h_t[j]=.9^j for j<t. Fixed decoder differences recover the target score vector at each internal clock state. It solves the seven targets without optimization or supplied position features, and removing recurrence breaks the solution. It demonstrates representability, not that the trained models learned that specific algorithm or can solve arbitrary sequences.

## 2. Do both architectures develop context-specific learning?

They show early context-specific **practice effects**, but the stronger endpoint phenotype is not established. During the first A block, Linear improves A by8.0033 nats but transfers only2.6682 to B: asymmetry5.3351[3.8031,7.0438]. Tanh improves A1.2628 versus B.3272: asymmetry.9356[.7619,1.1222]. B immediately before its first training block has CE13.7913 for Linear and4.6452 for Tanh. Raw magnitudes are not standardized across architectures, whose initial loss scales differ markedly.

The symmetric first B block uses its actual starting state at10 as baseline. First-block results:

{md(initial)}

I_A−T_A→B exactly equals the first baseline-adjusted A→B cost; these are algebraically the same evidence, not independent confirmations. At every boundary, raw incoming−outgoing CE and the change from the same-direction initial cue difference are supplied. To cancel static cue difficulty in the joint history criterion, complete A→B/B→A cycle costs are averaged: the initial A/B differences cancel exactly. Only8 complete cycles through160 qualify170; only16 through320 qualify330. Reacquisition after170 never enters the170 classification.

## Primary direct diagnostic: actual restored-Adam learning updates

At0,10,20,40,80,120,170,330, two separate clones start from the same model, policy AND optimizer state, including moments/counters. One receives an actual A update, the other B. For each j|i, Δ_j|i=L_j(before)−L_j(after i); negative effects are retained. Original model/policy/Adam/RNG hashes and short-trajectory fixtures prove the scientific path is unchanged. There are640 registered diagnostic updates, distinct from13,200 scientific updates.

The table records all signed within/cross effects. Columns delta_A_A and delta_B_A mean evaluation A/B after training A; delta_B_B and delta_A_B are evaluation B/A after training B.

{md(direct)}

At170, all20 Tanh models have D_A>0 but D_B<0: an A update improves A more than B, while a B update improves A more than B as well. All registered late endpoints170/330 follow A blocks. The diagnostic measures actual optimizer-mediated transfer conditioned on that history; it is not a momentum-free gradient or a pure readout of independent skills. We did not reset moments or add alternative diagnostic updates to obtain a desired pattern. At untrained0,19/20 models in **each** arm already satisfy the direct directional thresholds; that is another reason not to equate a one-update difference alone with learned splitting.

## 3. How consistently does functional splitting occur?

**0/20 per architecture at170 and330 pass the prespecified joint rule.** Direct evidence requires both directional own-minus-cross excesses>0, their mean>.001 nats, mean own-context gain>.0001, and neither own gain<−1e-6. Boundary evidence requires mean difficulty-cancelled cycle cost>.05 with at least half complete cycles>.05. Functional splitting at an endpoint is their conjunction. The thresholds were frozen in `scratchpad.md` and `audit/config.json` before training, not chosen for these counts. The prespecified3×3 amplitude sensitivity grid also yields0 positive cases in allcells, although other operational definitions could differ.

At170,2 Linear and9 Tanh models pass the boundary component; none passes the current direct component. Nineteen Linear models have own-gain too small/nonbeneficial for the direct test. This is **not evidence of shared learning**; it means a single late update carries little information about learning specificity. Tanh has measurable own-context learning at170, but directionally mixed rather than the required bidirectional excess. The secondary trajectory-persistence flag is also0 in both arms. First-block asymmetry, cycles, causal effects, stationary cases and full raw values all remain available rather than being erased by a binary label.

Reacquisition reports incoming CE gains after1/5/10 updates and first crossing of a fixed.05-nat gain. Floor-limited cells and end-of-run330 cells are explicitly censored, not assigned zero cost or zero reacquisition time:

{md(reacq)}

The count is boundary cells, not independent networks. Threshold crossings characterize the observed ten-update block; no claim is made about unobserved future recovery. See `tables/switch_costs.csv` and `tables/balanced_cycle_costs.csv` for every seed/direction/boundary.

## 4. Does Linear retain exact averaging despite splitting?

It retains exact averaging, but the phrase “despite splitting” is **not established under the primary functional criterion**. For any fixed linear recurrent weights and equal initial states, induction gives h_M,t=(h_A,t+h_B,t)/2, then z_M,t=(z_A,t+z_B,t)/2. Convexity of same-target CE in logits yields L_M≤(L_A+L_B)/2. This guarantee is independent of whether learning is context-specific, and does not imply arithmetic averaging of probabilities.

All40 trained endpoint controls pass. Maximum double hidden/logit errors are{linear.double_hidden_error.max():.3e}/{linear.double_logit_error.max():.3e}. Actual float32 maxima are{linear.float32_hidden_error.max():.3e}/{linear.float32_logit_error.max():.3e}, within frozen summand/propagation bounds. This is an algebraic control, not a statistical hypothesis test of exact linearity. There is no competent-and-split Linear subgroup available for the proposed strongest control.

## 5. Does Tanh have a gap when familiar acquisition and splitting are present?

Tanh has a positive gap with familiar acquisition, but there are **no joint strong-and-split cases** to establish the full conditional proposition. At170 the all-seed gap is.222074[.069694,.421477]. Among18 strong-familiar Tanh models it is.105914[.035999,.198568], an explicitly conditional selection. At330 all20 are strong and the gap is.140817[.028622,.329391]. Absolute Mixed CE is.430018 at170 and.184311 at330; relative impairment is not automatically catastrophic absolute failure.

{md(qualified)}

The paired all-seed Tanh−Linear gap at170 is{gapp['mean']:.6f}[{gapp.ci_low:.6f},{gapp.ci_high:.6f}]. Familiar performance also differs substantially: Tanh−Linear familiar CE≈.2074[.1145,.3448]. Common-strong paired estimates are supplied separately, but they are not exact performance matching and cannot substitute for missing common strong-and-split support. No failed acquisition case was removed from primary estimates. Final conditional verdict: `{verdict}`.

## 6. Learning, activation, recurrence and geometry: what explains the outcomes?

The structural explanation is firm for the averaging **possibility**: affine recurrence preserves logit interpolation, while tanh permits a positive gap. This does not uniquely explain the gap's magnitude, show that propagation rather than instantaneous nonlinear mismatch caused CE, or prove that the networks acquired independent context-specific skills. Architecture also changes optimization and learned representations. Actual Adam history affects the finite-update diagnostic, and familiar learning differs between arms.

Raw geometry is conspicuously insufficient. Linear hidden centroids remain about6.55 apart with position-matched separation≈6.85, yet late one-update effects are nearly stationary and the joint functional flag is0. Tanh geometry can also be separated without meeting the functional criterion. PCA is fit only on14 A/B states per model, jointly centered with deterministic component signs; seed0 is a fixed illustration, not selected for large separation. Every seed's projections and continuous geometry-to-transfer correlations are retained; correlations are descriptive, not causal mediation.

{md(geom)}

Tanh midpoint residuals and the imported D1 decomposition are position-resolved. D1 separates instantaneous j=tanh(mean familiar preactivation)−mean(tanh familiar preactivation) from r=tanh(mean preactivation+Wh e_previous)−tanh(mean preactivation), with exact ideal e=j+r and separately bounded observed rounding. At position1, e_previous=0, so r=0. Later propagation can modify the deviation. We reuse the existing implementation, not a new decomposition. Allnew Tanh source/adaptor linkages pass: maximum hidden/logit discrepancy{links.hidden_max_error.max():.3e}/{links.logit_max_error.max():.3e}, below predeclared2e-4/.002 bounds. D1 data come from this audited frozen NumPy adapter; primary outcomes remain the actual Torch float32 logits. Norms are not additive or percentages of CE explained.

## 7. How far does this explain the original canonical High-scale failure?

Only partially. This study restores current-token inputs and A10/B10 learning, unlike the previous explicit-history factorial. It demonstrates actual linear acquisition and validates the mathematical averaging guarantee. However, it does not recover the registered robust functional-splitting phenotype needed to isolate nonlinear failure of separately learned skills. Both new recurrent matrices start at spectral norm.9, rather than the original unnormalized High draw. Shared tensors are paired within the new arms, not claimed identical to historical checkpoint seeds. Canonical original models are a separate reference, not a third arm or an interchangeable control.

We therefore cannot conclude that nonlinear recurrence is the unique cause of the large canonical Mixed deficit. We also cannot conclude that the absence of a positive endpoint splitting flag proves absence of context-dependent learning. The supported result is narrower: early context-biased practice exists in both architectures; Linear learns and structurally averages; Tanh can have a positive gap; the critical joint phenotype remains unestablished. No new architectures, objectives, positional features, thresholds, diagnostics or hyperparameter searches were added in response.

## Secondary NoContext and Novel probes

`generalization.csv` retains every individual cue, allseven CEs/accuracies, absolute loss, uniform-relative loss and familiar-relative gap. NovelMean and30/60 means average variants within each trained model; see `tables/endpoint_summary.csv` and confidence tables. These protected results enter neither training nor checkpoint selection. They are secondary and not extrapolations of the special Mixed convexity guarantee.

## Five main figures

![Learning and switches](figures/fig_1_learning_switches.png)

Figure1: all20 A/B seed trajectories, mean/95% seed intervals, every executed context switch and the fixed170 endpoint. Loss is evaluated at frozen parameters before the next update.

![Learning-transfer diagnostics](figures/fig_2_learning_transfer.png)

Figure2: signed own/cross effects of one actual cloned A/B update, with allseeds. The optimizer is restored, not reset. Near-zero own gain is not automatically shared learning.

![Geometry and learning](figures/fig_3_geometry_and_transfer.png)

Figure3: familiar-only seed0 PCA illustrations and allseed centroid/logit-distance associations with direct specificity. Geometry never supplies the splitting flag.

![Protected generalization](figures/fig_4_mixed_generalization.png)

Figure4: familiar CE, absolute Mixed CE and gap, primary170 first and secondary330 below. All20 seeds and seed intervals remain.

![Averaging breakdown](figures/fig_5_midpoint_breakdown.png)

Figure5: source hidden/centered-logit residuals and imported D1 component norms. Double affine identities and float32 bounds are separate audits; component norms are not CE mediation.

## Execution, provenance and reproducibility

All40 runs completed330 scientific updates:13,200 total. All640 one-update diagnostic clones completed with scientific model/policy/Adam/RNG unchanged. All320 saved model+policy+optimizer checkpoint slots were sealed **before** protected evaluation. `learning_trajectories.csv` has26,480 A/B rows; `learning_transfer.csv`1280 signed diagnostic rows; `generalization.csv`640 endpoint/cue rows; transition predictions4480 rows. Failed/near-zero/mixed-sign findings and allflags remain. Original20 checkpoint hashes and prior source bytes are unchanged.

The deterministic capacity witness, source loss/gradient/clipping/Adam fixtures, causal input, neutral channel, identical initialization, no-probe-training guards, optimizer-moment restoration, unchanged trajectory, baseline cancellation, no future-window leakage, familiar-only PCA and seed-unit tests have RED→GREEN→REFACTOR logs. Reproducible configuration is `audit/config.json`; initialization/optimizer/checkpoint hashes are in `tables/` and `audit/training_seal.json`; exact sources in `audit/pretraining_gate.json`; source-to-claim map in `tables/claim_source_map.csv`. A Pandas future dtype warning from conditional missing-value summaries is retained in the log; it does not change stored values or seed coverage. No threshold was changed to increase the0/20 counts.

Read-only verification: `PYTHONPATH=.:src python -m experiments.recurrent_learning_transfer.verify`. Table-only figure regeneration: add `--regenerate /tmp/recurrent_learning_transfer_figures_new` with an empty destination. See the package README for stage order and the prohibition on replaying the registered scientific run. The final manifest binds every scientific output and current implementation. Stop after verification.
'''
 with (OUT/'RESULTS.md').open('x') as f:f.write(text)
 names=generate();dump(OUT/'audit/report_products.json',{'RESULTS.md':hash_file(OUT/'RESULTS.md'),**{'figures/'+n:hash_file(OUT/'figures'/n) for n in names}});append('## Findings and stopping interpretation\nAll40 trajectories completed. Primary endpoint joint functional-split evidence0/20 in BOTH arms; no posthoc threshold changes, resets or extra diagnostics. Early practice asymmetries positive, Linear19/20 near-stationary at170; Tanh170 directional excesses oppose (D_A positive,D_B negative). Linear acquisition/averaging demonstrated; Tanh competent-only gap positive, but critical strong+split comparison unestablished. Secondary330 kept separate. Five figures, allraw outcomes and source-to-claim map generated. Stop after independent verification, no further fitting.')
 print('Report and five figures complete; critical split phenotype not recovered',flush=True)

if __name__=='__main__':run()
