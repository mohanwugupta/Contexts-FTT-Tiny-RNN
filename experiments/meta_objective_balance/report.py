"""Eight scientific questions, exact paired evidence and researcher teach-back."""
from .common import *
from .statistics import interval
from .protocol import ARMS,SCREEN,CONFIRM
QUESTIONS=['Does increasing the relative weight on context-dependent learning improve unseen conflict performance?','Does emphasizing conflict-critical positions improve performance beyond increasing overall conflict weight?','Can these improvements occur without sacrificing C generalization?','Does functional A/B-specific learning remain observable during target acquisition?','Are benefits consistent across confirmation seeds?','Do shared and conflict objectives generate competing parameter updates?','Which objective balance gives the strongest joint pattern?','What limitations remain relative to human behavior?']
PRIMARY=['gap','ce','post_acquisition_gap','conflict_accuracy','conflict_ce','D_A','D_B']
def primary(f,selected):return f[f.dose.eq(1920)&f.candidate.isin(selected)&f.metric.isin(PRIMARY)]
def interpret(lo,hi):return 'decrease' if hi<0 else ('increase' if lo>0 else 'inconclusive direction; not equivalence')
def fmt(x):return f'{float(x):.5f}' if pd.notna(x) else 'missing'
def polish(text):
 import re
 text=re.sub(r'(?<=[A-Za-z]),(?=\d)',', ',text);text=re.sub(r'(\d+/\d+),(?=\d)',r'\1, ',text)
 return re.sub(r'\b(All|all|is|in|at|source|Source|for|after|before|through|from|to|than|each|Every|every|of|both|Both|Adam|episode|update|Figure|seed|seeds|steps|updates|blocks|norm|clip|Total|total|receives|costs|step|counter|position|CE|RNG|about|and|or)(?=\d)',r'\1 ',text)

def ci(r):return f"{fmt(r['mean'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"
def md(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows)+'\n'
def screen_factorial(f):
 rows=[];metrics=['gap','conflict_accuracy','conflict_ce','D_A','D_B'];pairs=[('B','A','beta'),('D','C','beta'),('F','E','beta'),('C','A','lambda'),('E','C','lambda'),('D','B','lambda'),('F','D','lambda')]
 for m in metrics:
  p=f[f.dose.eq(480)&f.arm.isin(ARMS)].pivot(index='seed',columns='arm',values=m).reindex(SCREEN)
  for a,b,kind in pairs:
   for s,v in (p[a].astype(float)-p[b].astype(float)).items():rows.append(dict(seed=s,contrast=a+'-'+b,kind=kind,metric=m,effect=v))
 return pd.DataFrame(rows)

def scientific_takeaways():
 f=read('confirm_seed_outcomes');f=f[f.dose.eq(1920)];p=read('confirm_paired_intervals');p=p[p.dose.eq(1920)]
 def effect(a,m):return p[p.candidate.eq(a)&p.metric.eq(m)].iloc[0]
 rows=[]
 for a in ['A','C','E']:
  g=f[f.arm.eq(a)];rows.append(f"{a}: C CE {fmt(g.ce.mean())}, familiar CE {fmt(g.familiar_ce.mean())}, conflict accuracy {fmt(g.conflict_accuracy.mean())}")
 a=effect('A','conflict_accuracy');e=effect('E','conflict_accuracy');db=effect('E','D_B')
 return '''## Scientific answer

**No selected objective demonstrated the registered joint improvement over the matched balanced control.** The final paired conflict-accuracy estimates are close to zero: A−C '''+ci(a)+''', E−C '''+ci(e)+'''. Both intervals span zero; this is uncertain benefit, not equivalence or proof that weighting can never help. Increasing conflict weight to .75 did not produce a confirmed unseen-conflict advantage in this experiment.

All14 seeds in every arm learned familiar A/B strongly, while C transfer remained heterogeneous: '''+'; '.join(rows)+'''. Strict C success is10/14 in each arm; ordinary C success is13/14 for A and C,12/14 for E. Final C-gap effects are A−C '''+ci(effect('A','gap'))+''' and E−C '''+ci(effect('E','gap'))+'''. Neither establishes superiority or noninferiority. The .1-nat screening tolerance must not be reused as an unregistered confirmation equivalence margin.

E's initial B-block specificity has a positive pointwise candidate−control interval: '''+ci(db)+'''. Its first-A change remains uncertain. At the final dose, both first-A and first-B cohort intervals are positive for C and E; A's first-A interval spans zero. Every final-dose initial-block floor flag is false. The actual practiced and unpracticed starting losses are also all above the inherited .05 threshold (audit/initial_floor_margins.json), so this is not merely a both-context conjunction artifact. Measured early floors do not explain these differences. Late near-zero losses still constrain endpoint interpretation. Positive nonfloor learning in both initial blocks occurs in8/14 A,11/14 C and11/14 E seeds; the descriptive familiar/C/Mixed-both/early-learning conjunction occurs in7/14,9/14 and8/14, respectively. No arm passes the original functional endpoint gate at170 or330. No objective dominates across the registered outcomes.

No critical-position-weighted arm survived the preregistered selection rule. All three within-λ β=.5 screen means had lower conflict accuracy than β=0, but these are exploratory six-seed observations, not confirmation evidence against critical-position weighting in general.

Shared and conflict gradients oppose on about half the paired updates, but their mean cosine is near zero with intervals spanning zero in all arms. Clipping is frequent (about96% A,99% C and99.5% E). These diagnostics describe local update compatibility; they do not identify harmful interference as the causal mechanism of the behavioral tradeoff. Descriptively, conflict accuracy is lower than at the archived source960 initialization in all forks; that historical anchor does not replace the matched paired-update control.

'''

def methods():return '''## Actual computation and design

The starting point is each seed's original P2-independent Transfer FOMAML episode-960 model **and outer Adam moments/step counter**. All arms within a seed copy these identical states. The model has eight identity recurrent units, four current-token inputs, three context inputs, four output logits and 172 prediction parameters. The two-parameter execution policy participates only in inner adaptation and target learning. Each sequence forward starts at h_0=0; no hidden state carries across repetitions, tasks or support/query passes. The native input and recurrent biases remain separate trained tensors; their sum is b in the recurrent equation. The RNN container has a tanh configuration flag, but its forward is replaced by the verified manual identity recurrence; nn.RNN.forward is not executed.

For each paired update, the current initialization is copied twice. A shared-sequence clone practices for 20 updates under two contexts and is queried under the withheld third. A separate conflict clone practices for 20 updates with stable P2 sequence/context assignments. Both clones start at exactly the same immutable initialization; there is no intervening outer step. Inner Adam is fresh, learning rate 0.005, original CE + 0.2 execution loss with detached confidence, joint prediction/policy norm-1 clipping. No execution-policy parameter is meta-updated.

From one conflict query forward, compute the all-position mean CE and the mean CE over **all query cells sharing an identical complete input-token prefix but requiring different next-token targets under their correct contexts**. The registered two-path tasks share three symbols and diverge at the fourth, so this is prediction position 3, under both contexts. The implementation identifies ambiguous prefixes rather than hardcoding that index; tests cover multiple ambiguous groups. Original P2 assigned query cells are duplicated equally, preserving historical mean weighting.

`L_meta = (1−λ)L_share + λ[(1−β)L_all + βL_critical]`.

Query gradients are differentiated with respect to each adapted prediction model, detached and mapped by parameter name to the initialization. This is first-order MAML: no derivative through the 20 Adam steps. Raw gradients combine as `g=(1−λ)g_share+λ[(1−β)g_all+βg_critical]`; only **then** the combined gradient is clipped once to norm 1 and inherited outer Adam 0.001 takes one step. No component normalization, gradient projection, precision change, weight decay or recurrent renormalization is introduced. The shared graph is released before conflict adaptation; all-position and critical gradients reuse the same conflict logits.

This is a **controlled fork**, not an exact continuation of the historical one-episode/one-outer-step algorithm. Condition C (λ=.5, β=0) is the matched paired-update control. Historical +0 and 10k outcomes are descriptive anchors, never substitutes for C. Labels +480/+960/+1920 count **additional episodes**, not outer steps or cumulative FOMAML episodes. The source already had 960 FOMAML episodes following 1920 Reptile episodes. After +1920, the outer Adam counter is 1920 (=960 inherited +960 paired updates), while total FOMAML episode exposure is 2880.

Six screen seeds 0–5 receive all six conditions for 480 episodes/240 pairs. Selected objectives and C use only seeds 6–19 for 1920 episodes/960 pairs, starting again from their original source960 models. +960 is a fixed intermediate measurement; no second selection occurs. These seeds have historical results but were untouched by this new screening/selection intervention. Screen and confirmation are never pooled for confirmatory inference.

The previously sealed original-generator blocks 5–12 are reused with corrected diagnostic full eight-symbol paths reserved. Within each block, shared and conflict episodes are paired by their original within-kind encounter order, shared first. All conditions receive identical pairs within seed. Every block contains 120 pairs, each with one shared and one conflict task, exact A/B/C query roles and original context exposure balance. Screening uses blocks 5–6, confirmation 5–12. Target sequence rotations/reversals and diagnostic paths remain excluded. Common subsequences are allowed. New paired registries and selection rules were sealed before scientific training; no sequence was chosen by outcomes.

At each target evaluation, fresh policy and Adam learn inputs `[0,1,2,0,3,1,2]` and targets `[1,2,0,3,1,2,3]` for 330 A10/B10 updates. C/Mixed never receive target practice; the learned C input column is bitwise fixed with zero Adam moments throughout target training. All 331 model/policy/Adam states, six cues and seven-position logits/probabilities/hidden states are retained. A/B one-update diagnostics copy the complete restored Adam/policy state, then discard diagnostic clones. Actual practiced CE is measured **before** its training update, at the same weights as the matching frozen evaluation.

Frozen cues are A=(1,0,0), B=(0,1,0), Mixed-average=(.5,.5,0), Mixed-both=(1,1,0), C=(0,0,1), NoContext=(0,0,0). C is a previously experienced context associated here with a held-out target sequence, not an entirely new sensory cue.

## Measurement and inference

`G_C(t)=CE_C(t)−[CE_A(t)+CE_B(t)]/2`. Absolute C and familiar CE are separate. Strong acquisition requires each A/B CE≤.5 and accuracy≥6/7. Original C/Mixed success uses CE≤.5 and accuracy≥6/7; strict C descriptives additionally require CE≤.1, accuracy7/7 and gap≤.1. No criterion selects target checkpoints. Post-acquisition gap includes **every state from the first strong A/B pass through330**, retaining later deterioration; never-acquired cases remain missing.

A-block specificity is A CE decrease minus B CE decrease across frozen start/end weights; B reverses the subtraction. Preserve signed own/cross gains, actual starting losses, inherited numerical tolerance and floor flag (both starting familiar CE≤.05). A small late change at floor is not evidence of shared learning. First-block coexistence requires positive own gains and specificity in **both** initial blocks, outside the inherited floor flag, plus familiar/C/Mixed-both success. It is descriptive and separate from the original endpoint gate. That gate retains direct update requirements (both D_A,D_B>0, meanD>.001, mean own gain>.0001, minimum own gain>−.000001) and balanced-cycle switch requirements (mean adjusted cost>.05 and at least half cycles>.05), at170/330.

Matched acquisition uses unchanged mean-familiar CE bins2–4,1–2,.5–1,.1–.5, first eligible block overall and separate A/B strata. Missing eligibility is missing, not zero; conditional paired n may be much less than14. Matching is descriptive conditioning, not randomized mediation or evidence of equivalent learning.

Reserved conflict capability is measured after the original 20-step adaptation on12 corrected-v2 tasks, each with stable context assignments. Conflict-position accuracy/CE, appropriate responses under both contexts, own/cross block gains and correct/wrong cue effects are separate from target C transfer. Average tasks within each seed **before** inference. Correct native branches and +0 histories are reused; only missing wrong-cue frozen predictions are evaluated. Initial correct CE is recovered exactly from final CE plus all four saved block gains, avoiding duplicate native evaluation.

Confirmation resamples all14 complete paired seeds10,000 times with RNG20260925; the same indices are used across arms/outcomes. Paired differences precede aggregation. Mean and median effects, their95% pointwise intervals and individual signs are retained. Missing observations retain planned-seed denominators; one eligible seed cannot support a population claim. Intervals are not simultaneous or multiplicity-adjusted. No equivalence margin was registered. The .1-nat C tolerance is a **screening preference**, not a confirmation noninferiority test. Six-seed exploratory intervals cannot validate the selected conditions. No post hoc multi-outcome score is a statistical success criterion.

'''

def main():
 assert json.loads((OUT/'audit/final_tables_verified.json').read_text())['passed'];selected=json.loads((OUT/'audit/selection_seal.json').read_text())['selected'];f=read('confirm_seed_outcomes');s=read('confirm_outcome_summary');p=read('confirm_paired_intervals');pri=primary(p,selected);screen=read('screen_seed_outcomes');factor=screen_factorial(screen);write_frame(OUT/'tables/screen_factorial_seed_effects.csv',factor);fr=[]
 for key,g in factor.groupby(['kind','contrast','metric']):fr.append(dict(zip(['kind','contrast','metric'],key),**interval(g.set_index('seed').effect.reindex(SCREEN).to_numpy(float),SCREEN)))
 fact=pd.DataFrame(fr);write_frame(OUT/'tables/screen_factorial_intervals.csv',fact)
 text='# Efficient meta-learning of sharing and context-specific skills\n\n'+scientific_takeaways()
 text+='All six screening objectives and the frozen confirmation design were executed. The confirmation comparison is '+', '.join(selected)+' versus matched balanced C, on fourteen paired seeds. The tables below separate C transfer, conflict adaptation and functional acquisition; C transfer alone is not success.\n\n'
 text+=md(['Arm','λ','β','Additional episodes','C CE','Familiar CE','C gap','Conflict accuracy','Conflict CE','Strong A/B','Strict C','Early both','Descriptive joint','Original late gate'],[[r.arm,*ARMS[r.arm],d,fmt(g.ce.mean()),fmt(g.familiar_ce.mean()),fmt(g.gap.mean()),fmt(g.conflict_accuracy.mean()),fmt(g.conflict_ce.mean()),f"{int(g.strong_familiar.eq(1).sum())}/14",f"{int(g.strict_all.eq(1).sum())}/14",f"{int(g.nonfloor_both.sum())}/14",f"{int(g.joint.sum())}/14",f"{int(g.functional_split.sum())}/14"] for d in [960,1920] for a in selected+['C'] for g in [f[f.arm.eq(a)&f.dose.eq(d)]] for r in [g.iloc[0]]])
 text+='\nRegistered final candidate−control effects (negative favors candidate for losses; positive for accuracy/specificity):\n\n'+md(['Contrast','Measure','Paired mean [95% CI]','Paired median [95% CI]','Negative / positive signs','Paired n','Direction'],[[r.contrast,r.metric,ci(r),f"{fmt(r['median'])} [{fmt(r.median_ci_low)}, {fmt(r.median_ci_high)}]",f'{r.n_negative}/{r.n_positive}',int(r.n),interpret(r.ci_low,r.ci_high)] for _,r in pri.iterrows()])
 text+='\nAny negative/positive count includes arbitrarily small signed changes; it is not a practical-effect threshold.\n\n'
 for i,q in enumerate(QUESTIONS,1):
  text+=f'## {i}. {q}\n\n'
  if i==1:
   text+='The confirmation arms all have β=0: A weights conflict .25, C .50, E .75. Thus A−C and E−C isolate this relative weighting under the matched paired-update algorithm. Inspect accuracy and critical CE together; changes in overall CE can conceal errors at the ambiguous prefix.\n\n';x=pri[pri.metric.isin(['conflict_accuracy','conflict_ce'])];text+=md(['Contrast','Outcome','Paired change [95% CI]','Interpretation'],[[r.contrast,r.metric,ci(r),interpret(r.ci_low,r.ci_high)] for _,r in x.iterrows()])
  elif i==2:
   text+='No β=.5 arm was selected. Confirmation therefore cannot validate critical-position emphasis. The within-λ screen comparisons B−A, D−C and F−E remain exploratory six-seed observations; none is pooled with confirmation.\n\n';x=fact[fact.kind.eq('beta')];text+=md(['Screen contrast','Metric','Exploratory mean [95% CI]','n'],[[r.contrast,r.metric,ci(r),int(r.n)] for _,r in x.iterrows()])
  elif i==3:
   text+='Use C loss, familiar loss, signed C gap and inclusive post-acquisition gap jointly. A smaller gap from worse familiar performance is not successful transfer; a zero-spanning paired interval does not establish no sacrifice. Both registered new checkpoints remain visible.\n\n';x=p[p.metric.isin(['gap','ce','familiar_ce','post_acquisition_gap'])];text+=md(['Additional episodes','Contrast','Outcome','Paired change [95% CI]','n'],[[r.dose,r.contrast,r.metric,ci(r),int(r.n)] for _,r in x.iterrows()])
  elif i==4:
   text+='Initial A and B learning remain separate measurements. The observed first-block counts, floor flags and late gate must not be conflated. Geometry is not a skill criterion.\n\n';rows=[]
   for (a,d),g in f.groupby(['arm','dose']):
    if a=='Baseline':continue
    rows.append([a,d,fmt(g.D_A.mean()),fmt(g.D_B.mean()),int(g.floor_A.sum()),int(g.floor_B.sum()),int(g.nonfloor_both.sum()),int(g.functional_split.sum())])
   text+=md(['Arm','Episodes','Mean firstA D','Mean firstB D','A floors','B floors','Both positive nonfloor /14','Late gate /14'],rows)
   x=read('confirm_matched_intervals');x=x[x.dose.eq(1920)&x.stratum.eq('all')&x.metric.eq('specificity')];text+='\nFixed-bin conditional first-block comparisons (all raw own/cross, starting losses and A/B strata remain in tables):\n\n'+md(['Contrast','Familiar CE bin','Specificity effect [95% CI]','Eligible paired n /14'],[[r.contrast,r.performance_range,ci(r),int(r.n)] for _,r in x.iterrows()])
  elif i==5:
   text+='E has a positive B-specificity effect in12/14 seeds, but conflict-accuracy signs are mixed (A:5 positive/5 negative/4 ties; E:7 positive/6 negative/1 tie). These counts include any signed change, however small. The primary table retains paired means, medians, intervals and both sign counts. Large mean changes with smaller median changes indicate heterogeneous tails. Inspect every seed in confirm_seed_outcomes.csv and confirm_paired_seed_effects.csv; diagnostic transitions/tasks are not independent networks. No favorable seed or checkpoint was removed.\n'
  elif i==6:
   text+='The cosine is computed from the shared gradient and β-weighted conflict gradient at the **same** meta-initialization, before weighted combination/clipping. Negative cosine indicates opposing local parameter directions, not by itself a causal explanation of behavioral failure. Raw component norms remain in their natural loss units; larger critical gradients are not normalized away.\n\n';hist=read('confirm_meta_history');stats=[]
   for (seed,a),g in hist.groupby(['seed','arm']):stats.append(dict(seed=seed,arm=a,cosine=g.gradient_cosine.mean(),opposing_fraction=g.gradient_cosine.lt(0).mean(),clipping_fraction=g.clipped.astype(float).mean()))
   ss=pd.DataFrame(stats);write_frame(OUT/'tables/confirm_gradient_seed_summary.csv',ss);gr=[]
   for a,g in ss.groupby('arm'):
    for metric in ['cosine','opposing_fraction','clipping_fraction']:gr.append(dict(arm=a,metric=metric,**interval(g.set_index('seed')[metric].reindex(CONFIRM),CONFIRM)))
   gr=pd.DataFrame(gr);write_frame(OUT/'tables/confirm_gradient_summary.csv',gr);text+=md(['Arm','Diagnostic','Whole-seed mean [95% CI]','n'],[[r.arm,r.metric,ci(r),int(r.n)] for _,r in gr.iterrows()])
  elif i==7:
   text+='The balanced control has the largest observed descriptive joint count (9/14), followed by E (8/14) and A (7/14). These count differences do not establish an optimum, and the selected candidates did not demonstrate a conflict-learning advantage. The opening table is the multi-outcome answer. The descriptive joint flag requires strong familiar acquisition, C success, Mixed-both success and positive nonfloor first A/B learning in the same target-trained seed. It is not the original endpoint gate, nor a new statistical score. Reserved conflict capability remains a separate continuous outcome. A winner must be justified across these outcomes and their paired uncertainty; successful C predictions alone do not satisfy the PRD.\n'
  else:
   text+='These are small teacher-forced next-token Linear RNNs and a fixed synthetic task ecology. C is familiar from prior sequences, not an arbitrary new sensory context. Human motor behavior, natural cue retrieval, response times and persistent separate skills were not fitted or established. The six-seed screen is noisy; fourteen confirmation seeds leave uncertainty and multiple pointwise comparisons. Failed acquisition bins, near-zero late losses, transient signed effects and seed heterogeneity constrain interpretation. Neither centroid distances nor algebraic decompositions prove two separately learned skills.\n'
 text+='''\n## Linear computation and mechanism limits

At sequence position t=1…7, `K_t=W_o Σ(k=0…t−1) W_h^k`. With context input columns w_A,w_B,w_C and context-free logits z_0,

`u=K_t(w_A+w_B)/2`, `v=K_t(w_A−w_B)/2`,

`z_A=z_0+u+v`, `z_B=z_0+u−v`, `z_C=z_0+K_tw_C`.

Mixed-average gives z_0+u, Mixed-both z_0+2u. Only Mixed-average has the familiar convex cross-entropy bound. C can predict accurately without equaling the A/B midpoint. Baseline sequence CE, common/contrast/C response norms, centered context sensitivity, context-column norms, recurrent spectral norm and A/B/C correct-class margins are retained. Class-common offsets are removed for magnitude comparisons. Across-model norm/cosine changes do not establish causal mediation; no context-strength sweep, oracle optimization or architecture change was run.

'''
 x=read('confirm_mechanism_intervals');x=x[x.dose.eq(1920)&x['update'].eq(330)&x.metric.isin(['baseline_ce','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm','A_margin','B_margin','C_margin'])];text+=md(['Contrast','Mechanism','Paired change [95% CI]','n'],[[r.contrast,r.metric,ci(r),int(r.n)] for _,r in x.iterrows()])
 text+='\n'+methods()+'''## Compute, fidelity and retained exceptions

All78 meta runs consumed97,920 new episodes,1,958,400 inner adaptation updates and48,960 paired outer updates. There are120 new330-update targets (39,600 updates),1,440 restored-Adam target-clone updates and28,800 reserved-conflict adaptation updates. Total new scientific optimizer steps: **2,077,200**. Twenty original+0 target/conflict branches are reused. Implementation fixtures add240 inner+6 outer updates, separately counted. Wrong-cue prediction and saved-data verification add no optimization.

Independent two-graph reference and sequential accumulated-gradient calculation agree. All saved query gradients are independently checked by manual Linear BPTT; every outer update is checked from inherited moments and weighted gradients, without optimizer replay. Every new target state, C constancy/reset contract, position distribution/CE/margin and numerical decomposition is checked using the unchanged predecessor suites. Original sources and complete ancestor seals are hash-bound. All scientific runs completed without numerical failure or repeated optimization; the desired joint improvement was not established.

A screening adapter initially passed archived Baseline rows into the strict six-arm selector. It stopped before writing a decision or running confirmation. RED31/GREEN32 fixes only the input filter; scientific tables and selection thresholds remain unchanged. The saved-gradient verifier was drafted before its dedicated fixture, backed out before scientific verification and accepted only after RED15/GREEN16. The prelaunch benchmark compared CPU1/2, complete paired operations, construction/serialization and accelerator availability; separate I/O/clone timings were recorded in a read-only followup, not falsely labeled prelaunch. All failures, deviations and recovery logs remain in SCRATCHPAD.md/audit. No unfavorable outcome changed a parameter, budget or threshold.

## Figures and exact evidence

- [Compact exploratory screen](figures/screen_summary.png) and [frozen screening decision](SCREENING_DECISION.md).
- [Figure1: actual practice](figures/figure_1.png), [first100](figures/figure_1_first100.png), [first20](figures/figure_1_first20.png).
- [Figure2: six frozen cues](figures/figure_2.png), [first120](figures/figure_2_first120.png).
- [Figure3: C/conflict tradeoff](figures/figure_3.png).
- [Figure4: initial functional learning](figures/figure_4.png), [matched bins](figures/figure_4_matched_all.png), [switch costs](figures/figure_4_switch_costs.png), [restored Adam](figures/figure_4_restored_adam.png).
- [Figure5: paired meta dynamics](figures/figure_5.png).

Every plot has SVG and exact raw-seed/interval tables. Full position logits and models remain in source-indexed target/diagnostic shards. CLAIM_LEDGER.csv maps each claim to exact sources and limits. RESEARCHER_HANDOFF.md explains the algorithms/results; IMPLEMENTATION_AUDIT.md and REPRODUCIBILITY.md cover hashes, budgets and table-only regeneration. No predecessor sealed artifact was overwritten.
'''
 (OUT/'REPORT.md').write_text(polish(text));dump(OUT/'audit/report_complete.json',dict(passed=True,questions=8,primary_final_rows=len(pri),confirmation_n=14,screen_n=6,no_pooled_inference=True));append('## Scientific report complete\nAll8PRDquestionsansweredwithfinalprimarypairedmean/medianintervals,secondary960visible,exploratorybeta/lambdaseparated,allfunctional/floor/conflict/jointoutcomesandmechanismcausalitylimits. Actualalgorithms/budgets/deviationsrecorded.')
if __name__=='__main__':main()
