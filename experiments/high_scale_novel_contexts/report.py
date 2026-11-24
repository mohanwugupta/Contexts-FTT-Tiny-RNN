"""Saved-table scientific report and exact source-to-claim ledger."""
from .core import *

def run():
 read=lambda n:pd.read_csv(OUT/'tables'/n,float_precision='round_trip');ci=read('intervals.csv');counts=read('counts.csv');pc=read('common_strong330_paired_intervals.csv');double=read('linear_affine_double.csv');single=read('linear_affine_float32.csv');config=json.loads((OUT/'audit/config.json').read_text());n=config['common_strong330_n']
 exact=[]
 for key,g in ci.groupby(['architecture','checkpoint','cue','qualification']):
  a,k,c,q=key;r=dict(architecture=a,checkpoint=k,cue=c,qualification=q);r['n_valid']=int(g.n_valid.iloc[0]);r['n_planned']=20
  for x in g.itertuples():
   for label,column in [('mean','mean'),('ci_low','ci_low'),('ci_high','ci_high')]:r[x.metric+'_'+label]=getattr(x,column)
  ct=counts[(counts.architecture==a)&(counts.checkpoint==k)&(counts.cue==c)&(counts.qualification==q)].iloc[0];r.update(above_chance=int(ct.above_chance),good_cue_prediction=int(ct.good_cue_prediction));exact.append(r)
 pd.DataFrame(exact).to_csv(OUT/'tables/exact_summary.csv',index=False)
 def get(a,k,c,q,m):return ci[(ci.architecture==a)&(ci.checkpoint==k)&(ci.cue==c)&(ci.qualification==q)&(ci.metric==m)].iloc[0]
 overview=[]
 for k in STEPS:
  for a in ARCHS:
   all_=get(a,k,'NovelMean','all','ce');good=get(a,k,'NovelMean','strong_familiar','ce');acc=get(a,k,'NovelMean','strong_familiar','accuracy');overview.append(f'| {a} | {k} | {all_["mean"]:.6f} | {good.n_valid}/20 | {good["mean"]:.6f} | {100*acc["mean"]:.2f}% |')
 pair_rows=[]
 for c in ['Mixed','NoContext',*NOVEL]:
  l=get('RNN-Linear',330,c,'common_strong330','ce');t=get('RNN-Tanh',330,c,'common_strong330','ce');p=pc[(pc.cue==c)&(pc.metric=='gap_Tanh_minus_Linear')].iloc[0];pair_rows.append(f'| {c} | {l["mean"]:.6f} | {t["mean"]:.6f} | {p["mean"]:.6f} [{p.ci_low:.6f}, {p.ci_high:.6f}] |')
 report='''# Novel-context generalization in frozen High-scale RNNs

**Linear's successful Mixed interpolation does not extend to these registered Novel cues.** Its Novel cross-entropy is catastrophically larger than Tanh's, including among networks that learned both familiar contexts. Accuracy gives a different, inconclusive architecture comparison: Linear can make fewer but much more overconfident errors. Neither architecture exhibits reliable transfer across all four Novel cues.

## Full cohort and familiar-qualified results

All20 original paired seeds per architecture, updates170/330, all8 cues are retained. Mean CE is across7 teacher-forced next-token positions. NovelMean first averages the four registered variants **within each seed**. Strong familiar means CE≤.5 and accuracy≥6/7 in each A/B context separately; no Novel outcome enters qualification. Main allseed estimates retain20 networks. Conditional results have smaller stated denominators.

| Architecture | Update | All20 NovelMean CE | Strong familiar n | Qualified NovelMean CE | Qualified Novel accuracy |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(overview)+f'''

[Exact per-cue means,95% intervals,accuracy,gaps and chance counts](tables/exact_summary.csv), [all640 seed/cue cells](tables/seed_values.csv), [all4480 position predictions/logits](tables/position_predictions.csv). NoContext and all four Novel cues remain separate. Counts use exact ln4=1.3862943611198906, not rounded1.386.

## Paired comparison at330: both architectures learned A/B

The familiar-only intersection contains **{n} paired seeds**. The same seed IDs are compared, not the two independently filtered14-Linear/13-Tanh groups. Positive paired gap differences mean a larger Tanh penalty; negative means a larger Linear penalty.

| Cue | Linear mean CE | Tanh mean CE | Paired gap Tanh−Linear [95% CI] |
| --- | --- | --- | --- |
'''+ '\n'.join(pair_rows)+'''

Across allfour variants in these9 pairs, Linear CE6234.949726 versus Tanh1.798325; paired gap difference−6233.282905 [−15165.897447,−647.183877]. But Novel accuracy is62.70% versus55.56%; paired Tanh−Linear accuracy−7.14 percentage points [−21.03,+5.56]. Thus the data support much worse Linear **CE**, not established Tanh accuracy superiority. Linear's large activations/logits make mistakes highly confident; CE and proportion correct cannot be interchanged. Allmeans and paired estimates use10,000 whole-seed bootstrap resamples,RNG20260925; intervals are pointwise, descriptive and unadjusted for multiple comparisons.

## Answers to the six questions

1. **Does Linear generalize better?** Not by Novel CE. All20 means are orders of magnitude worse than Tanh at both checkpoints. Accuracy is not uniformly worse: at330 allseed Novel accuracy is51.07% Linear versus47.86% Tanh. Neither is reliably accurate across the Novel set.

2. **After adequate A/B acquisition?** The severe CE disadvantage remains in the separate strong-familiar subsets and the9 common pairs. In the14 qualified Linear330 seeds, Mixed CE is.000994 with100% accuracy, but NovelMean CE7578.354941. At60°, all14 exceed chance on **each** side. Poor familiar learning therefore cannot explain the entire Novel failure. Own-qualified Tanh330 NovelMean CE1.729670 is also above the uniform reference on average; less catastrophic loss is not broad transfer success.

3. **Does60° hurt more than30°?** Yes at the cohort level, preserving side and pairing. In the common9, the60−30 CE increases are5792.908 [376.138,14537.926] on Linear's A side and7284.306 [526.678,16435.631] on its B side. Tanh increases are.953555 [.224990,1.996228] and1.760227 [1.163474,2.338195]. Allseed170 contrasts are also positive with intervals above zero on both sides. This is a paired cohort tendency, not a claim that every individual seed worsens monotonically. Registered30→60 changes angular support and signed coordinates, not a semantic dimension.

4. **A-side versus B-side?** They differ, and pooling would conceal it. At330 in the common group Tanh's60° B-side penalty exceeds its A side by1.235125 [.225182,2.143710]; its30° side contrast.428453 [−.094679,.994539] is inconclusive. Linear side contrasts have wide intervals spanning zero at both angles. Allseed and own-qualified contrasts at both checkpoints remain in `novelty_side_intervals.csv`; no favorable side is selected.

5. **Only Mixed, or also extrapolation?** These learned Linear solutions succeed systematically at Mixed once familiar learning is adequate, but not at the registered extrapolations. No qualified Linear330 seed passes the same CE/accuracy standard on either60° cue. This does not prove that all possible Linear solutions must fail: the affine identity gives an interpolation bound, not an extrapolation guarantee.

6. **Human-like context dependence plus successful Novel transfer?** Neither cohort establishes that full pattern here. Zero seeds in either architecture at either checkpoint meet the fixed CE≤.5/accuracy≥6/7 standard on **allfour** Novel cues, even before requiring good A/B. At330 some qualified Tanh models pass individual30° cues(5/13 A-side,3/13 B-side), but none passes either60° cue under that standard. Source A/B separation and block learning effects are retained as context-dependence evidence; they are not independent-skill classifications. This probe task does not quantitatively fit human behavior or execution/RT transfer, so even a successful case would not by itself establish human replication.

## Why the affine identity does not guarantee Novel transfer

All cue vectors were imported verbatim from the frozen registry; none was constructed or renormalized. Each model's third-context input column is exactly zero. For Linear recurrence with fixed tokens and zero initial state, the state and logits are affine in the persistent context:

`z(c) = (1−c1−c2) z(0) + c1 z(A) + c2 z(B)`.

This is a **logit**, not probability, identity. The corresponding hidden-state relationship is also verified. Mixed has weights(0,.5,.5), so convex CE implies CE_Mixed≤mean(CE_A,CE_B). NoContext is the untrained origin, outside the trained A/B segment. Novel30_A has weights(.633975,.866025,−.5) on(0,A,B); Novel60_A has(1.366025,.5,−.866025). The B-side swaps the A/B coefficients. Each Novel has one negative coefficient and lies outside the trained angular sector and convex hull. These are off-segment extrapolations; they cannot be expressed as convex mixtures of trained A/B. Their unit norms do not restore a familiar-only loss bound. All lie in the original A/B plane; none introduces a new semantic cue dimension.

All320 Linear double-precision affine checks(40 model/states×8 cues) pass fixed1e−10+1e−12×scale tolerance. Maximum hidden/logit residuals are'''+f'{double.hidden_max_error.max():.3e}/{double.logit_max_error.max():.3e}'+'''. All2240 float32 position checks pass separately propagated64eps bounds with explicit cue-cast error. At the large unnormalized scales, the observed maximum float32 logit residual is'''+f'{single.logit_max_error.max():.6f}'+'''—so float32 identity is **not** claimed bit-exact. Primary predictions are actual native float32 rollouts, not affine reconstructions. Exact coefficients and errors are in `cue_audit.csv`, `linear_affine_double.csv` and `linear_affine_float32.csv`.

## Figures and provenance

1. [Eight contexts, both architectures and checkpoints](figures/figure_1_novel_context_generalization.png)
2. [Individual Novel gaps and common-pair comparison](figures/figure_2_novel_gap_paired.png)
3. [Novel CE versus familiar learning](figures/figure_3_novel_vs_familiar.png)

All have SVG counterparts. Labeled symlog axes preserve exact zeros, negative gaps and very large CE without excluding seeds.240 archived native A/B/Mixed cells and1680 positions were reused exactly;400 missing-context native rollouts and320 Linear double mathematical rollouts were needed. Older canonical NumPy Novel traces were not substituted for non-bit-equivalent native Torch outputs. All80 checkpoint hashes and model states remain unchanged; no optimizer, training, cue tuning, architecture change or checkpoint selection. [Claim ledger](tables/claim_ledger.csv), [append-only scratchpad](scratchpad.md), source audit/configuration and reproduction guide retain all tests, including a corrected binary-exact synthetic test fixture; no scientific threshold or measurement was changed.
'''
 with (OUT/'REPORT.md').open('x') as f:f.write(report)
 claims=[('C1','Linear NovelCE is far worse thanTanh, including commonstrong330','common_strong330_paired_intervals.csv','Accuracy advantage notestablished forTanh'),('C2','PoorNovel performance persists after familiaracquisition','intervals.csv','Conditionaln2/4 at170,14/13 at330; commonn9'),('C3','60degree novelty worsensCE onbothsides','novelty_side_intervals.csv','Cohortpairingtendency,not everyseed monotonic'),('C4','Aside/Bside asymmetries retained','novelty_side_intervals.csv','Not everycontrast excludeszero'),('C5','Mixedsuccess doesnot imply extrapolationtransfer','counts.csv','No general impossibilitytheorem forLinear'),('C6','NoallfourNovel successunderinheritedstandard','joint_counts.csv','NoquantitativehumanRTfit'),('C7','Signedaffinecontext relationship verified','linear_affine_double.csv','Real/double identity,not probabilityaffinity'),('C8','Sourcefloat32 affineerrors bounded','linear_affine_float32.csv','Largeabsolute errors retained; primaryactualnative predictions')]
 pd.DataFrame([dict(claim_id=i,claim=c,source='tables/'+n,sha256=sha(OUT/'tables'/n),limitation=l) for i,c,n,l in claims]).to_csv(OUT/'tables/claim_ledger.csv',index=False)
 append('## Report interpretation\nLinear winsMixed buthascatastrophicNovelCE evencommonstrongn9; do not claimTanh accuracy superiority(pairedaccuracyCIcrosseszero). Botharchitectures showlarger60-degree CE,sidecontrastsnotuniform,nonepassesall4Novel atfixedstandard. Affinelogit continuationverified320double/2240float32 checks; negativecoefficients removefamiliar-onlybound,not guaranteeextrapolation failureinallLinear solutions. Humanpatternnotquantitativelyestablished. Exactmeans/accuracy/gaps/counts andallpositions retained.');print('Report, exactsummary and8-claimledger complete')

if __name__=='__main__':run()
