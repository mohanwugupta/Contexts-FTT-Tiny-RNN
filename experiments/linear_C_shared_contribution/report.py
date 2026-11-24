"""Concise archived diagnostic report and fully expanded exact deliverables."""
from .core import *

def run():
 read=lambda n:pd.read_csv(OUT/'tables'/n,float_precision='round_trip');s=read('seed_values.csv');p=read('position_decomposition.csv');long=read('condition_predictions.csv');ci=read('intervals.csv');checks=read('decomposition_checks.csv');shares=read('cohort_loss_accounting.csv')
 # Deliver source A/B qualification and allthree requested condition means in one seed table.
 archived=pd.read_csv(SOURCE/'tables/seed_values.csv',float_precision='round_trip');fam=archived[(archived.architecture=='RNN-Linear')&archived.cue.isin(['A','B'])].pivot(index=['seed','checkpoint'],columns='cue',values=['ce','accuracy']);fam.columns=[f'{c}_{m}' for m,c in fam.columns];fam=fam.reset_index();full=s.merge(fam,on=['seed','checkpoint'],validate='one_to_one');full['familiar_mean_ce']=(full.A_ce+full.B_ce)/2
 retrieval=long[long.condition=='AB_logit_mean'].groupby(['seed','checkpoint'])[['ce','accuracy']].mean().reset_index().rename(columns={'ce':'AB_logit_mean_ce','accuracy':'AB_logit_mean_accuracy'});full=full.merge(retrieval,on=['seed','checkpoint'],validate='one_to_one');full['restored_minus_retrieval_ce']=full.ce_restored-full.AB_logit_mean_ce;full.to_csv(OUT/'tables/seed_summary.csv',index=False)
 expand=long.pivot(index=['seed','checkpoint','position'],columns='condition',values=[*[f'logit_{j}' for j in range(4)],*[f'probability_{j}' for j in range(4)]]);expand.columns=[f'{c}_{field}' for field,c in expand.columns];p.merge(expand.reset_index(),on=['seed','checkpoint','position'],validate='one_to_one').to_csv(OUT/'tables/position_decomposition_full.csv',index=False)
 def estimate(k,q,m):return ci[(ci.level=='seed')&(ci.checkpoint==k)&(ci.qualification==q)&(ci.metric==m)].iloc[0]
 rows=[]
 for k in [330,170]:
  for qualification in ['all','strong_familiar']:
   c=estimate(k,qualification,'ce_C');r=estimate(k,qualification,'ce_restored');ca=estimate(k,qualification,'C_accuracy');ra=estimate(k,qualification,'restored_accuracy');rows.append(f'| {k} | {qualification} | {c.n_valid}/20 | {c["mean"]:.6f} | {r["mean"]:.6f} | {100*ca["mean"]:.2f}% | {100*ra["mean"]:.2f}% |')
 qualified=p[(p.checkpoint==330)&p.strong_familiar];position_rows=[]
 for t,g in qualified.groupby('position'):
  position_rows.append(f'| {t} | {int(g.target.iloc[0])} | {g.ce_C.mean():.6f} | {g.ce_restored.mean():.6f} | {g.ce_gain.mean():.6f} | {int(g.wrong_to_correct.sum())}/14 |')
 last3=float(qualified[qualified.position>=5].ce_gain.sum()/qualified.ce_gain.sum());m=estimate(330,'strong_familiar','margin_loss_gain');o=estimate(330,'strong_familiar','other_alternatives_loss_gain');share=shares[(shares.checkpoint==330)&(shares.qualification=='strong_familiar')].iloc[0]
 report='''# What does the Linear RNN lose under C?

**C supplies no familiar context drive. Adding the common A/B logit contribution restores the A/B midpoint—and, in every familiar-qualified network, restores perfect predictions.** The missing component is the recurrent response to the average familiar cue input, not evidence of a naturally retrieved common sequence memory. This diagnostic uses archived predictions only: zero network forward calls, training, optimization or parameter changes.

## Results, preserving both checkpoints

Strong familiar is the unchanged CE≤.5 and accuracy≥6/7 **in each A/B context**. All20 seeds are retained; the14 qualified networks at330 and two at170 are explicitly conditional subsets.

| Update | Group | n | Natural C CE | C+u CE | C accuracy | C+u accuracy |
| --- | --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

Among the14 qualified330 networks, natural C CE4932.710171 [95% seed CI1080.247684,10590.176222] becomes.000994058 [.00040594,.00167807]. All33 incorrect positions out of98 become correct, with zero correct-to-incorrect changes. This is a14-network result, not98 independent observations. The two qualified170 networks similarly recover allfive incorrect positions, but n=2 gives a weak population basis. Across all20, some poorly acquired A/B networks remain inaccurate after restoration; the midpoint is not a universal rescue of unlearned sequences. Restoration also turns9 correct positions incorrect at170 and4 at330. These negative effects are retained; none occurs in the familiar-qualified subsets.

Allconfidence intervals resample trained seeds,10,000 draws,RNG20260925. Position-level intervals and full-precision values are saved; checkpoints are never pooled as independent networks.

## 1. What computational contribution is absent?

C=[0,0,1] and NoContext=[0,0,0] have the same effective input because every frozen third-context weight is exactly zero. Input projections match; with identical initial state and tokens, the numerical recurrence for their state/output differences remains zero at allseven positions in both float32 and float64. All40 checkpoint hashes and zero-channel proofs are recorded. Thus archived NoContext outputs are reused as natural C; no C trajectory was rerun.

For each four-logit vector, `z0=zC`, `u=(zA+zB)/2−z0`, `v=(zA−zB)/2`. Therefore `zA=z0+u+v`, `zB=z0+u−v`, and the restored `z0+u=(zA+zB)/2`. Algebraic restoration and equal-weight A/B retrieval have exactly identical stored logits and CE here: they are **one construction, not two independent rescues**.

The same u/v are independently recovered from the frozen weight operators. If wA and wB are the context columns and Wh/Wo are recurrent/output weights:

`U_t = Wh U_(t−1) + (wA+wB)/2`, `U_0=0`, `u_t=Wo U_t`.

The analogous V recurrence uses(wA−wB)/2 and gives v. In closed form, u_t is Wo times the sum of recurrent powers applied to the common cue vector. It is a **position-dependent, recurrently accumulated cue contribution**; at fixed weights its difference term does not depend directly on which tokens were presented. The baseline still contains the learned token/bias dynamics. It is not a separately trained, ideal sequence-only solution.

This exactly explains C-versus-Mixed at the output level. C omits both u and ±v; accurate Mixed does not need ±v. It does not establish that u is a dedicated shared-skill representation, or that C can naturally retrieve it. Here C's input channel is disconnected by construction.

## 2. Which positions depend most on u?

Qualified330 results, allseven retained:

| Position | Target | C CE | C+u CE | CE gain | Incorrect→correct |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(position_rows)+f'''

Positions7,6,5 have the largest mean CE repair; together they account for{100*last3:.3f}% of the qualified cohort's total CE gain. Position5 is corrected most often(8/14), whereas position7 is corrected in4/14. Large loss improvements late in the sequence reflect very confident errors in some networks, not equal dependence at every seed/position. Actual margin-change intervals can span zero where already-correct networks lose excess margin while remaining correct.

## 3. Does the restoration recover Mixed?

Every restored prediction matches archived native Mixed at all280 network-position cells. In the native-logit construction, C+u equals A/B retrieval exactly. Native float32 Mixed is **not** asserted bit-identical: the largest raw logit difference from reconstructed midpoint is{checks.native_Mixed_error.max():.6f}, and the largest position CE difference is{checks.restored_native_Mixed_ce_difference.abs().max():.6f}, within inherited propagated roundoff bounds. In qualified330 networks the maximum CE discrepancy is only2.56e−8. Cached float64 Mixed agrees within fixed1e−10+1e−12×scale tolerance; maximum residual is{checks.double_Mixed_error.max():.3e}. Operator-derived u/v errors are at most{max(checks.double_operator_u_error.max(),checks.double_operator_v_error.max()):.3e}. No new model evaluation or loosened tolerance was used.

## 4. How much improvement is associated with the correct-class margin?

For the actual strongest incorrect class, let m=zy−maxwrong. The exact four-class loss is `CE=softplus(−m)+r`, with0≤r≤ln3; r accounts for the other wrong alternatives. Consequently CE gain equals the decrease in that binary margin loss plus the decrease in r. This accounts for competitor switches and allfour logits.

At330 in the qualified14, mean CE gain is4932.709177 nats per sequence-position average. The margin-loss term contributes{m['mean']:.6f} and the other-alternatives correction contributes{o['mean']:.9f} [{o.ci_low:.9f},{o.ci_high:.9f}]. The ratio of cohort sums is{share.margin_share_percent:.9f}% for the margin term; no unstable individual gain ratios were averaged. This is exact **algebraic loss accounting**, not a unique causal-mediation percentage.

The signed vector evidence is more specific than its norm. Across the98 repeated network-position observations, centered u raises the target logit in61, suppresses C's strongest wrong class in74, and improves that fixed competitor margin in71. The actual strongest-wrong margin increases in68. Different wrong alternatives can become strongest; this switch penalty is explicitly subtracted rather than adding max-margins incorrectly. Allfour centered class contributions are in the decomposition CSV. Large centered vector norms alone do not establish helpful CE effects.

## 5. Scope of the conclusion

For adequately learned networks, the common familiar cue response is sufficient to recover the accurate midpoint predictions; all14 qualified330 and both qualified170 models do so. For the full cohort, residual acquisition failures remain. The output-level intervention uses externally supplied A/B trajectories and is mathematically tied to Mixed. It does not demonstrate natural retrieval, an autonomous shared sequence memory, or the mechanism of a future context-selection architecture. No such architecture or additional experiment was introduced.

## Deliverables and provenance

- [Figure1: natural C versus common restoration by position](figures/figure_1_C_shared_by_position.png) ([SVG](figures/figure_1_C_shared_by_position.svg))
- [Figure2: centered decomposition and signed margin contributions](figures/figure_2_logit_decomposition.png) ([SVG](figures/figure_2_logit_decomposition.svg))
- [Figure3: every network, familiar qualification highlighted](figures/figure_3_individual_rescue.png) ([SVG](figures/figure_3_individual_rescue.svg))
- [Seed-level CSV, including A/B acquisition and retrieval identity](tables/seed_summary.csv)
- [Full position-level decomposition, allfour logits/probabilities](tables/position_decomposition_full.csv)
- [Long condition predictions](tables/condition_predictions.csv), [exact intervals](tables/intervals.csv), [C equivalence](tables/C_NoContext_identity.csv), [identity/operator controls](tables/decomposition_checks.csv), [checkpoint hashes](tables/checkpoint_hashes.csv), [claim ledger](tables/claim_ledger.csv).

Sources are the sealed unnormalized high_scale_initialization/v1 checkpoints and high_scale_novel_contexts/v1 native predictions/double caches. Their hashes and allsource files are recorded in the analysis audit. All20 seeds, negative effects, positions and logits remain. The append-only scratchpad, tests, implementation guide and final manifest document reproduction without model replay.
'''
 with (OUT/'REPORT.md').open('x') as f:f.write(report)
 claims=[('C1','C is NoContext due exactlyzero thirdinputcolumn','C_NoContext_identity.csv','Sourceweight/zero-difference proof; nofreshCrollout'),('C2','Commonu equals externally suppliedABlogitmean','decomposition_checks.csv','One algebraicconstruction,notindependent rescues'),('C3','Allqualified330 predictions recovered','seed_summary.csv','n14,all20remain; notguaranteeforunlearnednetworks'),('C4','Positions7/6/5 dominatequalifiedCErepair','position_decomposition_full.csv','Lossmassnotuniformindividual dependence'),('C5','Binarytopmargin accountsforalmostallCEchange','cohort_loss_accounting.csv','Algebraicpartition,notuniquecausalmediation'),('C6','u changes target/wrongmargins,notjustnorms','position_decomposition.csv','Competitorheldfixedforcomponentaddition; switchpenaltyseparate'),('C7','u/v matchfrozen linearcontextoperators','decomposition_checks.csv','Positiondependentcue response,not establishedabstractsequenceknowledge'),('C8','NativeMixed predictionsmatchbutfloat32logitsnotbitexact','decomposition_checks.csv','Inheritedboundsandcachedfloat64identityseparate')]
 pd.DataFrame([dict(claim_id=i,claim=c,source='tables/'+n,sha256=sha(OUT/'tables'/n),limitation=l) for i,c,n,l in claims]).to_csv(OUT/'tables/claim_ledger.csv',index=False)
 append('## Scientific interpretation and full deliverables\nQualified330 n14:Caccuracy66.3%→100%,33/33 mistakes corrected,noharm;qualified170n2alsoall5corrected. All20 acquisitionfailuresretained. Latepositions7/6/5 dominateCEgain,position5mostoftencorrected. uisrecurrentlyaccumulatedcommoncueinputresponse andlogitmeanconstruction,notnaturallyretrievedskill. Marginlossidentityaccountsforgain,notnorm-basedcausalclaim. Fullseed/positionwideCSVs,report andclaimledgergenerated.');print('Report andfullseed/positionCSVs generated')

if __name__=='__main__':run()
