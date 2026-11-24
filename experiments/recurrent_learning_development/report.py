"""Concise descriptive report and hashed claim ledger from saved analysis tables."""
import json,platform
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import ROOT,OUT,SOURCE,ARCHS,EPOCHS,sha

def run():
 read=lambda n:pd.read_csv(OUT/'tables'/n);ep=read('epoch_intervals.csv');directions=read('epoch_direction_intervals.csv');geom=read('geometry_intervals.csv')
 def val(arch,epoch,metric):return ep[(ep.architecture==arch)&(ep.epoch==epoch)&(ep.metric==metric)].iloc[0]
 def estimate(r):return f"{r['mean']:.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}]"
 rows=[]
 for a in ARCHS:
  for label,start,end in EPOCHS:
   s=val(a,label,'specificity');own=val(a,label,'own')['mean'];cross=val(a,label,'cross')['mean'];floor=val(a,label,'both_start_floor')['mean']
   rows.append(f'| {a} | {start}–{end} | {own:.6f} | {cross:.6f} | {estimate(s)} | {floor:.1%} |')
 summary='\n'.join(rows)
 report=f'''# Context-specific learning across development

**Both architectures show early preferential learning in the practiced context, followed by smaller absolute learning effects. The saved results do not establish a transition from separate skills to a shared representation, or persistent separate learning throughout training.** Linear becomes strongly floor-limited; Tanh retains some late context preference, mainly during A blocks.

## What was measured

All20 paired seeds per architecture and all33 A10/B10 blocks are included:1,320 network-block rows. For each block, practiced cue P and other cue Q,

\[
\Delta_P=L_P(s)-L_P(e),\quad\Delta_Q=L_Q(s)-L_Q(e),\quad S_{{block}}=\Delta_P-\Delta_Q.
\]

A and B use identical frozen model weights **at each endpoint**; the weights naturally change between block start s and end e. Negative gains and specificity are retained. A positive difference means preferential improvement, not evidence of two independent skills. Cross-context improvement is already substantial early on.

The table averages blocks within each seed before estimating cohort means and pointwise95% bootstrap intervals (10,000 whole-seed draws,RNG20260925). These are descriptive intervals, without multiple-comparison adjustment. Absolute specificity is not standardized for available loss reduction or different initial loss scales.

| Architecture | Updates | Own gain/block | Cross gain/block | Specificity [95% CI], nats/block | Both losses<.05 at block start |
| --- | --- | --- | --- | --- | --- |
{summary}

## Answers to the research questions

**Early context-specific learning? Yes, as a practice effect.** During updates0–40, both A and B block directions have positive mean specificity and seed intervals above zero. Linear A/B direction means are2.712/2.065 nats per block; Tanh means are.528/.334. These effects coexist with positive cross-context benefits. They establish preferential practiced-context improvement, not representational segregation.

**Does the evidence weaken? Yes, in absolute magnitude.** Mean block specificity falls from2.389 to approximately.000002 in Linear and from.431 to.003803 in Tanh. The paired within-seed late-minus-early changes are−2.388886 [−2.895212,−1.906130] and−.426966 [−.495206,−.368356], respectively. Shrinking effects alone do not establish increased sharing: overall own and cross learning also diminish.

**Does a performance floor explain the weakening? It limits the interpretation strongly for Linear, but cannot explain all Tanh weakening.** The .05-nat marker is inherited from the earlier reacquisition analysis; it is not a new splitting criterion or a numerical precision cutoff. A low starting CE bounds possible further positive improvement, though measurable small changes remain. Exact start/end losses, floor crossings and whole-block checks are saved. Linear is below the marker in both contexts for the entire170–330 period, all20 seeds. Low Linear gains therefore cannot distinguish shared from separate learning. Tanh has no both-start-floor cases through block120; the corresponding average is only2% during120–170 despite much smaller specificity. Floor limitations become more common later:46% of seed-blocks during170–330, and14/20 seeds at the start of the last block. Its attenuation starts before widespread floor effects. This descriptive timing does not identify its cause.

**Different developmental patterns? Yes.** Linear acquires familiar predictions quickly and its late changes become tiny. Tanh continues improving more gradually and retains a small positive pooled block effect. But late Tanh specificity is directionally asymmetric: A blocks average.007785 [.002928,.014083], B blocks−.000179 [−.003539,.002802]. A positive pooled mean does not demonstrate persistent bidirectional specificity.

**Persistent separate learning or transition to shared learning? Neither is established.** The original joint splitting criterion is unchanged and its archived result remains0/20 in each architecture at170 and330. This reanalysis adds block-level practice effects; it does not relabel them as criterion-passing splitting. Floor-limited, small-gain and discordant effects remain inconclusive about organization.

## Do the one-update diagnostics agree?

The existing model+policy+Adam-restored diagnostics at0,10,20,40,80,120,170,330 are shown separately from ten-update block gains. End-of-block practiced-direction signs generally agree: for example, first A-block end agreement is18/20 Linear and19/20 Tanh. But timing matters. At the start of the first B block (checkpoint10), Tanh's B diagnostic has mean specificity−.036203 while the subsequent full B block has+.359191;15/20 seeds have opposite signs. At the block end (20),19/20 agree. Linear's corresponding means are−.062483 and+2.619012, with9/20 opposite at the start. Instantaneous learning effects under the actual inherited optimizer state need not predict ten sequential updates. This is consistent with history dependence, but does not isolate an Adam-momentum mechanism. Untrained checkpoint0 already has positive directional effects; those alone cannot demonstrate acquired separate skills. All start/end matches and negative signs are retained in `diagnostic_agreement.csv` and the seed-level alignment table.

## What do switch costs and geometry add?

Saved adjusted switch cost is [incoming CE−outgoing CE] at a common weight state, minus the same-direction initial cue difference. If both current losses approach zero, this adjusted value can approach the negative initial difference. Its alternating offset is therefore not evidence of persistent switching difficulty. Figure1 includes raw costs as well; the original joint criterion's balanced-cycle correction remains unchanged. The boundary330 incoming-cue value is end-censored: no next block occurred.

Saved geometry exists only at170 and330. Linear hidden centroid distance remains6.554→6.548 and centered-logit distance8.578→8.546, despite tiny late learning effects; mean probability TV falls.000821→.000290. Tanh distances are1.659→1.561 and3.579→3.390, respectively. Distances depend on model scale. Their persistence does not establish persistent separate learning, and their reduction does not prove a shared representation. Figure3 leaves earlier geometry absent; no checkpoint was reevaluated or trajectory interpolated. Position-matched hidden separation, probability TV and descriptive associations are retained in tables.

## Figures and exact data

1. [Context-specific block learning, A/B losses and switch costs](figures/fig_1_context_specific_blocks.png) ([SVG](figures/fig_1_context_specific_blocks.svg)). Allseeds, intervals, switches and floor counts. Each block marker summarizes its preceding10 updates.
2. [Within-context versus cross-context learning](figures/fig_2_within_cross_learning.png) ([SVG](figures/fig_2_within_cross_learning.svg)). Full-block benefits and all saved one-update effects; their horizons differ.
3. [Learning specificity alongside separation](figures/fig_3_learning_and_separation.png) ([SVG](figures/fig_3_learning_and_separation.svg)). Directional diagnostics, sparse geometry and endpoint scatter; geometry supplies no learning classification.

The primary seed-level result is [block_learning.csv](tables/block_learning.csv), with exact A/B endpoint losses, signed gains and floor flags. Original measurements are retained in the three `original_learning_*`/`original_switch_costs` tables; source hashes and the untouched original splitting table are included. [Claim ledger](tables/claim_ledger.csv), append-only [scratchpad](scratchpad.md), test logs and the reproduction guide document provenance. This reanalysis loaded no checkpoints, evaluated no networks and performed no training. All source outputs remain sealed and unchanged.
'''
 with (OUT/'FINAL_LEARNING_DEVELOPMENT_REPORT.md').open('x') as f:f.write(report)
 claims=[('C1','Early preferential practiced-context learning in both architectures','epoch_direction_intervals.csv','Descriptive practice effect; not two distinct skills'),('C2','Absolute specificity decreases over time','paired_development_intervals.csv','Decline does not identify increased sharing'),('C3','Linear late learning is floor limited','floor_counts.csv','Marker bounds available improvement; not numerical identity'),('C4','Tanh attenuation precedes widespread floor effects','epoch_intervals.csv','Temporal description; not causal attribution'),('C5','Late Tanh pooled effect is directionally asymmetric','epoch_direction_intervals.csv','No persistent bidirectional-specificity claim'),('C6','Diagnostic agreement depends on start/end alignment','diagnostic_agreement.csv','One versus ten updates; inherited optimizer history'),('C7','Persistent geometry does not establish separate learning','geometry_intervals.csv','Sparse170/330; scale dependent'),('C8','Original joint splitting result unchanged','original_splitting_classification.csv','No rule/threshold changes; original0/20 counts')]
 pd.DataFrame([dict(claim_id=i,claim=c,table='tables/'+t,sha256=sha(OUT/'tables'/t),limitation=limit,source_manifest_sha=sha(SOURCE/'manifest.json')) for i,c,t,limit in claims]).to_csv(OUT/'tables/claim_ledger.csv',index=False)
 with (OUT/'audit/environment.json').open('x') as f:json.dump(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),f,indent=2)
 print('Concise report and8-claim ledger generated from saved tables')

if __name__=='__main__':run()
