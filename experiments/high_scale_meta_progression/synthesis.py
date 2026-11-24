"""Explicit interpretation of completed saved results; no new selection/gates."""
from .common import *
def main():
 if (OUT/'audit/synthesis_complete.json').exists():return
 read=lambda n:pd.read_csv(OUT/f'tables/{n}.csv',float_precision='round_trip');o=read('outcomes');g=read('functional_splitting');joint=read('joint_components');geom=read('meta_geometry');end=o[o['update'].eq(990)];rows=[]
 for key,z in end.groupby(['scale','phase','dose']):rows.append(dict(zip(['scale','phase','dose'],key),n=len(z),strong_familiar=int(z.strong_familiar.sum()),strict_C=int(z.strict_C.sum()),Mixed_average_success=int(z.Mixed_average_success.sum()),Mixed_average_CE=z.Mixed_average_ce.mean(),Mixed_both_success=int(z.Mixed_both_success.sum()),C_CE=z.C_ce.mean(),C_gap=z.gap.mean(),D_A=z.D_A.mean(),D_B=z.D_B.mean(),sustained_cost=z.sustained_cost.mean()))
 write_frame(OUT/'tables/observed_synthesis.csv',rows)
 q=joint[joint['update'].eq(990)&joint.dose.eq(1920)];co=q[q.strong_familiar.eq(1)&q.strict_C.eq(1)&q.Mixed_both_success.eq(1)&~q.floor_A&~q.floor_B&q.D_A.gt(0)&q.D_B.gt(0)&q.sustained_cost.gt(0)];columns=['seed','scale','phase','own_A','own_B','D_A','D_B','sustained_cost','A_to_B','B_to_A','C_ce','Mixed_both_ce','n_appropriate_tasks','later_headroom','positive_later_cycles'];write_frame(OUT/'tables/component_cooccurrence.csv',co[columns]);gate=g[g.functional_split];write_frame(OUT/'tables/original_gate_passes.csv',gate)
 # These are reporting statements about prespecified outcomes, not new eligibility rules.
 body='''# Observed scientific synthesis

**The full reliable joint phenotype is not established.** Meta-learning improves some transfer outcomes while leaving substantial early learning specificity in some conditions, but it does not produce uniform C/Mixed-both success or reliable positive switch costs in both directions.

Every registered990-update target trajectory reaches the inherited strong familiar A/B criterion. Fresh scales5/6 each improve from4/6 qualified at330 to6/6 by660; scale4 is6/6 throughout330/660/990. Thus the earlier330 failures at larger scales reflect slow acquisition in these six seeds, not permanent inability under extended practice. All990 results are close to the familiar loss floor, which limits evidence about persistent separate learning.

Mixed-average succeeds in all six seeds of every registered990 condition, consistent with the verified Linear midpoint/Jensen bound. Mixed-both is different: final FOMAML success is0–2/6 per condition. Improved familiar acquisition and Mixed-average interpolation therefore do not establish full-strength context composition or novel-C transfer.

Reptile markedly reduces mean C loss relative to Fresh at target990 (scale4:25.689→3.073; scale5:42.847→5.845; scale6:60.108→11.254 nats), while reducing both first-block specificity means and balanced/sustained early switch effects. Starting CE/headroom also changes, so this is not by itself evidence of representational sharing or causal mediation. Read the paired and performance-matched intervals alongside these means.

Among the fixed FOMAML conditions, scale4 O2 has the lowest mean final C CE,0.3165 nats, with strict C4/6 and Mixed-both2/6. Its paired C-gap change from Reptile is−2.7562 nats, pointwise95% CI[−6.4451,−0.1527]. First-A and first-B specificity changes are inconclusive (their intervals span zero). At this scale O1 also reaches strict C4/6, but Mixed-both1/6. These are partial transfer improvements, not a selected winning joint procedure. Across scales5/6, C success remains2–3/6 per final FOMAML condition. Mean and median effects remain separate in the full paired tables.

The original functional criterion passes only twice across all registered dose/endpoints: seed5/scale6/Reptile960 at170, and seed1/scale6/Transfer1920 at330. The latter is the only final-FOMAML330 pass, but its C CE is24.6115 and Mixed-both CE38.8959 at330; at990 these remain18.4407 and26.9866. It does not establish successful transfer alongside the original splitting gate. The original criterion has not been redefined at990.

The following cases co-occur on the prespecified early/nonfloor/sustained/familiar/C/Mixed-both components at990. This table is a descriptive intersection, not a new score, conflict-capability threshold, reliable bidirectional gate or selection decision. Every seed remains in joint_components.csv.

'''+co[columns].to_markdown(index=False,floatfmt='.5g')+'''

Both displayed cases have a negative early B→A directional mean. Their appropriate-conflict responses occur on6/12 and4/12 tasks, respectively; this does not establish uniformly successful conflict-dependent adaptation. Across final conditions, critical conflict accuracy means range approximately0.47–0.63, with all task averages and parent differences retained. There is no robust population-level joint conclusion from six seeds or these isolated cases.

Weight magnitude and effective computation both change. At scale4, Fresh→Reptile token RMS changes4.275→3.962 and output RMS4.186→3.775, while class-centered effective A/B logit contrast changes132.774→19.300. Recurrent spectral norm changes0.900→1.218. Later FOMAML contrasts can rise again while transfer changes. These unequal changes are consistent with changes beyond a single scalar temperature, but the study does not isolate norm versus orientation/recurrence as a causal mediator.

One strict forward-reference precision failure is retained (seed1/scale6/O2/+960 familiar query). It is consistent with the unchanged predecessor float32 propagation bound; the original gradient, CE and Adam checks pass. That separate certificate does not turn the strict forward failure into a pass. No threshold, model or prediction was changed.
'''
 (OUT/'OBSERVED_SYNTHESIS.md').write_text(body)
 for name in ['REPORT.md','RESEARCHER_HANDOFF.md']:
  with (OUT/name).open('a') as f:f.write('\n\n'+body)
 ledger=pd.read_csv(OUT/'CLAIM_LEDGER.csv');write_frame(OUT/'CLAIM_LEDGER_SYNTHESIS.csv',[dict(claim='Reliable joint phenotype not established',source='tables/observed_synthesis.csv;tables/component_cooccurrence.csv;tables/original_gate_passes.csv;tables/paired_intervals.csv;tables/conflict_seed.csv',limit='Six seeds; early co-occurrence is not reliable bidirectional splitting, persistent learning or uniform conflict success',status='qualified completed-data interpretation')]);dump(OUT/'audit/synthesis_complete.json',dict(passed=True,new_optimization=0,new_thresholds=0,selection=False,component_rows=len(co),original_gate_passes=len(gate)));append('## Scientific synthesis complete\nAllregistered990 branches familiar-qualified; Mixed-average universal butC/Mixed-both heterogeneous. Two isolated descriptive component intersections have negative B→A means and partial conflict capability. Original late gate passes only two registered dose/endpoints, including one finalT330 case with severe C/Mixed-both failure. No reliable joint phenotype or single winning procedure established. All numbers sourced to saved tables; strict precision failure retained.');print('Observed synthesis complete',flush=True)
if __name__=='__main__':main()
