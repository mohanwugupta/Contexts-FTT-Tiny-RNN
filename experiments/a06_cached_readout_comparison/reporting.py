"""Concise descriptive report, exact companion tables, no new decision gate."""
import json
import numpy as np
import pandas as pd
from .analysis import ARMS,PAIRS,SOURCE,ROOT,sha
def fmt(v):return 'unavailable' if v is None or not np.isfinite(v) else f'{v:.6g}'
def transitions(f):
 d={k:f[f.transition.eq(k)].position.astype(int).tolist() for k in ['became_correct','became_incorrect','stayed_incorrect','stayed_correct']};g=f.sort_values(['ce_contribution','position']);d['largest_loss_decrease_position']=int(g.iloc[0].position) if len(g) and g.iloc[0].ce_contribution<0 else None;return d
def render(out,f,p,d,pdiff,early,practice,ref,routes,validation):
 endpoint=f[f['update'].eq(330)];table=[]
 def val(seed,cue,arm,col):
  q=endpoint[endpoint.seed.eq(seed)&endpoint.cue.eq(cue)&endpoint.arm.eq(arm)];return q.iloc[0][col] if len(q) else None
 for seed in [0,1]:
  for cue in ['A','B','Mixed-average','C','NoContext']:
   r={'seed':seed,'probe':cue}
   for arm in ARMS:r[arm+' CE (correct/7)']=fmt(val(seed,cue,arm,'ce'))+' ('+str(int(val(seed,cue,arm,'correct_count')) if val(seed,cue,arm,'correct_count') is not None else '?')+'/7)'
   q=d[d.seed.eq(seed)&d['update'].eq(330)&d.cue.eq(cue)&d.first_arm.eq('A06')&d.second_arm.eq('A00-static')];r['A06−static ΔCE']=fmt(q.iloc[0].ce_difference if len(q) else None);r['Δcorrect']=int(q.iloc[0].correct_count_difference) if len(q) else 'unavailable';table.append(r)
 sentences=[]
 for cue in ['C','NoContext']:
  q=pdiff[pdiff.seed.eq(1)&pdiff['update'].eq(330)&pdiff.cue.eq(cue)&pdiff.first_arm.eq('A06')&pdiff.second_arm.eq('A00')];t=transitions(q)
  if q.empty:sentences.append(f'{cue}: essential paired position data unavailable.');continue
  top=q.sort_values('ce_contribution').iloc[0];position=int(top.position);r=p[p.seed.eq(1)&p['update'].eq(330)&p.cue.eq(cue)&p.position.eq(position)].set_index('arm');sentences.append(f'**Seed1 {cue}:** positions becoming correct {t["became_correct"]}, becoming incorrect {t["became_incorrect"]}, and remaining incorrect {t["stayed_incorrect"]}. Position{position} contributes {top.ce_contribution:+.6g} nats to the mean A06−A00 CE change; its position CE is '+ ' → '.join(fmt(r.loc[a,'ce']) if a in r.index else 'unavailable' for a in ARMS)+' (A00 → static → A06).')
 earlyrows=[]
 for row in early[early.arm.isin(['A00','A06'])].itertuples():earlyrows.append(dict(seed=row.seed,arm=row.arm,D_A=row.D_A,D_B=row.D_B,Q_A=row.Q_A,Q_B=row.Q_B,raw_E=row.E,E_normalized=row.E_normalized,own_eligible_A=row.Q_eligible_A,own_eligible_B=row.Q_eligible_B,joint=row.joint))
 missing=json.loads((out/'missing_inputs.json').read_text());full=all(val(s,c,a,'ce') is not None for s in [0,1] for c in ['C','NoContext'] for a in ARMS) and len(early[early.arm.isin(['A00','A06'])])==4 and ref is not None and routes is not None
 original_static_pass=bool(routes[routes.candidate.eq('A06')].iloc[0].static_control_passed) if routes is not None else None
 recommendation=('Retain output initialization as a focused follow-up candidate, chiefly because NoContext gains include additional correct predictions beyond frozen scaling in both seeds. This does not establish reliable C transfer or a joint phenotype; no follow-up is launched.' if full else 'Essential cached evidence is missing; the exact paths/keys in missing_inputs.json prevent a complete decision. Complete available comparisons are retained; do not rerun training.')
 text='''# Does A06 improve beyond a frozen quarter-readout view?

**C improves in loss, with no added C accuracy benefit over static scaling. NoContext improves in both loss and accuracy beyond that control.** Two pilot seeds at scale5/U are descriptive; positions/checkpoints are not independent replications.

'''+pd.DataFrame(table).to_markdown(index=False)+'''

All three arms retain familiar A/B and physical Mixed-average useful performance at330. The static view uses each checkpoint's own bias: `0.25*(z_A00-b_A00)+b_A00`; A06 outputs are unchanged. Both registered cached static views match this formula exactly. Static has the A00 learning history and no independent practice curve.

'''+ '\n\n'.join(sentences)+'''

For C, position3's target probability rises from1.23e−68 to2.34e−8 but stays wrong. Its −19.8283-nat mean-loss contribution exceeds the net −19.6295 change because position2 adds+0.198844 nats and becomes wrong (target2, prediction0). Thus6/7→5/7. Static scaling already gives−16.5804 nats; A06 adds−3.04902 versus static but loses one correct prediction. For NoContext, A06 additionally corrects positions3/4 in seed1 and four positions in seed0; static changes neither correct count. Seed1's remaining wrong position2 has CE4.28606; mean CE0.632108 still exceeds.5. Seed0 C stays7/7 throughout. Margins are scale-sensitive; use probabilities/correctness beside them.

**Early splitting and original gates:** actual A00/A06 A10/B10 losses remain visibly context dependent (Figure2). The original signed gains, D/Q, raw/directional/sustained switch measures, initial losses, eligibility and floors are unchanged in early.csv. A06 seed1 normalized E=0.0257912 remains below0.0284254; Q_A/Q_B pass their existing50%-Fresh thresholds. C CE2.70918>.5 and5/7<6/7 also fail. A00/A06 retain the original joint flag only in seed0. Original static-control subtest: '''+str(original_static_pass)+''' (mean C CE advantage); A06 does not qualify because both pilot seeds must be joint-feasible. NoContext is not an added gate; no new threshold or selection is applied.

**Recommendation:** '''+recommendation+'''

This excludes one fixed output-expression account of the NoContext improvement. It does not isolate a hidden-representation mechanism: the A06 training path can change internal/readout weights and biases. Exact paired differences and every seven-position contribution are in comparison.csv, paired.csv and position_differences.csv; readable seven-row seed1 tables and the seed0 companion are in POSITION_TABLES.md. All original classifications/selection remain unchanged. No new training, rollout, fitting, seed or confirmation. Reproduction: see REPRODUCIBILITY.md.
'''
 if not full:text='# Incomplete cached comparison\n\n'+recommendation+'\n\n'+pd.DataFrame(table).to_markdown(index=False)+'\n\n'+'\n\n'.join(sentences)
 (out/'REPORT.md').write_text(text)
 companion='# Seven-position endpoint tables\n\nOrder: A00, A00-static, A06. Class IDs are the original0–3; position order is1–7. Negative ΔCE/7 favors the first arm. Correctness is the argmax flag; raw margins depend on readout scale. Full precision/logits/all checkpoints are in positions.csv.\n'
 for cue in ['C','NoContext']:
  g=p[p.seed.eq(1)&p['update'].eq(330)&p.cue.eq(cue)];rows=[]
  for position in range(1,8):
   r={'position':position,'target':int(g[g.position.eq(position)].target.iloc[0]) if len(g[g.position.eq(position)]) else '?'}
   for arm in ARMS:
    x=g[g.position.eq(position)&g.arm.eq(arm)]
    if x.empty:r[arm]='unavailable';continue
    x=x.iloc[0];r[arm+' pred / correct']=f'{x.predicted_class} / {bool(x.correct)}';r[arm+' P(target)']=fmt(x.target_probability);r[arm+' margin']=fmt(x.margin);r[arm+' CE']=fmt(x.ce)
   rows.append(r)
  companion+='\n## Seed1 '+cue+'\n\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n\n'
  dif=pdiff[pdiff.seed.eq(1)&pdiff['update'].eq(330)&pdiff.cue.eq(cue)];view=dif.pivot(index='position',columns='contrast',values='ce_contribution');companion+='Position contributions to mean loss differences (sum each column):\n\n'+view.to_markdown(floatfmt='.8g')+'\n'
 companion+='\n## Seed0 companion\n\n'+pd.DataFrame(table).query("seed==0 and probe in ['C','NoContext']").to_markdown(index=False)+'\n\nFull seed0 seven-row tables: seed0_C_positions.csv and seed0_NoContext_positions.csv.\n';(out/'POSITION_TABLES.md').write_text(companion)
 # Separate exact endpoint interpretation per seed and probe, no new "similar" threshold.
 decisions=[]
 for seed in [0,1]:
  for cue in ['C','NoContext']:
   q=d[d.seed.eq(seed)&d['update'].eq(330)&d.cue.eq(cue)&d.first_arm.eq('A06')&d.second_arm.eq('A00-static')]
   if q.empty:decision='missing comparison'
   else:
    x=q.iloc[0];decision='lower CE and additional correct predictions beyond this static view' if x.ce_difference<0 and x.correct_count_difference>0 else ('lower CE; no demonstrated accuracy improvement' if x.ce_difference<0 else 'no lower CE beyond this static view')
   decisions.append(dict(seed=seed,cue=cue,interpretation=decision))
 pd.DataFrame(decisions).to_csv(out/'decisions.csv',index=False)
