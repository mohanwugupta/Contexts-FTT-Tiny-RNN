"""Registered ordered execution; no result-dependent procedure changes."""
from .common import *
from . import engine

def pilot(phase,deadline=None):
 configure();engine.require_identity();assert json.loads((OUT/'audit/preflight_complete.json').read_text())['passed'];receipt=OUT/f'audit/pilot_{phase}_complete.json'
 if receipt.exists():return json.loads(receipt.read_text())
 rows=[]
 if phase=='checkpoint':
  for seed in SEEDS:engine.materialize_control(seed,'F00')
  trigger=json.loads((OUT/'registry/A07_trigger.json').read_text())['triggered'];sgd=json.loads((OUT/'registry/sgd_calibration.json').read_text())['passed']
  for seed in [0,1]:
   engine.materialize_control(seed,'A00')
   for candidate in [f'A{i:02}' for i in range(1,8)]:
    if candidate=='A07' and not trigger:rows.append(dict(candidate=candidate,seed=seed,status='not_triggered'));continue
    if candidate=='A04' and not sgd:rows.append(dict(candidate=candidate,seed=seed,status='calibration_inconclusive'));continue
    rows.append(engine.target(seed,candidate,'pilot',deadline))
 elif phase=='pretraining':
  assert json.loads((OUT/'audit/pilot_checkpoint_complete.json').read_text())['passed']
  for seed in [0,1]:
   engine.materialize_control(seed,'B00');engine.background(seed,'B00',True,deadline);engine.background(seed,'B00',False,deadline)
   for candidate in [f'B{i:02}' for i in range(1,7)]:
    meta=engine.pretrain(seed,candidate,deadline)
    if meta['status']=='completed':
     rows.append(engine.target(seed,candidate,'pilot',deadline));engine.background(seed,candidate,False,deadline)
    else:rows.append(dict(candidate=candidate,seed=seed,status=meta['status']))
 else:raise ValueError('Two fixed pilot phases only')
 result=dict(passed=True,phase=phase,rows=rows,exposure=engine.Exposure().counts());put(receipt,result);append(f'## Pilot {phase} complete\nAll registered rows have explicit outcomes, including technical/conditional outcomes. No new threshold, horizon or hyperparameter. Exposure={result["exposure"]}. Fixed selection is permitted only after both pilot receipts.');return result

def confirm(deadline=None):
 configure();engine.require_identity();selection=json.loads((OUT/'selection.json').read_text());receipt=OUT/'audit/confirmation_complete.json'
 if receipt.exists():return json.loads(receipt.read_text())
 assert len(selection['selected'])<=2;rows=[]
 for candidate in selection['selected']:
  assert selection['frozen_runtime_sha']==sha(OUT/'registry/runtime_identity.json')
  for seed in [2,3,4,5]:
   control='A00' if candidate[0]=='A' else 'B00';engine.materialize_control(seed,control)
   if candidate[0]=='B':
    engine.background(seed,'B00',True,deadline);engine.background(seed,'B00',False,deadline);meta=engine.pretrain(seed,candidate,deadline)
    if meta['status']!='completed':rows.append(dict(candidate=candidate,seed=seed,status=meta['status']));continue
   rows.append(engine.target(seed,candidate,'confirmation',deadline))
   if candidate[0]=='B':engine.background(seed,candidate,False,deadline)
 result=dict(passed=True,selected=selection['selected'],rows=rows,status='no_candidates_qualified' if not rows else 'registered_confirmation_complete',exposure=engine.Exposure().counts());put(receipt,result);append(f'## Confirmation stopped at registered scope\nSelected={selection["selected"]}; primary reserved seeds2–5 only, unchanged procedures. No new selection or extension. Exposure={result["exposure"]}.');return result
