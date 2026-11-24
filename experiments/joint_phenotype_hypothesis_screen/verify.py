"""Read-only arithmetic, native contracts, immutable selection and hash seal."""
from .common import *
from . import engine,rules,adapter
def raw(r):
 z=r['logits'];assert z.shape==(10,5,7,4) and np.isfinite(z).all()
 for i in range(10):
  for j in range(5):
   expected=core.score(z[i,j])
   for field in ['probabilities','ce','margin']:np.testing.assert_allclose(r[field][i,j],expected[field],atol=1e-12,rtol=1e-12)
   np.testing.assert_array_equal(r['accuracy'][i,j],expected['accuracy']);np.testing.assert_allclose(r['probabilities'][i,j].sum(-1),1.,atol=1e-12,rtol=0)
 return True
def check_exposure(x):
 for k,limit in CAPS.items():assert 0<=x[k]<=limit
 return True
def schedule(cues):assert cues==[cue(k) for k in range(1,331)];return True
def math_status(candidate,checks):
 assert all(c['passed'] for c in checks)
 if candidate=='B06':assert len(checks)==10 and all(c['midpoint_status']=='inapplicable_gated' for c in checks)
 elif candidate in rules.ORDER:assert len(checks)==10 and all(c['midpoint_status']=='applicable' for c in checks)
 return True
def output_set(folder,entries):assert {str(p.relative_to(folder)) for p in Path(folder).rglob('*') if p.is_file() and p.name!='manifest.json'}==set(entries),'Changed sealed output file set'
def hashes(root,entries):
 for n,h in entries.items():assert sha(Path(root)/n)==h,'Changed dependency '+n

def review():
 engine.require_identity();inspection=json.loads((OUT/'registry/inspection.json').read_text());hashes(ROOT,inspection['source_files']);hashes(OUT,inspection['bank_hashes']);e=engine.Exposure();check_exposure(e.counts());saved_exposure=json.loads((OUT/'registry/exposure.json').read_text());assert all(e.counts()[k]==saved_exposure[k] for k in CAPS)
 assert len(e.keys)==len(e.events)
 scalar_rows=0;positions=0;snapshots_checked=0;restored_background=0;newtargets=0
 from experiments.heldout_context_meta.target import validate_training_records
 for dest in sorted((OUT/'runs').iterdir()):
  c=engine.accepted(dest)
  if not c:continue
  r=dict(np.load(dest/'observations.npz'));raw(r);positions+=10*5*7;scalar_rows+=50
  f=pd.read_csv(dest/'metrics.csv',float_precision='round_trip');expected=pd.DataFrame(engine.metric_rows(r,c['candidate'],int(c['seed']),c['role']));assert f.shape==expected.shape
  for field in ['ce','gap','gain_from_zero','accuracy','margin']:np.testing.assert_array_equal(f[field].to_numpy(),expected[field].to_numpy())
  for field in ['correct_count','floor','useful','strict','historical_strict_gap_rule']:np.testing.assert_array_equal(f[field].to_numpy(),expected[field].to_numpy())
  checks=json.loads((dest/'mathematics.json').read_text());math_status(c['candidate'],checks)
  states=torch.load(dest/'checkpoints.pt',weights_only=False)
  if c['candidate']=='A06_static':continue
  t=pd.read_csv(dest/'practice.csv',float_precision='round_trip');schedule(t.training_cue.tolist());assert t.status.eq('completed').all();select=t[t['update'].isin([k+1 for k in TIMES if k<330])];validate_training_records(select,f)
  for n,st in states.items():
   assert torch.equal(st['model']['recurrent.weight_ih_l0'][:,6],states[0]['model']['recurrent.weight_ih_l0'][:,6]);assert st['update']==n
   if c['candidate']=='A01':assert torch.equal(st['model']['recurrent.weight_ih_l0'][:,4:],states[0]['model']['recurrent.weight_ih_l0'][:,4:])
   if c['candidate']=='A02':
    for key,v in st['model'].items():
     if not key.startswith('sequence_head'):assert torch.equal(v,states[0]['model'][key])
   if c['candidate']=='B06':assert torch.equal(st['model']['V'][:,2],states[0]['model']['V'][:,2])
   if 'variant' in st:
    before=rng();m,p,o=adapter.restore(st);assert digest(m.state_dict())==digest(st['model']) and digest(p.state_dict())==digest(st['policy']) and digest(o.state_dict())==digest(st['optimizer']);assert digest(rng())==digest(st['rng']);set_rng(before)
    assert not st['settings']['mask_cue'] and st['settings']['cue_basis'] is None
    adam=st['optimizer']['policy']['state'] if c['candidate']=='A04' else st['optimizer']['state']
    if c['candidate']=='A04':assert not st['optimizer']['prediction']['state']
    assert len(adam)==(0 if n==0 else (2 if c['candidate']=='A04' else (9 if c['candidate']=='B06' else 8)))
    assert all(float(v['step'])==n for v in adam.values())
   snapshots_checked+=1
  if not c['reused']:
   newtargets+=1;assert e.run_count(f'target/{c["candidate"]}_s{c["seed"]}')==330
   continuation=torch.load(dest/'progress.pt',weights_only=False);assert continuation['cursor']==330 and digest(continuation['state'])==digest(states[330])
   if c['candidate']=='A05':assert t.loc[t.first_moment_reset,'update'].tolist()==list(range(11,331,10))
   if c['candidate']=='A04':assert t.physical_optimizer_calls.eq(2).all()
 for dest in sorted((OUT/'background').iterdir()):
  result=json.loads((dest/'complete.json').read_text());assert sha(dest/'results.pt')==result['sha256'];records=torch.load(dest/'results.pt',weights_only=False);assert len(records)==6
  for record in records:
   assert len(record['records'])==20;before=rng();m,p,o=adapter.restore(record['state']);assert digest(o.state_dict())==digest(record['state']['optimizer']);set_rng(before);restored_background+=1
 for candidate in [f'B{i:02}' for i in range(1,7)]:
  for seed in [0,1]:
   d=OUT/f'runs/{candidate}_s{seed}';meta=json.loads((d/'meta_complete.json').read_text());assert meta['inner_updates']==4800 and e.run_count(f'meta/{candidate}_s{seed}')==4800
   t=pd.read_csv(d/'meta_updates.csv.gz');episodes=pd.read_csv(d/'meta_episodes.csv',float_precision='round_trip');assert len(t)==4800 and len(episodes)==(80 if candidate=='B04' else 240);assert (episodes.inner_updates==(60 if candidate=='B04' else 20)).all();np.testing.assert_array_equal(episodes.epsilon.to_numpy(),engine.epsilons(candidate))
   current=torch.load(d/'meta_progress.pt',weights_only=False);final=torch.load(d/'meta_complete.pt',weights_only=False);assert digest(current['meta'])==digest(final['model'])==episodes.iloc[-1].after_hash and current['cursor']==4800 and current['inner']==0
 selection=json.loads((OUT/'selection.json').read_text());s=pd.read_csv(OUT/'registry/pilot_summary_at_selection.csv',float_precision='round_trip');assert sha(OUT/'registry/pilot_summary_at_selection.csv')==selection['pilot_summary_sha'];assert sha(OUT/'registry/pilot_metrics_at_selection.csv')==selection['pilot_metrics_sha'];rankrows=[]
 for row in selection['rows']:
  if row['status']!='completed':continue
  candidate=row['id'];a=s[s.candidate.eq(candidate)&s.seed.isin([0,1])];b=s[s.candidate.eq(row['control'])&s.seed.isin([0,1])];static=s[s.candidate.eq('A06_static')&s.seed.isin([0,1])] if candidate=='A06' else None;decision=rules.route(a,b,candidate,static)
  assert all(row[k]==v for k,v in decision.items());rankrows.append(row|dict(retention=-np.inf if row['retention'] is None else row['retention']))
 assert rules.rank(rankrows)[:2]==selection['selected'];assert len(selection['selected'])<=2
 confirmation=json.loads((OUT/'audit/confirmation_complete.json').read_text());assert confirmation['selected']==selection['selected']
 if not selection['selected']:assert confirmation['rows']==[] and all(r['role']!='confirmation' for r in [json.loads(p.read_text()) for p in (OUT/'runs').glob('*/complete.json')])
 assert newtargets==24 and scalar_rows==1800 and positions==12600
 for name,limit in CAPS.items():assert e.counts()[name]<=limit
 return dict(passed=True,new_target_histories=newtargets,scalar_rows=scalar_rows,position_rows=positions,all_four_probabilities=True,named_complete_states_checked=snapshots_checked,background_optimizer_restores=restored_background,source_files_checked=len(inspection['source_files']),selection_unchanged=True,selected=selection['selected'],compute=e.counts(),optimizer_operations=0)

def seal():
 assert not (OUT/'manifest.json').exists();r=review();visual=json.loads((OUT/'audit/visual_review.json').read_text());assert visual['passed'] and visual['actual_visual_inspection'];assert set(visual['reviewed_pngs'])=={f'figure_{i}_{n}.png' for i,n in [(1,'screen'),(2,'behavior'),(3,'confirmation')]};assert visual['hashes']=={p.name:sha(p) for p in (OUT/'figures').iterdir() if p.suffix in ['.png','.svg']};assert len(visual['hashes'])==6
 assert json.loads((OUT/'audit/table_only_regeneration.json').read_text())['passed'];assert json.loads((OUT/'audit/task_and_initialization_audit.json').read_text())['passed'];assert json.loads((OUT/'audit/report_complete_v2.json').read_text())['report_sha']==sha(OUT/'REPORT.md')
 assert json.loads((OUT/'audit/test_evidence.json').read_text())['passed'];assert json.loads((OUT/'audit/A06_actual_identity.json').read_text())['passed']
 ledger=pd.read_csv(OUT/'CLAIM_LEDGER.csv')
 for row in ledger.itertuples():assert sha(OUT/row.source)==row.sha256
 put(OUT/'audit/final_validation.json',r)
 inputs=json.loads((OUT/'registry/inspection.json').read_text())['source_files'];code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/joint_phenotype_hypothesis_screen','tests/joint_phenotype_hypothesis_screen'] for p in (ROOT/folder).glob('*.py')};code.update({n:sha(ROOT/n) for n in ['configs/joint_phenotype_hypothesis_screen.yaml','docs/joint_phenotype_hypothesis_screen/PRD.md','experiments/joint_phenotype_hypothesis_screen/README.md']});outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};put(OUT/'manifest.json',dict(program='joint_phenotype_hypothesis_screen/v1',inputs=inputs,code=code,outputs=outputs,validation=r,scientific_updates_stopped=True));print(json.dumps(dict(sealed=True,outputs=len(outputs),inputs=len(inputs),manifest_sha=sha(OUT/'manifest.json'))),flush=True)
def main():
 configure();p=OUT/'manifest.json'
 if p.exists():
  m=json.loads(p.read_text());output_set(OUT,m['outputs']);hashes(OUT,m['outputs']);hashes(ROOT,m['code']);hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),inputs=len(m['inputs']),manifest_sha=sha(p),optimizer_operations=0)),flush=True)
 else:print(json.dumps(review()),flush=True)
