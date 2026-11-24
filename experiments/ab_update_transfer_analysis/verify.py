"""Read-only scientific controls and complete immutable source-chain verification."""
from .common import *
from . import design

def check_exposure(r):
 if 'new_native_diagnostic_steps' in r:assert r['new_native_diagnostic_steps']+r['new_zero_gradient_adam_steps']==r['new_diagnostic_control_steps']
 assert 0<=r['new_diagnostic_control_steps']<=20000 and 0<=r['new_reconstruction_steps']<=21000 and 0<=r['new_fixture_steps']<=64 and r['new_meta_steps']==0
 assert r['new_diagnostic_control_steps']-r['failed_or_uncertain_attempts']+r['archived_branch_optimizer_steps_reused']==r['logical_completed_branch_steps']
def check_effects(f):
 np.testing.assert_allclose(f.gain,f.pre_ce-f.post_ce,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(f.specificity,f.own_gain-f.cross_gain,atol=1e-12,rtol=1e-12)
def check_raw(a):
 from experiments.context_experience_prior.verify import verify_position_rows
 z=np.asarray(a['logits']);p=np.asarray(a['probabilities']);assert z.shape==p.shape and z.shape[1:]==(5,7,4)
 f=pd.DataFrame({'target':np.tile(core.TARGET,z.shape[0]*5),'ce':np.asarray(a['ce']).ravel(),'correct':np.asarray(a['accuracy']).ravel(),'margin':np.asarray(a['margin']).ravel(),**{f'logit_{j}':z.reshape(-1,4)[:,j] for j in range(4)},**{f'probability_{j}':p.reshape(-1,4)[:,j] for j in range(4)}});verify_position_rows(f)
 # Convexity plus exact logit residual: CE is2-Lipschitz in the infinity norm.
 residual=abs(z[:,4]-(z[:,0]+z[:,1])/2).max(-1);gap=np.asarray(a['ce'])[:,4]-(np.asarray(a['ce'])[:,0]+np.asarray(a['ce'])[:,1])/2;tol=1e-10+1e-12*abs(z).max();assert np.all(gap<=2*residual+2*tol)

def hashes(root,items):
 root=Path(root).resolve()
 for name,h in items.items():
  p=(root/name).resolve();assert p.is_relative_to(root) and p.is_file() and sha(p)==h,'Changed/missing dependency '+name

def visual(folder,review):
 folder=Path(folder);actual={p.name:sha(p) for p in folder.iterdir() if p.suffix in ['.png','.svg']};pngs={n for n in actual if n.endswith('.png')};assert review['passed'] and review['hashes']==actual
 assert set(review['reviewed_pngs'])==pngs and len(review['reviewed_pngs'])==len(pngs)
 full=review['full_resolution_pngs'];assert set(full).issubset(pngs) and {int(n.split('_')[1]) for n in full}==set(range(1,7))
 assert {n[:-4] for n in pngs}=={n[:-4] for n in actual if n.endswith('.svg')}

def review():
 e=pd.read_csv(OUT/'tables/update_effects.csv',float_precision='round_trip');check_effects(e);a=pd.read_csv(OUT/'tables/adam_controls.csv',float_precision='round_trip');np.testing.assert_allclose(a.native_minus_zero_gain,a.post_ce_zero-a.post_ce_native,atol=1e-12,rtol=1e-12);assert a.held_gain.eq(0).all() and (a.held_counter==a.target_update).all();x=pd.read_csv(OUT/'registry/exposure.csv').iloc[0].to_dict();check_exposure(x)
 p=pd.read_csv(OUT/'tables/parameter_interventions.csv',float_precision='round_trip');struct=p[p.structural_check.eq(True)];np.testing.assert_array_equal(struct.gain,np.zeros(len(struct)));g=p[p.probe_context.isin(['C','NoContext'])];keys=['trajectory_id','target_update','source_context','probe_context'];v=g[g.variant.isin(['full','shared_only'])].pivot(index=keys,columns='variant',values='post_ce');np.testing.assert_array_equal(v.full,v.shared_only)
 reg=pd.read_csv(OUT/'registry/branches.csv');raw_positions=0
 for row in reg[reg.status.eq('completed')].itertuples():
  path=OUT/row.result;raw=path.parent/'raw.npz';assert sha(path)==row.sha256 and sha(raw)==row.raw_sha256
  with np.load(raw) as a:check_raw(a);raw_positions+=int(a['logits'].shape[0])*5*7
 assert len(reg)==5148 and reg[['trajectory_id','target_update','source_context','branch_type']].duplicated().sum()==0;timing=pd.read_csv(OUT/'tables/update_audits.csv');assert len(timing)==x['logical_completed_branch_steps'];assert int((~timing.reused_optimizer_step).sum())==x['new_diagnostic_control_steps']-x['failed_or_uncertain_attempts']
 # Repeated source-labelled zero controls are measurements of the same history, not independent networks.
 zero=reg[reg.branch_type.eq('zero_gradient_adam')&reg.status.eq('completed')];zero_equal=0
 for _,rows in zero.groupby(['trajectory_id','target_update']):
  if len(rows)==2:
   b=[torch.load(OUT/r.result,weights_only=False)['steps'][0]['state'] for r in rows.itertuples()];assert digest({k:b[0][k] for k in ['model','policy','optimizer']})==digest({k:b[1][k] for k in ['model','policy','optimizer']});zero_equal+=1
 source_states=pd.read_csv(OUT/'registry/source_states.csv');assert len(source_states[source_states.cohort.eq('final')])==144*9;assert source_states.Adam_entries.eq(0).equals(source_states.target_update.eq(0))
 # Source precision failures remain separate, with no altered tolerance or repaired label.
 source_fail=pd.read_csv(SOURCE/'tables/strict_precision_exceptions.csv');unique=source_fail.drop_duplicates(['seed','scale','condition','arm','dose','part']);assert len(unique)==3 and not unique.strict_passed.any()
 result=dict(passed=True,all_four_probabilities=True,registered_branch_cells=len(reg),completed=int(reg.status.eq('completed').sum()),missing=int(reg.status.ne('completed').sum()),zero_A_B_equal_pairs=zero_equal,verified_raw_position_predictions=raw_positions,structural_zero_rows=len(struct),complete_source_state_rows=len(source_states),preserved_source_strict_failures=3,optimizer_operations=0,**x)
 path=OUT/'audit/validation.json'
 if path.exists():assert json.loads(path.read_text())==result
 else:dump(path,result)
 return result

def seal():
 assert not (OUT/'manifest.json').exists()
 for name in ['inspect_complete','preflight_review','analysis_complete','artifacts_complete','state_restore_verified','numeric_regeneration','report_complete','visual_review','figure_regeneration']:assert json.loads((OUT/f'audit/{name}.json').read_text())['passed'],name
 visual(OUT/'figures',json.loads((OUT/'audit/visual_review.json').read_text()));result=review();assert (OUT/'REPORT.md').exists() and (OUT/'RESEARCHER_HANDOFF.md').exists()
 ledger=pd.read_csv(OUT/'CLAIM_LEDGER.csv');assert not ledger.sha256.eq('pending_finalization').any()
 for row in ledger[['source','sha256']].drop_duplicates().itertuples():assert sha(OUT/row.source)==row.sha256
 from .engine import identity
 identity();parent=json.loads((SOURCE/'manifest.json').read_text());inputs=dict(parent['inputs']);inputs.update(parent['code']);inputs.update({str((SOURCE/n).relative_to(ROOT)):h for n,h in parent['outputs'].items()});inputs[str((SOURCE/'manifest.json').relative_to(ROOT))]=sha(SOURCE/'manifest.json');hashes(ROOT,inputs)
 code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/ab_update_transfer_analysis','tests/ab_update_transfer_analysis'] for p in (ROOT/folder).glob('*.py')};code.update({n:sha(ROOT/n) for n in ['docs/ab_update_transfer_analysis/PRD.md','configs/ab_update_transfer_analysis.yaml','experiments/ab_update_transfer_analysis/README.md']});outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'};dump(OUT/'manifest.json',dict(program=design.VERSION,source_manifest=sha(SOURCE/'manifest.json'),inputs=inputs,code=code,outputs=outputs,exposure=result,scientific_budgets_stopped=True));print(json.dumps(dict(sealed=True,outputs=len(outputs),inputs=len(inputs),manifest_sha=sha(OUT/'manifest.json'))),flush=True)

def check_output_set(folder,entries):
 folder=Path(folder);actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'};assert actual==set(entries),'Changed output file set'

def main():
 configure();path=OUT/'manifest.json'
 if path.exists():
  m=json.loads(path.read_text());check_output_set(OUT,m['outputs']);hashes(OUT,m['outputs']);hashes(ROOT,m['code']);hashes(ROOT,m['inputs']);print(json.dumps(dict(passed=True,outputs=len(m['outputs']),inputs=len(m['inputs']),manifest_sha=sha(path),optimizer_operations=0)),flush=True)
 else:print(json.dumps(review()),flush=True)
