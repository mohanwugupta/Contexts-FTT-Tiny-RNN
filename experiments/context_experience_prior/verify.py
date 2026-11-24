"""Read-only full-grid, source, checkpoint and mathematical evidence audit."""
from .core import *
from .analysis import validate_grid

def verify_position_rows(f):
 z=f[[f'logit_{j}' for j in range(4)]].to_numpy(float);p=f[[f'probability_{j}' for j in range(4)]].to_numpy(float);y=f.target.to_numpy(int);q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True);np.testing.assert_allclose(p,q,rtol=1e-12,atol=1e-14);assert np.isfinite(z).all() and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-13);ce=z.max(1)+np.log(np.exp(z-z.max(1,keepdims=True)).sum(1))-z[np.arange(len(z)),y];np.testing.assert_allclose(f.ce,ce,rtol=1e-12,atol=1e-12);assert np.array_equal(f.correct.to_numpy(bool),z.argmax(1)==y);wrong=z.copy();wrong[np.arange(len(z)),y]=-np.inf;np.testing.assert_allclose(f.margin,z[np.arange(len(z)),y]-wrong.max(1),rtol=1e-12,atol=1e-12)

def verify_component_rows(f):
 for name in ['z0','u','v','C_response']:
  z=f[[f'{name}_raw_{j}' for j in range(4)]].to_numpy(float);c=f[[f'{name}_centered_{j}' for j in range(4)]].to_numpy(float);np.testing.assert_allclose(c,z-z.mean(1,keepdims=True),rtol=1e-12,atol=1e-12)

def main():
 seal=json.loads((OUT/'audit/protocol_seal.json').read_text());hashes(ROOT,seal['inputs']);hashes(OUT,seal['registry_hashes']);tgt=json.loads((OUT/'audit/target_complete.json').read_text());meta=json.loads((OUT/'audit/meta_complete.json').read_text());assert (meta['inner_updates_executed'],meta['outer_updates_executed'],tgt['new_target_steps'],tgt['diagnostic_steps'])==(384000,19200,26400,1440)
 f=pd.read_csv(OUT/'tables/evaluations.csv',float_precision='round_trip');validate_grid(f);assert f.status.eq('completed').all();dg=pd.read_csv(OUT/'tables/diagnostics.csv');assert len(dg)==2880 and dg.main_unchanged.all() and dg.optimizer_restored.all() and (dg.main_start_hash==dg.clone_start_hash).all() and dg.diagnostic_status.eq('completed').all()
 r=pd.read_csv(OUT/'tables/target_registry.csv');assert len(r)==120;checked=0;source_states=0
 for row in r.itertuples():
  pth=OUT/row.path;assert sha(pth)==row.sha256;b=torch.load(pth,weights_only=False);assert len(b['states'])==331 and b['target_policy_reset'] and b['target_optimizer_reset'];c=b['initial_C'];assert torch.count_nonzero(c)>0;zero=b['states'][0];assert not zero['optimizer']['state'] and all(torch.count_nonzero(v)==0 for v in zero['policy'].values())
  parent=torch.load(SOURCE/f'part_b/checkpoints/seed_{row.seed}_N{row.organization}.pt',weights_only=False) if row.prior=='P0' else None
  if row.prior!='P0':
   last=torch.load(OUT/f'meta/seed_{row.seed}_{row.prior}_{row.organization}.pt',weights_only=False)['records'][-1];assert all(torch.equal(zero['model'][n],v) for n,v in last['meta'].items())
  for k,st in enumerate(b['states']):
   assert st['update']==k and st['status']=='completed';assert torch.equal(st['model']['recurrent.weight_ih_l0'][:,6],c)
   for v in st['optimizer']['state'].values():
    assert int(v['step'])==k
    if tuple(v['exp_avg'].shape)==(8,7):assert torch.count_nonzero(v['exp_avg'][:,6])==torch.count_nonzero(v['exp_avg_sq'][:,6])==0
   if parent:
    src=parent['states'][k]
    for n,v in st['model'].items():
     a=src['model'][n];assert torch.equal(v[:,:6],a[:,:6]) if n=='recurrent.weight_ih_l0' else torch.equal(v,a)
    # Full policy/Adam structure and moments are compared recursively through hash.
    from experiments.nonzero_context_generalization.core import fingerprint
    assert fingerprint(st['policy'])==fingerprint(src['policy']) and fingerprint(st['optimizer'])==fingerprint(src['optimizer']);source_states+=1
   checked+=1
  print(f'Verified checkpoint {row.seed} {row.condition}',flush=True)
 n=0
 for x in pd.read_csv(OUT/'tables/position_predictions.csv.gz',chunksize=50000,float_precision='round_trip'):
  verify_position_rows(x);assert np.array_equal(x.token.to_numpy(),np.array(TARGET_PATH)[x.position.to_numpy()-1]);assert np.array_equal(x.target.to_numpy(),TARGET[x.position.to_numpy()-1]);n+=len(x)
 assert n==1668240
 dc=0
 for x in pd.read_csv(OUT/'tables/decomposition.csv.gz',chunksize=50000,float_precision='round_trip'):verify_component_rows(x);dc+=len(x)
 assert dc==278040
 checks=pd.read_csv(OUT/'tables/mathematical_checks.csv.gz');assert len(checks)==278040 and checks.passed.all();assert (checks.double_operator_logit_error<=checks.double_tolerance).all() and (checks.double_operator_hidden_error<=checks.double_tolerance).all();assert (checks.float32_logit_error<=checks.float32_logit_allowance+checks.double_tolerance).all();assert (checks.double_Jensen_gap<=2*checks.double_tolerance).all()
 tr=pd.read_csv(OUT/'tables/training.csv');assert len(tr)==39600 and tr.archive_reused.sum()==13200 and tr.new_optimizer_step_executed.sum()==26400 and tr.training_cue.isin(['A','B']).all();assert np.array_equal(tr.training_cue,np.where(((tr['update'].to_numpy()-1)//10)%2==0,'A','B'))
 eff=pd.read_csv(OUT/'tables/paired_seed_effects.csv')
 assert eff.groupby(['update','cue','metric','organization','contrast']).seed.apply(lambda x: sorted(x)==list(range(20))).all()
 hashes(ROOT,seal['inputs']);dump(OUT/'audit/verification.json',dict(passed=True,checkpoint_states=checked,P0_full_history_lift_states=source_states,position_rows=n,decomposition_rows=dc,mathematical_rows=len(checks),diagnostic_rows=len(dg),parent_unchanged=True,all_probabilities=True,no_training=True));append('## Read-only full audit GREEN\n39720model/policy/Adam states,13240P0fullhistory liftstates,1668240positionrows,278040decomposition/mathrows,2880diagnosticrows validated. Allfourprobabilities,CE,margins,token/targetindexing,nonzero/fixedC,Adamstepcounts,freshresets andsourcehashes pass. No evaluationreplay ortraining inverifier.')
if __name__=='__main__':main()
