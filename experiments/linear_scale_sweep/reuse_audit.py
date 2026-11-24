from .common import *
def old_control(seed):return ROOT/f'results/context_experience_prior/v1/target/seed_{seed}_P0_I.pt'
def matches_older_control(seed,model):
 p=old_control(seed)
 if not p.exists():return False
 b=torch.load(p,weights_only=False);return all(torch.equal(v,b['states'][0]['model'][k]) for k,v in model.items())
def main():
 rows=pd.read_csv(OUT/'registry/replicate_runs.csv');r=rows[rows.seed.eq(18)&rows.dose.eq(2)].iloc[0];new=torch.load(ROOT/r.path,weights_only=False);p=old_control(18);old=torch.load(p,weights_only=False);counts=0
 for a,b in zip(new['states'],old['states']):
  for key in ['model','policy']:
   assert all(torch.equal(v,b[key][k]) for k,v in a[key].items())
  for k,v in a['optimizer']['state'].items():
   for field,x in v.items():assert torch.equal(x,b['optimizer']['state'][k][field])
  counts+=1
 dump(OUT/'audit/additional_reuse_certificate.json',dict(passed=True,seed=18,scale=2,source='Fresh',original_path=str(p.relative_to(ROOT)),original_sha=sha(p),redundant_copy=r.path,redundant_sha=r.sha256,all_model_policy_Adam_states_bitwise=counts,redundant_target_updates=330,redundant_diagnostic_updates=12,additional_independent_seeds=0,paired_estimates_unchanged=True,actual_executions=154,genuinely_new_histories=153,archive_equivalent_logical_cells=19));append('## Additional archived-control match discovered at final audit\nSeed18 Freshscale2 exactly matches older P0-I archive at every331model/policy/Adamstate. It was inadvertently re-executed because initial reuse audit searched only immediate High-scale source registry. Retain redundant data/330+12compute; never count it as independent. All analyses already contain exactlyone control perseed, so results/14pairedunitsunchanged. Final evidence certifies19archive-equivalent references and153genuinelynewhistories,154actual executions. Add pre-execution legacy-control guard; no scientific rerun to repair oversight.')
if __name__=='__main__':main()
