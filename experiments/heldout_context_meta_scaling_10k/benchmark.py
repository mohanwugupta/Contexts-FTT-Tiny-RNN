"""Prespecified cloned-state benchmark; never contributes scientific checkpoints."""
import tempfile
from .common import *
from .runtime import restore,compact_episode
from .protocol import reserve_block
from .storage import Journal

def same(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b

def main():
 initialize_process();state,opt,src=load_source(0,'P2','I','Transfer');reserved={tuple(p) for e in json.loads((SOURCE/'registry/heldout_v2_seed_0.json').read_text())['episodes'] for p in e['paths']};eps=reserve_block(0,5,set(),reserved)['episodes'][:2];timing={};final=[];rng=torch.get_rng_state().clone()
 for full in [True,False]:
  m,o=restore(state,opt);start=time.perf_counter()
  for ep in eps:r=compact_episode(m,o,ep,'P2','Transfer',full=full);assert r['status']=='completed'
  timing['full' if full else 'compact']=time.perf_counter()-start;final.append((copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),r['gradient'].copy()))
 same(final[0][0],final[1][0]);same(final[0][1],final[1][1]);assert np.array_equal(final[0][2],final[1][2]);assert torch.equal(rng,torch.get_rng_state())
 with tempfile.TemporaryDirectory(prefix='heldout10k_benchmark_') as d:
  j=Journal(Path(d)/'test.sqlite');j.initialize({'episode':960,'model':state,'optimizer':opt});start=time.perf_counter()
  for e in range(961,971):j.commit(dict(episode=e,model=final[1][0],optimizer=final[1][1],torch_rng=rng),dict(episode=e,status='storage_fixture_no_optimizer'),registered=e==970)
  storage_seconds=(time.perf_counter()-start)/10;j.close()
 chosen='compact' if timing['compact']<=timing['full'] else 'full'
 dump(OUT/'audit/benchmark.json',dict(passed=True,method=chosen,seconds_two_episodes=timing,atomic_commit_seconds=storage_seconds,bitwise_model_and_optimizer=True,bitwise_query_gradient=True,global_torch_rng_unchanged=True,fixture_inner_updates_this_benchmark=80,fixture_outer_updates_this_benchmark=4,fixture_inner_updates_total=180,fixture_outer_updates_total=9,workers=8,threads=1,rough_meta_wall_seconds=1446400*(timing[chosen]/2+storage_seconds)/8,source_sha=sha(src),scientific_training=False))
 append('## Clone benchmark and continuation GREEN\nSource episode vs compact observation adapter matches every model tensor, outerAdam moment/counter and rawquerygradient bitwise for successive episodes from actual960/P2-I/Transfer. Entireupdate leavesglobaltorchRNGunchanged; keyedtaskstreamsrecoverexactly. Full vs compact timings and durable-transactioncost recorded inbenchmark.json; choose faster observation mode, no scientificprecision/update changes. Test04 used100inner+5outer fixtureupdates; benchmark adds80inner+4outer, total180inner+9outer, separatefromscientificbudgets. Do not rerunoptimizationfixturesduringreadonlyfinalchecks.')
 print(json.dumps(json.loads((OUT/'audit/benchmark.json').read_text())),flush=True)
if __name__=='__main__':main()
