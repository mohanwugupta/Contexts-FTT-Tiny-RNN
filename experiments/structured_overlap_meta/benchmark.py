"""Preflight hardware benchmark, independent fixture library and explicit budgets."""
import io
from .common import *
from .generator import family,episode
from .runtime import restore,meta_episode

def main():
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'];t=time.perf_counter();state,opt,src=source(0);load_seconds=time.perf_counter()-t;t=time.perf_counter();reserved={tuple(p) for n in ['diagnostic_paths','pilot_paths'] for p in json.loads((OUT/f'registry/{n}.json').read_text())};io_seconds=time.perf_counter()-t;eps=[]
 for j,k in enumerate([0,2]):
  f=family(np.random.default_rng(np.random.SeedSequence([20261121,581971,j])),dict(X='A',Y='B',Q='C'),(0,1),j%2,5,reserved);eps.append(episode(f,k));reserved.update(tuple(p) for p in f['base']+f['alternatives'])
 dump(OUT/'registry/benchmark_fixture_episodes.json',eps);rows=[];answers=[];torch.use_deterministic_algorithms(True)
 for threads in [1,2]:
  torch.set_num_threads(threads);t=time.perf_counter();m,o=restore(state,opt);construct=time.perf_counter()-t;t=time.perf_counter()
  for _ in range(20):clone=copy.deepcopy(m)
  cloning=(time.perf_counter()-t)/20;t=time.perf_counter();records=[meta_episode(m,o,e) for e in eps];seconds=time.perf_counter()-t;t=time.perf_counter();buf=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),torch_rng=torch.get_rng_state()),buf);serialization=time.perf_counter()-t;t=time.perf_counter();h=atomic_save(OUT/f'audit/benchmark_cpu{threads}_fixture.pt',dict(model=m.state_dict(),optimizer=o.state_dict(),torch_rng=torch.get_rng_state()));disk=time.perf_counter()-t;answers.append((copy.deepcopy(m.state_dict()),records));rows.append(dict(backend='cpu',threads=threads,episodes=2,inner_updates=40,outer_updates=2,seconds=seconds,seconds_per_episode=seconds/2,construct_restore_seconds=construct,deepcopy_seconds=cloning,serialization_seconds=serialization,disk_atomic_write_seconds=disk,bytes=buf.tell(),fixture_checkpoint_sha=h))
 for i,r in enumerate(rows):
  r['bitwise_equivalent_to_cpu1']=all(torch.equal(v,answers[i][0][n]) for n,v in answers[0][0].items()) and all(np.array_equal(a[field],b[field]) for a,b in zip(answers[0][1],answers[i][1]) for field in ['early_gradient','familiar_gradient','novel_gradient','combined_gradient'])
 chosen=min([r for r in rows if r['bitwise_equivalent_to_cpu1']],key=lambda r:r['seconds']);result=dict(passed=True,hardware=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),python=sys.version,torch=torch.__version__,numpy=np.__version__,source_checkpoint_load_seconds=load_seconds,registry_data_loading_seconds=io_seconds,cpu_trials=rows,selected=chosen,mps_available=torch.backends.mps.is_available(),cuda_available=torch.cuda.is_available(),native_accelerator_supported=False,accelerator_note='Unchanged source constructs CPU tensors/NumPy records; no supported accelerator in this environment',fixture_inner_updates=122,fixture_outer_updates=6,fixture_breakdown=dict(batch_equivalence_inner=2,two_checkpoint_reference_inner=40,two_checkpoint_reference_outer=2,benchmark_inner=80,benchmark_outer=4),scientific_pilot_updates_planned=23760,scientific_meta_updates_planned=710400)
 dump(OUT/'audit/benchmark.json',result);append('## Benchmark and numerical reference complete\nBatched-versus-independentnativefixture2innersteps;two-checkpointFOMAMLreference40inner+2outer;CPU1/2trials80inner+4outer. Totalimplementationfixtures122inner+6outer. Source/checkpoint/registryI/O, construction, deepcopy, serialization andatomicdiskwritealltimedbeforepilot. Fastestbitwise-equivalentCPUtrialselected;nofloatprecisionchange,componentnormalizationorarchitecturechange. Benchmarkusesindependentfixedfamiliesexcludedfrompilot;theirpathsremainreservedforfuturemeta. Actualhardwaremeasurementsinbenchmark.json.')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
