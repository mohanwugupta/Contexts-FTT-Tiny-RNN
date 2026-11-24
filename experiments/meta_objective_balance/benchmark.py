"""Hardware-specific fixed short clone benchmark, separate fixture budget."""
import io
from .common import *
from .runtime import restore,paired_update

def main():
 assert json.loads((OUT/'audit/registry_seal.json').read_text())['passed'];state,opt,_=source(0);pairs=json.loads((OUT/'registry/pairs_seed_0.json').read_text())['pairs'][:2];rows=[];answers=[]
 torch.use_deterministic_algorithms(True)
 for threads in [1,2]:
  torch.set_num_threads(threads);t=time.perf_counter();m,o=restore(state,opt);construct=time.perf_counter()-t;t=time.perf_counter();runs=[paired_update(m,o,p,.5,.5) for p in pairs];elapsed=time.perf_counter()-t;t=time.perf_counter();stream=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=o.state_dict()),stream);serialization=time.perf_counter()-t;answers.append((copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),runs));rows.append(dict(backend='cpu',threads=threads,pairs=2,inner_updates=80,outer_updates=2,seconds=elapsed,seconds_per_pair=elapsed/2,construction_seconds=construct,serialization_seconds=serialization,serialized_bytes=stream.tell()))
 for n,v in answers[0][0].items():assert torch.equal(v,answers[1][0][n]),'Thread count changes numerical computation'
 for x,y in zip(answers[0][2],answers[1][2]):np.testing.assert_array_equal(x['combined_gradient'],y['combined_gradient'])
 chosen=min(rows,key=lambda r:r['seconds']);mps=dict(available=bool(torch.backends.mps.is_available()),supported_by_unchanged_source=False,reason='Native source creates CPU tensors/NumPy query records; moving to accelerator requires altered device operations and separate equivalence. No precision/device substitution authorized by benchmark alone.')
 if mps['available']:
  try:
   m=source_model(state).to('mps');x,y=core.tensors();m(x,torch.tensor(CUES['A']))
  except Exception as e:mps['native_forward_failure']=type(e).__name__+': '+str(e)
  else:raise AssertionError('Unexpected native accelerator support: equivalence benchmark required before selection')
 result=dict(passed=True,hardware=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),python=sys.version,torch=torch.__version__,numpy=np.__version__,cpu_trials=rows,mps=mps,cuda_available=torch.cuda.is_available(),selected=chosen,threads_equivalent_bitwise=True,fixture_inner_updates=160,fixture_outer_updates=4,reference_fixture_inner_updates=80,reference_fixture_outer_updates=2,registered_scientific_pairs=48960,serial_compute_estimate_seconds=chosen['seconds_per_pair']*48960,parallel_wall_estimate_not_asserted=True)
 dump(OUT/'audit/benchmark.json',result);append('## Phase B benchmark/equivalence\nIndependent sequential versus two-graph reference passed (80 inner+2 outer fixture updates). CPU one/two-thread trials each used two pairs (160 inner+4 outer total); final parameters and combined gradients bitwise identical. Hardware measured timings in benchmark.json; choose fastest equivalent CPU setting, originalfloat32, native source operations. Accelerator capability/unsupported native-device behavior recorded. Total implementation fixture budget240 inner+6 outer; no hidden warmup optimization.')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
