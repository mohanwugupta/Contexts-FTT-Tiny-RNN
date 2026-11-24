from .common import *
def main():
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'];rows=[];states=[]
 for threads in [1,2]:
  initialize_process();torch.set_num_threads(threads);st,_=initial(0,'P2','I','Fresh',2);m=source_model(st);p=core.policy();o=core.optimizer(m,p);t=time.monotonic();r=core.step(m,p,o,'A');assert r['status']=='completed';core.trace(m,'A');elapsed=time.monotonic()-t;rows.append(dict(threads=threads,seconds=elapsed));states.append(copy.deepcopy(m.state_dict()))
 same=all(torch.equal(v,states[1][k]) for k,v in states[0].items());best=min(rows if same else rows[:1],key=lambda x:x['seconds']);dump(OUT/'audit/benchmark.json',dict(passed=True,threads=best['threads'],CPU1_CPU2_bitwise=same,trials=rows,hardware=platform.platform(),python=sys.version,torch=torch.__version__,fixture_optimizer_steps=2));append('## Benchmark complete\nCPU1/2 single native A-step fixtures compared,2 implementation optimizersteps separate. Selected fastest bitwise-equivalent tested CPU; native initializer bound to this exact configuration. Source scale2 history not retrained.')
if __name__=='__main__':main()
