"""Two fixed native fixture updates; fastest equivalent CPU configuration."""
from .common import *
from .protocol import scaled_initial
from .target import evaluate

def main():
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'];trials=[];states=[]
 for threads in [1,2]:
  initialize_process();torch.set_num_threads(threads);m=source_model(scaled_initial(0,5));p=core.policy();o=core.optimizer(m,p);t=time.monotonic();q=core.step(m,p,o,'A');dt=time.monotonic()-t;assert q['status']=='completed';rr=dict(evaluations=[],position_predictions=[],geometry=[],mathematical_checks=[],decomposition=[]);t=time.monotonic();evaluate(m,1,rr);ev=time.monotonic()-t;states.append(core.hash_state(m,p,o));trials.append(dict(threads=threads,step_seconds=dt,evaluation_seconds=ev,seconds=dt+ev))
 assert states[0]==states[1];best=min(trials,key=lambda r:r['seconds']);dump(OUT/'audit/benchmark.json',dict(passed=True,threads=best['threads'],trials=trials,bitwise_model_policy_Adam_parity=True,fixture_updates=2,hardware=platform.platform(),torch=torch.__version__));append('## Local benchmark complete\nTwo CPU1/2 one-A-step fixtures passed bitwise model/policy/Adam parity. Fastest measured step+full frozen six-cue geometry selected. Scientific target/metastates untouched. Total executed fixture optimizer steps285 (283 references+2 benchmark), separately charged.');print(best,flush=True)
if __name__=='__main__':main()
