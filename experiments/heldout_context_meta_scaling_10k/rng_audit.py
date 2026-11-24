"""Recover consumed PCG64 state from exact generator replay, with no model updates."""
from .common import *
from .protocol import source,reserve_block

def trace_block(seed,block,seen,reserved=None):
 original=source.draw;captures=[]
 def observed(rng,prefix=None,branch_not=None):
  p=original(rng,prefix,branch_not);captures.append((tuple(p),copy.deepcopy(rng.bit_generator.state)));return p
 try:
  source.draw=observed
  reg=source.make_block(seed,block,seen) if reserved is None else reserve_block(seed,block,seen,reserved)
 finally:source.draw=original
 usable=[(p,s) for p,s in captures if reserved is None or p not in reserved];cursor=0;states=[];rejected={}
 for e in reg['rejections']:rejected.setdefault(e['episode'],[]).append(e)
 for ep in reg['episodes']:
  for candidate in rejected.get(ep['episode'],[])+[ep]:
   n=len(candidate['paths']);assert [p for p,s in usable[cursor:cursor+n]]==[tuple(p) for p in candidate['paths']];cursor+=n
  states.append(usable[cursor-1][1])
 assert cursor==len(usable)
 return reg,states

def main():
 rows=[]
 for seed in range(20):
  old=ROOT/'results/context_experience_prior/v1';sc=ROOT/'results/meta_learning_scaling/v1';paths=[old/f'registry/seed_{seed}.json']+[sc/f'registry/seed_{seed}_block_{b}.json' for b in range(1,8)];seen=set()
  for p in paths:seen.update(source.signature(e) for e in json.loads(p.read_text())['episodes'])
  for b in range(1,4):seen.update(source.signature(e) for e in json.loads((SOURCE/f'registry/seed_{seed}_b{b}.json').read_text())['episodes'])
  reg,states=trace_block(seed,4,seen);frozen=json.loads((SOURCE/f'registry/seed_{seed}_b4.json').read_text());assert reg==frozen;rows.append(dict(seed=seed,episode=960,block=4,cursor=240,pcg64_consumption_state=states[-1]))
  seen.update(source.signature(e) for e in frozen['episodes'])
  for b in range(5,42):seen.update(source.signature(e) for e in json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text())['episodes'])
  reserved={tuple(p) for e in json.loads((OUT/f'registry/heldout_v2_seed_{seed}.json').read_text())['episodes'] for p in e['paths']};reg,states=trace_block(seed,42,seen,reserved);frozen=json.loads((OUT/f'registry/seed_{seed}_b42.json').read_text());assert reg==frozen and states[-1]==frozen['generation_rng_end'];rows.append(dict(seed=seed,episode=10000,block=42,cursor=160,pcg64_consumption_state=states[159],generation_rng_end=frozen['generation_rng_end'],unconsumed_preserved_slots=80))
 dump(OUT/'audit/task_rng_cursor_states.json',dict(passed=True,states=rows,original960_recovered_exactly=True,final10000_prefix_recovered_exactly=True,optimizer_updates=0,sampling_changed=False));print('Exact predecessor960 and final10000 task-RNG cursor states recovered for all20 seeds',flush=True)
if __name__=='__main__':main()
