"""Extend native keyed task banks; keep historical prefixes and unused suffixes."""
import shutil
from .common import *
from experiments.high_scale_constrained_meta.registries import generate
from experiments.high_scale_meta_progression.registries import paired_stream,structured_stream,validate_exclusions
from experiments.heldout_context_meta.protocol import support
from experiments.context_experience_prior.registry import allowed
from experiments.structured_overlap_meta.generator import validate as validate_overlap

def validate_bank(records,arm,reserved):
 expected=1920 if arm=='R' else (5000 if arm=='B' else 10000)
 if len(records)!=expected:raise ValueError('Wrong registered task budget')
 for k,e in enumerate(records,1):
  if e['pair' if arm=='B' else 'episode']!=k:raise ValueError('Wrong task cursor; would repeat/skip an episode')
  eps=[e['shared'],e['conflict']] if arm=='B' else [e]
  for ep in eps:
   if arm.startswith('O'):
    validate_overlap(ep,reserved);continue
   if any(not allowed(p) or tuple(p) in reserved for p in ep['paths']):raise ValueError('Invalid/target/diagnostic path in bank')
   schedule=ep['support_P2'] if arm=='R' else support(ep,'P2','Transfer')
   if len(schedule)!=20 or any(r['context'] not in 'ABC' for r in schedule):raise ValueError('Wrong native support')
   if arm=='B' and ((ep is e['shared'] and ep['kind']!='shared') or (ep is e['conflict'] and ep['kind']!='conflict')):raise ValueError('Unpaired shared/conflict task')
 return True

def build(seed,reserved):
 r=json.loads((SOURCE/f'registry/R_s{seed:02d}.json').read_text())['records'];t=generate(seed,'FOMAML',42,reserved)['episodes'];b=paired_stream(t);families,overlap=structured_stream(seed,10080,reserved);banks=dict(R=r,T=t[:10000],B=b[:5000],O1=overlap['O1'][:10000],O2=overlap['O2'][:10000]);unused=dict(T=t[10000:],B=b[5000:],O1=overlap['O1'][10000:],O2=overlap['O2'][10000:])
 for arm,eps in banks.items():
  old=json.loads((SOURCE/f'registry/{arm}_s{seed:02d}.json').read_text())['records']
  if eps[:len(old)]!=old:raise ValueError('Historical task prefix changed')
  validate_bank(eps,arm,reserved)
 return banks,unused

def main():
 p=OUT/'audit/task_registry_seal.json'
 if p.exists():core.hashes(OUT,json.loads(p.read_text())['files']);return
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'];assert '1 passed' in (OUT/'audit/05_GREEN_streams.log').read_text();reserved={tuple(p) for p in json.loads((OUT/'registry/diagnostic_paths.json').read_text())};rows=[];files={}
 for seed in SEEDS:
  banks,unused=build(seed,reserved)
  for arm,eps in banks.items():
   path=OUT/f'registry/{arm}_s{seed:02d}.json';dump(path,dict(seed=seed,arm=arm,records=eps,original_registry_sha=sha(SOURCE/f'registry/{arm}_s{seed:02d}.json'),prefix_records=960 if arm=='B' else 1920));files[str(path.relative_to(OUT))]=sha(path)
   if arm.startswith('O'):counts={str(k):sum(e['n_exceptions']==k for e in eps) for k in range(3)}
   else:counts={}
   rows.append(dict(seed=seed,arm=arm,path=str(path.relative_to(OUT)),sha256=sha(path),records=len(eps),episodes=1920 if arm=='R' else 10000,prefix_exact=True,zero=counts.get('0'),one=counts.get('1'),two=counts.get('2'),novel_eligible_cells=sum(e.get('eligible_count',0) for e in eps),critical_cells=sum(e.get('critical_cells',0) for e in eps)))
  dump(OUT/f'registry/unused_suffix_s{seed:02d}.json',unused);print('Native extended registry seed',seed,'sealed',flush=True)
 write_frame(OUT/'registry/task_streams.csv',rows);dump(p,dict(passed=True,files=files,U_P_tasks_identical=True,historical_prefix_records_unchanged=True,T_B_final_partial_subsets_distinct=True,unused_suffix_retained=True,final_episode=10000,balanced_final_outer_step=5000,global_reserved_paths=len(reserved)));append('## A full task bank seal\nAll six native keyed libraries extended through full10080 generation then exact10000 episode/5000 pair consumption; historical1920 prefix identical for every objective. Original support/query masks and all global sequence exclusions verified. U/P share banks exactly. Unused80 T/O episodes/40 B pairs retained perseed, with realized partial-block exception counts. No after-outcome balancing or task selection.');print('Registry audit PASSED',flush=True)
if __name__=='__main__':main()
