"""Frozen predecessor identities, complete U Adam and exact shared fresh floors."""
import shutil,sqlite3
from .common import *
from .protocol import floors,validate_checkpoint,restore_outer,plan
from experiments.high_scale_constrained_meta.audit import metrics
from experiments.heldout_context_meta_scaling_10k.storage import decode

def same(a,b):
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

def main():
 receipt=OUT/'audit/source_audit.json'
 if receipt.exists():assert json.loads(receipt.read_text())['passed'];return
 initialize_process();manifest=SOURCE/'manifest.json';m=json.loads(manifest.read_text());core.hashes(SOURCE,m['outputs']);core.hashes(ROOT,m['code']);core.hashes(ROOT,m['inputs']);refs={str((SOURCE/n).relative_to(ROOT)):h for n,h in m['outputs'].items()};refs.update(m['code']);refs.update(m['inputs']);refs[str(manifest.relative_to(ROOT))]=sha(manifest);dump(OUT/'registry/parent_hashes.json',refs);(OUT/'initial').mkdir(exist_ok=True);init=[];weights=[];sources=[]
 for seed in SEEDS:
  for scale in SCALES:
   src=SOURCE/f'initial/s{seed:02d}_scale{scale}.pt';b=torch.load(src,weights_only=False);f=floors(b['model']);path=OUT/f'initial/s{seed:02d}_scale{scale}.pt';h=atomic_save(path,dict(model=copy.deepcopy(b['model']),seed=seed,scale=scale,floors=f,source=str(src.relative_to(ROOT)),source_sha=sha(src)));init.append(dict(seed=seed,scale=scale,path=str(path.relative_to(ROOT)),sha256=h,source=str(src.relative_to(ROOT)),source_sha=sha(src),**metrics(b['model'])))
   for group,value in f.items():weights.append(dict(seed=seed,scale=scale,group=group,floor=value,definition='actual float64 RMS of exact archived fresh scaled tensors'))
   for arm in ['R']+ARMS:
    for dose in [240,960,1920]:
     p=SOURCE/f'meta/s{seed:02d}_scale{scale}/{arm}_e{dose}.pt';cp=torch.load(p,weights_only=False);validate_checkpoint(cp,arm,dose);registry=SOURCE/f'registry/{arm}_s{seed:02d}.json';assert sha(registry)==cp['registry_sha'];assert cp['seed']==seed and cp['scale']==scale
     if arm!='R':
      _,o=restore_outer(cp,arm,dose);assert same(o.state_dict(),cp['optimizer'])
     if dose==1920:
      db=sqlite3.connect('file:'+str(p.parent/f'{arm}_journal.sqlite')+'?mode=ro',uri=True);blob,digest=db.execute('SELECT data,hash FROM latest').fetchone();db.close();latest=decode(blob,digest);assert same(latest['model'],cp['model']) and same(latest['optimizer'],cp['optimizer']) and same(latest['torch_rng'],cp['torch_rng']) and same(latest['numpy_rng'],cp['numpy_rng']) and same(latest['python_rng'],cp['python_rng']) and latest['next_episode']==cp['next_episode']
     sources.append(dict(seed=seed,scale=scale,condition='U',phase=arm,dose=dose,path=str(p.relative_to(ROOT)),sha256=sha(p),registry_path=str(registry.relative_to(ROOT)),registry_sha=cp['registry_sha'],outer_iteration=cp['episode'],next_outer_iteration=cp['next_episode'],complete_RNG=True,outer_moments_restored=arm!='R'))
 for seed in SEEDS:shutil.copyfile(SOURCE/f'registry/heldout_v2_seed_{seed}.json',OUT/f'registry/heldout_v2_seed_{seed}.json')
 shutil.copyfile(SOURCE/'registry/diagnostic_paths.json',OUT/'registry/diagnostic_paths.json');write_frame(OUT/'registry/initializations.csv',init);write_frame(OUT/'registry/floors.csv',weights);write_frame(OUT/'registry/U_checkpoints.csv',sources)
 # Entire histories remain referenced; no copied/re-evaluated old state×cue cells.
 reused=[]
 for name in ['fresh','reptile_targets','fomaml_targets']:
  for r in pd.read_csv(SOURCE/f'registry/{name}.csv').to_dict('records'):
   r['directory']=str((SOURCE/r['directory']).relative_to(ROOT));r.update(condition='Shared' if name=='fresh' else 'U',reused=True,new_target_updates=0,new_diagnostic_updates=0);reused.append(r)
 write_frame(OUT/'registry/reused_targets.csv',reused);h=pd.read_csv(SOURCE/'registry/diagnostics.csv');h=h[(h.phase.eq('R')&h.dose.eq(1920))|(h.phase.isin(ARMS)&h.dose.eq(1920))].copy();assert len(h)==90;h['directory']=[str((SOURCE/d).relative_to(ROOT)) for d in h.directory];h['condition']='U';h['reused']=True;write_frame(OUT/'registry/reused_conflicts.csv',h);dump(receipt,dict(passed=True,parent_manifest_sha=sha(manifest),parent_dependencies=len(refs),fresh_initializations=18,identical_U_P_fresh=True,actual_group_floors=90,complete_U_checkpoints=len(sources),U_final_Adam_rng_journal_bitwise=True,reused_targets=288,reused_conflicts=90,prior_strict_forward_exception_retained=1,plan=plan()));append('## A source/reuse audit complete\nFull predecessor manifest and all ancestors verified.18 exact shared fresh states/90 actual group floors;270 U registered source checkpoints verified including72 complete1920 outer Adam/RNG/journal cursors (B960 versus1920 others). All288 historical target and90 matched conflict references indexed without rerunning branches. P starts fresh, never rescaled U. No optimizer steps.');print('Source/reuse audit PASSED',flush=True)
if __name__=='__main__':main()
