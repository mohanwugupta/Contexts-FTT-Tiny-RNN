"""Cache exact native saved-table measurements once per physical target history."""
import gzip,pickle
from .common import *
from experiments.high_scale_meta_progression import analysis as native
from experiments.heldout_context_meta_scaling_10k.adapters import bind

def all_registry(require_complete=True):
 frames=[pd.read_csv(OUT/'registry/reused_targets.csv')]
 for n in ['B_targets','C_targets','targets']:
  p=OUT/f'registry/{n}.csv'
  if p.exists():frames.append(pd.read_csv(p))
  elif require_complete:raise ValueError('Incomplete target registry '+n)
 f=pd.concat(frames,ignore_index=True)
 if require_complete:assert len(f)==1134
 assert not f.duplicated(['seed','scale','condition','phase','dose']).any();return f

def measure(r,cache=True):
 condition=r['condition'];phase=r['phase'];directory=(ROOT/r['path']).parent;assert sha(ROOT/r['path'])==r['sha256'];name=f's{int(r["seed"]):02d}_scale{int(r["scale"])}_{condition}_{phase}_e{int(r["dose"])}';folder=OUT/'measurement_cache';folder.mkdir(exist_ok=True);path=folder/(name+'.pkl.gz');receipt=folder/(name+'.json');identity=dict(checkpoint_sha=r['sha256'],analyzer_sha=sha(Path(__file__)),native_analyzer_sha=sha(ROOT/'experiments/high_scale_meta_progression/analysis.py'))
 if cache and receipt.exists():
  b=json.loads(receipt.read_text());assert all(b[k]==v for k,v in identity.items()) and b['cache_sha']==sha(path)
  with gzip.open(path,'rb') as f:return pickle.load(f)
 rr=dict(r,directory=str(directory.relative_to(ROOT)),phase=f'{condition}-{phase}')
 with bind(native,OUT=ROOT):tables=native.analyze_one(rr)
 for n,f in tables.items():f['phase']=phase;f['condition']=condition
 tables['outcomes']['NoContext_minus_C']=tables['outcomes']['NoContext_ce']-tables['outcomes']['C_ce']
 if cache:
  pending=path.with_suffix('.pending')
  with pending.open('xb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=3) as f:pickle.dump(tables,f,protocol=5)
   raw.flush();os.fsync(raw.fileno())
  os.replace(pending,path);dump(receipt,dict(identity,cache_sha=sha(path)))
 return tables

def collect(reg):
 tables={}
 for index,r in enumerate(reg.to_dict('records'),1):
  for name,f in measure(r).items():tables.setdefault(name,[]).append(f)
  if index%36==0:print('Cached table measurements',index,'/',len(reg),flush=True)
 return {name:pd.concat(frames,ignore_index=True) for name,frames in tables.items()}
