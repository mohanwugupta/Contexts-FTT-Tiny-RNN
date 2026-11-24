"""Deterministic identities and append-only diagnostic accounting."""
import copy,gzip,hashlib,json,os,platform,random,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from experiments.context_experience_prior import core
ROOT=core.ROOT;SOURCE=ROOT/'results/high_scale_constrained_scaling_10k/v1';OUT=ROOT/'results/ab_update_transfer_analysis/v1'
sha=core.sha

def digest(value):
 h=hashlib.sha256()
 def visit(x):
  if torch.is_tensor(x):h.update(b'tensor'+str(x.dtype).encode()+str(tuple(x.shape)).encode()+x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(b'array'+str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
  elif isinstance(x,dict):
   h.update(b'dict')
   for k in sorted(x,key=lambda a:str(a)):visit(k);visit(x[k])
  elif isinstance(x,(list,tuple)):
   h.update(type(x).__name__.encode())
   for v in x:visit(v)
  else:h.update((type(x).__name__+':'+repr(x)).encode())
 visit(value);return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.pending');torch.save(v,tmp)
 with tmp.open('rb') as f:os.fsync(f.fileno())
 os.replace(tmp,p)
def csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);payload=pd.DataFrame(rows).to_csv(index=False).encode()
 if p.exists():assert (gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())==payload,'Changed derived table: '+str(p)
 else:p.write_bytes(gzip.compress(payload,compresslevel=3,mtime=0) if p.suffix=='.gz' else payload)
def append(txt):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+txt+'\n')
def environment():return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),dtype='torch.float32',threads=1,workers=1)
def configure():torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def config(path):
 d=json.loads(Path(path).read_text());assert d['output_dir']==str(OUT.relative_to(ROOT));assert d['execution']['workers']==d['execution']['numerical_threads']==1
 return d
def rng():return dict(torch=torch.get_rng_state().clone(),numpy=copy.deepcopy(np.random.get_state()),python=random.getstate())
def set_rng(r):torch.set_rng_state(r['torch']);np.random.set_state(r['numpy']);random.setstate(r['python'])
