from pathlib import Path
import hashlib,json,os,tempfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def fingerprint(value):return hashlib.sha256(encoded(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def clean(value):
    import math
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if hasattr(value,'tolist'):return clean(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def write(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(clean(value),indent=2,allow_nan=False)+'\n'
    if immutable and path.exists():
        if path.read_text()!=data:raise ValueError('Immutable identity changed: '+str(path))
        return
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'w') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def identifier(value):
    import re
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value):raise ValueError('Invalid identifier')
    return value
def append(text):
    with (PACKAGE/'SCRATCHPAD.md').open('a') as f:f.write('\n'+text+'\n')
