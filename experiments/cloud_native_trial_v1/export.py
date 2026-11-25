"""Read native0600 files as their owner; create separate readable byte-exact exports."""
import hashlib
import json
from pathlib import Path
import stat
import zipfile
from .control import sha,save

def collect(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Export destination must be new')
    files=[]
    for p in sorted(root.rglob('*')):
        mode=p.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):raise ValueError('Unsafe native file '+str(p))
        if p.is_file():files.append(p)
    out.mkdir(parents=True)
    hashes={str(p.relative_to(root)):sha(p) for p in files}
    archive=out/'results.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in files:z.write(p,str(p.relative_to(root)))
    archive.chmod(0o644)
    # Re-read archive contents; this does not require host access to native files.
    save(out/'FILES.json',dict(zip_sha256=sha(archive),files=hashes,native_permissions_unchanged=True))
    verify(out)
    return dict(passed=True,files=len(hashes),zip_sha256=sha(archive))

def verify(out):
    out=Path(out);m=json.loads((out/'FILES.json').read_text());archive=out/'results.zip'
    if sha(archive)!=m['zip_sha256']:raise ValueError('Export ZIP checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(m['files']) or set(z.namelist())!=set(m['files']):raise ValueError('Missing/duplicate exported files')
        for name,h in m['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archive path')
            digest=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(1048576),b''):digest.update(b)
            if digest.hexdigest()!=h:raise ValueError('Exported bytes changed: '+name)
    return dict(passed=True,files=len(m['files']),zip_sha256=m['zip_sha256'])
