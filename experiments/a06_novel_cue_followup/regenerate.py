"""Rebuild exact derived tables and plots from accepted arrays, outside sealed evidence."""
from .verify import *

def destination(path):
    p=Path(path).resolve()
    if p==OUT.resolve() or OUT.resolve() in p.parents or p in OUT.resolve().parents:
        raise ValueError('Reproduction must be outside canonical evidence')
    if p.exists() and any(p.iterdir()):raise ValueError('Destination must be empty')
    p.mkdir(parents=True,exist_ok=True);return p

def main(path):
    dest=destination(path)
    if (OUT/'manifest.json').exists():sealed()
    with no_scientific_execution():
        from . import analysis,reporting,figures
        analysis.main(dest);reporting.main(dest);figures.main(dest/'figures')
        files={}
        for p in sorted(dest.rglob('*')):
            if p.is_file():
                name=str(p.relative_to(dest));assert sha(p)==sha(OUT/name),'Reproduction mismatch '+name;files[name]=sha(p)
    result=dict(passed=True,reproduced_files=files,new_optimizer_steps=0,model_forward_calls=0,unit='whole trained seed',sources='accepted immutable arrays and original saved tables; no network execution')
    put(dest/'REPRODUCTION.json',result)
    return dict(passed=True,files=len(files),destination=str(dest),new_optimizer_steps=0,model_forward_calls=0)
