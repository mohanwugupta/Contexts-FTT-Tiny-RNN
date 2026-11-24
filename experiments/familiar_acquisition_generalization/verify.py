"""Read-only scientific input/code/output verification."""
from pathlib import Path
import argparse,json
from .audit import ROOT,DEFAULT,sha


def check_outputs(out,files):
 out=Path(out)
 for n,h in files.items():
  if not (out/n).is_file() or sha(out/n)!=h:raise ValueError('Missing/modified output: '+n)
 actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','.DS_Store']}
 if actual!=set(files):raise ValueError('Unexpected/missing scientific output')
 return True


def run(out=DEFAULT,seal=False):
 out=Path(out);manifest=out/'manifest.json'
 if seal:
  if manifest.exists():raise FileExistsError('Immutable manifest')
  code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments','tests'] for p in (ROOT/folder/'familiar_acquisition_generalization').glob('*.py')}
  inputs=json.loads((out/'audit/source_hashes.json').read_text());files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','.DS_Store']}
  manifest.write_text(json.dumps(dict(inputs=inputs,code=code,files=files,models=80,original_seeds=20,reevaluations=0,model_updates=0),indent=2)+'\n')
 m=json.loads(manifest.read_text())
 for n,h in {**m['inputs'],**m['code']}.items():
  if sha(ROOT/n)!=h:raise ValueError('Scientific source/checkpoint/code changed: '+n)
 check_outputs(out,m['files']);print(json.dumps(dict(passed=True,models=80,outputs=len(m['files']),input_hashes=len(m['inputs']),model_updates=0)));return m

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT));p.add_argument('--seal',action='store_true');a=p.parse_args();run(Path(a.output),a.seal)
