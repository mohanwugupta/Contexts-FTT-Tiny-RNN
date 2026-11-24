import argparse,json
from pathlib import Path
from .port import ROOT,export,preflight
p=argparse.ArgumentParser(description='Deterministic cloud preparation only; no research launch command')
p.add_argument('action',choices=['export','preflight']);p.add_argument('--destination',type=Path);p.add_argument('--root',type=Path,default=ROOT)
a=p.parse_args()
if a.action=='export':
    if a.destination is None:p.error('--destination is required')
    result=export(a.destination)
else:result=preflight(a.root)
print(json.dumps(result,indent=2))
