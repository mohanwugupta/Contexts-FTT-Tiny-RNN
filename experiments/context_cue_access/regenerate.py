"""Rebuild key summaries and figures from sealed tables, with no model evaluation."""
from pathlib import Path
import argparse
from .seal import run as verify
from .analysis import derived_tables
from .plots import render_all
from context_sequence.provenance.hashing import sha256
from .io import dump


def run(source,destination):
    source=Path(source).resolve();dest=Path(destination).resolve();verify(source)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Destination must be empty and distinct')
    if dest==source:raise ValueError('Cannot mutate sealed source')
    dest.mkdir(parents=True,exist_ok=True);(dest/'tables').mkdir();(dest/'figures').mkdir()
    tables=derived_tables(source);checks=[]
    for name,frame in tables.items():
        p=dest/'tables'/f'{name}.csv';frame.to_csv(p,index=False)
        checks.append({'path':str(p.relative_to(dest)),'equal':sha256(p)==sha256(source/'tables'/p.name)})
    render_all(source,dest,tables)
    for p in (dest/'figures').glob('*'):checks.append({'path':str(p.relative_to(dest)),'equal':sha256(p)==sha256(source/'figures'/p.name)})
    dump(dest/'regeneration_check.json',checks)
    if not all(x['equal'] for x in checks):raise ValueError('Regenerated output identity mismatch; no source mutated')
    print(f'Regenerated {len(checks)} identical files without model evaluation')
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);a=p.parse_args();run(a.source,a.destination)
