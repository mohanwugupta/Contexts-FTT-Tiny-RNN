"""Read-only table regeneration. Never imports an experimental run function."""
from pathlib import Path
import argparse,json
from context_sequence.provenance.hashing import sha256
from .verify import run as verify,safe_destination
from .reporting import products


def run(source,destination):
 src=Path(source).resolve();dest=Path(destination).resolve();verify(src);safe_destination(dest);dest.mkdir(parents=True,exist_ok=True)
 expected=json.loads((src/'audit/r5_products.json').read_text());files=products(src,dest)
 if set(files)!=set(expected):raise ValueError('Regenerated product inventory changed')
 for name in files:
  if sha256(dest/name)!=expected[name]:raise ValueError('Table-only regeneration hash mismatch: '+name)
 print(json.dumps(dict(passed=True,products=len(files),figures=5,scientific_model_calls=0,hashes_identical=True)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);a=p.parse_args();run(a.source,a.destination)
