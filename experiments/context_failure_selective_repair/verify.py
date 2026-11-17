"""Independent hash seal and complete artifact-set verification, no training."""
from pathlib import Path
import argparse,json
from context_sequence.provenance.hashing import sha256
from .core import ROOT,CONTRACT,check_files,codes,dump,verify_stage


def safe_destination(path):
 p=Path(path).resolve()
 if any(name in str(p) for name in ['results/output_compatible_trajectories','results/context_cue_access','results/context_mixing_sequence_continuity']):raise ValueError('Cannot overwrite predecessor sealed data')
 if p.exists() and any(p.iterdir()):raise FileExistsError('Regeneration destination must be empty')
 return True

def verify_outputs(base,files):
 base=Path(base);check_files(base,files)
 actual={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and p.name!='.DS_Store' and str(p.relative_to(base))!='manifests/sealed_manifest.json'}
 if actual!=set(files):raise ValueError('Extra/missing scientific bytes: '+str(sorted(actual.symmetric_difference(files))))
 return True

def run(out,seal=False):
 out=Path(out).resolve();stages={s:verify_stage(out,s) for s in ['r0','r1','r2','r3','r4','r5']};p=out/'manifests/sealed_manifest.json'
 if seal:
  if p.exists():raise FileExistsError('Immutable seal already exists')
  if stages['r0']['status']['scientific_factorial_rollouts']!=0 or stages['r1']['status']['new_scientific_rollouts']!=0:raise ValueError('Unexpected matched-branch rerun')
  inputs=stages['r0']['inputs'];files={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='.DS_Store' and f!=p}
  dump(p,dict(status='completed_including_infeasible_negative_inconclusive',scientific_model_updates=0,optimization_steps=0,original_checkpoints_unchanged=True,inputs=inputs,code=codes(),contract=CONTRACT,files=files,ignored_os_metadata=['.DS_Store']))
 m=json.loads(p.read_text());check_files(ROOT,m['inputs']);check_files(ROOT,m['code']);verify_outputs(out,m['files'])
 if m['scientific_model_updates']!=0 or m['optimization_steps']!=0:raise ValueError('No training/optimization allowed')
 result=dict(passed=True,stages=6,files=len(m['files']),source_files=len(m['inputs']),original_checkpoints=20,model_updates=0,output_directory=str(out));print(json.dumps(result),flush=True);return m

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seal',action='store_true');a=p.parse_args();run(a.output,a.seal)
