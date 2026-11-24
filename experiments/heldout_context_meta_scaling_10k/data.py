"""Exact old table reuse plus new shards; never replay matched native branches."""
from .common import *

def registry():
 old=pd.read_csv(SOURCE/'tables/physical_run_registry.csv');old['directory']=[str((ROOT/r.directory if r.reused else SOURCE/r.directory).relative_to(ROOT)) for r in old.itertuples()];old['reused']=True;old['source_program']='heldout_context_meta/v1'
 new=pd.read_csv(OUT/'registry/new_targets.csv');new['source_program']='heldout_context_meta_scaling_10k/v1';r=pd.concat([old,new],ignore_index=True);assert len(r)==1200 and not r.run_id.duplicated().any();return r

def directory(row):return ROOT/row.directory if row.reused else OUT/row.directory

def collect():
 r=registry();frames={};positions=[pd.read_csv(SOURCE/'tables/C_positions.csv',float_precision='round_trip',low_memory=False)]
 for name in ['evaluations','training','geometry','diagnostics']:
  x=pd.read_csv(SOURCE/f'tables/{name}.csv',float_precision='round_trip',low_memory=False);x['reused']=True
  if name=='evaluations':x=x.drop(columns=[k for k in ['familiar_ce','strong_familiar','gap','success','strict_ce','strict_accuracy','strict_gap','strict_all'] if k in x])
  frames[name]=[x]
 for row in r[~r.reused].itertuples():
  d=directory(row)
  for name in frames:
   x=pd.read_csv(d/f'{name}.csv',float_precision='round_trip',low_memory=False)
   for k in ['seed','condition','prior','organization','arm','dose']:x[k]=getattr(row,k)
   x['physical_run_id']=row.run_id;x['reused']=False;frames[name].append(x)
  pos=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip',low_memory=False);pos=pos[(pos.cue=='C')&pos['update'].isin([0,40,80,120,170,330])].copy()
  for k in ['seed','condition','prior','organization','arm','dose']:pos[k]=getattr(row,k)
  positions.append(pos)
 return {n:pd.concat(x,ignore_index=True) for n,x in frames.items()},pd.concat(positions,ignore_index=True),r
