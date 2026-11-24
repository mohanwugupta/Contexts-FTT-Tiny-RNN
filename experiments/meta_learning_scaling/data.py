"""Exact table references: dose0 physical histories shared, never new replicates."""
from .common import *
TABLE_NAMES=['evaluations','geometry','training','diagnostics']

def registry(stage='full'):
 r=pd.read_csv(OUT/'registry/reuse_registry.csv');a=pd.read_csv(OUT/'registry/stage_a_new_targets.csv');parts=[r,a]
 if stage=='full':parts.append(pd.read_csv(OUT/'registry/stage_c_new_targets.csv'))
 r=pd.concat(parts,ignore_index=True);assert not r.run_id.duplicated().any();return r

def collect(stage='full'):
 r=registry(stage);parents={n:pd.read_csv(SOURCE/'tables'/f'{n}.csv',float_precision='round_trip') for n in TABLE_NAMES};frames={n:[] for n in TABLE_NAMES}
 for row in r.itertuples():
  for n in TABLE_NAMES:
   if row.reused:
    x=parents[n];x=x[(x.seed==row.seed)&(x.condition==row.condition)].copy();x=x.drop(columns=[k for k in ['seed','prior','organization','condition'] if k in x])
   else:
    path=OUT/row.directory/f'{n}.csv'
    try:x=pd.read_csv(path,float_precision='round_trip')
    except pd.errors.EmptyDataError:x=pd.DataFrame()
   conditions=[f'P1-{row.organization}',f'P2-{row.organization}'] if row.dose==0 else [row.condition]
   for c in conditions:
    z=x.copy();z['seed']=row.seed;z['condition']=c;z['organization']=row.organization;z['prior']=c[:2];z['dose']=row.dose;z['physical_run_id']=row.run_id;z['reused']=bool(row.reused);frames[n].append(z)
 return {n:pd.concat(v,ignore_index=True) for n,v in frames.items()},r
