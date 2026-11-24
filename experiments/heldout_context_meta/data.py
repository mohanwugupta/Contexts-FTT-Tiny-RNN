"""Reuse historical1920 target branches exactly; add only missing measurements."""
from .common import *
from .mechanisms import extra_measures
from experiments.context_experience_prior.target import saved_obs

def registry():
 r=pd.concat([pd.read_csv(OUT/'registry/historical.csv'),pd.read_csv(OUT/'registry/new_targets.csv')],ignore_index=True);assert len(r)==400 and not r.run_id.duplicated().any();return r

def directory(row):return ROOT/row.directory if row.reused else OUT/row.directory

def enrich_historical(row):
 d=OUT/'reuse_enrichment'/row.run_id;d.mkdir(parents=True,exist_ok=True)
 if (d/'geometry.csv').exists():return pd.read_csv(d/'geometry.csv',float_precision='round_trip')
 b=torch.load(ROOT/row.path,weights_only=False);path=directory(row);pos=pd.read_csv(path/'position_predictions.csv.gz',float_precision='round_trip');checks=pd.read_csv(path/'mathematical_checks.csv.gz',float_precision='round_trip');parts={(k,c):g.sort_values('position') for (k,c),g in pos.groupby(['update','cue'])};cc={k:g.sort_values('position').to_dict('records') for k,g in checks.groupby('update')};geo=[];dc=[]
 for k,st in enumerate(b['states']):
  obs={c:saved_obs(parts[k,c]) for c in CUES};g,x=extra_measures(st['model'],obs,cc[k]);geo.append(dict(update=k,**g));dc.extend(dict(update=k,position=t+1,**q) for t,q in enumerate(x))
 write_frame(d/'geometry.csv',geo);write_frame(d/'decomposition.csv.gz',dc);return pd.DataFrame(geo)

def collect():
 r=registry();frames={n:[] for n in ['evaluations','training','geometry','diagnostics']};positions=[]
 for row in r.itertuples():
  path=directory(row)
  for name in frames:
   try:x=pd.read_csv(path/f'{name}.csv',float_precision='round_trip',low_memory=False)
   except pd.errors.EmptyDataError:x=pd.DataFrame()
   if row.reused and name=='geometry':x=x.merge(enrich_historical(row),on='update',validate='one_to_one')
   for k in ['seed','condition','prior','organization','arm','dose']:x[k]=getattr(row,k)
   x['physical_run_id']=row.run_id;x['reused']=row.reused;frames[name].append(x)
  p=pd.read_csv(path/'position_predictions.csv.gz',float_precision='round_trip',low_memory=False);p=p[(p.cue=='C')&p['update'].isin([0,40,80,120,170,330])].copy()
  for k in ['seed','condition','prior','organization','arm','dose']:p[k]=getattr(row,k)
  positions.append(p)
 return {n:pd.concat(x,ignore_index=True) for n,x in frames.items()},pd.concat(positions,ignore_index=True),r
