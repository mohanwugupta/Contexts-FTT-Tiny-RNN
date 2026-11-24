"""Native fixed seed bootstrap, cached by values; paired factorial contrasts."""
from functools import lru_cache
import itertools
from .common import np,pd,SEEDS,ARMS
from experiments.meta_objective_balance.statistics import interval as native_interval,time_summary

@lru_cache(maxsize=32768)
def _interval(buf):return native_interval(np.frombuffer(buf,dtype=np.float64),SEEDS)
def interval(x):
 x=np.asarray(x,dtype=np.float64)
 if x.shape!=(6,):raise ValueError('Six planned unique trained seeds required')
 return dict(_interval(x.tobytes()))

def summarize(f,groups,metrics):
 rows=[]
 for key,g in f.groupby(groups,dropna=False):
  key=key if isinstance(key,tuple) else (key,)
  if g.seed.duplicated().any():raise ValueError('A seed appears more than once in a cohort')
  for m in metrics:rows.append(dict(zip(groups,key),metric=m,**interval(g.set_index('seed')[m].reindex(SEEDS).to_numpy(float))))
 return pd.DataFrame(rows)

def comparisons(out,metrics):
 rows=[];cis=[]
 def emit(a,b,kind,meta):
  for update in sorted(set(a['update'])|set(b['update'])):
   aa=a[a['update'].eq(update)].set_index('seed');bb=b[b['update'].eq(update)].set_index('seed')
   if aa.index.has_duplicates or bb.index.has_duplicates:raise ValueError('Incorrect pairing mixes trained seeds')
   for m in metrics:
    v=(aa[m].astype(float)-bb[m].astype(float)).reindex(SEEDS);tag=dict(meta,kind=kind,update=update,metric=m);rows.extend(dict(tag,seed=s,effect=x) for s,x in v.items());cis.append(dict(tag,**interval(v.to_numpy(float)),coverage='paired_registered_horizon' if v.notna().any() else 'not_estimable_missing_or_unregistered_horizon'))
 for (scale,phase,dose),g in out[out.condition.isin(['P','U'])].groupby(['scale','phase','dose']):emit(g[g.condition.eq('P')],g[g.condition.eq('U')],'P-U',dict(scale=scale,phase=phase,dose=dose,condition='P-U',reference='U'))
 for (scale,condition,phase,dose),a in out[out.condition.isin(['P','U'])].groupby(['scale','condition','phase','dose']):
  fresh=out[out.scale.eq(scale)&out.condition.eq('Shared')&out.phase.eq('Fresh')&out.dose.eq(0)]
  if not fresh.empty:emit(a,fresh,'own-Fresh',dict(scale=scale,condition=condition,phase=phase,dose=dose,reference='Shared-Fresh0'))
  if phase in ARMS:
   parent=out[out.scale.eq(scale)&out.condition.eq(condition)&out.phase.eq('R')&out.dose.eq(1920)]
   emit(a,parent,'own-Reptile1920',dict(scale=scale,condition=condition,phase=phase,dose=dose,reference=condition+'-R1920'))
   if dose==10000:
    before=out[out.scale.eq(scale)&out.condition.eq(condition)&out.phase.eq(phase)&out.dose.eq(1920)];emit(a,before,'10000-1920',dict(scale=scale,condition=condition,phase=phase,dose=dose,reference=condition+'-'+phase+'1920'))
 for (condition,phase,dose),g in out.groupby(['condition','phase','dose']):
  for a,b in [(5,4),(6,4),(6,5)]:
   aa=g[g.scale.eq(a)];bb=g[g.scale.eq(b)]
   if not aa.empty or not bb.empty:emit(aa,bb,'scale-pair',dict(scale=a,reference_scale=b,condition=condition,phase=phase,dose=dose,reference='scale'+str(b)))
 for (scale,condition,dose),g in out[out.phase.isin(ARMS)].groupby(['scale','condition','dose']):
  for ref,candidate in itertools.combinations(ARMS,2):emit(g[g.phase.eq(candidate)],g[g.phase.eq(ref)],'objective-pair',dict(scale=scale,condition=condition,phase=candidate,dose=dose,reference=ref))
 return pd.DataFrame(rows),pd.DataFrame(cis)
