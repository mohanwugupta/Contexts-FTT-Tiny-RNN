"""Whole planned seeds, fixed paired resampling; no seed/task pooling."""
import numpy as np
import pandas as pd
import warnings

def indices(n):return np.random.default_rng(20260925).integers(n,size=(10000,n))
def interval(x,seeds):
 x=np.asarray(x,float);assert len(x)==len(seeds) and len(set(seeds))==len(seeds);good=np.isfinite(x);n=int(good.sum());r=dict(n=n,n_planned=len(x),mean=np.nan,median=np.nan,ci_low=np.nan,ci_high=np.nan,median_ci_low=np.nan,median_ci_high=np.nan,n_negative=int((x<0).sum()),n_positive=int((x>0).sum()))
 if n:
  draws=x[indices(len(x))];valid=np.isfinite(draws);den=valid.sum(1);means=np.divide(np.nansum(draws,1),den,out=np.full(10000,np.nan),where=den>0)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(draws,axis=1)
  r.update(mean=float(np.nanmean(x)),median=float(np.nanmedian(x)),ci_low=float(np.nanquantile(means,.025)),ci_high=float(np.nanquantile(means,.975)),median_ci_low=float(np.nanquantile(med,.025)),median_ci_high=float(np.nanquantile(med,.975)))
 return r

def summarize(f,groups,metrics,seeds):
 rows=[]
 for key,g in f.groupby(groups,dropna=False):
  key=key if isinstance(key,tuple) else (key,)
  assert not g.seed.duplicated().any(),(groups,key)
  for m in metrics:rows.append(dict(zip(groups,key),metric=m,**interval(g.set_index('seed')[m].reindex(seeds).to_numpy(float),seeds)))
 return pd.DataFrame(rows)

def time_summary(f,groups,t,metrics,seeds):
 rows=[];idx=indices(len(seeds));weights=np.stack([(idx==i).sum(1)/len(seeds) for i in range(len(seeds))],1)
 for key,g in f.groupby(groups,dropna=False):
  key=key if isinstance(key,tuple) else (key,)
  for m in metrics:
   p=g.pivot(index='seed',columns=t,values=m).reindex(seeds);x=p.to_numpy(float);valid=np.isfinite(x);den=weights@valid;draw=np.divide(weights@np.nan_to_num(x),den,out=np.full_like(den,np.nan),where=den>0)
   for j,tt in enumerate(p.columns):
    v=x[:,j];ok=np.isfinite(v);dd=draw[:,j];rows.append(dict(zip(groups,key),**{t:tt},metric=m,mean=float(v[ok].mean()) if ok.any() else np.nan,ci_low=float(np.nanquantile(dd,.025)) if ok.any() else np.nan,ci_high=float(np.nanquantile(dd,.975)) if ok.any() else np.nan,n=int(ok.sum()),n_planned=len(seeds)))
 return pd.DataFrame(rows)
