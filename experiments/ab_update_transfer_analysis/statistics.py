"""Identical paired seed resamples for every condition; missingness retained."""
import warnings
from functools import lru_cache
import numpy as np
@lru_cache(None)
def indices():return np.random.default_rng(20261122).integers(6,size=(10000,6))
@lru_cache(100000)
def _interval(values):
 x=np.array(values,float);good=np.isfinite(x);n=int(good.sum());r=dict(n=n,n_planned=6,mean=np.nan,median=np.nan,ci_low=np.nan,ci_high=np.nan,median_ci_low=np.nan,median_ci_high=np.nan,n_negative=int((x<0).sum()),n_positive=int((x>0).sum()))
 if n:
  draws=x[indices()];den=np.isfinite(draws).sum(1);means=np.divide(np.nansum(draws,1),den,out=np.full(10000,np.nan),where=den>0)
  with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(draws,axis=1)
  r.update(mean=float(x[good].mean()),median=float(np.median(x[good])),ci_low=float(np.nanquantile(means,.025)),ci_high=float(np.nanquantile(means,.975)),median_ci_low=float(np.nanquantile(med,.025)),median_ci_high=float(np.nanquantile(med,.975)))
 return r
def interval(x):
 x=np.asarray(x,float);assert x.shape==(6,);return dict(_interval(tuple(x)))
