"""Small fixed familiar-only position diagnostics; never model-weight fitting."""
import numpy as np
from context_sequence.provenance.hashing import object_hash
from .config import CONTRACT


def _ridge(x,y):
    mean=x.mean(0);raw_sd=x.std(0,ddof=0);sd=np.maximum(raw_sd,1e-6);z=(x-mean)/sd;intercept=y.mean(0);centered=y-intercept
    gram=z.T@z+np.eye(8);weights=np.linalg.solve(gram,z.T@centered)
    return {'mean':mean,'sd':sd,'raw_sd':raw_sd,'weights':weights,'intercept':intercept,'condition_number':float(np.linalg.cond(gram)),
       'centered_rank':int(np.linalg.matrix_rank(x-mean)),'n_training_states':len(x)}


def _predict(model,x):return (((x-model['mean'])/model['sd'])@model['weights']+model['intercept']).argmax(1)


def _nearest(prototypes,x):return np.linalg.norm(x[:,None,:]-prototypes[None,:,:],axis=2).argmin(1)%7


def fit_familiar(a,b):
    a=np.array(a,float,copy=True);b=np.array(b,float,copy=True)
    if a.shape!=(7,8) or b.shape!=(7,8) or not np.isfinite(np.r_[a.ravel(),b.ravel()]).all():
        return {'fit_available':False,'prototype_status':'nonidentifiable_shared_decoder','ridge_status':'nonidentifiable_shared_decoder','reason':'insufficient_or_invalid_familiar_states','ridge_lambda':1.}
    labels=np.arange(7);y=np.eye(7);pool=np.r_[a,b];rp=_ridge(pool,np.r_[y,y]);ra=_ridge(a,y);rb=_ridge(b,y)
    proto_ab=float(np.mean(_nearest(a,b)==labels));proto_ba=float(np.mean(_nearest(b,a)==labels));ridge_ab=float(np.mean(_predict(ra,b)==labels));ridge_ba=float(np.mean(_predict(rb,a)==labels))
    proto_valid=proto_ab>=5/7 and proto_ba>=5/7;ridge_valid=ridge_ab>=5/7 and ridge_ba>=5/7
    degenerate=bool(np.any(rp['raw_sd']<1e-6) or rp['centered_rank']<7)
    return {'fit_available':True,'ridge_lambda':1.,'ridge_pool':rp,'ridge_A':ra,'ridge_B':rb,'prototypes':pool,'fit_sha':object_hash({'A':a.tolist(),'B':b.tolist(),'lambda':1.,'floor':1e-6,'ddof':0}),
       'prototype_A_to_B':proto_ab,'prototype_B_to_A':proto_ba,'ridge_A_to_B':ridge_ab,'ridge_B_to_A':ridge_ba,
       'prototype_transfer_validated':proto_valid,'ridge_transfer_validated':ridge_valid,'degenerate_covariance':degenerate,
       'prototype_status':'shared_decoder_sanity_passed' if proto_valid and not degenerate else 'nonidentifiable_shared_decoder',
       'ridge_status':'shared_decoder_sanity_passed' if ridge_valid and not degenerate else 'nonidentifiable_shared_decoder',
       'ridge_training_accuracy':float(np.mean(_predict(rp,pool)==np.tile(labels,2))),'n_training_states':14,'small_sample_warning':True}


def decode(fit,h):
    import pandas as pd
    if not fit['fit_available']:return pd.DataFrame([{'status':'nonidentifiable_shared_decoder','reason':fit.get('reason'),'position_accuracy':np.nan}])
    h=np.asarray(h,float);d=np.linalg.norm(h[:,None,:]-fit['prototypes'][None,:,:],axis=2);idx=d.argmin(1);ridge=_predict(fit['ridge_pool'],h);position=np.arange(7)
    return pd.DataFrame({'transition':position+1,'prototype_position':idx%7+1,'ridge_position':ridge+1,'prototype_correct':idx%7==position,'ridge_correct':ridge==position,
      'nearest_distance':d.min(1),'true_position_distance':np.minimum(d[np.arange(7),position],d[np.arange(7),position+7]),'prototype_status':fit['prototype_status'],'ridge_status':fit['ridge_status']})
