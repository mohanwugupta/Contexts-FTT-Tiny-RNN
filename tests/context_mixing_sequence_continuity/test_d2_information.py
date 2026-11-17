import numpy as np
import pandas as pd
import pytest
from experiments.context_mixing_sequence_continuity.history_diagnostic import history_mask,excess_table,seed_contrast
from experiments.context_mixing_sequence_continuity.familiar_decoder import fit_familiar,decode


def test_d2_sequence_repeated_token_map_and_fixed_history_mask():
    mask=history_mask((0,1,2,0,3,1,2,3));assert mask.tolist()==[True,False,True,True,False,False,True]
    assert mask.sum()==4 and len(mask)==7
    assert history_mask((0,1,0,2)).tolist()==[True,False,True]


def test_d2_excess_ce_familiar_reference_and_seed_unit():
    def trace(values):return pd.DataFrame({'transition':range(1,8),'token':[0,1,2,0,3,1,2],'target':[1,2,0,3,1,2,3],'raw_ce':values})
    a=trace(np.arange(7)/2);b=trace(np.arange(7)/3);m=trace(np.arange(7)+1)
    f=excess_table(a,b,m);np.testing.assert_allclose(f.excess_ce,m.raw_ce-(a.raw_ce+b.raw_ce)/2)
    f['seed']=0;g=f.copy();g['seed']=1;g['excess_ce']+=2
    c=seed_contrast(pd.concat([f,g]));assert len(c)==2 and c.seed.nunique()==2
    expected=f.excess_ce.iloc[[0,2,3,6]].mean()-f.excess_ce.iloc[[1,4,5]].mean();assert abs(c.history_contrast.iloc[0]-expected)<1e-12
    bad=b.copy();bad.loc[1,'target']=3
    with pytest.raises(ValueError):excess_table(a,bad,m)


def test_d2_decoder_training_firewall_and_no_probe_choice_of_lambda():
    rng=np.random.default_rng(17);a=rng.normal(size=(7,8));b=a+.01*rng.normal(size=(7,8));fit=fit_familiar(a,b)
    before=fit['fit_sha'];mean=fit['ridge_pool']['mean'].copy();coef=fit['ridge_pool']['weights'].copy()
    decode(fit,rng.normal(size=(7,8)));decode(fit,np.ones((7,8))*100)
    np.testing.assert_array_equal(mean,fit['ridge_pool']['mean']);np.testing.assert_array_equal(coef,fit['ridge_pool']['weights']);assert fit['fit_sha']==before
    assert fit['ridge_lambda']==1.
    with pytest.raises(TypeError):fit_familiar(a,b,ridge_lambda=.01)


def test_d2_familiar_cross_context_sanity_shared_and_nonshared():
    rng=np.random.default_rng(28);a=rng.normal(size=(7,8));b=a+.01*rng.normal(size=(7,8));fit=fit_familiar(a,b)
    assert fit['prototype_transfer_validated'] and fit['ridge_transfer_validated']
    assert fit['ridge_status']=='shared_decoder_sanity_passed'
    nonshared=fit_familiar(a,b[np.roll(np.arange(7),1)])
    assert not nonshared['ridge_transfer_validated'] and nonshared['ridge_status']=='nonidentifiable_shared_decoder'


def test_d2_underdetermined_and_degenerate_not_identifiable():
    fit=fit_familiar(np.zeros((7,8)),np.zeros((7,8)))
    assert fit['ridge_status']=='nonidentifiable_shared_decoder' and fit['degenerate_covariance']
    fit=fit_familiar(np.ones((3,8)),np.ones((3,8)))
    assert not fit['fit_available'] and fit['ridge_status']=='nonidentifiable_shared_decoder'
