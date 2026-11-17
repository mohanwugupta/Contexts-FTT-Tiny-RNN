import numpy as np


def bootstrap_indices():return np.random.default_rng(20260925).integers(0,20,size=(10000,20))


def summarize(values,indices):
    values=np.asarray(values,dtype=float)
    if values.shape!=(20,) or not np.isfinite(values).all():raise ValueError('Inference requires exactly 20 finite seed-level values')
    samples=values[indices];means=samples.mean(axis=1);medians=np.median(samples,axis=1)
    return {'n_seeds':20,'mean':float(values.mean()),'median':float(np.median(values)),
            'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975)),
            'median_ci_low':float(np.quantile(medians,.025)),'median_ci_high':float(np.quantile(medians,.975)),
            'fraction_positive':float(np.mean(values>0)),'resampling_unit':'seed','bootstrap_seed':20260925}
