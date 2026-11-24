import numpy as np
from experiments.structured_overlap_meta.history import record_metrics

def test_weighted_gradient_and_missing_critical_not_zero():
 r={'early_gradient':np.ones(172),'familiar_gradient':np.ones(172)*2,'novel_gradient':np.ones(172)*-1,'combined_gradient':np.ones(172)*.9}
 q=record_metrics(r);assert np.isclose(q['early_norm'],np.sqrt(172));assert np.isclose(q['early_novel_cosine'],-1);assert np.isclose(q['combined_norm'],np.sqrt(172)*.9)
