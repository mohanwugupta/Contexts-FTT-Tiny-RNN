import pandas as pd
from experiments.structured_overlap_meta.report import specificity_supported

def test_joint_claim_needs_both_positive_intervals_not_positive_means():
 f=pd.DataFrame([dict(metric='D_A',mean=.2,ci_low=.01),dict(metric='D_B',mean=.3,ci_low=-.01)])
 assert not specificity_supported(f)
 f.loc[f.metric.eq('D_B'),'ci_low']=.01;assert specificity_supported(f)
 assert not specificity_supported(f[f.metric.eq('D_A')])
