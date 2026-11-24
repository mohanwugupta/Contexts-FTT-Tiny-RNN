from experiments.high_scale_constrained_scaling_10k import continuation_audit as a
from experiments.high_scale_constrained_scaling_10k import train

def test_observational_trace_superset():
 for arm in ['R','T','B','O1','O2']:
  for initial in [0,1920]:
   for dose in range(1,10001):
    expected=train.trace_dose(arm,dose,initial) or (dose==(1922 if arm=='B' else 1921))
    assert a.trace_dose(arm,dose,initial)==expected
 assert a.ORIGINAL_RUN is train.run_meta
 assert a.ORIGINAL_TRACE is train.trace_dose
