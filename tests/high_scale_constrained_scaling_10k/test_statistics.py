import numpy as np,pandas as pd
from experiments.high_scale_constrained_scaling_10k.statistics import comparisons

def test_PU_pairing_signed_changes_and_missing_horizons_retained():
 rows=[]
 for s in range(6):
  for c in ['U','P']:
   for phase,dose in [('R',1920),('T',1920),('T',10000)]:
    for update in ([330,990] if not(c=='P' and dose==1920 and phase=='T') else [330]):rows.append(dict(seed=s,scale=4,condition=c,phase=phase,dose=dose,update=update,x=(s if c=='U' else -s)))
 raw,ci=comparisons(pd.DataFrame(rows),['x']);p=raw[raw.kind.eq('P-U')&raw.phase.eq('T')&raw.dose.eq(1920)&raw['update'].eq(330)];assert len(p)==6 and np.all(p.effect<=0)
 absent=ci[ci.kind.eq('P-U')&ci.phase.eq('T')&ci.dose.eq(1920)&ci['update'].eq(990)];assert len(absent)==1 and absent.iloc[0]['n']==0 and np.isnan(absent.iloc[0]['mean'])
 assert (ci.n_planned==6).all()
