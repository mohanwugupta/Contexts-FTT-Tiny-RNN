"""Paired reporting of all prespecified frozen-mechanism measurements."""
from .core import *
from .analysis import paired
from experiments.centered_context_linear.core import interval
METRICS=['baseline_ce','K_raw_norm','K_centered_norm','novel_contribution_norm','margin_change','recurrent_spectral_norm','u_norm','v_norm','hidden_separation','centered_logit_separation','baseline_margin']

def estimates(frame,metrics=METRICS):
 f=frame.copy();f['alpha']=1.;rows=[];ci=[]
 for metric in metrics:
  z=paired(f,metric);z['metric']=metric;rows.append(z)
  for k,g in z.groupby('update'):
   for effect in ['normalization','centering','interaction','NC_minus_NI']:
    mean,lo,hi=interval(g[effect]);ci.append(dict(update=k,metric=metric,effect=effect,n=len(g),mean=mean,ci_low=lo,ci_high=hi))
 return pd.concat(rows,ignore_index=True),pd.DataFrame(ci)

def main():
 f=pd.read_csv(OUT/'tables/primary_mechanism.csv',float_precision='round_trip');raw,ci=estimates(f[f['update'].isin(PROBES)]);raw.to_csv(OUT/'tables/mechanism_seed_contrasts.csv',index=False);ci.to_csv(OUT/'tables/mechanism_paired_intervals.csv',index=False)
 append('## Mechanism paired reporting complete\nAll11 prespecified mechanism quantities exported with20 matchedseed contrasts andwhole-seed bootstrap CIs. No model evaluations orparameter changes;quantities/thresholds unchanged.')
if __name__=='__main__':main()
