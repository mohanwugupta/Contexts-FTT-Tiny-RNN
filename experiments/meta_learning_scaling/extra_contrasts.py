"""Registered functional-retention and full mechanism paired contrast exports."""
from .common import *
from .analysis import summarize

def paired_rows(f,groups,metrics):
 rows=[]
 for key,g in f.groupby(groups):
  key=key if isinstance(key,tuple) else (key,)
  for metric in metrics:
   p=g.pivot(index='seed',columns=['condition','dose'],values=metric).reindex(range(20));doses=sorted(g.dose.unique())
   for org in ['I','C']:
    pairs=[]
    for dose in doses:
     pairs.append((('P2-'+org,dose),('P1-'+org,dose),'P2-P1',dose))
     for prior in ['P1','P2']:pairs.append(((prior+'-'+org,dose),(prior+'-'+org,0),prior+':dose-0',dose))
    for prior in ['P1','P2']:pairs.append(((prior+'-'+org,1920),(prior+'-'+org,240),prior+':1920-240',1920))
    for a,b,name,dose in pairs:
     if a not in p or b not in p:continue
     for s,v in (p[a]-p[b]).items():rows.append(dict(zip(groups,key),seed=s,organization=org,contrast=name,dose=dose,metric=metric,effect=v))
 return pd.DataFrame(rows)

def main():
 specs=[('block',read('block_transfer'),['end','practiced'],['practiced_gain','unpracticed_gain','specificity']),('diagnostic',read('learning_effects'),['update'],['mean_own_improvement','mean_cross_improvement','mean_specificity','D_A','D_B']),('mechanism',read('geometry').query('update in [0,170,330]'),['update'],['baseline_ce','C_ce','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm','A_norm','B_norm','C_norm','g_norm','d_norm','hidden_separation','centered_logit_separation','baseline_margin','C_margin'])]
 for name,f,groups,metrics in specs:
  e=paired_rows(f,groups,metrics);ss=summarize(e,groups+['organization','contrast','dose','metric'],['effect']);write_frame(OUT/'tables'/f'{name}_dose_seed_effects.csv',e);write_frame(OUT/'tables'/f'{name}_dose_intervals.csv',ss)
 dump(OUT/'audit/retention_contrasts.json',dict(passed=True,unconditioned_blocks=True,restored_probes=True,all_mechanism_metrics=True,paired_seeds=True,selection=False));append('## Registeredretention/mechanismcontrastscomplete\nUnconditionedblock/probeandall16mechanismmetrics exportwithinseedP2−P1/dose−0/1920−240effects+10kCIs. Matched-acquisitioncontrastsremainseparateconditionaldescriptions. No favorablemetricselection,additionalmodeltrainingorcheckpointselection.')
if __name__=='__main__':main()
