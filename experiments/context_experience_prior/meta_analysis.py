"""Describe meta-adaptation with exposure weights, infer over seeds only."""
from .core import *
from .analysis import seed_interval

def main():
 f=pd.read_csv(OUT/'tables/meta_task_performance.csv');u=pd.read_csv(OUT/'registry/updates.csv');e=pd.read_csv(OUT/'registry/episodes.csv');w=u.groupby(['seed','episode','prior','context','sequence']).size().rename('exposures').reset_index();f=f.merge(w,on=['seed','episode','prior','context','sequence'],how='left');f['exposures']=f.exposures.fillna(0);f['weighted_ce']=f.ce*f.exposures;f['weighted_accuracy']=f.accuracy*f.exposures
 ep=f.groupby(['seed','organization','prior','condition','episode','phase']).agg(ce=('weighted_ce','sum'),accuracy=('weighted_accuracy','sum'),n=('exposures','sum')).reset_index();assert ep.n.eq(20).all();ep['ce']/=20;ep['accuracy']/=20;ep=ep.merge(e[['seed','episode','kind','pair']],on=['seed','episode']);rows=[]
 for (s,c,kind),g in ep.groupby(['seed','condition','kind']):
  for window in ['all','last40']:
   x=g if window=='all' else g[g.episode>200]
   for phase,xx in x.groupby('phase'):rows.append(dict(seed=s,condition=c,kind=kind,window=window,phase=phase,ce=xx.ce.mean(),accuracy=xx.accuracy.mean(),n_episodes=len(xx)))
 seed=pd.DataFrame(rows);summ=[]
 for keys,g in seed.groupby(['condition','kind','window','phase']):
  for metric in ['ce','accuracy']:summ.append(dict(zip(['condition','kind','window','phase'],keys),metric=metric,**seed_interval(g[metric])))
 for name,x in [('meta_weighted_episode_performance',ep),('meta_seed_performance',seed),('meta_performance_summary',pd.DataFrame(summ))]:
  assert not (OUT/'tables'/f'{name}.csv').exists();x.to_csv(OUT/'tables'/f'{name}.csv',index=False)
if __name__=='__main__':main()
