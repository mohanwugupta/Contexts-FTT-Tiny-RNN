"""Descriptive first criterion crossings; no checkpoint chosen for use."""
from .core import *

def first_crossings(f):
 rows=[]
 for (seed,c),g in f.groupby(['seed','condition']):
  r=dict(seed=seed,condition=c);q=g[['update','strong_familiar']].drop_duplicates().sort_values('update');hit=q[q.strong_familiar]['update'];r['first_familiar']=int(hit.min()) if len(hit) else np.nan;r['familiar_success_states']=len(hit)
  for cue,key in [('C','C'),('Mixed-both','Mixed_both')]:
   x=g[g.cue==cue];hit=x[(x.ce<=.5)&(x.accuracy>=6/7)]['update'];r['first_'+key]=int(hit.min()) if len(hit) else np.nan;r[key+'_success_states']=len(hit);r[key+'_minus_familiar']=r['first_'+key]-r['first_familiar'];r[key+'_order']='not_observed' if not len(hit) else ('familiar_not_observed' if not np.isfinite(r['first_familiar']) else ('precedes' if r['first_'+key]<r['first_familiar'] else ('same_state' if r['first_'+key]==r['first_familiar'] else 'follows')))
  rows.append(r)
 return pd.DataFrame(rows)

def main():
 f=pd.read_csv(OUT/'tables/evaluations.csv');q=pd.read_csv(OUT/'tables/performance_gaps.csv');f=f.merge(q[['seed','condition','update','strong_familiar']],on=['seed','condition','update'],validate='many_to_one');r=first_crossings(f);assert len(r)==120 and r.groupby('condition').seed.apply(lambda x: sorted(x)==list(range(20))).all();r.to_csv(OUT/'tables/acquisition_timing.csv',index=False);print(r.groupby(['condition','C_order']).size().to_string())
if __name__=='__main__':main()
