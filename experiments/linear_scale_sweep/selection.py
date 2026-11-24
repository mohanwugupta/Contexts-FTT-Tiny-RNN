from .common import *
from .audit import build

def select(f,m):
 result={}
 for arm,g in f.groupby('arm'):
  ref=g[g.dose.eq(2)];rows=[]
  for s,q in g.groupby('dose'):
   match=m[m.arm.eq(arm)&m.dose.eq(s)&m.stratum.eq('all')];valid=bool(((match.n>=2)&(match['mean']>0)).any());eligible=bool(q.strong_familiar.eq(1).sum()>=5 and q.early_phenotype.sum()>=5 and q.sustained.sum()>=5 and q.early_cost.mean()>ref.early_cost.mean() and valid);joint=eligible and q.D_A.mean()>ref.D_A.mean() and q.D_B.mean()>ref.D_B.mean() and q.strict_all.eq(1).sum()>=5 and q.Mixed_both_success.sum()>=max(0,ref.Mixed_both_success.sum()-1);rows.append(dict(scale=int(s),eligible=eligible,joint_eligible=bool(joint),early=q.early_cost.mean(),median=q.early_cost.median(),sustained=q.sustained_cost.mean(),matched_control=valid,acquired=int(q.strong_familiar.eq(1).sum()),early_count=int(q.early_phenotype.sum()),sustained_count=int(q.sustained.sum())))
  table=pd.DataFrame(rows);e=table[table.eligible];j=table[table.joint_eligible];rank=lambda x,col:None if x.empty else int(x.sort_values([col,'median','scale'],ascending=[False,False,True]).iloc[0].scale);result[arm]=dict(split=rank(e,'early'),sustained=rank(e,'sustained'),joint=rank(j,'early'),rows=rows)
 return result

def main():
 assert json.loads((OUT/'audit/screen_analysis.json').read_text())['passed'];f=read('screen_outcomes');m=read('screen_matched_pairs');r=select(f,m);fresh=r['Fresh'];chosen=list(dict.fromkeys([s for s in [fresh['split'],fresh['sustained']] if s is not None]));rows=[dict(source=a,**v) for a,b in r.items() for v in b['rows']];write_frame(OUT/'tables/scale_eligibility.csv',rows);text='# Scale selection, frozen before new seeds\n\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n\nFresh candidate scales: '+str(chosen)+'. Scale2 is a separate paired reference control. Ranking: eligible mean balanced adjusted cycles1–3; separate sustained cycles2–3 leader if different. Eligibility and matched-control rules were fixed in PREREGISTRATION. No C requirement for Fresh. U/P best splitting and joint choices remain exploratory:\n\n'+json.dumps({a:{k:v for k,v in b.items() if k!='rows'} for a,b in r.items()},indent=2)+'\n\n'+('Fourteen new seeds6–19 will test only these Fresh choices and the necessary matched scale2 control, with no meta-training or reselection.' if chosen else 'No eligible Fresh scale: no new-seed replication is run. This is an unsuccessful screen, not evidence of equivalence.')+'\n';(OUT/'SCALE_SELECTION.md').write_text(text);dump(OUT/'audit/selection.json',dict(passed=True,selected=chosen,decision_sha=sha(OUT/'SCALE_SELECTION.md'),screen_outcomes_sha=sha(OUT/'tables/screen_outcomes.csv'),ranking=r,time=time.time()))
 if chosen:
  rows=build(range(6,20),['Fresh'],sorted(set(chosen+[2])));write_frame(OUT/'registry/replicate_initializations.csv',rows)
 append('## Selection frozen\nFresh='+str(chosen)+';14newseedscale2controls are paired references, not extra selected candidates. U/P exploratory only. No newseed outcome available at freeze; no meta-training or parameter tuning.')
if __name__=='__main__':main()
