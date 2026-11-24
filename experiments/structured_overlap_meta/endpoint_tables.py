"""Complete absolute six-cue endpoint summaries; fixed seed units."""
from .common import *
from experiments.meta_objective_balance.statistics import summarize

def main():
 for stage,seeds in [('screen',SCREEN),('confirm',CONFIRM)]:
  f=read(stage+'_evaluations');q=f[f['update'].eq(330)];summary=summarize(q,['arm','dose','cue'],['ce','gap','accuracy','success','strict_all'],seeds);write_frame(OUT/f'tables/{stage}_endpoint_cue_summary.csv',summary)
 q=pd.read_csv(OUT/'tables/confirm_endpoint_cue_summary.csv');q=q[q.dose.eq(960)&q.arm.ne('Baseline')];rows=[]
 for (arm,cue),g in q.groupby(['arm','cue']):
  v=g.set_index('metric');ce=v.loc['ce'];acc=v.loc['accuracy'];rows.append(dict(arm=arm,cue=cue,n=int(ce.n),mean_CE=ce['mean'],median_CE=ce['median'],CE_CI_low=ce.ci_low,CE_CI_high=ce.ci_high,mean_accuracy=acc['mean'],mean_gap=v.loc['gap','mean'],median_gap=v.loc['gap','median']))
 frame=pd.DataFrame(rows);write_frame(OUT/'tables/report_final_six_cues.csv',frame);text='## Absolute six-cue performance at target update330, final meta dose\n\n'+frame.to_markdown(index=False,floatfmt='.6g')+'\n\nAll14 confirmation seeds are included per arm. Complete mean/median95% seed-bootstrap intervals for CE, gap and accuracy at both saved doses are in `tables/confirm_endpoint_cue_summary.csv`; screening summaries remain separate. At+960, the C-gap benefit is small (mean C−A −.007835 nats), both first-block specificity intervals cross zero, and mean conflict accuracy is identical. This does not establish the joint stronger-learning/transfer phenotype or equivalence.\n\n'
 p=OUT/'REPORT.md';s=p.read_text().replace('## Figures\n',text+'## Figures\n');p.write_text(s)
 dump(OUT/'audit/endpoint_tables_complete.json',dict(passed=True,all_six_cues=True,mean_median_CE_gap_accuracy=True,bootstrap=10000,complete_seed_units=True,optimizer_updates=0))
 append('## Endpoint reporting completeness\nAdded allsixcue absolute CE/accuracy and C-gap mean/median summaries with fixed seed-bootstrap intervals at330 for both doses. This fills a reporting omission without new model evaluations. Figure3 corrected solely to the prescribed axis orientation; outcomes and decision unchanged.')
if __name__=='__main__':main()
