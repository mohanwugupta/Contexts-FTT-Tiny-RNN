"""Preregistered exploratory screening only; freeze before untouched-seed runs."""
from .common import *
from .protocol import ARMS,SCREEN

def choose(f):
 if set(f.arm)!=set(ARMS) or any(sorted(g.seed)!=SCREEN for _,g in f.groupby('arm')):raise ValueError('Screen requires all six prespecified seeds in every arm')
 rows=[]
 for a,g in f.groupby('arm'):
  rows.append(dict(arm=a,lambda_weight=ARMS[a][0],beta=ARMS[a][1],gap=float(g.gap.mean()),conflict_accuracy=float(g.conflict_accuracy.mean()),conflict_ce=float(g.conflict_ce.mean()),D_A=float(g.D_A.mean()),D_B=float(g.D_B.mean()),strong_count=int(g.strong_familiar.eq(1).sum()),nonfloor_both_count=int(g.nonfloor_both.sum())))
 s=pd.DataFrame(rows);limit=float(s[s.arm=='C'].gap.iloc[0])+.1;s['strong_eligible']=s.strong_count.eq(6);s['tolerance_eligible']=s.strong_eligible&s.gap.le(limit);s['specificity_safeguard']=s[['D_A','D_B']].min(axis=1)
 strong=s[s.arm.ne('C')&s.strong_eligible].copy();eligible=strong[strong.tolerance_eligible].copy()
 if len(strong)<2:raise ValueError('Fewer than two strong-acquisition experimental candidates; confirmation blocked')
 conflict_order=['conflict_accuracy','conflict_ce','specificity_safeguard','gap','arm'];directions=[False,True,False,True,True]
 if len(eligible)>=2:
  first=eligible.sort_values(conflict_order,ascending=directions).iloc[0].arm;other=eligible[eligible.arm.ne(first)];keep=[]
  for _,r in other.iterrows():
   dominated=((other.gap<=r.gap)&(other.conflict_accuracy>=r.conflict_accuracy)&((other.gap<r.gap)|(other.conflict_accuracy>r.conflict_accuracy))).any()
   if not dominated:keep.append(r.arm)
  second=other[other.arm.isin(keep)].sort_values(['gap','conflict_accuracy','specificity_safeguard','arm'],ascending=[True,False,False,True]).iloc[0].arm;met=True
 else:
  first=strong.sort_values(['gap','conflict_accuracy','arm'],ascending=[True,False,True]).iloc[0].arm;second=strong[strong.arm.ne(first)].sort_values(conflict_order,ascending=directions).iloc[0].arm;met=False
 return dict(selected=[first,second],reference='C',tolerance_met=met,c_gap_limit=limit,screen_seeds=SCREEN,exploratory=True,primary_significance_claim=False),s

def screen_candidates(f):
 return f[f.arm.isin(ARMS)&f.dose.eq(480)].copy()

def main():
 assert not (OUT/'audit/selection_seal.json').exists();f=screen_candidates(read('screen_seed_outcomes'));r,s=choose(f);write_frame(OUT/'tables/screen_candidate_summary.csv',s)
 text='# Screening decision, frozen before confirmation\n\n'+s.to_markdown(index=False)+'\n\nSelected: '+', '.join(r['selected'])+'; balanced control C retained. C-gap tolerance met: '+str(r['tolerance_met'])+'.\n\nAll six screening seeds are exploratory. Selection follows the pre-outcome SCRATCHPAD rule: strong acquisition on all six, fixed0.1-nat control tolerance, conflict leader then C-preserving Pareto complement, exact tie breaks. Specificity safeguards and nonfloor counts above do not by themselves establish persistent separate learning. Confirmation uses only seeds6–19, original960 source copies, fixed1920 episodes with960 intermediate, and no further selection.\n'
 (OUT/'SCREENING_DECISION.md').write_text(text);dump(OUT/'audit/selection_seal.json',dict(passed=True,**r,decision_sha=sha(OUT/'SCREENING_DECISION.md'),screen_table_sha=sha(OUT/'tables/screen_seed_outcomes.csv'),summary_sha=sha(OUT/'tables/screen_candidate_summary.csv'),time=time.time()))
 append('## Screening decision frozen\nSelected '+','.join(r['selected'])+' plus matched control C, using only six screen seeds and preregistered rules. See SCREENING_DECISION.md and selection_seal.json. No new confirmation optimizer/evaluation has run at this decision. Tolerance met: '+str(r['tolerance_met']))
if __name__=='__main__':main()
