"""Fixed single-candidate eligibility and deterministic tie breaks."""
from .common import *
def choose(f):
 if set(f.arm)!=set('ABC') or any(sorted(g.seed)!=SCREEN for _,g in f.groupby('arm')):raise ValueError('All six prespecified screen seeds required in all arms')
 rows=[]
 for a,g in f.groupby('arm'):
  rows.append(dict(arm=a,gap=float(g.gap.mean()),D_A=float(g.D_A.mean()),D_B=float(g.D_B.mean()),score=float((g.D_A.mean()+g.D_B.mean())/2),strong_count=int(g.strong_familiar.eq(1).sum()),conflict_accuracy=float(g.conflict_accuracy.mean()),nonfloor_both_count=int(g.nonfloor_both.sum()),Mixed_both_success_count=int(g.Mixed_both_success.sum())))
 s=pd.DataFrame(rows);limit=float(s[s.arm=='A'].gap.iloc[0])+.1;s['eligible']=s.arm.ne('A')&s.strong_count.ge(5)&s.gap.le(limit)&s.D_A.gt(0)&s.D_B.gt(0);eligible=s[s.eligible].sort_values(['score','conflict_accuracy','gap','arm'],ascending=[False,False,True,True]);selected=None if eligible.empty else str(eligible.iloc[0].arm)
 return dict(selected=selected,reference='A',passed=selected is not None,c_gap_limit=limit,screen_seeds=SCREEN,exploratory=True),s

def main():
 assert not (OUT/'audit/selection_seal.json').exists();f=read('screen_seed_outcomes');f=f[f.arm.isin(list('ABC'))&f.dose.eq(480)];r,s=choose(f);write_frame(OUT/'tables/screen_candidate_summary.csv',s)
 text='# Screening decision\n\n'+s.to_markdown(index=False)+'\n\nSelected: '+str(r['selected'])+'; reference A. Eligibility: ≥5/6 strong familiar learners, C-gap≤A+.1, positive mean first-A and first-B specificity. Rank mean specificity, conflict accuracy, C-gap, alphabetical ID. Mixed-both and floor counts are descriptive.\n\n'+('Confirmation uses untouched seeds6–19, source960 initialization, 960 episodes with saved480; no reselection.' if r['passed'] else 'Neither candidate is eligible. Fixed stopping rule: confirmation is not run. This is an unsuccessful screen, not evidence of equivalence.')+'\n'
 (OUT/'SCREENING_DECISION.md').write_text(text);dump(OUT/'audit/selection_seal.json',dict(**r,decision_sha=sha(OUT/'SCREENING_DECISION.md'),screen_table_sha=sha(OUT/'tables/screen_seed_outcomes.csv'),time=time.time()));append('## Fixed screening decision\n'+text)
if __name__=='__main__':main()
