import pandas as pd
from experiments.meta_objective_balance.report import QUESTIONS,primary,interpret

def test_report_has_eight_questions_and_registered_primary_cohort_not_best_dose():
 assert len(QUESTIONS)==8
 f=pd.DataFrame([dict(candidate=a,dose=d,metric=m) for a in ['A','E','F'] for d in [480,960,1920] for m in ['gap','ce','post_acquisition_gap','conflict_accuracy','conflict_ce','D_A','D_B','joint']])
 q=primary(f,['A','E']);assert len(q)==14 and set(q.dose)=={1920} and set(q.candidate)=={'A','E'} and 'joint' not in set(q.metric)
 assert interpret(-1,1)=='inconclusive direction; not equivalence' and interpret(-2,-1)=='decrease' and interpret(1,2)=='increase'

def test_screen_factorial_keeps_beta_and_lambda_comparisons_exploratory():
 from experiments.meta_objective_balance.report import screen_factorial
 f=pd.DataFrame([dict(seed=s,arm=a,dose=480,gap=float(ord(a)),conflict_accuracy=float(ord(a)),conflict_ce=float(ord(a)),D_A=float(ord(a)),D_B=float(ord(a))) for s in range(6) for a in 'ABCDEF'])
 q=screen_factorial(f);assert len(q)==7*5*6 and q.seed.nunique()==6
 b=q[q.kind.eq('beta')];assert set(b.contrast)=={'B-A','D-C','F-E'} and b.effect.eq(1).all()
