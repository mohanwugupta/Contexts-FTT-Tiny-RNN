from experiments.meta_learning_scaling.report import QUESTIONS,planned_primary

def test_report_answers_both_outcomes_and_fixed_contrasts():
 assert len(QUESTIONS)==7
 x=planned_primary();assert 'P2-P1' in x and 'P1:1920-240' in x and 'change_P2-P1:1920-240' in x
