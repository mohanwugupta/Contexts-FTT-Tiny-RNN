from pathlib import Path

def test_report_states_fixed_limits_and_all_arms():
 from experiments.recurrent_context_factorial.report import generate
 text=generate(write=False)
 for arm in ['HI','HC','NI','NC']:assert f'| {arm} |' in text
 assert 'not independent' in text and 'unassessed' in text and 'NoContext' in text
 assert 'Mixed-both' in text and 'mediation' in text
