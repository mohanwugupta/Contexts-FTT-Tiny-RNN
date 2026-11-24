
def test_report_preserves_primary_and_partial_robustness():
 from experiments.nonzero_context_generalization.report import generate
 text=generate(write=False)
 for arm in ['HI','HC','NI','NC']:assert f'| {arm} |' in text
 assert '26.00%' in text and '14.50%' in text
 assert 'mediation' in text and 'unassessed' in text and 'float32' in text
 assert 'α=1' in text and 'not 400' in text
