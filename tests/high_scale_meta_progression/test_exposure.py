from experiments.high_scale_meta_progression.finalize import expected_exposure

def test_budget_includes_actual_cadence_examples_and_queries():
 x=expected_exposure();assert x['meta_episodes']==172800 and x['inner_updates']==3456000 and x['outer_Adam_updates']==120960 and x['Reptile_interpolations']==34560
 assert x['target_updates_new']==162360 and x['target_clones_new']==3312 and x['conflict_updates']==56160 and x['optimizer_updates_scientific']==3798792
 assert x['support_sequence_presentations']==4838400 and x['meta_query_examples']==725760 and x['query_gradient_evaluations']==293760

def test_audit_budget_markdown_is_renderable():
 from experiments.high_scale_meta_progression.finalize import budget_markdown
 s=budget_markdown({'meta_episodes':172800});assert 'meta_episodes' in s and '172800' in s and 'Actual/planned complete' in s
