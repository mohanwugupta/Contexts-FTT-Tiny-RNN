from research_search.runners.fixture import receipt
def test_completed_fixture_covers_failure_restart_and_real_evaluation():
    r=receipt()
    assert r['passed'] and r['actual_optimizer_steps']==4
    for name in ['duplicate_rejected','queue_restarted','state_restored','simulated_failure_retained','uncertain_step_not_replayed','token_limit_enforced','optimizer_limit_enforced','fixed_evaluator_used','final_report_written']:assert r[name]
