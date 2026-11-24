import copy,pytest
from experiments.research_search_pilot_v2.protocol import BASELINE,LIMITS,validate_candidate


def test_limits_and_zero_prior():
    assert LIMITS==dict(tokens=50000,cpu_seconds=3600,optimizer_steps=6000,wall_seconds=7200,proposals=2,active_branches=1)
    assert BASELINE['prior']['episodes']==0 and BASELINE['target']['steps']==330


def test_one_registered_change_only():
    c=copy.deepcopy(BASELINE);c['target']['readout_initial_multiplier']=.25
    assert validate_candidate(c)==['target.readout_initial_multiplier']
    for change in [dict(prior=dict(episodes=1)),dict(model=dict(nominal_scale=4)),dict(target=dict(steps=100))]:
        bad=copy.deepcopy(BASELINE)
        for group,fields in change.items():bad[group].update(fields)
        with pytest.raises(ValueError):validate_candidate(bad)
    c['model']['centered']=True
    with pytest.raises(ValueError):validate_candidate(c)


def test_native_clean_cpu_error_branch_uses_defined_exception():
    from experiments.research_search_pilot_v2.worker_cli import bind_control_exception
    from research_search.runners.train import NativeRuntime
    from research_search.orchestration.budget import BudgetError
    bind_control_exception()
    r=object.__new__(NativeRuntime);r.store=type('S',(),{'check':lambda _:None})();r.cpu_deadline=0;r.deadline=None
    with pytest.raises(BudgetError,match='Clean CPU stop'):r.check()
