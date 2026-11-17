import pytest

from probe_modeling.recovery import recovery_identifiability, run_model_recovery
from probe_modeling.skill_states import fit_counterfactual_states


@pytest.mark.parametrize(
    "mechanism",
    ["equal_mixture", "contextual_inference", "responsibility_update", "third_state"],
)
def test_low_noise_generating_model_ranks_best(normalized_schedule, mechanism):
    schedule, _ = fit_counterfactual_states(normalized_schedule)
    results, confusion = run_model_recovery(
        schedule,
        mechanisms=[mechanism],
        n_simulations=1,
        noise_sd=0.002,
        seed=11,
    )
    assert results.iloc[0].selected_model == mechanism
    assert confusion.loc[mechanism, mechanism] == 1.0


def test_recovery_acceptance_requires_enough_simulations(normalized_schedule):
    schedule, _ = fit_counterfactual_states(normalized_schedule)
    results, _ = run_model_recovery(
        schedule,
        mechanisms=["equal_mixture"],
        n_simulations=1,
        noise_sd=0.002,
        seed=12,
    )
    status = recovery_identifiability(results)
    assert status.loc[0, "recovery_rate"] == 1.0
    assert not status.loc[0, "adequate_simulations"]
    assert not status.loc[0, "identifiable"]
