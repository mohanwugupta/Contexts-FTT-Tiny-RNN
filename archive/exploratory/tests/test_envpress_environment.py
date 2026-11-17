import numpy as np

from srn_environmental_pressures.config import EnvironmentalCondition, ExperimentConfig
from srn_environmental_pressures.environment.bayes_oracle import (
    cue_only_posterior,
    full_posterior,
    posterior_from_log_likelihood_ratio,
)
from srn_environmental_pressures.environment.cue_evidence import cue_sd_for_accuracy
from srn_environmental_pressures.environment.ftt_schedule import make_ftt_schedule
from srn_environmental_pressures.environment.outcomes import build_inputs
from srn_environmental_pressures.environment.simulation import generate_batch


def test_exact_ftt_schedule_and_fixed_exposure():
    schedule = make_ftt_schedule(n_blocks=8, block_length=10)
    assert schedule.shape == (80,)
    for block in range(8):
        expected = 0 if block % 2 == 0 else 1
        assert np.all(schedule[block * 10:(block + 1) * 10] == expected)
    config = ExperimentConfig(n_blocks=8, block_length=10)
    assert config.training_trials == 80
    assert config.weight_updates == 80


def test_validity_cue_accuracy_and_state_separation_match_configuration():
    condition = EnvironmentalCondition(
        context_validity=0.75, cue_accuracy=0.80, state_separation=0.60,
        outcome_noise_sd=0.0, state_variant="fixed_global",
    )
    batch = generate_batch(20_000, condition, n_blocks=1, block_length=10, seed=4)
    nominal = batch.nominal_context.numpy()
    latent = batch.latent_context.numpy()
    assert abs(np.mean(nominal == latent) - 0.75) < 0.01
    observed_accuracy = np.mean((batch.cues.numpy() < 0) == (latent == 0))
    assert abs(observed_accuracy - 0.80) < 0.01
    np.testing.assert_allclose(batch.skill_b.numpy() - batch.skill_a.numpy(), 0.60, atol=1e-6)
    assert cue_sd_for_accuracy(0.80) > 0


def test_current_outcome_never_enters_current_input():
    cues = np.array([-1.0, 0.0, 1.0])
    original = build_inputs(cues, np.array([0.2, 0.4, 0.8]))
    changed = build_inputs(cues, np.array([0.2, 99.0, 0.8]))
    np.testing.assert_allclose(original[1], changed[1])
    assert original[2, 1] != changed[2, 1]


def test_full_oracle_combines_nominal_prior_and_cue_likelihood():
    validity = 0.75
    cue_sd = cue_sd_for_accuracy(0.80)
    q_neutral_a = full_posterior(0.0, nominal_context=0, validity=validity, cue_sd=cue_sd)
    q_neutral_b = full_posterior(0.0, nominal_context=1, validity=validity, cue_sd=cue_sd)
    assert np.isclose(q_neutral_a, 0.75)
    assert np.isclose(q_neutral_b, 0.25)
    assert np.isclose(cue_only_posterior(0.0, cue_sd), 0.5)
    assert full_posterior(1_000.0, nominal_context=0, validity=1.0, cue_sd=cue_sd) == 1.0
    assert full_posterior(-1_000.0, nominal_context=1, validity=1.0, cue_sd=cue_sd) == 0.0
    hand = posterior_from_log_likelihood_ratio(prior_a=0.75, log_likelihood_ratio=-1.2)
    expected = 1 / (1 + np.exp(-(np.log(3) - 1.2)))
    assert np.isclose(hand, expected)
    # Cue conflict: full Bayes differs from both shortcuts.
    x = 0.4
    full = full_posterior(x, nominal_context=0, validity=validity, cue_sd=cue_sd)
    assert not np.isclose(full, validity)
    assert not np.isclose(full, cue_only_posterior(x, cue_sd))
