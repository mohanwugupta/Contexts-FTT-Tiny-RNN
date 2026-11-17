import numpy as np
import torch
from torch import nn

from srn_partial_observability_v1.analysis.ambiguity_tests import (
    evaluate_ambiguity,
    generate_ambiguity_batch,
)
from srn_partial_observability_v1.analysis.bayes_risk import risk_table
from srn_partial_observability_v1.analysis.hidden_probes import fit_hidden_probes
from srn_partial_observability_v1.models import ElmanSRN, GRUControl


def test_primary_model_is_plain_elman_srn_with_scalar_linear_readout():
    model = ElmanSRN(input_dim=4, hidden_dim=4)
    assert isinstance(model.recurrent, nn.RNN)
    assert model.recurrent.nonlinearity == "tanh"
    assert isinstance(model.output, nn.Linear)
    assert not any(isinstance(module, nn.GRU) for module in model.modules())
    prediction, hidden, final = model(torch.zeros(3, 7, 4))
    assert prediction.shape == (3, 7)
    assert hidden.shape == (3, 7, 4)
    assert final.shape == (1, 3, 4)


def test_gru_exists_only_as_compatible_optional_control():
    model = GRUControl(input_dim=4, hidden_dim=4)
    prediction, hidden, _ = model(torch.zeros(2, 5, 4))
    assert prediction.shape == (2, 5)
    assert hidden.shape == (2, 5, 4)


def test_ambiguity_grid_is_exact_and_reports_all_alignment_metrics():
    levels = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    batch = generate_ambiguity_batch(levels, episodes_per_level=3, history_length=8, seed=9)
    np.testing.assert_allclose(np.sort(np.unique(batch.q_star[:, -1])), levels, atol=1e-5)
    # Each history/state configuration is evaluated at every requested q.
    for match_id in range(3):
        matched = batch.diagnostic_match_id[:, -1].numpy() == match_id
        np.testing.assert_allclose(
            np.sort(batch.q_star[matched, -1].numpy()), levels, atol=1e-5
        )
        assert np.ptp(batch.skill_a[matched, -1].numpy()) < 1e-7
        assert np.ptp(batch.skill_b[matched, -1].numpy()) < 1e-7
    rows, summary, _ = evaluate_ambiguity(ElmanSRN(), batch)
    assert "q_implied_unclipped" in rows
    assert {
        "posterior_mean_r2", "posterior_mean_slope", "posterior_mean_intercept",
        "posterior_mean_mae", "responsibility_r2", "responsibility_slope",
        "responsibility_intercept", "responsibility_mae",
    }.issubset(summary)


def test_sampling_risk_is_twice_posterior_mean_risk_analytically_and_numerically():
    table = risk_table(n_samples=200_000, seed=11)
    np.testing.assert_allclose(
        table.analytic_sampling_risk, 2 * table.analytic_mean_risk, rtol=1e-12
    )
    np.testing.assert_allclose(
        table.numerical_sampling_risk / table.numerical_mean_risk, 2.0, atol=0.08
    )


def test_hidden_probes_are_episode_disjoint():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(400, 3))
    hidden = latent @ rng.normal(size=(3, 4))
    probes = fit_hidden_probes(
        hidden,
        {"S_A": latent[:, 0], "S_B": latent[:, 1], "q_star": latent[:, 2]},
        np.repeat(np.arange(40), 10),
        seed=2,
    )
    assert probes.episode_disjoint.all()
    assert probes.heldout_r2.min() > 0.98
