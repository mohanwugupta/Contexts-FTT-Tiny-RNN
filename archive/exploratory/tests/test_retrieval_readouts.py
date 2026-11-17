import numpy as np

from retrieval_modeling.categorical_readout import categorical_log_likelihood
from retrieval_modeling.continuous_readout import continuous_log_likelihood, continuous_readout
from retrieval_modeling.evidence_strength import evidence_scaled_sigma
from retrieval_modeling.history_prior import history_responsibility


def test_continuous_readout_is_weighted_log_rt_and_does_not_mutate_experts():
    a = np.array([400.0])
    b = np.array([600.0])
    before_a, before_b = a.copy(), b.copy()
    prediction = continuous_readout(a, b, np.array([0.25]))
    assert prediction.item() == 550.0
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


def test_categorical_likelihood_prefers_switching_data():
    rng = np.random.default_rng(4)
    a = np.full(1000, 6.0)
    b = np.full(1000, 6.8)
    q = np.full(1000, 0.5)
    observed = np.where(rng.random(1000) < q, a, b) + rng.normal(0, 0.04, 1000)
    categorical = categorical_log_likelihood(observed, a, b, q, np.full(1000, 0.04)).sum()
    continuous = continuous_log_likelihood(observed, 0.5 * (a + b), np.full(1000, 0.04)).sum()
    assert categorical > continuous


def test_history_is_a_prior_and_eta_zero_removes_history():
    contexts = ["A"] * 5 + ["Both"] + ["B"] * 5 + ["Both"]
    with_history = history_responsibility(contexts, beta=1.0, rho=0.9, eta=1.0)
    probes = with_history[np.array(contexts) == "Both"]
    assert probes.q_A.iloc[0] > probes.q_A.iloc[1]
    no_history = history_responsibility(contexts, beta=1.0, rho=0.9, eta=0.0)
    np.testing.assert_allclose(no_history.loc[np.array(contexts) == "Both", "q_A"], 0.5)


def test_absolute_evidence_changes_variance_not_responsibility():
    evidence = np.array([1.0, 0.2])
    sigma = evidence_scaled_sigma(np.array([0.1, 0.1]), evidence, kappa=1.0)
    assert sigma[0] < sigma[1]

