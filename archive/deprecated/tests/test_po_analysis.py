import numpy as np
import torch

from partial_observability_gru.analysis.hidden_probes import fit_hidden_probes
from partial_observability_gru.analysis.interventions import intervene_linear_state
from partial_observability_gru.analysis.human_probes import evaluate_human_style_probes
from partial_observability_gru.analysis.posterior_alignment import alignment_metrics
from partial_observability_gru.environment.episode import EpisodeConfig, generate_batch
from partial_observability_gru.models import build_model


def test_hidden_probe_decodes_known_distributed_variables():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(240, 3))
    rotation = rng.normal(size=(3, 6))
    hidden = latent @ rotation
    episode = np.repeat(np.arange(24), 10)
    probes = fit_hidden_probes(hidden, {"S_A": latent[:, 0], "S_B": latent[:, 1], "q_star": latent[:, 2]}, episode)
    assert probes.set_index("target").r2.min() > 0.98


def test_linear_state_intervention_reaches_requested_decoded_values():
    hidden = np.array([[0.0, 0.0], [0.2, -0.1]])
    direction = np.array([2.0, 0.0])
    moved = intervene_linear_state(hidden, direction, intercept=0.0, target_value=0.75)
    np.testing.assert_allclose(moved @ direction, 0.75)


def test_alignment_recovers_exact_posterior_mean_and_belief():
    q = np.linspace(0.1, 0.9, 30)
    a, b = np.linspace(-1, 0, 30), np.linspace(1, 2, 30)
    y = q * a + (1 - q) * b
    metrics = alignment_metrics(y, y, a, b, q)
    assert metrics["posterior_mean_r2"] == 1.0
    assert metrics["belief_r2"] == 1.0


def test_human_style_both_and_novel_are_zero_shot_and_symmetric_normatively():
    batch = generate_batch(4, EpisodeConfig(length=8, uncertainty="moderate"), seed=9)
    model = build_model("deterministic", batch.inputs.shape[-1], hidden_dim=3)
    probes = evaluate_human_style_probes(model, "deterministic", batch)
    assert {"Clear_A", "Clear_B", "Midpoint", "Both", "Novel", "Default"} == set(probes.probe)
    assert not probes.seen_during_training.any()
    symmetric = probes[probes.probe.isin(["Midpoint", "Novel"])]
    np.testing.assert_allclose(symmetric.bayes_q_equal_prior, 0.5)
