import json

from partial_observability_gru.pipeline import PartialObservabilityConfig, run_partial_observability_analysis


def test_partial_observability_pipeline_writes_complete_contract(tmp_path):
    config = PartialObservabilityConfig(
        uncertainty_levels=("moderate",), skill_separations=("medium",),
        objectives=("deterministic", "gaussian", "classifier", "stochastic", "distributional"),
        hidden_dims=(3,), seeds=(3,), train_episodes=8, eval_episodes=5,
        episode_length=12, epochs=2, batches_per_epoch=2, batch_size=4,
        make_plots=True, show_progress=False,
    )
    result = run_partial_observability_analysis(tmp_path, config=config)
    required = {
        "training_conditions.csv", "network_performance.csv", "ambiguity_behavior.csv",
        "bayesian_ground_truth.csv", "posterior_alignment.csv",
        "dimensionality_comparison.csv", "hidden_state_probes.csv",
        "belief_interventions.csv", "skill_interventions.csv", "objective_comparison.csv",
        "continuous_vs_sampling.csv", "human_style_probe_predictions.csv",
        "distilled_dynamics.csv", "run_config.json",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    assert len(list((tmp_path / "figures_partial_observability").glob("*.png"))) == 8
    assert set(result["performance"].objective) == set(config.objectives)
    record = json.loads((tmp_path / "run_config.json").read_text())
    assert record["target_leakage_safeguard"] == "inputs[t] contain y[t-1], never y[t]"
