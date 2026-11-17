import json

from gru_partial_observability_v1.pipeline import V1AnalysisConfig, run_v1_analysis


def test_v1_pipeline_writes_exact_contract_and_report(tmp_path):
    config = V1AnalysisConfig(
        seeds=(2,), epochs=2, batches_per_epoch=2, batch_size=8,
        train_episode_length=12, eval_episodes=6, ambiguity_episodes_per_q=2,
        make_plots=True, show_progress=False,
    )
    result = run_v1_analysis(tmp_path, config=config)
    required = {
        "run_config_v1.json", "training_conditions.csv", "network_performance.csv",
        "ambiguity_behavior.csv", "bayesian_ground_truth.csv",
        "posterior_mean_alignment.csv", "implied_responsibility.csv",
        "perfect_vs_uncertain.csv", "continuous_vs_sampling.csv",
        "hidden_state_probes.csv", "belief_interventions.csv", "seed_summary.csv",
        "proof_of_concept_report.md",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    assert len(list((tmp_path / "figures_v1").glob("*.png"))) == 7
    assert set(result["performance"].condition) == {"Perfect", "Uncertain"}
    record = json.loads((tmp_path / "run_config_v1.json").read_text())
    assert record["gru_inputs"] == ["x_t", "y_t_minus_1", "previous_outcome_mask", "episode_start"]
