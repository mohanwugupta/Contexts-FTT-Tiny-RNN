import json
from dataclasses import replace

import pandas as pd
import pytest

from srn_partial_observability_v1.pipeline import AnalysisConfig, final_config, run_analysis


CSV_OUTPUTS = {
    "training_conditions.csv", "network_performance.csv", "baseline_performance.csv",
    "ambiguity_behavior.csv", "bayesian_ground_truth.csv", "posterior_mean_alignment.csv",
    "implied_responsibility.csv", "perfect_vs_uncertain.csv",
    "continuous_vs_sampling.csv", "hidden_state_probes.csv",
    "dimensionality_comparison.csv", "belief_interventions.csv",
    "gru_robustness_control.csv", "seed_summary.csv",
}


def test_srn_pipeline_writes_complete_contract_and_documents_staged_skips(tmp_path):
    config = AnalysisConfig(
        seeds=(2,), epochs=2, batches_per_epoch=2, batch_size=8,
        train_episode_length=12, eval_episodes=6, ambiguity_episodes_per_q=2,
        make_plots=True, show_progress=False, run_dimensionality=False,
        run_gru_control=False,
    )
    result = run_analysis(tmp_path, config=config)
    names = {path.name for path in tmp_path.iterdir()}
    assert CSV_OUTPUTS.issubset(names)
    assert {"run_config_v1.json", "proof_of_concept_report.md"}.issubset(names)
    assert len(list((tmp_path / "figures_v1").glob("*.png"))) == 9
    assert pd.read_csv(tmp_path / "dimensionality_comparison.csv").status.eq("not_run").all()
    assert pd.read_csv(tmp_path / "gru_robustness_control.csv").status.eq("not_run").all()
    assert set(result["network_performance"].condition) == {"Perfect", "Uncertain"}
    record = json.loads((tmp_path / "run_config_v1.json").read_text())
    assert record["primary_architecture"] == "Elman SRN"
    assert record["inputs"] == ["x_t", "y_t_minus_1", "previous_outcome_mask", "episode_start"]
    assert record["human_data_used"] is False


def test_optional_stages_run_only_when_forced_in_software_smoke(tmp_path):
    config = AnalysisConfig(
        seeds=(3,), epochs=1, batches_per_epoch=1, batch_size=5,
        train_episode_length=8, eval_episodes=4, ambiguity_episodes_per_q=1,
        make_plots=False, show_progress=False, run_dimensionality=True,
        dimensionality_hidden_dims=(1, 4), dimensionality_seeds=(3,),
        run_gru_control=True, control_seeds=(3,), force_staged_analyses=True,
    )
    run_analysis(tmp_path, config=config)
    assert pd.read_csv(tmp_path / "dimensionality_comparison.csv").status.eq("completed").all()
    control = pd.read_csv(tmp_path / "gru_robustness_control.csv")
    assert control.status.eq("completed").all()
    assert set(control.architecture) == {"SRN", "GRU"}


def test_final_profile_is_blocked_until_an_mvp_behavioral_gate_passes(tmp_path):
    with pytest.raises(ValueError, match="requires mvp_gate_record"):
        final_config().validate()
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"behavioral_stage_gate_passed": False}))
    with pytest.raises(ValueError, match="intentionally blocked"):
        replace(final_config(), mvp_gate_record=str(failed)).validate()
    passed = tmp_path / "passed.json"
    passed.write_text(json.dumps({"behavioral_stage_gate_passed": True}))
    replace(final_config(), mvp_gate_record=str(passed)).validate()
