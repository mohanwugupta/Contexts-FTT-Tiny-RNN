from pathlib import Path

from probe_modeling.epoch_pipeline import EpochAnalysisConfig, run_epoch_analysis


def test_epoch_pipeline_writes_v2_contract(tmp_path, epoch_keypress_frame):
    source = tmp_path / "combined.csv"
    epoch_keypress_frame.to_csv(source, index=False)
    output = tmp_path / "v2"
    config = EpochAnalysisConfig(
        seed=9,
        neural_seeds=(9,),
        gate_epochs=5,
        recovery_simulations=1,
        recovery_regimes=("easy",),
        make_plots=False,
    )
    run_epoch_analysis(source, output, config=config)
    required = {
        "clean_probe_data_v2.csv", "counterfactual_AB_states_v2.csv",
        "epoch_behavioral_summary.csv", "mixed1_metrics.csv", "mixed2_metrics.csv",
        "mixed_test_metrics.csv", "post_mixed_aftereffects.csv", "model_fits_v2.csv",
        "model_predictions_v2.csv", "model_comparison_by_epoch.csv",
        "temporal_prediction_tests.csv", "counterbalance_generalization.csv",
        "gate_states_v2.csv", "gate_reset_interventions.csv", "distilled_dynamics.csv",
        "model_recovery_results_v2.csv", "model_recovery_confusion_v2.csv",
        "model_recovery_human_parameter_region.csv", "run_config_v2.json",
    }
    assert required.issubset({path.name for path in Path(output).iterdir()})

