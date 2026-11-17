from pathlib import Path

from probe_modeling.pipeline import ProbeAnalysisConfig, run_probe_analysis


def test_pipeline_writes_required_exports(tmp_path, keypress_probe_frame):
    source = tmp_path / "combined.csv"
    keypress_probe_frame.to_csv(source, index=False)
    out_dir = tmp_path / "results"
    config = ProbeAnalysisConfig(
        seed=7,
        neural_seeds=(7,),
        gate_epochs=10,
        recovery_simulations=1,
        make_plots=False,
    )
    run_probe_analysis({"Combined": source}, out_dir, config=config)
    required = {
        "clean_probe_data.csv",
        "counterfactual_AB_states.csv",
        "behavioral_probe_metrics.csv",
        "model_fits.csv",
        "model_predictions.csv",
        "model_comparison.csv",
        "gate_states.csv",
        "cue_intervention_results.csv",
        "state_intervention_results.csv",
        "post_probe_aftereffects.csv",
        "model_recovery_results.csv",
        "model_recovery_confusion.csv",
        "run_config.json",
        "participant_splits.csv",
    }
    assert required.issubset({p.name for p in Path(out_dir).iterdir()})

