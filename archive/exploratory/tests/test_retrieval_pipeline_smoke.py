from pathlib import Path

import pandas as pd

from retrieval_modeling.pipeline import RetrievalAnalysisConfig, run_retrieval_analysis


def _raw(dataset):
    labels = (
        ["Maze Training", "Melody Training", "Mixed Training", "Mixed Test"]
        if dataset == "Combined"
        else ["Maze Training", "Melody Training", "Space Training", "No Context Training"]
    )
    rows = []
    for person in range(3):
        participant = f"{dataset[0]}{person}"
        for block, label in enumerate(labels, 1):
            base = 400 if "Maze" in label else 600 if "Melody" in label else 500
            for sequence in range(10):
                for key in range(8):
                    rows.append(
                        {
                            "ID": participant, "new_trialNumber": block,
                            "sequenceNumber": sequence, "keypressNumber": key,
                            "trialType": label, "counterBalance": f"cb{person % 2}",
                            "correct": True, "rt": base + sequence + key + person,
                        }
                    )
    return pd.DataFrame(rows)


def test_retrieval_pipeline_writes_contract_and_figures(tmp_path):
    combined, novel = tmp_path / "combined.csv", tmp_path / "novel.csv"
    _raw("Combined").to_csv(combined, index=False)
    _raw("Novel").to_csv(novel, index=False)
    output = tmp_path / "retrieval"
    config = RetrievalAnalysisConfig(
        seed=3, gate_seeds=(), recovery_simulations=0,
        make_plots=True, show_progress=False,
    )
    run_retrieval_analysis(combined, novel, output, config=config)
    required = {
        "context_feature_table.csv", "context_geometry.csv",
        "retrieval_model_fits.csv", "retrieval_predictions.csv",
        "retrieval_responsibilities.csv", "relative_absolute_evidence.csv",
        "continuous_vs_categorical.csv", "participant_AB_separation_test.csv",
        "combined_to_novel_transfer.csv", "novel_to_combined_transfer.csv",
        "tiny_gate_fits.csv", "tiny_gate_states.csv", "distilled_retrieval_rule.csv",
        "retrieval_model_recovery.csv", "retrieval_recovery_confusion.csv",
        "run_config_retrieval.json",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    figures = list((output / "figures_retrieval").glob("*.png"))
    assert len(figures) == 8
