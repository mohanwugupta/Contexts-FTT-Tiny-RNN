import json

import numpy as np
import pandas as pd

from human_low_rank_interpretation.config import LowRankInterpretationConfig
from human_low_rank_interpretation.pipeline import (
    REQUIRED_CSVS,
    run_low_rank_interpretation_pipeline,
)


def _schedule(probe_training, probe_test):
    return [
        "Maze Training", "Melody Training", probe_training,
        "Maze Training", "Melody Training", probe_training,
        "Maze Training", "Melody Training", "Maze Training", "Melody Training",
        "Maze Training", "Melody Training", "Maze Training", "Melody Training",
        "Maze Training", "Melody Training", probe_test,
    ]


def _raw(path, dataset):
    probe_training, probe_test = (
        ("Mixed Training", "Mixed Test") if dataset == "Combined"
        else ("Space Training", "test spaceA")
    )
    rows = []
    for participant_index in range(10):
        participant = f"{dataset.lower()}_{participant_index}"
        for block, context in enumerate(_schedule(probe_training, probe_test), start=1):
            for sequence in range(10):
                for position in range(8):
                    raw_position = 2 * position if context.startswith("Maze") else position
                    context_shift = -.10 if context.startswith("Maze") else (
                        .10 if context.startswith("Melody") else .01
                    )
                    log_rt = (
                        6.0 + .01 * participant_index + context_shift
                        + .015 * np.sin(position + participant_index * .1)
                        - .005 * sequence + .001 * block
                    )
                    row = {
                        "ID": participant, "new_trialNumber": block,
                        "sequenceNumber": sequence, "keypressNumber": raw_position,
                        "rt": float(np.exp(log_rt)), "correct": True,
                        "trialType": context, "counterBalance": 2 + participant_index % 3,
                    }
                    rows.append(row)
                    if sequence == 1 and position == participant_index % 7 + 1:
                        rows.append({**row, "rt": row["rt"] * 1.1, "correct": False})
    pd.DataFrame(rows).to_csv(path, index=False)


def _v3_outputs(root):
    root.mkdir()
    pure, mixed, novel = [], [], []
    for dataset, context in (("Combined", "Both"), ("Novel", "Novel")):
        for participant_index in range(10):
            uid = f"{dataset}:{dataset.lower()}_{participant_index}"
            for position in range(1, 8):
                delta = -.22 + .04 * np.sin(position) + .005 * participant_index * (position - 4)
                midpoint = 6.0 + .03 * np.cos(position)
                pure.append({
                    "participant_uid": uid, "dataset": dataset,
                    "transition_position": position, "r_a": midpoint + delta / 2,
                    "r_b": midpoint - delta / 2, "midpoint": midpoint, "delta": delta,
                    "n_a": 50, "n_b": 50, "pure_source_contexts": "A|B",
                    "probe_rows_used_for_profile": 0,
                })
                epochs = ("Mixed1", "Mixed2", "MixedTest") if dataset == "Combined" else (
                    "Novel1", "Novel2", "NovelTest"
                )
                target = mixed if dataset == "Combined" else novel
                for epoch_index, epoch in enumerate(epochs):
                    for repetition in range(1, 5):
                        deviation = (
                            .04 * np.sin(position + epoch_index * .2)
                            + .002 * participant_index
                        )
                        target.append({
                            "participant_uid": uid, "dataset": dataset, "context": context,
                            "transition_position": position, "midpoint_deviation": deviation,
                            "delta": delta, "probe_split": "train" if repetition % 2 else "test",
                            "probe_epoch": epoch, "sequence_repetition": repetition,
                            "unit_id": f"{uid}|{epoch}|r{repetition}|t{position}",
                        })
    pd.DataFrame(pure).to_csv(root / "pure_ab_token_profiles.csv", index=False)
    pd.DataFrame(mixed).to_csv(root / "mixed_midpoint_deviation.csv", index=False)
    pd.DataFrame(novel).to_csv(root / "novel_midpoint_deviation.csv", index=False)
    srn = pd.DataFrame([
        {
            "seed": seed, "transition_position": position,
            "a_rt": .3 + .01 * position + .002 * seed,
            "b_rt": .7 - .01 * position - .002 * seed,
            "row_type": "seed",
        }
        for seed in range(3) for position in range(1, 8)
    ])
    srn.to_csv(root / "srn_mixed_profile.csv", index=False)
    srn.to_csv(root / "srn_novel_profile.csv", index=False)
    pd.DataFrame({
        "seed": range(3), "weights_unchanged": True, "optimizer_steps": 0,
        "backward_calls": 0, "human_outcomes_used": 0,
    }).to_csv(root / "srn_frozen_audit.csv", index=False)
    (root / "human_run_config.json").write_text(json.dumps({
        "analysis_version": "3.0.0", "final_inference_ready": False,
    }))


def test_low_rank_pipeline_writes_complete_frozen_contract(tmp_path):
    v3 = tmp_path / "v3"
    _v3_outputs(v3)
    combined = tmp_path / "combined.csv"; novel = tmp_path / "novel.csv"
    _raw(combined, "Combined"); _raw(novel, "Novel")
    output = tmp_path / "v31"
    config = LowRankInterpretationConfig(
        cv_folds=2, bootstrap_replicates=5, parallel_analysis_replicates=10,
        semantic_recovery_replicates=5, participant_limit_per_dataset=None,
        make_plots=True, show_progress=False, profile="smoke",
    )
    tables = run_low_rank_interpretation_pipeline(
        v3, combined, novel, output, config=config
    )
    names = {path.name for path in output.iterdir()}
    assert set(REQUIRED_CSVS).issubset(names)
    assert {"low_rank_run_config.json", "low_rank_interpretation_report.md"}.issubset(names)
    assert len(list((output / "figures_human_low_rank_v31").glob("*.png"))) == 11
    for filename in REQUIRED_CSVS:
        assert not pd.read_csv(output / filename).empty
    record = json.loads((output / "low_rank_run_config.json").read_text())
    assert record["probe_rows_used_for_primary_basis"] == 0
    assert record["human_outcomes_used_to_modify_srn"] == 0
    assert record["srn_weights_changed"] is False
    assert record["srn_seed_count"] == 3
    assert set(tables["component_loadings"].reference_dataset) == {"Combined"}
    assert tables["transition_feature_table"].feature_table_frozen.all()
    assert not tables["shared_vs_separate_basis"].heldout_participant_in_basis.any()
    comparison = tables["shared_vs_separate_basis"]
    assert {
        "mean_fold_log_likelihood_gain_over_shared",
        "paired_gain_over_shared_se", "comparison_folds",
    }.issubset(comparison.columns)
    assert np.allclose(
        comparison.heldout_log_likelihood_gain_over_shared,
        comparison.mean_fold_log_likelihood_gain_over_shared * comparison.comparison_folds,
    )
    assert tables["cross_dataset_component_alignment"].independent_before_alignment.all()
