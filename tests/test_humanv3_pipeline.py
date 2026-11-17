import hashlib
import json

import numpy as np
import pandas as pd
import torch

from ftt_srn_execution_v2.config import ModelCondition
from ftt_srn_execution_v2.models import DualHeadElmanSRN
from human_configured_state.config import HumanConfiguredStateConfig
from human_configured_state.pipeline import REQUIRED_CSVS, run_human_configured_state_pipeline


def _frozen_source(root):
    root.mkdir()
    checkpoint_dir = root / "model_checkpoints"
    checkpoint_dir.mkdir()
    manifest = []
    for seed in (0, 1, 2):
        torch.manual_seed(900 + seed)
        model = DualHeadElmanSRN(ModelCondition("High", 0.5, True))
        path = checkpoint_dir / f"seed_{seed}_human_matched.pt"
        torch.save({"state_dict": model.state_dict()}, path)
        manifest.append({
            "seed": seed,
            "model_condition": "High",
            "checkpoint": "human_matched",
            "relative_path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    pd.DataFrame(manifest).to_csv(root / "model_checkpoint_manifest.csv", index=False)
    pd.DataFrame({"seed": [0, 1, 2]}).to_csv(root / "seed_summary.csv", index=False)
    pd.DataFrame([
        {
            "seed": seed,
            "model_condition": "High",
            "checkpoint": "human_matched",
            "context": context,
            "normalized_rt": normalized_rt,
        }
        for seed in (0, 1, 2)
        for context, normalized_rt in (("A", 0.3), ("B", 0.7))
    ]).to_csv(root / "pure_context_counterfactuals.csv", index=False)
    (root / "run_config_v2.json").write_text(json.dumps({
        "model_version": "2.1.0",
        "splitting_gate_passed": True,
        "execution_grounding_gate_passed": True,
    }))


def _schedule(probe_training, probe_test):
    return [
        "Maze Training", "Melody Training", probe_training,
        "Maze Training", "Melody Training", probe_training,
        "Maze Training", "Melody Training", "Maze Training", "Melody Training",
        "Maze Training", "Melody Training", "Maze Training", "Melody Training",
        "Maze Training", "Melody Training", probe_test,
    ]


def _human_csv(path, *, dataset):
    probe_training, probe_test = (
        ("Mixed Training", "Mixed Test")
        if dataset == "Combined"
        else ("Space Training", "test spaceA")
    )
    rows = []
    for participant_index in range(6):
        participant = f"{dataset.lower()}_{participant_index}"
        counterbalance = 2 + participant_index % 3
        participant_shift = 0.015 * participant_index
        for block, context in enumerate(_schedule(probe_training, probe_test), start=1):
            if context.startswith("Maze"):
                context_shift = -0.12
            elif context.startswith("Melody"):
                context_shift = 0.12
            elif dataset == "Combined":
                context_shift = 0.025
            else:
                context_shift = -0.015
            for sequence in range(10):
                for ordinal_position in range(8):
                    raw_keypress = 2 * ordinal_position if context.startswith("Maze") else ordinal_position
                    token_shape = (ordinal_position - 3.5) * (
                        0.008 if context.startswith("Maze") else -0.006
                    )
                    probe_shape = 0.0
                    if context in {probe_training, probe_test}:
                        probe_shape = 0.025 * np.sin(ordinal_position + participant_index * 0.2)
                    log_rt = (
                        6.15 + participant_shift + context_shift + token_shape + probe_shape
                        - 0.006 * sequence + 0.001 * block
                    )
                    rows.append({
                        "ID": participant,
                        "new_trialNumber": block,
                        "sequenceNumber": sequence,
                        "keypressNumber": raw_keypress,
                        "rt": float(np.exp(log_rt)),
                        "correct": True,
                        "trialType": context,
                        "counterBalance": counterbalance,
                    })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_human_v3_pipeline_writes_complete_leak_free_contract(tmp_path):
    source = tmp_path / "source"
    _frozen_source(source)
    combined = tmp_path / "combined.csv"
    novel = tmp_path / "novel.csv"
    _human_csv(combined, dataset="Combined")
    _human_csv(novel, dataset="Novel")
    output = tmp_path / "human_v3"
    config = HumanConfiguredStateConfig(
        srn_endpoint_epsilon_rt=1e-8,
        cv_folds=3,
        recovery_replicates=1,
        participant_limit_per_dataset=None,
        make_plots=True,
        show_progress=False,
        profile="smoke",
    )
    tables = run_human_configured_state_pipeline(
        combined, novel, source, output, config=config
    )

    names = {path.name for path in output.iterdir()}
    assert set(REQUIRED_CSVS).issubset(names)
    assert {"human_run_config.json", "human_behavioral_report.md"}.issubset(names)
    assert len(list((output / "figures_human_configured_v3").glob("*.png"))) == 11
    for filename in REQUIRED_CSVS:
        assert not pd.read_csv(output / filename).empty

    record = json.loads((output / "human_run_config.json").read_text())
    assert record["probe_rows_used_for_pure_profiles"] == 0
    assert record["human_outcomes_used_for_srn_seed_selection"] == 0
    assert record["human_outcomes_used_for_srn_profile_shape"] == 0
    assert record["srn_seed_count"] == 3
    assert record["srn_aggregation"] == "all_registered_seed_mean"
    assert record["participants"] == {"Combined": 6, "Novel": 6}

    profiles = tables["pure_ab_token_profiles"]
    assert set(profiles.pure_source_contexts) == {"A|B"}
    assert set(profiles.transition_position) == set(range(1, 8))
    audit = tables["srn_frozen_audit"]
    assert audit.weights_unchanged.all()
    assert (audit.optimizer_steps == 0).all()
    assert (audit.backward_calls == 0).all()
    assert (audit.human_outcomes_used == 0).all()

    for model_name in (
        "model_M0_scalar_context", "model_M1_scalar_mixture",
        "model_M2_token_specific", "model_M3_low_rank", "model_M4_srn_informed",
    ):
        predictions = tables[model_name]
        assert not predictions.heldout_participant_in_group_training.any()
        assert predictions.target_transition_preserved.all()
        assert predictions.ab_profiles_frozen_before_probe_fit.all()
        assert set(predictions.probe_context) == {"Both", "Novel"}
    assert set(tables["shared_vs_separate_probe_profile"].model) == {
        "shared_profile", "separate_profile",
    }
