import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ftt_srn_execution_v2.config import FIXED_SEQUENCE, final_config
from ftt_srn_execution_v2.inputs.context_basis import context_vector
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_paper_entrypoints_and_configs_exist():
    required = [
        "configs/srn_final.json", "configs/human_final.json",
        "scripts/run_srn_final.py", "scripts/run_human_final.py",
        "scripts/run_all_final.py", "scripts/make_paper_figures.py",
        "docs/ANALYSIS_OVERVIEW.md", "docs/REPRODUCIBILITY.md",
        "docs/RESULT_MAP.md", "README.md",
    ]
    assert all((ROOT / path).is_file() for path in required)


def test_frozen_sequence_context_curriculum_and_scale_contract():
    config = final_config()
    assert FIXED_SEQUENCE == (0, 1, 2, 0, 3, 1, 2, 3)
    assert config.fixed_sequence == FIXED_SEQUENCE
    assert config.hidden_dim == 8
    assert [(condition.name, condition.init_scale) for condition in config.conditions] == [
        ("High", .5), ("Low", .02), ("High_seq_only", .5),
    ]
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    assert len(schedule) == 170
    for block, frame in schedule.groupby("block_index"):
        assert len(frame) == 10
        assert frame.context.unique().tolist() == ["A" if block % 2 else "B"]
    np.testing.assert_array_equal(context_vector("A"), [1, 0, 0])
    np.testing.assert_array_equal(context_vector("B"), [0, 1, 0])
    np.testing.assert_allclose(
        context_vector("Mixed_norm"), [1 / np.sqrt(2), 1 / np.sqrt(2), 0]
    )
    np.testing.assert_array_equal(context_vector("Novel"), [0, 0, 1])


def test_frozen_outputs_preserve_no_update_and_no_leakage_guards():
    execution = json.loads((
        ROOT / "outputs/ftt_srn_execution_v2_final_checkpointed/run_config_v2.json"
    ).read_text())
    human = json.loads((ROOT / "outputs/human_configured_state_v3/human_run_config.json").read_text())
    low_rank = json.loads((
        ROOT / "outputs/human_low_rank_interpretation_v31/low_rank_run_config.json"
    ).read_text())
    assert execution["mixed_novel_training_updates"] == 0
    assert execution["novel_context_column_frozen_zero"] is True
    assert human["probe_rows_used_for_pure_profiles"] == 0
    assert human["human_outcomes_used_for_srn_seed_selection"] == 0
    assert low_rank["probe_rows_used_for_primary_basis"] == 0
    assert low_rank["srn_weights_changed"] is False


def test_frozen_human_rank_model_order_and_headline_schemas():
    low_rank = json.loads((
        ROOT / "outputs/human_low_rank_interpretation_v31/low_rank_run_config.json"
    ).read_text())
    assert low_rank["selected_rank"] == 2
    assert low_rank["candidate_ranks"] == [1, 2, 3, 7]
    expected_order = {
        "mixed_model_comparison.csv": [
            "M3_low_rank", "M1_scalar_mixture", "M2_token_specific",
            "M4_srn_informed", "M0_scalar_context",
        ],
        "novel_model_comparison.csv": [
            "M3_low_rank", "M2_token_specific", "M1_scalar_mixture",
            "M4_srn_informed", "M0_scalar_context",
        ],
    }
    for filename, order in expected_order.items():
        frame = pd.read_csv(ROOT / "outputs/human_configured_state_v3" / filename)
        assert frame.sort_values("primary_rank").model.tolist() == order
        assert {
            "model", "heldout_log_likelihood", "heldout_rmse",
            "heldout_participants", "primary_rank",
        }.issubset(frame.columns)
    assert {
        "component", "transition_position", "loading", "stable_component",
        "interpretation_label", "semantic_label_assigned",
    }.issubset(pd.read_csv(
        ROOT / "outputs/human_low_rank_interpretation_v31/component_loadings.csv"
    ).columns)
    assert {
        "seed", "checkpoint", "probe_context", "correct_token_advantage",
        "full_prediction_identity_preservation",
    }.issubset(pd.read_csv(
        ROOT / "outputs/ftt_srn_context_sequence_v23_final/token_structured_causal_effects.csv"
    ).columns)


def test_frozen_headline_csvs_match_pre_cleanup_manifest():
    manifest = json.loads((ROOT / "reproducibility/baseline_manifest.json").read_text())
    for analysis in manifest["frozen_outputs"].values():
        for record in analysis["csvs"]:
            path = ROOT / record["path"]
            assert path.is_file()
            assert _sha(path) == record["sha256"]


def test_frozen_srn_checkpoint_manifest_matches_checkpoint_bytes():
    output_root = ROOT / "outputs/ftt_srn_execution_v2_final_checkpointed"
    manifest = pd.read_csv(output_root / "model_checkpoint_manifest.csv")
    assert len(manifest) == 240
    assert manifest[["model_condition", "seed", "checkpoint"]].duplicated().sum() == 0
    for record in manifest.itertuples(index=False):
        checkpoint = output_root / record.relative_path
        assert checkpoint.is_file()
        assert _sha(checkpoint) == record.sha256


def test_paper_figure_bundle_has_complete_hash_traced_sources():
    provenance = json.loads((
        ROOT / "outputs/final/paper_figure_sources.json"
    ).read_text())
    assert provenance["schema_version"] == "1.0.0"
    assert len(provenance["figures"]) == 5
    assert len(provenance["source_tables"]) == 17
    for filename in provenance["figures"]:
        figure = ROOT / "outputs/final/figures" / filename
        assert figure.is_file()
        assert figure.stat().st_size > 10_000
    for record in provenance["source_tables"]:
        source = ROOT / record["source"]
        paper_copy = ROOT / record["paper_copy"]
        assert source.is_file() and paper_copy.is_file()
        assert _sha(source) == record["sha256"]
        assert _sha(paper_copy) == record["sha256"]
