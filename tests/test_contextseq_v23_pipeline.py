import hashlib
import json

import pandas as pd
import torch

from ftt_srn_context_sequence_interaction.config import smoke_config
from ftt_srn_context_sequence_interaction.pipeline import REQUIRED_CSVS, run_interaction_pipeline
from ftt_srn_execution_v2.config import ModelCondition
from ftt_srn_execution_v2.models import DualHeadElmanSRN


CHECKPOINTS = (
    "early_after_initial_ab", "human_matched", "several_switches", "final"
)


def _frozen_source(root):
    root.mkdir()
    checkpoint_dir = root / "model_checkpoints"
    checkpoint_dir.mkdir()
    manifest = []
    torch.manual_seed(42)
    for seed in (0, 1, 2):
        for checkpoint in CHECKPOINTS:
            model = DualHeadElmanSRN(ModelCondition("High", 0.5, True))
            path = checkpoint_dir / f"seed_{seed}_{checkpoint}.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            manifest.append({
                "seed": seed, "model_condition": "High", "checkpoint": checkpoint,
                "relative_path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    pd.DataFrame(manifest).to_csv(root / "model_checkpoint_manifest.csv", index=False)
    pd.DataFrame({"seed": [0, 1, 2]}).to_csv(root / "seed_summary.csv", index=False)
    pure = []
    for seed in (0, 1, 2):
        for checkpoint in CHECKPOINTS:
            pure.extend([
                {"seed": seed, "model_condition": "High", "checkpoint": checkpoint,
                 "context": "A", "normalized_rt": 0.3},
                {"seed": seed, "model_condition": "High", "checkpoint": checkpoint,
                 "context": "B", "normalized_rt": 0.7},
            ])
    pd.DataFrame(pure).to_csv(root / "pure_context_counterfactuals.csv", index=False)
    (root / "run_config_v2.json").write_text(json.dumps({
        "model_version": "2.1.0", "splitting_gate_passed": True,
        "execution_grounding_gate_passed": True,
    }))


def test_pipeline_loads_frozen_checkpoints_and_writes_complete_contract(tmp_path):
    source = tmp_path / "source"
    _frozen_source(source)
    output = tmp_path / "interaction"
    result = run_interaction_pipeline(source, output, config=smoke_config())
    names = {path.name for path in output.iterdir()}
    assert set(REQUIRED_CSVS).issubset(names)
    assert {"interaction_run_config.json", "interaction_report.md"}.issubset(names)
    assert len(list((output / "figures_context_sequence_v23").glob("*.png"))) == 10
    for filename in REQUIRED_CSVS:
        frame = pd.read_csv(output / filename)
        assert not frame.empty
        if filename != "frozen_checkpoint_audit.csv":
            assert frame.splitting_gate_passed.all()
            assert frame.execution_grounding_gate_passed.all()
    record = json.loads((output / "interaction_run_config.json").read_text())
    assert record["optimizer_steps"] == 0
    assert record["backward_calls"] == 0
    assert record["primary_checkpoint"] == "human_matched"
    assert set(result["full_context_patching"].target_identity_preserved) == {True}
    assert set(result["wrong_token_patching"].wrong_token_is_distinct) == {True}
    assert (
        result["wrong_token_patching"].reference_context_norm
        - result["wrong_token_patching"].patched_context_norm
    ).abs().max() < 1e-8
    synthetic_names = (
        "full_context_patching", "shared_context_patching", "specific_context_patching",
        "wrong_token_patching", "random_direction_controls", "sequence_direction_controls",
        "full_context_ablation", "shared_context_ablation", "specific_context_ablation",
    )
    assert len(result["off_manifold_metrics"]) == sum(
        len(result[name]) for name in synthetic_names
    )
    assert result["sequence_direction_controls"].target_identity_preserved.all()
    assert result["random_direction_controls"].target_identity_preserved.all()
    assert (
        result["random_direction_controls"].reference_context_norm
        - result["random_direction_controls"].control_direction_norm
    ).abs().max() < 1e-8
    audit = result["frozen_checkpoint_audit"]
    assert audit.weights_unchanged.all()
    assert (audit.optimizer_steps == 0).all()
    assert (audit.backward_calls == 0).all()
