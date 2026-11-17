from pathlib import Path

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import ConfirmatoryElmanSRN
from srn_mixed_novel_diagnostics.config import DiagnosticConfig
from srn_mixed_novel_diagnostics.core import axis_geometry, decompose_vectors, factorization, output_projectors
from srn_mixed_novel_diagnostics.pipeline import TABLE_NAMES, run_pipeline


def test_axis_factorization_and_subspace_invariants():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(7, 8)); b = rng.normal(size=(7, 8))
    ga = axis_geometry(a, b, a); gb = axis_geometry(a, b, b)
    assert np.allclose(ga["q"], 0) and np.allclose(gb["q"], 1)
    assert ga["mean_off_axis_residual"] < 1e-10 and gb["mean_off_axis_residual"] < 1e-10
    values = factorization(a, b)
    assert np.isclose(values["P_sequence"] + values["P_context"] + values["P_interaction"], 1)
    row, null, _ = output_projectors(rng.normal(size=(4, 8)))
    relevant, null_values, relevant_fraction, null_fraction = decompose_vectors(b-a, row, null)
    assert np.allclose(relevant + null_values, b-a)
    assert np.isclose(relevant_fraction + null_fraction, 1)


def _write_checkpoint(path: Path, condition: str, seed: int, scale: float):
    torch.manual_seed(seed + (0 if condition.startswith("High") else 100))
    model = ConfirmatoryElmanSRN(hidden_dim=8, init_scale=scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "seed": seed, "init_scale": scale,
                "model_state_dict": model.state_dict()}, path)


def test_pipeline_smoke_obeys_firewalls_and_writes_contract(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    for condition, scale in (("High_monotonic", 2.), ("Low_monotonic", .02)):
        _write_checkpoint(checkpoints / f"{condition}_seed_0.pt", condition, 0, scale)
    output = tmp_path / "output"
    config = DiagnosticConfig(checkpoint_dir=checkpoints, output_dir=output, seeds=(0,),
                              make_plots=True, profile="test")
    tables = run_pipeline(config)
    assert set(tables) == set(TABLE_NAMES)
    assert (output / "SRN_MIXED_NOVEL_REPRESENTATION_DIAGNOSTICS.md").exists()
    for name in TABLE_NAMES:
        frame = pd.read_csv(output / f"{name}.csv")
        assert "seed" in frame and frame.seed.notna().all()
    audit = pd.read_csv(output / "frozen_evaluation_audit.csv")
    assert audit.parameters_unchanged.all()
    assert audit.optimizer_steps.sum() == 0
    assert audit.human_rows_used.sum() == 0
    assert audit.probe_training_rows.sum() == 0
    decoder = tables["sequence_decoder_results"]
    assert set(decoder.decoder_training_data) == {"familiar_A_B_only"}
    assert tables["novel_input_audit"].UNTRAINED_NOVEL_INPUT_DIRECTION.all()
    assert tables["history_dependence"].groupby(["condition", "seed", "probe_context"]).current_probe_input_sha256.nunique().max() == 1
    registered = tables["history_dependence"].query("history_intervention_type == 'registered_reset_protocol'")
    assert registered.groupby(["condition", "seed", "probe_context"]).sequence_loss.nunique().max() == 1
    for number in range(1, 11):
        assert len(list(output.glob(f"figure_{number:02d}_*.png"))) == 1
