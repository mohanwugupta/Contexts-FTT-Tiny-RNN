import numpy as np
import torch
from torch import nn

from srn_environmental_pressures.analysis.success_gate import evaluate_success
from srn_environmental_pressures.config import EnvironmentalCondition, ExperimentConfig
from srn_environmental_pressures.models import ElmanSRN
from srn_environmental_pressures.training.trainer import train_condition


def test_primary_network_is_fixed_eight_unit_elman_srn():
    model = ElmanSRN()
    assert model.hidden_dim == 8
    assert isinstance(model.recurrent, nn.RNN)
    assert model.recurrent.nonlinearity == "tanh"
    assert not any(isinstance(module, nn.GRU) for module in model.modules())
    prediction, hidden, final = model(torch.zeros(3, 9, 4))
    assert prediction.shape == (3, 9)
    assert hidden.shape == (3, 9, 8)
    assert final.shape == (1, 3, 8)


def test_flat_midpoint_is_explicitly_rejected_by_success_gate():
    q = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95])
    rows = {
        "q_star": q,
        "skill_a": np.full_like(q, 0.2),
        "skill_b": np.full_like(q, 0.8),
        "bayes_mean": q * 0.2 + (1 - q) * 0.8,
        "srn_prediction": np.full_like(q, 0.5),
        "phase_only_prediction": np.where(q < 0.5, 0.65, 0.35),
        "cue_only_prediction": q * 0.2 + (1 - q) * 0.8 + 0.03,
        "hard_map_prediction": np.where(q >= 0.5, 0.2, 0.8),
        "global_midpoint_prediction": np.full_like(q, 0.5),
        "matched_state_history_id": np.zeros_like(q),
    }
    summary = evaluate_success(rows)
    assert summary["flat_midpoint_rejected"] is False
    assert summary["passed"] is False


def test_training_has_one_fixed_update_per_trial_and_pair_checkpoints():
    config = ExperimentConfig(
        n_blocks=4, block_length=10, training_batch_size=8,
        diagnostic_histories=2,
    )
    condition = EnvironmentalCondition(0.75, 0.80, 0.60, 0.01, "fixed_global")
    result = train_condition(condition, config, seed=3, evaluate_checkpoints=False)
    assert len(result.training_history) == 40
    assert result.weight_updates == 40
    assert sorted(result.checkpoints) == [1, 2]

