import numpy as np
import torch

from ftt_srn_execution_v2.config import ExecutionConfig, ExperimentConfig, ModelCondition
from ftt_srn_execution_v2.environment.speed_accuracy import (
    calibrate_speed_accuracy,
    execution_cost,
)
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from ftt_srn_execution_v2.training.trainer import train_network


def test_execution_calibration_is_monotone_and_has_interior_optima():
    table = calibrate_speed_accuracy(ExecutionConfig())
    ordered = table.sort_values("sequence_confidence")
    assert np.all(np.diff(ordered.optimal_speed) >= -1e-12)
    relevant = ordered[ordered.sequence_confidence.between(.4, .95)]
    assert relevant.optimal_speed.between(0.0, 1.0, inclusive="neither").any()
    assert relevant.iloc[-1].optimal_speed > relevant.iloc[0].optimal_speed


def test_execution_confidence_is_detached_from_cost_graph():
    confidence = torch.tensor([0.7], requires_grad=True)
    speed = torch.tensor([0.4], requires_grad=True)
    loss, _ = execution_cost(confidence, speed, ExecutionConfig())
    loss.sum().backward()
    assert confidence.grad is None
    assert speed.grad is not None


def test_ftt_schedule_is_exact_and_contains_only_pure_contexts():
    schedule = make_training_schedule(n_blocks=17, block_length=10)
    assert len(schedule) == 170
    assert set(schedule.context) == {"A", "B"}
    for block, group in schedule.groupby("block_index"):
        assert len(group) == 10
        assert group.context.nunique() == 1
        assert group.context.iloc[0] == ("A" if block % 2 == 1 else "B")


def test_training_preserves_novel_neutrality_and_sequence_only_learns():
    config = ExperimentConfig(n_blocks=4, checkpoint_sequences=(20, 40), show_progress=False)
    condition = ModelCondition("High_seq_only", 0.5, False)
    result = train_network(condition, config, seed=2)
    np.testing.assert_allclose(result.model.novel_context_column().detach().numpy(), 0.0)
    assert result.optimizer_steps == 40
    # A four-block smoke fit only needs to show that the deterministic skill is
    # being acquired; mastery is evaluated at the preregistered 17-block end.
    assert result.training_dynamics.sequence_accuracy.iloc[-1] >= .70
    assert (
        result.training_dynamics.sequence_accuracy.iloc[-10:].mean()
        > result.training_dynamics.sequence_accuracy.iloc[:10].mean()
    )
    assert not hasattr(result, "competence_a")
    assert not hasattr(result, "competence_b")
