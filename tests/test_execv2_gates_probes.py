import numpy as np

from ftt_srn_execution_v2.config import ExperimentConfig, ModelCondition
from ftt_srn_execution_v2.diagnostics.splitting_gate import splitting_gate
from ftt_srn_execution_v2.probes.counterfactuals import run_counterfactual_probes
from ftt_srn_execution_v2.training.trainer import train_network


def test_splitting_gate_passes_separated_and_fails_identical_geometry():
    passed = splitting_gate(
        high_separation=1.0, low_separation=0.2,
        high_transition_cost=.3, low_transition_cost=.05,
        high_mastery=.95, low_mastery=.95, high_context_weight_norm=.8,
        high_transfer_cost=.2, low_transfer_cost=.02,
    )
    failed = splitting_gate(
        high_separation=.2, low_separation=.2,
        high_transition_cost=.05, low_transition_cost=.05,
        high_mastery=.95, low_mastery=.95, high_context_weight_norm=.8,
        high_transfer_cost=.02, low_transfer_cost=.02,
    )
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_probe_counterfactuals_change_only_context_and_never_weights():
    config = ExperimentConfig(n_blocks=2, checkpoint_sequences=(20,), show_progress=False)
    result = train_network(ModelCondition("High", .5, True), config, seed=4)
    before = {name: value.detach().clone() for name, value in result.model.state_dict().items()}
    rows, hidden = run_counterfactual_probes(result.model, checkpoint_label="final")
    after = result.model.state_dict()
    assert set(rows.context) == {"A", "B", "Mixed_norm", "Mixed_additive", "Novel"}
    assert rows.network_weight_identity.nunique() == 1
    assert rows.sequence_input_identity.nunique() == 1
    assert len(hidden) == 5 * 7
    assert all(np.array_equal(before[name], after[name]) for name in before)
    assert result.optimizer_steps == 20

