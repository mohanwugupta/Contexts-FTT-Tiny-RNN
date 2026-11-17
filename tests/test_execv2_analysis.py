import numpy as np
import pandas as pd
import torch

from ftt_srn_execution_v2.analysis.hidden_geometry import hidden_state_geometry
from ftt_srn_execution_v2.analysis.interpolation import probe_interpolation
from ftt_srn_execution_v2.config import ExecutionConfig, ModelCondition
from ftt_srn_execution_v2.diagnostics.execution_grounding import execution_grounding_gate
from ftt_srn_execution_v2.diagnostics.transfer_probe import cross_context_transfer
from ftt_srn_execution_v2.diagnostics.transition_costs import transition_costs
from ftt_srn_execution_v2.environment.speed_accuracy import calibrate_speed_accuracy
from ftt_srn_execution_v2.models import DualHeadElmanSRN


def test_interpolation_uses_within_network_endpoints_and_masks_small_denominators():
    rows = []
    for seed, values in ((0, {"A": .8, "B": .4, "Mixed_norm": .6, "Novel": .5}),
                         (1, {"A": .50, "B": .49, "Mixed_norm": .495, "Novel": .495})):
        for context, normalized_rt in values.items():
            rows.append({"seed": seed, "model_condition": "High", "checkpoint": "final",
                         "context": context, "normalized_rt": normalized_rt})
    result = probe_interpolation(pd.DataFrame(rows), reliable_separation=.02)
    mixed = result[(result.seed == 0) & (result.probe_type == "Mixed_norm")].iloc[0]
    close = result[(result.seed == 1) & (result.probe_type == "Mixed_norm")].iloc[0]
    assert np.isclose(mixed.interpolation_index, .5)
    assert bool(mixed.between_endpoints)
    assert np.isnan(close.interpolation_index)
    assert close.interpolation_status == "endpoint_too_close"


def test_hidden_geometry_uses_one_shared_ab_axis():
    rows = []
    positions = {
        "A": [(1, 0), (1, 0)], "B": [(-1, 0), (-1, 0)],
        "Mixed_norm": [(0, 0), (0, 0)], "Novel": [(0, 1), (0, 1)],
    }
    for context, states in positions.items():
        for token_position, (h0, h1) in enumerate(states, 1):
            rows.append({"seed": 0, "model_condition": "High", "checkpoint": "final",
                         "context": context, "token_position": token_position,
                         "execution_speed": .5, "h0": h0, "h1": h1})
    geometry = hidden_state_geometry(pd.DataFrame(rows)).set_index("context")
    assert geometry.loc["A", "ab_centroid_distance"] == 2
    assert geometry.loc["Mixed_norm", "projection_on_ab_axis_b0_a1"] == .5
    assert geometry.loc["Mixed_norm", "off_axis_distance"] == 0
    assert geometry.loc["Novel", "off_axis_distance"] == 1


def test_execution_grounding_requires_within_context_and_controlled_relationship():
    confidence = np.tile(np.linspace(.1, .9, 20), 2)
    frame = pd.DataFrame({
        "correct_token_confidence": confidence,
        "execution_speed": .1 + .8 * confidence,
        "context": np.repeat(["A", "B"], 20),
        "block_index": np.tile(np.repeat([1, 2], 10), 2),
    })
    gate = execution_grounding_gate(frame, calibrate_speed_accuracy(ExecutionConfig()))
    assert gate["passed"] is True
    reversed_frame = frame.copy()
    reversed_frame["execution_speed"] = 1 - reversed_frame.execution_speed
    assert execution_grounding_gate(
        reversed_frame, calibrate_speed_accuracy(ExecutionConfig())
    )["passed"] is False


def test_transfer_probe_is_frozen_and_directionally_complete():
    torch.manual_seed(7)
    model = DualHeadElmanSRN(ModelCondition("High", .5, True))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = cross_context_transfer(model, checkpoint="initial")
    assert set(zip(result.source_context, result.target_context)) == {("A", "B"), ("B", "A")}
    assert (result.probe_optimizer_updates == 0).all()
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before)


def test_transition_cost_matches_validated_absolute_block_boundary_jump():
    frame = pd.DataFrame({
        "global_sequence": [1, 2, 3, 4],
        "block_index": [1, 1, 2, 2],
        "sequence_in_block": [1, 2, 1, 2],
        "sequence_ce": [1.0, .4, .9, .3],
        "execution_cost": [.8, .5, .2, .1],
        "normalized_rt": [.7, .6, .4, .3],
    })
    result = transition_costs(frame, {"final": 4}).iloc[0]
    assert np.isclose(result.sequence_transition_cost, .5)
    assert np.isclose(result.sequence_transition_signed, .5)
    assert np.isclose(result.execution_transition_cost, .3)
    assert np.isclose(result.execution_transition_signed, -.3)
