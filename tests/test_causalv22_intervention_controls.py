import numpy as np
import torch

from ftt_srn_causal_representation.controls.off_manifold import NaturalManifold
from ftt_srn_causal_representation.controls.random_directions import orthogonal_random_direction
from ftt_srn_causal_representation.interventions.readout import readout_hidden
from ftt_srn_causal_representation.interventions.recurrent_rollout import recurrent_rollout
from ftt_srn_causal_representation.interventions.steering import steer_hidden
from ftt_srn_execution_v2.config import FIXED_SEQUENCE, ModelCondition
from ftt_srn_execution_v2.inputs.context_basis import context_vector
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.models import DualHeadElmanSRN


def test_random_control_is_norm_matched_and_orthogonal_when_possible():
    direction = np.array([3.0, 4.0, 0.0, 0.0])
    basis = (direction / np.linalg.norm(direction)).reshape(-1, 1)
    random = orthogonal_random_direction(direction, basis, rng=np.random.default_rng(2))
    assert np.isclose(np.linalg.norm(random), np.linalg.norm(direction))
    assert abs(float(random @ basis[:, 0])) < 1e-9


def test_zero_steering_is_exact_readout_identity_and_does_not_mutate_state():
    torch.manual_seed(3)
    model = DualHeadElmanSRN(ModelCondition("High", .5, True))
    hidden = torch.randn(8)
    stored = hidden.clone()
    direction = np.arange(1, 9, dtype=float)
    intervened = steer_hidden(hidden.numpy(), direction, alpha=0.0)
    natural = readout_hidden(model, hidden.numpy(), target_token=2)
    zero = readout_hidden(model, intervened, target_token=2)
    assert natural == zero
    assert torch.equal(hidden, stored)


def test_recurrent_rollout_feeds_modified_state_forward_and_keeps_targets_fixed():
    torch.manual_seed(5)
    model = DualHeadElmanSRN(ModelCondition("High", .5, True))
    tokens = encode_sequence(FIXED_SEQUENCE[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.tensor(FIXED_SEQUENCE[1:], dtype=torch.long)
    context = context_vector("A", as_tensor=True).unsqueeze(0)
    with torch.no_grad():
        _, _, hidden, _ = model(tokens, context)
    start = 2
    natural = recurrent_rollout(model, tokens, context, targets, start, hidden[0, start].numpy())
    shifted_state = hidden[0, start].numpy().copy()
    shifted_state[0] += .5
    shifted = recurrent_rollout(model, tokens, context, targets, start, shifted_state)
    assert list(natural.target_token) == list(targets[start:].numpy())
    assert list(shifted.target_token) == list(natural.target_token)
    assert np.isclose(natural.iloc[0].execution_speed,
                      readout_hidden(model, hidden[0, start].numpy(), int(targets[start]))["execution_speed"])
    assert not np.allclose(shifted.execution_speed.iloc[1:], natural.execution_speed.iloc[1:])


def test_off_manifold_metrics_are_finite_for_nonzero_intervention():
    natural = np.vstack([np.eye(8), -np.eye(8), np.zeros((1, 8))])
    manifold = NaturalManifold.fit(natural)
    metrics = manifold.metrics(np.full(8, .25))
    assert metrics["nearest_natural_euclidean"] > 0
    assert np.isfinite(metrics["mahalanobis_distance"])
    assert np.isfinite(metrics["hidden_state_norm"])
