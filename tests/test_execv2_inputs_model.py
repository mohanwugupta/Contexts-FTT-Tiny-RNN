import numpy as np
import torch

from ftt_srn_execution_v2.config import ModelCondition
from ftt_srn_execution_v2.inputs.context_basis import context_vector
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.models import DualHeadElmanSRN


def test_tokens_are_strict_four_dimensional_onehots():
    encoded = encode_sequence([0, 1, 2, 0, 3, 1, 2, 3])
    assert encoded.shape == (8, 4)
    np.testing.assert_allclose(encoded.sum(axis=1), 1.0)
    assert set(np.unique(encoded)) == {0.0, 1.0}


def test_three_dimensional_context_basis_and_norm_matched_mixed():
    np.testing.assert_array_equal(context_vector("A"), [1, 0, 0])
    np.testing.assert_array_equal(context_vector("B"), [0, 1, 0])
    np.testing.assert_array_equal(context_vector("Novel"), [0, 0, 1])
    mixed = context_vector("Mixed_norm")
    np.testing.assert_allclose(mixed, [1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    assert np.isclose(np.linalg.norm(mixed), 1.0)
    np.testing.assert_array_equal(context_vector("Mixed_additive"), [1, 1, 0])


def test_high_low_initialization_and_neutral_novel_column():
    for scale_name, expected_sd in (("High", 0.5), ("Low", 0.02)):
        torch.manual_seed(11)
        model = DualHeadElmanSRN(ModelCondition(scale_name, expected_sd, True))
        weights = model.non_exempt_initial_weights()
        assert abs(float(weights.std(unbiased=False)) - expected_sd) < max(0.08, expected_sd * 0.20)
        np.testing.assert_allclose(model.novel_context_column().detach().numpy(), 0.0)
        assert model.context_alpha == 1.0

