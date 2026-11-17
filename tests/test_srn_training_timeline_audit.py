from __future__ import annotations

import pandas as pd
import pytest
import torch

from confirmatory_srn.config import ConfirmatoryConfig
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from srn_training_timeline_audit.core import (
    assert_canonical_checkpoint, frozen_schedule_detector, schedule_hash,
)


def _payload(hidden=8, scale=2.0):
    return {"init_scale": scale, "model_state_dict": {
        "recurrent.weight_hh_l0": torch.zeros(hidden, hidden),
        "recurrent.weight_ih_l0": torch.zeros(hidden, 7),
        "sequence_head.weight": torch.zeros(4, hidden),
    }}


def test_architecture_guard_rejects_old_six_unit_checkpoint():
    with pytest.raises(AssertionError, match="FAIL: hidden_size mismatch"):
        assert_canonical_checkpoint(_payload(hidden=6), "High_monotonic")


def test_architecture_guard_accepts_canonical_scales():
    assert assert_canonical_checkpoint(_payload(), "High_monotonic") == (8, 7, 4)
    assert assert_canonical_checkpoint(_payload(scale=.02), "Low_monotonic") == (8, 7, 4)


def test_architecture_guard_rejects_wrong_scale():
    with pytest.raises(AssertionError, match="initialization_scale mismatch"):
        assert_canonical_checkpoint(_payload(scale=.5), "High_monotonic")


def test_frozen_schedule_detector_distinguishes_acquisition():
    frozen = pd.DataFrame({"seed": [0]*6, "context": ["A","A","A","B","B","B"],
                           "raw_sequence_loss": [1.,1.,1.,2.,2.,2.]})
    acquisition = frozen.copy(); acquisition["raw_sequence_loss"] = [3.,2.,1.,4.,3.,2.]
    assert frozen_schedule_detector(frozen)
    assert not frozen_schedule_detector(acquisition)


def test_schedule_is_exact_canonical_order_and_hash_is_stable():
    config = ConfirmatoryConfig(show_progress=False)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    assert len(schedule) == 170
    assert schedule.groupby("block_index").size().eq(10).all()
    assert schedule.groupby("block_index").context.first().tolist() == [
        "A" if index % 2 else "B" for index in range(1, 18)]
    assert schedule_hash(config) == schedule_hash(config)
