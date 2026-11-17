"""Exact mapping from the human 17-block experiment to model update indices."""

from __future__ import annotations

import pandas as pd


TRAINING_BLOCKS = {1: "A", 2: "B", 3: "A", 4: "B", 7: "A", 8: "B", 9: "A", 10: "B", 11: "A", 12: "B", 13: "A", 14: "B"}


def build_timeline() -> pd.DataFrame:
    rows, completed = [], 0
    for block in range(1, 18):
        if block in TRAINING_BLOCKS:
            context = TRAINING_BLOCKS[block]
            phase = f"familiar {context}" if block <= 4 else f"return-to-{context}"
            for sequence in range(1, 11):
                completed += 1
                rows.append({"human_phase": phase, "human_block": block, "human_sequence_index": sequence,
                             "probe_name": None, "model_training_sequence_index": completed,
                             "weights_frozen_before_probe": False, "context": context, "updates_weights": True})
        elif block == 5:
            for probe in ("Mixed1", "Novel1", "NoContext1"):
                rows.append({"human_phase": probe, "human_block": block, "human_sequence_index": 1,
                             "probe_name": probe, "model_training_sequence_index": completed,
                             "weights_frozen_before_probe": True, "context": probe.rstrip("1"), "updates_weights": False})
        elif block == 6:
            for probe in ("Mixed2", "Novel2", "NoContext2"):
                rows.append({"human_phase": probe, "human_block": block, "human_sequence_index": 1,
                             "probe_name": probe, "model_training_sequence_index": completed,
                             "weights_frozen_before_probe": True, "context": probe.rstrip("2"), "updates_weights": False})
        elif block == 15:
            rows.append({"human_phase": "familiar A test", "human_block": block, "human_sequence_index": 1,
                         "probe_name": "FamiliarATest", "model_training_sequence_index": completed,
                         "weights_frozen_before_probe": True, "context": "A", "updates_weights": False})
        elif block == 16:
            rows.append({"human_phase": "familiar B test", "human_block": block, "human_sequence_index": 1,
                         "probe_name": "FamiliarBTest", "model_training_sequence_index": completed,
                         "weights_frozen_before_probe": True, "context": "B", "updates_weights": False})
        else:
            for probe in ("MixedTest", "NovelTest", "NoContextTest"):
                rows.append({"human_phase": probe, "human_block": block, "human_sequence_index": 1,
                             "probe_name": probe, "model_training_sequence_index": completed,
                             "weights_frozen_before_probe": True,
                             "context": "NoContext" if probe.startswith("NoContext") else probe.replace("Test", ""),
                             "updates_weights": False})
    frame = pd.DataFrame(rows)
    assert completed == 120
    mapping = frame.dropna(subset=["probe_name"]).set_index("probe_name").model_training_sequence_index.to_dict()
    expected = {"Mixed1": 40, "Mixed2": 40, "MixedTest": 120, "Novel1": 40, "Novel2": 40, "NovelTest": 120,
                "NoContext1": 40, "NoContext2": 40, "NoContextTest": 120, "FamiliarATest": 120, "FamiliarBTest": 120}
    if mapping != expected: raise AssertionError(f"T3: human timeline mismatch: {mapping}")
    return frame


def training_schedule(timeline: pd.DataFrame) -> pd.DataFrame:
    return timeline[timeline.updates_weights].sort_values("model_training_sequence_index").reset_index(drop=True)
