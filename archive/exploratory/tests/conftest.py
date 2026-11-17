import pandas as pd
import pytest


@pytest.fixture
def keypress_probe_frame():
    """Small keypress-level schedule with two primary Both probe blocks."""
    rows = []
    schedule = [
        (1, "Maze Training"),
        (2, "Melody Training"),
        (3, "Mixed Training"),
        (4, "Mixed Training"),
        (5, "Maze Test"),
        (6, "test melodyA"),
    ]
    for pid_i, pid in enumerate(("P1", "P2")):
        for block, label in schedule:
            for seq in range(2):
                for key in range(4):
                    rows.append(
                        {
                            "ID": pid,
                            "new_trialNumber": block,
                            "sequenceNumber": seq,
                            "keypressNumber": key,
                            "trialType": label,
                            "counterBalance": "Maze-Melody-Mixed",
                            "correct": not (pid_i == 1 and block == 3 and seq == 1 and key == 3),
                            "rt": 400 + 20 * pid_i + 10 * block + key,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def normalized_schedule():
    rows = []
    contexts = ["A"] * 4 + ["B"] * 4 + ["Probe"] * 3 + ["A"] * 3 + ["Probe"] * 3 + ["B"] * 3
    for pid_i, pid in enumerate(("P1", "P2", "P3", "P4")):
        for t, context in enumerate(contexts):
            block = 1 + t // 3
            seq = t % 3 + 1
            is_probe = context == "Probe"
            a = 6.0 - 0.012 * t + 0.02 * pid_i
            b = 6.3 - 0.010 * t + 0.02 * pid_i
            y = (0.65 * a + 0.35 * b) if is_probe else (a if context == "A" else b)
            rows.append(
                {
                    "participant_id": pid,
                    "dataset": "Combined",
                    "block_id": block,
                    "sequence_in_block": seq,
                    "global_sequence": t + 1,
                    "context": context,
                    "probe_type": "Both" if is_probe else None,
                    "is_probe": is_probe,
                    "is_primary_probe": is_probe,
                    "probe_block_number": (1 if 8 <= t <= 10 else 2) if is_probe else pd.NA,
                    "previous_context": "B" if 8 <= t <= 10 else ("A" if 14 <= t <= 16 else None),
                    "next_context": "A" if 8 <= t <= 10 else ("B" if 14 <= t <= 16 else None),
                    "mean_rt": float(__import__("numpy").exp(y)),
                    "log_rt": y,
                    "error": 0.0,
                    "break_duration": float("nan"),
                    "counterbalance": "Maze-Melody-Mixed",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def epoch_keypress_frame():
    """Two mirrored 17-block Combined schedules at keypress resolution."""
    rows = []
    schedules = {
        "A_before_Mixed": [
            "Melody Training", "Maze Training", "Melody Training", "Maze Training",
            "Mixed Training", "Mixed Training",
            "Melody Training", "Maze Training", "Melody Training", "Maze Training",
            "Melody Training", "Maze Training", "Melody Training", "Maze Training",
            "test melodyA", "Maze Test", "Mixed Test",
        ],
        "B_before_Mixed": [
            "Maze Training", "Melody Training", "Maze Training", "Melody Training",
            "Mixed Training", "Mixed Training",
            "Maze Training", "Melody Training", "Maze Training", "Melody Training",
            "Maze Training", "Melody Training", "Maze Training", "Melody Training",
            "Maze Test", "test melodyA", "Mixed Test",
        ],
    }
    for cb_i, (counterbalance, labels) in enumerate(schedules.items()):
        for person in range(2):
            pid = f"{counterbalance[:1]}{cb_i}{person}"
            for block, label in enumerate(labels, start=1):
                for raw_sequence in range(10):
                    for key in range(8):
                        rows.append(
                            {
                                "ID": pid,
                                "new_trialNumber": block,
                                "sequenceNumber": raw_sequence,
                                "keypressNumber": key,
                                "trialType": label,
                                "counterBalance": counterbalance,
                                "correct": True,
                                "rt": 360 + 25 * cb_i + 4 * block + 2 * raw_sequence + key,
                            }
                        )
                    # An intervening error does not make a subsequently completed
                    # eight-key sequence incomplete.
                    if block == 3 and raw_sequence == 4 and person == 0:
                        rows.append(
                            {
                                "ID": pid,
                                "new_trialNumber": block,
                                "sequenceNumber": raw_sequence,
                                "keypressNumber": 99,
                                "trialType": label,
                                "counterBalance": counterbalance,
                                "correct": False,
                                "rt": 500,
                            }
                        )
    return pd.DataFrame(rows)
