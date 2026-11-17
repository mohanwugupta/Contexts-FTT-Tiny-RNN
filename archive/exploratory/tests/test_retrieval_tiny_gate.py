import numpy as np
import pandas as pd

from retrieval_modeling.tiny_gate import fit_tiny_gate_ladder


def test_shared_tiny_gate_ladder_stops_at_two_dimensions():
    rows, splits = [], []
    for dataset, probe in [("Combined", "Both"), ("Novel", "Novel")]:
        for person in range(3):
            participant = f"{dataset[0]}{person}"
            splits.append(
                {"dataset": dataset, "participant_id": participant, "split": "test" if person == 2 else "train"}
            )
            for time, context in enumerate(["A"] * 3 + ["B"] * 3 + [probe] * 5):
                a, b = 6.0, 6.5
                q = 1 if context == "A" else 0 if context == "B" else .5
                rows.append(
                    {
                        "dataset": dataset, "participant_id": participant,
                        "global_sequence": time + 1, "retrieval_context": context,
                        "pred_A_log_rt": a, "pred_B_log_rt": b,
                        "log_rt": q * a + (1 - q) * b,
                    }
                )
    fits, states = fit_tiny_gate_ladder(
        pd.DataFrame(rows), pd.DataFrame(splits), seeds=(2,), epochs=25, learning_rate=.03
    )
    assert set(fits.model) == {"logistic_gate", "recurrent_gate_1d", "recurrent_gate_2d"}
    assert fits.hidden_dim.max() == 2
    assert fits.shared_across_datasets.all()
    assert set(states.dataset) == {"Combined", "Novel"}

