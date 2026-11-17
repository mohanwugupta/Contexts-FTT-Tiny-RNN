"""Feed a modified hidden state into subsequent Elman recurrence."""

from __future__ import annotations

import pandas as pd
import torch

from .readout import readout_hidden


def recurrent_rollout(model, tokens, context, targets, start_position: int, modified_state):
    state = torch.as_tensor(modified_state, dtype=tokens.dtype).view(1, 1, -1)
    rows = []
    initial = readout_hidden(model, state.view(-1), int(targets[start_position]))
    rows.append({
        "token_position": start_position + 1,
        "horizon": 0,
        "target_token": int(targets[start_position]),
        **initial,
        **{f"h{index}": float(value) for index, value in enumerate(state.view(-1))},
    })
    with torch.no_grad():
        for position in range(start_position + 1, tokens.shape[1]):
            logits, speed, hidden, state = model(
                tokens[:, position:position + 1], context, state
            )
            del logits, speed
            values = readout_hidden(model, hidden[0, 0], int(targets[position]))
            rows.append({
                "token_position": position + 1,
                "horizon": position - start_position,
                "target_token": int(targets[position]),
                **values,
                **{f"h{index}": float(value) for index, value in enumerate(hidden[0, 0])},
            })
    return pd.DataFrame(rows)
