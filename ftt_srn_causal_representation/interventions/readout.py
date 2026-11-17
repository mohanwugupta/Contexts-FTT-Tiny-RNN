"""Instantaneous output-head readout without recurrent mutation."""

from __future__ import annotations

import torch


def readout_hidden(model, hidden, target_token: int) -> dict:
    state = torch.as_tensor(hidden, dtype=model.sequence_head.weight.dtype)
    with torch.no_grad():
        logits = model.sequence_head(state)
        speed = torch.sigmoid(model.execution_head(state)).squeeze()
        probabilities = torch.softmax(logits, dim=-1)
    prediction = int(logits.argmax().item())
    return {
        "execution_speed": float(speed),
        "normalized_rt": float(1.0 - speed),
        "correct_token_probability": float(probabilities[int(target_token)]),
        "next_token_prediction": prediction,
        "next_token_correct": bool(prediction == int(target_token)),
        "next_token_ce": float(-torch.log(probabilities[int(target_token)].clamp_min(1e-12))),
        **{f"logit_{index}": float(value) for index, value in enumerate(logits)},
    }
