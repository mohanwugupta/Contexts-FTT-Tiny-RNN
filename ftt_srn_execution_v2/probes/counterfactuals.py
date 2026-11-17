"""Frozen A/B/Mixed/Novel readouts with identical weights and token state."""

from __future__ import annotations

import hashlib

import pandas as pd
import torch
from torch import nn

from ..config import FIXED_SEQUENCE
from ..inputs.context_basis import context_vector
from ..inputs.token_onehot import encode_sequence


PROBE_CONTEXTS = ("A", "B", "Mixed_norm", "Mixed_additive", "Novel")


def _weight_identity(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_counterfactual_probes(model, *, checkpoint_label: str):
    tokens = encode_sequence(FIXED_SEQUENCE[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(FIXED_SEQUENCE[1:], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    identity = _weight_identity(model)
    rows, hidden_rows = [], []
    model.eval()
    with torch.no_grad():
        for context_name in PROBE_CONTEXTS:
            context = context_vector(context_name, as_tensor=True).unsqueeze(0)
            logits, speed, hidden, _ = model(tokens, context)
            probability = torch.softmax(logits, -1).gather(
                -1, targets.view(1, -1, 1)
            ).squeeze(-1)
            rows.append({
                "checkpoint": checkpoint_label,
                "context": context_name,
                "probe_type": context_name,
                "sequence_ce": float(criterion(logits.squeeze(0), targets)),
                "sequence_accuracy": float((logits.argmax(-1) == targets.view(1, -1)).float().mean()),
                "correct_token_confidence": float(probability.mean()),
                "execution_speed": float(speed.mean()),
                "normalized_rt": float((1.0 - speed).mean()),
                "network_weight_identity": identity,
                "sequence_input_identity": "fixed_sequence_0-1-2-0-3-1-2-3",
                "probe_optimizer_updates": 0,
            })
            for position in range(hidden.shape[1]):
                hidden_rows.append({
                    "checkpoint": checkpoint_label,
                    "context": context_name,
                    "token_position": position + 1,
                    "execution_speed": float(speed[0, position]),
                    "normalized_rt": float(1.0 - speed[0, position]),
                    **{f"h{index}": float(hidden[0, position, index]) for index in range(hidden.shape[-1])},
                })
    return pd.DataFrame(rows), pd.DataFrame(hidden_rows)

