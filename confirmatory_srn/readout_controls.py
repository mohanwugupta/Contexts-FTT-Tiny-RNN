"""Normative and unrestricted controls evaluated on frozen representations."""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from execution_readout_diagnostic.policies import deterministic_normative_speed

from .config import ConfirmatoryConfig


def evaluate_readout_controls(
    transitions: pd.DataFrame, config: ConfirmatoryConfig,
) -> pd.DataFrame:
    hidden_columns = [f"hidden_state_{index}" for index in range(1, 9)]
    rows = []
    for (condition, seed), group in transitions.groupby(["condition", "seed"], sort=True):
        ordered = group.sort_values(["training_sequence", "transition_position"])
        head = nn.Linear(8, 1)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        optimizer = torch.optim.Adam(head.parameters(), lr=config.learning_rate)
        for training_sequence, one_sequence in ordered.groupby(
            "training_sequence", sort=True
        ):
            hidden = torch.as_tensor(
                one_sequence[hidden_columns].to_numpy(dtype="float32")
            )
            confidence = torch.as_tensor(
                one_sequence.correct_token_confidence.to_numpy(dtype="float32")
            )
            normative = deterministic_normative_speed(
                confidence, config.execution_candidate
            )
            unrestricted = torch.sigmoid(head(hidden)).squeeze(-1)
            uncertainty = 1 - confidence.detach()
            loss = (
                config.normative_c_time * (1 - unrestricted)
                + config.normative_c_risk * uncertainty
                * unrestricted.pow(config.normative_gamma)
            ).mean()
            for index, source in enumerate(one_sequence.itertuples(index=False)):
                common = {
                    "condition": condition,
                    "seed": int(seed),
                    "training_sequence": int(training_sequence),
                    "context": source.context,
                    "transition_position": int(source.transition_position),
                    "correct_token_confidence": float(confidence[index]),
                    "representation_frozen": True,
                    "human_rows_used": 0,
                }
                rows.extend([
                    {
                        **common,
                        "readout": "primary_monotonic",
                        "execution_speed": float(source.execution_speed)
                        if bool(source.execution_active) else float("nan"),
                        "normalized_RT": float(source.normalized_RT)
                        if bool(source.execution_active) else float("nan"),
                        "trainable_readout_parameters": 2 if bool(source.execution_active) else 0,
                    },
                    {
                        **common,
                        "readout": "deterministic_normative",
                        "execution_speed": float(normative[index]),
                        "normalized_RT": float(1 - normative[index]),
                        "trainable_readout_parameters": 0,
                    },
                    {
                        **common,
                        "readout": "unrestricted_hidden_state",
                        "execution_speed": float(unrestricted[index].detach()),
                        "normalized_RT": float(1 - unrestricted[index].detach()),
                        "trainable_readout_parameters": 9,
                    },
                ])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return pd.DataFrame(rows)
