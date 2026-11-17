"""Independent, token-matched estimation and evaluation forward passes."""

from __future__ import annotations

import pandas as pd
import torch

from ftt_srn_execution_v2.config import FIXED_SEQUENCE
from ftt_srn_execution_v2.inputs.context_basis import context_vector
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence


CAUSAL_CONTEXTS = ("A", "B", "Mixed_norm", "Novel")


def _hidden_columns(frame):
    return sorted(
        [column for column in frame if column.startswith("h") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )


def collect_matched_states(model, *, pass_role: str, pass_id: str,
                           contexts=CAUSAL_CONTEXTS) -> pd.DataFrame:
    if pass_role not in {"estimation", "evaluation"}:
        raise ValueError("pass_role must be estimation or evaluation.")
    tokens = encode_sequence(FIXED_SEQUENCE[:-1], as_tensor=True).unsqueeze(0)
    targets = torch.as_tensor(FIXED_SEQUENCE[1:], dtype=torch.long)
    rows = []
    model.eval()
    with torch.no_grad():
        for context_name in contexts:
            context = context_vector(context_name, as_tensor=True).unsqueeze(0)
            logits, speed, hidden, _ = model(tokens, context)
            probabilities = torch.softmax(logits, -1)
            confidence = probabilities.gather(-1, targets.view(1, -1, 1)).squeeze(-1)
            prediction = logits.argmax(-1)
            for position in range(hidden.shape[1]):
                rows.append({
                    "pass_role": pass_role,
                    "pass_id": pass_id,
                    "context": context_name,
                    "token_position": position + 1,
                    "input_token": int(FIXED_SEQUENCE[position]),
                    "target_token": int(targets[position]),
                    "execution_speed": float(speed[0, position]),
                    "normalized_rt": float(1.0 - speed[0, position]),
                    "correct_token_probability": float(confidence[0, position]),
                    "next_token_prediction": int(prediction[0, position]),
                    "next_token_correct": bool(prediction[0, position] == targets[position]),
                    **{f"h{index}": float(hidden[0, position, index])
                       for index in range(hidden.shape[-1])},
                })
    return pd.DataFrame(rows)


def estimate_ab_directions(states: pd.DataFrame) -> pd.DataFrame:
    if set(states.pass_role) != {"estimation"}:
        raise ValueError("A/B directions must come only from the estimation pass.")
    pure = states[states.context.isin(["A", "B"])]
    hidden_columns = _hidden_columns(pure)
    a = pure[pure.context == "A"].set_index("token_position")
    b = pure[pure.context == "B"].set_index("token_position")
    if set(a.index) != set(b.index) or a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("A/B direction estimation requires one matched state per token position.")
    rows = []
    for position in sorted(a.index):
        row_a, row_b = a.loc[position], b.loc[position]
        if int(row_a.target_token) != int(row_b.target_token):
            raise ValueError("Matched A/B states must retain the same next-token target.")
        h_a = row_a[hidden_columns].to_numpy(dtype=float)
        h_b = row_b[hidden_columns].to_numpy(dtype=float)
        difference = h_a - h_b
        distance = float((difference @ difference) ** .5)
        unit = difference / distance if distance > 1e-12 else difference.copy()
        midpoint = (h_a + h_b) / 2.0
        rows.append({
            "token_position": int(position),
            "target_a": int(row_a.target_token),
            "target_b": int(row_b.target_token),
            "estimation_pass_id": row_a.pass_id,
            "ab_distance": distance,
            **{f"h_a{index}": value for index, value in enumerate(h_a)},
            **{f"h_b{index}": value for index, value in enumerate(h_b)},
            **{f"mu{index}": value for index, value in enumerate(midpoint)},
            **{f"d{index}": value for index, value in enumerate(difference)},
            **{f"unit_d{index}": value for index, value in enumerate(unit)},
        })
    return pd.DataFrame(rows)
