"""Audits of learned token and context input associations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def context_weight_metrics(model, *, checkpoint: str) -> pd.DataFrame:
    weight = model.recurrent.weight_ih_l0.detach().cpu().numpy()
    a = weight[:, 4]
    b = weight[:, 5]
    novel = weight[:, 6]
    return pd.DataFrame([{
        "checkpoint": checkpoint,
        "token_input_weight_norm": float(np.linalg.norm(weight[:, :4])),
        "context_a_weight_norm": float(np.linalg.norm(a)),
        "context_b_weight_norm": float(np.linalg.norm(b)),
        "context_ab_weight_norm": float(np.linalg.norm(weight[:, 4:6])),
        "context_ab_difference_norm": float(np.linalg.norm(a - b)),
        "novel_context_weight_norm": float(np.linalg.norm(novel)),
        "novel_context_column_exact_zero": bool(np.array_equal(novel, np.zeros_like(novel))),
    }])
