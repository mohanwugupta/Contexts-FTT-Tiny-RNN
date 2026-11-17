"""Realized recurrent initialization regimes and anonymous layout loading."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import nn

from episodic_meta_learning_capacity.model import VariableElmanSRN, initialization_statistics, model_hash


@dataclass(frozen=True)
class LayoutRegistry:
    arrays: tuple[np.ndarray, ...]


def load_layout_registry(path):
    payload = json.loads(path.read_text()); arrays = []
    for records in payload["layouts"].values():
        arrays.append(np.asarray([(row["x_norm"], row["y_norm"]) for row in records], dtype=np.float32))
    if len(arrays) != 4 or any(array.shape != (8, 2) for array in arrays): raise ValueError("Coordinate source must provide four anonymous 8-position arrays.")
    return LayoutRegistry(tuple(arrays))


def spectral_radius(model):
    values = model.recurrent.weight_hh_l0.detach().cpu().numpy(); return float(np.max(np.abs(np.linalg.eigvals(values))))


def initialize_model(width, regime, regimes, seed):
    torch.manual_seed(int(seed)); specification = regimes[regime]
    scale = float(specification.get("scale", 1.0)); model = VariableElmanSRN(42, int(width), scale)
    if specification["kind"] == "orthogonal_xavier":
        with torch.no_grad():
            nn.init.xavier_uniform_(model.recurrent.weight_ih_l0); nn.init.orthogonal_(model.recurrent.weight_hh_l0)
            nn.init.xavier_uniform_(model.sequence_head.weight)
            model.recurrent.bias_ih_l0.zero_(); model.recurrent.bias_hh_l0.zero_(); model.sequence_head.bias.zero_()
            rho = spectral_radius(model); model.recurrent.weight_hh_l0.mul_(float(specification["target_spectral_radius"]) / max(rho, 1e-12))
    statistics = initialization_statistics(model)
    statistics.update({"initialization_regime": regime, "parameter_norm": float(torch.sqrt(sum((value.detach() ** 2).sum() for value in model.parameters()))),
                       "initial_state_sha256": model_hash(model)})
    return model, statistics
