"""Pre-probe calibration and differentiable speed–accuracy cost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import ExecutionConfig


def execution_cost(sequence_confidence: torch.Tensor, speed: torch.Tensor,
                   config: ExecutionConfig):
    confidence_detached = sequence_confidence.detach()
    motor_correct = torch.sigmoid(
        config.kappa * (confidence_detached - speed - config.delta)
    )
    normalized_rt = 1.0 - speed
    cost = config.c_error * (1.0 - motor_correct) + config.c_time * normalized_rt
    return cost, motor_correct


def calibrate_speed_accuracy(config: ExecutionConfig) -> pd.DataFrame:
    config.validate()
    confidence_grid = np.linspace(0.0, 1.0, config.confidence_grid_size)
    speed_grid = np.linspace(0.0, 1.0, config.speed_grid_size)
    rows = []
    for confidence in confidence_grid:
        motor_correct = 1.0 / (
            1.0 + np.exp(-config.kappa * (confidence - speed_grid - config.delta))
        )
        cost = config.c_error * (1.0 - motor_correct) + config.c_time * (1.0 - speed_grid)
        best = int(np.argmin(cost))
        rows.append({
            "sequence_confidence": confidence,
            "optimal_speed": float(speed_grid[best]),
            "optimal_normalized_rt": float(1.0 - speed_grid[best]),
            "minimum_execution_cost": float(cost[best]),
            "motor_correct_at_optimum": float(motor_correct[best]),
            "interior_optimum": bool(0 < best < len(speed_grid) - 1),
            "kappa": config.kappa,
            "delta": config.delta,
            "c_error": config.c_error,
            "c_time": config.c_time,
            "lambda_exec": config.lambda_exec,
        })
    table = pd.DataFrame(rows)
    table["monotone_optimal_speed"] = bool(np.all(np.diff(table.optimal_speed) >= -1e-12))
    return table

