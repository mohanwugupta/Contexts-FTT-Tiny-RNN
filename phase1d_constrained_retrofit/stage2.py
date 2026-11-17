"""Disabled-by-default from-scratch compositional-bias experiment.

This is intentionally isolated from the primary retrofit.  It is invoked only when
``stage2_enabled`` is explicit, and it never exposes lambda=.5 during acquisition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import ConfirmatoryElmanSRN
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from phase1b_direct_path.core import _forward_cues, _tokens, centered_logits


def run_optional_stage2(config, output):
    if not config.stage2_enabled:
        return pd.DataFrame()
    if .5 in config.stage2_local_lambdas:
        raise AssertionError("Stage 2 midpoint firewall failed.")
    directory = Path(output) / "stage2"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    tokens = _tokens(config.sequence)
    targets = torch.tensor(config.sequence[1:], dtype=torch.long)
    schedule = make_training_schedule(config.n_blocks, config.block_length)
    for seed in config.seeds:
        torch.manual_seed(int(seed))
        model = ConfirmatoryElmanSRN(hidden_dim=8, init_scale=2.)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.stage2_learning_rate)
        for trial in schedule.itertuples(index=False):
            optimizer.zero_grad()
            endpoint_logits, _ = _forward_cues(model, tokens, (0., 1.))
            current = 0 if trial.context == "A" else 1
            sequence_loss = F.cross_entropy(endpoint_logits[current], targets)
            endpoint_centered = centered_logits(endpoint_logits).detach()
            values = torch.tensor(config.stage2_local_lambdas, dtype=torch.float32)
            local_logits, _ = _forward_cues(model, tokens, values)
            lam = values.reshape(-1, 1, 1)
            local_target = (1 - lam) * endpoint_centered[0] + lam * endpoint_centered[1]
            composition = (centered_logits(local_logits) - local_target).square().mean()
            total = sequence_loss + config.stage2_composition_alpha * composition
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            model.enforce_novel_neutral()
        rows.append({
            "seed": seed, "optimizer_updates": len(schedule),
            "local_lambdas": ";".join(map(str, config.stage2_local_lambdas)),
            "midpoint_training_steps": 0, "human_rows_used": 0,
            "composition_alpha": config.stage2_composition_alpha,
            "final_sequence_loss": float(sequence_loss.detach()),
            "final_local_composition_loss": float(composition.detach()),
        })
        torch.save({"seed": seed, "model_state_dict": model.state_dict(),
                    "midpoint_training_steps": 0}, directory / f"stage2_seed_{seed}.pt")
    result = pd.DataFrame(rows)
    result.to_csv(directory / "STAGE2_training_summary.csv", index=False)
    return result
