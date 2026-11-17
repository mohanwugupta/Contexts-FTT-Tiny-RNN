"""Read-only post-primary controller and implicit-GRU interventions."""

from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from execution_objective_search.objectives import predictive_statistics
from confirmatory_srn.model import normalized_rt

from .engine import combined_hash, forward_values, make_batch, restore


def _score_rows(run, model, policy, output, targets, label, context, extra=None):
    confidence, _, _ = predictive_statistics(output["logits"], targets.squeeze(0)); rt = normalized_rt(policy(confidence)); rows = []
    for position in range(7):
        logits = output["logits"][0, position]
        rows.append({"group": run.group, "architecture": run.architecture, "seed": run.seed, "checkpoint": 40,
                     "intervention": label, "recipient_context": context, "transition_position": position + 1,
                     "correct_token_nll": float(F.cross_entropy(logits.unsqueeze(0), targets[0, position].view(1))),
                     "correct_token_probability": float(confidence[0, position]), "normalized_RT": float(rt[0, position]), **(extra or {})})
    return rows


def controller_interventions(run, registry, sequence):
    model, policy = restore(run, 40); before = combined_hash(model, policy); model.eval(); policy.eval()
    swap, ablation, average = [], [], []
    with torch.no_grad():
        base = {context: (make_batch(registry, context, sequence), None) for context in ("A", "B", "Mixed")}
        for context, (batch, _) in list(base.items()): base[context] = (batch, forward_values(model, policy, batch))
        for recipient, donor in (("A", "B"), ("B", "A")):
            batch = base[recipient][0]; forced = base[donor][1][0]["context_state"]
            values = forward_values(model, policy, batch, overrides={"context_state": forced})
            swap.extend(_score_rows(run, model, policy, values[0], values[1], "context_state_swap", recipient, {"donor_context": donor}))
        for context in ("A", "B"):
            batch = base[context][0]; reference = base[context][1][0]["context_state"]
            zero = torch.zeros_like(reference); generator = torch.Generator().manual_seed(run.seed * 1009 + (0 if context == "A" else 1))
            random = torch.randn(reference.shape, generator=generator); random *= reference.norm() / random.norm().clamp_min(1e-12)
            for label, forced in (("zero", zero), ("matched_random", random)):
                values = forward_values(model, policy, batch, overrides={"context_state": forced})
                ablation.extend(_score_rows(run, model, policy, values[0], values[1], f"context_{label}", context, {"ablation": label}))
        midpoint = .5 * (base["A"][1][0]["context_state"] + base["B"][1][0]["context_state"])
        values = forward_values(model, policy, base["Mixed"][0], overrides={"context_state": midpoint})
        average.extend(_score_rows(run, model, policy, values[0], values[1], "arithmetic_context_average", "Mixed"))
    if combined_hash(model, policy) != before: raise AssertionError("T12: controller intervention mutated its parent.")
    return pd.DataFrame(swap), pd.DataFrame(ablation), pd.DataFrame(average)


def _gru_with_patch(model, batch, donor_hidden, patch_position, kind, alpha=.5):
    tokens, evidence, targets = batch; inputs = torch.cat([tokens, evidence], -1); state = None; hidden = []
    for position in range(7):
        current, state = model.gru(inputs[:, position:position + 1], state); value = current[:, 0]
        if position == patch_position:
            donor = donor_hidden[:, position]
            if kind == "donor_patch": value = donor
            elif kind == "difference_removal":
                difference = value - donor
                value = value - ((value * difference).sum(-1, keepdim=True) / (difference.square().sum(-1, keepdim=True) + 1e-12)) * difference
            elif kind == "interpolation": value = (1 - alpha) * value + alpha * donor
            state = value.unsqueeze(0)
        hidden.append(value)
    hseq = torch.stack(hidden, 1)
    return {"logits": model.sequence_head(hseq), "hidden": hseq, "context_state": None, "gate": None, "terminal": state}, targets


def gru_interventions(run, registry, sequence):
    model, policy = restore(run, 40); before = combined_hash(model, policy); model.eval(); policy.eval(); rows = []
    with torch.no_grad():
        base = {}
        for context in ("A", "B"):
            batch = make_batch(registry, context, sequence); values = forward_values(model, policy, batch); base[context] = (batch, values[0]["hidden"])
        for recipient, donor in (("A", "B"), ("B", "A")):
            for position in range(7):
                for kind in ("donor_patch", "difference_removal", "interpolation"):
                    output, targets = _gru_with_patch(model, base[recipient][0], base[donor][1], position, kind)
                    rows.extend(_score_rows(run, model, policy, output, targets, kind, recipient,
                                            {"donor_context": donor, "patched_transition": position + 1, "interpolation_alpha": .5 if kind == "interpolation" else np.nan}))
    if combined_hash(model, policy) != before: raise AssertionError("T12: GRU intervention mutated its parent.")
    return pd.DataFrame(rows)


def run_interventions(runs, passing, registry, sequence):
    swaps, ablations, averages, gru = [], [], [], []
    for run in runs:
        if (run.group, run.seed) not in passing: continue
        if run.architecture == "B":
            a, b, c = controller_interventions(run, registry, sequence); swaps.append(a); ablations.append(b); averages.append(c)
        elif run.architecture == "A": gru.append(gru_interventions(run, registry, sequence))
    empty = lambda: pd.DataFrame(columns=["status"])
    return {"swap": pd.concat(swaps, ignore_index=True) if swaps else empty(),
            "ablation": pd.concat(ablations, ignore_index=True) if ablations else empty(),
            "average": pd.concat(averages, ignore_index=True) if averages else empty(),
            "gru": pd.concat(gru, ignore_index=True) if gru else empty()}
