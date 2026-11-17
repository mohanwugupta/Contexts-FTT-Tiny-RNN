from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.model import normalized_rt
from srn_mixed_novel_diagnostics.core import CONTEXT_VECTORS
from srn_nonlinearity_compositionality.math import exact_forward
from srn_training_timeline_audit.core import model_from_checkpoint, sha256_file

from .math import (
    backbone, centered_logits, decompose, orthogonal_matched,
    prediction_metrics, softmax, wrong_positions,
)


CONTEXTS = ("A", "B", "Mixed", "Novel", "NoContextProbe")


def module_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_canonical(config, condition, seed):
    path = Path(config.checkpoint_dir) / f"{condition}_seed_{seed}.pt"
    try:
        model, policy, payload = model_from_checkpoint(path, config)
    except (AssertionError, ValueError) as error:
        raise AssertionError("FAIL: NONCANONICAL_SRN") from error
    hidden = int(model.recurrent.hidden_size)
    scale = float(payload["init_scale"])
    expected = 2.0 if condition == "High_monotonic" else .02
    assert_canonical_metadata(hidden, scale, condition)
    return model, policy, payload, path, sha256_file(path)


def assert_canonical_metadata(hidden_size, init_scale, condition):
    expected = 2.0 if condition == "High_monotonic" else .02
    if int(hidden_size) != 8 or float(init_scale) != expected:
        raise AssertionError("FAIL: NONCANONICAL_SRN")
    return True


def execution_proxy(policy, correct_probabilities):
    if policy is None:
        return np.nan
    with torch.no_grad():
        values = torch.tensor(correct_probabilities, dtype=torch.float32).reshape(1, -1)
        return float(normalized_rt(policy(values)).mean())


def frozen_runs(config, model):
    return {name: exact_forward(model, config.sequence, CONTEXT_VECTORS[name], audit_manual=False)
            for name in CONTEXTS}


def extract_states(config, condition, seed, model, policy, checkpoint_hash, runs):
    rows = []
    for context, run in runs.items():
        z = centered_logits(run["logits"])
        metrics = prediction_metrics(run["logits"], run["targets"])
        execution = execution_proxy(policy, metrics["correct_probabilities"])
        for k in range(7):
            row = {
                "seed": seed, "scale_condition": condition, "probe_type": context,
                "transition": k + 1, "current_token": int(config.sequence[k]),
                "correct_token": int(run["targets"][k]),
                "raw_transition_loss": float(metrics["transition_losses"][k]),
                "raw_sequence_loss": metrics["raw_loss"],
                "correct_token_probability": float(metrics["correct_probabilities"][k]),
                "sequence_accuracy": metrics["sequence_accuracy"],
                "execution_proxy": execution, "checkpoint_hash": checkpoint_hash,
            }
            row.update({f"hidden_state_{i+1}": float(run["hidden"][k, i]) for i in range(8)})
            row.update({f"raw_logit_{i+1}": float(run["logits"][k, i]) for i in range(4)})
            row.update({f"centered_logit_{i+1}": float(z[k, i]) for i in range(4)})
            rows.append(row)
    return pd.DataFrame(rows)


def geometry_tables(config, condition, seed, checkpoint_hash, runs):
    task_a, task_b = centered_logits(runs["A"]["logits"]), centered_logits(runs["B"]["logits"])
    task_s, task_d = backbone(task_a, task_b)
    hidden_s, hidden_d = backbone(runs["A"]["hidden"], runs["B"]["hidden"])
    task_rows, hidden_rows, specificity = [], [], []
    for probe in config.probes:
        task_probe = centered_logits(runs[probe]["logits"])
        hidden_probe = runs[probe]["hidden"]
        for k in range(7):
            for space, value, shared, modulation, a, b, output in (
                ("task", task_probe, task_s, task_d, task_a, task_b, task_rows),
                ("hidden", hidden_probe, hidden_s, hidden_d, runs["A"]["hidden"],
                 runs["B"]["hidden"], hidden_rows),
            ):
                result = decompose(value[k], shared[k], modulation[k], config.epsilon)
                output.append({
                    "seed": seed, "scale_condition": condition, "probe": probe,
                    "transition": k + 1, "space": space, "alpha": result["alpha"],
                    "off_axis_norm": result["off_axis_norm"],
                    "normalized_off_axis_norm": result["normalized_off_axis_norm"],
                    "distance_to_shared": result["distance_to_shared"],
                    "distance_to_A": float(np.linalg.norm(value[k]-a[k])),
                    "distance_to_B": float(np.linalg.norm(value[k]-b[k])),
                    "axis_explained_fraction": result["axis_explained_fraction"],
                    "AB_modulation_norm": float(np.linalg.norm(modulation[k])),
                    "checkpoint_hash": checkpoint_hash,
                })
                wrong = wrong_positions(k)
                wrong_distances = np.array([np.linalg.norm(value[k]-shared[j]) for j in wrong])
                correct_distance = float(np.linalg.norm(value[k]-shared[k]))
                specificity.append({
                    "seed": seed, "scale_condition": condition, "probe": probe,
                    "transition": k + 1, "space": space,
                    "correct_position_distance": correct_distance,
                    "mean_wrong_position_distance": float(wrong_distances.mean()),
                    "nearest_wrong_position_distance": float(wrong_distances.min()),
                    "sequence_specificity_margin": float(wrong_distances.mean()-correct_distance),
                    "wrong_positions": ";".join(str(j+1) for j in wrong),
                    "checkpoint_hash": checkpoint_hash,
                })
    arrays = {"task_s": task_s, "task_d": task_d, "hidden_s": hidden_s, "hidden_d": hidden_d}
    return pd.DataFrame(task_rows), pd.DataFrame(hidden_rows), pd.DataFrame(specificity), arrays


def _task_transition_rows(config, condition, seed, probe, checkpoint_hash, run, shared, modulation):
    original = centered_logits(run["logits"]); targets = run["targets"]
    projector = np.eye(4) - np.ones((4, 4)) / 4
    rows = []
    for k in range(7):
        own = decompose(original[k], shared[k], modulation[k], config.epsilon)
        correct = {
            "B0_baseline": original[k], "B1_shared": shared[k],
            "B2_axis": shared[k] + own["alpha"] * modulation[k],
            "B3_clamped_axis": shared[k] + np.clip(own["alpha"], -1, 1) * modulation[k],
        }
        wrong_shared, wrong_axis = [], []
        for j in wrong_positions(k):
            wrong_shared.append(shared[j])
            wrong_decomp = decompose(original[k], shared[j], modulation[j], config.epsilon)
            wrong_axis.append(shared[j] + wrong_decomp["alpha"] * modulation[j])
        candidates = {
            **correct,
            "C1_wrong_shared": np.stack(wrong_shared),
            "C2_wrong_axis": np.stack(wrong_axis),
            "C3_orthogonal_shared": original[k] + orthogonal_matched(
                shared[k]-original[k], projector),
            "C3_orthogonal_axis": original[k] + orthogonal_matched(
                correct["B2_axis"]-original[k], projector),
        }
        for intervention, value in candidates.items():
            values = value[None, :] if value.ndim == 1 else value
            metrics = [prediction_metrics(v[None, :], [targets[k]]) for v in values]
            mean_logits = values.mean(axis=0)
            rows.append({
                "seed": seed, "scale_condition": condition, "probe": probe,
                "transition": k + 1, "intervention": intervention,
                "raw_transition_loss": float(np.mean([x["raw_loss"] for x in metrics])),
                "raw_loss": float(np.mean([x["raw_loss"] for x in metrics])),
                "correct_token_probability": float(np.mean([x["correct_token_probability"] for x in metrics])),
                "transition_correct": float(np.mean([x["sequence_accuracy"] for x in metrics])),
                "alpha": own["alpha"], "off_axis_norm": own["off_axis_norm"],
                "sequence_position_source": ("correct" if intervention.startswith("B") else
                                             "all_wrong_mean" if intervention.startswith(("C1", "C2")) else
                                             "orthogonal_matched"),
                "checkpoint_hash": checkpoint_hash,
                **{f"patched_logit_{i+1}": float(mean_logits[i]) for i in range(4)},
            })
    return rows


def task_patches(config, condition, seed, policy, checkpoint_hash, runs, arrays):
    rows = []
    for probe in config.probes:
        rows.extend(_task_transition_rows(
            config, condition, seed, probe, checkpoint_hash, runs[probe],
            arrays["task_s"], arrays["task_d"]))
    frame = pd.DataFrame(rows)
    frame["transition_execution_proxy"] = [
        execution_proxy(policy, [value]) for value in frame.correct_token_probability]
    return frame


def _rollout(model, sequence, context, patch_hidden):
    tokens = np.eye(4, dtype=np.float32)[np.asarray(sequence[:-1], dtype=int)]
    ctx = torch.tensor(context, dtype=torch.float32)
    w_ih = model.recurrent.weight_ih_l0.detach(); w_hh = model.recurrent.weight_hh_l0.detach()
    bias = model.recurrent.bias_ih_l0.detach() + model.recurrent.bias_hh_l0.detach()
    w_out = model.sequence_head.weight.detach(); b_out = model.sequence_head.bias.detach()
    previous = torch.tensor(patch_hidden, dtype=torch.float32)
    hidden = [previous]; logits = [previous @ w_out.T + b_out]
    for k in range(1, len(tokens)):
        combined = torch.cat((torch.tensor(tokens[k]), ctx))
        previous = torch.tanh(w_ih @ combined + w_hh @ previous + bias)
        hidden.append(previous); logits.append(previous @ w_out.T + b_out)
    return torch.stack(logits).numpy(), torch.stack(hidden).numpy()


def hidden_patches(config, condition, seed, model, policy, checkpoint_hash, runs, arrays):
    before = module_hash(model)
    w_out = model.sequence_head.weight.detach().numpy().astype(np.float64)
    b_out = model.sequence_head.bias.detach().numpy().astype(np.float64)
    shared, modulation = arrays["hidden_s"], arrays["hidden_d"]
    rows = []
    for probe in config.probes:
        run = runs[probe]; original = run["hidden"]; targets = run["targets"]
        for k in range(7):
            own = decompose(original[k], shared[k], modulation[k], config.epsilon)
            candidates = {
                "D0_baseline_transition": original[k], "D1_shared_transition": shared[k],
                "D2_axis_transition": shared[k] + own["alpha"] * modulation[k],
                "D4_orthogonal_shared_transition": original[k] + orthogonal_matched(shared[k]-original[k]),
                "D3_wrong_shared_transition": np.stack([shared[j] for j in wrong_positions(k)]),
            }
            for intervention, value in candidates.items():
                values = value[None, :] if value.ndim == 1 else value
                logits = values @ w_out.T + b_out
                metrics = [prediction_metrics(v[None, :], [targets[k]]) for v in logits]
                rows.append({
                    "seed": seed, "scale_condition": condition, "probe": probe,
                    "transition": k + 1, "patch_mode": "transition_wise",
                    "intervention": intervention,
                    "raw_transition_loss": float(np.mean([x["raw_loss"] for x in metrics])),
                    "raw_loss": float(np.mean([x["raw_loss"] for x in metrics])),
                    "correct_token_probability": float(np.mean([x["correct_token_probability"] for x in metrics])),
                    "transition_correct": float(np.mean([x["sequence_accuracy"] for x in metrics])),
                    "alpha": own["alpha"], "off_axis_norm": own["off_axis_norm"],
                    "sequence_position_source": ("all_wrong_mean" if "wrong" in intervention else
                                                 "orthogonal_matched" if "orthogonal" in intervention else "correct"),
                    "checkpoint_hash": checkpoint_hash,
                })

        entry = config.free_rollout_entry_transition - 1
        own = decompose(original[entry], shared[entry], modulation[entry], config.epsilon)
        rollout_candidates = {
            "D0_baseline_rollout": [(original[entry], "correct")],
            "D1_shared_rollout": [(shared[entry], "correct")],
            "D2_axis_rollout": [(shared[entry] + own["alpha"]*modulation[entry], "correct")],
            "D4_orthogonal_shared_rollout": [(original[entry] + orthogonal_matched(
                shared[entry]-original[entry]), "orthogonal_matched")],
            "D3_wrong_shared_rollout": [(shared[j], "all_wrong_mean") for j in wrong_positions(entry)],
        }
        for intervention, candidates in rollout_candidates.items():
            results = [_rollout(model, config.sequence, CONTEXT_VECTORS[probe], value)[0]
                       for value, _ in candidates]
            for k in range(7):
                metrics = [prediction_metrics(value[k:k+1], [targets[k]]) for value in results]
                rows.append({
                    "seed": seed, "scale_condition": condition, "probe": probe,
                    "transition": k + 1, "patch_mode": "single_entry_free_rollout",
                    "intervention": intervention,
                    "raw_transition_loss": float(np.mean([x["raw_loss"] for x in metrics])),
                    "raw_loss": float(np.mean([x["raw_loss"] for x in metrics])),
                    "correct_token_probability": float(np.mean([x["correct_token_probability"] for x in metrics])),
                    "transition_correct": float(np.mean([x["sequence_accuracy"] for x in metrics])),
                    "alpha": own["alpha"], "off_axis_norm": own["off_axis_norm"],
                    "sequence_position_source": candidates[0][1], "checkpoint_hash": checkpoint_hash,
                })
    if module_hash(model) != before:
        raise AssertionError("Frozen hidden-state intervention changed model parameters.")
    frame = pd.DataFrame(rows)
    frame["transition_execution_proxy"] = [
        execution_proxy(policy, [value]) for value in frame.correct_token_probability]
    return frame


def aggregate_patch_rows(frame, policy, space):
    keys = ["seed", "scale_condition", "probe", "intervention", "checkpoint_hash"]
    if "patch_mode" in frame:
        keys.append("patch_mode")
    rows = []
    for values, group in frame.groupby(keys, sort=False):
        key = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        probabilities = group.sort_values("transition").correct_token_probability.to_numpy()
        raw_loss = float(group.raw_transition_loss.mean())
        rows.append({**key, "space": space, "raw_loss": raw_loss,
                     "mean_token_loss": raw_loss,
                     "sum_token_loss": float(group.raw_transition_loss.sum()),
                     "correct_token_probability": float(probabilities.mean()),
                     "sequence_accuracy": float(group.transition_correct.mean()),
                     "execution_proxy": float(group.transition_execution_proxy.mean()),
                     "alpha": float(group.alpha.mean()),
                     "off_axis_norm": float(group.off_axis_norm.mean()),
                     "transition_count": int(group.transition.nunique()),
                     "sequence_position_source": group.sequence_position_source.iloc[0]})
    return pd.DataFrame(rows)


def add_rescue(raw):
    raw = raw.copy()
    baseline = raw[raw.intervention.isin(["B0_baseline", "D0_baseline_transition",
                                          "D0_baseline_rollout"])][
        ["seed", "scale_condition", "probe", "space", "patch_mode", "raw_loss"]
    ].rename(columns={"raw_loss": "baseline_raw_loss"})
    keys = ["seed", "scale_condition", "probe", "space", "patch_mode"]
    raw = raw.merge(baseline, on=keys, how="left")
    raw["raw_rescue"] = raw.baseline_raw_loss - raw.raw_loss
    return raw
