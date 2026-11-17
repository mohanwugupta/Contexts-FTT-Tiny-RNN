"""Stage B: read-only native inner-loop adaptation of locked H128 checkpoints."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from episodic_meta_learning_capacity.human import human_features
from episodic_meta_learning_capacity.model import model_hash
from human_meta_prior_transfer.checkpoints import load_locked_checkpoint
from human_meta_prior_transfer.config import load_human_meta_transfer_config
from human_svg_geometry_transfer.geometry import load_coordinate_registry


SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)
CONTEXTS = ("A", "B", "Mixed", "Novel")


def _tensor_batch(registry, contexts):
    sequence = torch.as_tensor(SEQUENCE, dtype=torch.long)
    tokens = F.one_hot(sequence[:-1], 4).float().unsqueeze(0).repeat(len(contexts), 1, 1)
    features = torch.stack([torch.as_tensor(human_features(registry, context), dtype=torch.float32) for context in contexts])
    targets = sequence[1:].unsqueeze(0).repeat(len(contexts), 1)
    return tokens, features, targets


def _loss(model, registry, contexts, params=None):
    tokens, features, targets = _tensor_batch(registry, contexts)
    logits, hidden, _ = model(tokens, features, params=params)
    return F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1)), logits, hidden, targets


def native_adapt(model, registry, schedule, *, steps=10, alpha=.01):
    """The selected synthetic operator: full-batch plain SGD, first order, no clipping."""
    adapted = OrderedDict(model.named_parameters()); rows = []
    for step in range(1, int(steps) + 1):
        loss, _, _, _ = _loss(model, registry, schedule, adapted)
        gradients = torch.autograd.grad(loss, tuple(adapted.values()), create_graph=False)
        norm = float(torch.sqrt(sum((gradient.detach() ** 2).sum() for gradient in gradients)))
        adapted = OrderedDict((name, value - float(alpha) * gradient.detach()) for (name, value), gradient in zip(adapted.items(), gradients))
        rows.append({"inner_step": step, "support_NLL_before_step": float(loss.detach()), "gradient_norm": norm,
                     "optimizer": "plain_full_batch_SGD", "alpha": float(alpha), "gradient_clipping": False})
    return adapted, pd.DataFrame(rows)


def _parameter_hash(params):
    digest = hashlib.sha256()
    for name, value in sorted(params.items()): digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _probe(model, registry, params, group, seed, phase):
    rows = []
    policy = MonotonicExecutionPolicy(); policy_hash = _parameter_hash(policy.state_dict())
    for context in CONTEXTS:
        loss, logits, hidden, targets = _loss(model, registry, [context], params)
        probability = logits.softmax(-1); correct = probability.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        speed = policy(correct.detach()); rt = normalized_rt(speed)
        for position in range(7):
            row = {"training_condition": group, "group": group, "seed": int(seed), "phase": phase,
                   "context": context, "canonical_context": context, "transition_position": position + 1,
                   "current_token": SEQUENCE[position], "target_token": SEQUENCE[position + 1],
                   "predicted_token": int(logits[0, position].argmax()),
                   "correct_token_probability": float(correct[0, position].detach()),
                   "correct_token_nll": float(-torch.log(correct[0, position].detach().clamp_min(1e-12))),
                   "token_correct": bool(logits[0, position].argmax() == targets[0, position]),
                   "normalized_RT": float(rt[0, position].detach()), "rt_role": "descriptive_post_adaptation_only",
                   "rt_gradient_to_srn": False, "execution_policy_sha256": policy_hash,
                   "weight_policy": "functional_frozen_probe", "optimizer_updates": 0}
            row.update({f"hidden_state_{unit + 1}": float(value.detach()) for unit, value in enumerate(hidden[0, position])})
            rows.append(row)
    if policy_hash != _parameter_hash(policy.state_dict()): raise AssertionError("Descriptive RT policy changed during probing.")
    return pd.DataFrame(rows)


def _whole_profile_canonicalize(frame):
    parts = []
    for (group, seed, phase), one in frame.groupby(["group", "seed", "phase"], sort=True):
        means = one[one.context.isin(["A", "B"])].groupby("context").normalized_RT.mean()
        fast, slow = ("A", "B") if means.A <= means.B else ("B", "A")
        value = one.copy(); value["fast_source_context"] = fast; value["slow_source_context"] = slow
        value["canonical_context"] = value.context.replace({fast: "Fast", slow: "Slow"})
        value["canonicalization_scope"] = "whole_7_transition_profile"
        parts.append(value)
    return pd.concat(parts, ignore_index=True)


def _outside(value, first, second):
    return max(min(first, second) - value, 0., value - max(first, second))


def _summaries(raw, permutation_draws):
    canonical = _whole_profile_canonicalize(raw)
    hcols = [column for column in raw if column.startswith("hidden_state_")]
    familiar, mixed, novel, geometry = [], [], [], []
    for (group, seed), post in canonical[canonical.phase.eq("post")].groupby(["group", "seed"], sort=True):
        pre = canonical[(canonical.group.eq(group)) & canonical.seed.eq(seed) & canonical.phase.eq("pre")]
        post_lookup = post.set_index(["canonical_context", "transition_position"]); pre_lookup = pre.set_index(["canonical_context", "transition_position"])
        post_dist = np.mean([np.linalg.norm(post_lookup.loc[("Fast", k), hcols].to_numpy(float) - post_lookup.loc[("Slow", k), hcols].to_numpy(float)) for k in range(1, 8)])
        pre_dist = np.mean([np.linalg.norm(pre_lookup.loc[("Fast", k), hcols].to_numpy(float) - pre_lookup.loc[("Slow", k), hcols].to_numpy(float)) for k in range(1, 8)])
        matrix = post[post.canonical_context.isin(["Fast", "Slow"])].sort_values(["canonical_context", "transition_position"])[hcols].to_numpy(float)
        rng = np.random.default_rng(77000 + int(seed)); null = []
        for _ in range(permutation_draws):
            indices = rng.permutation(14); first, second = matrix[indices[:7]], matrix[indices[7:]]
            null.append(float(np.linalg.norm(first.mean(0) - second.mean(0))))
        observed_centroid = float(np.linalg.norm(matrix[:7].mean(0) - matrix[7:].mean(0)))
        a_b = post[post.context.isin(["A", "B"])]
        familiar.append({"group": group, "seed": seed, "AB_accuracy": float(a_b.token_correct.mean()),
                         "AB_NLL": float(a_b.correct_token_nll.mean()), "AB_confidence": float(a_b.correct_token_probability.mean()),
                         "AB_hidden_distance_pre": float(pre_dist), "AB_hidden_distance_post": float(post_dist),
                         "AB_hidden_distance_change": float(post_dist - pre_dist), "observed_AB_centroid_distance": observed_centroid,
                         "permutation_95": float(np.quantile(null, .95)), "above_permutation_null": observed_centroid > np.quantile(null, .95),
                         "mean_absolute_AB_confidence_gap": float(post.pivot(index="transition_position", columns="canonical_context", values="correct_token_probability").eval("abs(Fast-Slow)").mean()),
                         "whole_profile_canonicalization": True})
        for context, destination in (("Mixed", mixed), ("Novel", novel)):
            between_conf, between_nll, confidence_deficits, nll_deficits, rt_deficits, corridor, coordinates, residuals = [], [], [], [], [], [], [], []
            for k in range(1, 8):
                fast, slow, probe = (post_lookup.loc[(name, k)] for name in ("Fast", "Slow", context))
                between_conf.append(min(fast.correct_token_probability, slow.correct_token_probability) <= probe.correct_token_probability <= max(fast.correct_token_probability, slow.correct_token_probability))
                between_nll.append(min(fast.correct_token_nll, slow.correct_token_nll) <= probe.correct_token_nll <= max(fast.correct_token_nll, slow.correct_token_nll))
                confidence_deficits.append(_outside(float(probe.correct_token_probability), float(fast.correct_token_probability), float(slow.correct_token_probability)))
                nll_deficits.append(_outside(float(probe.correct_token_nll), float(fast.correct_token_nll), float(slow.correct_token_nll)))
                rt_deficits.append(_outside(float(probe.normalized_RT), float(fast.normalized_RT), float(slow.normalized_RT)))
                fh, sh, ph = (row[hcols].to_numpy(float) for row in (fast, slow, probe)); direction = sh - fh; denominator = float(direction @ direction)
                coordinate = float((ph - fh) @ direction / denominator) if denominator > 1e-14 else np.nan
                clipped = float(np.clip(coordinate, 0, 1)) if np.isfinite(coordinate) else 0.; projection = fh + clipped * direction
                shared = (fh + sh) / 2; difference = (sh - fh) / 2; norm = float(difference @ difference)
                corridor.append(float(np.linalg.norm(ph - projection))); coordinates.append(coordinate)
                residuals.append(float(np.linalg.norm(ph - (shared + ((ph - shared) @ difference / norm) * difference))) if norm > 1e-14 else np.nan)
                geometry.append({"group": group, "seed": seed, "probe_context": context, "transition_position": k,
                                 "projection_coordinate": coordinate, "inside_AB_corridor": bool(np.isfinite(coordinate) and 0 <= coordinate <= 1),
                                 "off_corridor_distance": corridor[-1], "shared_difference_residual": residuals[-1]})
            destination.append({"group": group, "seed": seed, "probe_context": context,
                                "confidence_between_fraction": float(np.mean(between_conf)), "NLL_between_fraction": float(np.mean(between_nll)),
                                "confidence_deficit": float(np.mean(confidence_deficits)), "prediction_NLL_deficit": float(np.mean(nll_deficits)),
                                "RT_deficit_descriptive": float(np.mean(rt_deficits)), "off_corridor_distance": float(np.mean(corridor)),
                                "inside_corridor_fraction": float(np.mean([np.isfinite(x) and 0 <= x <= 1 for x in coordinates])),
                                "mean_projection_coordinate": float(np.nanmean(coordinates)), "shared_difference_residual": float(np.nanmean(residuals))})
    return canonical, pd.DataFrame(familiar), pd.DataFrame(mixed), pd.DataFrame(novel), pd.DataFrame(geometry)


def _gate(config, familiar, mixed, cohort):
    settings = config.payload["stage_b"]; seed_count = int(familiar.seed.nunique()); required = int(math.ceil(float(settings["gate_seed_fraction"]) * seed_count))
    familiar_rows = []
    for group, one in familiar.groupby("group"):
        checks = {"B_F1_AB_above_chance": int((one.AB_accuracy > .25).sum()) >= required,
                  "B_F2_hidden_distance_increased": int((one.AB_hidden_distance_change > 0).sum()) >= required,
                  "B_F3_above_permutation_null": int(one.above_permutation_null.sum()) >= required,
                  "B_F4_profiles_nonidentical": int((one.mean_absolute_AB_confidence_gap > float(settings["profile_difference_epsilon"])).sum()) >= required}
        familiar_rows.append({"cohort": cohort, "row_type": "familiar_group", "group": group, "n_seeds": len(one), "required_passing_seeds": required,
                              **checks, "gate_passed": bool(all(checks.values()))})
    pivot = mixed.pivot(index="seed", columns="group")
    comparable = pivot.index
    deficit_expected = pivot["prediction_NLL_deficit"]["P2_meta"] < pivot["prediction_NLL_deficit"]["P1_shuffled_meta"]
    joint = {"primary_deficit_seed_count": int(deficit_expected.sum()),
             "structured_lower_mean_prediction_deficit": float(pivot["prediction_NLL_deficit"]["P2_meta"].mean()) < float(pivot["prediction_NLL_deficit"]["P1_shuffled_meta"].mean()),
             "structured_higher_mean_between_fraction": float(pivot["confidence_between_fraction"]["P2_meta"].mean()) > float(pivot["confidence_between_fraction"]["P1_shuffled_meta"].mean()),
             "structured_lower_mean_corridor_distance": float(pivot["off_corridor_distance"]["P2_meta"].mean()) < float(pivot["off_corridor_distance"]["P1_shuffled_meta"].mean())}
    p2_familiar = next(row["gate_passed"] for row in familiar_rows if row["group"] == "P2_meta")
    primary_consistent = joint["primary_deficit_seed_count"] >= required
    passed = bool(p2_familiar and primary_consistent and joint["structured_lower_mean_prediction_deficit"] and joint["structured_higher_mean_between_fraction"] and joint["structured_lower_mean_corridor_distance"])
    if config.force_stage_b_pass: passed = True
    overall = {"cohort": cohort, "row_type": "overall", "group": "P2_meta_vs_P1_shuffled_meta", "n_seeds": len(comparable), "required_passing_seeds": required,
               **joint, "primary_deficit_seed_consistency": primary_consistent, "P2_familiar_gate_passed": p2_familiar,
               "gate_passed": passed, "forced_for_integration_test": bool(config.force_stage_b_pass)}
    return pd.concat([pd.DataFrame(familiar_rows), pd.DataFrame([overall])], ignore_index=True, sort=False)


def _paired_bootstrap(config, mixed, novel):
    rows = []; rng = np.random.default_rng(88001)
    for probe, frame in (("Mixed", mixed), ("Novel", novel)):
        for metric, direction in (("prediction_NLL_deficit", "lower"), ("confidence_between_fraction", "higher"), ("off_corridor_distance", "lower")):
            pivot = frame.pivot(index="seed", columns="group", values=metric).dropna()
            delta = (pivot.P2_meta - pivot.P1_shuffled_meta).to_numpy(float)
            if not len(delta): continue
            draws = rng.choice(delta, size=(config.bootstrap_draws, len(delta)), replace=True).mean(1)
            rows.append({"probe_context": probe, "metric": metric, "contrast": "P2_meta - P1_shuffled_meta",
                         "expected_direction": direction, "n_paired_seeds": len(delta),
                         "mean_paired_difference": float(delta.mean()), "CI95_lower": float(np.quantile(draws, .025)),
                         "CI95_upper": float(np.quantile(draws, .975)),
                         "standardized_effect": float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan,
                         "bootstrap_draws": config.bootstrap_draws, "bootstrap_unit": "whole paired seed"})
    return pd.DataFrame(rows)


def run_native_meta_test(config, progress):
    transfer_config = load_human_meta_transfer_config(config.human_meta_config, smoke=False)
    registry = load_coordinate_registry(transfer_config.coordinate_registry, {"A": 0, "B": 1, "Mixed": 2, "Novel": 3})
    raw, manifest, adaptation = [], [], []
    settings = config.payload["stage_b"]; schedule = tuple(settings["support_schedule"])
    pilot_seeds = tuple(map(int, config.payload["seeds"]["stage_b_pilot"]))
    if config.smoke: pilot_seeds = config.stage_b_seeds

    def execute(seed_set, cohort):
      for group in settings["groups"]:
        for completed, seed in enumerate(seed_set, 1):
            progress(f"Stage B {cohort} native adaptation {group} seed {seed} ({completed}/{len(seed_set)})")
            payload, path = load_locked_checkpoint(transfer_config, group, seed); model = __import__("episodic_meta_learning_capacity.model", fromlist=["VariableElmanSRN"]).VariableElmanSRN(42, 128, 1.0)
            model.load_state_dict(payload["model_state_dict"]); before_hash = model_hash(model)
            pre = _probe(model, registry, None, group, seed, "pre"); pre["cohort"] = cohort; raw.append(pre)
            adapted, curve = native_adapt(model, registry, schedule, steps=10, alpha=.01)
            curve["group"] = group; curve["seed"] = seed; curve["cohort"] = cohort; adaptation.append(curve)
            post = _probe(model, registry, adapted, group, seed, "post"); post["cohort"] = cohort; raw.append(post)
            if model_hash(model) != before_hash: raise AssertionError("T2: functional native adaptation mutated the locked checkpoint.")
            for order, context in enumerate(schedule, 1):
                manifest.append({"cohort": cohort, "group": group, "seed": seed, "support_order": order, "context": context,
                                 "sequence": "0-1-2-0-3-1-2-3", "role": "support", "used_for_inner_adaptation": True,
                                 "used_as_query": False, "mixed_or_novel": False, "inner_steps_reused": 10,
                                 "full_batch_each_step": True, "checkpoint_path": str(path), "checkpoint_model_sha256": before_hash})
    execute(pilot_seeds, "pilot")
    pilot_raw = pd.concat(raw, ignore_index=True); _, pilot_familiar, pilot_mixed, _, _ = _summaries(pilot_raw, config.permutation_draws)
    pilot_gate = _gate(config, pilot_familiar, pilot_mixed, "pilot")
    pilot_pass = bool(pilot_gate[pilot_gate.row_type.eq("overall")].iloc[0].gate_passed)
    remaining = tuple(seed for seed in config.stage_b_seeds if seed not in pilot_seeds)
    if not config.smoke and pilot_pass and remaining: execute(remaining, "confirmatory")
    raw = pd.concat(raw, ignore_index=True); canonical, familiar, mixed, novel, geometry = _summaries(raw, config.permutation_draws)
    bootstrap = _paired_bootstrap(config, mixed, novel)
    if remaining and pilot_pass:
        confirm_gate = _gate(config, familiar, mixed, "confirmatory_20_seed")
        gate = pd.concat([pilot_gate, confirm_gate], ignore_index=True, sort=False)
    else:
        gate = pilot_gate
    return {"raw": raw, "canonical": canonical, "manifest": pd.DataFrame(manifest), "adaptation": pd.concat(adaptation, ignore_index=True),
            "familiar": familiar, "mixed": mixed, "novel": novel, "geometry": geometry, "bootstrap": bootstrap, "gate": gate}
