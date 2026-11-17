"""Seed-local diagnostic battery; no model or probe fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (
    CONTEXT_VECTORS, axis_geometry, decompose_vectors, evaluate_hidden,
    factorization, output_projectors, run_context,
)


def analyze_seed(network, sequence, runs, sequence_decoder, context_score,
                 cue_lambdas, history_repetitions):
    common = {"condition": network.condition, "condition_label": network.condition_label,
              "seed": network.seed}
    hidden_a, hidden_b = runs["A"]["hidden"], runs["B"]["hidden"]
    targets = runs["A"]["targets"]

    # B: familiar axis.
    geometry_rows, geometries = [], {}
    for name, run in runs.items():
        geometry = axis_geometry(hidden_a, hidden_b, run["hidden"])
        geometries[name] = geometry
        geometry_rows.append({**common, "probe_context": name,
                              **{k: v for k, v in geometry.items() if np.isscalar(v)}})
    if not np.allclose(geometries["A"]["q"], 0, atol=1e-8) or not np.allclose(
            geometries["B"]["q"], 1, atol=1e-8):
        raise AssertionError("T3 failed: familiar axis endpoints are not exact.")
    # Hidden states are stored as float32 in the frozen checkpoints; endpoint
    # projection therefore has a small (~1e-7) round-off residual.
    if geometries["A"]["mean_off_axis_residual"] > 1e-6 or geometries["B"]["mean_off_axis_residual"] > 1e-6:
        raise AssertionError("T3 failed: familiar endpoints have off-axis residual.")

    # C: familiar two-way decomposition.
    factors = factorization(hidden_a, hidden_b)
    factor_row = {**common, **{k: v for k, v in factors.items() if np.isscalar(v)}}

    # D: output row/null spaces.
    weight = network.model.sequence_head.weight.detach().cpu().numpy().astype(float)
    row_projector, null_projector, output_rank = output_projectors(weight)
    subspace_rows = []
    ab_delta = hidden_b - hidden_a
    _, _, rel_fraction, null_fraction = decompose_vectors(ab_delta, row_projector, null_projector)
    subspace_rows.append({**common, "vector_type": "familiar_AB_separation", "probe_context": "A_vs_B",
                          "output_rank": output_rank, "relevant_fraction": rel_fraction,
                          "null_fraction": null_fraction})
    for probe in ("Mixed", "Novel", "NoContextProbe"):
        residual = geometries[probe]["residual_vector"]
        _, _, rel_fraction, null_fraction = decompose_vectors(residual, row_projector, null_projector)
        subspace_rows.append({**common, "vector_type": "probe_off_axis_deviation", "probe_context": probe,
                              "output_rank": output_rank, "relevant_fraction": rel_fraction,
                              "null_fraction": null_fraction})

    # F: exact frozen interpolation sweep.
    sweep_rows = []
    for cue_lambda in cue_lambdas:
        vector = (1 - cue_lambda) * CONTEXT_VECTORS["A"] + cue_lambda * CONTEXT_VECTORS["B"]
        run = run_context(network.model, sequence, vector)
        geometry = axis_geometry(hidden_a, hidden_b, run["hidden"])
        metrics = evaluate_hidden(sequence_decoder, context_score, run["hidden"], targets,
                                  network.model.sequence_head)
        sweep_rows.append({**common, "cue_lambda": cue_lambda, **metrics,
                           "mean_axis_q": geometry["mean_q"],
                           "normalized_off_axis_residual": geometry["normalized_off_axis_residual"]})
    if not np.array_equal((1 - cue_lambdas[0]) * CONTEXT_VECTORS["A"] + cue_lambdas[0] * CONTEXT_VECTORS["B"], CONTEXT_VECTORS["A"]):
        raise AssertionError("T6 failed at lambda=0.")
    if not np.array_equal((1 - cue_lambdas[-1]) * CONTEXT_VECTORS["A"] + cue_lambdas[-1] * CONTEXT_VECTORS["B"], CONTEXT_VECTORS["B"]):
        raise AssertionError("T6 failed at lambda=1.")

    # G: input audit.
    c_a, c_b, c_n = CONTEXT_VECTORS["A"], CONTEXT_VECTORS["B"], CONTEXT_VECTORS["Novel"]
    familiar_basis = np.column_stack([c_a, c_b])
    input_projection = familiar_basis @ np.linalg.pinv(familiar_basis) @ c_n
    context_weights = network.model.recurrent.weight_ih_l0.detach().cpu().numpy()[:, 4:]
    v_a, v_b, v_n = context_weights @ c_a, context_weights @ c_b, context_weights @ c_n
    induced_basis = np.column_stack([v_a, v_b])
    induced_projection = induced_basis @ np.linalg.pinv(induced_basis) @ v_n
    cosine = lambda x, y: float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-15))
    novel_only = np.flatnonzero((np.abs(c_n) > 0) & (np.abs(c_a) + np.abs(c_b) == 0))
    novel_gradient_possible = bool(np.any((np.abs(c_a[novel_only]) + np.abs(c_b[novel_only])) > 0))
    novel_weights_zero = bool(np.allclose(context_weights[:, novel_only], 0, atol=1e-12))
    untrained = bool(len(novel_only) and not novel_gradient_possible)
    if len(novel_only) and not untrained:
        raise AssertionError("T7 failed: Novel-only dimensions were not flagged.")
    novel_row = {**common,
        "novel_input_norm": np.linalg.norm(c_n),
        "fraction_novel_input_norm_in_familiar_span": np.linalg.norm(input_projection) / max(np.linalg.norm(c_n), 1e-15),
        "novel_input_orthogonal_residual": np.linalg.norm(c_n - input_projection),
        "induced_novel_norm": np.linalg.norm(v_n), "cosine_novel_A_induced": cosine(v_n, v_a),
        "cosine_novel_B_induced": cosine(v_n, v_b),
        "induced_distance_to_familiar_span": np.linalg.norm(v_n - induced_projection),
        "novel_only_dimension_indices": ";".join(map(str, novel_only)),
        "novel_dimension_active_during_training": novel_gradient_possible,
        "novel_input_weights_exactly_zero": novel_weights_zero,
        "UNTRAINED_NOVEL_INPUT_DIRECTION": untrained,
    }

    # H: same current probe inputs under five standardized initial states.
    def repeated_terminal(context_name, repetitions):
        state = None
        for _ in range(repetitions):
            state = run_context(network.model, sequence, CONTEXT_VECTORS[context_name], state)["final_state"]
        return state

    # These SRNs reset state at every sequence boundary in both training and
    # their registered frozen-probe evaluation. Thus an actual preceding A or
    # B sequence supplies a zero initial state. Carrying a familiar terminal
    # state is retained as the requested *counterfactual* sensitivity test and
    # cannot be treated as evidence of history dependence in the actual model.
    initial_states = {
        "reset_hidden_state": (None, "registered_reset_protocol"),
        "actual_preceding_A_history": (None, "registered_reset_protocol"),
        "actual_preceding_B_history": (None, "registered_reset_protocol"),
        "matched_familiar_A_terminal_state": (repeated_terminal("A", history_repetitions), "counterfactual_state_carryover"),
        "matched_familiar_B_terminal_state": (repeated_terminal("B", history_repetitions), "counterfactual_state_carryover"),
    }
    history_rows = []
    reference_inputs = None
    for probe in ("Mixed", "Novel"):
        for history_name, (initial_state, intervention_type) in initial_states.items():
            run = run_context(network.model, sequence, CONTEXT_VECTORS[probe], initial_state)
            if reference_inputs is None:
                reference_inputs = run["input"].copy()
            if not np.array_equal(reference_inputs, run["input"]):
                raise AssertionError("T8 failed: current probe inputs differ by history condition.")
            geometry = axis_geometry(hidden_a, hidden_b, run["hidden"])
            metrics = evaluate_hidden(sequence_decoder, context_score, run["hidden"], targets,
                                      network.model.sequence_head)
            history_rows.append({**common, "probe_context": probe, "history_condition": history_name,
                                 "history_intervention_type": intervention_type,
                                 "state_carried_across_sequence_boundary": intervention_type == "counterfactual_state_carryover",
                                 **metrics, "mean_axis_q": geometry["mean_q"],
                                 "normalized_off_axis_residual": geometry["normalized_off_axis_residual"],
                                 "current_probe_input_sha256": __import__("hashlib").sha256(run["input"].tobytes()).hexdigest()})

    # I: output-level causal patches after A-H have completed successfully.
    causal_rows = []
    shared = (hidden_a + hidden_b) / 2
    familiar_mu = shared.mean(axis=0)
    for probe in ("Mixed", "Novel"):
        baseline = runs[probe]["hidden"]
        probe_context_component = (baseline - shared).mean(axis=0)
        shared_patch = shared + probe_context_component
        context_patch = (baseline - baseline.mean(axis=0)) + familiar_mu
        delta = shared_patch - baseline
        relevant_delta, null_delta, _, _ = decompose_vectors(delta, row_projector, null_projector)
        patches = {
            "baseline": baseline, "shared_sequence_patch": shared_patch,
            "context_component_patch": context_patch,
            "output_relevant_patch": baseline + relevant_delta,
            "output_null_patch": baseline + null_delta,
        }
        baseline_metrics = evaluate_hidden(sequence_decoder, context_score, baseline, targets,
                                           network.model.sequence_head)
        for patch_name, patched_hidden in patches.items():
            metrics = evaluate_hidden(sequence_decoder, context_score, patched_hidden, targets,
                                      network.model.sequence_head)
            causal_rows.append({**common, "probe_context": probe, "patch_type": patch_name, **metrics,
                                "delta_loss_vs_baseline": metrics["sequence_loss"] - baseline_metrics["sequence_loss"],
                                "delta_confidence_vs_baseline": metrics["correct_token_confidence"] - baseline_metrics["correct_token_confidence"],
                                "delta_decoder_accuracy_vs_baseline": metrics["sequence_decoder_accuracy"] - baseline_metrics["sequence_decoder_accuracy"]})

    return {
        "geometry": pd.DataFrame(geometry_rows), "factorization": pd.DataFrame([factor_row]),
        "subspace": pd.DataFrame(subspace_rows), "sweep": pd.DataFrame(sweep_rows),
        "novel": pd.DataFrame([novel_row]), "history": pd.DataFrame(history_rows),
        "causal": pd.DataFrame(causal_rows),
    }
