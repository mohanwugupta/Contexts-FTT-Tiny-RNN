"""White-box extraction and seed-local representation calculations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from confirmatory_srn.model import ConfirmatoryElmanSRN


CONTEXT_VECTORS = {
    "A": np.array([1., 0., 0.], dtype=np.float32),
    "B": np.array([0., 1., 0.], dtype=np.float32),
    "Mixed": np.array([.5, .5, 0.], dtype=np.float32),
    "Novel": np.array([0., 0., 1.], dtype=np.float32),
    "NoContextProbe": np.array([0., 0., 0.], dtype=np.float32),
}

CONDITION_LABELS = {
    "High_monotonic": "High-Scale + High Context",
    "Low_monotonic": "Low-Scale + Low Context",
    "Lazy+RichCtx": "Low-Scale + High Context",
    "Rich+LazyCtx": "High-Scale + Low Context",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class FrozenNetwork:
    condition: str
    condition_label: str
    seed: int
    path: Path
    checkpoint_sha256: str
    model: ConfirmatoryElmanSRN


def load_frozen_network(checkpoint_dir: Path, condition: str, seed: int) -> FrozenNetwork:
    path = checkpoint_dir / f"{condition}_seed_{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["seed"]) != seed or payload["condition"] != condition:
        raise ValueError(f"Checkpoint identity mismatch in {path}")
    state = payload["model_state_dict"]
    hidden = int(state["recurrent.weight_hh_l0"].shape[0])
    init_scale = float(payload.get("init_scale", .02))
    model = ConfirmatoryElmanSRN(hidden_dim=hidden, init_scale=init_scale)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return FrozenNetwork(
        condition, CONDITION_LABELS.get(condition, condition), seed, path,
        sha256_file(path), model,
    )


def _tokens(sequence: tuple[int, ...]) -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32)[torch.tensor(sequence[:-1])].unsqueeze(0)


def run_context(model: ConfirmatoryElmanSRN, sequence: tuple[int, ...], context_vector,
                initial_state: torch.Tensor | None = None) -> dict[str, np.ndarray | float]:
    tokens = _tokens(sequence)
    targets = torch.tensor(sequence[1:], dtype=torch.long)
    context = torch.as_tensor(context_vector, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        logits, hidden, final = model(tokens, context, initial_state)
        probabilities = torch.softmax(logits, dim=-1)
        correct = probabilities[0, torch.arange(len(targets)), targets]
        losses = -torch.log(correct.clamp_min(1e-12))
        predicted = logits.argmax(-1)[0]
    return {
        "input": tokens[0].numpy(), "hidden": hidden[0].numpy(),
        "logits": logits[0].numpy(), "probabilities": probabilities[0].numpy(),
        "correct_probability": correct.numpy(), "loss": losses.numpy(),
        "predicted": predicted.numpy(), "accuracy": float((predicted == targets).float().mean()),
        "targets": targets.numpy(), "final_state": final.detach(),
    }


def extract_frozen_steps(network: FrozenNetwork, sequence, contexts) -> tuple[pd.DataFrame, dict[str, dict]]:
    before = state_hash(network.model)
    rows, runs = [], {}
    for context_name in contexts:
        vector = CONTEXT_VECTORS[context_name]
        result = run_context(network.model, sequence, vector)
        runs[context_name] = result
        for k in range(len(sequence) - 1):
            row = {
                "condition": network.condition,
                "condition_label": network.condition_label,
                "seed": network.seed,
                "context": context_name,
                "position": k + 1,
                "current_token": sequence[k],
                "target_token": sequence[k + 1],
                "correct_token_probability": result["correct_probability"][k],
                "cross_entropy": result["loss"][k],
                "predicted_token": result["predicted"][k],
                "position_correct": int(result["predicted"][k] == sequence[k + 1]),
                "sequence_accuracy": result["accuracy"],
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "training_row": False,
            }
            row.update({f"input_{j}": value for j, value in enumerate(result["input"][k])})
            row.update({f"context_input_{j}": value for j, value in enumerate(vector)})
            row.update({f"hidden_{j}": value for j, value in enumerate(result["hidden"][k])})
            row.update({f"logit_{j}": value for j, value in enumerate(result["logits"][k])})
            row.update({f"probability_{j}": value for j, value in enumerate(result["probabilities"][k])})
            rows.append(row)
    after = state_hash(network.model)
    if before != after:
        raise AssertionError("T1 failed: frozen evaluation changed network parameters.")
    return pd.DataFrame(rows), runs


def _decoder_metrics(model, x, y) -> dict[str, float]:
    probabilities = model.predict_proba(x)
    prediction = model.classes_[probabilities.argmax(1)]
    ordered = np.sort(probabilities, axis=1)
    return {
        "decoder_accuracy": float(np.mean(prediction == y)),
        "decoder_cross_entropy": float(log_loss(y, probabilities, labels=model.classes_)),
        "decoder_confidence": float(probabilities.max(1).mean()),
        "decoder_margin": float((ordered[:, -1] - ordered[:, -2]).mean()),
    }


def fit_sequence_decoder(runs: dict[str, dict], c: float, seed: int):
    """Fit only on familiar activity; cross-context familiar tests are held out."""
    rows = []
    positions = np.arange(1, runs["A"]["hidden"].shape[0] + 1)
    for train_name, test_name in (("A", "B"), ("B", "A")):
        decoder = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=seed)
        decoder.fit(runs[train_name]["hidden"], positions)
        rows.append({"decoder_train_contexts": train_name, "test_context": test_name,
                     "held_out_familiar": True, **_decoder_metrics(decoder, runs[test_name]["hidden"], positions)})
    decoder = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=seed)
    x = np.vstack([runs["A"]["hidden"], runs["B"]["hidden"]])
    y = np.tile(positions, 2)
    decoder.fit(x, y)
    for name, run in runs.items():
        if name in {"A", "B"}:
            continue
        rows.append({"decoder_train_contexts": "A+B", "test_context": name,
                     "held_out_familiar": False, **_decoder_metrics(decoder, run["hidden"], positions)})
    return decoder, pd.DataFrame(rows)


def fit_context_decoder(runs: dict[str, dict], c: float, seed: int):
    hidden_a, hidden_b = runs["A"]["hidden"], runs["B"]["hidden"]
    x = np.vstack([hidden_a, hidden_b])
    y = np.r_[np.zeros(len(hidden_a), dtype=int), np.ones(len(hidden_b), dtype=int)]
    decoder = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=seed)
    decoder.fit(x, y)
    raw_a, raw_b = decoder.decision_function(hidden_a).mean(), decoder.decision_function(hidden_b).mean()
    if abs(raw_b - raw_a) < 1e-12:
        raise RuntimeError("Familiar context decoder has zero score range.")

    def score(hidden):
        raw = decoder.decision_function(hidden)
        return 2 * (raw - raw_a) / (raw_b - raw_a) - 1

    rows = []
    for name, run in runs.items():
        scores = score(run["hidden"])
        rows.append({"probe_context": name, "mean_context_score": scores.mean(),
                     "sd_context_score": scores.std(ddof=1),
                     "hard_B_fraction": (scores > 0).mean(),
                     "decoder_train_contexts": "A+B"})
    return score, pd.DataFrame(rows)


def axis_geometry(hidden_a: np.ndarray, hidden_b: np.ndarray, hidden_p: np.ndarray):
    direction = hidden_b - hidden_a
    denominator = np.sum(direction * direction, axis=1)
    q = np.divide(np.sum((hidden_p - hidden_a) * direction, axis=1), denominator,
                  out=np.zeros_like(denominator), where=denominator > 1e-14)
    projected = hidden_a + q[:, None] * direction
    residual_vector = hidden_p - projected
    residual = np.linalg.norm(residual_vector, axis=1)
    ab_norm = np.linalg.norm(direction, axis=1)
    normalized = float(residual.mean() / max(ab_norm.mean(), 1e-12))
    return {
        "q": q, "projected": projected, "residual_vector": residual_vector,
        "residual": residual, "mean_q": float(q.mean()), "median_q": float(np.median(q)),
        "sd_q": float(q.std(ddof=1)), "mean_off_axis_residual": float(residual.mean()),
        "normalized_off_axis_residual": normalized,
        "fraction_q_inside_unit_interval": float(np.mean((q >= 0) & (q <= 1))),
    }


def factorization(hidden_a: np.ndarray, hidden_b: np.ndarray) -> dict[str, float | np.ndarray]:
    # Accumulate means/deviations in float64. Several saturated High-scale
    # trajectories lose ~1e-7 under float32 subtract/add cancellation.
    values = np.stack([hidden_a, hidden_b], axis=0).astype(np.float64)
    mu = values.mean(axis=(0, 1))
    sequence_component = values.mean(axis=0) - mu
    context_component = values.mean(axis=1) - mu
    interaction = values - mu - sequence_component[None, :, :] - context_component[:, None, :]
    reconstructed = mu + sequence_component[None, :, :] + context_component[:, None, :] + interaction
    if not np.allclose(reconstructed, values, atol=1e-10):
        raise AssertionError("T4 failed: factorization does not reconstruct hidden states.")
    ss_sequence = 2 * np.sum(sequence_component ** 2)
    ss_context = len(hidden_a) * np.sum(context_component ** 2)
    ss_interaction = np.sum(interaction ** 2)
    total = ss_sequence + ss_context + ss_interaction
    return {
        "mu": mu, "sequence_component": sequence_component,
        "context_component": context_component, "interaction": interaction,
        "SS_sequence": float(ss_sequence), "SS_context": float(ss_context),
        "SS_context_x_sequence": float(ss_interaction), "SS_total": float(total),
        "P_sequence": float(ss_sequence / total), "P_context": float(ss_context / total),
        "P_interaction": float(ss_interaction / total),
        "interaction_fraction": float(ss_interaction / total),
        "factorization_index": float(1 - ss_interaction / total),
    }


def output_projectors(weight: np.ndarray):
    _, singular, vh = np.linalg.svd(weight, full_matrices=True)
    tolerance = max(weight.shape) * np.finfo(float).eps * singular.max()
    rank = int(np.sum(singular > tolerance))
    basis = vh[:rank].T
    row = basis @ basis.T
    null = np.eye(weight.shape[1]) - row
    return row, null, rank


def decompose_vectors(vectors: np.ndarray, row: np.ndarray, null: np.ndarray):
    relevant = vectors @ row
    null_values = vectors @ null
    if not np.allclose(vectors, relevant + null_values, atol=1e-8):
        raise AssertionError("T5 failed: output subspaces do not reconstruct vectors.")
    if not np.allclose(np.sum(relevant * null_values, axis=1), 0, atol=1e-7):
        raise AssertionError("T5 failed: output subspaces are not orthogonal.")
    total = float(np.sum(vectors ** 2))
    relevant_ss = float(np.sum(relevant ** 2))
    null_ss = float(np.sum(null_values ** 2))
    return relevant, null_values, relevant_ss / max(total, 1e-15), null_ss / max(total, 1e-15)


def evaluate_hidden(sequence_decoder, context_score, hidden, targets, sequence_head):
    with torch.no_grad():
        logits = sequence_head(torch.as_tensor(hidden, dtype=torch.float32)).numpy()
    probability = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    correct = probability[np.arange(len(targets)), targets]
    position = np.arange(1, len(hidden) + 1)
    return {
        "sequence_loss": float((-np.log(np.clip(correct, 1e-12, 1))).mean()),
        "correct_token_confidence": float(correct.mean()),
        "sequence_accuracy": float(np.mean(logits.argmax(1) == targets)),
        "sequence_decoder_accuracy": _decoder_metrics(sequence_decoder, hidden, position)["decoder_accuracy"],
        "context_decoder_score": float(np.mean(context_score(hidden))),
    }
