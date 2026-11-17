"""
srn.metrics
===========
All measurement and analysis functions used by both analysis pipelines.

Public API
----------
Output diagnostics (used in participant-timeline runner):
  compute_sequence_accuracy, compute_prediction_entropy,
  compute_prediction_confidence, compute_correct_trial_surprisal

Geometry / representational analysis (shared by both scripts):
  compute_effective_dimensionality, compute_jacobian_stats, compute_pca_overlap,
  compute_per_context_dimensionality

Transfer / context-sensitivity (mechanistic sweep):
  compute_directional_transfer, compute_ce_bounds

Gradient alignment — two-level API:
  compute_gradient_alignment_tensors   (low-level, explicit tensors)
  compute_gradient_alignment           (high-level, builds tensors from config)

Weight & collapse analysis (sequential-all pipeline):
  compute_weight_norms, compute_collapse_metrics,
  run_context_perturbation_analysis, build_shared_pca,
  collect_checkpoint_hidden_states, compute_output_divergence,
  run_context_ablation_test, run_mechanistic_analysis,
  compute_transition_cost
"""

from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "sklearn not found — geometry metrics (PCA, decodability) will return NaN.",
        ImportWarning,
        stacklevel=2,
    )

from .config import ModelConfig, VOCAB_SIZE, FIXED_SEQUENCE
from .model import ContextSequenceRNN, gaussian_vector, sequence_to_gaussian


# =============================================================================
# Output diagnostics
# =============================================================================

def compute_sequence_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of time-steps where argmax(logits) == target.

    Parameters
    ----------
    logits  : (seq_len, vocab) or (1, seq_len, vocab)
    targets : (seq_len,)
    """
    preds = logits.squeeze(0).argmax(dim=-1)
    return (preds == targets).float().mean().item()


def compute_prediction_entropy(logits: torch.Tensor) -> float:
    """Mean Shannon entropy of the softmax distribution over the sequence (nats).

    High entropy ⟹ model is spreading probability mass (uncertain).
    Low  entropy ⟹ model is confident (well-learnt or overfit).

    logits : (seq_len, vocab) or (1, seq_len, vocab)
    """
    probs = torch.softmax(logits.squeeze(0), dim=-1)          # (seq_len, vocab)
    H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)        # (seq_len,)
    return H.mean().item()


def compute_prediction_confidence(logits: torch.Tensor) -> float:
    """Mean max-softmax probability over the sequence.

    Complement of entropy: rises toward 1.0 as the model becomes certain.

    logits : (seq_len, vocab) or (1, seq_len, vocab)
    """
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    return probs.max(dim=-1).values.mean().item()


def compute_correct_trial_surprisal(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Mean −log p(correct token) over *correctly-predicted* positions only.

    Human-RT analogue: conditioned on the model being right, how surprised
    was it?  Returns NaN when no position is correct.

    logits  : (seq_len, vocab) or (1, seq_len, vocab)
    targets : (seq_len,)
    """
    logits_2d = logits.squeeze(0)                             # (seq_len, vocab)
    probs = torch.softmax(logits_2d, dim=-1)
    preds = logits_2d.argmax(dim=-1)
    correct = (preds == targets)
    if correct.sum() == 0:
        return float("nan")
    p_correct_class = probs[torch.arange(len(targets)), targets]
    surprisal = -torch.log(p_correct_class + 1e-9)
    return surprisal[correct].mean().item()


# =============================================================================
# Geometry / representational analysis
# =============================================================================

def compute_effective_dimensionality(hidden_states: np.ndarray) -> float:
    """Effective Dimensionality (Participation Ratio).

    D_eff = (Σλ)² / Σλ²  where λ are eigenvalues of the hidden-state covariance.

    Parameters
    ----------
    hidden_states : (N, hidden_dim) array
    """
    if not SKLEARN_AVAILABLE:
        return np.nan

    cov = np.cov(hidden_states.T)
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0

    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0

    sum_eig    = np.sum(eigenvalues)
    sum_sq_eig = np.sum(eigenvalues ** 2)
    if sum_sq_eig == 0:
        return 0.0

    return float((sum_eig ** 2) / sum_sq_eig)


def compute_jacobian_stats(
    model: ContextSequenceRNN,
    inputs: torch.Tensor,
    context: torch.Tensor,
) -> Dict:
    """Spectral radius of the recurrent Jacobian J_t = diag(1 − h_t²) · W_hh.

    Returns
    -------
    dict with key ``mean_spectral_radius``
    """
    model.eval()
    with torch.no_grad():
        _, h_seq = model(inputs, context)           # (1, seq, hidden)
    h_seq = h_seq.squeeze(0)                        # (seq, hidden)

    W_hh = model.rnn.weight_hh_l0.detach()         # (hidden, hidden)
    spectral_radii = []

    for t in range(h_seq.shape[0]):
        h_t = h_seq[t]
        diag_deriv = torch.diag(1 - h_t ** 2)
        J_t = torch.matmul(diag_deriv, W_hh)
        try:
            eigvals = torch.linalg.eigvals(J_t)
            spectral_radii.append(torch.max(torch.abs(eigvals)).item())
        except Exception:
            pass

    return {
        "mean_spectral_radius": float(np.mean(spectral_radii)) if spectral_radii else 0.0
    }


def compute_pca_overlap(
    hidden_states_A: np.ndarray,
    hidden_states_B: np.ndarray,
) -> Dict:
    """Cosine similarity between the top principal components of two contexts.

    Returns
    -------
    dict with key ``subspace_overlap_cos``
    """
    if not SKLEARN_AVAILABLE:
        return {"subspace_overlap_cos": np.nan}

    n_comp_A = min(3, hidden_states_A.shape[1], hidden_states_A.shape[0])
    n_comp_B = min(3, hidden_states_B.shape[1], hidden_states_B.shape[0])

    pca_A = PCA(n_components=n_comp_A).fit(hidden_states_A)
    pca_B = PCA(n_components=n_comp_B).fit(hidden_states_B)

    pc1_A = pca_A.components_[0]
    pc1_B = pca_B.components_[0]
    cos_sim = float(
        np.abs(np.dot(pc1_A, pc1_B))
        / (np.linalg.norm(pc1_A) * np.linalg.norm(pc1_B) + 1e-9)
    )
    return {"subspace_overlap_cos": cos_sim}


def compute_per_context_dimensionality(hidden_states: np.ndarray) -> Dict:
    """Dimensionality and behavioral-convergence metrics for a single context.

    Returns
    -------
    dict with keys: eff_dim, h_variance, h_mean_norm, h_saturation
    """
    eff_dim     = compute_effective_dimensionality(hidden_states)
    h_variance  = float(np.mean(np.var(hidden_states, axis=0)))
    h_mean_norm = float(np.mean(np.linalg.norm(hidden_states, axis=1)))
    # Mean |h_t|: near 1.0 means tanh units are fully pinned → Jacobian ≈ 0
    h_saturation = float(np.mean(np.abs(hidden_states)))
    return {
        "eff_dim":      eff_dim,
        "h_variance":   h_variance,
        "h_mean_norm":  h_mean_norm,
        "h_saturation": h_saturation,
    }


# =============================================================================
# Transfer / context-sensitivity (mechanistic sweep)
# =============================================================================

def compute_directional_transfer(loss_history: List[Dict]) -> Dict:
    """Context-sensitivity metrics from a chronological loss history.

    At each context switch, Transfer Index (TI) = loss(new_ctx) − loss(old_ctx).

    Primary metrics
    ---------------
    context_sensitivity : Mean |TI|. Near zero = context-insensitive.
    ti_sign_consistency : Fraction of switches where TI > 0.
                          ~0.5 = chaotic,  ~1.0 = true splitting.
    ti_signed_mean      : Mean signed TI (only meaningful when sign_consistency ≠ 0.5).
    """
    switches_maze_to_melody: List[float] = []
    switches_melody_to_maze: List[float] = []

    prev_context   = None
    prev_loss_curr = None

    for entry in loss_history:
        curr_context = entry["context"]
        loss_curr    = entry["loss_curr"]

        if prev_context is not None and curr_context != prev_context:
            ti = loss_curr - prev_loss_curr
            if prev_context == "Maze" and curr_context == "Melody":
                switches_maze_to_melody.append(ti)
            elif prev_context == "Melody" and curr_context == "Maze":
                switches_melody_to_maze.append(ti)

        prev_context   = curr_context
        prev_loss_curr = loss_curr

    all_switches = switches_maze_to_melody + switches_melody_to_maze

    if not all_switches:
        return {
            "context_sensitivity": np.nan,
            "ti_sign_consistency": np.nan,
            "ti_signed_mean":      np.nan,
            "ti_maze_to_melody":   np.nan,
            "ti_melody_to_maze":   np.nan,
            "ti_asymmetry":        np.nan,
            "n_switches":          0,
        }

    mean_m2mel = float(np.mean(switches_maze_to_melody)) if switches_maze_to_melody else np.nan
    mean_mel2m = float(np.mean(switches_melody_to_maze)) if switches_melody_to_maze else np.nan

    return {
        "context_sensitivity": float(np.mean([abs(x) for x in all_switches])),
        "ti_sign_consistency": float(np.mean([1.0 if x > 0 else 0.0 for x in all_switches])),
        "ti_signed_mean":      float(np.mean(all_switches)),
        "ti_maze_to_melody":   mean_m2mel,
        "ti_melody_to_maze":   mean_mel2m,
        "ti_asymmetry":        float(abs(mean_m2mel - mean_mel2m))
                               if not (np.isnan(mean_m2mel) or np.isnan(mean_mel2m))
                               else np.nan,
        "n_switches": len(all_switches),
    }


def compute_ce_bounds(fixed_sequence: List[int], vocab_size: int) -> Dict:
    """Task-level CE reference values (chance only, no model required).

    Returns
    -------
    dict with keys: ce_chance, acc_chance
    """
    return {
        "ce_chance":  float(np.log(vocab_size)),
        "acc_chance": 1.0 / vocab_size,
    }


# =============================================================================
# Gradient alignment — two-level API
# =============================================================================

def compute_gradient_alignment_tensors(
    model: nn.Module,
    inputs_A: torch.Tensor,
    targets_A: torch.Tensor,
    context_A: torch.Tensor,
    inputs_B: torch.Tensor,
    targets_B: torch.Tensor,
    context_B: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """Cosine similarity between the gradient vectors for two explicit inputs.

    Low-level API used by the mechanistic sweep.  For the high-level convenience
    wrapper (takes config + context_encodings dict), see ``compute_gradient_alignment``.

    Returns cosine similarity in [−1, 1], or 0.0 if any gradient is zero.
    """
    model.eval()
    model.zero_grad()

    # Gradient for context A
    logits_A, _ = model(inputs_A, context_A)
    loss_A = criterion(logits_A.view(-1, logits_A.shape[-1]), targets_A.view(-1))
    loss_A.backward()
    grads_A = [p.grad.view(-1).clone() for p in model.parameters() if p.grad is not None]
    if not grads_A:
        model.train()
        return 0.0
    vec_A = torch.cat(grads_A)

    # Gradient for context B
    model.zero_grad()
    logits_B, _ = model(inputs_B, context_B)
    loss_B = criterion(logits_B.view(-1, logits_B.shape[-1]), targets_B.view(-1))
    loss_B.backward()
    grads_B = [p.grad.view(-1).clone() for p in model.parameters() if p.grad is not None]
    if not grads_B:
        model.train()
        return 0.0
    vec_B = torch.cat(grads_B)

    norm_A, norm_B = torch.norm(vec_A), torch.norm(vec_B)
    if norm_A == 0 or norm_B == 0:
        model.train()
        model.zero_grad()
        return 0.0

    cos_sim = (torch.dot(vec_A, vec_B) / (norm_A * norm_B)).item()
    model.train()
    model.zero_grad()
    return float(cos_sim)


def compute_gradient_alignment(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> float:
    """Cross-context cosine similarity of the full gradient vector.

    High-level wrapper used by the sequential-all pipeline.  Builds the
    sequence tensors from *config* and delegates to
    ``compute_gradient_alignment_tensors``.

    Positive  → both contexts exert the same learning pressure (collapse-driving).
    Negative  → conflicting gradients (context-splitting pressure).
    Returns NaN when fewer than 2 contexts are available or gradients are zero.
    """
    ctx_keys = list(context_encodings.keys())
    if len(ctx_keys) < 2:
        return float("nan")

    inputs    = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets   = torch.tensor(fixed_sequence[1:], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    ctx_A = context_encodings[ctx_keys[0]].unsqueeze(0)
    ctx_B = context_encodings[ctx_keys[1]].unsqueeze(0)

    was_training = model.training
    result = compute_gradient_alignment_tensors(
        model,
        inputs, targets, ctx_A,
        inputs, targets, ctx_B,
        criterion,
    )
    model.train(was_training)

    na = np.abs(result)
    # The tensor helper returns 0.0 on zero-grad; surface as NaN here for clarity
    if na < 1e-12 and result == 0.0:
        return float("nan")
    return result


# =============================================================================
# Weight & collapse analysis
# =============================================================================

def compute_weight_norms(model: nn.Module, vocab_size: int = VOCAB_SIZE) -> Dict:
    """Extract L2 norms of input, context, recurrent, and output weight matrices.

    Returns
    -------
    dict with keys: norm_input_weights, norm_context_weights,
    norm_recurrent_weights, context_input_ratio, norm_output_weights
    """
    result: Dict = {}
    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy()
        if "weight_ih" in name:
            result["norm_input_weights"]   = float(np.linalg.norm(w[:, :vocab_size]))
            result["norm_context_weights"] = float(np.linalg.norm(w[:, vocab_size:]))
            result["context_input_ratio"]  = result["norm_context_weights"] / (
                result["norm_input_weights"] + 1e-9)
        elif "weight_hh" in name:
            result["norm_recurrent_weights"] = float(np.linalg.norm(w))
        elif "output_layer.weight" in name:
            result["norm_output_weights"] = float(np.linalg.norm(w))
    return result


def compute_collapse_metrics(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> Dict:
    """Compute context-collapse geometry from a model checkpoint.

    Runs the model on the fixed sequence for each context, then computes:

    - euclidean_ctx_distance      : mean element-wise Euclidean distance between
                                    matched hidden-state vectors across contexts
    - cosine_ctx_distance         : mean cosine distance across matched steps
    - centroid_ctx_distance       : distance between context-trajectory centroids
    - within_traj_distance        : mean step-to-step distance within each context
    - collapse_index              : euclidean_ctx_distance / within_traj_distance
    - context_decodability        : linear classifier accuracy (context labels)
    - sequence_step_decodability  : linear classifier accuracy (step-index labels)
    - effective_dimensionality    : participation ratio of hidden-state covariance
    - trajectory_separation       : mean PCA distance between context trajectories
    """
    if not SKLEARN_AVAILABLE:
        return {}

    inputs   = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma)
    ctx_keys = list(context_encodings.keys())
    ctx_hidden: Dict[str, np.ndarray] = {}

    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            _, h_seq = model(inputs, ctx_vec.unsqueeze(0))
            ctx_hidden[ctx_name] = h_seq.squeeze(0).cpu().numpy()

    metrics: Dict = {}

    # Cross-context distance metrics (requires ≥ 2 contexts)
    if len(ctx_keys) >= 2:
        ha = ctx_hidden[ctx_keys[0]]
        hb = ctx_hidden[ctx_keys[1]]

        diffs     = ha - hb
        euc_dists = np.linalg.norm(diffs, axis=1)
        metrics["euclidean_ctx_distance"] = float(np.mean(euc_dists))

        cos_sims = []
        for a, b in zip(ha, hb):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-9 and nb > 1e-9:
                cos_sims.append(1.0 - float(np.dot(a, b) / (na * nb)))
        metrics["cosine_ctx_distance"]   = float(np.mean(cos_sims)) if cos_sims else np.nan
        metrics["centroid_ctx_distance"] = float(np.linalg.norm(ha.mean(axis=0) - hb.mean(axis=0)))
    else:
        metrics.update(
            euclidean_ctx_distance=np.nan,
            cosine_ctx_distance=np.nan,
            centroid_ctx_distance=np.nan,
        )

    # Within-trajectory distance (mean step-to-step norm across all contexts)
    within_dists = []
    for h in ctx_hidden.values():
        for t in range(len(h) - 1):
            within_dists.append(np.linalg.norm(h[t + 1] - h[t]))
    within = float(np.mean(within_dists)) if within_dists else 1e-9
    metrics["within_traj_distance"] = within
    metrics["collapse_index"] = (
        metrics.get("euclidean_ctx_distance", np.nan) / max(within, 1e-9)
    )

    # Stacked representations for classifier + PCA
    all_h      = np.concatenate(list(ctx_hidden.values()), axis=0)
    ctx_labels = np.concatenate([
        np.array([k] * v.shape[0]) for k, v in ctx_hidden.items()
    ])
    seq_len_per_ctx = list(ctx_hidden.values())[0].shape[0]
    step_labels     = np.tile(np.arange(seq_len_per_ctx), len(ctx_hidden))

    # Context decodability
    if len(np.unique(ctx_labels)) > 1:
        clf_ctx = LogisticRegression(random_state=42, max_iter=200)
        clf_ctx.fit(all_h, ctx_labels)
        metrics["context_decodability"] = float(clf_ctx.score(all_h, ctx_labels))
    else:
        metrics["context_decodability"] = np.nan

    # Sequence-step decodability
    if len(np.unique(step_labels)) > 1:
        clf_step = LogisticRegression(random_state=42, max_iter=200)
        try:
            clf_step.fit(all_h, step_labels)
            metrics["sequence_step_decodability"] = float(clf_step.score(all_h, step_labels))
        except Exception:
            metrics["sequence_step_decodability"] = np.nan
    else:
        metrics["sequence_step_decodability"] = np.nan

    # Effective dimensionality
    # joint: participation ratio of the combined (all-context) hidden-state cloud.
    # per_context: average PR within each context individually.
    # expansion: joint − per_context — positive when rich network uses two distinct
    #   subspaces (joint cloud is higher-dimensional than either context alone);
    #   near zero when both contexts share the same subspace (collapsed / poor).
    joint_eff_dim      = compute_effective_dimensionality(all_h)
    per_ctx_eff_dims   = [compute_effective_dimensionality(h) for h in ctx_hidden.values()]
    per_ctx_eff_dim    = float(np.mean(per_ctx_eff_dims)) if per_ctx_eff_dims else np.nan
    metrics["effective_dimensionality"]            = joint_eff_dim
    metrics["per_context_effective_dimensionality"] = per_ctx_eff_dim
    metrics["dimensionality_expansion"]            = (
        joint_eff_dim - per_ctx_eff_dim
        if not (np.isnan(joint_eff_dim) or np.isnan(per_ctx_eff_dim))
        else np.nan
    )

    # PCA trajectory separation
    pca_n = min(2, all_h.shape[0], all_h.shape[1])
    if pca_n >= 2 and len(ctx_keys) >= 2:
        pca   = PCA(n_components=pca_n)
        X_pca = pca.fit_transform(all_h)
        n_per = seq_len_per_ctx
        sep   = np.mean([
            np.linalg.norm(X_pca[i] - X_pca[i + n_per])
            for i in range(n_per)
        ])
        metrics["trajectory_separation"] = float(sep)
    else:
        metrics["trajectory_separation"] = np.nan

    return metrics


def run_context_perturbation_analysis(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
    modes: Optional[List[str]] = None,
) -> List[Dict]:
    """Causal perturbation test: measure the effect of disrupting the context signal.

    For each context and perturbation mode, computes:
    - baseline_loss / baseline_accuracy (correct context)
    - perturbed_loss / perturbed_accuracy (disrupted context)
    - delta_loss = perturbed − baseline
    - delta_accuracy = perturbed − baseline
    - hidden_state_shift : mean L2 distance between baseline and perturbed hidden states

    Modes
    -----
    ``"zero"``    : replace context with the zero vector
    ``"shuffle"`` : assign a randomly chosen *other* context vector
    ``"swap"``    : for the first two contexts, swap their vectors
    """
    if modes is None:
        modes = ["zero", "shuffle", "swap"]

    criterion   = nn.CrossEntropyLoss()
    inputs_full = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma)
    inputs      = inputs_full[:, :-1, :]                         # (1, seq_len-1, vocab)
    targets     = torch.tensor(fixed_sequence[1:], dtype=torch.long)

    ctx_keys = list(context_encodings.keys())
    results: List[Dict] = []

    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            ctx_batch = ctx_vec.unsqueeze(0)

            logits_base, h_base = model(inputs, ctx_batch)
            loss_base = criterion(logits_base.squeeze(0), targets).item()
            acc_base  = compute_sequence_accuracy(logits_base, targets)
            h_base_np = h_base.squeeze(0).cpu().numpy()

            for mode in modes:
                if mode == "zero":
                    pert_vec = torch.zeros_like(ctx_vec)
                elif mode == "shuffle":
                    others = [k for k in ctx_keys if k != ctx_name]
                    pert_vec = context_encodings[others[0]] if others else ctx_vec
                elif mode == "swap":
                    if len(ctx_keys) >= 2:
                        swap_map = {ctx_keys[0]: ctx_keys[1], ctx_keys[1]: ctx_keys[0]}
                        pert_name = swap_map.get(ctx_name, ctx_name)
                        pert_vec  = context_encodings[pert_name]
                    else:
                        pert_vec = torch.zeros_like(ctx_vec)
                else:
                    continue

                logits_pert, h_pert = model(inputs, pert_vec.unsqueeze(0))
                loss_pert = criterion(logits_pert.squeeze(0), targets).item()
                acc_pert  = compute_sequence_accuracy(logits_pert, targets)
                h_pert_np = h_pert.squeeze(0).cpu().numpy()

                hidden_shift = float(np.mean(
                    np.linalg.norm(h_pert_np - h_base_np, axis=1)
                ))

                results.append({
                    "context":            ctx_name,
                    "perturbation_mode":  mode,
                    "baseline_loss":      loss_base,
                    "perturbed_loss":     loss_pert,
                    "delta_loss":         loss_pert - loss_base,
                    "baseline_accuracy":  acc_base,
                    "perturbed_accuracy": acc_pert,
                    "delta_accuracy":     acc_pert - acc_base,
                    "hidden_state_shift": hidden_shift,
                })

    return results


def build_shared_pca(
    checkpoint_hidden_states: Dict[str, np.ndarray],
    n_components: int = 2,
):
    """Fit a single PCA on pooled hidden states from all checkpoints.

    Parameters
    ----------
    checkpoint_hidden_states : dict mapping checkpoint_key → (N, hidden) array
    n_components             : number of PCA dimensions to retain

    Returns
    -------
    Fitted sklearn PCA object, or None if sklearn is unavailable.
    """
    if not SKLEARN_AVAILABLE:
        return None
    all_states = np.concatenate(list(checkpoint_hidden_states.values()), axis=0)
    n_comp = min(n_components, all_states.shape[0], all_states.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(all_states)
    return pca


def collect_checkpoint_hidden_states(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> np.ndarray:
    """Return stacked hidden states (all contexts) for a single model checkpoint.

    Returns
    -------
    np.ndarray of shape (n_contexts * seq_len, hidden_dim)
    """
    inputs = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma)
    all_h: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for ctx_vec in context_encodings.values():
            _, h_seq = model(inputs, ctx_vec.unsqueeze(0))
            all_h.append(h_seq.squeeze(0).cpu().numpy())
    return np.concatenate(all_h, axis=0)


def compute_output_divergence(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> Dict:
    """Jensen-Shannon divergence between output distributions across contexts.

    As the model collapses, JS divergence falls toward 0.

    Returns
    -------
    dict with keys: output_js_divergence, correct_token_prob_gap
    """
    ctx_keys = list(context_encodings.keys())
    if len(ctx_keys) < 2:
        return {"output_js_divergence": np.nan, "correct_token_prob_gap": np.nan}

    inputs  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets = torch.tensor(fixed_sequence[1:], dtype=torch.long)

    probs_all: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for ctx_name in ctx_keys[:2]:
            ctx_vec = context_encodings[ctx_name].unsqueeze(0)
            logits, _ = model(inputs, ctx_vec)
            p = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()   # (seq_len, vocab)
            probs_all.append(p)

    pa, pb = probs_all[0], probs_all[1]
    m = 0.5 * (pa + pb)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.mean(np.sum(p * np.log(p / (q + 1e-9) + 1e-9), axis=-1)))

    js = float(0.5 * _kl(pa, m) + 0.5 * _kl(pb, m))

    tgt_np   = targets.numpy()
    p_corr_a = pa[np.arange(len(tgt_np)), tgt_np]
    p_corr_b = pb[np.arange(len(tgt_np)), tgt_np]
    prob_gap = float(np.mean(np.abs(p_corr_a - p_corr_b)))

    return {"output_js_divergence": js, "correct_token_prob_gap": prob_gap}


def run_context_ablation_test(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> List[Dict]:
    """Recurrent-only competence test: run with zeroed / mean context.

    For each original context, compares performance under:
    - ``"zero"`` : context vector replaced with zeros  (pure recurrent dynamics)
    - ``"mean"`` : context vector replaced with the mean of all context vectors

    Returns one row per (context_name × condition) with delta_loss,
    delta_accuracy, and hidden_state_shift relative to the true-context baseline.
    """
    criterion = nn.CrossEntropyLoss()
    inputs    = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets   = torch.tensor(fixed_sequence[1:], dtype=torch.long)

    ctx_vecs = list(context_encodings.values())
    mean_ctx = torch.stack(ctx_vecs).mean(dim=0)
    zero_ctx = torch.zeros_like(mean_ctx)

    results: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            logits_true, h_true = model(inputs, ctx_vec.unsqueeze(0))
            loss_true = criterion(logits_true.squeeze(0), targets).item()
            acc_true  = compute_sequence_accuracy(logits_true, targets)
            h_true_np = h_true.squeeze(0).cpu().numpy()

            for condition, pert_vec in [("zero", zero_ctx), ("mean", mean_ctx)]:
                logits_p, h_p = model(inputs, pert_vec.unsqueeze(0))
                loss_p = criterion(logits_p.squeeze(0), targets).item()
                acc_p  = compute_sequence_accuracy(logits_p, targets)
                h_p_np = h_p.squeeze(0).cpu().numpy()
                shift  = float(np.mean(np.linalg.norm(h_p_np - h_true_np, axis=1)))
                results.append({
                    "context":            ctx_name,
                    "context_condition":  condition,
                    "loss":               loss_p,
                    "accuracy":           acc_p,
                    "delta_loss":         loss_p - loss_true,
                    "delta_accuracy":     acc_p  - acc_true,
                    "hidden_state_shift": shift,
                })
    return results


def run_mechanistic_analysis(
    model: "ContextSequenceRNN",
    config: ModelConfig,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
    active_contexts: Optional[List[str]] = None,
    context_encodings: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    """Diagnose HOW the model solved the task (R4 & R5).

    Computes:
    1. Weight attribution : context vs. input weight norm ratio
    2. Context decodability : linear classifier on hidden states
    3. Sequence-step decodability
    4. PCA geometry
    5. Centroid distance (‖mean(h_A) − mean(h_B)‖₂ in original hidden space)
    5b. Trajectory separation (mean PCA distance between contexts)
    6. Effective dimensionality
    7. Jacobian spectral radius per context
    """
    if not SKLEARN_AVAILABLE:
        return {}

    if context_encodings is None:
        context_encodings = {
            "Maze":   torch.tensor([1.0, 0.0]),
            "Melody": torch.tensor([0.0, 1.0]),
        }

    results: Dict = {}

    # 1. Weight attribution
    for name, param in model.named_parameters():
        if "weight_ih" in name:
            weights = param.detach().numpy()
            w_input   = weights[:, :vocab_size]
            w_context = weights[:, vocab_size:]
            norm_in   = np.linalg.norm(w_input)
            norm_cx   = np.linalg.norm(w_context)
            results["weight_ratio_cx_in"] = norm_cx / (norm_in + 1e-9)
            results["norm_cx"]            = norm_cx
            results["norm_in"]            = norm_in
            break

    # 2. Collect hidden states
    inputs = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma)

    hidden_states: List[np.ndarray] = []
    labels: List[str] = []

    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            _, h_seq = model(inputs, ctx_vec.unsqueeze(0))
            h_np     = h_seq.squeeze(0).cpu().numpy()
            hidden_states.append(h_np)
            labels.extend([ctx_name] * h_np.shape[0])

    X = np.concatenate(hidden_states, axis=0)
    y = np.array(labels)

    # 3. Context decodability
    if len(np.unique(y)) > 1:
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        results["context_decodability"] = float(clf.score(X, y))
    else:
        results["context_decodability"] = np.nan

    # 3b. Sequence-step decodability
    seq_len_for_decode = hidden_states[0].shape[0]
    step_labels = np.tile(np.arange(seq_len_for_decode), len(hidden_states))
    if len(np.unique(step_labels)) > 1:
        clf_step = LogisticRegression(random_state=42, max_iter=200)
        try:
            clf_step.fit(X, step_labels)
            results["sequence_step_decodability"] = float(clf_step.score(X, step_labels))
        except Exception:
            results["sequence_step_decodability"] = np.nan
    else:
        results["sequence_step_decodability"] = np.nan

    # 4. PCA geometry
    pca   = PCA(n_components=min(2, X.shape[0]))
    X_pca = pca.fit_transform(X)
    results["pca_variance_explained"] = float(pca.explained_variance_ratio_.sum())
    results["pca_coords"]             = X_pca
    results["pca_labels"]             = y

    # 5. Centroid distance (original hidden-state space)
    if len(hidden_states) >= 2:
        centroid_a = hidden_states[0].mean(axis=0)
        centroid_b = hidden_states[1].mean(axis=0)
        results["centroid_ctx_distance"] = float(np.linalg.norm(centroid_a - centroid_b))
    else:
        results["centroid_ctx_distance"] = np.nan

    # 5b. Trajectory separation (PCA space)
    unique_ctx_labels = np.unique(y)
    if len(unique_ctx_labels) >= 2:
        ctx_a = unique_ctx_labels[0]
        ctx_b = unique_ctx_labels[1]
        idx_a = np.where(y == ctx_a)[0]
        idx_b = np.where(y == ctx_b)[0]
        if len(idx_a) == len(idx_b):
            coords_a = X_pca[idx_a]
            coords_b = X_pca[idx_b]
            results["trajectory_separation"] = float(np.mean(
                [np.linalg.norm(a - b) for a, b in zip(coords_a, coords_b)]
            ))
        else:
            results["trajectory_separation"] = np.nan
    else:
        results["trajectory_separation"] = np.nan

    # 6. Effective dimensionality (joint + per-context breakdown)
    joint_eff_dim    = compute_effective_dimensionality(X)
    per_ctx_eff_dims = [compute_effective_dimensionality(h) for h in hidden_states]
    per_ctx_eff_dim  = float(np.mean(per_ctx_eff_dims)) if per_ctx_eff_dims else np.nan
    results["effective_dimensionality"]             = joint_eff_dim
    results["per_context_effective_dimensionality"] = per_ctx_eff_dim
    results["dimensionality_expansion"]             = (
        joint_eff_dim - per_ctx_eff_dim
        if not (np.isnan(joint_eff_dim) or np.isnan(per_ctx_eff_dim))
        else np.nan
    )

    # 7. Jacobian spectral radius per context
    for ctx_name, ctx_vec in context_encodings.items():
        j_stats = compute_jacobian_stats(model, inputs, ctx_vec.unsqueeze(0))
        results[f"jacobian_spectral_radius_{ctx_name}"] = j_stats["mean_spectral_radius"]

    return results


# =============================================================================
# Transition cost analysis
# =============================================================================

def compute_transition_cost(
    df: pd.DataFrame,
    dv: str,
    use_global_step: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Compute transition costs between context blocks in the participant timeline.

    For each block (model_type × participant_id × block_id × context), derives
    ``dv_start`` (DV at the first trial) and ``dv_end`` (DV at the last trial).
    A *transition event* is the boundary between two consecutive blocks:

        trans_cost = |dv_start_current_block − dv_end_previous_block|

    Transition type
    ---------------
    ``"switch"`` : context changed from previous block  (e.g. Maze → Melody)
    ``"stay"``   : context unchanged                    (e.g. Maze → Maze)

    Direction (normalised relative to each participant's opening context)
    -----------------
    ``"A→B"``  : first-seen context → other context
    ``"B→A"``  : other context → first-seen context
    ``"other"`` : any other pattern

    Parameters
    ----------
    df              : per-step learning-curve DataFrame produced by the pipeline.
                      Required columns: ``model_type``, ``participant_id``,
                      ``block_id``, ``context``, ``sequence_in_block``
                      (or ``global_step`` for ordering), ``training_active``,
                      and the DV column named by *dv*.
    dv              : name of the dependent-variable column (e.g. ``"loss"``).
    use_global_step : if True, order blocks by ``min(global_step)``; else by
                      ``min(block_id)``.

    Returns
    -------
    dict with three DataFrames:
      ``"events"`` — one row per transition event (full detail)
      ``"unit"``   — mean cost aggregated by
                     (model_type, participant_id, trans_type, trans_dir_norm)
      ``"block"``  — block-level summary with all transition annotations
    """
    # ── Filter to training-active rows ───────────────────────────────────────
    if "training_active" in df.columns:
        df_filt = df[df["training_active"] == True].copy()
    else:
        df_filt = df.copy()

    # ── Normalise context labels ──────────────────────────────────────────────
    def _norm_ctx(x: str) -> str:
        s = str(x).strip().lower()
        if "maze" in s:
            return "Maze"
        if "melody" in s:
            return "Melody"
        return str(x).strip().title()

    df_filt = df_filt.copy()
    df_filt["context"] = df_filt["context"].apply(_norm_ctx)
    df_filt = df_filt.dropna(subset=["context"])

    # ── Block-level summary ───────────────────────────────────────────────────
    seq_col   = "sequence_in_block" if "sequence_in_block" in df_filt.columns else None
    order_col = "global_step"       if (use_global_step and "global_step" in df_filt.columns) else "block_id"

    def _block_summary(g: pd.DataFrame) -> pd.Series:
        if seq_col and seq_col in g.columns:
            start_row = g.loc[g[seq_col].idxmin()]
            end_row   = g.loc[g[seq_col].idxmax()]
        else:
            start_row = g.iloc[0]
            end_row   = g.iloc[-1]
        return pd.Series({
            "dv_start":    float(start_row[dv]),
            "dv_end":      float(end_row[dv]),
            "block_order": float(g[order_col].min()),
        })

    block = (
        df_filt
        .groupby(["model_type", "participant_id", "block_id", "context"],
                 sort=False, observed=True)
        .apply(_block_summary)
        .reset_index()
        .sort_values(["model_type", "participant_id", "block_order"])
        .reset_index(drop=True)
    )

    # ── Starting context per (model_type, participant_id) ────────────────────
    starts = (
        block
        .sort_values("block_order")
        .groupby(["model_type", "participant_id"], sort=False, observed=True)
        .first()
        .reset_index()[["model_type", "participant_id", "context"]]
        .rename(columns={"context": "start_context"})
    )
    block = block.merge(starts, on=["model_type", "participant_id"], how="left")

    def _other_ctx(ctx: str) -> str:
        if ctx == "Maze":
            return "Melody"
        if ctx == "Melody":
            return "Maze"
        return ""

    block["other"] = block["start_context"].apply(_other_ctx)

    # ── Transition columns ────────────────────────────────────────────────────
    def _add_transitions(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("block_order").copy()
        g["prev_context"] = g["context"].shift(1)
        g["prev_end"]     = g["dv_end"].shift(1)
        g["is_transition"] = g["prev_context"].notna()
        g["is_switch"]     = g["is_transition"] & (g["context"] != g["prev_context"])

        g["trans_cost"] = np.where(
            g["is_transition"],
            np.abs(g["dv_start"] - g["prev_end"]),
            np.nan,
        )
        g["trans_type"] = np.where(
            ~g["is_transition"], None,
            np.where(g["is_switch"], "switch", "stay"),
        )

        def _dir_raw(row):
            if not row["is_transition"]:
                return None
            return f"{row['prev_context']}→{row['context']}"

        def _dir_norm(row):
            if not row["is_transition"]:
                return None
            sc, oc = row["start_context"], row["other"]
            pc, cc = row["prev_context"],  row["context"]
            if pc == sc and cc == oc:
                return "A→B"
            if pc == oc and cc == sc:
                return "B→A"
            return "other"

        g["trans_dir_raw"]  = g.apply(_dir_raw,  axis=1)
        g["trans_dir_norm"] = g.apply(_dir_norm, axis=1)
        return g

    block2 = (
        block
        .groupby(["model_type", "participant_id"], sort=False, observed=True,
                 group_keys=False)
        .apply(_add_transitions)
        .reset_index(drop=True)
    )

    # ── Events (one row per transition) ──────────────────────────────────────
    events_cols = [
        "model_type", "participant_id", "block_id", "block_order",
        "start_context", "prev_context", "context",
        "trans_type", "trans_dir_raw", "trans_dir_norm", "trans_cost",
    ]
    events = block2[block2["trans_type"].notna()][events_cols].copy().reset_index(drop=True)

    # ── Unit summary (mean cost per type × direction) ─────────────────────────
    unit = (
        events
        .groupby(["model_type", "participant_id", "trans_type", "trans_dir_norm"],
                 observed=True)
        .agg(
            mean_cost=("trans_cost", "mean"),
            n_trans=("trans_cost", "count"),
        )
        .reset_index()
    )

    return {"events": events, "unit": unit, "block": block2}
