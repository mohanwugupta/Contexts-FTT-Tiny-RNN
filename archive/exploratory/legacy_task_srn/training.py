"""
srn.training
============
Experiment runners that execute model training on participant timelines or
train models to asymptote for representational-geometry analysis.

Public API
----------
run_participant_experiment           — row-by-row participant timeline
run_to_asymptote                     — full-sequence convergence training
run_matched_performance_analysis     — CE/accuracy-matched geometry comparison
run_participant_experiment_with_mechanistics  — mechanistic sweep (PRD2 pipeline)
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import ModelConfig, VOCAB_SIZE, FIXED_SEQUENCE, PERF_THRESHOLDS
from .model import ContextSequenceRNN, gaussian_vector, sequence_to_gaussian
from .metrics import (
    compute_sequence_accuracy,
    compute_prediction_entropy,
    compute_prediction_confidence,
    compute_correct_trial_surprisal,
    compute_gradient_alignment_tensors,
    compute_jacobian_stats,
    compute_per_context_dimensionality,
    compute_effective_dimensionality,
    compute_pca_overlap,
    compute_weight_norms,
    run_mechanistic_analysis,
    compute_directional_transfer,
)


# =============================================================================
# 1. Participant-timeline experiment runner
# =============================================================================

def run_participant_experiment(
    df_participant: pd.DataFrame,
    config: ModelConfig,
    vocab_size: int = VOCAB_SIZE,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    single_context_mode: bool = False,
    train_context: str = "Maze",
    context_encodings: Optional[Dict[str, torch.Tensor]] = None,
    context_col: str = "group",
    snapshot_loss_thresholds: Optional[List[float]] = None,
    snapshot_accuracy_thresholds: Optional[List[float]] = None,
    snapshot_step_checkpoints: Optional[List[int]] = None,
    step_interval: Optional[int] = None,
) -> Tuple[Dict, nn.Module, Dict, List[Dict]]:
    """Run one participant's trial sequence through the model.

    Snapshot modes (all optional, combinable):

    - ``snapshot_loss_thresholds``     : capture the first time CE ≤ threshold
    - ``snapshot_accuracy_thresholds`` : capture the first time accuracy ≥ threshold
    - ``snapshot_step_checkpoints``    : capture at specific global_step values
    - ``step_interval``                : capture every N steps (dense checkpointing)

    All captured snapshots share the same ``snapshots`` dict keyed by strings:
    ``"ce_<thr>"``, ``"acc_<thr>"``, ``"step_<n>"``.

    Returns
    -------
    results           : per-step dict (loss, accuracy, entropy, confidence, …)
    model             : final trained model
    snapshots         : Dict[str, nn.Module]
    snapshot_metadata : List[Dict] — per-snapshot metadata rows
    """
    if context_encodings is None:
        context_encodings = {
            "Maze":   torch.tensor([1.0, 0.0]),
            "Melody": torch.tensor([0.0, 1.0]),
        }

    context_dim = next(iter(context_encodings.values())).shape[0]
    model       = ContextSequenceRNN(vocab_size, context_dim=context_dim, config=config)
    optimizer   = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion   = nn.CrossEntropyLoss()

    results = {
        "block_id":           [],
        "sequence_in_block":  [],
        "global_step":        [],
        "context":            [],
        "loss":               [],
        "transfer_loss":      [],
        "accuracy":           [],
        "transfer_accuracy":  [],
        "entropy":            [],
        "confidence":         [],
        "correct_surprisal":  [],
        "rt":                 [],
        "training_active":    [],
        "counterbalance":     [],
        "weight_ratio_cx_in":      [],   # context / input weight norm ratio at each step
        "norm_input_weights":      [],   # ||W_input||
        "norm_context_weights":    [],   # ||W_context||
        "norm_recurrent_weights":  [],   # ||W_hh||
        "norm_output_weights":     [],   # ||W_output||
    }

    snapshots:              Dict[str, nn.Module] = {}
    thresholds_remaining:   set = set(snapshot_loss_thresholds or [])
    acc_thresholds_remaining: set = set(snapshot_accuracy_thresholds or [])
    step_checkpoints_remaining: set = set(snapshot_step_checkpoints or [])
    snapshot_metadata: List[Dict] = []

    global_step = 0

    for _, row in df_participant.iterrows():
        block_id          = row["new_trialNumber"]
        seq_num           = row.get("sequenceNumber", 0)
        curr_context_name = row[context_col]
        rt                = row["meanRT"]
        cb                = row.get("counterBalance", np.nan)
        global_step      += 1

        if curr_context_name not in context_encodings:
            curr_context_name = list(context_encodings.keys())[0]

        curr_context_vec  = context_encodings[curr_context_name].unsqueeze(0)

        all_ctx_names = list(context_encodings.keys())
        if len(all_ctx_names) > 1:
            others             = [c for c in all_ctx_names if c != curr_context_name]
            other_context_name = others[0]
        else:
            other_context_name = curr_context_name
        other_context_vec = context_encodings[other_context_name].unsqueeze(0)

        # Control condition: skip training on non-target contexts
        train_active = True
        if single_context_mode and curr_context_name != train_context:
            train_active = False

        inputs  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
        targets = torch.tensor(fixed_sequence[1:], dtype=torch.long)

        # --- 1. EVALUATION (before training update) ---
        model.eval()
        with torch.no_grad():
            logits, _       = model(inputs, curr_context_vec)
            loss_val        = criterion(logits.squeeze(0), targets).item()
            acc             = compute_sequence_accuracy(logits, targets)
            entropy         = compute_prediction_entropy(logits)
            conf            = compute_prediction_confidence(logits)
            c_surp          = compute_correct_trial_surprisal(logits, targets)

            logits_tx, _    = model(inputs, other_context_vec)
            loss_tx         = criterion(logits_tx.squeeze(0), targets).item()
            acc_tx          = compute_sequence_accuracy(logits_tx, targets)

        # --- Snapshots: CE thresholds ---
        for thr in list(thresholds_remaining):
            if loss_val <= thr:
                key = f"ce_{thr}"
                snapshots[key] = copy.deepcopy(model)
                thresholds_remaining.discard(thr)
                snapshot_metadata.append({
                    "snapshot_type": "ce", "snapshot_value": thr,
                    "global_step": global_step, "block_id": block_id,
                    "context": curr_context_name, "loss": loss_val,
                    "accuracy": acc, "confidence": conf,
                })

        # --- Snapshots: accuracy thresholds ---
        for thr in list(acc_thresholds_remaining):
            if acc >= thr:
                key = f"acc_{thr}"
                snapshots[key] = copy.deepcopy(model)
                acc_thresholds_remaining.discard(thr)
                snapshot_metadata.append({
                    "snapshot_type": "accuracy", "snapshot_value": thr,
                    "global_step": global_step, "block_id": block_id,
                    "context": curr_context_name, "loss": loss_val,
                    "accuracy": acc, "confidence": conf,
                })

        # --- Snapshots: explicit step checkpoints ---
        if global_step in step_checkpoints_remaining:
            key = f"step_{global_step}"
            snapshots[key] = copy.deepcopy(model)
            step_checkpoints_remaining.discard(global_step)
            snapshot_metadata.append({
                "snapshot_type": "step", "snapshot_value": global_step,
                "global_step": global_step, "block_id": block_id,
                "context": curr_context_name, "loss": loss_val,
                "accuracy": acc, "confidence": conf,
            })

        # --- Snapshots: dense step-interval checkpointing ---
        if step_interval and global_step % step_interval == 0:
            key = f"step_{global_step}"
            if key not in snapshots:
                snapshots[key] = copy.deepcopy(model)
            snapshot_metadata.append({
                "snapshot_type": "interval", "snapshot_value": global_step,
                "global_step": global_step, "block_id": block_id,
                "context": curr_context_name, "loss": loss_val,
                "accuracy": acc, "confidence": conf,
            })

        # --- Record data ---
        results["block_id"].append(block_id)
        results["sequence_in_block"].append(seq_num)
        results["global_step"].append(global_step)
        results["context"].append(curr_context_name)
        results["loss"].append(loss_val)
        results["transfer_loss"].append(loss_tx)
        results["accuracy"].append(acc)
        results["transfer_accuracy"].append(acc_tx)
        results["entropy"].append(entropy)
        results["confidence"].append(conf)
        results["correct_surprisal"].append(c_surp)
        results["rt"].append(rt)
        results["training_active"].append(train_active)
        results["counterbalance"].append(cb)
        # weight norms: pre-update (consistent with pre-update loss_val above)
        _wn = compute_weight_norms(model, vocab_size=vocab_size)
        results["weight_ratio_cx_in"].append(_wn.get("context_input_ratio",     np.nan))
        results["norm_input_weights"].append(_wn.get("norm_input_weights",      np.nan))
        results["norm_context_weights"].append(_wn.get("norm_context_weights",  np.nan))
        results["norm_recurrent_weights"].append(_wn.get("norm_recurrent_weights", np.nan))
        results["norm_output_weights"].append(_wn.get("norm_output_weights",    np.nan))

        # --- 2. TRAINING (stepwise through sequence) ---
        if train_active:
            model.train()
            for t in range(len(fixed_sequence) - 1):
                token_idx      = fixed_sequence[t]
                next_token_idx = fixed_sequence[t + 1]

                step_input_vec = gaussian_vector(token_idx, vocab_size, config.sigma)
                step_input     = torch.tensor(
                    step_input_vec, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0)                                  # (1, 1, vocab)
                step_target    = torch.tensor([next_token_idx], dtype=torch.long)

                optimizer.zero_grad()
                step_logits, _ = model(step_input, curr_context_vec)
                loss           = criterion(step_logits.squeeze(0), step_target)
                loss.backward()
                optimizer.step()

    return results, model, snapshots, snapshot_metadata


# =============================================================================
# 2. Asymptote training
# =============================================================================

def run_to_asymptote(
    model: nn.Module,
    config: ModelConfig,
    fixed_sequence: List[int],
    vocab_size: int,
    context_encodings: Dict[str, torch.Tensor],
    max_epochs: int = 1000,
    tol: float = 1e-4,
    patience: int = 30,
    epoch_interval: Optional[int] = None,
    track_gradients: bool = False,
    snapshot_loss_thresholds: Optional[List[float]] = None,
    snapshot_accuracy_thresholds: Optional[List[float]] = None,
    alternating_contexts: bool = False,
) -> Tuple[nn.Module, List[float], Dict]:
    """Train a model on the fixed sequence until CE converges.

    By default, interleaves all contexts every epoch (one full-sequence pass
    per context per epoch).  When ``alternating_contexts=True``, each epoch
    trains on exactly **one** context, cycling through them in order
    (A, B, A, B, …).  This mirrors the block structure of the participant
    experiment and is the recommended setting for parameter sweeps.

    Stops when mean CE improvement < ``tol`` for ``patience`` consecutive epochs.

    Full-sequence training is intentional: single-step training gives h_0 = 0
    on every step, so W_hh receives zero gradient.  Full-sequence passes allow
    BPTT to flow through the recurrent weights.

    Parameters
    ----------
    alternating_contexts         : if True, train one context per epoch
                                   (cycling A→B→A→B…) rather than all contexts
                                   within each epoch.  Mirrors participant block
                                   structure.
    epoch_interval               : if set, deep-copy model every N epochs into
                                   ``extra_info['epoch_snapshots']``
    track_gradients              : if True, log per-epoch gradient norms into
                                   ``extra_info['grad_history']``
    snapshot_loss_thresholds     : capture a model copy the first time mean CE
                                   drops at or below each value; stored in
                                   ``extra_info['ce_snapshots']`` as ``"ce_<thr>"``
    snapshot_accuracy_thresholds : capture a model copy the first time mean
                                   accuracy reaches or exceeds each value;
                                   stored in ``extra_info['ce_snapshots']`` as
                                   ``"acc_<thr>"`` (checked every epoch_interval)

    Returns
    -------
    best_model   : deep-copy of the model at its lowest mean CE
    loss_history : list of per-epoch mean CE values
    extra_info   : dict with keys ``epoch_snapshots``, ``grad_history``,
                   ``accuracy_history``, ``ce_snapshots``
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    loss_history: List[float] = []
    best_loss        = float("inf")
    no_improve_count = 0
    best_model       = copy.deepcopy(model)

    epoch_snapshots:   Dict[int, nn.Module]  = {}
    accuracy_at_epoch: Dict[int, float]      = {}
    grad_history:      List[Dict]            = []
    ce_snapshots:      Dict[str, nn.Module]  = {}

    ce_thrs_remaining:  set = set(snapshot_loss_thresholds  or [])
    acc_thrs_remaining: set = set(snapshot_accuracy_thresholds or [])

    # Pre-build full-sequence tensors once
    inputs_seq  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets_seq = torch.tensor(fixed_sequence[1:], dtype=torch.long)

    ctx_keys = list(context_encodings.keys())

    for epoch in range(max_epochs):
        epoch_loss, n = 0.0, 0
        grad_ctx_accum, grad_rec_accum, grad_out_accum = 0.0, 0.0, 0.0
        grad_ctx_steps, grad_rec_steps, grad_out_steps = 0, 0, 0

        # Select which contexts to train this epoch:
        # alternating → one context, cycling by epoch index
        # interleaved → all contexts (original behaviour)
        if alternating_contexts:
            epoch_ctx_vecs = [context_encodings[ctx_keys[epoch % len(ctx_keys)]]]
        else:
            epoch_ctx_vecs = list(context_encodings.values())

        for ctx_vec in epoch_ctx_vecs:
            optimizer.zero_grad()
            logits, _ = model(inputs_seq, ctx_vec.unsqueeze(0))
            loss      = criterion(logits.squeeze(0), targets_seq)
            loss.backward()

            if track_gradients:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if "weight_ih" in name:
                        g = param.grad.detach()
                        grad_ctx_accum += g[:, vocab_size:].norm().item()
                        grad_ctx_steps += 1
                    elif "weight_hh" in name:
                        grad_rec_accum += param.grad.detach().norm().item()
                        grad_rec_steps += 1
                    elif "output_layer.weight" in name:
                        grad_out_accum += param.grad.detach().norm().item()
                        grad_out_steps += 1

            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / n
        loss_history.append(avg)

        # ── CE threshold snapshots (checked every epoch, free — avg already known) ─
        for thr in list(ce_thrs_remaining):
            if avg <= thr:
                ce_snapshots[f"ce_{thr}"] = copy.deepcopy(model)
                ce_thrs_remaining.discard(thr)

        if track_gradients and (grad_ctx_steps + grad_rec_steps + grad_out_steps) > 0:
            grad_history.append({
                "epoch":          epoch,
                "loss":           avg,
                "grad_context":   grad_ctx_accum / grad_ctx_steps if grad_ctx_steps > 0 else np.nan,
                "grad_recurrent": grad_rec_accum / grad_rec_steps if grad_rec_steps > 0 else np.nan,
                "grad_output":    grad_out_accum / grad_out_steps if grad_out_steps > 0 else np.nan,
            })

        if epoch_interval and (epoch + 1) % epoch_interval == 0:
            epoch_snapshots[epoch + 1] = copy.deepcopy(model)
            model.eval()
            acc_vals: List[float] = []
            with torch.no_grad():
                for ctx_vec in context_encodings.values():
                    logits_e, _ = model(inputs_seq, ctx_vec.unsqueeze(0))
                    acc_vals.append(compute_sequence_accuracy(logits_e, targets_seq))
            mean_acc = float(np.mean(acc_vals))
            accuracy_at_epoch[epoch + 1] = mean_acc
            model.train()

            # ── accuracy threshold snapshots (checked at epoch_interval) ──────
            for thr in list(acc_thrs_remaining):
                if mean_acc >= thr:
                    ce_snapshots[f"acc_{thr}"] = copy.deepcopy(model)
                    acc_thrs_remaining.discard(thr)

        if best_loss - avg > tol:
            best_loss        = avg
            no_improve_count = 0
            best_model       = copy.deepcopy(model)
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    extra_info = {
        "epoch_snapshots":  epoch_snapshots,
        "grad_history":     grad_history,
        "accuracy_history": accuracy_at_epoch,
        "ce_snapshots":     ce_snapshots,
    }
    return best_model, loss_history, extra_info


# =============================================================================
# 3. Unified participant-timeline → asymptote training (single run per participant)
# =============================================================================

def run_participant_to_asymptote(
    df_participant: pd.DataFrame,
    config: ModelConfig,
    vocab_size: int = VOCAB_SIZE,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    context_encodings: Optional[Dict[str, torch.Tensor]] = None,
    context_col: str = "group",
    snapshot_loss_thresholds: Optional[List[float]] = None,
    snapshot_accuracy_thresholds: Optional[List[float]] = None,
    step_interval: Optional[int] = None,
    max_epochs: int = 1000,
    tol: float = 1e-4,
    patience: int = 1000,
    epoch_interval: Optional[int] = 10,
    track_gradients: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Dict, nn.Module, nn.Module, Dict, List[Dict], List[float], Dict]:
    """Train through participant timeline then continue for a fixed epoch budget.

    This is the unified training entry-point used by the pipeline.  It sets a
    deterministic seed *before* model initialisation so every participant always
    receives the same weight initialisation regardless of when / how the script
    is run.

    Phases
    ------
    A. Participant timeline — one gradient step per trial, row-by-row through
       ``df_participant``.  Step-interval snapshots are captured here (for
       collapse / perturbation / driver / ablation analysis).  CE threshold
       snapshots are intentionally **not** captured here because the short
       participant trial sequence may not reach low CE values for all models.
    B. Fixed-budget continuation — full-sequence BPTT epochs from the *same
       model state* for exactly ``max_epochs`` epochs (default 1000).  Early
       stopping is disabled by default (``patience = max_epochs``).  CE /
       accuracy threshold snapshots are captured here, guaranteeing full
       coverage down to CE = 0.1 for both Rich and Poor.  Epoch snapshots and
       gradient dynamics are also collected here.

    Parameters
    ----------
    seed : if given, calls ``torch.manual_seed(seed)`` and
           ``numpy.random.seed(seed)`` before model initialisation.

    Returns
    -------
    results           : per-step dict (loss, accuracy, rt, …) from phase A
    model_participant : model state immediately after phase A (before asymptote)
    best_model        : model at lowest CE during phase B (asymptote)
    snapshots         : CE / accuracy threshold snapshots captured in phase A
    snapshot_metadata : metadata list from phase A snapshots
    loss_history      : per-epoch CE losses from phase B
    extra_info        : ``epoch_snapshots``, ``grad_history``,
                        ``accuracy_history`` from phase B
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ── Phase A: participant timeline ─────────────────────────────────────────
    # Step-interval snapshots only — CE/acc threshold snaps moved to Phase B
    # so all thresholds are guaranteed to be reached during asymptote training.
    results, model, snapshots, snapshot_metadata = run_participant_experiment(
        df_participant               = df_participant,
        config                       = config,
        vocab_size                   = vocab_size,
        fixed_sequence               = fixed_sequence,
        context_encodings            = context_encodings,
        context_col                  = context_col,
        snapshot_loss_thresholds     = None,   # ← disabled; captured in Phase B
        snapshot_accuracy_thresholds = None,   # ← disabled; captured in Phase B
        step_interval                = step_interval,
    )
    model_participant = copy.deepcopy(model)   # preserve post-timeline state

    # ── Phase B: asymptote continuation from same model state ─────────────────
    # CE / accuracy threshold snapshots captured here — training runs until
    # convergence so all thresholds down to CE = 0.1 are always covered.
    best_model, loss_history, extra_info = run_to_asymptote(
        model                        = model,
        config                       = config,
        fixed_sequence               = list(fixed_sequence),
        vocab_size                   = vocab_size,
        context_encodings            = context_encodings,
        max_epochs                   = max_epochs,
        tol                          = tol,
        patience                     = patience,
        epoch_interval               = epoch_interval,
        track_gradients              = track_gradients,
        snapshot_loss_thresholds     = snapshot_loss_thresholds,
        snapshot_accuracy_thresholds = snapshot_accuracy_thresholds,
    )

    return (results, model_participant, best_model,
            snapshots, snapshot_metadata, loss_history, extra_info)


# =============================================================================
# 4. Performance-matched geometry comparison (legacy — now superseded by the
#    unified pipeline which reuses CE snapshots from run_participant_to_asymptote)
# =============================================================================

def run_matched_performance_analysis(
    df_all: pd.DataFrame,
    valid_pids: List[str],
    configs: List[ModelConfig],
    context_encodings: Dict[str, torch.Tensor],
    context_col: str,
    thresholds: Optional[List[float]] = None,
    vocab_size: int = VOCAB_SIZE,
    n_participants: int = 5,
) -> pd.DataFrame:
    """Compare representational geometry at matched predictive performance.

    For each config × participant, trains the model following the participant
    timeline while capturing model snapshots each time CE crosses a threshold.
    Then runs the full mechanistic analysis on each snapshot.

    Returns
    -------
    DataFrame with one row per (config, participant, threshold) and geometry
    metrics, enabling comparisons like "Rich vs Poor when both have CE ≤ 1.0".

    Parameters
    ----------
    n_participants : how many participants to use (capped for speed)
    """
    if thresholds is None:
        thresholds = PERF_THRESHOLDS

    rows = []
    pids = valid_pids[:n_participants]

    for config in configs:
        if config.name == "Control (Rich)":
            continue

        print(f"   [matched] {config.name} — {len(pids)} participants × {len(thresholds)} thresholds")
        for pid in pids:
            # df_all here is the raw participant CSV — infer its id column
            pid_col = next(
                (c for c in ["ID", "participant_id", "pid", "subject"] if c in df_all.columns),
                df_all.columns[0],
            )
            p_data = df_all[df_all[pid_col] == pid]
            sort_cols = [c for c in ["new_trialNumber", "sequenceNumber", "global_step"]
                         if c in p_data.columns]
            if sort_cols:
                p_data = p_data.sort_values(by=sort_cols)
            p_data = p_data.reset_index(drop=True)
            _, _, snaps, _meta = run_participant_experiment(
                p_data, config,
                context_encodings=context_encodings,
                context_col=context_col,
                snapshot_loss_thresholds=thresholds,
            )

            for snap_key, snap_model in snaps.items():
                if not snap_key.startswith("ce_"):
                    continue
                thr  = float(snap_key[3:])
                mech = run_mechanistic_analysis(
                    snap_model, config,
                    context_encodings=context_encodings,
                )
                row = {"model_type": config.name, "participant_id": pid, "ce_threshold": thr}
                row.update({k: v for k, v in mech.items() if not isinstance(v, np.ndarray)})
                rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# 4. Mechanistic participant experiment (PRD2 pipeline)
# =============================================================================

def run_participant_experiment_with_mechanistics(
    df_participant: pd.DataFrame,
    config: ModelConfig,
    vocab_size: int = 4,
    fixed_sequence: Optional[List[int]] = None,
) -> Tuple[Dict, Dict]:
    """Run the experimental timeline with inline mechanistic probes (PRD2).

    Specific to the Maze / Melody two-context dataset.  Context encodings are
    hardcoded as one-hot 2D vectors:
        Maze   → [1, 0]
        Melody → [0, 1]

    Parameters
    ----------
    df_participant : single-participant DataFrame with columns
                     ``new_trialNumber``, ``sequenceNumber``, ``trialType``
    config         : ModelConfig (``track_gradients`` controls whether per-layer
                     gradient norms are recorded for each training step)
    vocab_size     : number of token types (default 4)
    fixed_sequence : 8-element integer list; defaults to ``FIXED_SEQUENCE``

    Returns
    -------
    results      : timeline dict (block_id, global_step, context, loss, grad_alignment)
    mechanistics : comprehensive mechanistic metrics dict
    """
    if fixed_sequence is None:
        fixed_sequence = list(FIXED_SEQUENCE)

    MODEL_DIM = 2
    context_encodings = {
        "Maze":   torch.tensor([1.0, 0.0]),
        "Melody": torch.tensor([0.0, 1.0]),
    }

    model     = ContextSequenceRNN(vocab_size, context_dim=MODEL_DIM, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Fixed probe inputs for gradient-alignment measurement
    probe_input  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    probe_target = torch.tensor(fixed_sequence[1:], dtype=torch.long)
    probe_ctx_A  = context_encodings["Maze"].unsqueeze(0)
    probe_ctx_B  = context_encodings["Melody"].unsqueeze(0)

    # Initial hidden-state saturation (BEFORE any training)
    probe_full = sequence_to_gaussian(fixed_sequence, vocab_size, config.sigma)
    initial_h_sats: Dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            _, h_init = model(probe_full, ctx_vec.unsqueeze(0))
            h_np = h_init.squeeze(0).cpu().numpy()
            initial_h_sats[ctx_name] = float(np.mean(np.abs(h_np)))
    model.train()

    # Gradient-norm trackers
    grad_norms:     List[float] = []
    grad_norms_ih:  List[float] = []
    grad_norms_hh:  List[float] = []
    grad_norms_out: List[float] = []

    results = {
        "block_id":       [],
        "global_step":    [],
        "context":        [],
        "loss":           [],
        "grad_alignment": [],
    }

    loss_history: List[Dict] = []    # for compute_directional_transfer

    global_step       = 0
    inputs            = None         # will be set in the loop (reused in post-hoc section)

    for _, row in df_participant.iterrows():
        block_id          = row["new_trialNumber"]
        seq_num           = row.get("sequenceNumber", 0)
        curr_context_name = row["trialType"]
        global_step      += 1

        curr_context_vec  = context_encodings[curr_context_name].unsqueeze(0)
        other_context_name = "Melody" if curr_context_name == "Maze" else "Maze"
        other_context_vec  = context_encodings[other_context_name].unsqueeze(0)

        inputs  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
        targets = torch.tensor(fixed_sequence[1:], dtype=torch.long)

        # 1. Performance probe (before update)
        model.eval()
        with torch.no_grad():
            logits, _    = model(inputs, curr_context_vec)
            loss_val     = criterion(logits.squeeze(0), targets).item()
            logits_tx, _ = model(inputs, other_context_vec)
            loss_tx      = criterion(logits_tx.squeeze(0), targets).item()

        # 2. Gradient-alignment probe (start of each block)
        grad_sim = np.nan
        if config.track_gradients and seq_num == 1:
            grad_sim = compute_gradient_alignment_tensors(
                model,
                probe_input, probe_target, probe_ctx_A,
                probe_input, probe_target, probe_ctx_B,
                criterion,
            )

        results["block_id"].append(block_id)
        results["global_step"].append(global_step)
        results["context"].append(curr_context_name)
        results["loss"].append(loss_val)
        results["grad_alignment"].append(grad_sim)

        loss_history.append({
            "step":       global_step,
            "context":    curr_context_name,
            "loss_curr":  loss_val,
            "loss_other": loss_tx,
        })

        # 3. Training step (token-by-token)
        model.train()
        for t in range(len(fixed_sequence) - 1):
            token_idx      = fixed_sequence[t]
            next_token_idx = fixed_sequence[t + 1]

            step_input_vec = gaussian_vector(token_idx, vocab_size, config.sigma)
            step_input     = torch.tensor(
                step_input_vec, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)
            step_target    = torch.tensor([next_token_idx], dtype=torch.long)

            optimizer.zero_grad()
            step_logits, _ = model(step_input, curr_context_vec)
            loss           = criterion(step_logits.squeeze(0), step_target)
            loss.backward()

            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms.append(total_norm)

            g_ih  = model.rnn.weight_ih_l0.grad
            g_hh  = model.rnn.weight_hh_l0.grad
            g_out = model.output_layer.weight.grad
            if g_ih  is not None: grad_norms_ih.append(g_ih.data.norm(2).item())
            if g_hh  is not None: grad_norms_hh.append(g_hh.data.norm(2).item())
            if g_out is not None: grad_norms_out.append(g_out.data.norm(2).item())

            optimizer.step()

    # -------------------------------------------------------------------------
    # Post-hoc mechanistic analysis
    # -------------------------------------------------------------------------
    mechanistics: Dict = {}

    # Context-sensitivity metrics
    ctx_sens = compute_directional_transfer(loss_history)
    mechanistics.update(ctx_sens)

    # Initial loss per context
    initial_loss: Dict[str, float] = {}
    for entry in loss_history:
        ctx = entry["context"]
        if ctx not in initial_loss:
            initial_loss[ctx] = entry["loss_curr"]
        if len(initial_loss) == 2:
            break
    if loss_history and "Melody" not in initial_loss:
        initial_loss["Melody"] = loss_history[0]["loss_other"]
    if loss_history and "Maze" not in initial_loss:
        initial_loss["Maze"] = loss_history[0]["loss_other"]

    mechanistics["initial_loss_Maze"]   = initial_loss.get("Maze",   np.nan)
    mechanistics["initial_loss_Melody"] = initial_loss.get("Melody", np.nan)

    mechanistics["initial_h_saturation_Maze"]   = initial_h_sats.get("Maze",   np.nan)
    mechanistics["initial_h_saturation_Melody"] = initial_h_sats.get("Melody", np.nan)

    mechanistics["mean_grad_norm"] = float(np.mean(grad_norms))  if grad_norms  else np.nan
    mechanistics["max_grad_norm"]  = float(np.max(grad_norms))   if grad_norms  else np.nan
    mechanistics["mean_grad_norm_ih"]  = float(np.mean(grad_norms_ih))  if grad_norms_ih  else np.nan
    mechanistics["mean_grad_norm_hh"]  = float(np.mean(grad_norms_hh))  if grad_norms_hh  else np.nan
    mechanistics["mean_grad_norm_out"] = float(np.mean(grad_norms_out)) if grad_norms_out else np.nan

    _ih  = mechanistics["mean_grad_norm_ih"]
    _out = mechanistics["mean_grad_norm_out"]
    mechanistics["grad_ratio_ih_out"] = (
        float(_ih / max(_out, 1e-12))
        if not (np.isnan(_ih) or np.isnan(_out))
        else np.nan
    )

    # Collect hidden states
    h_states: Dict[str, np.ndarray] = {}
    inputs_full = sequence_to_gaussian(fixed_sequence, vocab_size, config.sigma)

    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            _, h_seq = model(inputs_full, ctx_vec.unsqueeze(0))
            h_states[ctx_name] = h_seq.squeeze(0).cpu().numpy()

            j_stats = compute_jacobian_stats(model, inputs_full, ctx_vec.unsqueeze(0))
            mechanistics[f"jacobian_spec_rad_{ctx_name}"] = j_stats["mean_spectral_radius"]

            ctx_dim = compute_per_context_dimensionality(h_states[ctx_name])
            mechanistics[f"eff_dim_{ctx_name}"]      = ctx_dim["eff_dim"]
            mechanistics[f"h_variance_{ctx_name}"]   = ctx_dim["h_variance"]
            mechanistics[f"h_mean_norm_{ctx_name}"]  = ctx_dim["h_mean_norm"]
            mechanistics[f"h_saturation_{ctx_name}"] = ctx_dim["h_saturation"]

    all_h_states = np.concatenate([h_states["Maze"], h_states["Melody"]], axis=0)
    mechanistics["eff_dim_overall"]    = compute_effective_dimensionality(all_h_states)
    mechanistics["eff_dim_asymmetry"]  = abs(
        mechanistics["eff_dim_Maze"] - mechanistics["eff_dim_Melody"]
    )
    mechanistics.update(compute_pca_overlap(h_states["Maze"], h_states["Melody"]))

    # Causal ablation (delta loss) — ``inputs`` still holds the last-iteration tensor
    if inputs is not None:
        with torch.no_grad():
            l_mA, _ = model(inputs, context_encodings["Maze"].unsqueeze(0))
            loss_base_A = criterion(l_mA.squeeze(0), targets).item()

            l_mB, _ = model(inputs, context_encodings["Melody"].unsqueeze(0))
            loss_base_B = criterion(l_mB.squeeze(0), targets).item()

            mechanistics["final_loss_Maze"]   = loss_base_A
            mechanistics["final_loss_Melody"] = loss_base_B

            l_sA, _ = model(inputs, context_encodings["Melody"].unsqueeze(0))
            loss_swap_A = criterion(l_sA.squeeze(0), targets).item()

            l_sB, _ = model(inputs, context_encodings["Maze"].unsqueeze(0))
            loss_swap_B = criterion(l_sB.squeeze(0), targets).item()

            delta_maze   = loss_swap_A - loss_base_A
            delta_melody = loss_swap_B - loss_base_B

            mechanistics["ablation_swap_delta_maze"]   = delta_maze
            mechanistics["ablation_swap_delta_melody"] = delta_melody
            mechanistics["ablation_swap_delta"]        = (abs(delta_maze) + abs(delta_melody)) / 2

            zero_ctx = torch.zeros(1, MODEL_DIM)
            l_z, _   = model(inputs, zero_ctx)
            loss_zero = criterion(l_z.squeeze(0), targets).item()
            avg_base  = (loss_base_A + loss_base_B) / 2
            mechanistics["ablation_zero_delta"] = loss_zero - avg_base

    # Final accuracy per context
    inputs_acc  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets_acc = torch.tensor(fixed_sequence[1:], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            logits_acc, _ = model(inputs_acc, ctx_vec.unsqueeze(0))
            preds = logits_acc.squeeze(0).argmax(dim=-1)
            acc   = (preds == targets_acc).float().mean().item()
            mechanistics[f"final_accuracy_{ctx_name}"] = acc

    return results, mechanistics
