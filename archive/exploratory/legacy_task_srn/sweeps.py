"""
srn.sweeps
==========
Parameter-robustness sweeps (PRD4 Parts 1–4).

Public API
----------
_compute_trajectory_summary   — compact scalar summary of a separation trajectory
_sweep_run_one                — train one config with alternating context blocks
run_hidden_size_sweep         — sweep hidden_dim × {Rich, Poor}
run_learning_rate_sweep       — sweep learning_rate × {Rich, Poor}
run_init_scale_sweep          — sweep init_scale
compute_model_summary_metrics — aggregate paper-ready metrics from multi-DataFrame inputs
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig, VOCAB_SIZE, FIXED_SEQUENCE, RICH_CONFIG, POOR_CONFIG
from .model import ContextSequenceRNN, sequence_to_gaussian
from .metrics import (
    compute_collapse_metrics, compute_sequence_accuracy,
    compute_weight_norms,
)


# =============================================================================
# 1. Trajectory-summary helper
# =============================================================================

def _compute_trajectory_summary(
    sep_vals:     List[float],
    ctx_dec_vals: List[float],
    seq_dec_vals: List[float],
    epoch_vals:   List[int],
    final_loss:   float,
    final_acc:    float,
) -> Dict:
    """Compress a separation trajectory into compact scalar metrics (PRD4 Part 4).

    Parameters
    ----------
    sep_vals      : trajectory_separation at each checkpoint
    ctx_dec_vals  : context_decodability at each checkpoint
    seq_dec_vals  : sequence_step_decodability at each checkpoint
    epoch_vals    : checkpoint indices (epochs or steps)
    final_loss    : CE loss at convergence
    final_acc     : accuracy at convergence
    """
    sep     = np.array(sep_vals,     dtype=float)
    ctx_dec = np.array(ctx_dec_vals, dtype=float)
    seq_dec = np.array(seq_dec_vals, dtype=float)
    epochs  = np.array(epoch_vals,   dtype=float)

    _nan: Dict = {
        "peak_separation":             np.nan,
        "step_to_peak":                np.nan,
        "final_separation":            np.nan,
        "auc_separation":              np.nan,
        "compression_amount":          np.nan,
        "compression_ratio":           np.nan,
        "peak_context_decodability":   np.nan,
        "final_context_decodability":  np.nan,
        "peak_sequence_decodability":  np.nan,
        "final_sequence_decodability": np.nan,
        "final_loss":                  final_loss,
        "final_accuracy":              final_acc,
    }
    if len(sep) == 0 or np.all(np.isnan(sep)):
        return _nan

    valid = ~np.isnan(sep)
    if not valid.any():
        return _nan

    peak_idx     = int(np.nanargmax(sep))
    peak_sep     = float(sep[peak_idx])
    step_to_peak = int(epochs[peak_idx])
    final_sep    = float(sep[valid][-1])
    auc          = float(np.trapz(sep[valid], epochs[valid])) if valid.sum() > 1 else np.nan

    ctx_valid = ctx_dec[valid]
    seq_valid = seq_dec[valid]

    return {
        "peak_separation":             peak_sep,
        "step_to_peak":                step_to_peak,
        "final_separation":            final_sep,
        "auc_separation":              auc,
        "compression_amount":          peak_sep - final_sep,
        "compression_ratio":           final_sep / peak_sep if peak_sep > 1e-9 else np.nan,
        "peak_context_decodability":   float(np.nanmax(ctx_dec)) if len(ctx_valid) else np.nan,
        "final_context_decodability":  float(ctx_valid[-1])      if len(ctx_valid) else np.nan,
        "peak_sequence_decodability":  float(np.nanmax(seq_dec)) if len(seq_valid) else np.nan,
        "final_sequence_decodability": float(seq_valid[-1])      if len(seq_valid) else np.nan,
        "final_loss":                  final_loss,
        "final_accuracy":              final_acc,
    }


# =============================================================================
# 2. Single-config sweep worker
# =============================================================================

_BLOCK_SIZE  = 10   # full-sequence passes per context block
_MAX_EPOCHS  = 500  # total training passes (= 25 full A→B cycles)


def _sweep_run_one(
    cfg: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int],
    vocab_size: int,
) -> List[Dict]:
    """Train one model config for exactly ``_MAX_EPOCHS`` passes in a block
    structure (``_BLOCK_SIZE`` passes of context A, then ``_BLOCK_SIZE`` of B,
    repeat) and return one tidy row **per epoch**.

    Row schema
    ----------
    Identifiers:
      model_type, participant_id, block_id, sequence_in_block,
      global_step, context, checkpoint_type, step_or_epoch, checkpoint_value

    Per-epoch performance (every row):
      loss, accuracy, weight_ratio_cx_in, norm_input_weights,
      norm_context_weights, norm_recurrent_weights, norm_output_weights,
      hidden_dim, learning_rate, init_scale

    Geometry (populated at block boundaries — every ``_BLOCK_SIZE`` epochs,
    i.e. the last pass of each block — NaN otherwise):
      euclidean_ctx_distance, cosine_ctx_distance, centroid_ctx_distance,
      within_traj_distance, collapse_index, context_decodability,
      sequence_step_decodability, effective_dimensionality, trajectory_separation

    Summary scalars broadcast to **every** row:
      peak_separation, final_separation, final_loss, final_accuracy,
      starting_dimensionality, ending_dimensionality,
      tc_loss_mean_switch, tc_loss_mean_stay, tc_loss_switch_minus_stay,
      tc_accuracy_mean_switch, tc_accuracy_mean_stay,
      tc_accuracy_switch_minus_stay  (and per-direction variants)

    Transition cost is computed directly from the per-epoch training rows
    (no post-hoc simulation needed) using ``compute_transition_cost``.
    """
    import torch.nn as _nn

    ctx_keys    = list(context_encodings.keys())
    n_ctx       = len(ctx_keys)
    context_dim = next(iter(context_encodings.values())).shape[0]

    model     = ContextSequenceRNN(vocab_size, context_dim=context_dim, config=cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = _nn.CrossEntropyLoss()

    inputs_seq  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, cfg.sigma)
    targets_seq = torch.tensor(fixed_sequence[1:], dtype=torch.long)
    # For mean-accuracy evaluation across all contexts
    inputs_e    = inputs_seq
    tgts_e      = targets_seq

    run_id = (
        f"sweep_{cfg.name}_h{cfg.hidden_dim}"
        f"_lr{cfg.learning_rate:.5f}_s{cfg.init_scale:.3f}"
    )

    # Geometry column names (populated at block boundaries only)
    _GEO_KEYS: List[str] = [
        "euclidean_ctx_distance", "cosine_ctx_distance", "centroid_ctx_distance",
        "within_traj_distance", "collapse_index", "context_decodability",
        "sequence_step_decodability", "effective_dimensionality",
        "per_context_effective_dimensionality", "dimensionality_expansion",
        "trajectory_separation",
    ]

    training_rows: List[Dict] = []   # one row per epoch — used for transition cost

    model.train()
    for epoch in range(1, _MAX_EPOCHS + 1):
        # ── Block / context assignment ────────────────────────────────────────
        block_id          = (epoch - 1) // _BLOCK_SIZE + 1
        seq_in_block      = (epoch - 1) % _BLOCK_SIZE          # 0 … _BLOCK_SIZE-1
        ctx_cycle_idx     = (block_id - 1) % n_ctx
        ctx_name          = ctx_keys[ctx_cycle_idx]
        ctx_vec           = context_encodings[ctx_name].unsqueeze(0)

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        logits, _ = model(inputs_seq, ctx_vec)
        loss_t    = criterion(logits.squeeze(0), targets_seq)
        loss_t.backward()
        optimizer.step()
        ep_loss = loss_t.item()

        # ── Evaluation (mean accuracy across all contexts) ────────────────────
        model.eval()
        acc_vals: List[float] = []
        with torch.no_grad():
            for cv in context_encodings.values():
                lg, _ = model(inputs_e, cv.unsqueeze(0))
                acc_vals.append(compute_sequence_accuracy(lg, tgts_e))
        ep_acc = float(np.mean(acc_vals))

        # ── Weight attribution ────────────────────────────────────────────────
        wn = compute_weight_norms(model, vocab_size)

        # ── Geometry — only at the last pass of each block ────────────────────
        is_block_end = (seq_in_block == _BLOCK_SIZE - 1)
        if is_block_end:
            col_m = compute_collapse_metrics(
                model, cfg, context_encodings, fixed_sequence, vocab_size
            )
        else:
            col_m = {}

        row: Dict = {
            # Block / context identifiers
            "model_type":        cfg.name,
            "participant_id":    run_id,
            "block_id":          block_id,
            "sequence_in_block": seq_in_block,
            "global_step":       epoch,
            "context":           ctx_name,
            # Tidy aliases used by downstream pipeline functions
            "checkpoint_type":   "epoch",
            "step_or_epoch":     epoch,
            "checkpoint_value":  epoch,
            # Performance
            "loss":              ep_loss,
            "accuracy":          ep_acc,
            # Sweep parameters
            "hidden_dim":        cfg.hidden_dim,
            "learning_rate":     cfg.learning_rate,
            "init_scale":        cfg.init_scale,
            # Geometry (NaN for non-block-end rows)
            **{k: col_m.get(k, np.nan) for k in _GEO_KEYS},
            # Weight norms
            "weight_ratio_cx_in":     wn.get("context_input_ratio",     np.nan),
            "norm_input_weights":     wn.get("norm_input_weights",      np.nan),
            "norm_context_weights":   wn.get("norm_context_weights",    np.nan),
            "norm_recurrent_weights": wn.get("norm_recurrent_weights",  np.nan),
            "norm_output_weights":    wn.get("norm_output_weights",     np.nan),
        }
        training_rows.append(row)

    if not training_rows:
        return []

    # ── Summary scalars from geometry trajectory ──────────────────────────────
    snap_rows    = [r for r in training_rows if not np.isnan(r.get("trajectory_separation", np.nan))]
    sep_vals     = [r["trajectory_separation"]      for r in snap_rows]
    ctx_dec_vals = [r["context_decodability"]       for r in snap_rows]
    seq_dec_vals = [r["sequence_step_decodability"] for r in snap_rows]
    ep_vals      = [r["step_or_epoch"]              for r in snap_rows]

    final_loss = training_rows[-1]["loss"]
    final_acc  = training_rows[-1]["accuracy"]

    summary = _compute_trajectory_summary(
        sep_vals, ctx_dec_vals, seq_dec_vals, ep_vals, final_loss, final_acc
    )

    start_geo = next((r for r in training_rows if not np.isnan(r.get("effective_dimensionality", np.nan))), {})
    end_geo   = next((r for r in reversed(training_rows) if not np.isnan(r.get("effective_dimensionality", np.nan))), {})

    # ── Transition cost — per block boundary + run-level mean ────────────────
    # Jump = |loss_start_of_block_(b+1) − loss_end_of_block_b|
    # Placed on the BLOCK-END row (sequence_in_block == _BLOCK_SIZE-1) so that
    # it co-occupies the same row as the geometry metrics, enabling direct
    # correlation (e.g. centroid_ctx_distance ~ tc_loss_jump).
    block_end:   Dict[int, Dict] = {}   # block_id → {"loss": ..., "accuracy": ...}
    block_start: Dict[int, Dict] = {}   # block_id → {"loss": ..., "accuracy": ...}

    for r in training_rows:
        bid = r["block_id"]
        if r["sequence_in_block"] == 0:
            block_start[bid] = {"loss": r["loss"], "accuracy": r["accuracy"]}
        if r["sequence_in_block"] == _BLOCK_SIZE - 1:
            block_end[bid]   = {"loss": r["loss"], "accuracy": r["accuracy"]}

    # jump for block b  =  |start(b+1) − end(b)|  → placed on block-end row b
    jump_by_block_end: Dict[int, Dict] = {}   # block_id (b) → jump values
    sorted_blocks = sorted(block_end.keys())
    tc_loss_jumps: List[float] = []
    tc_acc_jumps:  List[float] = []

    for i in range(len(sorted_blocks) - 1):
        b      = sorted_blocks[i]
        b_next = sorted_blocks[i + 1]
        if b in block_end and b_next in block_start:
            lj = abs(block_start[b_next]["loss"]     - block_end[b]["loss"])
            aj = abs(block_start[b_next]["accuracy"] - block_end[b]["accuracy"])
            jump_by_block_end[b] = {"tc_loss_jump": lj, "tc_accuracy_jump": aj}
            tc_loss_jumps.append(lj)
            tc_acc_jumps.append(aj)

    # Inject per-row jump values onto block-end rows (NaN elsewhere)
    for r in training_rows:
        if r["sequence_in_block"] == _BLOCK_SIZE - 1 and r["block_id"] in jump_by_block_end:
            r["tc_loss_jump"]     = jump_by_block_end[r["block_id"]]["tc_loss_jump"]
            r["tc_accuracy_jump"] = jump_by_block_end[r["block_id"]]["tc_accuracy_jump"]
        else:
            r["tc_loss_jump"]     = np.nan
            r["tc_accuracy_jump"] = np.nan

    tc_scalars: Dict = {
        "tc_loss_mean_switch":     float(np.mean(tc_loss_jumps)) if tc_loss_jumps else np.nan,
        "tc_accuracy_mean_switch": float(np.mean(tc_acc_jumps))  if tc_acc_jumps  else np.nan,
    }

    summary_broadcast = {
        **summary,
        "starting_dimensionality": start_geo.get("effective_dimensionality", np.nan),
        "ending_dimensionality":   end_geo.get("effective_dimensionality",   np.nan),
        **tc_scalars,
    }

    for row in training_rows:
        row.update(summary_broadcast)

    return training_rows


# =============================================================================
# 3. Robustness sweeps
# =============================================================================

def run_hidden_size_sweep(
    context_encodings: Dict[str, torch.Tensor],
    output_dir: Path,
    hidden_sizes: Optional[List[int]] = None,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> pd.DataFrame:
    """PRD4 Part 1: sweep hidden_dim × {Rich, Poor} and record per-epoch geometry.

    For each (model_type, hidden_dim), trains a fresh model for ``_MAX_EPOCHS``
    epochs in alternating context blocks of ``_BLOCK_SIZE``.  One tidy row is
    produced per epoch; geometry metrics are populated at block boundaries only.

    Parameters
    ----------
    hidden_sizes : list of integer hidden dimensions to evaluate.
                   Defaults to 20 log-spaced integers from 2 → 64.

    Outputs
    -------
    ``output_dir / hidden_size_sweep.csv``

    Returns
    -------
    DataFrame with one row per (model_type, hidden_dim, epoch)
    """
    if hidden_sizes is None:
        hidden_sizes = sorted(set(
            int(round(v))
            for v in np.logspace(np.log10(2), np.log10(64), num=20)
        ))

    rows: List[Dict] = []
    for hd in hidden_sizes:
        for base_cfg in [RICH_CONFIG, POOR_CONFIG]:
            cfg = ModelConfig(
                name=base_cfg.name,
                hidden_dim=hd,
                init_scale=base_cfg.init_scale,
                learning_rate=base_cfg.learning_rate,
                context_alpha=base_cfg.context_alpha,
                sigma=base_cfg.sigma,
            )
            print(f"      [hidden_size_sweep] {cfg.name}  hidden_dim={hd} …")
            rows.extend(_sweep_run_one(cfg, context_encodings, fixed_sequence, vocab_size))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "hidden_size_sweep.csv", index=False)
    return df


def run_learning_rate_sweep(
    context_encodings: Dict[str, torch.Tensor],
    output_dir: Path,
    learning_rates: Optional[List[float]] = None,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> pd.DataFrame:
    """PRD4 Part 2: sweep learning_rate × {Rich, Poor} and record per-checkpoint geometry.

    Parameters
    ----------
    learning_rates : list of floats to evaluate.
                     Defaults to 30 log-spaced values from 0.0001 → 0.1.

    Outputs
    -------
    ``output_dir / learning_rate_sweep.csv``

    Returns
    -------
    DataFrame with one row per (model_type, learning_rate, epoch_checkpoint)
    """
    if learning_rates is None:
        learning_rates = np.logspace(np.log10(0.0001), np.log10(0.1), num=30).tolist()

    rows: List[Dict] = []
    for lr in learning_rates:
        for base_cfg in [RICH_CONFIG, POOR_CONFIG]:
            cfg = ModelConfig(
                name=base_cfg.name,
                hidden_dim=base_cfg.hidden_dim,
                init_scale=base_cfg.init_scale,
                learning_rate=lr,
                context_alpha=base_cfg.context_alpha,
                sigma=base_cfg.sigma,
            )
            print(f"      [lr_sweep] {cfg.name}  lr={lr:.4f} …")
            rows.extend(_sweep_run_one(cfg, context_encodings, fixed_sequence, vocab_size))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "learning_rate_sweep.csv", index=False)
    return df


def run_init_scale_sweep(
    context_encodings: Dict[str, torch.Tensor],
    output_dir: Path,
    init_scales: Optional[List[float]] = None,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> pd.DataFrame:
    """Sweep init_scale across a log-spaced grid for both Rich and Poor configs.

    Produces per-checkpoint tidy rows (same schema as hidden_size_sweep and
    learning_rate_sweep) with summary scalars broadcast onto every row.

    Parameters
    ----------
    init_scales : list of floats.  Defaults to 50 log-spaced values from 0.01 → 4.0.

    Outputs
    -------
    ``output_dir / init_scale_sweep.csv``

    Returns
    -------
    DataFrame with one row per (init_scale, epoch_checkpoint).
    """
    if init_scales is None:
        init_scales = np.logspace(np.log10(0.01), np.log10(4.0), num=50).tolist()

    rows: List[Dict] = []
    for scale in init_scales:
        # Use RICH_CONFIG's learning rate / hidden_dim as the neutral baseline
        cfg = ModelConfig(
            name=f"scale_{scale:.4f}",
            hidden_dim=RICH_CONFIG.hidden_dim,
            init_scale=float(scale),
            learning_rate=RICH_CONFIG.learning_rate,
            context_alpha=RICH_CONFIG.context_alpha,
            sigma=RICH_CONFIG.sigma,
        )
        print(f"      [init_scale_sweep]  init_scale={scale:.4f} …")
        rows.extend(_sweep_run_one(cfg, context_encodings, fixed_sequence, vocab_size))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "init_scale_sweep.csv", index=False)
    return df


def run_scale_lr_sweep(
    context_encodings: Dict[str, torch.Tensor],
    output_dir: Path,
    init_scales: Optional[List[float]] = None,
    learning_rates: Optional[List[float]] = None,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> pd.DataFrame:
    """Joint sweep of init_scale × learning_rate for both Rich and Poor configs.

    Uses the same value ranges as the solo sweeps:
      - init_scale    : 50 log-spaced values from 0.01 → 4.0
      - learning_rate : 30 log-spaced values from 0.0001 → 0.1

    Each combination is trained for ``_MAX_EPOCHS`` epochs in alternating
    context blocks of ``_BLOCK_SIZE``.  One tidy row is produced per epoch;
    geometry metrics and transition-cost jumps are populated at block-end rows
    (``sequence_in_block == _BLOCK_SIZE - 1``), enabling direct correlation
    (e.g. ``centroid_ctx_distance ~ tc_loss_jump``) across the full 2-D grid.

    Outputs
    -------
    ``output_dir / scale_lr_sweep.csv``

    Returns
    -------
    DataFrame with one row per (model_type, init_scale, learning_rate, epoch)
    """
    if init_scales is None:
        init_scales = np.logspace(np.log10(0.01), np.log10(4.0), num=50).tolist()
    if learning_rates is None:
        learning_rates = np.logspace(np.log10(0.0001), np.log10(0.1), num=30).tolist()

    rows: List[Dict] = []
    total = len(init_scales) * len(learning_rates)
    done  = 0
    for scale in init_scales:
        for lr in learning_rates:
            cfg = ModelConfig(
                name=f"scale_{scale:.4f}_lr_{lr:.5f}",
                hidden_dim=RICH_CONFIG.hidden_dim,
                init_scale=float(scale),
                learning_rate=float(lr),
                context_alpha=RICH_CONFIG.context_alpha,
                sigma=RICH_CONFIG.sigma,
            )
            done += 1
            print(
                f"      [scale_lr_sweep] {done}/{total}  "
                f"scale={scale:.4f}  lr={lr:.5f} …"
            )
            rows.extend(_sweep_run_one(cfg, context_encodings, fixed_sequence, vocab_size))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "scale_lr_sweep.csv", index=False)
    return df


# =============================================================================
# 4. Paper-ready summary metrics
# =============================================================================

def compute_model_summary_metrics(
    collapse_df: pd.DataFrame,
    perturbation_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config_meta: Dict,
) -> Dict:
    """Compute the compact inferential metric set from PRD4 Part 4.

    Parameters
    ----------
    collapse_df     : combined collapse_dynamics DataFrame
    perturbation_df : combined context_perturbation DataFrame
    results_df      : per-step learning DataFrame (contains ``loss``, ``accuracy``)
    config_meta     : dict with at minimum model_type, hidden_dim, learning_rate,
                      init_scale, participant_id
    """
    mt  = config_meta.get("model_type", "")
    pid = config_meta.get("participant_id", "")

    # Separation trajectory
    sub_col = pd.DataFrame()
    if not collapse_df.empty and "model_type" in collapse_df.columns:
        mask = collapse_df["model_type"] == mt
        if "participant_id" in collapse_df.columns:
            mask &= collapse_df["participant_id"] == pid
        sub_col = collapse_df[mask]

    x_col        = "step_or_epoch" if "step_or_epoch" in sub_col.columns else "checkpoint_value"
    sep_vals     = sub_col["trajectory_separation"].tolist()       if not sub_col.empty and "trajectory_separation"       in sub_col.columns else []
    ctx_dec_vals = sub_col["context_decodability"].tolist()        if not sub_col.empty and "context_decodability"        in sub_col.columns else []
    seq_dec_vals = sub_col["sequence_step_decodability"].tolist()  if not sub_col.empty and "sequence_step_decodability"  in sub_col.columns else []
    ep_vals      = sub_col[x_col].tolist()                         if not sub_col.empty and x_col in sub_col.columns      else []

    # Final performance
    sub_res = pd.DataFrame()
    if not results_df.empty and "model_type" in results_df.columns:
        mask_r = results_df["model_type"] == mt
        if "participant_id" in results_df.columns:
            mask_r &= results_df["participant_id"] == pid
        sub_res = results_df[mask_r]
    final_loss = float(sub_res["loss"].iloc[-1])     if not sub_res.empty and "loss"     in sub_res.columns else np.nan
    final_acc  = float(sub_res["accuracy"].iloc[-1]) if not sub_res.empty and "accuracy" in sub_res.columns else np.nan

    traj = _compute_trajectory_summary(sep_vals, ctx_dec_vals, seq_dec_vals,
                                       ep_vals, final_loss, final_acc)

    # Perturbation: early vs. late zero-context ablation cost
    early_ablation_cost = late_ablation_cost  = np.nan
    early_hidden_shift  = late_hidden_shift   = np.nan

    if not perturbation_df.empty and "model_type" in perturbation_df.columns:
        sub_pert = perturbation_df[perturbation_df["model_type"] == mt]
        pert_x   = "checkpoint_value" if "checkpoint_value" in sub_pert.columns else None
        if pert_x and not sub_pert.empty and "perturbation_mode" in sub_pert.columns:
            zero_pert   = sub_pert[sub_pert["perturbation_mode"] == "zero"]
            checkpoints = sorted(zero_pert[pert_x].unique())
            if len(checkpoints) >= 2:
                e_sub = zero_pert[zero_pert[pert_x] == checkpoints[0]]
                l_sub = zero_pert[zero_pert[pert_x] == checkpoints[-1]]
                early_ablation_cost = float(e_sub["delta_loss"].mean())        if "delta_loss"        in e_sub.columns else np.nan
                late_ablation_cost  = float(l_sub["delta_loss"].mean())        if "delta_loss"        in l_sub.columns else np.nan
                early_hidden_shift  = float(e_sub["hidden_state_shift"].mean()) if "hidden_state_shift" in e_sub.columns else np.nan
                late_hidden_shift   = float(l_sub["hidden_state_shift"].mean()) if "hidden_state_shift" in l_sub.columns else np.nan

    return {
        **traj,
        **config_meta,
        "early_context_ablation_cost": early_ablation_cost,
        "late_context_ablation_cost":  late_ablation_cost,
        "early_hidden_shift":          early_hidden_shift,
        "late_hidden_shift":           late_hidden_shift,
    }
