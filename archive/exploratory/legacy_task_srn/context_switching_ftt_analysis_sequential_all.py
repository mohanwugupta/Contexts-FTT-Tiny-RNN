#!/usr/bin/env python3
"""
Context-Specific Learning: Rich vs. Poor SRN Analysis
=====================================================

Thin entry-point script.  All shared model, training, metric and plotting
logic now lives in the ``srn`` package — import with ``from srn import *``.

This file retains only:
  • ``process_dataset()`` — per-dataset orchestration loop
  • ``main()``            — batch runner over all FTTData CSVs
"""

import copy
import warnings

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List

# ── srn package — all shared components ──────────────────────────────────────
from srn import (
    # config
    ModelConfig, VOCAB_SIZE, FIXED_SEQUENCE,
    RICH_CONFIG, POOR_CONFIG, PERF_THRESHOLDS, ACC_THRESHOLDS,
    # model
    ContextSequenceRNN, sequence_to_gaussian,
    # metrics
    run_mechanistic_analysis, compute_collapse_metrics, compute_weight_norms,
    run_context_perturbation_analysis, build_shared_pca,
    collect_checkpoint_hidden_states, compute_gradient_alignment,
    compute_output_divergence, run_context_ablation_test,
    # training
    run_participant_experiment, run_to_asymptote,
    # sweeps
    run_hidden_size_sweep, run_learning_rate_sweep, compute_model_summary_metrics,
    # plotting
    plot_pca_geometry, plot_mechanistic_summary,
    plot_rich_poor_comparison, plot_normalized_model_vs_human,
    plot_accuracy_entropy_curves, plot_asymptote_geometry,
    plot_weight_gradient_dynamics, plot_collapse_dynamics,
    plot_normalized_collapse, plot_phase_portrait,
    plot_dual_decoding_dynamics, plot_perturbation_dynamics,
    plot_shared_pca_panels, plot_collapse_driver_dynamics,
    plot_matched_geometry, _plot_matched_geometry_generic,
    plot_hidden_size_sweep, plot_learning_rate_sweep,
)

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    hidden_dim: int
    init_scale: float
    learning_rate: float
    context_alpha: float  # Strength of context signal (0.0 to 1.0)
    sigma: float = 0.7    # Gaussian input width

# Global Constants for Consistency across datasets
VOCAB_SIZE = 4
FIXED_SEQUENCE = [0, 1, 2, 0, 3, 1, 2, 3]

# Define the two regimes
RICH_CONFIG = ModelConfig(
    name="Rich",
    hidden_dim=6,       # Standard capacity
    init_scale=0.5,     # Stronger initial weights (High Gain)
    learning_rate=0.005, # Aggressive learning
    context_alpha=1.0   # Full context reliance
)

POOR_CONFIG = ModelConfig(
    name="Poor",
    hidden_dim=6,       # Standard capacity
    init_scale=0.02,    # Weak initial weights (Low Gain)
    learning_rate=0.005,
    context_alpha=1.0
)

# -----------------------------------------------------------------------------
# 2. Unified Model Architecture
# -----------------------------------------------------------------------------

def gaussian_vector(index: int, vocab_size: int, sigma: float) -> np.ndarray:
    """Generates a Gaussian-smooth one-hot vector."""
    x = np.arange(vocab_size)
    vec = np.exp(-0.5 * ((x - index) / sigma) ** 2)
    return vec / vec.sum()

def sequence_to_gaussian(seq: List[int], vocab_size: int, sigma: float) -> torch.Tensor:
    """Converts integer sequence to Gaussian input batch."""
    mat = np.stack([gaussian_vector(i, vocab_size, sigma) for i in seq], axis=0)
    return torch.tensor(mat, dtype=torch.float32).unsqueeze(0) # (1, seq_len, vocab)

# -----------------------------------------------------------------------------
# Prediction diagnostics — accuracy, entropy, confidence
# -----------------------------------------------------------------------------

def compute_sequence_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of time-steps where argmax(logits) == target.

    logits: (seq_len, vocab)  or (1, seq_len, vocab)
    targets: (seq_len,)
    """
    preds = logits.squeeze(0).argmax(dim=-1)
    return (preds == targets).float().mean().item()


def compute_prediction_entropy(logits: torch.Tensor) -> float:
    """Mean Shannon entropy of the softmax distribution over the sequence (nats).

    Tracks *output uncertainty*: high entropy = model still spreading probability
    mass; low entropy = confident (possibly overfit or well-learnt).
    logits: (seq_len, vocab) or (1, seq_len, vocab)
    """
    probs = torch.softmax(logits.squeeze(0), dim=-1)          # (seq_len, vocab)
    H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)        # (seq_len,)
    return H.mean().item()


def compute_prediction_confidence(logits: torch.Tensor) -> float:
    """Mean max-softmax probability over the sequence.

    Complement of entropy: rises toward 1.0 as the model becomes certain.
    Used as a confidence-based measure that separates 'is the model correct'
    (accuracy) from 'is the model sure' (confidence).
    logits: (seq_len, vocab) or (1, seq_len, vocab)
    """
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    return probs.max(dim=-1).values.mean().item()


def compute_correct_trial_surprisal(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean −log p(correct token) over correctly-predicted positions only.

    This is the human-RT analogue: conditioned on the model being right,
    how surprised was it?  A high value means the model was barely correct
    (low confidence even on its best guess), analogous to slow RTs on
    correct trials in behavioural data.
    Returns nan when no position is correct.
    logits: (seq_len, vocab)  or (1, seq_len, vocab)
    targets: (seq_len,)
    """
    logits_2d = logits.squeeze(0)                             # (seq_len, vocab)
    probs      = torch.softmax(logits_2d, dim=-1)
    preds      = logits_2d.argmax(dim=-1)
    correct    = (preds == targets)
    if correct.sum() == 0:
        return float('nan')
    p_correct_class = probs[torch.arange(len(targets)), targets]
    surprisal = -torch.log(p_correct_class + 1e-9)
    return surprisal[correct].mean().item()

class ContextSequenceRNN(nn.Module):
    def __init__(self, vocab_size: int, context_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input: Sequence (vocab) + Context (context_dim)
        input_dim = vocab_size + context_dim
        
        self.rnn = nn.RNN(input_dim, config.hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(config.hidden_dim, vocab_size)
        
        # Initialization Control
        self._initialize_weights(config.init_scale)

    def _initialize_weights(self, scale: float):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=scale)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        x: (batch, seq_len, vocab_size)
        context: (batch, context_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Scale context by alpha (Strength of context signal)
        # alpha=0 implies the model CANNOT see the context
        scaled_context = context * self.config.context_alpha
        
        context_expanded = scaled_context.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Concatenate Input + Context
        rnn_input = torch.cat([x, context_expanded], dim=-1)
        
        rnn_out, _ = self.rnn(rnn_input)
        logits = self.output_layer(rnn_out)
        return logits, rnn_out

# -----------------------------------------------------------------------------
# 3. Experiment Runner
# -----------------------------------------------------------------------------

def run_participant_experiment(df_participant: pd.DataFrame, 
                               config: ModelConfig, 
                               vocab_size: int = VOCAB_SIZE,
                               fixed_sequence: List[int] = FIXED_SEQUENCE,
                               single_context_mode: bool = False,
                               train_context: str = 'Maze',
                               context_encodings: Dict[str, torch.Tensor] = None,
                               context_col: str = 'group',
                               snapshot_loss_thresholds: Optional[List[float]] = None,
                               snapshot_accuracy_thresholds: Optional[List[float]] = None,
                               snapshot_step_checkpoints: Optional[List[int]] = None,
                               step_interval: Optional[int] = None,
                               ) -> Tuple[Dict, nn.Module, Dict, List[Dict]]:
    """Run one participant's trial sequence through the model.

    Snapshot modes (all optional, combinable):
    - snapshot_loss_thresholds     : capture first time CE ≤ threshold
    - snapshot_accuracy_thresholds : capture first time accuracy ≥ threshold
    - snapshot_step_checkpoints    : capture at specific global_step values
    - step_interval                : capture every N steps (dense checkpointing)

    All captured snapshots share the same ``snapshots`` dict keyed by strings:
    ``"ce_<thr>"``, ``"acc_<thr>"``, ``"step_<n>"``.  Each capture also appends
    a row to ``snapshot_metadata`` for downstream CSV export.

    Returns
    -------
    results           : per-step dict (loss, accuracy, entropy, confidence, …)
    model             : final trained model
    snapshots         : Dict[str, nn.Module]
    snapshot_metadata : List[Dict] — per-snapshot metadata rows
    """
    
    # Setup
    if context_encodings is None:
        context_encodings = {
            'Maze': torch.tensor([1., 0.]), 
            'Melody': torch.tensor([0., 1.])
        }
    
    # Determine context dimension from the vectors
    first_vec = next(iter(context_encodings.values()))
    context_dim = first_vec.shape[0]
    
    model = ContextSequenceRNN(vocab_size, context_dim=context_dim, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    results = {
        'block_id': [],          # The Trial Number (1-12)
        'sequence_in_block': [], # The index within the block (1-9)
        'global_step': [],       # Continuous index (1-108)
        'context': [],
        'loss': [],
        'transfer_loss': [],     # Loss on the OTHER context (counterfactual)
        'accuracy': [],          # Fraction of argmax correct (current context)
        'transfer_accuracy': [], # Fraction of argmax correct (transfer context)
        'entropy': [],           # Mean output entropy (prediction uncertainty)
        'confidence': [],        # Mean max-softmax prob (how sure is the model)
        'correct_surprisal': [], # −log p(correct) on correctly-predicted tokens (RT analogue)
        'rt': [],
        'training_active': [],   # [NEW] Log if training happened on this step
        'counterbalance': []     # Counterbalance group (1=context_A first, 2=context_B first)
    }

    # Snapshot tracking: capture model state the first time CE ≤ each threshold
    snapshots: Dict[str, nn.Module] = {}
    thresholds_remaining: set = set(snapshot_loss_thresholds or [])
    acc_thresholds_remaining: set = set(snapshot_accuracy_thresholds or [])
    step_checkpoints_remaining: set = set(snapshot_step_checkpoints or [])
    snapshot_metadata: List[Dict] = []
    
    # Training Loop
    # df_participant has one row per SEQUENCE (not per keypress)
    # Structure: 12 Blocks x 9 Sequences = ~108 rows
    global_step = 0
    
    for _, row in df_participant.iterrows():
        block_id = row['new_trialNumber']     # 1 to 12
        seq_num = row.get('sequenceNumber', 0) # 1 to 9 (usually)
        curr_context_name = row[context_col]
        rt = row['meanRT']
        cb = row.get('counterBalance', np.nan)
        global_step += 1
        
        # Handle case where context might not be in the map
        if curr_context_name not in context_encodings:
             curr_context_name = list(context_encodings.keys())[0]

        curr_context_vec = context_encodings[curr_context_name].unsqueeze(0)
        
        # Determine the "Transfer" context (the one we aren't seeing right now)
        all_ctx_names = list(context_encodings.keys())
        if len(all_ctx_names) > 1:
            others = [c for c in all_ctx_names if c != curr_context_name]
            other_context_name = others[0]
        else:
            other_context_name = curr_context_name
            
        other_context_vec = context_encodings[other_context_name].unsqueeze(0)
        
        # Determine if we should TRAIN on this trial (Control Condition Logic)
        train_active = True
        if single_context_mode and (curr_context_name != train_context):
            train_active = False

        # Prepare inputs (8 keypresses -> 7 predictions)
        inputs = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
        targets = torch.tensor(fixed_sequence[1:], dtype=torch.long)
        
        # 1. EVALUATION (Before Learning Update for this sequence)
        model.eval()
        with torch.no_grad():
            # Current Context Loss (Average over the 7 predictions)
            logits, _ = model(inputs, curr_context_vec)
            loss_val = criterion(logits.squeeze(0), targets).item()

            # Accuracy, entropy, confidence, correct-trial surprisal
            acc      = compute_sequence_accuracy(logits, targets)
            entropy  = compute_prediction_entropy(logits)
            conf     = compute_prediction_confidence(logits)
            c_surp   = compute_correct_trial_surprisal(logits, targets)

            # Transfer Probe: How well does it perform on the OTHER context?
            logits_tx, _ = model(inputs, other_context_vec)
            loss_tx  = criterion(logits_tx.squeeze(0), targets).item()
            acc_tx   = compute_sequence_accuracy(logits_tx, targets)

        # Snapshot: capture model the first time CE ≤ each threshold
        for thr in list(thresholds_remaining):
            if loss_val <= thr:
                key = f"ce_{thr}"
                snapshots[key] = copy.deepcopy(model)
                thresholds_remaining.discard(thr)
                snapshot_metadata.append({
                    'snapshot_type': 'ce', 'snapshot_value': thr,
                    'global_step': global_step, 'block_id': block_id,
                    'context': curr_context_name, 'loss': loss_val,
                    'accuracy': acc, 'confidence': conf,
                })

        # Snapshot: accuracy thresholds — first time accuracy ≥ threshold
        for thr in list(acc_thresholds_remaining):
            if acc >= thr:
                key = f"acc_{thr}"
                snapshots[key] = copy.deepcopy(model)
                acc_thresholds_remaining.discard(thr)
                snapshot_metadata.append({
                    'snapshot_type': 'accuracy', 'snapshot_value': thr,
                    'global_step': global_step, 'block_id': block_id,
                    'context': curr_context_name, 'loss': loss_val,
                    'accuracy': acc, 'confidence': conf,
                })

        # Snapshot: explicit step checkpoints
        if global_step in step_checkpoints_remaining:
            key = f"step_{global_step}"
            snapshots[key] = copy.deepcopy(model)
            step_checkpoints_remaining.discard(global_step)
            snapshot_metadata.append({
                'snapshot_type': 'step', 'snapshot_value': global_step,
                'global_step': global_step, 'block_id': block_id,
                'context': curr_context_name, 'loss': loss_val,
                'accuracy': acc, 'confidence': conf,
            })

        # Snapshot: dense step-interval checkpointing
        if step_interval and global_step % step_interval == 0:
            key = f"step_{global_step}"
            if key not in snapshots:
                snapshots[key] = copy.deepcopy(model)
            snapshot_metadata.append({
                'snapshot_type': 'interval', 'snapshot_value': global_step,
                'global_step': global_step, 'block_id': block_id,
                'context': curr_context_name, 'loss': loss_val,
                'accuracy': acc, 'confidence': conf,
            })

        # Record Data
        results['block_id'].append(block_id)
        results['sequence_in_block'].append(seq_num)
        results['global_step'].append(global_step)
        results['context'].append(curr_context_name)
        results['loss'].append(loss_val)
        results['transfer_loss'].append(loss_tx)
        results['accuracy'].append(acc)
        results['transfer_accuracy'].append(acc_tx)
        results['entropy'].append(entropy)
        results['confidence'].append(conf)
        results['correct_surprisal'].append(c_surp)
        results['rt'].append(rt)
        results['training_active'].append(train_active)
        results['counterbalance'].append(cb)
        
        # 2. TRAINING (Stepwise/Sequential through the 8 keypresses)
        if train_active:
            model.train()
            # Train on each transition in the sequence
            for t in range(len(fixed_sequence) - 1):
                token_idx = fixed_sequence[t]
                next_token_idx = fixed_sequence[t+1]
                
                # Single step input
                step_input_vec = gaussian_vector(token_idx, vocab_size, config.sigma)
                step_input = torch.tensor(step_input_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0) #(1,1,v)
                step_target = torch.tensor([next_token_idx], dtype=torch.long)
                
                # Forward pass with CURRENT context
                optimizer.zero_grad()
                step_logits, _ = model(step_input, curr_context_vec)
                loss = criterion(step_logits.squeeze(0), step_target)
                loss.backward()
                optimizer.step()
            
    return results, model, snapshots, snapshot_metadata

# -----------------------------------------------------------------------------
# 4. Asymptote Training & Performance-Matched Analysis
# -----------------------------------------------------------------------------

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
) -> Tuple[nn.Module, List[float], Dict]:
    """Continue training a model on the fixed sequence until CE converges.

    Interleaves all contexts every epoch (one full sequence pass per context).
    Stops when mean CE improves by < ``tol`` for ``patience`` consecutive epochs.

    Parameters
    ----------
    epoch_interval   : if set, deep-copy model every N epochs into extra_info['epoch_snapshots']
    track_gradients  : if True, log per-epoch gradient norms into extra_info['grad_history']

    Returns
    -------
    best_model   : deep-copy of the model at its lowest mean CE
    loss_history : list of per-epoch mean CE values
    extra_info   : dict with optional keys 'epoch_snapshots' and 'grad_history'
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion  = nn.CrossEntropyLoss()
    model.train()

    loss_history: List[float] = []
    best_loss        = float('inf')
    no_improve_count = 0
    best_model       = copy.deepcopy(model)

    epoch_snapshots: Dict[int, nn.Module] = {}
    accuracy_at_epoch: Dict[int, float]   = {}
    grad_history: List[Dict] = []

    # Pre-build full-sequence tensors once (seq_len = len-1, all predictions)
    # Training on the full sequence in one forward/backward pass is critical so
    # that W_hh receives non-zero gradients (single-step passes have h_0=0 →
    # grad(W_hh)=0 by construction).
    inputs_seq  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets_seq = torch.tensor(fixed_sequence[1:], dtype=torch.long)

    for epoch in range(max_epochs):
        epoch_loss, n = 0.0, 0
        grad_ctx_accum, grad_rec_accum, grad_out_accum = 0.0, 0.0, 0.0
        grad_ctx_steps, grad_rec_steps, grad_out_steps = 0, 0, 0

        for ctx_vec in context_encodings.values():
            optimizer.zero_grad()
            logits, _ = model(inputs_seq, ctx_vec.unsqueeze(0))
            loss = criterion(logits.squeeze(0), targets_seq)
            loss.backward()

            if track_gradients:
                # Gradient norms for context slice, recurrent weights, output weights.
                # Each weight appears exactly once per backward pass; use independent
                # counters so normalisation is correct even if any param is missing.
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if 'weight_ih' in name:
                            g = param.grad.detach()
                            grad_ctx_accum  += g[:, vocab_size:].norm().item()
                            grad_ctx_steps  += 1
                        elif 'weight_hh' in name:
                            grad_rec_accum  += param.grad.detach().norm().item()
                            grad_rec_steps  += 1
                        elif 'output_layer.weight' in name:
                            grad_out_accum  += param.grad.detach().norm().item()
                            grad_out_steps  += 1

            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / n
        loss_history.append(avg)

        if track_gradients and (grad_ctx_steps + grad_rec_steps + grad_out_steps) > 0:
            grad_history.append({
                'epoch': epoch,
                'loss': avg,
                'grad_context':   grad_ctx_accum / grad_ctx_steps if grad_ctx_steps > 0 else np.nan,
                'grad_recurrent': grad_rec_accum / grad_rec_steps if grad_rec_steps > 0 else np.nan,
                'grad_output':    grad_out_accum / grad_out_steps if grad_out_steps > 0 else np.nan,
            })

        if epoch_interval and (epoch + 1) % epoch_interval == 0:
            epoch_snapshots[epoch + 1] = copy.deepcopy(model)
            # Accuracy eval: full sequence forward pass across all contexts
            model.eval()
            acc_vals: List[float] = []
            with torch.no_grad():
                for ctx_vec in context_encodings.values():
                    logits_e, _ = model(inputs_seq, ctx_vec.unsqueeze(0))
                    acc_vals.append(compute_sequence_accuracy(logits_e, targets_seq))
            accuracy_at_epoch[epoch + 1] = float(np.mean(acc_vals))
            model.train()

        if best_loss - avg > tol:
            best_loss        = avg
            no_improve_count = 0
            best_model       = copy.deepcopy(model)
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    extra_info = {
        'epoch_snapshots':  epoch_snapshots,
        'grad_history':     grad_history,
        'accuracy_history': accuracy_at_epoch,
    }
    return best_model, loss_history, extra_info


# Performance-matched snapshot thresholds — shared across all analyses
PERF_THRESHOLDS = [1.3, 1.1, 0.9, 0.7, 0.5]


def run_matched_performance_analysis(
    df_all: pd.DataFrame,
    valid_pids: List[str],
    configs: List[ModelConfig],
    context_encodings: Dict[str, torch.Tensor],
    context_col: str,
    thresholds: List[float] = None,
    vocab_size: int = VOCAB_SIZE,
    n_participants: int = 5,
) -> pd.DataFrame:
    """Compare representational geometry at *matched* predictive performance.

    For each config × participant, trains the model following the participant
    timeline while capturing model snapshots each time CE crosses a threshold.
    Then runs the full mechanistic analysis (geometry) on each snapshot.

    The result DataFrame has one row per (config, participant, threshold) with
    geometry metrics — allowing comparisons like "Rich vs Poor when both have
    CE ≤ 1.0" rather than just "at end of training".

    Parameters
    ----------
    n_participants : how many participants to use (capped for speed; set to
                     len(valid_pids) for full sample)
    """
    if thresholds is None:
        thresholds = PERF_THRESHOLDS

    rows = []
    pids = valid_pids[:n_participants]

    for config in configs:
        if config.name == 'Control (Rich)':
            continue  # only compare Rich vs Poor

        print(f"   [matched] {config.name} — {len(pids)} participants × {len(thresholds)} thresholds")
        for pid in pids:
            p_data = df_all[df_all['ID'] == pid].sort_values(
                by=["new_trialNumber", "sequenceNumber"]
            )
            _, _, snaps, _meta = run_participant_experiment(
                p_data, config,
                context_encodings=context_encodings,
                context_col=context_col,
                snapshot_loss_thresholds=thresholds,
            )

            for snap_key, snap_model in snaps.items():
                if not snap_key.startswith('ce_'):
                    continue
                thr = float(snap_key[3:])
                mech = run_mechanistic_analysis(snap_model, config,
                                                context_encodings=context_encodings)
                row = {
                    'model_type':    config.name,
                    'participant_id': pid,
                    'ce_threshold':  thr,
                }
                # Keep only scalar metrics
                row.update({k: v for k, v in mech.items()
                             if not isinstance(v, np.ndarray)})
                rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# 5. Mechanistic Analysis (R4 & R5)
# -----------------------------------------------------------------------------

def compute_effective_dimensionality(hidden_states: np.ndarray) -> float:
    """
    Computes Effective Dimensionality (Participation Ratio).
    D_eff = (sum λ)^2 / sum(λ^2)
    """
    if not SKLEARN_AVAILABLE:
        return np.nan
        
    cov = np.cov(hidden_states.T)
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0
    
    # Filter small negatives due to numerical error
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    if len(eigenvalues) == 0:
        return 0.0
        
    sum_eig = np.sum(eigenvalues)
    sum_sq_eig = np.sum(eigenvalues**2)
    
    if sum_sq_eig == 0:
        return 0.0
        
    d_eff = (sum_eig**2) / sum_sq_eig
    return d_eff

def compute_jacobian_stats(model: ContextSequenceRNN, inputs: torch.Tensor, context: torch.Tensor) -> Dict:
    """
    Computes Jacobian spectral radius of the RNN dynamics.
    J_t = diag(1 - h_t^2) * W_hh
    """
    model.eval()
    with torch.no_grad():
        _, h_seq = model(inputs, context) # (1, seq, hidden)
    h_seq = h_seq.squeeze(0) # (seq, hidden)
    
    # Get Recurrent Weights (W_hh)
    # PyTorch RNN stores weight_hh_l0 of shape (hidden, hidden)
    W_hh = model.rnn.weight_hh_l0.detach() # (hidden, hidden)
    
    spectral_radii = []
    
    for t in range(h_seq.shape[0]):
        h_t = h_seq[t]
        # derivative of tanh(x) is 1 - tanh^2(x) = 1 - h_t^2
        # Jacobian J = diag(1 - h_t^2) @ W_hh
        diag_deriv = torch.diag(1 - h_t**2)
        J_t = torch.matmul(diag_deriv, W_hh)
        
        # Calculate eigenvalues
        try:
            # torch.linalg.eigvals returns complex values
            eigvals = torch.linalg.eigvals(J_t)
            # Spectral radius = max absolute eigenvalue
            spec_rad = torch.max(torch.abs(eigvals)).item()
            spectral_radii.append(spec_rad)
        except:
            pass
            
    return {
        'mean_spectral_radius': np.mean(spectral_radii) if spectral_radii else 0.0
    }

def run_mechanistic_analysis(model: ContextSequenceRNN, 
                             config: ModelConfig,
                             fixed_sequence: List[int] = FIXED_SEQUENCE, 
                             vocab_size: int = VOCAB_SIZE,
                             active_contexts: List[str] = None,
                             context_encodings: Dict[str, torch.Tensor] = None) -> Dict:
    """
    R4 & R5: Diagnoses HOW the model solved the task.
    1. Weight Attribution: Ratio of Context weights to Input weights.
    2. Context Decodability: Can we decode context from hidden states?
    3. Representational Geometry: PCA of hidden states.
    """
    if not SKLEARN_AVAILABLE:
        return {}
        
    results = {}
    
    # --- 1. Weight Attribution ---
    # RNN Input weights are (hidden_dim, input_dim) where input_dim = vocab + context
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            weights = param.detach().numpy() # Shape (Hidden, Vocab+Context)
            
            # Slice weights
            w_input = weights[:, :vocab_size]
            w_context = weights[:, vocab_size:]
            
            norm_in = np.linalg.norm(w_input)
            norm_cx = np.linalg.norm(w_context)
            
            # Compute Ratio (avoid div by zero)
            ratio = norm_cx / (norm_in + 1e-9)
            results['weight_ratio_cx_in'] = ratio
            results['norm_cx'] = norm_cx
            results['norm_in'] = norm_in
            break # Assuming single layer RNN
            
    # --- 2. Collect Hidden States for Both Contexts ---
    if context_encodings is None:
         context_encodings = {
            'Maze': torch.tensor([1., 0.]), 
            'Melody': torch.tensor([0., 1.])
        }
    
    inputs = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma) 
    # inputs shape: (1, seq_len, vocab)
    
    hidden_states = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            # Run model on the *entire sequence* to see trajectory
            _, h_seq = model(inputs, ctx_vec.unsqueeze(0))
            # h_seq shape: (1, seq_len, hidden)
            
            h_numpy = h_seq.squeeze(0).cpu().numpy()
            hidden_states.append(h_numpy)
            labels.extend([ctx_name] * h_numpy.shape[0])
            
    X = np.concatenate(hidden_states, axis=0) # (N * seq_len, hidden)
    y = np.array(labels)
    
    # --- 3. Context Decodability ---
    # Can a linear classifier predict Context (Maze vs Melody) from Hidden State?
    # Only if we have at least 2 classes
    if len(np.unique(y)) > 1:
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y) # Train on all (descriptive statistic of separability)
        acc = clf.score(X, y)
        results['context_decodability'] = acc
    else:
        results['context_decodability'] = np.nan

    # --- 3b. Sequence-Step Decodability ---
    # Can a linear classifier predict the position in the sequence (step 0–N-1)?
    # High value → hidden states encode temporal structure irrespective of context.
    # Complement of context_decodability: useful for dual decoding analysis (D6).
    seq_len_for_decode = hidden_states[0].shape[0]  # length of one context's trajectory
    step_labels = np.tile(np.arange(seq_len_for_decode), len(hidden_states))
    if len(np.unique(step_labels)) > 1:
        clf_step = LogisticRegression(random_state=42, max_iter=200)
        try:
            clf_step.fit(X, step_labels)
            results['sequence_step_decodability'] = clf_step.score(X, step_labels)
        except Exception:
            results['sequence_step_decodability'] = np.nan
    else:
        results['sequence_step_decodability'] = np.nan
    
    # --- 4. PCA Geometry ---
    pca = PCA(n_components=min(2, X.shape[0]))
    X_pca = pca.fit_transform(X)
    
    results['pca_variance_explained'] = pca.explained_variance_ratio_.sum()
    results['pca_coords'] = X_pca
    results['pca_labels'] = y
    
    # --- 5. Trajectory Separation (New Metric for Stats) ---
    # Only calculate if we have at least 2 contexts to compare
    unique_ctx_labels = np.unique(y)
    if len(unique_ctx_labels) >= 2:
        # Assuming equal-length trajectories: compare first two found contexts
        ctx_a = unique_ctx_labels[0]
        ctx_b = unique_ctx_labels[1]
        
        indices_a = np.where(y == ctx_a)[0]
        indices_b = np.where(y == ctx_b)[0]
        
        # Only compare if lengths match (simple case)
        if len(indices_a) == len(indices_b):
            coords_a = X_pca[indices_a]
            coords_b = X_pca[indices_b]
            separations = [np.linalg.norm(a - b) for a, b in zip(coords_a, coords_b)]
            results['trajectory_separation'] = np.mean(separations)
        else:
            results['trajectory_separation'] = np.nan
    else:
        results['trajectory_separation'] = np.nan
    
    # --- 6. Effective Dimensionality ---
    results['effective_dimensionality'] = compute_effective_dimensionality(X)

    # --- 7. Recurrent Dynamics (Jacobian) ---
    for ctx_name, ctx_vec in context_encodings.items():
        j_stats = compute_jacobian_stats(model, inputs, ctx_vec.unsqueeze(0))
        results[f'jacobian_spectral_radius_{ctx_name}'] = j_stats['mean_spectral_radius']

    return results


# -----------------------------------------------------------------------------
# 5b. Weight Norms, Collapse Metrics, Perturbation Analysis, Shared PCA
# -----------------------------------------------------------------------------

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
        if 'weight_ih' in name:
            result['norm_input_weights']   = float(np.linalg.norm(w[:, :vocab_size]))
            result['norm_context_weights'] = float(np.linalg.norm(w[:, vocab_size:]))
            result['context_input_ratio']  = result['norm_context_weights'] / (
                result['norm_input_weights'] + 1e-9)
        elif 'weight_hh' in name:
            result['norm_recurrent_weights'] = float(np.linalg.norm(w))
        elif 'output_layer.weight' in name:
            result['norm_output_weights'] = float(np.linalg.norm(w))
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

    - euclidean_ctx_distance   : mean element-wise Euclidean distance between
                                 matched hidden-state vectors across contexts
    - cosine_ctx_distance      : mean cosine distance across matched steps
    - centroid_ctx_distance    : distance between context-trajectory centroids
    - within_traj_distance     : mean step-to-step distance within each context
    - collapse_index           : euclidean_ctx_distance / within_traj_distance
                                 (> 1 ⟹ contexts more separated than they move)
    - context_decodability     : linear classifier accuracy (context labels)
    - sequence_step_decodability: linear classifier accuracy (step-index labels)
    - effective_dimensionality : participation ratio of hidden-state covariance
    - trajectory_separation    : mean PCA distance between context trajectories
    """
    if not SKLEARN_AVAILABLE:
        return {}

    inputs = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma)
    ctx_keys   = list(context_encodings.keys())
    ctx_hidden = {}   # ctx_name → (seq_len, hidden) numpy array

    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            _, h_seq = model(inputs, ctx_vec.unsqueeze(0))
            ctx_hidden[ctx_name] = h_seq.squeeze(0).cpu().numpy()

    # Only compute cross-context metrics when we have exactly 2 contexts
    metrics: Dict = {}
    if len(ctx_keys) >= 2:
        ha = ctx_hidden[ctx_keys[0]]   # (seq_len, hidden)
        hb = ctx_hidden[ctx_keys[1]]

        # Euclidean distance (mean over matched time-steps)
        diffs      = ha - hb
        euc_dists  = np.linalg.norm(diffs, axis=1)
        metrics['euclidean_ctx_distance'] = float(np.mean(euc_dists))

        # Cosine distance (1 − cosine similarity)
        cos_sims = []
        for a, b in zip(ha, hb):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-9 and nb > 1e-9:
                cos_sims.append(1.0 - float(np.dot(a, b) / (na * nb)))
        metrics['cosine_ctx_distance']   = float(np.mean(cos_sims)) if cos_sims else np.nan

        # Centroid distance
        ca = ha.mean(axis=0)
        cb = hb.mean(axis=0)
        metrics['centroid_ctx_distance'] = float(np.linalg.norm(ca - cb))
    else:
        metrics.update(euclidean_ctx_distance=np.nan,
                       cosine_ctx_distance=np.nan,
                       centroid_ctx_distance=np.nan)

    # Within-trajectory distance (mean step-to-step norm across all contexts)
    within_dists = []
    for h in ctx_hidden.values():
        for t in range(len(h) - 1):
            within_dists.append(np.linalg.norm(h[t + 1] - h[t]))
    within = float(np.mean(within_dists)) if within_dists else 1e-9
    metrics['within_traj_distance'] = within
    metrics['collapse_index'] = (
        metrics.get('euclidean_ctx_distance', np.nan) / max(within, 1e-9)
    )

    # Build stacked hidden-state matrix for classifier + PCA
    all_h  = np.concatenate(list(ctx_hidden.values()), axis=0)
    ctx_labels  = np.concatenate([
        np.array([k] * v.shape[0]) for k, v in ctx_hidden.items()
    ])
    seq_len_per_ctx = list(ctx_hidden.values())[0].shape[0]
    step_labels = np.tile(np.arange(seq_len_per_ctx), len(ctx_hidden))

    # Context decodability
    if len(np.unique(ctx_labels)) > 1:
        clf_ctx = LogisticRegression(random_state=42, max_iter=200)
        clf_ctx.fit(all_h, ctx_labels)
        metrics['context_decodability'] = float(clf_ctx.score(all_h, ctx_labels))
    else:
        metrics['context_decodability'] = np.nan

    # Sequence-step decodability
    if len(np.unique(step_labels)) > 1:
        clf_step = LogisticRegression(random_state=42, max_iter=200)
        try:
            clf_step.fit(all_h, step_labels)
            metrics['sequence_step_decodability'] = float(clf_step.score(all_h, step_labels))
        except Exception:
            metrics['sequence_step_decodability'] = np.nan
    else:
        metrics['sequence_step_decodability'] = np.nan

    # Effective dimensionality
    metrics['effective_dimensionality'] = compute_effective_dimensionality(all_h)

    # PCA trajectory separation
    pca_n = min(2, all_h.shape[0], all_h.shape[1])
    if pca_n >= 2 and len(ctx_keys) >= 2:
        pca = PCA(n_components=pca_n)
        X_pca = pca.fit_transform(all_h)
        n_per = seq_len_per_ctx
        sep = np.mean([
            np.linalg.norm(X_pca[i] - X_pca[i + n_per])
            for i in range(n_per)
        ])
        metrics['trajectory_separation'] = float(sep)
    else:
        metrics['trajectory_separation'] = np.nan

    return metrics


def run_context_perturbation_analysis(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
    modes: List[str] = None,
) -> List[Dict]:
    """Causal perturbation test: how much does disrupting the context signal hurt?

    For each context and each perturbation mode, computes:
    - baseline_loss / baseline_accuracy with the correct context
    - perturbed_loss / perturbed_accuracy with the disrupted context
    - delta_loss = perturbed − baseline (positive ⟹ perturbation hurts)
    - delta_accuracy = perturbed − baseline (negative ⟹ perturbation hurts)
    - hidden_state_shift: mean L2 distance between baseline and perturbed hidden states

    Modes
    -----
    "zero"    : replace context with the zero vector
    "shuffle" : assign a randomly chosen *other* context vector
    "swap"    : for the first two contexts, swap their vectors
    """
    if modes is None:
        modes = ["zero", "shuffle", "swap"]

    criterion = nn.CrossEntropyLoss()
    # inputs uses the full sequence (seq_len steps); the model produces one logit
    # per input step, but targets only covers the *next-token* predictions (seq_len-1).
    # We therefore use inputs[:,:-1,:] so the output has the same length as targets.
    inputs_full = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma)
    inputs      = inputs_full[:, :-1, :]          # (1, seq_len-1, vocab)
    targets     = torch.tensor(fixed_sequence[1:], dtype=torch.long)  # (seq_len-1,)

    ctx_keys = list(context_encodings.keys())
    results  = []

    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            ctx_batch = ctx_vec.unsqueeze(0)

            # Baseline
            logits_base, h_base = model(inputs, ctx_batch)
            loss_base = criterion(logits_base.squeeze(0), targets).item()
            acc_base  = compute_sequence_accuracy(logits_base, targets)
            h_base_np = h_base.squeeze(0).cpu().numpy()

            for mode in modes:
                # Build perturbed context
                if mode == "zero":
                    pert_vec = torch.zeros_like(ctx_vec)
                elif mode == "shuffle":
                    others = [k for k in ctx_keys if k != ctx_name]
                    if others:
                        pert_vec = context_encodings[others[0]]
                    else:
                        pert_vec = ctx_vec
                elif mode == "swap":
                    # swap first two contexts; fallback to zero if only one context
                    if len(ctx_keys) >= 2:
                        swap_map = {ctx_keys[0]: ctx_keys[1], ctx_keys[1]: ctx_keys[0]}
                        pert_name = swap_map.get(ctx_name, ctx_name)
                        pert_vec  = context_encodings[pert_name]
                    else:
                        pert_vec = torch.zeros_like(ctx_vec)
                else:
                    continue

                pert_batch = pert_vec.unsqueeze(0)
                logits_pert, h_pert = model(inputs, pert_batch)
                loss_pert = criterion(logits_pert.squeeze(0), targets).item()
                acc_pert  = compute_sequence_accuracy(logits_pert, targets)
                h_pert_np = h_pert.squeeze(0).cpu().numpy()

                hidden_shift = float(np.mean(
                    np.linalg.norm(h_pert_np - h_base_np, axis=1)
                ))

                results.append({
                    'context':             ctx_name,
                    'perturbation_mode':   mode,
                    'baseline_loss':       loss_base,
                    'perturbed_loss':      loss_pert,
                    'delta_loss':          loss_pert - loss_base,
                    'baseline_accuracy':   acc_base,
                    'perturbed_accuracy':  acc_pert,
                    'delta_accuracy':      acc_pert - acc_base,
                    'hidden_state_shift':  hidden_shift,
                })

    return results


def build_shared_pca(
    checkpoint_hidden_states: Dict[str, np.ndarray],
    n_components: int = 2,
):
    """Fit a single PCA on the pooled hidden states from all checkpoints.

    Parameters
    ----------
    checkpoint_hidden_states : dict mapping checkpoint_key → (N, hidden) array
    n_components             : number of PCA dimensions to retain

    Returns
    -------
    pca : fitted sklearn PCA object
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
    """Return stacked hidden states (all contexts) for a single model checkpoint."""
    inputs = sequence_to_gaussian(fixed_sequence, vocab_size, sigma=config.sigma)
    all_h  = []
    model.eval()
    with torch.no_grad():
        for ctx_vec in context_encodings.values():
            _, h_seq = model(inputs, ctx_vec.unsqueeze(0))
            all_h.append(h_seq.squeeze(0).cpu().numpy())
    return np.concatenate(all_h, axis=0)


# ─── PRD4 Part 3: Collapse-driver helpers ─────────────────────────────────────

def compute_gradient_alignment(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> float:
    """Cross-context cosine similarity of the full gradient vector (Extension A).

    Computes the gradient of CE loss w.r.t. all parameters for Context A and
    Context B on the fixed sequence, then returns their cosine similarity.

    Positive → both contexts exert the same learning pressure (collapse-driving).
    Negative → conflicting gradients (context-splitting pressure).
    Returns nan when fewer than 2 contexts are available or gradients are zero.
    """
    ctx_keys = list(context_encodings.keys())
    if len(ctx_keys) < 2:
        return float('nan')

    inputs    = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets   = torch.tensor(fixed_sequence[1:], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    grad_vecs: List[torch.Tensor] = []
    was_training = model.training
    model.train()

    for ctx_name in ctx_keys[:2]:
        ctx_vec = context_encodings[ctx_name].unsqueeze(0)
        model.zero_grad()
        logits, _ = model(inputs, ctx_vec)
        loss = criterion(logits.squeeze(0), targets)
        loss.backward()
        g = torch.cat([
            p.grad.detach().flatten()
            for p in model.parameters()
            if p.grad is not None
        ])
        grad_vecs.append(g)

    model.zero_grad()
    model.train(was_training)

    if len(grad_vecs) < 2:
        return float('nan')
    na, nb = grad_vecs[0].norm(), grad_vecs[1].norm()
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return float(torch.dot(grad_vecs[0], grad_vecs[1]) / (na * nb))


def compute_output_divergence(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> Dict:
    """Jensen-Shannon divergence between output distributions across contexts (Extension B).

    Measures how redundant the two context-conditioned output distributions are.
    As the model collapses, JS divergence should fall toward 0.

    Returns
    -------
    dict with keys:
      output_js_divergence    : mean JS divergence (nats) across sequence positions
      correct_token_prob_gap  : mean |p(correct | ctx_A) − p(correct | ctx_B)|
    """
    ctx_keys = list(context_encodings.keys())
    if len(ctx_keys) < 2:
        return {'output_js_divergence': np.nan, 'correct_token_prob_gap': np.nan}

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

    tgt_np      = targets.numpy()
    p_corr_a    = pa[np.arange(len(tgt_np)), tgt_np]
    p_corr_b    = pb[np.arange(len(tgt_np)), tgt_np]
    prob_gap    = float(np.mean(np.abs(p_corr_a - p_corr_b)))

    return {'output_js_divergence': js, 'correct_token_prob_gap': prob_gap}


def run_context_ablation_test(
    model: nn.Module,
    config: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> List[Dict]:
    """Recurrent-only competence test: run with zeroed / mean context (Extension C).

    For each original context, compares performance under:
    - ``zero`` : context vector replaced with zeros  (pure recurrent dynamics)
    - ``mean`` : context vector replaced with the mean of all context vectors

    Returns one row per (context_name × condition).  Each row also includes
    ``delta_loss`` and ``delta_accuracy`` relative to the true-context baseline,
    and ``hidden_state_shift`` (mean L2 distance of hidden trajectories).
    """
    criterion = nn.CrossEntropyLoss()
    inputs    = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
    targets   = torch.tensor(fixed_sequence[1:], dtype=torch.long)

    ctx_vecs  = list(context_encodings.values())
    mean_ctx  = torch.stack(ctx_vecs).mean(dim=0)
    zero_ctx  = torch.zeros_like(mean_ctx)

    results: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for ctx_name, ctx_vec in context_encodings.items():
            logits_true, h_true = model(inputs, ctx_vec.unsqueeze(0))
            loss_true = criterion(logits_true.squeeze(0), targets).item()
            acc_true  = compute_sequence_accuracy(logits_true, targets)
            h_true_np = h_true.squeeze(0).cpu().numpy()

            for condition, pert_vec in [('zero', zero_ctx), ('mean', mean_ctx)]:
                logits_p, h_p  = model(inputs, pert_vec.unsqueeze(0))
                loss_p  = criterion(logits_p.squeeze(0), targets).item()
                acc_p   = compute_sequence_accuracy(logits_p, targets)
                h_p_np  = h_p.squeeze(0).cpu().numpy()
                shift   = float(np.mean(np.linalg.norm(h_p_np - h_true_np, axis=1)))
                results.append({
                    'context':           ctx_name,
                    'context_condition': condition,
                    'loss':              loss_p,
                    'accuracy':          acc_p,
                    'delta_loss':        loss_p - loss_true,
                    'delta_accuracy':    acc_p  - acc_true,
                    'hidden_state_shift': shift,
                })
    return results


def plot_mechanistic_summary(results_list: List[Dict], output_dir: Path):
    """Aggregates mechanistic metrics across participants."""
    if not results_list:
        return

    df = pd.DataFrame(results_list)
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Decodability vs Model Type
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x='model_type', y='context_decodability', 
                palette={'Rich': 'darkgreen', 'Poor': 'firebrick', 'Control (Rich)': 'royalblue'}, ax=ax, errorbar='se')
    ax.set_title("Context Decodability (Hidden States)", fontsize=14)
    ax.set_ylabel("Classification Accuracy (Chance=0.5)")
    ax.set_ylim(0.0, 1.1)
    ax.axhline(0.5, linestyle='--', color='gray')
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / "mech_decodability.png")
    plt.close(fig)
    
    # 2. Weight Ratio
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df, x='model_type', y='weight_ratio_cx_in', 
                palette={'Rich': 'darkgreen', 'Poor': 'firebrick', 'Control (Rich)': 'royalblue'}, ax=ax, errorbar='se')
    ax.set_title("Weight Norm Ratio (Context / Input)", fontsize=14)
    ax.set_ylabel("||W_context|| / ||W_input||")
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / "mech_weight_ratio.png")
    plt.close(fig)

    # 3. Effective Dimensionality
    if 'effective_dimensionality' in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(data=df, x='model_type', y='effective_dimensionality', 
                    palette={'Rich': 'darkgreen', 'Poor': 'firebrick', 'Control (Rich)': 'royalblue'}, ax=ax, errorbar='se')
        ax.set_title("Effective Dimensionality (Participation Ratio)", fontsize=14)
        ax.set_ylabel("Dimensionality")
        sns.despine()
        plt.tight_layout()
        fig.savefig(output_dir / "mech_eff_dim.png")
        plt.close(fig)

    # 4. Jacobian Recurrent Dynamics
    jac_cols = [c for c in df.columns if 'jacobian_spectral_radius' in c]
    if jac_cols:
        df_jac = df.melt(id_vars=['model_type'], value_vars=jac_cols, var_name='Context', value_name='Spectral_Radius')
        df_jac['Context'] = df_jac['Context'].str.replace('jacobian_spectral_radius_', '')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=df_jac, x='model_type', y='Spectral_Radius', hue='Context',
                    palette='viridis', ax=ax, errorbar='se')
        ax.set_title("Recurrent Dynamics Stability (Jacobian)", fontsize=14)
        ax.set_ylabel("Spectral Radius (Max Eigenvalue)")
        ax.axhline(1.0, linestyle='--', color='gray', alpha=0.5, label='Chaotic Edge')
        sns.despine()
        plt.tight_layout()
        fig.savefig(output_dir / "mech_jacobian_dynamics.png")
        plt.close(fig)

def plot_pca_geometry(mech_stats: Dict, model_name: str, output_dir: Path):
    """Plots the 2D PCA projection of hidden states for a single model instance."""
    X_pca = mech_stats['pca_coords']
    y = mech_stats['pca_labels']
    
    # Create DataFrame for plotting
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Context'] = y
    
    # Add sequence step 
    unique_ctx = np.unique(y)
    num_ctx = len(unique_ctx)
    if num_ctx > 0:
        seq_len = len(y) // num_ctx
        seq_steps = np.tile(np.arange(seq_len), num_ctx)
    else:
        seq_steps = []
        
    df_pca['Step'] = seq_steps
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check if we have at least 2 categories for hue/style
    if num_ctx > 1:
        # Create a dynamic palette
        palette = sns.color_palette("husl", num_ctx)
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Context', style='Context', 
                        palette=palette, s=100, ax=ax)
    else:
        # Single color if only 1 context
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', color='purple', s=100, ax=ax)

    # connect points with lines
    # Only iterate over contexts actually present
    for i, ctx in enumerate(unique_ctx):
        color = 'purple' if num_ctx == 1 else sns.color_palette("husl", num_ctx)[i]
        subset = df_pca[df_pca['Context'] == ctx].sort_values('Step')
        ax.plot(subset['PC1'], subset['PC2'], color=color, alpha=0.5)
        
    ax.set_title(f"Hidden State Geometry ({model_name})", fontsize=16)
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / f"mech_pca_{model_name}.png")
    plt.close(fig)

# -----------------------------------------------------------------------------
# 5. Main Execution & Analysis
# -----------------------------------------------------------------------------

# ── New plotting helpers (PRD3 deliverables) ──────────────────────────────────

def plot_collapse_dynamics(collapse_df: pd.DataFrame, output_dir: Path):
    """Figure D — collapse metrics over training steps/epochs.

    Four panels: trajectory_separation, context_decodability,
    sequence_step_decodability, effective_dimensionality.
    Lines split by model_type (Rich vs Poor) and training_phase.
    """
    if collapse_df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}

    metrics = [
        ('trajectory_separation',      'Trajectory Separation'),
        ('context_decodability',        'Context Decodability'),
        ('sequence_step_decodability',  'Seq-Step Decodability'),
        ('effective_dimensionality',    'Effective Dimensionality'),
        ('collapse_index',              'Collapse Index'),
        ('loss',                        'CE Loss'),
    ]
    avail = [(c, l) for c, l in metrics if c in collapse_df.columns]
    n = len(avail)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    x_col = 'step_or_epoch' if 'step_or_epoch' in collapse_df.columns else 'checkpoint_value'
    for ax_i, (col, label) in enumerate(avail):
        ax = axes[ax_i]
        plot_df = collapse_df[collapse_df[col].notna()]
        for mt, grp in plot_df.groupby('model_type'):
            color = pal.get(mt, 'steelblue')
            agg = grp.groupby(x_col)[col].mean().reset_index()
            ax.plot(agg[x_col], agg[col], label=mt, color=color, linewidth=1.5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Step / Epoch')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    for ax_i in range(len(avail), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle('Collapse Dynamics Over Training', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'collapse_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_normalized_collapse(collapse_df: pd.DataFrame, output_dir: Path):
    """Figure E — trajectory separation normalised per participant, aligned to peak."""
    if collapse_df.empty or 'trajectory_separation' not in collapse_df.columns:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}

    x_col = 'step_or_epoch' if 'step_or_epoch' in collapse_df.columns else 'checkpoint_value'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mt in zip(axes, ['Rich', 'Poor']):
        sub = collapse_df[collapse_df['model_type'] == mt].copy()
        if sub.empty:
            ax.set_title(f'{mt} (no data)')
            continue
        groups = sub.groupby('participant_id') if 'participant_id' in sub.columns else [(mt, sub)]
        for pid, grp in groups:
            s = grp.sort_values(x_col)['trajectory_separation'].values
            if s.max() < 1e-9:
                continue
            s_norm = s / s.max()
            peak_idx = np.argmax(s)
            aligned  = s_norm - s_norm[peak_idx]   # shift so peak = 0 (time 0)
            xvals    = np.arange(len(aligned)) - peak_idx
            ax.plot(xvals, aligned, alpha=0.4, color=pal.get(mt, 'steelblue'), linewidth=0.8)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{mt} — Normalised Separation (aligned to peak)', fontsize=11)
        ax.set_xlabel('Steps relative to peak')
        ax.set_ylabel('Normalised trajectory separation')

    plt.tight_layout()
    fig.savefig(output_dir / 'collapse_normalised.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_phase_portrait(collapse_df: pd.DataFrame, output_dir: Path):
    """Figure F — phase portrait: loss (x) vs trajectory separation (y)."""
    if collapse_df.empty:
        return
    needed = {'loss', 'trajectory_separation'}
    if not needed.issubset(collapse_df.columns):
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}
    fig, ax = plt.subplots(figsize=(7, 6))

    for mt, grp in collapse_df.groupby('model_type'):
        ax.scatter(
            grp['loss'], grp['trajectory_separation'],
            label=mt, color=pal.get(mt, 'steelblue'),
            alpha=0.4, s=15
        )
        # Average trend
        agg = grp.dropna(subset=['loss', 'trajectory_separation'])
        if len(agg) > 3:
            bins   = pd.cut(agg['loss'], bins=10)
            trend  = agg.groupby(bins)['trajectory_separation'].mean()
            mids   = [b.mid for b in trend.index]
            ax.plot(mids, trend.values, color=pal.get(mt, 'steelblue'), linewidth=2)

    ax.set_xlabel('CE Loss')
    ax.set_ylabel('Trajectory Separation')
    ax.set_title('Phase Portrait: Loss vs Context Separation', fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / 'collapse_phase_portrait.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_perturbation_dynamics(
    perturbation_df: pd.DataFrame, output_dir: Path
):
    """Plot delta_loss and hidden_state_shift per perturbation mode, over checkpoints."""
    if perturbation_df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')

    for metric, ylabel in [
        ('delta_loss',          'Δ CE Loss (perturbed − baseline)'),
        ('hidden_state_shift',  'Mean Hidden-State Shift (L2)'),
    ]:
        if metric not in perturbation_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        x_col = 'checkpoint_value' if 'checkpoint_value' in perturbation_df.columns else 'global_step'
        if x_col not in perturbation_df.columns:
            plt.close(fig)
            continue
        for (mt, mode), grp in perturbation_df.groupby(['model_type', 'perturbation_mode']):
            agg = grp.groupby(x_col)[metric].mean().reset_index()
            label = f"{mt} / {mode}"
            ax.plot(agg[x_col], agg[metric], label=label, linewidth=1.5)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Perturbation Analysis — {ylabel}', fontsize=12)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = 'context_perturbation_delta_loss.png' if metric == 'delta_loss' \
            else 'context_perturbation_hidden_shift.png'
        fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_weight_gradient_dynamics(wg_df: pd.DataFrame, output_dir: Path):
    """Plot weight norms and gradient norms over asymptote training epochs."""
    if wg_df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}

    weight_metrics = [
        ('norm_context_weights',   '||W_context||'),
        ('norm_input_weights',     '||W_input||'),
        ('norm_recurrent_weights', '||W_recurrent||'),
        ('context_input_ratio',    'Context / Input Ratio'),
        ('norm_output_weights',    '||W_output||'),
    ]
    grad_metrics = [
        ('grad_context',   '∇ context weights'),
        ('grad_recurrent', '∇ recurrent weights'),
        ('grad_output',    '∇ output weights'),
    ]

    all_panels = weight_metrics + grad_metrics
    avail = [(c, l) for c, l in all_panels if c in wg_df.columns]
    if not avail:
        return

    ncols = 4
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    x_col = 'epoch'

    for ax_i, (col, label) in enumerate(avail):
        ax = axes[ax_i]
        for mt, grp in wg_df.groupby('model_type'):
            agg = grp.groupby(x_col)[col].mean().reset_index()
            ax.plot(agg[x_col], agg[col], label=mt,
                    color=pal.get(mt, 'steelblue'), linewidth=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    for ax_i in range(len(avail), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle('Weight & Gradient Dynamics (Asymptote Training)', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / 'weight_gradient_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_shared_pca_panels(shared_pca_df: pd.DataFrame, output_dir: Path):
    """Plot early / mid / late / asymptote hidden-state trajectories on shared PCA axes."""
    if shared_pca_df.empty or 'PC1' not in shared_pca_df.columns:
        return
    plt.style.use('seaborn-v0_8-whitegrid')

    model_types = shared_pca_df['model_type'].unique()
    for mt in model_types:
        sub = shared_pca_df[shared_pca_df['model_type'] == mt]
        checkpoints = sorted(sub['checkpoint_value'].unique())
        n_panels = min(len(checkpoints), 4)
        # Pick evenly spaced checkpoints + last one
        idxs = np.linspace(0, len(checkpoints) - 1, n_panels, dtype=int)
        chosen = [checkpoints[i] for i in idxs]

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
        axes = axes[0]
        ctx_list = sorted(sub['context'].unique())
        colors   = sns.color_palette('husl', len(ctx_list))

        for ax, ckpt in zip(axes, chosen):
            cp_sub = sub[sub['checkpoint_value'] == ckpt]
            for ci, ctx in enumerate(ctx_list):
                c_sub = cp_sub[cp_sub['context'] == ctx].sort_values('step_in_sequence')
                ax.plot(c_sub['PC1'], c_sub['PC2'], color=colors[ci],
                        label=ctx, linewidth=1.5, marker='o', markersize=4)
            ax.set_title(f'Checkpoint {ckpt}', fontsize=10)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=7)

        fig.suptitle(f'Shared PCA — {mt}', fontsize=13)
        plt.tight_layout()
        safe = mt.replace(' ', '_').replace('/', '_')
        fig.savefig(output_dir / f'shared_pca_{safe}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_dual_decoding_dynamics(collapse_df: pd.DataFrame, output_dir: Path):
    """Figure dual — context decodability vs sequence-step decodability over training."""
    needed = {'context_decodability', 'sequence_step_decodability'}
    if collapse_df.empty or not needed.issubset(collapse_df.columns):
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}

    x_col = 'step_or_epoch' if 'step_or_epoch' in collapse_df.columns else 'checkpoint_value'
    model_types = collapse_df['model_type'].unique()
    n = len(model_types)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for ax, mt in zip(axes[0], model_types):
        sub = collapse_df[collapse_df['model_type'] == mt]
        agg = sub.groupby(x_col)[
            ['context_decodability', 'sequence_step_decodability']
        ].mean().reset_index()
        color = pal.get(mt, 'steelblue')
        ax.plot(agg[x_col], agg['context_decodability'],
                label='Context decodability', color=color, linewidth=1.8)
        ax.plot(agg[x_col], agg['sequence_step_decodability'],
                label='Seq-step decodability', color=color,
                linewidth=1.8, linestyle='--')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6)
        ax.set_title(f'{mt}', fontsize=12)
        ax.set_xlabel('Step / Epoch')
        ax.set_ylabel('Decoding Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    fig.suptitle('Context vs Sequence-Step Decodability Over Training', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / 'dual_decoding_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ─── PRD4 Part 1 & 2: Robustness sweeps ──────────────────────────────────────

def _compute_trajectory_summary(
    sep_vals: List[float],
    ctx_dec_vals: List[float],
    seq_dec_vals: List[float],
    epoch_vals: List[int],
    final_loss: float,
    final_acc: float,
) -> Dict:
    """Summarise a separation trajectory into the compact scalar metrics from PRD4.

    Parameters
    ----------
    sep_vals      : trajectory_separation at each checkpoint
    ctx_dec_vals  : context_decodability at each checkpoint
    seq_dec_vals  : sequence_step_decodability at each checkpoint
    epoch_vals    : checkpoint indices (epochs or steps)
    final_loss    : CE loss at convergence
    final_acc     : accuracy at convergence
    """
    sep     = np.array(sep_vals,      dtype=float)
    ctx_dec = np.array(ctx_dec_vals,  dtype=float)
    seq_dec = np.array(seq_dec_vals,  dtype=float)
    epochs  = np.array(epoch_vals,    dtype=float)

    _nan_dict: Dict = {
        'peak_separation': np.nan, 'step_to_peak': np.nan,
        'final_separation': np.nan, 'auc_separation': np.nan,
        'compression_amount': np.nan, 'compression_ratio': np.nan,
        'peak_context_decodability': np.nan, 'final_context_decodability': np.nan,
        'peak_sequence_decodability': np.nan, 'final_sequence_decodability': np.nan,
        'final_loss': final_loss, 'final_accuracy': final_acc,
    }
    if len(sep) == 0 or np.all(np.isnan(sep)):
        return _nan_dict

    valid = ~np.isnan(sep)
    if not valid.any():
        return _nan_dict

    peak_idx     = int(np.nanargmax(sep))
    peak_sep     = float(sep[peak_idx])
    step_to_peak = int(epochs[peak_idx])
    final_sep    = float(sep[valid][-1])
    auc          = float(np.trapz(sep[valid], epochs[valid])) if valid.sum() > 1 else np.nan

    ctx_valid  = ctx_dec[valid]
    seq_valid  = seq_dec[valid]

    return {
        'peak_separation':            peak_sep,
        'step_to_peak':               step_to_peak,
        'final_separation':           final_sep,
        'auc_separation':             auc,
        'compression_amount':         peak_sep - final_sep,
        'compression_ratio':          final_sep / peak_sep if peak_sep > 1e-9 else np.nan,
        'peak_context_decodability':  float(np.nanmax(ctx_dec))  if len(ctx_valid) else np.nan,
        'final_context_decodability': float(ctx_valid[-1])        if len(ctx_valid) else np.nan,
        'peak_sequence_decodability': float(np.nanmax(seq_dec))  if len(seq_valid) else np.nan,
        'final_sequence_decodability': float(seq_valid[-1])       if len(seq_valid) else np.nan,
        'final_loss':                 final_loss,
        'final_accuracy':             final_acc,
    }


def _sweep_run_one(
    cfg: ModelConfig,
    context_encodings: Dict[str, torch.Tensor],
    fixed_sequence: List[int],
    vocab_size: int,
) -> Dict:
    """Train one model config to asymptote and extract trajectory summary metrics.

    Shared logic reused by both `run_hidden_size_sweep` and `run_learning_rate_sweep`.
    """
    context_dim = next(iter(context_encodings.values())).shape[0]
    model       = ContextSequenceRNN(vocab_size, context_dim=context_dim, config=cfg)
    best_model, loss_hist, extra = run_to_asymptote(
        model, cfg, fixed_sequence, vocab_size, context_encodings,
        epoch_interval=10, track_gradients=False,
    )

    epoch_snaps = extra.get('epoch_snapshots', {})
    sep_vals, ctx_dec_vals, seq_dec_vals, ep_vals = [], [], [], []
    for ep in sorted(epoch_snaps.keys()):
        col_m = compute_collapse_metrics(epoch_snaps[ep], cfg, context_encodings,
                                         fixed_sequence, vocab_size)
        sep_vals.append(col_m.get('trajectory_separation',      np.nan))
        ctx_dec_vals.append(col_m.get('context_decodability',   np.nan))
        seq_dec_vals.append(col_m.get('sequence_step_decodability', np.nan))
        ep_vals.append(ep)

    final_loss = loss_hist[-1] if loss_hist else np.nan
    inputs_e   = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, cfg.sigma)
    tgts_e     = torch.tensor(fixed_sequence[1:], dtype=torch.long)
    best_model.eval()
    acc_vals: List[float] = []
    with torch.no_grad():
        for ctx_vec in context_encodings.values():
            logits_e, _ = best_model(inputs_e, ctx_vec.unsqueeze(0))
            acc_vals.append(compute_sequence_accuracy(logits_e, tgts_e))
    final_acc = float(np.mean(acc_vals)) if acc_vals else np.nan

    summary = _compute_trajectory_summary(
        sep_vals, ctx_dec_vals, seq_dec_vals, ep_vals, final_loss, final_acc
    )
    summary['model_type']    = cfg.name
    summary['hidden_dim']    = cfg.hidden_dim
    summary['learning_rate'] = cfg.learning_rate
    summary['init_scale']    = cfg.init_scale
    return summary


def run_hidden_size_sweep(
    context_encodings: Dict[str, torch.Tensor],
    output_dir: Path,
    hidden_sizes: Optional[List[int]] = None,
    fixed_sequence: List[int] = FIXED_SEQUENCE,
    vocab_size: int = VOCAB_SIZE,
) -> pd.DataFrame:
    """PRD4 Part 1: sweep hidden_dim × {Rich, Poor} and record collapse-trajectory metrics.

    For each (model_type, hidden_dim), trains a fresh model to asymptote with
    ``run_to_asymptote`` (epoch snapshots every 10 epochs), then derives the
    compact summary metrics defined in PRD4 Part 4.

    Outputs
    -------
    ``output_dir / hidden_size_sweep.csv``  — one row per (model_type, hidden_dim)
    """
    if hidden_sizes is None:
        # 20 log-spaced integer values from 2 → 64 gives good resolution across capacity regimes
        hidden_sizes = sorted(set(
            int(round(v))
            for v in np.logspace(np.log10(2), np.log10(64), num=20)
        ))

    rows = []
    for hd in hidden_sizes:
        for base_cfg in [RICH_CONFIG, POOR_CONFIG]:
            cfg = ModelConfig(
                name=base_cfg.name, hidden_dim=hd,
                init_scale=base_cfg.init_scale, learning_rate=base_cfg.learning_rate,
                context_alpha=base_cfg.context_alpha, sigma=base_cfg.sigma,
            )
            print(f"      [hidden_size_sweep] {cfg.name}  hidden_dim={hd} …")
            rows.append(_sweep_run_one(cfg, context_encodings, fixed_sequence, vocab_size))

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
    """PRD4 Part 2: sweep learning_rate × {Rich, Poor} and record collapse-trajectory metrics.

    Outputs
    -------
    ``output_dir / learning_rate_sweep.csv``  — one row per (model_type, learning_rate)
    """
    if learning_rates is None:
        # 30 log-spaced values from 0.0001 → 0.1 gives good resolution across lr regimes
        learning_rates = np.logspace(np.log10(0.0001), np.log10(0.1), num=30).tolist()

    rows = []
    for lr in learning_rates:
        for base_cfg in [RICH_CONFIG, POOR_CONFIG]:
            cfg = ModelConfig(
                name=base_cfg.name, hidden_dim=base_cfg.hidden_dim,
                init_scale=base_cfg.init_scale, learning_rate=lr,
                context_alpha=base_cfg.context_alpha, sigma=base_cfg.sigma,
            )
            print(f"      [lr_sweep] {cfg.name}  lr={lr:.4f} …")
            rows.append(_sweep_run_one(cfg, context_encodings, fixed_sequence, vocab_size))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "learning_rate_sweep.csv", index=False)
    return df


# ─── PRD4 Part 4: Paper-ready summary metrics ─────────────────────────────────

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
    results_df      : per-step learning DataFrame (contains 'loss', 'accuracy')
    config_meta     : dict with at minimum model_type, hidden_dim, learning_rate,
                      init_scale, participant_id
    """
    mt  = config_meta.get('model_type', '')
    pid = config_meta.get('participant_id', '')

    # ── Separation trajectory ──────────────────────────────────────────────────
    sub_col = pd.DataFrame()
    if not collapse_df.empty and 'model_type' in collapse_df.columns:
        mask = collapse_df['model_type'] == mt
        if 'participant_id' in collapse_df.columns:
            mask &= collapse_df['participant_id'] == pid
        sub_col = collapse_df[mask]

    x_col   = 'step_or_epoch' if 'step_or_epoch' in sub_col.columns else 'checkpoint_value'
    sep_vals     = sub_col['trajectory_separation'].tolist()      if not sub_col.empty and 'trajectory_separation'       in sub_col.columns else []
    ctx_dec_vals = sub_col['context_decodability'].tolist()       if not sub_col.empty and 'context_decodability'        in sub_col.columns else []
    seq_dec_vals = sub_col['sequence_step_decodability'].tolist() if not sub_col.empty and 'sequence_step_decodability'  in sub_col.columns else []
    ep_vals      = sub_col[x_col].tolist()                        if not sub_col.empty and x_col in sub_col.columns     else []

    # ── Final performance ──────────────────────────────────────────────────────
    sub_res = pd.DataFrame()
    if not results_df.empty and 'model_type' in results_df.columns:
        mask_r = results_df['model_type'] == mt
        if 'participant_id' in results_df.columns:
            mask_r &= results_df['participant_id'] == pid
        sub_res = results_df[mask_r]
    final_loss = float(sub_res['loss'].iloc[-1])     if not sub_res.empty and 'loss'     in sub_res.columns else np.nan
    final_acc  = float(sub_res['accuracy'].iloc[-1]) if not sub_res.empty and 'accuracy' in sub_res.columns else np.nan

    traj = _compute_trajectory_summary(sep_vals, ctx_dec_vals, seq_dec_vals,
                                       ep_vals, final_loss, final_acc)

    # ── Perturbation early vs late (zero-context ablation) ────────────────────
    early_ablation_cost = late_ablation_cost = np.nan
    early_hidden_shift  = late_hidden_shift  = np.nan

    if not perturbation_df.empty and 'model_type' in perturbation_df.columns:
        sub_pert = perturbation_df[perturbation_df['model_type'] == mt]
        pert_x   = 'checkpoint_value' if 'checkpoint_value' in sub_pert.columns else None
        if pert_x and not sub_pert.empty and 'perturbation_mode' in sub_pert.columns:
            zero_pert   = sub_pert[sub_pert['perturbation_mode'] == 'zero']
            checkpoints = sorted(zero_pert[pert_x].unique())
            if len(checkpoints) >= 2:
                e_sub = zero_pert[zero_pert[pert_x] == checkpoints[0]]
                l_sub = zero_pert[zero_pert[pert_x] == checkpoints[-1]]
                early_ablation_cost = float(e_sub['delta_loss'].mean())        if 'delta_loss'        in e_sub.columns else np.nan
                late_ablation_cost  = float(l_sub['delta_loss'].mean())        if 'delta_loss'        in l_sub.columns else np.nan
                early_hidden_shift  = float(e_sub['hidden_state_shift'].mean()) if 'hidden_state_shift' in e_sub.columns else np.nan
                late_hidden_shift   = float(l_sub['hidden_state_shift'].mean()) if 'hidden_state_shift' in l_sub.columns else np.nan

    return {
        **traj,
        **config_meta,
        'early_context_ablation_cost': early_ablation_cost,
        'late_context_ablation_cost':  late_ablation_cost,
        'early_hidden_shift':          early_hidden_shift,
        'late_hidden_shift':           late_hidden_shift,
    }


# ─── PRD4 plotting ────────────────────────────────────────────────────────────

def plot_hidden_size_sweep(df: pd.DataFrame, output_dir: Path):
    """PRD4 Part 1 figures: peak/final separation and compression vs hidden_dim."""
    if df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}
    panels = [
        ('peak_separation',    'Peak Separation'),
        ('final_separation',   'Final Separation'),
        ('compression_amount', 'Compression Amount (peak − final)'),
        ('final_loss',         'Final CE Loss'),
    ]
    avail = [(c, l) for c, l in panels if c in df.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, avail):
        for mt, grp in df.groupby('model_type'):
            grp_s = grp.sort_values('hidden_dim')
            ax.plot(grp_s['hidden_dim'], grp_s[col], marker='o',
                    label=mt, color=pal.get(mt, 'steelblue'), linewidth=1.8)
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle('Hidden-Size Robustness Sweep', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / 'hidden_size_sweep.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_learning_rate_sweep(df: pd.DataFrame, output_dir: Path):
    """PRD4 Part 2 figures: peak/final separation and compression vs learning_rate."""
    if df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}
    panels = [
        ('peak_separation',    'Peak Separation'),
        ('final_separation',   'Final Separation'),
        ('compression_amount', 'Compression Amount (peak − final)'),
        ('final_loss',         'Final CE Loss'),
    ]
    avail = [(c, l) for c, l in panels if c in df.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, avail):
        for mt, grp in df.groupby('model_type'):
            grp_s = grp.sort_values('learning_rate')
            ax.plot(grp_s['learning_rate'], grp_s[col], marker='o',
                    label=mt, color=pal.get(mt, 'steelblue'), linewidth=1.8)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle('Learning-Rate Robustness Sweep', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / 'learning_rate_sweep.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_collapse_driver_dynamics(driver_df: pd.DataFrame, output_dir: Path):
    """PRD4 Part 3 figure: gradient alignment, JS divergence, separation, prob-gap over training."""
    if driver_df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal   = {'Rich': 'darkgreen', 'Poor': 'firebrick'}
    x_col = 'checkpoint_value' if 'checkpoint_value' in driver_df.columns else 'step_or_epoch'
    panels = [
        ('gradient_alignment',     'Gradient Alignment\n(cosine sim, ctx A vs B)'),
        ('output_js_divergence',   'Output JS Divergence\n(ctx A vs B)'),
        ('trajectory_separation',  'Trajectory Separation'),
        ('correct_token_prob_gap', 'Correct-Token Prob Gap'),
    ]
    avail = [(c, l) for c, l in panels if c in driver_df.columns]
    if not avail:
        return
    ncols = min(len(avail), 4)
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_i, (col, label) in enumerate(avail):
        ax  = axes[ax_i]
        sub = driver_df[driver_df[col].notna()]
        for mt, grp in sub.groupby('model_type'):
            agg = grp.groupby(x_col)[col].mean().reset_index()
            ax.plot(agg[x_col], agg[col], label=mt,
                    color=pal.get(mt, 'steelblue'), linewidth=1.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    for ax_i in range(len(avail), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle('Collapse Driver Dynamics', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / 'collapse_driver_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_dataset(data_path_obj: Path):
    data_filename = data_path_obj.name
    print(f"\n==============================================================================")
    print(f"📦 Processing Dataset: {data_filename}")
    print(f"==============================================================================")
    
    # 1. Load Data
    if not data_path_obj.exists():
        print(f"⚠️ File {data_path_obj} not found. Skipping...")
        return

    df = pd.read_csv(data_path_obj)
    
    # Filter for standard participants
    participant_counts = df['ID'].value_counts()
    if participant_counts.empty:
        print("   ⚠️ Dataset is empty or has no IDs.")
        return

    mode_count = participant_counts.mode()[0]
    valid_pids = participant_counts[participant_counts == mode_count].index.tolist()
    
    print(f"   Data loaded. Processing {len(valid_pids)} participants (mode trials: {mode_count}).")
    print(f"   Simulation Config -> Sequence: {FIXED_SEQUENCE}, Vocab Size: {VOCAB_SIZE}")
    print(f"   Configs: Rich(alpha={RICH_CONFIG.context_alpha}, init={RICH_CONFIG.init_scale}), Poor(alpha={POOR_CONFIG.context_alpha}, init={POOR_CONFIG.init_scale})")
    
    dataset_slug = data_path_obj.stem
    output_dir = Path(f"results_{dataset_slug}")
    output_dir.mkdir(exist_ok=True)
    
    # --- DYNAMIC CONTEXT IDENTIFICATION ---
    # For datasets where the per-trial context type is in 'trialType' (e.g. mixed
    # Maze-Melody datasets), use that column instead of 'group', which in those
    # files just encodes the participant cohort (e.g. 'Maze-Melody').
    if 'trialType' in df.columns and df['trialType'].nunique() > 1:
        context_col = 'trialType'
    else:
        context_col = 'group'

    available_groups = sorted(df[context_col].dropna().unique().tolist())
    if not available_groups:
        print("   ⚠️ No context column or empty groups found. Defaulting to 'Default'.")
        df[context_col] = 'Default'
        available_groups = ['Default']

    print(f"   Identified Contexts (from column '{context_col}'): {available_groups}")

    # Report counterbalance distribution if present
    if 'counterBalance' in df.columns:
        cb_counts = df.drop_duplicates('ID')['counterBalance'].value_counts().sort_index()
        print(f"   Counterbalance distribution (participants): {cb_counts.to_dict()}")

    # Build Context Encodings
    num_contexts = len(available_groups)
    context_dim = max(2, num_contexts) 
    
    context_encodings = {}
    eye = torch.eye(context_dim)
    
    for i, grp in enumerate(available_groups):
        context_encodings[grp] = eye[i]

    # Handle Control Condition Target
    if 'Maze' in available_groups:
        dominant_context = 'Maze' 
    else:
        dominant_context = available_groups[0]
    
    # 2. Run Experiments (Rich vs Poor + Control Condition)
    
    # R2: Control Condition (Single Context Training)
    # We use the RICH configuration but restrict training to 'Maze' only.
    CONTROL_CONFIG = ModelConfig(name='Control (Rich)', hidden_dim=6, init_scale=0.5, learning_rate=0.005, context_alpha=1.0)
    
    configs = [RICH_CONFIG, POOR_CONFIG, CONTROL_CONFIG]
    all_results = []
    mechanistic_results = [] # Store diagnostic stats
    all_pca_data = [] # Collect all PCA data for Rich/Poor
    
    for config in configs:
        is_control = (config.name == 'Control (Rich)')
        mode_str = "Single-Context" if is_control else "Standard"
        print(f"\n🚀 Running {config.name} Regime (dim={config.hidden_dim}, alpha={config.context_alpha}) [{mode_str}]...")
        
        for i, pid in enumerate(valid_pids):
            # Sort by Block (Trial) then Sequence Index
            p_data = df[df['ID'] == pid].sort_values(by=["new_trialNumber", "sequenceNumber"])
            
            # Pass single_context_mode flag if Control
            # Ensure we train on a context that actually exists in the data
            target_ctx = dominant_context if is_control else 'Maze'
            if target_ctx not in available_groups:
                target_ctx = dominant_context

            res, model, _snaps, _snap_meta = run_participant_experiment(p_data, config, 
                                                    single_context_mode=is_control,
                                                    train_context=target_ctx,
                                                    context_encodings=context_encodings,
                                                    context_col=context_col)
            
            # Convert to DF and tag
            res_df = pd.DataFrame(res)
            res_df['model_type'] = config.name
            res_df['participant_id'] = pid
            all_results.append(res_df)
            
            # R4: Run Mechanistic Analysis on trained model (Skip for control if desired, but interesting to see)
            if not is_control:
                mech_stats = run_mechanistic_analysis(model, config=config, 
                                                      active_contexts=available_groups,
                                                      context_encodings=context_encodings)
                if mech_stats:
                    mech_stats['model_type'] = config.name
                    mech_stats['participant_id'] = pid
                    
                    # Save first participant's geometry plot
                    if i == 0:
                         plot_pca_geometry(mech_stats, config.name, output_dir)
                    
                    # Export PCA Data for R Plotting (All Participants)
                    X_pca = mech_stats['pca_coords']
                    y = mech_stats['pca_labels']
                    
                    # Fix: Calculate sequence steps dynamically based on number of visible contexts 
                    # instead of hardcoding // 2 (which fails for single-context datasets)
                    num_ctx_pca = len(np.unique(y))
                    if num_ctx_pca > 0:
                        seq_len_pca = len(y) // num_ctx_pca
                        seq_steps = np.tile(np.arange(seq_len_pca), num_ctx_pca)
                    else:
                        seq_steps = np.zeros(len(y))
                    
                    df_pca_r = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    df_pca_r['Context'] = y
                    df_pca_r['Step'] = seq_steps
                    df_pca_r['Model_Type'] = config.name
                    df_pca_r['Participant_ID'] = pid
                    all_pca_data.append(df_pca_r)
                    
                    # Keep scalar metrics for summary
                    scalars = {k: v for k, v in mech_stats.items() if not isinstance(v, np.ndarray)}
                    mechanistic_results.append(scalars)
            
    final_df = pd.concat(all_results)
    final_df.to_csv(output_dir / "rich_poor_learning_dynamics.csv", index=False)

    # Save all PCA data to one file
    if all_pca_data:
        all_pca_df = pd.concat(all_pca_data, ignore_index=True)
        all_pca_df.to_csv(output_dir / "all_pca_data.csv", index=False)

    # Export Mechanistic Metrics for Inferential Stats (e.g., t-test on weight ratios)
    if mechanistic_results:
        mech_df = pd.DataFrame(mechanistic_results)
        mech_df.to_csv(output_dir / "mechanistic_metrics.csv", index=False)
    
    # 3. Visualization
    print("   Generating plots...")
    plot_rich_poor_comparison(final_df, output_dir)
    plot_normalized_model_vs_human(final_df, output_dir)
    plot_accuracy_entropy_curves(final_df, output_dir)
    
    if mechanistic_results:
        print("   Generating mechanistic diagnostic plots...")
        plot_mechanistic_summary(mechanistic_results, output_dir)

    # 4. Asymptote Analysis — retrain Rich & Poor to convergence, then compare geometry
    print("   Running asymptote training (Rich & Poor)...")
    asymptote_mechs   = []
    wg_rows: List[Dict] = []     # D4: weight/gradient dynamics
    collapse_asym_rows: List[Dict] = []  # D2: asymptote collapse dynamics
    perturb_asym_rows: List[Dict] = []   # D3: asymptote perturbation
    shared_pca_rows: List[Dict]  = []    # D5: shared PCA trajectories
    collapse_driver_rows: List[Dict] = []   # PRD4-3: gradient alignment + output divergence
    context_ablation_rows: List[Dict] = []  # PRD4-3C: recurrent-only competence test

    for config in [RICH_CONFIG, POOR_CONFIG]:
        context_dim_asym = next(iter(context_encodings.values())).shape[0]
        asym_model = ContextSequenceRNN(VOCAB_SIZE, context_dim=context_dim_asym, config=config)
        best_model, asym_loss, extra = run_to_asymptote(
            asym_model, config, FIXED_SEQUENCE, VOCAB_SIZE, context_encodings,
            epoch_interval=10, track_gradients=True,
        )
        mech = run_mechanistic_analysis(best_model, config,
                                        context_encodings=context_encodings)
        scalars = {k: v for k, v in mech.items() if not isinstance(v, np.ndarray)}
        scalars['model_type']   = config.name
        scalars['asymptote_ce'] = asym_loss[-1] if asym_loss else float('nan')
        asymptote_mechs.append(scalars)
        print(f"      {config.name}: converged at CE = {scalars['asymptote_ce']:.4f} "
              f"after {len(asym_loss)} epochs")
        if mech:
            plot_pca_geometry(mech, f"{config.name}_asymptote", output_dir)

        # D4: Build weight/gradient dynamics rows from epoch snapshots + grad_history
        epoch_snaps = extra.get('epoch_snapshots', {})
        grad_hist   = {g['epoch']: g for g in extra.get('grad_history', [])}
        acc_hist    = extra.get('accuracy_history', {})
        for ep, snap in epoch_snaps.items():
            wnorms = compute_weight_norms(snap, VOCAB_SIZE)
            col_m  = compute_collapse_metrics(snap, config, context_encodings)
            grad_row = grad_hist.get(ep - 1, {})
            snap_acc = acc_hist.get(ep, np.nan)
            row: Dict = {'model_type': config.name, 'epoch': ep,
                         'loss': asym_loss[ep - 1] if ep - 1 < len(asym_loss) else np.nan,
                         'accuracy': snap_acc}
            row.update(wnorms)
            row['grad_context']   = grad_row.get('grad_context',   np.nan)
            row['grad_recurrent'] = grad_row.get('grad_recurrent', np.nan)
            row['grad_output']    = grad_row.get('grad_output',    np.nan)
            row.update({k: col_m.get(k, np.nan)
                        for k in ('context_decodability', 'sequence_step_decodability',
                                  'trajectory_separation')})
            wg_rows.append(row)

            # D2: asymptote collapse dynamics
            col_row = {'participant_id': 'asymptote', 'model_type': config.name,
                       'training_phase': 'asymptote', 'checkpoint_type': 'epoch',
                       'checkpoint_value': ep, 'step_or_epoch': ep,
                       'loss': row['loss'], 'accuracy': snap_acc}
            col_row.update(col_m)
            col_row.update(wnorms)
            collapse_asym_rows.append(col_row)

            # D3: perturbation at this epoch checkpoint
            pert_results = run_context_perturbation_analysis(
                snap, config, context_encodings, FIXED_SEQUENCE, VOCAB_SIZE
            )
            for pr in pert_results:
                pr.update({'participant_id': 'asymptote', 'model_type': config.name,
                            'checkpoint_type': 'epoch', 'checkpoint_value': ep})
                perturb_asym_rows.append(pr)

            # PRD4-3A+B: gradient alignment + output divergence (asymptote checkpoints)
            asym_grad_align = compute_gradient_alignment(snap, config, context_encodings)
            asym_out_div    = compute_output_divergence(snap, config, context_encodings)
            asym_driver_row: Dict = {
                'participant_id':             'asymptote',
                'model_type':                 config.name,
                'checkpoint_type':            'epoch',
                'checkpoint_value':           ep,
                'gradient_alignment':         asym_grad_align,
                'trajectory_separation':      col_m.get('trajectory_separation', np.nan),
                'context_decodability':       col_m.get('context_decodability', np.nan),
                'sequence_step_decodability': col_m.get('sequence_step_decodability', np.nan),
                'loss':                       row['loss'],
                'accuracy':                   snap_acc,
            }
            asym_driver_row.update(asym_out_div)
            collapse_driver_rows.append(asym_driver_row)

            # PRD4-3C: context ablation (recurrent-only competence)
            asym_abl = run_context_ablation_test(snap, config, context_encodings)
            for ar in asym_abl:
                ar.update({'participant_id': 'asymptote', 'model_type': config.name,
                           'checkpoint_type': 'epoch', 'checkpoint_value': ep})
                context_ablation_rows.append(ar)

        # D5: Shared PCA — pool hidden states from all epoch snapshots, fit one PCA
        if epoch_snaps and SKLEARN_AVAILABLE:
            hs_pool = {
                str(ep): collect_checkpoint_hidden_states(snap, config, context_encodings)
                for ep, snap in epoch_snaps.items()
            }
            shared_pca = build_shared_pca(hs_pool)
            if shared_pca is not None:
                ctx_keys_list = list(context_encodings.keys())
                seq_len_spca  = list(context_encodings.values())[0].shape[0]  # dim ≠ seq_len, re-run model
                # We need actual seq_len from run
                _inp_spca = sequence_to_gaussian(FIXED_SEQUENCE, VOCAB_SIZE, sigma=config.sigma)
                seq_len_spca = _inp_spca.shape[1]

                for ep, snap in epoch_snaps.items():
                    snap.eval()
                    with torch.no_grad():
                        for ctx_name, ctx_vec in context_encodings.items():
                            _, h_ep = snap(_inp_spca, ctx_vec.unsqueeze(0))
                            h_np    = h_ep.squeeze(0).cpu().numpy()
                            proj    = shared_pca.transform(h_np)
                            for step_i, (pc1, pc2) in enumerate(proj):
                                shared_pca_rows.append({
                                    'participant_id':  'asymptote',
                                    'model_type':      config.name,
                                    'checkpoint_type': 'epoch',
                                    'checkpoint_value': ep,
                                    'context':         ctx_name,
                                    'step_in_sequence': step_i,
                                    'PC1': float(pc1),
                                    'PC2': float(pc2),
                                })

    plot_asymptote_geometry(asymptote_mechs, output_dir)
    if asymptote_mechs:
        pd.DataFrame(asymptote_mechs).to_csv(output_dir / "asymptote_geometry.csv", index=False)

    # D4: Save weight/gradient dynamics
    if wg_rows:
        wg_df = pd.DataFrame(wg_rows)
        wg_df.to_csv(output_dir / "weight_gradient_dynamics.csv", index=False)
        plot_weight_gradient_dynamics(wg_df, output_dir)

    # 5. Performance-Matched Geometry — CE-matched and accuracy-matched (D1)
    print("   Running performance-matched geometry analysis (CE + accuracy thresholds)...")

    # Accuracy thresholds mirroring CE thresholds (roughly)
    ACC_THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9]

    matched_rows_ce  : List[Dict] = []
    matched_rows_acc : List[Dict] = []
    snap_meta_rows   : List[Dict] = []
    collapse_part_rows: List[Dict] = []   # D2: participant training collapse
    perturb_part_rows : List[Dict] = []   # D3: participant training perturbation

    n_match = min(5, len(valid_pids))
    pids_match = valid_pids[:n_match]

    for config in [RICH_CONFIG, POOR_CONFIG]:
        print(f"   [matched+collapse] {config.name} — {n_match} participants …")
        for pid in pids_match:
            p_data = df[df['ID'] == pid].sort_values(
                by=["new_trialNumber", "sequenceNumber"])

            res_p, _, snaps_p, snap_meta_p = run_participant_experiment(
                p_data, config,
                context_encodings=context_encodings,
                context_col=context_col,
                snapshot_loss_thresholds=PERF_THRESHOLDS,
                snapshot_accuracy_thresholds=ACC_THRESHOLDS,
                step_interval=5,
            )

            # Annotate snapshot metadata with participant/model info
            for row in snap_meta_p:
                row['participant_id'] = pid
                row['model_type']     = config.name
                snap_meta_rows.append(row)

            # CE-matched geometry (D1.3 Figure A)
            for snap_key, snap_model in snaps_p.items():
                if not snap_key.startswith('ce_'):
                    continue
                thr  = float(snap_key[3:])
                mech = run_mechanistic_analysis(snap_model, config,
                                                context_encodings=context_encodings)
                row: Dict = {'model_type': config.name, 'participant_id': pid,
                              'ce_threshold': thr}
                row.update({k: v for k, v in mech.items()
                             if not isinstance(v, np.ndarray)})
                matched_rows_ce.append(row)

            # Accuracy-matched geometry (D1.3 Figure B)
            for snap_key, snap_model in snaps_p.items():
                if not snap_key.startswith('acc_'):
                    continue
                thr  = float(snap_key[4:])
                mech = run_mechanistic_analysis(snap_model, config,
                                                context_encodings=context_encodings)
                row = {'model_type': config.name, 'participant_id': pid,
                       'acc_threshold': thr}
                row.update({k: v for k, v in mech.items()
                             if not isinstance(v, np.ndarray)})
                matched_rows_acc.append(row)

            # D2: collapse dynamics during participant training (step_interval snapshots)
            for snap_key, snap_model in snaps_p.items():
                if not snap_key.startswith('step_'):
                    continue
                step_n = int(snap_key[5:])
                col_m  = compute_collapse_metrics(snap_model, config, context_encodings)
                wnorms = compute_weight_norms(snap_model, VOCAB_SIZE)
                # Get loss/acc at this step from results
                step_idx = step_n - 1  # 0-indexed
                step_loss = res_p['loss'][step_idx] if step_idx < len(res_p['loss']) else np.nan
                step_acc  = res_p['accuracy'][step_idx] if step_idx < len(res_p['accuracy']) else np.nan
                step_conf = res_p['confidence'][step_idx] if step_idx < len(res_p['confidence']) else np.nan
                step_ent  = res_p['entropy'][step_idx] if step_idx < len(res_p['entropy']) else np.nan

                col_row = {
                    'participant_id':   pid,
                    'model_type':       config.name,
                    'training_phase':   'participant',
                    'checkpoint_type':  'step',
                    'checkpoint_value': step_n,
                    'step_or_epoch':    step_n,
                    'loss':             step_loss,
                    'accuracy':         step_acc,
                    'confidence':       step_conf,
                    'entropy':          step_ent,
                }
                col_row.update(col_m)
                col_row.update(wnorms)
                collapse_part_rows.append(col_row)

                # D3: perturbation at this step checkpoint
                pert_results = run_context_perturbation_analysis(
                    snap_model, config, context_encodings, FIXED_SEQUENCE, VOCAB_SIZE
                )
                for pr in pert_results:
                    pr.update({'participant_id': pid, 'model_type': config.name,
                                'checkpoint_type': 'step', 'checkpoint_value': step_n})
                    perturb_part_rows.append(pr)

                # PRD4-3A+B: gradient alignment + output divergence
                grad_align = compute_gradient_alignment(snap_model, config, context_encodings)
                out_div    = compute_output_divergence(snap_model, config, context_encodings)
                driver_row: Dict = {
                    'participant_id':             pid,
                    'model_type':                 config.name,
                    'checkpoint_type':            'step',
                    'checkpoint_value':           step_n,
                    'gradient_alignment':         grad_align,
                    'trajectory_separation':      col_m.get('trajectory_separation', np.nan),
                    'context_decodability':       col_m.get('context_decodability', np.nan),
                    'sequence_step_decodability': col_m.get('sequence_step_decodability', np.nan),
                    'loss':                       step_loss,
                    'accuracy':                   step_acc,
                }
                driver_row.update(out_div)
                collapse_driver_rows.append(driver_row)

                # PRD4-3C: context ablation (recurrent-only competence)
                abl_results = run_context_ablation_test(snap_model, config, context_encodings)
                for ar in abl_results:
                    ar.update({'participant_id': pid, 'model_type': config.name,
                               'checkpoint_type': 'step', 'checkpoint_value': step_n})
                    context_ablation_rows.append(ar)

    # Save matched geometry CSVs
    if matched_rows_ce:
        df_ce = pd.DataFrame(matched_rows_ce)
        df_ce.to_csv(output_dir / "matched_geometry_ce.csv", index=False)
        plot_matched_geometry(df_ce, output_dir)

    if matched_rows_acc:
        df_acc = pd.DataFrame(matched_rows_acc)
        df_acc.to_csv(output_dir / "matched_geometry_accuracy.csv", index=False)
        # Figure B: accuracy-matched geometry
        _plot_matched_geometry_generic(df_acc, 'acc_threshold',
                                        'Accuracy threshold (matched performance)',
                                        output_dir / "matched_geometry_accuracy.png")

    if snap_meta_rows:
        pd.DataFrame(snap_meta_rows).to_csv(output_dir / "snapshot_metadata.csv", index=False)

    # D2: Combined collapse dynamics
    collapse_all_rows = collapse_part_rows + collapse_asym_rows
    if collapse_all_rows:
        collapse_df = pd.DataFrame(collapse_all_rows)
        collapse_df.to_csv(output_dir / "collapse_dynamics.csv", index=False)
        plot_collapse_dynamics(collapse_df, output_dir)
        plot_normalized_collapse(collapse_df, output_dir)
        plot_phase_portrait(collapse_df, output_dir)
        plot_dual_decoding_dynamics(collapse_df, output_dir)

    # D3: Combined perturbation
    perturb_all = perturb_part_rows + perturb_asym_rows
    if perturb_all:
        perturb_df = pd.DataFrame(perturb_all)
        perturb_df.to_csv(output_dir / "context_perturbation.csv", index=False)
        plot_perturbation_dynamics(perturb_df, output_dir)

    # D5: Shared PCA trajectories
    if shared_pca_rows:
        pd.DataFrame(shared_pca_rows).to_csv(
            output_dir / "shared_pca_trajectories.csv", index=False)
        plot_shared_pca_panels(pd.DataFrame(shared_pca_rows), output_dir)

    # PRD4-3: Collapse driver dynamics CSV + plot
    if collapse_driver_rows:
        driver_df = pd.DataFrame(collapse_driver_rows)
        driver_df.to_csv(output_dir / "collapse_driver_dynamics.csv", index=False)
        plot_collapse_driver_dynamics(driver_df, output_dir)

    # PRD4-3C: Context ablation dynamics CSV
    if context_ablation_rows:
        pd.DataFrame(context_ablation_rows).to_csv(
            output_dir / "context_ablation_dynamics.csv", index=False)

    # PRD4-4: Paper-ready summary metrics
    print("   Computing paper-ready summary metrics…")
    summary_rows: List[Dict] = []
    collapse_all_df = pd.DataFrame(collapse_all_rows) if collapse_all_rows else pd.DataFrame()
    perturb_all_df  = pd.DataFrame(perturb_all)       if perturb_all       else pd.DataFrame()
    for cfg_s in [RICH_CONFIG, POOR_CONFIG]:
        for pid_s in valid_pids[:n_match]:
            meta = {
                'model_type':    cfg_s.name,
                'hidden_dim':    cfg_s.hidden_dim,
                'learning_rate': cfg_s.learning_rate,
                'init_scale':    cfg_s.init_scale,
                'participant_id': pid_s,
            }
            summary_rows.append(compute_model_summary_metrics(
                collapse_all_df, perturb_all_df, final_df, meta))
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "model_summary_metrics.csv", index=False)

    # PRD4-1 & PRD4-2: Robustness sweeps (hidden-size + learning-rate)
    print("   Running robustness sweeps (hidden-size + learning-rate)…")
    hs_df = run_hidden_size_sweep(context_encodings, output_dir)
    if not hs_df.empty:
        plot_hidden_size_sweep(hs_df, output_dir)
    lr_df = run_learning_rate_sweep(context_encodings, output_dir)
    if not lr_df.empty:
        plot_learning_rate_sweep(lr_df, output_dir)

    print(f"\n✅ Analysis Complete for {data_filename}. Results in {output_dir}")

def plot_normalized_model_vs_human(data_input, output_dir):
    """
    Plots normalized trajectories for Human (RT) and Model (Loss) over time.
    Separates trajectories into disconnected blocks based on trials (1-12).
    Plots every sequence individually (1-9 per block).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise ValueError("Input data must be pandas DataFrame")
    
    group_cols = ['model_type', 'context', 'block_id', 'sequence_in_block']
    agg_df = df.groupby(group_cols)[['loss', 'transfer_loss', 'rt', 'global_step']].mean().reset_index()
    agg_df = agg_df.sort_values(by=['block_id', 'sequence_in_block'])
    
    model_types = df['model_type'].unique()
    
    for m_type in model_types:
        model_data = agg_df[agg_df['model_type'] == m_type].copy()
        
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())
            
        model_data['norm_rt'] = normalize(model_data['rt'])
        model_data['norm_loss'] = normalize(model_data['loss'])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        def plot_series(sub_df, y_col, color, label, marker):
            blocks = sub_df['block_id'].unique()
            first_label = True
            for b_id in sorted(blocks):
                block_data = sub_df[sub_df['block_id'] == b_id].sort_values('sequence_in_block')
                x = block_data['global_step'].values
                y = block_data[y_col].values
                lbl = label if first_label else None
                ax.plot(x, y, color=color, label=lbl, marker=marker, linewidth=2, alpha=0.9)
                first_label = False

        maze_df = model_data[model_data['context'] == 'Maze']
        plot_series(maze_df, 'norm_rt', 'lightblue', 'Maze Human RT', 'o')
        plot_series(maze_df, 'norm_loss', 'darkblue', 'Maze Model Loss', None)
        
        melody_df = model_data[model_data['context'] == 'Melody']
        plot_series(melody_df, 'norm_rt', 'salmon', 'Melody Human RT', 's')
        plot_series(melody_df, 'norm_loss', 'darkred', 'Melody Model Loss', None)
        
        ax.set_title(f"Normalized Trajectories: {m_type} Model (Solid) vs Human (Markers)", fontsize=16)
        ax.set_xlabel("Cumulative Sequence Index", fontsize=12)
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        plt.tight_layout()
        
        fig.savefig(output_dir / f"{m_type}_normalized_trajectories.png", dpi=300)
        plt.close(fig)

def plot_rich_poor_comparison(df, output_dir):
    """Generates diagnostic plots for Rich vs Poor hypothesis."""
    agg_df = df.groupby(['model_type', 'global_step'])[['loss', 'transfer_loss', 'rt']].mean().reset_index()
    agg_df['generalization_gap'] = agg_df['transfer_loss'] - agg_df['loss']

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Generalization Gap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=agg_df, x='global_step', y='generalization_gap', hue='model_type',
                 palette={'Rich': 'darkgreen', 'Poor': 'firebrick', 'Control (Rich)': 'royalblue'}, linewidth=1.5, ax=ax)
    
    ax.set_title("Context Specificity (Generalization Gap)", fontsize=16)
    ax.set_ylabel("Transfer Loss - Training Loss")
    ax.set_xlabel("Sequence Index")
    ax.axhline(0, linestyle='--', color='gray', alpha=0.5)
    
    fig.savefig(output_dir / "generalization_gap_analysis.png", dpi=300)
    plt.close(fig)
    
    # 2. Learning Curves (Loss)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=agg_df, x='global_step', y='loss', hue='model_type',
                 palette={'Rich': 'darkgreen', 'Poor': 'firebrick', 'Control (Rich)': 'royalblue'}, linewidth=1.5, ax=ax)
    ax.set_title("Learning Dynamics (Cross Entropy Loss)", fontsize=16)
    fig.savefig(output_dir / "learning_curves.png", dpi=300)
    plt.close(fig)


def plot_accuracy_entropy_curves(df: pd.DataFrame, output_dir: Path):
    """4-panel figure: loss, accuracy, entropy, confidence — per model type.

    Loss and accuracy are kept in separate panels so nobody can claim the
    entropy/confidence pattern is just an artefact of correct-trial rate.
    The 'correct_surprisal' column (−log p(correct) on correct positions) is
    the human-RT analogue — shown separately to make that framing explicit.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick', 'Control (Rich)': 'royalblue'}

    metrics = [
        ('loss',              'Mean CE Loss',          'CE loss (↓ = better)'),
        ('accuracy',          'Accuracy',              'Fraction correct (↑ = better)'),
        ('entropy',           'Prediction Entropy',    'Output entropy / nats (↓ = more certain)'),
        ('confidence',        'Prediction Confidence', 'Max-softmax probability (↑ = more confident)'),
        ('correct_surprisal', 'Correct-Trial Surprisal\n(RT analogue)',
                                                       '−log p(correct) on correct trials (↓ = less surprised)'),
    ]

    agg = (df.groupby(['model_type', 'global_step'])[
               [m for m, _, _ in metrics]].mean().reset_index())

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
    axes = axes.flatten()

    for ax_i, (col, title, ylabel) in enumerate(metrics):
        ax = axes[ax_i]
        if col not in agg.columns:
            ax.set_visible(False)
            continue
        sns.lineplot(data=agg, x='global_step', y=col, hue='model_type',
                     palette=pal, linewidth=1.5, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Sequence Index")
        ax.set_ylabel(ylabel)
        ax.get_legend().remove() if ax.get_legend() else None

    # Shared legend on the last unused panel
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title='Model', fontsize=10, loc='center')
    axes[-1].set_visible(True)
    axes[-1].axis('off')

    fig.suptitle("Learning Trajectories — Loss, Accuracy, Entropy & Confidence",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "learning_curves_full.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_matched_geometry_generic(
    matched_df: pd.DataFrame,
    threshold_col: str,
    threshold_label: str,
    save_path: Path,
):
    """Generic matched-geometry bar plot for any threshold column."""
    if matched_df.empty:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}
    geo_metrics = [
        ('context_decodability',    'Context Decodability'),
        ('weight_ratio_cx_in',      'Context / Input Weight Ratio'),
        ('effective_dimensionality','Effective Dimensionality'),
        ('trajectory_separation',   'PCA Trajectory Separation'),
    ]
    geo_metrics = [(c, l) for c, l in geo_metrics if c in matched_df.columns]
    if not geo_metrics:
        return
    n = len(geo_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, geo_metrics):
        sns.barplot(data=matched_df, x=threshold_col, y=col, hue='model_type',
                    palette=pal, errorbar='se', ax=ax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(threshold_label)
        ax.set_ylabel(label)
    fig.suptitle("Representational Geometry at Matched Performance", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_matched_geometry(matched_df: pd.DataFrame, output_dir: Path):
    """Compare mechanistic metrics at matched CE performance levels.

    For each threshold (matched performance) and each metric, shows Rich vs Poor
    side by side — ensuring geometry differences are not just mastery differences.
    """
    if matched_df.empty:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}

    geo_metrics = [
        ('context_decodability',    'Context Decodability'),
        ('weight_ratio_cx_in',      'Context / Input Weight Ratio'),
        ('effective_dimensionality','Effective Dimensionality'),
        ('trajectory_separation',   'PCA Trajectory Separation'),
    ]
    geo_metrics = [(c, l) for c, l in geo_metrics if c in matched_df.columns]

    if not geo_metrics:
        return

    n = len(geo_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, (col, label) in zip(axes, geo_metrics):
        sns.barplot(data=matched_df, x='ce_threshold', y=col, hue='model_type',
                    palette=pal, errorbar='se', ax=ax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("CE threshold (matched performance)")
        ax.set_ylabel(label)
        ax.axhline(0.5 if col == 'context_decodability' else 0,
                   linestyle='--', color='gray', alpha=0.4)

    fig.suptitle("Representational Geometry at Matched Predictive Performance",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "matched_geometry.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_asymptote_geometry(
    asymptote_mechs: List[Dict],
    output_dir: Path,
):
    """Compare geometry metrics (Rich vs Poor) after training to asymptote."""
    if not asymptote_mechs:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    df = pd.DataFrame(asymptote_mechs)
    pal = {'Rich': 'darkgreen', 'Poor': 'firebrick'}

    geo_metrics = [
        ('context_decodability',    'Context Decodability'),
        ('weight_ratio_cx_in',      'Context / Input Weight Ratio'),
        ('effective_dimensionality','Effective Dimensionality'),
        ('trajectory_separation',   'PCA Trajectory Separation'),
    ]
    geo_metrics = [(c, l) for c, l in geo_metrics if c in df.columns]
    if not geo_metrics:
        return

    n = len(geo_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, (col, label) in zip(axes, geo_metrics):
        sns.barplot(data=df, x='model_type', y=col, palette=pal,
                    errorbar='se', ax=ax)
        sns.stripplot(data=df, x='model_type', y=col, color='black',
                      alpha=0.3, size=4, ax=ax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel(label)

    fig.suptitle("Representational Geometry After Training to Asymptote",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "asymptote_geometry.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    print("🔬 Rich vs. Poor Model Analysis Initiated [Batch Mode]...")
    
    # Look for files in various potential locations relative to script
    potential_dirs = [
        Path("FTTData"),
        Path("CogSci/FTTData"),
        Path("../FTTData")
    ]
    
    data_files = []
    
    for d in potential_dirs:
        if d.exists():
            print(f"   Scanning directory: {d.resolve()}")
            # Find all CSVs
            found = list(d.glob("*.csv"))
            data_files.extend(found)
            
    # Remove duplicates if any (by resolving absolute path)
    unique_files = {f.resolve(): f for f in data_files}.values()
    
    if not unique_files:
        print("❌ No CSV files found in logical FTTData directories.")
        return
        
    sorted_files = sorted(list(unique_files), key=lambda x: x.name)
    print(f"   Found {len(sorted_files)} datasets.")
    
    for data_path in sorted_files:
        process_dataset(data_path)

if __name__ == "__main__":
    main()
