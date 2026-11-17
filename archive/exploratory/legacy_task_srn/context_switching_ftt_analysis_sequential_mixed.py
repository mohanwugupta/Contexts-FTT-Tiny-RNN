#!/usr/bin/env python3
"""
Context-Specific Learning: Rich vs. Poor SRN Analysis
=====================================================

This script implements a "Rich" vs "Poor" model comparison to minimize/maximize
context-dependency in sequence learning.

Hypothesis:
- "Rich" models (high capacity, high gain) will quickly separate contexts but fail to generalize.
- "Poor" models (low capacity, low gain) will be forced to find shared structure, aiding transfer.

Key Features:
- Unified Model Architecture (Context-Gated RNN)
- Parameter Sweeps for Rich/Poor regimes
- Online Transfer Probing (measure generalization during training)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
import os # Added for path operations if needed

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not found. Mechanistic analysis will be limited.")

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
    # Cue-belief gating parameters (PRD: cue-only belief + entropy-gated plasticity).
    # Cognitively: the model maintains a running estimate of context identity from
    # observable cues alone, and plasticity is gated by its own epistemic certainty.
    belief_lambda: float = 0.01    # Inertia of past belief (higher = slower context switch)
    belief_beta: float = 8.0      # Cue evidence strength (larger = sharper belief updates)
    plasticity_floor: float = 0.0 # Minimum plasticity gain (0 = full suppression during Mixed)
    plasticity_sharpness: float = 1.0  # Exponent on entropy→gain mapping (>1 = sharper gate)

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
# Mixed Phase / Belief Gate Experiment Control
# -----------------------------------------------------------------------------

# Set to True to activate cue-driven belief updating and entropy-gated plasticity.
# When True, all model runs use belief state [1-b, b] as the context vector,
# and gradient updates are scaled by m = 1 - H_norm(b). No explicit 'Mixed' label
# check is performed — gating emerges from the model's own epistemic uncertainty.
ENABLE_CUE_BELIEF_GATE = True

# Mixed phase injection parameters
N_MIXED_BLOCKS = 2   # Number of Mixed-cue blocks to insert after criterion is met
MIXED_BLOCK_SIZE = 10  # Sequences per block (1 block = 10 sequences = 80 items)

# Fixed 2-D context basis vectors.
# Pure trials always receive strict one-hot corners of context space;
# ambiguity (Mixed) is expressed via a continuous mixture, never a 3rd dimension.
CTX_MAZE   = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # shape (1, 2)
CTX_MELODY = torch.tensor([[0.0, 1.0]], dtype=torch.float32)  # shape (1, 2)

# Policy for unexpected context labels.
# 'error'         -> raise ValueError immediately (default; preferred for scientific runs).
# 'warn_as_mixed' -> emit a warning and treat as ambiguous cue.
UNKNOWN_LABEL_POLICY: str = 'error'

# How to report transfer_loss on Mixed rows (no unambiguous 'other' context exists).
# 'skip' -> transfer_loss = NaN (recommended; diagnostics only apply to pure trials).
# 'both' -> log transfer_maze_loss and transfer_melody_loss (extension, not yet implemented).
LOG_TRANSFER_ON_MIXED: str = 'skip'

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


def update_belief_logit(b: float, e: float, belief_lambda: float, belief_beta: float) -> float:
    """
    Update context belief b = P(Melody) via leaky log-odds integration.

    Operates in logit (log-odds) space for numerical stability:
        logit(b') = lambda * logit(b) + beta * e

    Cognitive rationale: this is a bounded, memoryful accumulator of cue evidence.
    - lambda (0 < lambda < 1): controls how quickly old evidence decays.
      High lambda = longer memory, slow context switches.
    - beta: controls how strongly each cue shifts the estimate.
      High beta = rapid, confident context identification.
    - e in {-1, 0, +1}: directional evidence (Maze=-1, Mixed=0, Melody=+1).

    The update is identical in structure to a simple recurrent filter on log-odds,
    equivalent to a running Bayesian estimate under a geometric forgetting prior.
    """
    b = float(np.clip(b, 1e-6, 1.0 - 1e-6))
    logit_b = np.log(b / (1.0 - b))
    logit_b = belief_lambda * logit_b + belief_beta * e
    # Sigmoid maps back to probability
    return float(1.0 / (1.0 + np.exp(-logit_b)))


def plasticity_from_belief(b: float, floor: float = 0.0, sharpness: float = 1.0) -> float:
    """
    Compute plasticity gain m from belief entropy.

    m = 1 - H_norm(b),  where H_norm is normalised binary entropy in [0, 1].

    Cognitive rationale: when the network is confident about context (b near 0 or 1),
    entropy is low and learning proceeds at full rate.  When context is ambiguous
    (b near 0.5), entropy is maximal and weight updates are suppressed.  This
    implements a form of epistemic-certainty–gated synaptic plasticity, analogous
    to neuromodulatory gating of consolidation (e.g., acetylcholine / uncertainty
    modulation of hippocampal encoding).

    Crucially, no explicit 'Mixed' label is checked.  Ambiguity arises naturally
    from the belief state, so the same mechanism handles novel or noisy contexts.

    Args:
        b:         Current belief P(Melody)
        floor:     Minimum plasticity (prevents full suppression if desired)
        sharpness: Exponent > 1 sharpens the gain curve (more binary gating)
    Returns:
        m in [floor, 1]
    """
    b = float(np.clip(b, 1e-6, 1.0 - 1e-6))
    H = -(b * np.log(b) + (1.0 - b) * np.log(1.0 - b))  # binary entropy in nats
    H_norm = H / np.log(2.0)                              # normalise to [0, 1]
    m = 1.0 - H_norm
    m = max(m, floor)
    m = m ** sharpness
    return float(m)


def parse_context_label(label: str) -> tuple:
    """
    Normalise a raw context label and return (ctx_type, is_ambiguous).

    ctx_type is one of {'maze', 'melody', 'mixed'}.
    is_ambiguous is True only for 'mixed' (or anything mapped to it by policy).

    Unknown labels either raise ValueError (UNKNOWN_LABEL_POLICY='error', default)
    or are warned and treated as 'mixed' (UNKNOWN_LABEL_POLICY='warn_as_mixed').
    This prevents unexpected labels from silently contaminating belief states.
    """
    norm = label.strip().lower()
    if norm in ('maze', 'melody', 'mixed'):
        return norm, (norm == 'mixed')
    # Unknown label — apply policy
    if UNKNOWN_LABEL_POLICY == 'warn_as_mixed':
        import warnings as _w
        _w.warn(
            f"Unknown context label '{label}' treated as Mixed (ambiguous cue). "
            f"Set UNKNOWN_LABEL_POLICY='error' to raise instead."
        )
        return 'mixed', True
    raise ValueError(
        f"Unknown context label '{label}'. "
        f"Expected one of: maze, melody, mixed. "
        f"Set UNKNOWN_LABEL_POLICY='warn_as_mixed' to continue instead of raising."
    )


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
                               context_encodings: Dict[str, torch.Tensor] = None) -> Dict:
    """
    Simulate one participant's sequence-learning experiment.

    Context representation (ENABLE_CUE_BELIEF_GATE=True):

      Pure Maze/Melody trials
        SRN input   = strict one-hot [1,0] or [0,1]  (identical to baseline SRN)
        Belief b    snaps to 0.0 (Maze) or 1.0 (Melody)  — certainty under clear cue
        Plasticity  m = 1.0  — normal learning
        transfer_loss computed against strict opposite one-hot

      Mixed trials
        Belief b    updated with e=0 → lambda decay pulls b → 0.5
        SRN input   = [1-b, b]  ≈ [0.5, 0.5]  (blended context signal)
        Plasticity  m = 1 - H_norm(b) ≈ 0.0  (suppressed by belief entropy)
        transfer_loss = NaN  (no unambiguous opposite context)

    Cognitive rationale:
      Discrete one-hots preserve the baseline context separation on pure trials.
      Mixed trials introduce genuine cue ambiguity at the *input* level and gate
      consolidation via epistemic uncertainty — without any explicit label check.
      This instantiates FTT: context-specific FTT knowledge is learned during
      pure blocks; Mixed probes blended expression without consolidating it.
    """
    # Fixed 2D basis — never changes even if dataset has more groups
    context_dim = 2

    model = ContextSequenceRNN(vocab_size, context_dim=context_dim, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    results = {
        'block_id': [],             # Block number (new_trialNumber)
        'sequence_in_block': [],    # Sequence index within block
        'global_step': [],          # Continuous counter
        'context': [],              # Raw group label from data
        'loss': [],                 # CE loss under current context input
        'transfer_loss': [],        # CE under strict opposite one-hot; NaN for Mixed
        'rt': [],                   # Human mean RT (NaN on Mixed rows)
        'training_active': [],      # Whether weight update ran this step
        'belief_b': [],             # 0 on Maze, 1 on Melody, ~0.5 on Mixed
        'plasticity_m': [],         # Gain: 1.0 on pure, low on Mixed
        'cue_evidence': [],         # e in {-1, 0, +1}
        'is_mixed': [],             # True for Mixed rows
        'is_ambiguous_cue': [],     # True for Mixed / unknown-as-mixed
        'context_input_maze': [],   # Actual context[0] fed to SRN
        'context_input_melody': [], # Actual context[1] fed to SRN
    }

    b = 0.5  # initialise at maximum uncertainty
    global_step = 0

    for _, row in df_participant.iterrows():
        block_id = row['new_trialNumber']
        seq_num  = row.get('sequenceNumber', 0)
        raw_label = str(row['group']).strip()
        rt        = row['meanRT']
        global_step += 1

        # ------------------------------------------------------------------
        # 2. Context input, belief snapshot, and plasticity gate
        # ------------------------------------------------------------------
        _norm_label = raw_label.strip().lower()
        _use_belief_gate = ENABLE_CUE_BELIEF_GATE and (_norm_label in ('maze', 'melody', 'mixed'))

        if _use_belief_gate:
            # Label parsing only needed for belief-gate path
            ctx_type, is_ambiguous = parse_context_label(raw_label)
            is_mixed = (ctx_type == 'mixed')
            if ctx_type == 'maze':
                # Unambiguous Maze: strict one-hot input, certain belief, full plasticity.
                b = 0.0
                curr_context_vec     = CTX_MAZE.clone()
                transfer_context_vec = CTX_MELODY.clone()  # baseline definition
                m = 1.0
                e = -1.0

            elif ctx_type == 'melody':
                # Unambiguous Melody: strict one-hot input, certain belief, full plasticity.
                b = 1.0
                curr_context_vec     = CTX_MELODY.clone()
                transfer_context_vec = CTX_MAZE.clone()    # baseline definition
                m = 1.0
                e = +1.0

            else:  # mixed / ambiguous
                # e=0 → no directional push; lambda decay draws b toward 0.5 over trials.
                e = 0.0
                b = update_belief_logit(b, e, config.belief_lambda, config.belief_beta)
                curr_context_vec     = torch.tensor([[1.0 - b, b]], dtype=torch.float32)
                transfer_context_vec = None   # no meaningful opposite one-hot
                m = plasticity_from_belief(b, config.plasticity_floor,
                                           config.plasticity_sharpness)
        else:
            # Legacy path (ENABLE_CUE_BELIEF_GATE=False): one-hot lookup, m=1 always.
            # parse_context_label is NOT called here — arbitrary group labels valid.
            ctx_lower = raw_label.lower()
            is_mixed      = (ctx_lower == 'mixed')
            is_ambiguous  = is_mixed
            if context_encodings is None:
                context_encodings = {
                    'Maze':   torch.tensor([1., 0.]),
                    'Melody': torch.tensor([0., 1.])
                }
            lookup   = raw_label if raw_label in context_encodings \
                       else list(context_encodings.keys())[0]
            curr_context_vec = context_encodings[lookup].unsqueeze(0)
            all_ctx  = list(context_encodings.keys())
            others   = [c for c in all_ctx if c != lookup]
            other    = others[0] if others else lookup
            transfer_context_vec = context_encodings[other].unsqueeze(0)
            m, e = 1.0, 0.0

        # ------------------------------------------------------------------
        # 3. Control condition (single_context_mode)
        # ------------------------------------------------------------------
        train_active = True
        if single_context_mode and raw_label != train_context:
            train_active = False

        # ------------------------------------------------------------------
        # 4. Evaluate before weight update
        # ------------------------------------------------------------------
        inputs  = sequence_to_gaussian(fixed_sequence[:-1], vocab_size, config.sigma)
        targets = torch.tensor(fixed_sequence[1:], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            logits, _ = model(inputs, curr_context_vec)
            loss_val  = criterion(logits.squeeze(0), targets).item()

            # Transfer loss: strict opposite one-hot (matches baseline definition).
            # NaN on Mixed rows — no unambiguous complement exists.
            # NEVER used to derive m or make any training decision.
            if transfer_context_vec is not None:
                logits_tx, _ = model(inputs, transfer_context_vec)
                loss_tx = criterion(logits_tx.squeeze(0), targets).item()
            else:
                loss_tx = float('nan')

        # ------------------------------------------------------------------
        # 5. Log
        # ------------------------------------------------------------------
        results['block_id'].append(block_id)
        results['sequence_in_block'].append(seq_num)
        results['global_step'].append(global_step)
        results['context'].append(raw_label)
        results['loss'].append(loss_val)
        results['transfer_loss'].append(loss_tx)
        results['rt'].append(rt)
        results['training_active'].append(train_active)
        results['belief_b'].append(b)
        results['plasticity_m'].append(m)
        results['cue_evidence'].append(e)
        results['is_mixed'].append(is_mixed)
        results['is_ambiguous_cue'].append(is_ambiguous)
        results['context_input_maze'].append(float(curr_context_vec[0, 0].item()))
        results['context_input_melody'].append(float(curr_context_vec[0, 1].item()))

        # ------------------------------------------------------------------
        # 6. Train — entropy-gated gradient scaling
        # ------------------------------------------------------------------
        # On pure trials m=1.0, so mul_(1.0) is a no-op (no overhead).
        # On Mixed trials m≈0 suppresses weight updates algorithmically.
        if train_active:
            model.train()
            for t in range(len(fixed_sequence) - 1):
                token_idx      = fixed_sequence[t]
                next_token_idx = fixed_sequence[t + 1]

                step_input_vec = gaussian_vector(token_idx, vocab_size, config.sigma)
                step_input  = torch.tensor(step_input_vec, dtype=torch.float32)\
                                   .unsqueeze(0).unsqueeze(0)   # (1, 1, vocab)
                step_target = torch.tensor([next_token_idx], dtype=torch.long)

                optimizer.zero_grad()
                step_logits, _ = model(step_input, curr_context_vec)
                loss = criterion(step_logits.squeeze(0), step_target)
                loss.backward()

                if ENABLE_CUE_BELIEF_GATE and m < 1.0:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(m)

                optimizer.step()

    return results, model

# -----------------------------------------------------------------------------
# 4. Mechanistic Analysis (R4 & R5)
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

def inject_mixed_phase(df_participant: pd.DataFrame,
                       n_mixed_blocks: int = N_MIXED_BLOCKS,
                       block_size: int = MIXED_BLOCK_SIZE) -> pd.DataFrame:
    """
    Insert Mixed-cue trial blocks into a participant's sequence of rows
    after the network has had the opportunity to form distinct context
    representations (>= 2 Maze blocks AND >= 2 Melody blocks seen).

    Cognitive rationale: Mixed cues are epistemically meaningful only once
    the learner has established separate context representations.  Injecting
    Mixed trials too early would be confounded with initial acquisition.
    This mirrors experimental designs where exposure to individual contexts
    precedes any ambiguous-cue probes.

    Block structure:
        prefix: original rows up to (and including) the criterion row
        mixed : n_mixed_blocks x block_size new rows with group='Mixed'
        suffix: remaining original rows (block IDs shifted to stay monotonic)

    Mixed rows carry meanRT=NaN (no human RT available; excluded from RT plots).

    Args:
        df_participant:  One participant's sorted sequence rows.
        n_mixed_blocks:  How many Mixed blocks to insert.
        block_size:      Sequences per Mixed block.

    Returns:
        Augmented DataFrame with mixed rows inserted and block IDs renumbered.
    """
    df_sorted = df_participant.sort_values(
        by=['new_trialNumber', 'sequenceNumber']
    ).reset_index(drop=True)

    groups_norm = df_sorted['group'].astype(str).str.strip().str.lower()

    maze_blocks: set = set()
    melody_blocks: set = set()
    insert_after = None

    for idx in range(len(df_sorted)):
        g = groups_norm.iloc[idx]
        b_id = df_sorted['new_trialNumber'].iloc[idx]
        if g == 'maze':
            maze_blocks.add(b_id)
        elif g == 'melody':
            melody_blocks.add(b_id)

        if len(maze_blocks) >= 2 and len(melody_blocks) >= 2:
            insert_after = idx
            break

    if insert_after is None:
        # Criterion never met: single-context dataset or insufficient data
        return df_participant.copy()

    prefix = df_sorted.iloc[:insert_after + 1].copy()
    suffix = df_sorted.iloc[insert_after + 1:].copy()

    prefix_max_block = int(prefix['new_trialNumber'].max())
    pid = df_sorted['ID'].iloc[0] if 'ID' in df_sorted.columns else None

    # Build Mixed rows
    mixed_rows = []
    for mb in range(n_mixed_blocks):
        block_id = prefix_max_block + 1 + mb
        for seq in range(1, block_size + 1):
            row_dict = {col: np.nan for col in df_sorted.columns}
            row_dict['new_trialNumber'] = block_id
            row_dict['sequenceNumber'] = seq
            row_dict['group'] = 'Mixed'
            row_dict['meanRT'] = np.nan
            if pid is not None:
                row_dict['ID'] = pid
            mixed_rows.append(row_dict)

    df_mixed = pd.DataFrame(mixed_rows, columns=df_sorted.columns)

    # Shift suffix block IDs upward to keep ordering monotonic after insertion
    if not suffix.empty:
        suffix = suffix.copy()
        suffix['new_trialNumber'] = suffix['new_trialNumber'] + n_mixed_blocks

    result = pd.concat([prefix, df_mixed, suffix], ignore_index=True)
    return result


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
    available_groups = sorted(df['group'].dropna().unique().tolist())
    if not available_groups:
        print("   ⚠️ No 'group' column or empty groups found. Defaulting to 'Default'.")
        df['group'] = 'Default'
        available_groups = ['Default']
        
    print(f"   Identified Contexts: {available_groups}")

    # ------------------------------------------------------------------
    # Unknown-label validation (context-switching datasets only)
    # Only applies when the dataset already contains at least one recognised
    # FTT context label (maze / melody / mixed).  Pure single-context datasets
    # (e.g. Letters, Digits) are silently skipped — their labels are handled
    # by the legacy one-hot path and never reach parse_context_label().
    # ------------------------------------------------------------------
    if ENABLE_CUE_BELIEF_GATE:
        allowed_labels = {'maze', 'melody', 'mixed'}
        raw_labels     = df['group'].dropna().astype(str).str.strip().str.lower().unique()
        recognised_present = set(raw_labels) & allowed_labels
        if recognised_present:
            # Only validate when *some* FTT labels are present; flag any extras.
            unknown_labels = set(raw_labels) - allowed_labels
            if unknown_labels:
                counts = (
                    df['group'].astype(str).str.strip().str.lower()
                      .value_counts()
                      .loc[lambda s: s.index.isin(unknown_labels)]
                      .to_dict()
                )
                msg = (
                    f"Unknown group label(s) found alongside FTT contexts: {counts}. "
                    f"Expected labels are {allowed_labels}. "
                    f"Set UNKNOWN_LABEL_POLICY to 'warn_as_mixed' to treat them as Mixed."
                )
                if UNKNOWN_LABEL_POLICY == 'error':
                    raise ValueError(msg)
                else:
                    import warnings
                    warnings.warn(msg, UserWarning, stacklevel=2)

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

    # When belief gate is on, mechanistic analysis uses the fixed 2D Maze/Melody
    # encoding basis (same as run_participant_experiment), not the potentially
    # higher-dimensional encodings built from the raw dataset groups.
    mech_context_encodings = context_encodings
    if ENABLE_CUE_BELIEF_GATE:
        mech_context_encodings = {
            'Maze':   torch.tensor([1., 0.]),
            'Melody': torch.tensor([0., 1.])
        }

    for config in configs:
        is_control = (config.name == 'Control (Rich)')
        mode_str = "Single-Context" if is_control else "Standard"
        print(f"\n🚀 Running {config.name} Regime (dim={config.hidden_dim}, alpha={config.context_alpha}) [{mode_str}]...")
        
        for i, pid in enumerate(valid_pids):
            # Sort by Block (Trial) then Sequence Index
            p_data = df[df['ID'] == pid].sort_values(by=["new_trialNumber", "sequenceNumber"])

            # --- Mixed Phase Injection ---
            # Insert Mixed-cue blocks once >= 2 blocks of each context have been seen.
            # Skip injection for single-context datasets (criterion never met).
            # The belief gate in run_participant_experiment handles ambiguity gracefully
            # regardless: suppression emerges from entropy, not from an explicit label check.
            if ENABLE_CUE_BELIEF_GATE and len(available_groups) >= 2:
                p_data = inject_mixed_phase(p_data,
                                            n_mixed_blocks=N_MIXED_BLOCKS,
                                            block_size=MIXED_BLOCK_SIZE)
            
            # Pass single_context_mode flag if Control
            # Ensure we train on a context that actually exists in the data
            target_ctx = dominant_context if is_control else 'Maze'
            if target_ctx not in available_groups:
                target_ctx = dominant_context

            res, model = run_participant_experiment(p_data, config,
                                                    single_context_mode=is_control,
                                                    train_context=target_ctx,
                                                    context_encodings=mech_context_encodings)
            
            # Convert to DF and tag
            res_df = pd.DataFrame(res)
            res_df['model_type'] = config.name
            res_df['participant_id'] = pid
            all_results.append(res_df)
            
            # R4: Run Mechanistic Analysis on trained model
            if not is_control:
                mech_stats = run_mechanistic_analysis(model, config=config,
                                                      active_contexts=['Maze', 'Melody'],
                                                      context_encodings=mech_context_encodings)
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
    
    if ENABLE_CUE_BELIEF_GATE:
        print("   Generating plasticity/belief dynamics plot...")
        plot_plasticity_dynamics(final_df, output_dir)

    if mechanistic_results:
        print("   Generating mechanistic diagnostic plots...")
        plot_mechanistic_summary(mechanistic_results, output_dir)
        
    # R3: Run Parameter Sweep
    run_parameter_sweep(df, valid_pids, output_dir, context_encodings)

    print(f"\n✅ Analysis Complete for {data_filename}. Results in {output_dir}")

def run_parameter_sweep(df_all: pd.DataFrame, valid_pids: List[str], output_dir: Path, context_encodings: Dict):
    """
    R3: Runs a grid search over Alpha (Context Gating) and Sigma (Input Scale).
    Metric: Generalization Gap (Transfer Loss - Training Loss) averaged over the last block.
    """
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    scales = [0.02, 0.1, 0.3, 0.5]
    
    heatmap_data = []

    print(f"\n🔍 Starting Parameter Sweep (Grid: {len(alphas)}x{len(scales)})...")
    
    for a in alphas:
        for s in scales:
            # Create temp config
            # Use 'Rich' dim (Standard Capacity) for the sweep
            cfg = ModelConfig(name='Sweep', hidden_dim=6, init_scale=0.5, context_alpha=a, sigma=s, learning_rate=0.005)
            
            total_gap = 0
            count = 0
            
            # Run on subset of participants to save time (e.g. first 3)
            # or all if fast enough. 1 year of data is small, usually.
            # Let's use first 5 representative PIDs
            subset_pids = valid_pids[:5]
            
            for pid in subset_pids:
                p_data = df_all[df_all['ID'] == pid].sort_values(by=["new_trialNumber", "sequenceNumber"])
                res, _ = run_participant_experiment(p_data, cfg, context_encodings=context_encodings)
                
                # Calculate metric: Mean Generalization Gap in Last Block (Trial 12)
                res_df = pd.DataFrame(res)
                last_block = res_df[res_df['block_id'] == 12]
                if not last_block.empty:
                    gap = (last_block['transfer_loss'] - last_block['loss']).mean()
                    total_gap += gap
                    count += 1
            
            avg_gap = total_gap / count if count > 0 else 0
            heatmap_data.append({'Alpha': a, 'Sigma': s, 'Gap': avg_gap})
    
    # Convert to DataFrame for plotting
    df_heatmap = pd.DataFrame(heatmap_data)
    
    # Pivot for heatmap: Alpha (rows) x Sigma (columns)
    df_pivot = df_heatmap.pivot_table(index='Alpha', columns='Sigma', values='Gap')
    
    # 4. Visualization
    print("   Generating parameter sweep heatmap...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, cmap="coolwarm", annot=True, fmt=".3f", linewidths=.5)
    plt.title("Generalization Gap: Parameter Sweep (Alpha x Sigma)", fontsize=16)
    plt.xlabel("Sigma (Input Scale)")
    plt.ylabel("Alpha (Context Gating)")
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_sweep_heatmap.png")
    plt.close()
    
    print("   Saving sweep results...")
    df_heatmap.to_csv(output_dir / "rich_poor_parameter_sweep_results.csv", index=False)

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


def plot_plasticity_dynamics(final_df: pd.DataFrame, output_dir: Path):
    """
    Plots plasticity gain (m) and belief b = P(Melody) over cumulative sequence
    index, with a vertical line at the onset of Mixed trials.

    Interpretation:
    - m (filled area): how much the SRN is allowed to learn at each step.
      Near 1 on pure context trials; drops toward 0 during Mixed trials as
      belief entropy rises.
    - b (dashed): the model's running estimate of P(Melody).
      Reflects accumulated cue evidence; Mixed cues hold it near 0.5.
    - Orange dotted line: first Mixed trial seen (Mixed onset).

    One subplot per model type; NaN rt rows (Mixed) are handled by aggregation.
    """
    if 'plasticity_m' not in final_df.columns:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'Rich': 'darkgreen', 'Poor': 'firebrick', 'Control (Rich)': 'royalblue'}
    model_types = final_df['model_type'].unique()
    n_types = len(model_types)

    fig, axes = plt.subplots(n_types, 1, figsize=(14, 4 * n_types), sharex=False)
    if n_types == 1:
        axes = [axes]

    for ax, m_type in zip(axes, model_types):
        subset = final_df[final_df['model_type'] == m_type]
        agg = (subset.groupby('global_step')[['plasticity_m', 'belief_b', 'is_mixed']]
                     .mean()
                     .reset_index())

        color = colors.get(m_type, 'steelblue')

        # Plasticity gain
        ax.fill_between(agg['global_step'], agg['plasticity_m'],
                        alpha=0.20, color=color)
        ax.plot(agg['global_step'], agg['plasticity_m'],
                color=color, linewidth=1.8, label='Plasticity gain (m)')

        # Belief
        ax.plot(agg['global_step'], agg['belief_b'],
                color='gray', linestyle='--', linewidth=1.2, alpha=0.85,
                label='Belief  b = P(Melody)')

        # Mixed onset marker
        mixed_steps = agg.loc[agg['is_mixed'] > 0.5, 'global_step']
        if not mixed_steps.empty:
            ax.axvline(mixed_steps.min(), color='darkorange', linestyle=':',
                       linewidth=2.0, label='Mixed onset')

        ax.set_title(f"Plasticity Gate & Belief: {m_type}", fontsize=13)
        ax.set_ylabel("Value  (0 \u2013 1)", fontsize=11)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(0.5, linestyle='--', color='lightgray', alpha=0.6, linewidth=0.8)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        sns.despine(ax=ax)

    axes[-1].set_xlabel("Cumulative Sequence Index", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "plasticity_belief_dynamics.png", dpi=300)
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
