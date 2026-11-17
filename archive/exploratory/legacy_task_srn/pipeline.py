"""
srn.pipeline
============
High-level orchestration of the full Rich / Poor analysis pipeline.

Design principle
----------------
Every participant × config pair is trained exactly **once**: the model follows
the participant's empirical trial sequence first (collecting per-step learning
curves and CE/accuracy threshold snapshots), then training continues from the
*same model state* until asymptote (collecting weight / gradient / collapse
dynamics at each epoch checkpoint).

Data that was previously computed in three separate passes —
  1. participant timeline
  2. fresh-model asymptote run
  3. matched-performance re-run
— all derive from this single unified run.

Reproducibility
---------------
Each participant index ``i`` receives seed ``GLOBAL_SEED + i`` before model
initialisation, guaranteeing identical weight initialisations across dataset
runs and across machines.

Public API
----------
process_dataset(data_path, output_dir)  — full pipeline for one CSV file
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA as _PCA

from .config import (
    ModelConfig, VOCAB_SIZE, FIXED_SEQUENCE,
    RICH_CONFIG, POOR_CONFIG, LAZY_RICH_CTX_CONFIG, RICH_LAZY_CTX_CONFIG,
    PERF_THRESHOLDS, ACC_THRESHOLDS,
    GLOBAL_SEED,
)
from .metrics import (
    run_mechanistic_analysis,
    compute_collapse_metrics,
    compute_weight_norms,
    run_context_perturbation_analysis,
    compute_gradient_alignment,
    compute_output_divergence,
    run_context_ablation_test,
    collect_checkpoint_hidden_states,
    compute_transition_cost,
)
from .training import (
    run_participant_to_asymptote,
)
from .sweeps import (
    run_hidden_size_sweep,
    run_learning_rate_sweep,
    run_init_scale_sweep,
    run_scale_lr_sweep,
    compute_model_summary_metrics,
)
from .plotting import (
    plot_mechanistic_summary,
    plot_pca_geometry,
    plot_rich_poor_comparison,
    plot_accuracy_entropy_curves,
    plot_collapse_dynamics,
    plot_normalized_collapse,
    plot_phase_portrait,
    plot_perturbation_dynamics,
    plot_weight_gradient_dynamics,
    plot_shared_pca_panels,
    plot_dual_decoding_dynamics,
    plot_matched_geometry,
    plot_asymptote_geometry,
    plot_hidden_size_sweep,
    plot_learning_rate_sweep,
    plot_collapse_driver_dynamics,
    _plot_matched_geometry_generic,
    DISPLAY_NAMES,
)


# ---------------------------------------------------------------------------
# Column-inference helpers
# ---------------------------------------------------------------------------

def _infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *df* (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _build_context_encodings(
    contexts: List[str],
) -> Dict[str, torch.Tensor]:
    """One-hot context encodings ordered by the supplied context list."""
    n = len(contexts)
    return {
        ctx: torch.eye(n)[i]
        for i, ctx in enumerate(contexts)
    }


def _snap_step(key: str) -> int:
    """Extract the numeric step from a snapshot key (e.g. 'step_40' → 40)."""
    try:
        return int(key.split("_")[-1])
    except ValueError:
        return 0


def _snap_type(key: str) -> str:
    """Extract the snapshot-type prefix (e.g. 'ce_1.0' → 'ce')."""
    return key.split("_")[0]


# ---------------------------------------------------------------------------
# Unified training phase
# ---------------------------------------------------------------------------

def _run_unified_phase(
    df_raw:            pd.DataFrame,
    pid_col:           str,
    context_col:       str,
    participants:      List[str],
    configs:           Dict[str, ModelConfig],
    context_encodings: Dict[str, torch.Tensor],
    global_seed:       int = GLOBAL_SEED,
) -> Tuple[
    pd.DataFrame,   # df_all             — per-step learning-curve rows
    List[Dict],     # asymptote_mechs    — mechanistic analysis after asymptote
    List[Dict],     # mech_stats_list    — mechanistic analysis after participant timeline
    List[Dict],     # collapse_rows      — collapse metrics per CE/acc snapshot
    List[Dict],     # perturb_rows       — perturbation analysis per snapshot
    List[Dict],     # driver_rows        — gradient + JS divergence per snapshot
    List[Dict],     # ablation_rows      — context ablation per snapshot
    List[Dict],     # wg_rows            — weight / gradient norms per asymptote epoch
    List[Dict],     # collapse_asym_rows — collapse metrics per asymptote epoch
    List[Dict],     # shared_pca_rows    — raw hidden states for shared PCA
    List[Dict],     # matched_rows       — matched-geometry rows (from CE snapshots)
]:
    """Run one combined participant-timeline + asymptote pass for every pid × config.

    Each participant index ``i`` (0-based, sorted order) receives seed
    ``global_seed + i``.  Within a participant, every config uses the *same*
    seed so that differences between Rich and Poor reflect only init_scale /
    architecture, not random-seed variation.

    All downstream analysis data — including the matched-performance geometry
    that was previously computed in a third separate pass — comes directly from
    this single unified run.
    """
    all_step_rows:      List[Dict] = []
    asymptote_mechs:    List[Dict] = []
    mech_stats_list:    List[Dict] = []
    collapse_rows:      List[Dict] = []
    perturb_rows:       List[Dict] = []
    driver_rows:        List[Dict] = []
    ablation_rows:      List[Dict] = []
    wg_rows:            List[Dict] = []
    collapse_asym_rows: List[Dict] = []
    shared_pca_rows:    List[Dict] = []
    matched_rows:       List[Dict] = []

    for pid_idx, pid in enumerate(participants):
        seed = global_seed + pid_idx
        p_df = df_raw[df_raw[pid_col] == pid].reset_index(drop=True)

        for cfg_name, cfg in configs.items():
            print(f"   🚀  {cfg_name} | participant {pid} | seed {seed}")

            (results, model_participant, best_model,
             snapshots, _snap_meta,
             loss_history, extra_info) = run_participant_to_asymptote(
                df_participant               = p_df,
                config                       = cfg,
                vocab_size                   = VOCAB_SIZE,
                fixed_sequence               = list(FIXED_SEQUENCE),
                context_encodings            = context_encodings,
                context_col                  = context_col,
                snapshot_loss_thresholds     = PERF_THRESHOLDS,
                snapshot_accuracy_thresholds = ACC_THRESHOLDS,
                step_interval                = 10,
                max_epochs                   = 500,   # fixed budget — no early stop
                tol                          = 1e-4,
                patience                     = 500,   # = max_epochs → never triggers
                epoch_interval               = 10,
                track_gradients              = True,
                seed                         = seed,
            )

            # ── per-step learning-curve rows ──────────────────────────────────
            n = len(results["global_step"])
            for i in range(n):
                row = {k: v[i] for k, v in results.items()}
                row["model_type"]     = cfg_name
                row["participant_id"] = pid
                all_step_rows.append(row)

            # ── mechanistic analysis — post-participant-timeline state ─────────
            mech = run_mechanistic_analysis(
                model_participant, cfg, context_encodings=context_encodings,
            )
            mech["model_type"]     = cfg_name
            mech["participant_id"] = pid
            mech_stats_list.append(mech)

            # ── mechanistic analysis — post-asymptote state ───────────────────
            mech_asym = run_mechanistic_analysis(
                best_model, cfg, context_encodings=context_encodings,
            )
            mech_asym["model_type"]     = cfg_name
            mech_asym["participant_id"] = pid
            mech_asym["source"]         = "asymptote"
            asymptote_mechs.append(mech_asym)

            # ── snapshot-level geometry probes (step-interval timeline snaps) ──
            # Only step_* keys are produced here (CE/acc snaps now live in
            # extra_info["ce_snapshots"] from the asymptote phase).
            for snap_key, snap_model in snapshots.items():
                if not snap_key.startswith("step_"):
                    continue
                step_val = _snap_step(snap_key)

                col_met = compute_collapse_metrics(snap_model, cfg, context_encodings)
                col_met.update({
                    "model_type":       cfg_name,
                    "participant_id":   pid,
                    "snapshot_key":     snap_key,
                    "snapshot_type":    "step",
                    "step_or_epoch":    step_val,
                    "checkpoint_value": step_val,
                })
                collapse_rows.append(col_met)

                for pr in run_context_perturbation_analysis(snap_model, cfg, context_encodings):
                    pr.update({
                        "model_type":       cfg_name,
                        "participant_id":   pid,
                        "checkpoint_value": step_val,
                    })
                    perturb_rows.append(pr)

                ga = compute_gradient_alignment(snap_model, cfg, context_encodings)
                od = compute_output_divergence(snap_model, cfg, context_encodings)
                driver_rows.append({
                    "model_type":         cfg_name,
                    "participant_id":     pid,
                    "checkpoint_value":   step_val,
                    "gradient_alignment": ga,
                    **od,
                })

                for ar in run_context_ablation_test(snap_model, cfg, context_encodings):
                    ar.update({
                        "model_type":       cfg_name,
                        "participant_id":   pid,
                        "checkpoint_value": step_val,
                    })
                    ablation_rows.append(ar)

            # ── matched-performance geometry (CE/acc snapshots from asymptote) ─
            # Captured in extra_info["ce_snapshots"] by run_to_asymptote.
            # Training runs to convergence so all thresholds 1.4 → 0.1 are
            # always covered for both Rich and Poor.
            for snap_key, snap_model in extra_info.get("ce_snapshots", {}).items():
                if snap_key.startswith("ce_"):
                    thr        = float(snap_key[3:])
                    mech_match = run_mechanistic_analysis(
                        snap_model, cfg, context_encodings=context_encodings,
                    )
                    matched_row = {
                        "model_type":     cfg_name,
                        "participant_id": pid,
                        "ce_threshold":   thr,
                    }
                    matched_row.update({
                        k: v for k, v in mech_match.items()
                        if not isinstance(v, np.ndarray)
                    })
                    matched_rows.append(matched_row)

            # ── asymptote epoch data ──────────────────────────────────────────
            epoch_snaps    = extra_info.get("epoch_snapshots",  {})
            acc_hist       = extra_info.get("accuracy_history", {})
            grad_hist_list = extra_info.get("grad_history",     [])
            grad_by_ep: Dict[int, Dict] = {
                int(g["epoch"] + 1): g for g in grad_hist_list
            }

            for ep in sorted(epoch_snaps.keys()):
                snap_m = epoch_snaps[ep]

                wn = compute_weight_norms(snap_m)
                wg_rows.append({
                    "model_type":     cfg_name,
                    "participant_id": pid,
                    "epoch":          ep,
                    "loss": (loss_history[ep - 1]
                             if ep <= len(loss_history) else np.nan),
                    "accuracy": acc_hist.get(ep, np.nan),
                    "weight_ratio_cx_in": wn.get("context_input_ratio", np.nan),
                    **wn,
                    **{k: grad_by_ep.get(ep, {}).get(k, np.nan)
                       for k in ("grad_context", "grad_recurrent", "grad_output")},
                })

                col_met = compute_collapse_metrics(snap_m, cfg, context_encodings)
                wn_cm = compute_weight_norms(snap_m)
                col_met.update({
                    "model_type":              cfg_name,
                    "participant_id":          pid,
                    "checkpoint_type":         "asymptote_epoch",
                    "step_or_epoch":           ep,
                    "checkpoint_value":        ep,
                    "loss": (loss_history[ep - 1]
                             if ep <= len(loss_history) else np.nan),
                    "accuracy":                acc_hist.get(ep, np.nan),
                    "weight_ratio_cx_in":      wn_cm.get("context_input_ratio",     np.nan),
                    "norm_input_weights":      wn_cm.get("norm_input_weights",      np.nan),
                    "norm_context_weights":    wn_cm.get("norm_context_weights",    np.nan),
                    "norm_recurrent_weights":  wn_cm.get("norm_recurrent_weights",  np.nan),
                    "norm_output_weights":     wn_cm.get("norm_output_weights",     np.nan),
                })
                collapse_asym_rows.append(col_met)

                h_all       = collect_checkpoint_hidden_states(snap_m, cfg, context_encodings)
                seq_per_ctx = h_all.shape[0] // len(context_encodings)
                for ci, ctx_name in enumerate(context_encodings.keys()):
                    for si in range(seq_per_ctx):
                        h = h_all[ci * seq_per_ctx + si]
                        shared_pca_rows.append({
                            "model_type":       cfg_name,
                            "participant_id":   pid,
                            "checkpoint_value": ep,
                            "context":          ctx_name,
                            "step_in_sequence": si,
                            **{f"h{j}": float(h[j]) for j in range(len(h))},
                        })

    df_all = pd.DataFrame(all_step_rows)
    return (df_all, asymptote_mechs, mech_stats_list,
            collapse_rows, perturb_rows, driver_rows, ablation_rows,
            wg_rows, collapse_asym_rows, shared_pca_rows, matched_rows)
    """Run Rich and Poor models on every participant's timeline.

    Returns
    -------
    df_all             : combined per-step results DataFrame
    mech_stats_list    : mechanistic analysis dicts (one per pid × config)
    collapse_rows      : collapse metrics per snapshot
    perturb_rows       : perturbation analysis per snapshot
    driver_rows        : collapse-driver (gradient alignment + JS div) per snapshot
    ablation_rows      : context ablation per snapshot
    """
    all_step_rows:  List[Dict] = []
    mech_stats_list: List[Dict] = []
    collapse_rows:   List[Dict] = []
    perturb_rows:    List[Dict] = []
    driver_rows:     List[Dict] = []
    ablation_rows:   List[Dict] = []

    for cfg_name, cfg in configs.items():
        print(f"   🚀  {cfg_name} — {len(participants)} participants …")
        for pid in participants:
            p_df = df_raw[df_raw[pid_col] == pid].reset_index(drop=True)

            results_dict, model, snapshots, _ = run_participant_experiment(
                df_participant              = p_df,
                config                     = cfg,
                context_encodings          = context_encodings,
                context_col                = context_col,
                snapshot_loss_thresholds   = PERF_THRESHOLDS,
                snapshot_accuracy_thresholds = ACC_THRESHOLDS,
                step_interval              = 10,
            )

            # Flatten dict-of-lists → list-of-dicts and tag
            n = len(results_dict["global_step"])
            for i in range(n):
                row = {k: v[i] for k, v in results_dict.items()}
                row["model_type"]    = cfg_name
                row["participant_id"] = pid
                all_step_rows.append(row)

            # Mechanistic analysis on final model
            mech = run_mechanistic_analysis(
                model              = model,
                config             = cfg,
                context_encodings  = context_encodings,
            )
            mech["model_type"]    = cfg_name
            mech["participant_id"] = pid
            mech_stats_list.append(mech)

            # Snapshot-level geometry probes
            for snap_key, snap_model in snapshots.items():
                step_val = _snap_step(snap_key)
                ctx_name = _snap_type(snap_key)

                col_met = compute_collapse_metrics(
                    snap_model, cfg, context_encodings,
                )
                wn_snap = compute_weight_norms(snap_model)
                col_met.update({
                    "model_type":              cfg_name,
                    "participant_id":          pid,
                    "snapshot_key":            snap_key,
                    "snapshot_type":           ctx_name,
                    "step_or_epoch":           step_val,
                    "checkpoint_value":        step_val,
                    "weight_ratio_cx_in":      wn_snap.get("context_input_ratio",     np.nan),
                    "norm_input_weights":      wn_snap.get("norm_input_weights",      np.nan),
                    "norm_context_weights":    wn_snap.get("norm_context_weights",    np.nan),
                    "norm_recurrent_weights":  wn_snap.get("norm_recurrent_weights",  np.nan),
                    "norm_output_weights":     wn_snap.get("norm_output_weights",     np.nan),
                })
                collapse_rows.append(col_met)

                for pr in run_context_perturbation_analysis(snap_model, cfg, context_encodings):
                    pr.update({
                        "model_type":       cfg_name,
                        "participant_id":   pid,
                        "checkpoint_value": step_val,
                    })
                    perturb_rows.append(pr)

                ga = compute_gradient_alignment(snap_model, cfg, context_encodings)
                od = compute_output_divergence(snap_model, cfg, context_encodings)
                driver_rows.append({
                    "model_type":         cfg_name,
                    "participant_id":     pid,
                    "checkpoint_value":   step_val,
                    "gradient_alignment": ga,
                    **od,
                })

                for ar in run_context_ablation_test(snap_model, cfg, context_encodings):
                    ar.update({
                        "model_type":       cfg_name,
                        "participant_id":   pid,
                        "checkpoint_value": step_val,
                    })
                    ablation_rows.append(ar)

                # CE-threshold snapshots → matched-performance geometry
                if snap_key.startswith("ce_"):
                    thr        = float(snap_key[3:])
                    mech_match = run_mechanistic_analysis(
                        snap_model, cfg, context_encodings=context_encodings,
                    )
                    matched_row = {
                        "model_type":     cfg_name,
                        "participant_id": pid,
                        "ce_threshold":   thr,
                    }
                    matched_row.update({
                        k: v for k, v in mech_match.items()
                        if not isinstance(v, np.ndarray)
                    })
                    matched_rows.append(matched_row)

            # ── asymptote epoch data ──────────────────────────────────────────
            epoch_snaps    = extra_info.get("epoch_snapshots",  {})
            acc_hist       = extra_info.get("accuracy_history", {})
            grad_hist_list = extra_info.get("grad_history",     [])
            grad_by_ep: Dict[int, Dict] = {
                int(g["epoch"] + 1): g for g in grad_hist_list
            }

            for ep in sorted(epoch_snaps.keys()):
                snap_m = epoch_snaps[ep]

                wn = compute_weight_norms(snap_m)
                wg_rows.append({
                    "model_type":     cfg_name,
                    "participant_id": pid,
                    "epoch":          ep,
                    "loss": (loss_history[ep - 1]
                             if ep <= len(loss_history) else np.nan),
                    "accuracy": acc_hist.get(ep, np.nan),
                    "weight_ratio_cx_in": wn.get("context_input_ratio", np.nan),
                    **wn,
                    **{k: grad_by_ep.get(ep, {}).get(k, np.nan)
                       for k in ("grad_context", "grad_recurrent", "grad_output")},
                })

                col_met = compute_collapse_metrics(snap_m, cfg, context_encodings)
                wn_cm = compute_weight_norms(snap_m)
                col_met.update({
                    "model_type":              cfg_name,
                    "participant_id":          pid,
                    "checkpoint_type":         "asymptote_epoch",
                    "step_or_epoch":           ep,
                    "checkpoint_value":        ep,
                    "loss": (loss_history[ep - 1]
                             if ep <= len(loss_history) else np.nan),
                    "accuracy":                acc_hist.get(ep, np.nan),
                    "weight_ratio_cx_in":      wn_cm.get("context_input_ratio",     np.nan),
                    "norm_input_weights":      wn_cm.get("norm_input_weights",      np.nan),
                    "norm_context_weights":    wn_cm.get("norm_context_weights",    np.nan),
                    "norm_recurrent_weights":  wn_cm.get("norm_recurrent_weights",  np.nan),
                    "norm_output_weights":     wn_cm.get("norm_output_weights",     np.nan),
                })
                collapse_asym_rows.append(col_met)

                h_all       = collect_checkpoint_hidden_states(snap_m, cfg, context_encodings)
                seq_per_ctx = h_all.shape[0] // len(context_encodings)
                for ci, ctx_name in enumerate(context_encodings.keys()):
                    for si in range(seq_per_ctx):
                        h = h_all[ci * seq_per_ctx + si]
                        shared_pca_rows.append({
                            "model_type":       cfg_name,
                            "participant_id":   pid,
                            "checkpoint_value": ep,
                            "context":          ctx_name,
                            "step_in_sequence": si,
                            **{f"h{j}": float(h[j]) for j in range(len(h))},
                        })

    df_all = pd.DataFrame(all_step_rows)
    return (df_all, asymptote_mechs, mech_stats_list,
            collapse_rows, perturb_rows, driver_rows, ablation_rows,
            wg_rows, collapse_asym_rows, shared_pca_rows, matched_rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_dataset(data_path: Path, output_dir: Optional[Path] = None) -> None:
    """Run the complete Rich-vs-Poor pipeline on a single CSV dataset.

    Each participant × config pair is trained exactly once: participant timeline
    first, then continued to asymptote from the same model state.  All analysis
    data (learning curves, geometry snapshots, asymptote dynamics, matched
    performance geometry) comes from this single unified pass.

    Parameters
    ----------
    data_path  : path to a participant-timeline CSV
    output_dir : where to write results (default: ``<script_dir>/results_<stem>``)
    """
    data_path = Path(data_path)
    if output_dir is None:
        output_dir = data_path.parent.parent / f"results_{data_path.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"📦  Dataset     : {data_path.name}")
    print(f"📁  Output      : {output_dir}")
    print(f"🌱  Global seed : {GLOBAL_SEED}")
    print(f"{'=' * 70}")

    # ── load & inspect ────────────────────────────────────────────────────────
    df_raw = pd.read_csv(data_path)
    print(f"   Rows: {len(df_raw):,}  |  Columns: {list(df_raw.columns)}")

    pid_col     = _infer_column(df_raw, ["ID", "participant_id", "pid", "subject"])
    # Prefer trial-level context columns over group-label columns so that
    # datasets like mazeMelodyID_clean_all (group="Maze-Melody", trialType="Maze"/"Melody")
    # get per-trial context encodings rather than a single group-label encoding.
    context_col = _infer_column(df_raw, ["trialType", "context", "condition", "group"])

    if pid_col is None:
        df_raw["_pid"] = "p001"
        pid_col = "_pid"
        print("   ⚠️   No participant-id column — treating as one participant.")

    if context_col is None:
        df_raw["_ctx"] = "Default"
        context_col = "_ctx"
        print("   ⚠️   No context column — treating as single-context dataset.")

    participants = sorted(df_raw[pid_col].unique())
    contexts     = sorted(df_raw[context_col].unique())

    print(f"   Participants : {len(participants)}")
    print(f"   Contexts     : {contexts}")

    context_encodings = _build_context_encodings(contexts)
    configs = {
        "Rich":          RICH_CONFIG,
        "Poor":          POOR_CONFIG,
        "Lazy+RichCtx":  LAZY_RICH_CTX_CONFIG,
        "Rich+LazyCtx":  RICH_LAZY_CTX_CONFIG,
    }

    # ── Phase 1: unified participant-timeline + asymptote training ────────────
    print("\n── Phase 1: participant timelines + asymptote (unified) ──")
    (df_all, asymptote_mechs, mech_stats_list,
     collapse_rows, perturb_rows, driver_rows, ablation_rows,
     wg_rows, collapse_asym_rows, shared_pca_rows,
     matched_rows) = _run_unified_phase(
        df_raw, pid_col, context_col,
        participants, configs, context_encodings,
        global_seed=GLOBAL_SEED,
    )

    # ── Save + plot: learning curves ──────────────────────────────────────────
    df_all.to_csv(output_dir / "training_results.csv", index=False)
    print(f"   Saved training_results.csv  ({len(df_all):,} rows)")

    # ── Transition cost analysis ──────────────────────────────────────────────
    for _tc_dv in ("loss", "accuracy"):
        if _tc_dv in df_all.columns:
            try:
                _tc = compute_transition_cost(df_all, dv=_tc_dv)
                if not _tc["events"].empty:
                    _tc["events"].to_csv(
                        output_dir / f"transition_cost_{_tc_dv}_events.csv", index=False
                    )
                    _tc["unit"].to_csv(
                        output_dir / f"transition_cost_{_tc_dv}_unit.csv", index=False
                    )
                    _tc["block"].to_csv(
                        output_dir / f"transition_cost_{_tc_dv}_block.csv", index=False
                    )
                    print(f"   Saved transition_cost_{_tc_dv}_*.csv  "
                          f"({len(_tc['events'])} events)")
            except Exception as _tc_err:
                print(f"   ⚠️   Transition cost ({_tc_dv}) failed: {_tc_err}")

    plot_rich_poor_comparison(df_all, output_dir)
    plot_accuracy_entropy_curves(df_all, output_dir)
    plot_mechanistic_summary(mech_stats_list, output_dir)

    for ms in mech_stats_list:
        if "pca_coords" in ms:
            tag = f"{ms['model_type']}_p{ms['participant_id']}"
            plot_pca_geometry(ms, tag, output_dir)

    # ── Save + plot: participant-snapshot geometry ────────────────────────────
    if collapse_rows:
        df_col = pd.DataFrame(collapse_rows)
        df_col.to_csv(output_dir / "collapse_participant.csv", index=False)
        plot_collapse_dynamics(df_col, output_dir)
        plot_normalized_collapse(df_col, output_dir)
        plot_phase_portrait(df_col, output_dir)
        plot_dual_decoding_dynamics(df_col, output_dir)

    if perturb_rows:
        df_pert = pd.DataFrame(perturb_rows)
        df_pert.to_csv(output_dir / "context_perturbation.csv", index=False)
        plot_perturbation_dynamics(df_pert, output_dir)

    if driver_rows:
        df_drv = pd.DataFrame(driver_rows)
        df_drv.to_csv(output_dir / "collapse_driver_dynamics.csv", index=False)
        plot_collapse_driver_dynamics(df_drv, output_dir)

    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(
            output_dir / "context_ablation_dynamics.csv", index=False
        )

    # ── Save + plot: asymptote geometry ──────────────────────────────────────
    plot_asymptote_geometry(asymptote_mechs, output_dir)

    for ms in asymptote_mechs:
        if "pca_coords" in ms:
            plot_pca_geometry(
                ms, f"{ms['model_type']}_asym_p{ms['participant_id']}", output_dir
            )

    if wg_rows:
        df_wg = pd.DataFrame(wg_rows)
        df_wg.to_csv(output_dir / "weight_gradient_dynamics.csv", index=False)
        plot_weight_gradient_dynamics(df_wg, output_dir)

    if collapse_asym_rows:
        df_ca = pd.DataFrame(collapse_asym_rows)
        df_ca.to_csv(output_dir / "collapse_asymptote.csv", index=False)

        asym_plot_dir = output_dir / "asymptote"
        asym_plot_dir.mkdir(exist_ok=True)
        plot_collapse_dynamics(df_ca, asym_plot_dir)
        plot_normalized_collapse(df_ca, asym_plot_dir)
        plot_phase_portrait(df_ca, asym_plot_dir)
        plot_dual_decoding_dynamics(df_ca, asym_plot_dir)

        if wg_rows:
            df_wg_asym = pd.DataFrame(wg_rows)
            grad_driver_rows = [
                {
                    "model_type":         r["model_type"],
                    "participant_id":     r["participant_id"],
                    "checkpoint_value":   r["epoch"],
                    "step_or_epoch":      r["epoch"],
                    "gradient_alignment": r.get("grad_context", np.nan),
                }
                for _, r in df_wg_asym.iterrows()
            ]
            df_gd = pd.DataFrame(grad_driver_rows)
            sep_cols = [
                "model_type", "participant_id", "step_or_epoch",
                "trajectory_separation", "context_decodability",
            ]
            sep_cols = [c for c in sep_cols if c in df_ca.columns]
            df_gd = df_gd.merge(
                df_ca[sep_cols].rename(
                    columns={"step_or_epoch": "checkpoint_value"}
                ),
                on=["model_type", "participant_id", "checkpoint_value"],
                how="left",
            )
            df_gd.to_csv(asym_plot_dir / "collapse_driver_asymptote.csv", index=False)
            plot_collapse_driver_dynamics(df_gd, asym_plot_dir)

    # ── Save + plot: shared PCA ────────────────────────────────────────────────
    if shared_pca_rows:
        df_spca = pd.DataFrame(shared_pca_rows)
        h_cols  = [c for c in df_spca.columns if c.startswith("h")]
        X       = df_spca[h_cols].values
        pca     = _PCA(n_components=min(2, X.shape[1], X.shape[0]))
        coords  = pca.fit_transform(X)
        df_spca["PC1"] = coords[:, 0]
        df_spca["PC2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
        df_spca["model_label"] = df_spca["model_type"].map(
            lambda x: DISPLAY_NAMES.get(x, x)
        )
        df_spca.to_csv(output_dir / "shared_pca_trajectories.csv", index=False)
        plot_shared_pca_panels(df_spca, output_dir)

    # ── Phase 2: matched-performance geometry (from unified CE snapshots) ──────
    print("\n── Phase 2: matched-performance geometry (from unified snapshots) ──")
    if matched_rows:
        matched_df = pd.DataFrame(matched_rows)
        matched_df.to_csv(output_dir / "matched_geometry_ce.csv", index=False)
        plot_matched_geometry(matched_df, output_dir)
        _plot_matched_geometry_generic(
            matched_df, "ce_threshold", "CE threshold",
            output_dir / "matched_geometry_ce_generic.png",
        )
    else:
        print("   ⚠️   No CE-threshold snapshots captured — skipping matched geometry.")

    # ── Phase 3: robustness parameter sweeps ──────────────────────────────────
    print("\n── Phase 3: parameter sweeps ──")
    df_hdim = run_hidden_size_sweep(context_encodings=context_encodings,
                                    output_dir=output_dir)
    if not df_hdim.empty:
        plot_hidden_size_sweep(df_hdim, output_dir)

    df_lr = run_learning_rate_sweep(context_encodings=context_encodings,
                                    output_dir=output_dir)
    if not df_lr.empty:
        plot_learning_rate_sweep(df_lr, output_dir)

    df_scale = run_init_scale_sweep(context_encodings=context_encodings,
                                    output_dir=output_dir)
    if not df_scale.empty:
        print(f"   Saved init_scale_sweep.csv  ({len(df_scale)} rows)")

    df_scale_lr = run_scale_lr_sweep(context_encodings=context_encodings,
                                     output_dir=output_dir)
    if not df_scale_lr.empty:
        print(f"   Saved scale_lr_sweep.csv  ({len(df_scale_lr)} rows)")

    # ── Phase 4: model summary metrics ────────────────────────────────────────
    df_col_all = (
        pd.DataFrame(collapse_rows + collapse_asym_rows)
        if (collapse_rows or collapse_asym_rows) else pd.DataFrame()
    )
    df_pert_all = pd.DataFrame(perturb_rows) if perturb_rows else pd.DataFrame()

    summary_rows = []
    for cfg_name, cfg in configs.items():
        for pid in participants[:min(5, len(participants))]:
            smry = compute_model_summary_metrics(
                collapse_df     = df_col_all,
                perturbation_df = df_pert_all,
                results_df      = df_all,
                config_meta     = {
                    "model_type":    cfg_name,
                    "participant_id": pid,
                    "hidden_dim":    cfg.hidden_dim,
                    "learning_rate": cfg.learning_rate,
                    "init_scale":    cfg.init_scale,
                },
            )
            summary_rows.append(smry)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "model_summary_metrics.csv", index=False
        )

    print(f"\n✅  Complete — results in {output_dir}\n")
