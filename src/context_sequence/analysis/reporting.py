"""Claim-to-evidence reports; no outcome-dependent panels or model redesign."""
from pathlib import Path
import numpy as np
import pandas as pd


def table(frame):
    if not len(frame):return 'No estimable rows.'
    def s(v):
        if isinstance(v,(float,np.floating)):return 'unavailable' if not np.isfinite(v) else f'{v:.5g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(s(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def write(out,name,body):
    p=Path(out)/name
    with p.open('x') as f:f.write(body+'\n')


def e1_report(out,summary,static,cv,bootstrap):
    mse=cv.groupby(['cohort','model']).squared_error.mean().unstack()
    means=summary[summary.probe.isin(['A','B','Mixed','NoContext','Novel30Mean','Novel60Mean'])].groupby(['cohort','condition','probe'])[['raw_ce','closure_task_norm','error_task_norm','taylor_remainder_task_norm']].mean().reset_index()
    body='# E1 — Static compatibility and dynamical accessibility\n\nA linear output head maps the hidden midpoint to the arithmetic midpoint of logits. Because A and B share a correct label and CE is convex in logits, midpoint CE cannot exceed their mean endpoint CE. This algebraic guarantee does not imply natural recurrent access.\n\n'
    body+=f'All {len(static)} static transition rows passed affine and convexity checks. Maximum affine discrepancy: {static.affine_error.max():.4g}; minimum Jensen gap: {static.convexity_gap.min():.4g}. Float32 recorded-head roundoff is exported separately.\n\n'
    body+='The exported identity is e_t = b_t + [f(m_(t−1)+e_(t−1))−f(m_(t−1))]. J_t e_(t−1) is only a first-order approximation; its Taylor remainder and output-relevant norms are separate columns. Jacobian singular values and finite products describe local finite-horizon amplification, not autonomous attractor stability. Single-patch rows explicitly label the exogenous placement and then audit recurrent continuation.\n\n'+table(means)+'\n\nNested held-out-seed prediction MSE (lower is better):\n\n'+table(mse.reset_index())+'\n\nPaired held-out-seed and C0–C3 effects:\n\n'+table(bootstrap)+'\n\nThe dynamics-augmented model adds only prespecified mean closure and finite-horizon amplification features. All conditions and cues of a seed remain held out together. Incremental prediction is not causal mediation. Familiar acquisition source logs remain historical changing-weight observations; these final rollouts are not learning curves.\n\nEvidence: tables/e1_transition_dynamics.csv, e1_seed_summary.csv, e1_static_convexity_audit.csv, e1_single_patch_replay.csv, e1_grouped_cv_predictions.csv, e1_effects_bootstrap.csv; figures/e1_accessibility.{png,pdf,svg}; audit/e1_historical_regression.csv. All inference resamples trained seed clusters within cohort. Teacher forcing fixes seven external tokens.\n'
    write(out,'E1_RESULTS.md',body)


def e2_report(out,controls,invariants):
    selected=controls[controls.comparison_kind.isin(['anchor_Mixed_minus_NoContext','exact_Novel_A_beta_sign','exact_Novel_B_beta_sign'])&controls.metric.eq('raw_ce')]
    means=selected.groupby(['cohort','condition','comparison_kind']).effect.mean().reset_index()
    structural=invariants[invariants.structural_identity_expected]
    body='# E2 — Context information and cue geometry\n\nThe full finite registry contains 35 grid locations plus eight exact anchors and six cue-magnitude rays. Alpha=x−y and beta=x+y; all third-channel values are zero. Rounded grid locations and exact radical Novel anchors retain distinct identities. Raw norm, nearest-endpoint angle and Euclidean distance to the familiar segment are exported separately. Zero-cue angle is undefined.\n\n'+table(means)+'\n\n'
    body+=f'{len(structural)} C1/C3 same-alpha comparisons passed the frozen precision bounds; maximum logit difference {structural.max_logit_difference.max():.4g}, maximum CE difference {structural.max_ce_difference.max():.4g}. Unrestricted models are measured by the same comparisons without requiring an identity.\n\n'
    body+='Same-alpha beta-sign comparisons are norm matched; same-norm/different-direction pairs and A/B magnitude rays are separately labeled. The beta finite difference is descriptive and does not establish psychological familiarity. No beta/rho pathway was trained. All grid points are repeated measurements of a seed, not independent networks.\n\nEvidence: tables/e2_grid_transition.csv, e2_grid_seed.csv, e2_matched_geometry_controls.csv, e2_projection_invariants.csv, e2_controls_bootstrap.csv; figures/e2_context_grid.{png,pdf,svg}. Heatmap color scales are condition-specific, and every panel shows across-seed means.\n'
    write(out,'E2_RESULTS.md',body)


def factorial_report(out,prefix,status,familiar=None,feasibility=None,effects=None):
    body=f'# {prefix.upper()} — Causal separation × closure\n\nImplementation status: {status["implementation_status"]}. Scientific status: {status["scientific_status"]}. Evidence status: {status.get("evidence_status",status["scientific_status"])}.\n\n'
    if familiar is None:
        body+='No frozen candidate satisfied every prospectively declared familiar-only pilot criterion. All 48 attempted settings and their five reserved seeds are retained, including failures. No confirmatory model or protected probe was evaluated for this factorial. This is insufficient familiar support, not evidence against recurrent closure. The registered grid was not expanded.\n'
    else:
        means=familiar.groupby(['sequence_id','condition'])[['A_ce','B_ce','hidden_separation','task_separation','closure_fixed_H','margin_attainment']].mean().reset_index()
        body+=table(means)+'\n\nFamiliar-only gates, frozen before protected evaluation:\n\n'+table(feasibility)+'\n\nRegistered paired effects (all seed pairs retained):\n\n'+table(effects[effects.metric.eq('raw_ce')&effects.probe.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])])+'\n\n'
        body+='The manipulation directly regularizes hidden separation; output/task separation is an independent check. Closure uses recursive neutral rollout and fixed-H normalization, not Phase 1L separation normalization. Its cue is derived from A/B and numerically equals physical Mixed; no protected behavioral targets are trained. Competence comparability requires each endpoint mean weak/strong gap ≤0.30 nat at fixed closure. Failed engagement or competence imbalance limits causal interpretation without deleting seeds. Interaction is an intended-treatment effect, not mediation proof.\n'
    body+=f'\nEvidence: manifests/e3_selected_config_manifest.json, tables/e3_pilot_ab_only.csv, e3_pilot_candidate_audit.csv, figures/e3_pilot_feasibility.*; all available {prefix} training, feasibility, transition and contrast tables are hash-indexed in manifests/{prefix}.json.\n'
    write(out,prefix.upper()+'_RESULTS.md',body)


def final_report(out,statuses):
    body='# Dynamical generalization — final report\n\nThe four questions concern algebraic output compatibility versus dynamical access (E1), independent contextual coordinates (E2), a randomized separation × closure training intervention (E3), and replication across fixed sequence classes (E4). The original four-experiment paper freeze remains a separate historical result.\n\n'+table(pd.DataFrame(statuses))+'\n\n'
    for stage in ['E1','E2','E3','E4']:
        p=Path(out)/(stage+'_RESULTS.md')
        if p.exists():body+='\n'+p.read_text()+'\n'
    body+='\n## Plain-language interpretation\n\nA good familiar midpoint output is mathematically expected from the shared target and linear head. E1 quantifies whether recurrent steps reach and sustain that state, while keeping exact nonlinear displacement distinct from local linearization. E2 measures contextual information that the bottleneck necessarily discards; it does not identify a psychological novelty signal. Strong causal claims about separation require both successful process manipulation and comparable familiar learning in E3. When familiar-only feasibility fails, that causal question and its sequence replication remain unresolved. No architecture, cue axis, coefficient grid or output head was changed to obtain a preferred probe result.\n\nAll predictive validation holds out complete seed groups. Bootstrap draws resample seeds within fixed sequence. Replication is reported sequence by sequence; three additional sequences cannot establish architecture-wide universality. Negative or blocked results terminate the registered program.\n\n## Provenance and reproduction\n\nEach stage manifest records environment, source/config hashes, checkpoint identity, output hashes and run registry. See audit/ and manifests/ for exact before/after integrity evidence and tables/claim_source_map.csv for figure/claim sources. Reproduction instructions are in docs/dynamical_generalization.md.\n'
    write(out,'FINAL_DYNAMICAL_GENERALIZATION_REPORT.md',body)
