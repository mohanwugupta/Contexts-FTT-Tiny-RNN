# Epoch-structured Combined-context modeling (PRD2)

The canonical pipeline now analyzes the Combined schedule as four distinct
informative epochs: Mixed 1, Mixed 2, the first A/B return block, and Mixed
Test. It removes raw sequence 0 from every block before any metric or model,
leaving original sequences 2–10 as nine analyzed observations.

This implementation addresses the PRD's downstream question: given established context-specific Maze (A) and Melody (B) skill states, what retrieves or learns from a combined-cue or novel third context?

## What is implemented

- Validated keypress-to-sequence normalization with nine completed sequences per 17-block Combined schedule.
- First-class `Mixed_1`, `Mixed_2`, `Post_Mixed_Return`, and `Mixed_Test` epoch labels.
- Explicit errors for missing/unknown context labels and participant-safe chronological ordering.
- Restart-free split A/B trajectories with fatigue/recovery and zero Mixed-RT leakage.
- Primary log-RT midpoint deviation `D`; separation-masked mixture ratios are secondary.
- Explicit M0–M6 ladder: equal/static retrieval, counterbalance-static retrieval, recurrent inference, gate adaptation, A/B updating, and third-state formation.
- Grouped 80/20 participant splits, separate and combined counterbalance fits, A↔B mirroring, and cross-counterbalance prediction.
- Prospective Mixed 1→2, Mixed 1/2→Mixed Test, and Mixed 1/2→post-Mixed A/B tests without target refitting.
- Non-recurrent and 1/2/4-unit neural gates plus block, pre-Mixed-1, and between-Mixed reset interventions.
- Progressive held-out distillation and recovery at easy, human-like, and empirical-neighborhood parameters.
- All required v2 tables and eight epoch-specific figures.

The existing next-item sequence RNN is not used as the behavioral probe model.

## Run

From this repository:

```bash
python scripts/run_probe_analysis.py
```

The defaults are the final-analysis configuration: 20 neural seeds, 300 gate
epochs, and 20 recovery simulations per mechanism in all three regimes. Results
go to `outputs/epoch_analysis_v2/`.

A faster implementation check can be run with:

```bash
python scripts/run_probe_analysis.py \
  --output outputs/epoch_analysis_v2_verification \
  --neural-seeds 1 \
  --gate-epochs 60 \
  --recovery-simulations 1 \
  --recovery-regimes easy
```

That configuration is automatically labeled `verification_only`; it cannot satisfy the recovery acceptance criterion.

PRD2 is intentionally Combined-only. The header-only third-context file is not
loaded or recorded by this replacement pipeline.

Run tests with:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache \
XDG_CACHE_HOME=/private/tmp/xdg-cache \
python -m pytest -q
```

## Current data status

`mixedBreak_clean.csv` contains the Combined cohort. PRD2 retains 76 complete
participants, 17 blocks each, and 9 observations per block (11,628 rows). The
third-context CSV is still header-only and is outside this Combined-only update.

## Interpretation safeguards

- Mixed RT never fits or updates the foundational A/B state trajectories.
- Held-out participants' Mixed RT never estimates population mechanism parameters.
- Post-Mixed Return and Mixed Test are explicit prospective targets.
- Counterbalance/history is named and modeled explicitly.
- Mixture indices are masked when counterfactual A/B RTs differ by less than 2%, because the ratio is ill-conditioned there.
- Recovery tables record the exact 17×9 schedule and remain verification-only below 20 simulations per mechanism/regime.
