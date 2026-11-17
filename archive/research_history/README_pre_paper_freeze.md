# Context-configured motor-sequence learning

This is the reproducible paper repository for the SRN and human analyses of how
context configures the active state of an otherwise identical learned motor
sequence.

The paper pipeline is deliberately ordered SRNs first, humans second. High-scale
8-unit tanh Elman SRNs learn context-specific A/B representations and an
execution-control consequence; frozen Mixed and Novel probes are graded, and
matched sequence-position A/B components have stronger causal effects than
mismatched controls. The SRN is a mechanistic existence proof. Human behavior is
then analyzed at matched ordinal transition positions, where a participant-held-
out low-rank configured-state model wins and the selected solution is rank 2.
The exact SRN and human transition profiles do **not** match; that negative result
is part of the paper's inferential boundary.

## Quick start

Python 3.11 is the validated runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The expected human inputs are:

```text
data/mixedBreak_clean.csv
data/threeContextsBreak_clean.csv
```

Run the complete paper analysis and regenerate paper figures:

```bash
python scripts/run_all_final.py
```

The full pipeline retrains the registered SRN cohort. To reuse the existing
checksum-validated v2.1 checkpoints while rerunning all frozen causal and human
analyses:

```bash
python scripts/run_all_final.py --reuse-execution
```

Individual entry points are:

```bash
python scripts/run_srn_final.py --config configs/srn_final.json
python scripts/run_human_final.py --config configs/human_final.json
python scripts/make_paper_figures.py
```

Paper-facing figures and their copied source tables are written under
`outputs/final/`. Registered analysis outputs remain in their versioned frozen
directories so the pre-cleanup checksums and model-checkpoint provenance remain
auditable.

For a read-only visual validation of loss and model-generated normalized RT over
all training sequences, including R-style Lump/Cross/Split overlays on the
observed traces, run:

```bash
python scripts/run_srn_validation_diagnostics.py
```

This writes only to `outputs/srn_trial_validation/`; it does not retrain or
modify a frozen model. See
[the SRN trial-validation note](docs/SRN_TRIAL_VALIDATION.md) for interpretation.

Run the new frozen confirmatory High/Low/sequence-only analysis (60 primary
networks, checkpoints, probes, canonical state-space fits, low-rank bridge,
causal controls, report, and figures) with:

```bash
python scripts/run_confirmatory_srn.py --profile confirmatory \
  --output outputs/confirmatory_srn_final
```

Use `--profile smoke` for the fast six-network integration check.

Run the stepwise structured-context ecology, with automatic stage gates, using:

```bash
python scripts/run_structured_context_ecology.py --profile ecology \
  --output outputs/structured_context_ecology_final
```

Run the ordinal-sequence scaffold plus context-specific spatial-layout ecology:

```bash
python scripts/run_ordinal_layout_ecology.py --profile ecology \
  --output outputs/ordinal_layout_ecology_final
```

The runner stops later stages automatically if sequence learning, High/Low
splitting, state-space behavior, or RT sanity fails at a preceding stage.

Analyze saved Stage 2 transition RT as a function of time since an A/B switch:

```bash
python scripts/run_posthoc_transition_switch_analysis.py \
  --output outputs/posthoc_transition_switch_analysis
```

This posthoc runner is read-only: it never loads a checkpoint, constructs an
optimizer, or updates a parameter. It requires
`stage2_layout_transition_training.csv`, which future ordinal-layout runs save
automatically. Older runs that predate this logging table stop with a diagnostic
report because final checkpoints cannot reconstruct their historical RT stream.

Run the read-only mechanism-first Mixed/Novel probe diagnostics with:

```bash
python scripts/run_probe_diagnostics.py \
  --config configs/mechanism_first_probe_diagnostics.json
```

The runner hashes every frozen input, separates condition level from centered
transition shape, re-scores existing held-out human models, gates history on
identifiability, and audits an independent no-context basis for Novel. It does
not train an SRN or rerun a probe. Use `--from-stage N --through-stage M` to
restart a bounded stage range. See
[the mechanism-first guide](docs/MECHANISM_FIRST_PROBE_DIAGNOSTICS.md).

Overlay Mixed and Novel sequence RT on the original A/B learning curve, with
read-only A/B/Mixed/Novel branches from replay-verified matched checkpoints:

```bash
python scripts/run_time_matched_sequence_probe_overlay.py \
  --config time_matched_probe_config.json
```

This diagnostic preserves the frozen learner and validates its deterministic
replay against every saved final model/readout hash before using intermediate
probes. It writes to `outputs/time_matched_sequence_probe_overlay/`.

Locate the time-matched probe deficit in next-token prediction versus its
expression through the learned execution readout with:

```bash
python scripts/run_probe_prediction_vs_execution.py \
  --config probe_prediction_execution_config.json
```

This extension exports transition probability, NLL, RT, and readout parameters;
computes paired excess over the worse A/B endpoint; and applies each seed's
checkpoint-20 readout as a counterfactual transformation. It does not fit a new
model or update a probe.

Run the coordinate-only human-SVG geometry experiment with:

```bash
python scripts/run_human_svg_geometry_transfer.py
```

The four coordinate arrays in `human_svg_target_coordinates_v3.json` are used
in file order; their embedded names are ignored. The default run screens five
paired seeds and advances passing geometries to the frozen 20-seed analysis.
Use `--smoke` for the two-seed engineering check.

Run the prespecified cue-sharing and hidden-capacity frontier with:

```bash
python scripts/run_differentiation_transfer_frontier.py
```

This first runs the five-seed S0--S4 screen and the original-G2 equivalence
control. Registered stop conditions are honored automatically; only admissible
conditions proceed to the 20-seed and H=8/16/32 stages.

Run the hard-gated emergent retrieval/context-inference program with:

```bash
python scripts/run_emergent_retrieval_program.py
```

The runner first reproduces the exact G2 shared-x baseline, tests the explicit
C1 responsibility-weighted sufficiency control, and only then advances through
E1--E4 and the temporal C2 control. Confirmatory seed extension depends solely
on familiar A/B gates. Mixed, Novel, neutral, interpolation, history, and
ablation results are frozen post-training diagnostics. Use `--smoke --output
/tmp/emergent-retrieval-smoke` for the two-seed/20-sequence engineering run.

Run the hierarchical compositional pretraining ecology with:

```bash
python scripts/run_hierarchical_pretraining_ecology.py
```

This compares no pretraining (P0), structure-shuffled pretraining (P1), and
paired structured pretraining (P2) on the unchanged eight-unit High-scale SRN.
The frozen internal synthetic probes gate confirmatory human G2 transfer; human
Mixed/Novel outcomes cannot select the ecology. Use `--smoke --output
/tmp/hierarchical-pretraining-smoke` for the compact engineering run.

Run the sequential structural-prior curriculum with:

```bash
python scripts/run_structural_prior_curriculum.py
```

The E0--E3 ecologies progressively remove stable factor semantics. Every level
must pass both its paired 100-world pilot and independent 1,000-world
confirmatory gate before the next is attempted. Human G2 files and figures are
created only after an ecology is selected. Use `--smoke --output
/tmp/structural-prior-curriculum-smoke` to verify the gate/firewall machinery.

Compare the preregistered P0--P5 pretraining ecologies with paired shuffled
controls and a shared frozen unseen-family battery using:

```bash
python scripts/run_compare_pretraining_ecologies.py
```

The 5-seed pilot gates the 20-seed confirmatory stage; only the top two
confirmatory-passing structured ecologies, the top ecology's matched P1
control, and P0 can reach human G2 transfer. Use `--smoke --output
/tmp/compare-pretraining-ecologies-smoke` for the engineering check.

Run the frozen implementation audits for the context-controlled recurrent
architectures before confirmatory training:

```bash
python scripts/run_context_controlled_preflight.py
```

This audits neutral-history uncertainty and causal target isolation without
optimizer updates, freezes the pilot-selected A/B architectures and relevant
source/checkpoint hashes, and registers 20 held-out seeds. Once it passes, run
the synthetic-only confirmatory stage with:

```bash
python scripts/run_context_controlled_recurrent_architectures.py \
  --profile confirmatory \
  --config configs/context_controlled_recurrent_architectures_confirmatory.json \
  --workers 4 \
  --checkpoint-dir /Users/mg9965/ResearchCheckpoints/Contexts-FTT-Tiny-RNN/context_controlled_recurrent_architectures
```

The human-data firewall remains closed unless a later, deliberate invocation
adds `--open-human-firewall`. The runner migrates existing exact-name Box
checkpoints to the local directory, resumes them, and then uses atomic compact
checkpoints plus append-only 2,500-world metric chunks. Training and frozen
validation cells run independently across the requested workers; final tables
are assembled once in deterministic architecture/condition/seed order.

Run the capacity × learning-objective episodic meta-learning experiment with:

```bash
python scripts/run_episodic_meta_learning_capacity.py
```

This runs the complete H8/H32/H64/H128 pilot before applying synthetic V1--V5
gates. Only passing capacities are confirmed and allowed into human G2
transfer. Use `--smoke --output /tmp/episodic-meta-capacity-smoke` for the
registered H8/H64, 100-episode engineering run.

Calibrate width-specific SRN inner-loop dynamics before any further structural
meta-learning interpretation with:

```bash
python scripts/run_srn_meta_learning_calibration.py
```

The runner evaluates the frozen width × initialization × learning-rate × step
grid, applies the registered clipping rescue only where needed, and gates the
confirmatory, meta-batch, and revised structural stages. Use `--smoke --output
/tmp/srn-meta-learning-calibration-smoke` for the fast engineering check.

Run the locked H128 structural-prior transfer into the human-style A/B task:

```bash
python scripts/run_human_meta_prior_transfer.py
```

The runner verifies the upstream gate, hash-locks 20 distinct paired synthetic
checkpoints, resets the target optimizer and monotonic RT readout, and evaluates
frozen first-exposure probes behind the familiar gate. Use `--smoke --output
/tmp/human-meta-prior-transfer-smoke` for the three-seed engineering path.

## Repository map

- `configs/`: portable, frozen paper-level contracts and relative paths.
- `scripts/run_*_final.py`: canonical execution path.
- `ftt_srn_execution_v2/`: architecture, curriculum, execution objective, and
  frozen probes.
- `ftt_srn_causal_representation/`: legacy full-subspace causal suite retained
  for the strict necessity boundary.
- `ftt_srn_context_sequence_interaction/`: matched/mismatched current-state
  components and natural A/B execution models.
- `human_configured_state/`: validated preprocessing and held-out M0–M4 model
  comparison.
- `human_svg_geometry_transfer/`: hard-gated G0/G1/G2 test of exact SVG and
  shared-x spatial geometry in the frozen Stage-2 SRN.
- `differentiation_transfer_frontier/`: hard-gated S0--S4 shared-cue and
  H=8/16/32 capacity frontier built on the validated G2 shared-x geometry.
- `emergent_retrieval_program/`: V0/C1/C2 controls, E1--E4 two-dimensional
  context-inference architectures, hard familiar gates, frozen transfer/history
  diagnostics, and prespecified successful-model ablations.
- `hierarchical_pretraining_ecology/`: paired structured/shuffled synthetic task
  worlds, unchanged-SRN pretraining, optimizer-reset human transfer, early and
  late frozen probes, and differentiation/transfer diagnostics.
- `structural_prior_curriculum/`: sequential E0--E3 compositional ecologies,
  modified-position diagnostics, most-abstract-stage selection, conditional
  human transfer, and success-contingent weight-subsystem ablations.
- `srn_meta_learning_calibration/`: fresh-start inner-loop stability grid,
  width-specific initialization and clipping selection, confirmatory
  adaptation, meta-batch calibration, and the gated revised structural test.
- `human_meta_prior_transfer/`: locked-checkpoint A/B acquisition, immutable
  first-exposure probes, familiar gating, canonicalized transfer outcomes,
  hidden geometry, prior retention, and conditional subsystem ablations.
- `human_low_rank_interpretation/`: rank selection, stability, cross-dataset
  replication, semantic limits, and human–SRN comparison.
- `reproducibility/`: pre-cleanup checksums, numerical snapshots, and equivalence
  tooling.
- `srn_validation/`: read-only trial diagnostics and the Python translation of
  the legacy no-fatigue Lump/Cross/Split fits.
- `confirmatory_srn/`: frozen execution-head-free confirmatory pipeline with the
  two-parameter monotonic confidence readout and no human fitting.
- `structured_context_ecology/`: gated literal-item, ordinal-context, salience,
  Mixed, and structured-Novel environmental analysis.
- `ordinal_layout_ecology/`: gated shared-ordinal, dual-layout, Mixed corridor,
  sparse-overlap follow-up, and Novel-layout analysis.
- `posthoc_transition_switch/`: read-only transition-profile, switch-cost,
  block-recovery, and frozen A/B endpoint analysis.
- `mechanism_probe_diagnostics/`: read-only level/shape, q/delta, held-out
  rescoring, history-identifiability, independent-default, and adjudication
  pipeline.
- `time_matched_probe_overlay/`: exact Stage-2 replay, immutable same-snapshot
  sequence probes, endpoint classification, uncertainty, and temporal overlays.
- `probe_prediction_execution/`: transition prediction-loss, execution-RT, and
  fixed-readout counterfactual diagnostics on the verified matched checkpoints.
- `archive/`: historical analyses, scripts, specifications, and old outputs;
  none are imported by the paper runners.

See [the analysis overview](docs/ANALYSIS_OVERVIEW.md),
[reproducibility guide](docs/REPRODUCIBILITY.md), and
[result map](docs/RESULT_MAP.md) for details. The completed reorganization and
before/after equivalence audit are documented in
[CLEANUP_REPORT.md](CLEANUP_REPORT.md).
