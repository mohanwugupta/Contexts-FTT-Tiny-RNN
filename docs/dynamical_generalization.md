# Prospective dynamical generalization program

This is a separately identified extension to the original paper freeze, implementing `PRD_RNN_DYNAMICAL_GENERALIZATION_PROGRAM.md`. E1 and E2 use the existing 120 checkpoints only. E3 introduces familiar-only regularization in a new 2 × 2 design, conditional on code integrity and a fixed A/B-only pilot. E4 repeats a feasible frozen E3 design on three independently registered sequence classes. No human model or RT readout is fit.

## Reproduce

Python 3.11 and the pinned dependencies in `requirements.lock` are required. Install the package with `python -m pip install --no-deps -e .`, or run the commands below with `PYTHONPATH=src`. Historical checkpoints and source tables must be present at the paths in `configs/dynamical_generalization_v1.json`; the audit names missing paths and never replaces them with training.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/unit tests/regression tests/integration
python3 scripts/audit_dynamical_generalization_inputs.py --config configs/dynamical_generalization_v1.json
python3 scripts/run_dynamical_generalization.py --stage e1 --mode eval_only --output results/dynamical_generalization/my_reproduction
python3 scripts/run_dynamical_generalization.py --stage e2 --mode eval_only --output results/dynamical_generalization/my_reproduction
python3 scripts/run_dynamical_generalization.py --stage e3 --mode ab_pilot --output results/dynamical_generalization/my_reproduction
python3 scripts/run_dynamical_generalization.py --stage e3 --mode confirmatory --output results/dynamical_generalization/my_reproduction
python3 scripts/run_dynamical_generalization.py --stage e4 --mode confirmatory --output results/dynamical_generalization/my_reproduction
python3 scripts/verify_dynamical_generalization.py --output results/dynamical_generalization/my_reproduction
```

Use a fresh output root and the same root for all sequential stages. Each stage refuses a second start; completed outputs and source hashes are immutable. E3 confirmation reads the hash-locked pilot selection. If no candidate meets its criteria, E3 reports `insufficient_familiar_support` without confirmatory training; E4 writes its frozen design and `blocked_by_e3` report. This is a permitted final scientific result. Pilot seeds 1000–1004 never enter confirmatory inference. No post-result coefficient expansion or probe-based exclusion is allowed.

`freeze_dynamical_generalization_design.py` created the prospective registries before evaluation. It refuses to replace them. Do not rerun it against the supplied freeze. The scratchpad and append-only decision trail record all conventions, source discoveries, RED/GREEN evidence and scientific gates.

## Measures and numerical contract

Natural states/logits retain the historical explicit float32 recurrence. Affine midpoint and CE convexity checks recompute the linear head in float64 on those states; recorded float32 output-head roundoff is separately exported. E1 computes exact nonlinear error propagation, a smooth-real analytical Jacobian of the same tanh formula, the linear approximation, and its explicit Taylor remainder. An exogenous injection row is labeled as a placement rather than a recurrence identity. Finite Jacobian products are local finite-horizon diagnostics; singular values above one do not prove an autonomous attractor or instability.

The E2 registry contains 35 declared decimal-coordinate grid points, eight exact anchors and six magnitude rays. Nominal grid axes retain their declared decimal values; the stored raw vectors are hashed individually. Exact Novel anchors retain their original radical values. Zero cue has undefined angular distance. Same-alpha beta-sign pairs and same-norm direction controls are distinct estimands. C1/C3 same-alpha equivalence is a structural precision check, not a required outcome for unrestricted models.

E3 applies current-context A/B CE plus weak/strong hidden-separation regularization, with or without recursive midpoint-closure loss. Closure normalization is fixed by seven transitions and eight hidden units. Its detached target and neutral physical cue are derived from A/B; that neutral cue equals physical Mixed, which is explicitly disclosed. Protected behavioral targets never train. The complete 48-candidate pilot uses objective-equivalent caching (80 distinct objective settings per pilot seed), reports all attempts, and selects a common setting by the frozen familiar-only rule. Final familiar feasibility is written and hashed before protected model evaluation.

All bootstrap effects use 10,000 whole-seed draws, RNG seed 20260925, with the actual index matrix saved. E1 prediction uses nested grouped seed folds with within-training standardization and regularization selection. Per-sequence replication effects remain separate; heterogeneity summarizes fixed-sequence means descriptively.

## Historical boundary

The source files and artifacts of the old paper freeze remain byte-identical. The new configuration records the original paper's exact source-file scope and digest in addition to the expanded program source manifest. The old paper's global source-tree guard naturally includes newly added files if run on this expanded checkout; reproduce that historical freeze from commit `1240883`, with its original artifact set. Historical arithmetic and state/CE parity are regression-tested here without revising the old contract or reports.

The current run exports stage reports, all per-transition and per-seed measurements, full per-update parameter histories for new training, genuine pre-update losses, registry audits, figure/table maps, output hashes and a final report. Check the reports for null, confounded or blocked results before making a causal claim.

## Completed registered run

The [final report](../results/dynamical_generalization/dg_v1_20261103/FINAL_DYNAMICAL_GENERALIZATION_REPORT.md) records E1/E2 completion and the E3/E4 stop. All 48 pilot candidates failed the conjunction of familiar-competence criteria; none was selected. The [pilot numerical summary](../results/dynamical_generalization/dg_v1_20261103/PILOT_FEASIBILITY_SUMMARY.md) distinguishes this competence failure from the learning checks, which all candidates passed. E4 is explicitly design-only. The final manifest verifies 470 output files; 42 unit/regression/integration tests pass. See [figure captions](dynamical_figure_captions.md). Lightweight reports/manifests are also versioned under artifacts/manifests/dynamical_generalization_v1/.
