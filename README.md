# Context and sequence generalization: four frozen computational experiments

## Agent navigation and latest research

Start with [REPO_MAP.md](REPO_MAP.md) and [AGENTS.md](AGENTS.md). The latest [context failure/selective repair report](results/context_failure_selective_repair/v1_1/FINAL_CONTEXT_FAILURE_SELECTIVE_REPAIR_REPORT.md) preserves the negative, feasibility-limited result. [PUBLICATION.md](PUBLICATION.md) explains the private GitHub snapshot and release evidence bundle. The original paper freeze below remains a separate program.

This repository evaluates how a context-dependent sequence model handles unfamiliar cues. The four paper experiments ask whether familiar A/B differentiation accompanies a generalization cost; whether a useful shared hidden state remains available; whether the model can generate and maintain that state under physical cue changes; and whether two historical constructive mechanisms improve natural generalization. The analyses use frozen checkpoints, 20 registered seeds per condition, and seven teacher-forced predictions per sequence. They do not retrain or tune models.

The final evaluation is in [the four-experiment report](results/paper/v2_20260925_novelty30_60_r2/FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md). A2, the canonical matched-acquisition robustness test, has insufficient common support: only one of 20 seed pairs meets the prespecified familiar-performance caliper, and its matched intermediate checkpoint is unavailable. The result is reported without a new-Novel matched contrast. The other seven analyses and all four computational figures are included, including unfavorable outcomes.

## Reproduce the frozen paper analysis

Use Python 3.11. The package and pinned environment are in `pyproject.toml` and `requirements.lock`. The existing frozen checkpoints and source tables under `analysis/` and `outputs/` are required. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python scripts/audit_paper_inputs.py --config configs/paper_freeze_v2.json
python scripts/freeze_novelty_registry.py --config configs/novelty_registry_v1.json
python -m pytest -q tests/unit/test_paper_contract.py tests/regression/test_paper_evaluator.py tests/integration/test_paper_pipeline.py
python scripts/run_paper_freeze.py --config configs/paper_freeze_v2.json --mode evaluation_only --output results/paper/my_reproduction
python scripts/verify_paper_freeze.py --config configs/paper_freeze_v2.json --result results/paper/my_reproduction
```

The pipeline requires a **fresh** output directory and refuses to overwrite a freeze. Choose a new path for each reproduction. It checks the paper contract, cue registry, 120 checkpoint hashes, and historical evaluator regressions before new-cue model evaluation; an optimizer/backward firewall remains active. It exports transition and seed tables, paired bootstrap summaries, four PNG/PDF/SVG figures, captions, source maps, audits, a final report, and a hash manifest. `scripts/reproduce_models.py --mode reproduce --output ...` is a separate optional historical training workflow and is never called by evaluation.

## Paper structure and evidence

| Experiment | Analyses | Frozen source |
| --- | --- | --- |
| 1. Differentiation and generalization | A1–A2 | Canonical High/Low monotonic checkpoints and genuine acquisition logs |
| 2. Available shared state | A3 | A/B-derived midpoint placement with wrong-position, orthogonal, and endpoint controls |
| 3. Generation and maintenance | A4–A5 | Single injection, cue continuation, identical-prestate branches, and takeover |
| 4. Constructive test | A6–A8 | Separate sequence-only C0–C3 cohort, geometry–behavior effects, effective-cue audit |

The canonical High/Low cohort and the constructive sequence-only C0–C3 cohort have distinct training objectives and are never pooled as one baseline. Novel30 and Novel60 are fixed out-of-support A/B-plane cues; historical “Novel” labels are mapped to `LegacyNovel` in supplementary data. The physical third cue channel remains inactive in the historical models. See [methods](docs/methods.md), [analysis contract](docs/analysis_contract.md), [experiment map](docs/experiment_map.md), [novelty registry](docs/novelty_registry.md), and [provenance](docs/provenance.md).

Historical human datasets and analyses are retained in `data/` and the older packages, but no human refitting or collection is part of this computational freeze. Their manuscript role is documented separately in [data notes](data/README.md). The full pre-cleanup hash snapshot is `artifacts/manifests/pre_freeze_snapshot.json`; [the archive crosswalk](archive/artifact_index.json) maps historical documents and runners to preserved copies. Older paper runners remain available as compatibility entry points, but they do not implement this freeze.

## Prospective dynamical extension

The separately registered [dynamical generalization program](docs/dynamical_generalization.md) implements output-compatibility/closure diagnostics, an independent cue-coordinate grid, and a gated separation × closure factorial with fixed-sequence replication. Its runners, registries, scratchpad and outputs are separate from the historical paper freeze. See that guide for stage modes, familiar-only pilot rules and explicit blocked outcomes.

The [completed dynamical report](results/dynamical_generalization/dg_v1_20261103/FINAL_DYNAMICAL_GENERALIZATION_REPORT.md) contains E1/E2 results. The registered E3 pilot found no setting meeting all familiar-competence criteria, so confirmatory E3 and sequence replication E4 were not run. The pilot and design-only outcomes are preserved explicitly; no search expansion followed.
