# Current source and cloud execution map

Cloud preparation starts at [CLOUD_SETUP.md](CLOUD_SETUP.md). The original [REPO_MAP.md](REPO_MAP.md) and [AGENTS.md](AGENTS.md) retain the frozen experiment's entry instructions. Later experimental source is included without copying private evidence into Git.

| Path | Responsibility |
| --- | --- |
| experiments/research_search_cloud_v1/ | Portable private-evidence exporter and deterministic cloud preflight; no research launch command |
| docs/research_search_cloud_v1/ | Install dependencies, cloud preflight task and private transfer instructions |
| tests/research_search_cloud_v1/ | Export exclusions, tamper/escape detection and missing-usage stop tests |
| experiments/research_search_integration_v1_1/ | Local integration with hash-pinned executable discovery; Mac runtime is not a cloud backend |
| experiments/research_search_integration_v1/ | Preserved earlier integration implementation |
| experiments/research_search_pilot_v2/ | Preserved bounded pilot implementation |
| research_search/ | Native deterministic metrics, queue, budgets, task logic and frozen original campaign implementation |
| experiments/ | Separate scientific programs; never pool cohorts or rerun training to reproduce tables |
| tests/ | Program-specific regression and contract tests |
| src/context_sequence/ | Canonical evaluation, provenance and analysis code |
| confirmatory_srn/, execution_objective_search/, ftt_srn_*/ | Native model/objective and related source dependencies |
| configs/ | Fixed program configuration; source publication does not authorize launching it |

Original sources, checkpoints, evaluator rules and sealed outputs remain unchanged. The original72-hour campaign must not be launched by cloud setup. Private evidence is restored separately at its original paths; cloud preflight verifies its manifest. Full remote role coordination requires independent cloud isolation validation and actual provider-reported token accounting, with manual approval per call. Prepared source is not evidence of a running cloud task.

## Verified selected cached analysis (November25,2026)

The owner approved public publication of these research results. The exact38-file bundle is [artifacts/selected-analysis-results.zip](artifacts/selected-analysis-results.zip). Its independently verified report, two primary PNG/SVG figure pairs, seed/task tables, claim map and table-only reproduction are in [results/research_search_cloud/selected_analysis_verified_v3_1/REPORT.md](results/research_search_cloud/selected_analysis_verified_v3_1/REPORT.md). The report preserves the export recovery, stale accounting metadata, zero joint success and unresolved parent/critic isolation. It authorizes no further calls or campaign.

| Path | Responsibility |
| --- | --- |
| experiments/cloud_confidence_selected_v1/ | Frozen saved-readout table arithmetic, all-four-probability checks and figures |
| experiments/worker_isolation_preflight_v1/ | Synthetic Docker/seccomp restriction fixtures; no research data |
| experiments/cloud_confidence_container_v2/ | Explicit frozen-table container adapter, source staging and approval gates |
| experiments/cloud_runtime_tls_v1/ | Temporary build-CA secret propagation; same scientific worker |
| results/research_search_cloud/selected_analysis_verified_v3_1/verify_saved_tables.py | Archive-only checksum/table reproduction; optional139-source verification |

Older setup/private-evidence notes describe their original snapshots. The public result ZIP is the selected16-history zero-prior cached study, not all historical checkpoints or a public credential/confirmation package. Keep cohorts separate and restore predecessor evidence only when needed; never retrain for reproduction.

## Supervised full-prior native Cloud trial

Start with [the one-paste Cloud handoff](docs/cloud_native_trial_v1/START_IN_CLOUD.md) and [fixed protocol/source audit](docs/cloud_native_trial_v1/README.md). `cloud_native_trial.sh` reuses the successful Cloud image; `experiments/cloud_native_trial_v1/` wraps native pretraining/target/scoring in a separate restricted Linux worker and read-only same-identity exporter. One registered cue-diversity candidate plus control, two seeds/four tasks, full1920 prior episodes,330 A/B target updates;6h wall/5h CPU ceilings. This is supervised numerical execution with zero automatic agent calls; it does not certify or start the sealed72h autonomous campaign. Original unavailable competence-generation failure and versioned shared-only repair are retained in its scratchpad.
