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
