# First cloud task — deterministic preflight only, no research cycle

Use the updated repository code and restore the privately supplied cloud evidence bundle at repository root, preserving relative paths. Do not read or copy local auth.json; no credential migration or API-key fallback. Do not run research_search launch/probe-backend or any integration call/compute command.

This task is a separate model call. Require the owner's explicit approval of this task, reserve19000 reported tokens from the cloud pilot's100000 total, and retain actual platform token telemetry. Include cloud setup/task tokens in the total. Missing telemetry is unknown usage, not zero; hold the reservation and stop before further role calls. This file is not approval by itself.

1. Run `bash docs/research_search_cloud_v1/install.sh` during environment setup.
2. Run `PYTHONPATH=.:src python -m pytest -q tests/research_search_cloud_v1`.
3. Run `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.research_search_cloud_v1 preflight > CLOUD_PREFLIGHT_RESULT.json`.
4. Return that exact JSON, test results, errors, runtime and actual provider-reported input/output/cached token usage. If unavailable, record null and stop. Preserve negative/blocked results. Do not run a scientific experiment or invent a successful multi-agent cycle.

The existing worker has local acceptance only. Cloud role isolation, independent evaluator file protection and aggregate per-call usage must be validated before executing the two-specialist/coordinator/critic cycle. Linux must not be represented as passing the original arm64 software seal. No training, model forward, checkpoints, confirmation or72h campaign is authorized by this task. Research scores/thresholds and historical classifications remain unchanged. No stored data may be uploaded to a public GitHub branch.
