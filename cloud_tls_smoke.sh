#!/usr/bin/env bash
# Run this from an independently extracted verified kit; preserve the repository.
set -euo pipefail
kit_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python3 -I "$kit_directory/experiments/cloud_runtime_tls_v1/bootstrap.py" --tests-only
python3 -I "$kit_directory/experiments/cloud_runtime_tls_v1/bootstrap.py" "$@"
