#!/usr/bin/env bash
set -euo pipefail
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r docs/research_search_cloud_v1/requirements-cloud.txt
# Installation is not an agent call and never starts a campaign.
