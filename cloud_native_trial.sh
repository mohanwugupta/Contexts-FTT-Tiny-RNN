#!/usr/bin/env bash
# Supervised numerical trial only. No Codex/API dispatch, installs or image builds.
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -ne 2 ]]; then
  echo "Usage: bash cloud_native_trial.sh start|resume|status|stop|export OUTPUT_DIRECTORY" >&2
  exit 2
fi
trial_action="$1"
trial_directory="$2"
export PYTHONPATH=.:src
case "$trial_action" in
  status)
    exec python3 -m experiments.cloud_native_trial_v1 status --out "$trial_directory"
    ;;
  start|resume|stop|export) ;;
  *) echo "Unknown action: $trial_action" >&2; exit 2 ;;
esac
trial_image_file=/tmp/srn-cached-adapter-tls-v1/image-id
if [[ ! -s "$trial_image_file" ]]; then
  echo "Accepted image-id file is absent. Return to the successful Cloud environment; no rebuild or fallback was started." >&2
  exit 3
fi
trial_image="$(tr -d '\r\n' < "$trial_image_file")"
trial_socket=/var/run/docker.sock
if [[ ! -S "$trial_socket" ]]; then trial_socket=/run/codex-docker/docker.sock; fi
if [[ ! -S "$trial_socket" ]]; then echo "Accepted Docker socket unavailable" >&2; exit 3; fi
# This only inspects the already available image and never pulls it.
docker --host="unix://$trial_socket" image inspect "$trial_image" >/dev/null
case "$trial_action" in
  start)
    python3 -m experiments.cloud_native_trial_v1 prepare --out "$trial_directory"
    exec python3 -m experiments.cloud_native_trial_v1 run --out "$trial_directory" --image "$trial_image" --socket "$trial_socket"
    ;;
  resume)
    exec python3 -m experiments.cloud_native_trial_v1 run --out "$trial_directory" --image "$trial_image" --socket "$trial_socket"
    ;;
  stop|export)
    exec python3 -m experiments.cloud_native_trial_v1 "$trial_action" --out "$trial_directory" --image "$trial_image" --socket "$trial_socket"
    ;;
esac
